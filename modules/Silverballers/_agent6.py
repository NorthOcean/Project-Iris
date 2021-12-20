"""
@Author: Conghao Wong
@Date: 2021-12-15 16:06:16
@LastEditors: Conghao Wong
@LastEditTime: 2021-12-15 20:33:40
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import List

import tensorflow as tf
from tensorflow import keras

from .. import applications as A
from .. import models as M
from ._agent import Agent
from ._args import AgentArgs
from ._layers import FFTlayer, GraphConv, IFFTlayer, OuterLayer, TrajEncoding


class Agent6Model(M.prediction.Model):

    def __init__(self, Args: AgentArgs,
                 feature_dim: int = 128,
                 id_depth: int = 16,
                 keypoints_number: int = 3,
                 training_structure=None,
                 *args, **kwargs):

        super().__init__(Args, training_structure, *args, **kwargs)

        self.args = Args

        # Parameters
        self.d = feature_dim
        self.n_key = keypoints_number
        self.d_id = id_depth

        # Preprocess
        preprocess_list = ()
        for index, operation in enumerate(['Move', 'Scale', 'Rotate']):
            if self.args.preprocess[index] == '1':
                preprocess_list += (operation,)
        
        self.set_preprocess(*preprocess_list)
        self.set_preprocess_parameters(move=0)

        # Layers
        self.te1 = TrajEncoding(self.d//2, activation=tf.nn.relu, useFFT=True)
        self.te2 = TrajEncoding(self.d//2, activation=tf.nn.relu, useFFT=True)
        self.ie = TrajEncoding(units=self.d//2, activation=tf.nn.tanh)
        self.concat = keras.layers.Concatenate(axis=-1)

        self.fft = FFTlayer()
        self.ifft = IFFTlayer()

        # Transformer is used as a feature extractor
        self.T = A.Transformer(num_layers=4,
                               d_model=self.d,
                               num_heads=8,
                               dff=512,
                               input_vocab_size=None,
                               target_vocab_size=None,
                               pe_input=Args.obs_frames,
                               pe_target=Args.obs_frames,
                               include_top=False)

        # Outer product
        self.outer = OuterLayer(self.d//2, self.d//2, reshape=True)
        self.outer_fc = keras.layers.Dense(self.d//2, tf.nn.tanh)

        # Trainable adj matrix and gcn layer
        self.adj_fc = keras.layers.Dense(self.args.Kc, tf.nn.tanh)
        self.gcn = GraphConv(units=self.d)

        # Decoder layers
        self.decoder_fc1 = keras.layers.Dense(self.d, tf.nn.tanh)
        self.decoder_fc2 = keras.layers.Dense(4 * self.n_key)
        self.decoder_reshape = keras.layers.Reshape(
            [self.args.Kc, self.n_key, 4])

    def call(self, inputs: List[tf.Tensor],
             training=None, mask=None):

        # unpack inputs
        trajs = inputs[0]   # (batch, obs, 2)
        bs = trajs.shape[0]

        # feature embedding and encoding -> (batch, obs, d/2)
        f1 = self.te1.call(trajs)
        f2 = self.te2.call(trajs)
        f = self.outer.call(f1, f2)
        spec_features = self.outer_fc(f)

        all_predictions = []
        rep_time = 1 if training else self.args.K
        for _ in range(rep_time):
            # assign random ids and embedding -> (batch, obs, d)
            ids = tf.random.normal([bs, self.args.obs_frames, self.d_id])
            id_features = self.ie.call(ids)

            # transformer inputs
            t_inputs = self.concat([spec_features, id_features])
            t_outputs = self.concat(self.fft.call(trajs))

            # transformer -> (batch, obs, d)
            behavior_features, _ = self.T.call(inputs=t_inputs,
                                               targets=t_outputs,
                                               training=training)

            # multi-style features -> (batch, Kc, d)
            adj = tf.transpose(self.adj_fc(t_inputs), [0, 2, 1])
            m_features = self.gcn.call(behavior_features, adj)

            # predicted keypoints -> (batch, Kc, key, 2)
            y = self.decoder_fc1(m_features)
            y = self.decoder_fc2(y)
            y = self.decoder_reshape(y)

            y = self.ifft.call(real=y[:, :, :, :2], imag=y[:, :, :, 2:])
            all_predictions.append(y)

        return tf.concat(all_predictions, axis=1)


class Agent6(Agent):

    def __init__(self, Args: List[str], association: M.prediction.Structure = None, *args, **kwargs):
        super().__init__(Args, association=association, *args, **kwargs)

        self.set_model_type(new_type=Agent6Model)