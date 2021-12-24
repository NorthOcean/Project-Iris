"""
@Author: Conghao Wong
@Date: 2021-12-23 09:35:54
@LastEditors: Conghao Wong
@LastEditTime: 2021-12-23 09:49:42
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import List

import tensorflow as tf
from tensorflow import keras

from ... import applications as A
from ... import models as M
from .._args import AgentArgs
from .._baseAgent import BaseAgentStructure
from .._layers import FFTlayer, GraphConv, IFFTlayer, OuterLayer, TrajEncoding


class Agent47DModel(M.prediction.Model):

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
        self.te = TrajEncoding(self.d//2, tf.nn.relu, useFFT=True)

        self.outer = OuterLayer(self.d//2, self.d//2, reshape=False)
        self.pooling = keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            data_format='channels_first')

        self.outer_fc = keras.layers.Dense(self.d//2, tf.nn.tanh)

        self.ie = TrajEncoding(self.d//2, tf.nn.tanh)
        self.concat = keras.layers.Concatenate(axis=-1)

        self.fft = FFTlayer()
        self.ifft = IFFTlayer()

        # Transformer is used as a feature extractor
        self.T = A.TransformerEncoder(num_layers=4,
                                      num_heads=8,
                                      dim_model=self.d,
                                      dim_forward=512,
                                      steps=self.args.obs_frames,
                                      include_top=False)

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
        f = self.te.call(trajs)             # (batch, obs, d/2)
        f = self.outer.call(f, f)           # (batch, obs, d/2, d/2)
        f = self.pooling(f)                 # (batch, obs, d/4, d/4)
        f = tf.reshape(f, [f.shape[0], f.shape[1], -1])
        spec_features = self.outer_fc(f)    # (batch, obs, d/2)

        all_predictions = []
        rep_time = 1 if training else self.args.K
        for _ in range(rep_time):
            # assign random ids and embedding -> (batch, obs, d/2)
            ids = tf.random.normal([bs, self.args.obs_frames, self.d_id])
            id_features = self.ie.call(ids)

            # transformer -> (batch, obs, d)
            t_inputs = self.concat([spec_features, id_features])
            behavior_features = self.T.call(t_inputs, training=training)

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


class Agent47D(BaseAgentStructure):

    def __init__(self, Args: List[str],
                 association: M.prediction.Structure = None,
                 *args, **kwargs):

        super().__init__(Args, association=association, *args, **kwargs)

        self.set_model_type(new_type=Agent47DModel)
