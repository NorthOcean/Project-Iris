"""
@Author: Conghao Wong
@Date: 2021-12-22 20:27:46
@LastEditors: Conghao Wong
@LastEditTime: 2022-04-13 20:51:09
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import List

import tensorflow as tf
from tensorflow import keras

from ... import applications as A
from ... import models as M
from ..__args import AgentArgs
from ..__layers import FFTlayer, GraphConv, IFFTlayer, OuterLayer, TrajEncoding
from .__baseAgent import BaseAgentStructure


class Agent47CModel(M.prediction.Model):

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
        self.T = A.Transformer(num_layers=4,
                               d_model=self.d,
                               num_heads=8,
                               dff=512,
                               input_vocab_size=None,
                               target_vocab_size=None,
                               pe_input=Args.obs_frames,
                               pe_target=Args.obs_frames,
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
        """
        Run the first stage `agent47` model.

        :param inputs: a list of tensors, including `trajs`
            - a batch of observed trajs, shape is `(batch, obs, 2)`

        :param training: set to `True` when training, or leave it `None`
        :return predictions: predicted keypoints, shape = `(batch, Kc, N_key, 2)`
        """

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


class Agent47C(BaseAgentStructure):
    
    """
    Training structure for the `Agent47C` model.
    Note that it is only used to train the single model.
    Please use the `Silverballers` structure if you want to test any
    agent-handler based silverballers models.
    """

    def __init__(self, Args: List[str],
                 *args, **kwargs):

        super().__init__(Args, *args, **kwargs)

        self.set_model_type(new_type=Agent47CModel)
