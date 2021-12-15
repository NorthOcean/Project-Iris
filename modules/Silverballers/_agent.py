"""
@Author: Conghao Wong
@Date: 2021-10-28 19:38:56
@LastEditors: Conghao Wong
@LastEditTime: 2021-12-15 09:51:16
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import List

import tensorflow as tf
from tensorflow import keras

from .. import applications as A
from .. import models as M
from ._args import AgentArgs
from ._layers import FFTlayer, GraphConv, IFFTlayer, TrajEncoding


class AgentModel(M.prediction.Model):
    """
    The first stage `Agent` model from `Silverballers`
    """

    def __init__(self, Args: AgentArgs,
                 feature_dim: int = 128,
                 id_depth: int = 64,
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
        self.set_preprocess('Move', 'Scale', 'Rotate')
        self.set_preprocess_parameters(move=0)

        # Layers
        self.te = TrajEncoding(
            units=self.d//2, activation=tf.nn.tanh, useFFT=True)
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

        # feature embedding and encoding -> (batch, obs, d)
        spec_features = self.te.call(trajs)

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


class Agent(M.prediction.Structure):

    model_type = AgentModel

    def __init__(self, Args: List[str],
                 association: M.prediction.Structure = None,
                 *args, **kwargs):

        super().__init__(Args, *args, **kwargs)

        self.args = AgentArgs(Args)
        self.important_args += ['Kc', 'key_points', 'depth', 'preprocess']

        self.set_model_inputs('traj')
        self.set_model_groundtruths('gt')

        self.set_loss(self.l2_loss)
        self.set_loss_weights(1.0)

        self.set_metrics(self.min_FDE)
        self.set_metrics_weights(1.0)

        self.association = association

    @property
    def p_index(self) -> tf.Tensor:
        """
        Time step of predicted key points.
        """
        p_index = [int(i) for i in self.args.key_points.split('_')]
        return tf.cast(p_index, tf.int32)

    @property
    def p_len(self) -> int:
        """
        Length of predicted key points.
        """
        return len(self.p_index)

    def set_model_type(self, new_type=AgentModel):
        self.model_type = new_type

    def create_model(self, *args, **kwargs):
        model = self.model_type(self.args,
                                feature_dim=128,
                                id_depth=self.args.depth,
                                keypoints_number=self.p_len,
                                training_structure=self,
                                *args, **kwargs)

        opt = keras.optimizers.Adam(self.args.lr)
        return model, opt

    def l2_loss(self, outputs: List[tf.Tensor],
                labels: tf.Tensor,
                *args, **kwargs) -> tf.Tensor:
        """
        L2 distance between predictions and labels on predicted key points
        """
        labels_pickled = tf.gather(labels, self.p_index, axis=1)
        return M.prediction.Loss.ADE(outputs[0], labels_pickled)

    def min_FDE(self, outputs: List[tf.Tensor],
                labels: tf.Tensor,
                *args, **kwargs) -> tf.Tensor:
        """
        minimum FDE among all predictions
        """
        # shape = (batch, Kc*K)
        distance = tf.linalg.norm(
            outputs[0][:, :, -1, :] -
            tf.expand_dims(labels[:, -1, :], 1), axis=-1)

        return tf.reduce_mean(tf.reduce_min(distance, axis=-1))
