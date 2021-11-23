"""
@Author: Conghao Wong
@Date: 2021-10-28 19:38:56
@LastEditors: Conghao Wong
@LastEditTime: 2021-10-28 20:47:30
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import List

import tensorflow as tf
from tensorflow import keras

from .. import applications as A
from .. import models as M
from ._args import SArgs
from ._layers import FFTlayer, GraphConv, IFFTlayer, TrajEncoding


class AgentModel(M.prediction.Model):
    """
    The first stage `Agent` model from `Silverballers`
    """

    def __init__(self, Args: SArgs,
                 feature_dim: int = 128,
                 keypoints_number: int = 3,
                 training_structure=None,
                 *args, **kwargs):

        super().__init__(Args, training_structure, *args, **kwargs)

        self.args = Args

        # Parameters
        self.d = feature_dim
        self.n_key = keypoints_number

        # Preprocess
        self.set_preprocess('move')
        self.set_preprocess_parameters(move=0)

        # Layers
        self.te = TrajEncoding(units=self.d,
                               activation=tf.nn.relu,
                               useFFT=True)
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
        self.gcn = GraphConv(units=self.d, activation=tf.nn.relu)

        # Decoder layers
        self.decoder = keras.Sequential([
            keras.layers.Dense(self.d, tf.nn.relu),
            keras.layers.Dense(self.d, tf.nn.relu),
            keras.layers.Dense(4*(self.n_key + self.args.obs_frames)),
            keras.layers.Reshape(
                [self.args.Kc, self.n_key + self.args.obs_frames, 4])
        ])

    def call(self, inputs: List[tf.Tensor],
             training=None, mask=None):

        # unpack inputs
        trajs = inputs[0]

        # feature embedding and encoding -> (batch, obs, d)
        spec_features = self.te.call(trajs)
        spec_obs = tf.concat(self.fft.call(trajs), axis=-1)
        behavior_features, _ = self.T.call(inputs=spec_features,
                                           targets=spec_obs,
                                           training=training)

        # multi-style features -> (batch, Kc, d)
        adj = tf.transpose(self.adj_fc(spec_features), [0, 2, 1])
        m_features = self.gcn.call(behavior_features, adj)

        # predicted keypoints (spectrums) -> (batch, Kc, obs+key, 4)
        spec = self.decoder.call(m_features)

        # ifft -> (batch, Kc, obs+key, 2)
        pred = self.ifft.call(real=spec[:, :, :, :2],
                              imag=spec[:, :, :, 2:])
        
        return pred[:, :, self.args.obs_frames:, :]


class Agent(M.prediction.Structure):

    def __init__(self, Args: List[str], *args, **kwargs):
        super().__init__(Args, *args, **kwargs)

        self.args = SArgs(Args)
        self.important_args += ['Kc', 'key_points']

        self.set_model_inputs('traj')
        self.set_model_groundtruths('gt')

        self.set_loss(self.l2_loss)
        self.set_loss_weights(1.0)
        
        self.set_metrics(self.min_FDE)
        self.set_metrics_weights(1.0)

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

    def create_model(self, *args, **kwargs):
        model = AgentModel(self.args,
                           feature_dim=128,
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
