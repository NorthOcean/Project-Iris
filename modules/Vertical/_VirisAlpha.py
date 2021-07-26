"""
@Author: Conghao Wong
@Date: 2021-07-08 20:58:48
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-21 19:59:02
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .. import applications as A
from .. import models as M
from ._args import VArgs
from ._layers import ContextEncoding, GraphConv, TrajEncoding


class VIrisAlphaModel(M.prediction.Model):
    """
    VIrisAlphaModel
    ---------------
    Alpha model for `Vertical` project.
    - two stage model
    - first stage: important points (<- this model)
    - second stage: interpolation

    Inputs
    ------
    :param inputs: a list of tensors, including:
        - trajs, shape = (batch, obs, 2)
        - maps, shape = (batch, a, a)

    Outputs
    -------
    :return outputs: important points, shape = (batch, Kc, n, 2)
        where:
        - `Kc` is the number of style categories
        - `n` is the number of important points
    """

    def __init__(self, Args: VArgs,
                 pred_number: int,
                 training_structure=None,
                 *args, **kwargs):

        super().__init__(Args, training_structure,
                         *args, **kwargs)

        # Preprocess
        self.set_preprocess('move')
        self.set_preprocess_parameters(move=0)

        # Args
        self.n_pred = pred_number

        # Layers
        self.concat = keras.layers.Concatenate(axis=-1)

        self.te = TrajEncoding(units=64,
                               activation=tf.nn.tanh,
                               useFFT=False)

        self.ce = ContextEncoding(units=64,
                                  output_channels=Args.obs_frames,
                                  activation=tf.nn.tanh)

        self.transformer = A.Transformer(num_layers=4,
                                         d_model=128,
                                         num_heads=8,
                                         dff=512,
                                         input_vocab_size=None,
                                         target_vocab_size=None,
                                         pe_input=Args.obs_frames,
                                         pe_target=Args.obs_frames,
                                         include_top=False)

        self.gcn = GraphConv(units=128,
                             activation=None)

        self.adj_fc = keras.layers.Dense(Args.K_train, tf.nn.tanh)

        self.fc1 = keras.layers.Dense(128, activation=tf.nn.tanh)
        self.fc2 = keras.layers.Dense(self.n_pred * 2)

        self.reshape = keras.layers.Reshape([Args.K_train, self.n_pred, 2])

    def call(self, inputs: List[tf.Tensor],
             training=None, mask=None):

        # unpack inputs
        trajs, maps = inputs[:2]

        traj_feature = self.te(trajs)
        context_feature = self.ce(maps)

        # transformer inputs shape = (batch, obs, 128)
        t_inputs = self.concat([traj_feature, context_feature])
        t_outputs = trajs

        # transformer
        me, mc, md = A.create_transformer_masks(t_inputs, t_outputs)
        # output shape = (batch, obs, 128)
        t_features, _ = self.transformer(t_inputs, t_outputs,
                                         training,
                                         me, mc, md)

        adj = tf.transpose(self.adj_fc(t_inputs), [0, 2, 1])
        # (batch, Kc, 128)
        m_features = self.gcn(features=t_features,
                              adjMatrix=adj)

        # shape = (batch, Kc, 2*n)
        vec = self.fc2(self.fc1(m_features))

        # shape = (batch, Kc, n, 2)
        return self.reshape(vec)


class VIrisAlpha(M.prediction.Structure):

    def __init__(self, Args: List[str], *args, **kwargs):
        super().__init__(Args, *args, **kwargs)

        self.args = VArgs(Args)

        self.set_model_inputs('traj', 'maps', 'paras')
        self.set_model_groundtruths('gt')

        self.set_loss(self.loss_unnamed)
        self.set_loss_weights(1.0)

        self.set_metrics(self.min_FDE)
        self.set_metrics_weights(1.0)

    @property
    def p_index(self) -> tf.Tensor:
        p_index = [int(i) for i in self.args.p_index.split('_')]
        return tf.cast(p_index, tf.int32)
    
    @property
    def p_len(self) -> int:
        return len(self.p_index)

    def create_model(self, model_type=VIrisAlphaModel,
                     *args, **kwargs):
                     
        model = model_type(self.args,
                           pred_number=self.p_len,
                           training_structure=self,
                           *args, **kwargs)

        opt = keras.optimizers.Adam(self.args.lr)
        return model, opt

    def loss_unnamed(self, outputs: List[tf.Tensor],
                     labels: tf.Tensor) -> tf.Tensor:
        
        labels_pickled = tf.gather(labels, self.p_index, axis=1)
        return M.prediction.Loss.ADE(outputs[0], labels_pickled)

    def min_FDE(self, outputs, labels) -> tf.Tensor:
        distance = tf.linalg.norm(
            outputs[0][:, :, -1, :] - tf.expand_dims(labels[:, -1, :], 1), axis=-1)   # shape = [batch, K]
        return tf.reduce_mean(tf.reduce_min(distance, axis=-1))
