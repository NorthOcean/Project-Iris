"""
@Author: Conghao Wong
@Date: 2021-07-08 20:58:48
@LastEditors: Conghao Wong
@LastEditTime: 2021-12-31 10:11:52
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
from .__args import VArgs
from .__layers import ContextEncoding, GraphConv, TrajEncoding


class VIrisAlphaModel(M.prediction.Model):
    """
    Deterministic first stage `Vertical-D` model
    """

    def __init__(self, Args: VArgs,
                 pred_number: int,
                 training_structure=None,
                 *args, **kwargs):

        super().__init__(Args, training_structure,
                         *args, **kwargs)
        
        self.args = Args

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

        self.adj_fc = keras.layers.Dense(Args.Kc, tf.nn.tanh)

        self.fc1 = keras.layers.Dense(128, activation=tf.nn.tanh)
        self.fc2 = keras.layers.Dense(self.n_pred * 2)

        self.reshape = keras.layers.Reshape([Args.Kc, self.n_pred, 2])

    def call(self, inputs: List[tf.Tensor],
             training=None, mask=None) -> tf.Tensor:
        """
        Run the first stage deterministic  `Vertical-D` model

        :param inputs: a list of tensors, which includes `trajs` and `maps`
            - trajs, shape = `(batch, obs, 2)`
            - maps, shape = `(batch, a, a)`
            
        :param training: controls run as the training mode or the test mode

        :return predictions: predicted trajectories, shape = `(batch, Kc, N, 2)`
        """

        # unpack inputs
        trajs, maps = inputs[:2]

        traj_feature = self.te.call(trajs)
        context_feature = self.ce.call(maps)

        # transformer inputs shape = (batch, obs, 128)
        t_inputs = self.concat([traj_feature, context_feature])
        t_outputs = trajs

        # transformer
        # output shape = (batch, obs, 128)
        t_features, _ = self.transformer.call(t_inputs,
                                              t_outputs,
                                              training=training)

        adj = tf.transpose(self.adj_fc(t_inputs), [0, 2, 1])
        # (batch, Kc, 128)
        m_features = self.gcn.call(features=t_features,
                                   adjMatrix=adj)

        # shape = (batch, Kc, 2*n)
        vec = self.fc2(self.fc1(m_features))

        # shape = (batch, Kc, n, 2)
        return self.reshape(vec)


class VIrisAlpha(M.prediction.Structure):
    """
    Training structure for the deterministic first stage `Vertical-D`
    """

    def __init__(self, Args: List[str], *args, **kwargs):
        super().__init__(Args, *args, **kwargs)

        self.args = VArgs(Args)
        self.important_args += ['Kc', 'p_index']

        self.set_model_inputs('traj', 'maps')
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
        p_index = [int(i) for i in self.args.p_index.split('_')]
        return tf.cast(p_index, tf.int32)

    @property
    def p_len(self) -> int:
        """
        Length of predicted key points.
        """
        return len(self.p_index)

    def create_model(self, model_type=VIrisAlphaModel,
                     *args, **kwargs):

        model = model_type(self.args,
                           pred_number=self.p_len,
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
        return M.prediction.loss.ADE(outputs[0], labels_pickled)

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
