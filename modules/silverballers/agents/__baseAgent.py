"""
@Author: Conghao Wong
@Date: 2021-12-22 19:20:26
@LastEditors: Conghao Wong
@LastEditTime: 2022-04-15 09:15:39
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import List

import tensorflow as tf
from tensorflow import keras

from ... import models as M
from ..__args import AgentArgs


class BaseAgentStructure(M.prediction.Structure):

    model_type = None

    def __init__(self, Args: List[str],
                 *args, **kwargs):

        super().__init__(Args, *args, **kwargs)

        self.args = AgentArgs(Args)
        self.important_args += ['Kc', 'key_points', 'depth', 'preprocess']

        self.set_model_inputs('traj')
        self.set_model_groundtruths('gt')

        self.set_loss(self.l2_loss)
        self.set_loss_weights(1.0)

        if self.args.metric == 'fde':
            self.set_metrics(self.min_FDE)
        elif self.args.metric == 'ade':
            self.set_metrics(self.l2_loss)
        else:
            raise ValueError(self.log('Metric error!', level='error'))

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

    def set_model_type(self, new_type):
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
