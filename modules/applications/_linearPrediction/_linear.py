"""
@Author: Conghao Wong
@Date: 2021-09-16 19:44:00
@LastEditors: Conghao Wong
@LastEditTime: 2021-09-16 20:10:05
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf
from tensorflow import keras

class LinearLayer(keras.layers.Layer):
    def __init__(self, obs_frames, pred_frames, diff=0.95, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.h = obs_frames
        self.f = pred_frames
        self.diff = diff

        if self.diff == 0:
            P = tf.linalg.diag(tf.ones(self.h))
        else:
            P = tf.linalg.diag(tf.nn.softmax([(i+1)**self.diff for i in range(self.h)]))

        self.x = tf.range(self.h, dtype=tf.float32)
        self.x_p = tf.range(self.f, dtype=tf.float32) + self.h
        A = tf.transpose(tf.stack([
            tf.ones([self.h]),
            self.x
        ]))
        self.A_p = tf.transpose(tf.stack([
            tf.ones([self.f]),
            self.x_p
        ]))
        self.W = tf.linalg.inv(tf.transpose(A) @ P @ A) @ tf.transpose(A) @ P

    def call(self, inputs: tf.Tensor, **kwargs):
        """
        Linear prediction

        :param inputs: input trajs, shape = (batch, obs, 2)
        :param results: linear pred, shape = (batch, pred, 2)
        """
        x = inputs[:, :, 0:1]
        y = inputs[:, :, 1:2]
        
        Bx = self.W @ x
        By = self.W @ y

        results = tf.stack([
            self.A_p @ Bx,
            self.A_p @ By,
        ])

        results = tf.transpose(results[:, :, :, 0], [1, 2, 0])
        return results[:, -self.f:, :]