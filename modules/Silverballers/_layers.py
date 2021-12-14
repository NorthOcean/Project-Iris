"""
@Author: Conghao Wong
@Date: 2021-10-28 19:53:54
@LastEditors: Conghao Wong
@LastEditTime: 2021-12-14 16:10:49
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from ..Vertical._layers import FFTlayer, GraphConv, IFFTlayer, TrajEncoding


class OuterLayer(keras.layers.Layer):
    """
    Compute the outer product of two vectors.

    :param a_dim: the last dimension of the first input feature
    :param b_dim: the last dimension of the second input feature
    :param reshape: if `reshape == True`, output shape = `(..., a_dim * b_dim)`
        else output shape = `(..., a_dim, b_dim)`
    """

    def __init__(self, a_dim: int, b_dim: int,
                 reshape=False,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.M = a_dim
        self.N = b_dim
        self.reshape = reshape

    def call(self, tensorA: tf.Tensor, tensorB: tf.Tensor):
        """
        Compute the outer product of two vectors.

        :param tensorA: shape = (..., M)
        :param tensorB: shape = (..., N)
        :return outer: shape = (..., M, N) if `reshape` is `False`,
            else its output shape = (..., M*N)
        """

        _a = tf.expand_dims(tensorA, axis=-1)
        _b = tf.expand_dims(tensorB, axis=-2)

        _a = tf.repeat(_a, self.N, axis=-1)
        _b = tf.repeat(_b, self.M, axis=-2)

        outer = _a * _b

        if not self.reshape:
            return outer
        else:
            return tf.reshape(outer, list(outer.shape[:-2]) + [self.M*self.N])
