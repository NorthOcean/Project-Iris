"""
@Author: Conghao Wong
@Date: 2021-12-21 15:20:57
@LastEditors: Conghao Wong
@LastEditTime: 2021-12-21 15:22:00
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf
from tensorflow import keras


class GraphConv(keras.layers.Layer):
    """
    Graph conv layer
    """

    def __init__(self, units: int,
                 activation=None,
                 *args, **kwargs):
        """
        Init a graph convolution layer

        :param units: output feature dimension
        :param activation: activations used in the output layer
        """
        super().__init__(*args, **kwargs)

        self.fc = keras.layers.Dense(units, activation)

    def call(self, features: tf.Tensor,
             adjMatrix: tf.Tensor,
             *args, **kwargs) -> tf.Tensor:
        """
        Run the graph convolution operation

        :param features: feature sequences, shape = (batch, N, M)
        :param adjMatrix: adj matrix, shape = (batch, N, N)
        :return outputs: shape = (batch, N, units)
        """

        dot = tf.matmul(adjMatrix, features)
        return self.fc(dot)