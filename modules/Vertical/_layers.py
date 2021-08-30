"""
@Author: Conghao Wong
@Date: 2021-07-08 15:17:59
@LastEditors: Conghao Wong
@LastEditTime: 2021-08-30 09:46:59
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras


class FFTlayer(keras.layers.Layer):
    """
    Calculate DFT for the batch inputs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.concat = keras.layers.Concatenate()

    def call(self, inputs: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        :param inputs: batch inputs, shape = (batch, N, M)
        :return fft: fft results (r and i), shape = ((batch, N, M), (batch, N, M))
        """

        ffts = []
        for index in range(0, inputs.shape[-1]):
            seq = tf.cast(tf.gather(inputs, index, axis=-1), tf.complex64)
            seq_fft = tf.signal.fft(seq)
            ffts.append(tf.expand_dims(seq_fft, -1))

        ffts = self.concat(ffts)
        return (tf.math.real(ffts), tf.math.imag(ffts))


class IFFTlayer(keras.layers.Layer):
    """
    Calculate IDFT for the batch inputs
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.concat = keras.layers.Concatenate()

    def call(self, real: tf.Tensor, imag: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        :param real: batch inputs of real part, shape = (batch, N, M)
        :param real: batch inputs of imaginary part, shape = (batch, N, M)
        :return ifft: ifft results, shape = (batch, N, M)
        """

        ffts = []
        for index in range(0, real.shape[-1]):
            r = tf.gather(real, index, axis=-1)
            i = tf.gather(imag, index, axis=-1)
            ffts.append(
                tf.expand_dims(
                    tf.math.real(
                        tf.signal.ifft(
                            tf.complex(r, i)
                        )
                    ), axis=-1
                )
            )

        return self.concat(ffts)


class ContextEncoding(keras.layers.Layer):
    """
    Encode context maps into the context feature
    """

    def __init__(self, output_channels: int,
                 units: int = 64,
                 activation=None,
                 *args, **kwargs):
        """
        Init a context encoding module

        :param output_channels: output channels 
        :param units: output feature dimension
        :param activation: activations used in the output layer
        """

        super().__init__(*args, **kwargs)

        self.pool = keras.layers.MaxPooling2D([5, 5])
        self.flatten = keras.layers.Flatten()
        self.fc = keras.layers.Dense(output_channels * units, activation)
        self.reshape = keras.layers.Reshape((output_channels, units))

    def call(self, context_map: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Encode context maps into context features

        :param context_map: maps, shape = `(batch, a, a)`
        :return feature: features, shape = `(batch, output_channels, units)`
        """
        pool = self.pool(context_map[:, :, :, tf.newaxis])
        flat = self.flatten(pool)
        fc = self.fc(flat)
        return self.reshape(fc)


class TrajEncoding(keras.layers.Layer):
    """
    Encode trajectories into the traj feature
    """

    def __init__(self, units: int = 64,
                 activation=None,
                 useFFT=None,
                 *args, **kwargs):
        """
        Init a trajectory encoding module

        :param units: feature dimension
        :param activation: activations used in the output layer
        :param useFFT: controls if encode trajectories in `freq domain`
        """

        super().__init__(*args, **kwargs)

        self.useFFT = useFFT

        if (self.useFFT):
            self.fft = FFTlayer()
            self.concat = keras.layers.Concatenate()
            self.fc2 = keras.layers.Dense(units, tf.nn.relu)

        self.fc1 = keras.layers.Dense(units, activation)

    def call(self, trajs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Encode trajectories into the high-dimension features

        :param trajs: trajs, shape = `(batch, N, 2)`
        :return features: features, shape = `(batch, N, units)`
        """
        if self.useFFT:
            t_r, t_i = self.fft.call(trajs)
            concat = self.concat([t_r, t_i])
            trajs = self.fc2(concat)

        return self.fc1(trajs)


class LinearPrediction(keras.layers.Layer):
    """
    Linear prediction from start points (not contain) to end points.
    """

    def __init__(self, useFFT=None, include_obs=None, *args, **kwargs):
        """
        Init a linear prediction layer

        :param useFFT: controls if applys `DFT` on outputs
        :param include_obs: controls if includes observations in outputs
        """
        super().__init__(*args, **kwargs)

        self.useFFT = useFFT
        self.include_obs = include_obs

        if self.useFFT:
            self.fft = FFTlayer()

        self.concat = keras.layers.Concatenate(axis=-2)
        self.concat1 = keras.layers.Concatenate(axis=-1)

    def call(self, start, end, n, obs=None, *args, **kwargs) -> tf.Tensor:
        """
        Run the linear prediction operation
        
        :param start: start points, shape = (batch, 1, 2)
        :param end: end points, shape == (batch, 1, 2)
        :param n: number of prediction points, DOES NOT contain the start point

        :return pred: linear predictions, DOES NOT contain the start point
        """
        results = []
        for i in range(1, n):
            p = i / n
            results.append((end - start) * p + start)

        results.append(end)
        pred = self.concat(results)  # (batch, n, 2)

        # output shape = (batch, obs+n, 2)
        if self.include_obs:
            pred = self.concat([obs, pred])

        # output shape = (batch, obs+n, 4)
        if self.useFFT:
            pred_r, pred_i = self.fft(pred)
            pred = self.concat1([pred_r, pred_i])

        return pred


class GraphConv(keras.layers.Layer):
    """
    Graph conv layer
    """

    def __init__(self, units: int,
                 activation=None,
                 *args, **kwargs):
        """
        Init a graph convolution layer

        :param units: feature dimension
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
