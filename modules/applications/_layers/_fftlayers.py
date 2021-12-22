"""
@Author: Conghao Wong
@Date: 2021-12-21 15:17:38
@LastEditors: Conghao Wong
@LastEditTime: 2021-12-21 15:17:38
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import Tuple

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
