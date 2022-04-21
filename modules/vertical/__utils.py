"""
@Author: Conghao Wong
@Date: 2021-07-09 10:40:52
@LastEditors: Conghao Wong
@LastEditTime: 2022-04-21 11:02:37
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf


class Utils():

    @staticmethod
    def LinearInterpolation(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """
        Piecewise linear interpolation
        (Results do not contain the start point)

        :param x: index, shape = `(n)`, where `m = x[-1] - x[0]`
        :param y: values, shape = `(..., n, 2)`
        :return yp: linear interpolations, shape = `(..., m, 2)`
        """
        linear_results = []
        for output_index in range(x.shape[0] - 1):
            p_start = x[output_index]
            p_end = x[output_index+1]

            # shape = (..., 2)
            start = tf.gather(y, output_index, axis=-2)
            end = tf.gather(y, output_index+1, axis=-2)

            for p in tf.range(p_start+1, p_end+1):
                linear_results.append(tf.expand_dims(
                    (end - start) * (p - p_start) / (p_end - p_start)
                    + start
                , axis=-2))   # (..., 1, 2)

        # shape = (..., n, 2)
        return tf.concat(linear_results, axis=-2)