"""
@Author: Conghao Wong
@Date: 2021-01-08 15:08:07
@LastEditors: Conghao Wong
@LastEditTime: 2022-04-21 11:00:37
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import numpy as np
import tensorflow as tf


def difference(trajs: tf.Tensor, direction='back', ordd=1) -> list[tf.Tensor]:
    """
    :param trajs: trajectories, shape = `[(K,) batch, pred, 2]`
    :param direction: string, canbe `'back'` or `'forward'`
    :param ord: repeat times

    :return result: results list, `len(results) = ord + 1`
    """
    outputs = [trajs]
    for repeat in range(ordd):
        outputs_current = \
            outputs[-1][:, :, 1:, :] - outputs[-1][:, :, :-1, :] if len(trajs.shape) == 4 else \
            outputs[-1][:, 1:, :] - outputs[-1][:, :-1, :]
        outputs.append(outputs_current)
    return outputs


def calculate_cosine(vec1: np.ndarray,
                     vec2: np.ndarray):

    length1 = np.linalg.norm(vec1, axis=-1)
    length2 = np.linalg.norm(vec2, axis=-1)

    return (np.sum(vec1 * vec2, axis=-1) + 0.0001) / ((length1 * length2) + 0.0001)


def calculate_length(vec1):
    return np.linalg.norm(vec1, axis=-1)


def activation(x: np.ndarray, a=1, b=1):
    return np.less_equal(x, 0) * a * x + np.greater(x, 0) * b * x
