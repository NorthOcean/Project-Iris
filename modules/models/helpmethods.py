"""
@Author: Conghao Wong
@Date: 2021-01-08 09:14:00
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-16 16:20:59
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras


def dir_check(target_dir: str) -> str:
    """
    Used for check if the `target_dir` exists.
    It not exist, it will make it.
    """
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    return target_dir


# help methods for linear predict
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)


def __predict_linear(x, y, x_p, diff_weights=0):
    if diff_weights == 0:
        P = np.diag(np.ones(shape=[x.shape[0]]))
    else:
        P = np.diag(softmax([(i+1)**diff_weights for i in range(x.shape[0])]))

    A = np.stack([np.ones_like(x), x]).T
    A_p = np.stack([np.ones_like(x_p), x_p]).T
    Y = y.T
    B = np.matmul(np.matmul(np.matmul(np.linalg.inv(
        np.matmul(np.matmul(A.T, P), A)), A.T), P), Y)
    Y_p = np.matmul(A_p, B)
    return Y_p, B


def predict_linear_for_person(position, time_pred, different_weights=0.95) -> np.ndarray:
    """
    对二维坐标的最小二乘拟合
    注意：`time_pred`中应当包含现有的长度，如`len(position)=8`, `time_pred=20`时，输出长度为20
    """
    time_obv = position.shape[0]
    t = np.arange(time_obv)
    t_p = np.arange(time_pred)
    x = position.T[0]
    y = position.T[1]

    x_p, _ = __predict_linear(t, x, t_p, diff_weights=different_weights)
    y_p, _ = __predict_linear(t, y, t_p, diff_weights=different_weights)

    return np.stack([x_p, y_p]).T


def calculate_ADE_FDE_numpy(pred, GT) -> Tuple[np.ndarray, np.ndarray]:
    if len(pred.shape) == 3:    # [K, pred, 2]
        ade = []
        fde = []
        for p in pred:
            all_loss = np.linalg.norm(p - GT, ord=2, axis=1)
            ade.append(np.mean(all_loss))
            fde.append(all_loss[-1])

        min_index = np.argmin(np.array(ade))
        ade = ade[min_index]
        fde = fde[min_index]

        # # ADE of the mean traj
        # mean_traj = np.mean(pred, axis=0)
        # mean_traj_loss = np.linalg.norm(mean_traj - GT, ord=2, axis=1)
        # ade = np.mean(mean_traj_loss)
        # fde = mean_traj_loss[-1]

    else:
        all_loss = np.linalg.norm(pred - GT, ord=2, axis=1)
        ade = np.mean(all_loss)
        fde = all_loss[-1]

    return ade, fde


def GraphConv_layer(output_units, activation=None):
    return keras.layers.Dense(output_units, activation=activation)


def GraphConv_func(features, A, output_units=64, activation=None, layer=None):
    dot = tf.matmul(A, features)
    if layer == None:
        res = keras.layers.Dense(output_units, activation=activation)(dot)
    else:
        res = layer(dot)
    return res
