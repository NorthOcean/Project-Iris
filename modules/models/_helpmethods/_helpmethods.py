'''
Author: Conghao Wong
Date: 2021-01-08 09:14:00
LastEditors: Conghao Wong
LastEditTime: 2021-04-15 11:13:26
Description: file content
'''

import os
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
# from sklearn.manifold import TSNE
from tensorflow import keras
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tqdm import tqdm


def dir_check(target_dir) -> str:
    """
    Used for check if the `target_dir` exists.
    """
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    return target_dir


def reduce_dim(x, out_dim, pca=True):
    if not pca:
        # TODO: sklearn not available
        pass
        # tsne = TSNE(n_components=out_dim)
        # result = tsne.fit_transform(x)
        # return result

    else:
        x = tf.constant(x)
        s, u, v = tf.linalg.svd(x)
        return tf.matmul(u[:, :out_dim], tf.linalg.diag(s[:out_dim])).numpy()


def calculate_feature_lower_dim(feature, reduction_dimension=2, pca=True, regulation=True):
    current_dimension = feature.shape[1]
    if reduction_dimension < current_dimension:
        feature_vector_low_dim = reduce_dim(
            feature,
            out_dim=reduction_dimension,
            pca=pca,
        )
    else:
        feature_vector_low_dim = feature

    if regulation:
        feature_vector_low_dim = (feature_vector_low_dim - np.min(feature_vector_low_dim))/(
            np.max(feature_vector_low_dim) - np.min(feature_vector_low_dim))

    return feature_vector_low_dim


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


class GraphConv(keras.layers.Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs,
                 ):

        super(GraphConv, self).__init__(**kwargs)
        self.activation = activations.get(activation)
        self.units = units
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

    def build(self, input_shape):
        feature_shape = input_shape[0]

        self.kernel = self.add_weight(
            shape=(feature_shape[-1], self.units),
            initializer=self.kernel_initializer,
            name='kernel',
            regularizer=self.kernel_regularizer
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.kernel_regularizer
            )
        else:
            self.bias = None

    def call(self, inputs):
        if (not type(inputs) == list) or (not len(inputs) == 2):
            raise ValueError(
                'Input of `GraphConv` layers should be a list with length of 2. (features, A)')

        features = inputs[0]
        A = inputs[1]
        output = A @ features @ self.kernel
        if self.use_bias:
            output += self.bias

        return self.activation(output)

    def get_config(self):
        config = {"activation": self.activation,
                  "units": self.units,
                  "use_bias": self.use_bias,
                  "kernel_initializer": self.kernel_initializer,
                  "bias_initializer": self.bias_initializer,
                  "kernel_regularizer": self.kernel_regularizer,
                  "bias_regularizer": self.bias_regularizer}
        base_config = super(GraphConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TrainableAdjMatrix(keras.layers.Layer):
    def __init__(self, units,
                 activation=None,
                 process=True,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 **kwargs):
        super(TrainableAdjMatrix, self).__init__(**kwargs)
        self.units = units
        self.process = process
        self.activation = activations.get(activation)
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        self.A = self.add_weight(
            shape=self.units,
            initializer=self.kernel_initializer,
            name='adj_matrix',
            regularizer=self.kernel_regularizer,
        )

    def call(self, inputs):
        if not self.process:
            return self.activation(self.A)
        else:
            A = self.activation(self.A)
            D = K.sum(A, axis=-1, keepdims=False)
            A_ = tf.stack([A[i]/D[i] for i in range(self.units[0])])
            return A_

    def get_config(self):
        config = {"activation": self.activation,
                  "units": self.units,
                  "process": self.process,
                  "kernel_initializer": self.kernel_initializer,
                  "kernel_regularizer": self.kernel_regularizer}
        base_config = super(TrainableAdjMatrix, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
