"""
@Author: Conghao Wong
@Date: 2021-07-27 19:06:00
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-30 11:18:04
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import List
from .. import applications as A
from .. import models as M
from ._args import VArgs
from ._layers import ContextEncoding, GraphConv, TrajEncoding

import tensorflow as tf
from tensorflow import keras


class VEncoder(keras.Model):

    def __init__(self, Args: VArgs, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Layers
        self.te = TrajEncoding(units=64,
                               activation=tf.nn.tanh,
                               useFFT=False)

        self.ce = ContextEncoding(units=64,
                                  output_channels=Args.obs_frames,
                                  activation=tf.nn.tanh)

        self.transformer = A.Transformer(num_layers=4,
                                         d_model=128,
                                         num_heads=8,
                                         dff=512,
                                         input_vocab_size=None,
                                         target_vocab_size=None,
                                         pe_input=Args.obs_frames,
                                         pe_target=Args.obs_frames,
                                         include_top=False)

        self.gcn = GraphConv(units=128,
                             activation=None)

        self.adj_fc = keras.layers.Dense(Args.Kc, tf.nn.tanh)

    def call(self, trajs: tf.Tensor,
             maps: tf.Tensor,
             training=None, mask=None):
        """
        Encode all inputs into the latent space vector

        :param trajs: trajectories, shape = `(batch, obs, 2)`
        :param maps: context maps, shape = `(batch, a, a)`
        :return features: features in the latent space, shape = `(batch, Kc, 128)`
        """

        traj_feature = self.te.call(trajs)
        context_feature = self.ce.call(maps)

        # transformer inputs shape = (batch, obs, 128)
        t_inputs = tf.concat([traj_feature, context_feature], axis=-1)
        t_outputs = trajs

        # transformer
        # output shape = (batch, obs, 128)
        t_features, _ = self.transformer.call(t_inputs, 
                                              t_outputs,
                                              training=training)

        # multi-style prediction
        adj = tf.transpose(self.adj_fc(t_inputs), [0, 2, 1])
        # (batch, Kc, 128)
        return self.gcn.call(features=t_features,
                             adjMatrix=adj)


class VDecoder(keras.Model):

    def __init__(self, Args: VArgs,
                 pred_number: int,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Generator layers
        self.g1 = keras.layers.Dense(128, activation=tf.nn.relu)

        # Layers
        self.fc1 = keras.layers.Dense(128, activation=tf.nn.tanh)
        self.fc2 = keras.layers.Dense(pred_number * 2)
        self.reshape = keras.layers.Reshape([Args.Kc, pred_number, 2])

    def call(self, features: tf.Tensor,
             noise: tf.Tensor,
             training=None, mask=None) -> tf.Tensor:
        """
        Decode features, and generate multiple predictions 
        by adding randomly sampled Gaussian noise.

        :param features: features from the latent space, shape = `(batch, Kc, 128)`
        :param noise: sampled Gaussian noise, shape = `(batch, Kc, 128)`
        :return trajs: output trajectories, shape = `(batch, Kc, N, 2)`
        """

        f = self.g1(features + noise)
        f = self.fc1(f)
        f = self.fc2(f)
        return self.reshape(f)


class VIrisAlphaGModel(M.prediction.Model):

    def __init__(self, Args: VArgs,
                 pred_number: int,
                 training_structure=None,
                 *args, **kwargs):

        super().__init__(Args, training_structure, *args, **kwargs)

        self.args = Args

        # Args
        self.n_pred = pred_number

        # Preprocess
        self.set_preprocess('move')
        self.set_preprocess_parameters(move=0)

        # Layers
        self.encoder = VEncoder(Args)
        self.decoder = VDecoder(Args, pred_number)

    def call(self, inputs: List[tf.Tensor],
             training=None, mask=None):

        # unpack inputs
        trajs, maps = inputs[:2]
        K = self.args.K_train if training else self.args.K
        sigma = 1.0 if training else self.args.sigma

        # encode
        features = self.encoder.call(trajs, maps)

        # decode
        predictions = []
        for _ in range(K):
            z = tf.random.normal(features.shape, 0.0, sigma)
            predictions.append(self.decoder.call(features, z))

        # shape = (batch, Kc*K, N, 2)
        predictions = tf.concat(predictions, axis=1)

        return (predictions, features)


class VIrisAlphaG(M.prediction.Structure):

    def __init__(self, Args: List[str], *args, **kwargs):
        super().__init__(Args, *args, **kwargs)

        self.args = VArgs(Args)
        
        self.important_args += ['Kc', 'K_train', 'p_index', 'K']

        self.set_model_inputs('traj', 'maps')
        self.set_model_groundtruths('gt')

        self.set_loss(self.l2_loss, self.p_loss)
        self.set_loss_weights(1.0, 1.0)

        self.set_metrics(self.min_FDE)
        self.set_metrics_weights(1.0)

    @property
    def p_index(self) -> tf.Tensor:
        p_index = [int(i) for i in self.args.p_index.split('_')]
        return tf.cast(p_index, tf.int32)

    @property
    def p_len(self) -> int:
        return len(self.p_index)

    def create_model(self, *args, **kwargs):

        model = VIrisAlphaGModel(self.args,
                                 pred_number=self.p_len,
                                 training_structure=self,
                                 *args, **kwargs)

        opt = keras.optimizers.Adam(self.args.lr)
        return model, opt

    def l2_loss(self, outputs: List[tf.Tensor],
                labels: tf.Tensor,
                *args, **kwargs) -> tf.Tensor:

        labels_pickled = tf.gather(labels, self.p_index, axis=1)
        return M.prediction.Loss.ADE(outputs[0], labels_pickled)

    def p_loss(self, outputs: List[tf.Tensor],
               labels: tf.Tensor = None,
               *args, **kwargs) -> tf.Tensor:
        
        features = tf.reshape(outputs[1], [-1, 128])

        mu_real = tf.reduce_mean(features, axis=0)  # (128)
        std_real = tf.math.reduce_std(features, axis=0)  # (128)

        return (tf.reduce_mean(tf.abs(mu_real - 0)) + 
                tf.reduce_mean(tf.abs(std_real - 1)))

    def min_FDE(self, outputs: List[tf.Tensor],
                labels: tf.Tensor,
                *args, **kwargs) -> tf.Tensor:

        # shape = (batch, Kc*K)
        distance = tf.linalg.norm(
            outputs[0][:, :, -1, :] -
            tf.expand_dims(labels[:, -1, :], 1), axis=-1)

        return tf.reduce_mean(tf.reduce_min(distance, axis=-1))
