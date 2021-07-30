"""
@Author: Conghao Wong
@Date: 2021-07-27 19:06:00
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-30 16:40:19
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import List, Tuple

import tensorflow as tf
from tensorflow import keras

from .. import applications as A
from .. import models as M
from ._args import VArgs
from ._layers import ContextEncoding, GraphConv, TrajEncoding
from ._VirisAlpha import VIrisAlpha


class VEncoder(keras.Model):
    """
    Encoder used in the first stage generative `Vertical-G`
    """

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
    """
    Decoder used in the first stage generative `Vertical-G`
    """

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
    """
    Generative first stage `Vertical-G` model
    """

    def __init__(self, Args: VArgs,
                 pred_number: int,
                 training_structure=None,
                 *args, **kwargs):
        """
        Init a first stage `Vertical-G` model

        :param Args:
        :param pred_number: number of timesteps to predict
        """

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
             training=None, mask=None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Run the first stage generative `Vertical-G` model

        :param inputs: a list of tensors, which includes `trajs` and `maps`
            - trajs, shape = `(batch, obs, 2)`
            - maps, shape = `(batch, a, a)`
            
        :param training: controls run as the training mode or the test mode

        :return predictions: predicted trajectories, shape = `(batch, Kc*K, N, 2)`
        :return features: features in the latent space, shape = `(batch, Kc, 128)`
        """

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


class VIrisAlphaG(VIrisAlpha):
    """
    Training structure for the generative first stage `Vertical-G`
    """

    def __init__(self, Args: List[str], *args, **kwargs):
        super().__init__(Args, *args, **kwargs)

        self.important_args += ['K']

        self.set_loss(self.l2_loss, self.p_loss)
        self.set_loss_weights(1.0, 1.0)

    def create_model(self, *args, **kwargs):
        return VIrisAlpha.create_model(self, VIrisAlphaGModel)

    def p_loss(self, outputs: List[tf.Tensor],
               labels: tf.Tensor = None,
               *args, **kwargs) -> tf.Tensor:
        """
        a simple loss function to make features in line with 
        the normalized Gaussian distribution
        """
        features = tf.reshape(outputs[1], [-1, 128])

        mu_real = tf.reduce_mean(features, axis=0)  # (128)
        std_real = tf.math.reduce_std(features, axis=0)  # (128)

        return (tf.reduce_mean(tf.abs(mu_real - 0)) +
                tf.reduce_mean(tf.abs(std_real - 1)))
