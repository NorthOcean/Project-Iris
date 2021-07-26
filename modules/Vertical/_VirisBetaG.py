"""
@Author: Conghao Wong
@Date: 2021-07-22 14:41:43
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-22 16:04:31
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import List, Tuple
import tensorflow as tf
from modules import applications as A
from modules import models as M
from tensorflow import keras

from ._args import VArgs
from ._layers import ContextEncoding, FFTlayer, IFFTlayer, TrajEncoding
from ._utils import Utils as U
from ._VirisBeta import VIrisBetaModel


class Encoder(keras.Model):
    """
    Encoder
    -------
    Encoder in Generative `Vertical`
    - two stage model
    - first stage: important points
    - second stage: interpolation (<- used in this model)
    """

    def __init__(self, Args: VArgs,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.fft = FFTlayer()

        self.te = TrajEncoding(units=64,
                               activation=tf.nn.tanh,
                               useFFT=True)

        self.ce = ContextEncoding(units=64,
                                  output_channels=Args.obs_frames,
                                  activation=tf.nn.tanh)

        self.transformer = A.Transformer(num_layers=4,
                                         d_model=128,
                                         num_heads=8,
                                         dff=512,
                                         input_vocab_size=None,
                                         target_vocab_size=4,
                                         pe_input=Args.obs_frames,
                                         pe_target=Args.obs_frames + Args.pred_frames,
                                         include_top=False)

    def call(self, trajs, maps,
             points, points_index,
             training=None, mask=None) -> tf.Tensor:
        """
        :param trajs: obs traj, shape = (batch, obs, 2)
        :param maps: traj maps, shape = (batch, a, a)
        :param points: important points output from alpha model.
            shape = (batch, n, 2)
        :param point_index: time positions of important points.
            shape = (n).
            For example, `tf.Tensor([0, 6, 11])`

        :return p_fft: latent feature in the `freq domain`.
            shape = (batch, obs+pred, 128)
        """

        traj_feature = self.te.call(trajs)
        context_feature = self.ce.call(maps)

        # transformer inputs shape = (batch, obs, 128)
        t_inputs = tf.concat([traj_feature, context_feature], axis=-1)

        # transformer target shape = (batch, obs+pred, 4)
        points_index = tf.concat([[-1], points_index], axis=0)
        points = tf.concat([trajs[:, -1:, :], points], axis=1)

        # add the last obs point to finish linear interpolation
        linear_pred = U.LinearInterpolation(points_index, points)
        traj = tf.concat([trajs, linear_pred], axis=-2)
        lfft_r, lfft_i = self.fft.call(traj)
        t_outputs = tf.concat([lfft_r, lfft_i], axis=-1)

        # output shape = (batch, obs+pred, 128)
        me, mc, md = A.create_transformer_masks(t_inputs, t_outputs)
        p_fft, _ = self.transformer.call(t_inputs, t_outputs, True,
                                         me, mc, md)

        return p_fft


class Decoder(keras.Model):
    """
    Decoder
    -------
    Decoder in Generative `Vertical`
    - two stage model
    - first stage: important points
    - second stage: interpolation (<- used in this model) 
    """

    def __init__(self, Args: VArgs, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.args = Args
        self.ifft = IFFTlayer()
        self.fc1 = keras.layers.Dense(128, activation=tf.nn.relu)
        self.fc2 = keras.layers.Dense(4)

    def call(self, features, noise, training=None, mask=None):
        """
        :param features: latent features in freq domain. $ f \sim N(0, I)$, shape is `(batch, obs+pred, 128)`
        :param noise: noise vector, shape is `(..., 128)`

        :return pred: prediction, shape is `(batch, pred, 2)`
        """
        features += noise
        fc1 = self.fc1(features)
        fc2 = self.fc2(fc1 + features)

        pred = self.ifft.call(real=fc2[:, :, :2],
                              imag=fc2[:, :, 2:])

        return pred[:, self.args.obs_frames:, :]


class VIrisBetaGModel(M.prediction.Model):
    """
    VIrisBetaGModel
    ---


    """

    def __init__(self, Args: VArgs,
                 points: int,
                 asSecondStage=False,
                 p_index: str = None,
                 training_structure=None,
                 *args, **kwargs):

        super().__init__(Args, training_structure, *args, **kwargs)

        # pre-process
        self.set_preprocess('move')
        self.set_preprocess_parameters(move=0)

        # parameters
        self.n_pred = points
        self.asSecondStage = asSecondStage

        if self.asSecondStage and p_index:
            pi = [int(i) for i in p_index.split('_')]
            self.points_index = tf.cast(pi, tf.float32)

        # layers
        self.encoder = Encoder(self.args)
        self.decoder = Decoder(self.args)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor],
             points: tf.Tensor,
             points_index: tf.Tensor,
             training=None, mask=None,
             *args, **kwargs):

        K = self.args.K_train if training else self.args.K
        sigma = 1.0 if training else self.args.sigma
        trajs, maps = inputs[:2]

        features = self.encoder.call(trajs, maps, points, points_index)

        samples = []
        for _ in range(K):
            z = tf.random.normal(features.shape, 0.0, sigma)
            samples.append(self.decoder.call(features, noise=z))

        # shape = (batch, K, pred, 2)
        pred = tf.transpose(tf.stack(samples), [1, 0, 2, 3])

        return (pred, features)

    def forward(self, model_inputs: List[tf.Tensor],
                training=None,
                *args, **kwargs) -> List[tf.Tensor]:

        return VIrisBetaModel.forward(self, model_inputs,
                                      training,
                                      *args, **kwargs)


class VIrisBetaG(M.prediction.Structure):
    def __init__(self, Args: List[str], *args, **kwargs):
        super().__init__(Args, *args, **kwargs)

        self.args = VArgs(Args)

        self.set_model_inputs('trajs', 'maps', 'paras', 'gt')
        self.set_model_groundtruths('gt')

        self.set_loss('ade', 'diff', self.p_loss)
        self.set_loss_weights(0.8, 0.2, 1.0)

        self.set_metrics('ade', 'fde')
        self.set_metrics_weights(0.7, 0.3)

    def create_model(self, *args, **kwargs):
        model = VIrisBetaGModel(self.args,
                                points=self.args.points,
                                training_structure=self,
                                *args, **kwargs)

        opt = keras.optimizers.Adam(self.args.lr)
        return model, opt

    def load_forward_dataset(self, model_inputs: List[tf.Tensor], **kwargs):
        trajs = model_inputs[0]
        maps = model_inputs[1]
        paras = model_inputs[2]
        proposals = model_inputs[3]
        return tf.data.Dataset.from_tensor_slices((trajs, maps, paras, proposals))

    def p_loss(self, model_outputs: Tuple[tf.Tensor], labels=None):
        features = tf.reshape(model_outputs[1], [-1, 128])

        mu_real = tf.reduce_mean(features, axis=0)  # (128)
        std_real = tf.math.reduce_std(features, axis=0)  # (128)

        return tf.reduce_mean(tf.abs(mu_real - 0)) + tf.reduce_mean(tf.abs(std_real - 1))
