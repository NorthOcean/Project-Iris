"""
@Author: Conghao Wong
@Date: 2021-12-21 15:03:58
@LastEditors: Conghao Wong
@LastEditTime: 2021-12-31 10:36:06
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import List

import tensorflow as tf
from tensorflow import keras

from ... import applications as A
from ... import models as M
from ..__args import HandlerArgs
from ..__baseHandler import BaseHandlerModel, BaseHandlerStructure
from ..__layers import OuterLayer


class BurnwoodModel(BaseHandlerModel):
    """
    Second stage `Silverballers` model.
    """

    def __init__(self, Args: HandlerArgs,
                 feature_dim: int,
                 points: int,
                 asHandler=False,
                 key_points: str = None,
                 training_structure=None,
                 *args, **kwargs):

        super().__init__(Args, feature_dim, points,
                         asHandler=asHandler,
                         key_points=key_points,
                         training_structure=training_structure,
                         *args, **kwargs)

        # Parameters
        self.d = feature_dim
        self.points = points
        self.key_points = key_points
        self.steps = self.args.obs_frames + self.args.pred_frames

        # Layers
        self.fft = A.layers.FFTlayer()

        self.linear = A.layers.LinearInterpolation()

        self.te = A.layers.TrajEncoding(units=self.d//2,
                                        activation=tf.nn.tanh,
                                        useFFT=True)

        self.ce = A.layers.ContextEncoding(units=self.d//2,
                                           output_channels=self.steps,
                                           activation=tf.nn.tanh)

        self.outer = OuterLayer(self.d, self.d, reshape=True)
        self.outer_fc = keras.layers.Dense(self.d, tf.nn.tanh)

        self.T = A.TransformerEncoder(num_layers=4, num_heads=8,
                                      dim_model=self.d, dim_forward=512,
                                      steps=self.steps,
                                      dim_output=4,
                                      include_top=True)

        self.ifft = A.layers.IFFTlayer()

    def call(self, inputs: List[tf.Tensor],
             keypoints: tf.Tensor,
             keypoints_index: tf.Tensor,
             training=None, mask=None,
             *args, **kwargs):
        """
        Run the Burnwood model

        :param inputs: a list of tensors, which includes `trajs` and `maps`
            - trajs, shape = `(batch, obs, 2)`
            - maps, shape = `(batch, a, a)`

        :param keypoints: predicted keypoints, shape is `(batch, n_k, 2)`
        :param keypoints_index: index of predicted keypoints, shape is `(n_k)`
        :param training: controls run as the training mode or the test mode

        :return predictions: predictions, shape = `(batch, pred, 2)`
        """

        # Unpack inputs
        trajs, maps = inputs[:2]

        # Concat keypoints with the last observed point
        keypoints_index = tf.concat([[-1], keypoints_index], axis=0)
        keypoints = tf.concat([trajs[:, -1:, :], keypoints], axis=1)

        # Calculate linear interpolation and concat -> (batch, obs+pred, 2)
        # linear shape = (batch, pred, 2)
        linear = self.linear.call(keypoints_index, keypoints)
        trajs = tf.concat([trajs, linear], axis=-2)

        # Encode trajectory features and context features
        traj_feature = self.te.call(trajs)      # (batch, obs+pred, d/2)
        context_feature = self.ce.call(maps)    # (batch, obs+pred, d/2)
        f = tf.concat([traj_feature, context_feature], axis=-1)

        # Outer product
        f = self.outer.call(f, f)
        f = self.outer_fc(f)

        # Encode features with Transformer Encoder
        # (batch, obs+pred, 4)
        p_fft = self.T.call(inputs=f, training=training)
        p = self.ifft.call(real=p_fft[:, :, :2], imag=p_fft[:, :, 2:])

        return p[:, self.args.obs_frames:, :]


class Burnwood(BaseHandlerStructure):

    def __init__(self, Args: List[str],
                 association: M.prediction.Structure = None,
                 *args, **kwargs):

        super().__init__(Args, association=association, *args, **kwargs)
        self.set_model_type(new_type=BurnwoodModel)
