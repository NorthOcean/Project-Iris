"""
@Author: Conghao Wong
@Date: 2021-07-08 20:58:48
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-12 15:06:00
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from argparse import Namespace
from typing import List, Tuple

import tensorflow as tf
from tensorflow import keras

from .. import applications as A
from .. import models as M
from ._args import VArgs
from ._layers import ContextEncoding, TrajEncoding
from ._utils import Utils as U

from ..satoshi._alpha_transformer import SatoshiAlphaTransformerModel as VIrisAlphaModel


class __VIrisAlphaModel(M.prediction.Model):
    def __init__(self, Args,
                 training_structure=None,
                 *args, **kwargs):

        super().__init__(Args, training_structure,
                         *args, **kwargs)

        # Preprocess
        self.set_preprocess('move')
        self.set_preprocess_parameters(move=0)

        # Layers
        self.concat = keras.layers.Concatenate()

        self.te = TrajEncoding(units=64,
                               activation=tf.nn.tanh,
                               useFFT=True)

        self.ce = ContextEncoding(units=64,
                                  output_channels=self.args.obs_frames,
                                  activation=tf.nn.tanh)

        self.transformer = A.Transformer(num_layers=4,
                                         d_model=128,
                                         num_heads=8,
                                         dff=512,
                                         input_vocab_size=None,
                                         target_vocab_size=4,
                                         pe_input=Args.obs_frames,
                                         pe_target=Args.obs_frames + Args.pred_frames,
                                         include_top=True)

    def call(self, inputs, training=None, mask=None):
        # TODO
        pass

    def forward(self, model_inputs: List[tf.Tensor],
                training=None,
                *args, **kwargs):

        return U.forward(self, model_inputs, training, *args, **kwargs)


class VIrisAlpha(M.prediction.Structure):

    def __init__(self, Args: List[str], *args, **kwargs):
        super().__init__(Args, *args, **kwargs)
        
        self.args = VArgs(Args)
        
        self.set_model_inputs('traj', 'maps')
        self.set_model_groundtruths('destination')

        self.set_loss(self.min_FDE)
        self.set_loss_weights(1.0)

        self.set_metrics(self.min_FDE)
        self.set_metrics_weights(1.0)

    def create_model(self, model_type=None,
                     *args, **kwargs):

        if model_type is None:
            model_type = VIrisAlphaModel
            
        model = model_type(self.args, 
                           training_structure=self,
                           *args, **kwargs)

        opt = keras.optimizers.Adam(self.args.lr)
        return model, opt

    def min_FDE(self, outputs: List[tf.Tensor],
                labels: tf.Tensor) -> tf.Tensor:
        """
        Calculate minimun FDE

        :param outputs: a list of tensor, and `outputs[0].shape` is `(batch, K, 2)`
        :param labels: labels, shape is `(batch, pred, 2)`
        """
        # shape = (batch, K)
        distance = tf.linalg.norm(outputs[0] - labels[:, -1:, :], axis=-1)
        return tf.reduce_mean(tf.reduce_min(distance, axis=-1))
