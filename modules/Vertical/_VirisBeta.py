"""
@Author: Conghao Wong
@Date: 2021-07-08 15:45:53
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-09 15:57:03
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
from ..satoshi._args import SatoshiArgs
from ..satoshi._beta_transformer import SatoshiBetaTransformerModel
from ._args import VArgs
from ._layers import (ContextEncoding, FFTlayer, IFFTlayer, LinearPrediction,
                      TrajEncoding)
from ._utils import Utils as U


class VIrisBetaModel(M.prediction.Model):
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

        self.lp = LinearPrediction(useFFT=True, include_obs=True)

        self.transformer = A.Transformer(num_layers=4,
                                         d_model=128,
                                         num_heads=8,
                                         dff=512,
                                         input_vocab_size=None,
                                         target_vocab_size=4,
                                         pe_input=Args.obs_frames,
                                         pe_target=Args.obs_frames + Args.pred_frames,
                                         include_top=True)

        self.decoder = IFFTlayer()

    def call(self, inputs: List[tf.Tensor],
             outputs: tf.Tensor = None,
             training=None, mask=None):

        # unpack inputs
        trajs, maps, _, destinations = inputs[:4]

        traj_feature = self.te(trajs)
        context_feature = self.ce(maps)

        # transformer inputs shape = (batch, obs, 128)
        t_inputs = self.concat([traj_feature, context_feature])

        # transformer target shape = (batch, obs+pred, 4)
        t_outputs = self.lp(start=trajs[:, -1:, :],
                            end=destinations[:, :1, :],
                            n=self.args.pred_frames,
                            obs=trajs)

        # transformer output shape = (batch, obs+pred, 4)
        me, mc, md = A.create_transformer_masks(t_inputs, t_outputs)
        p_fft, _ = self.transformer(t_inputs, t_outputs, True,
                                    me, mc, md)

        # decode
        p = self.decoder(real=p_fft[:, :, :2],
                         imag=p_fft[:, :, 2:])

        return p[:, self.args.obs_frames:, :]

    def forward(self, model_inputs: List[tf.Tensor],
                training=None,
                *args, **kwargs):

        return U.forward(self, model_inputs, training, *args, **kwargs)


class VIrisBeta(M.prediction.Structure):
    def __init__(self, Args: List[str], *args, **kwargs):
        super().__init__(Args, *args, **kwargs)

        self.args = VArgs(Args)

        self.set_model_inputs('trajs', 'maps', 'paras', 'destinations')
        self.set_model_groundtruths('gt')

        self.set_loss('ade', 'diff')
        self.set_loss_weights(0.8, 0.2)

        self.set_metrics('ade', 'fde')
        self.set_metrics_weights(1.0, 0.0)

    def create_model(self, *args, **kwargs):
        model = VIrisBetaModel(self.args, training_structure=self)
        opt = keras.optimizers.Adam(self.args.lr)
        return model, opt

    def load_forward_dataset(self, model_inputs: Tuple[tf.Tensor], **kwargs):
        trajs = model_inputs[0]
        maps = model_inputs[1]
        paras = model_inputs[2]
        proposals = model_inputs[3]
        return tf.data.Dataset.from_tensor_slices((trajs, maps, paras, proposals))

    def print_test_result_info(self, loss_dict, dataset_name, **kwargs):
        self.log_parameters(title='rest results',
                            **dict({'dataset': dataset_name}, **loss_dict))

        self.logger.info('Results: {}, {}, {}.'.format(
            self.args.load,
            dataset_name,
            loss_dict
        ))
