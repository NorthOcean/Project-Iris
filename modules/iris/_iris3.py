"""
@Author: Conghao Wong
@Date: 2021-06-24 09:14:08
@LastEditors: Conghao Wong
@LastEditTime: 2021-06-24 20:08:08
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import Dict, List, Tuple

import modules.models as M
import numpy as np
import tensorflow as tf
from tensorflow import keras as keras

from ._alpha import IrisAlpha, IrisAlphaModel
from ._beta_cvae import IrisBetaCVAE as IrisBeta
from ..satoshi._beta_transformer import SatoshiBetaTransformer as IrisGamma
from ..satoshi._args import SatoshiArgs


class Iris3Model(IrisAlphaModel):

    def __init__(self, Args, training_structure=None, *args, **kwargs):
        super().__init__(Args, training_structure=training_structure, *args, **kwargs)

    def post_process(self, outputs: Tuple[tf.Tensor],
                     training=False,
                     **kwargs) -> Tuple[tf.Tensor]:

        # shape = [(batch, K, 2)]
        outputs = super().post_process(outputs, training=training, **kwargs)

        if training:
            return outputs

        # shape = [(batch, obs, 2), (batch, 100, 100), None]
        model_inputs = kwargs['model_inputs']
        Kc = outputs[0].shape[1]
        K = self.args.K

        # prepare new inputs into beta model
        # new batch_size (total) is batch*Kc
        Kc_per_batch = self.args.max_batch_size // self.args.batch_size
        beta_results = []
        for k_point in range(0, Kc, Kc_per_batch):
            k_start = k_point
            k_end = min(k_point + Kc_per_batch, Kc)
            k = k_end - k_start

            beta_inputs = [tf.repeat(inp, k, axis=0) for inp in model_inputs]
            beta_inputs.append(tf.reshape(outputs[0][:, k_start:k_end, :], [-1, 1, 2]))
            beta_results.append(self.training_structure.beta(
                beta_inputs,
                return_numpy=False)[0][:, :, -1:, :])

        # choose beta destinations
        beta_des = tf.concat(beta_results, axis=1)
        beta_des = tf.reshape(beta_des, [-1, 1, 2])  # (batch*Kc*K, 1, 2)

        if self.args.linear or self.args.loadc.startswith('l'):
            # prepare new inputs into linear model
            obs_pos = model_inputs[0][:, -1, :][:,
                                                tf.newaxis, :]   # (batch, 1, 2)

            # (batch, Kc*K, 2)
            des_pos = tf.reshape(beta_des, [-1, Kc*K, 2])

            final_results = []
            p = self.args.pred_frames
            for step in range(1, p+1):
                pred = (des_pos - obs_pos) * step/p + obs_pos   # (batch, K, 2)
                final_results.append(pred)

            # shape = (batch, K*K_c, pred, 2)
            final_results = tf.transpose(final_results, [1, 2, 0, 3])

        else:
            # prepare new inputs into gamma model
            # new batch_size is batch*Kc*K
            gamma_inputs = [tf.repeat(inp, K, axis=0) for inp in beta_inputs[:-1]]
            gamma_inputs.append(beta_des)

            # feed into gamma model
            # output shape = (batch*Kc*K, pred, 2)
            gamma_results = self.training_structure.gamma(
                gamma_inputs,
                return_numpy=False)[0]

            # re-organize outputs
            # shape = (batch, K*K_c, pred, 2)
            final_results = tf.reshape(
                beta_results,
                [-1, K*Kc, self.args.pred_frames, 2])

        return M.prediction.Process.update((final_results,), outputs)


class Iris3(IrisAlpha):

    alpha_model = Iris3Model
    beta_model = IrisBeta
    gamma_model = IrisGamma

    def __init__(self, args, arg_type=SatoshiArgs):
        super().__init__(args, arg_type=arg_type)

        # set inputs and groundtruths
        self.set_model_inputs('trajs', 'maps', 'map_paras')
        self.set_model_groundtruths('gt')

        # set metrics
        self.set_metrics('ade', 'fde')
        self.set_metrics_weights(1.0, 0.0)

        # assign models
        self.alpha = self
        self.beta = self.beta_model(args)
        self.gamma = self.gamma_model(args)

        # load gamma weights
        if ('null' in [self.args.loada, self.args.loadb]) and \
                (self.args.loadc == 'null' and self.args.linear == 0):
            self.logger.error(e := ('`IrisAlpha` or `IrisBeta` or `IrisGamma` not' + 
                              ' found! Please specific their paths via' + 
                              ' `--loada` or `--loadb`.'))
            raise FileNotFoundError(e)

        if self.args.loadc.startswith('l'):
            self.args.linear = 1
        else:
            self.gamma.load_args(args, args.loadc, SatoshiArgs)
            self.gamma._model = self.gamma.load_from_checkpoint(args.loadc)

        # load other weights
        self.alpha.load_args(args, args.loada, SatoshiArgs)
        self.beta.load_args(args, args.loadb, SatoshiArgs)
        self.alpha._model = self.alpha.load_from_checkpoint(args.loada)
        self.beta._model = self.beta.load_from_checkpoint(args.loadb)

    def run_train_or_test(self):
        self.logger.info(
            'Start test model from `{}` and `{}`'.format(
                self.args.loada, self.args.loadb))

        self.run_test()

    def create_model(self, model_type=None):
        return super().create_model(model_type=self.alpha_model)

    def print_test_result_info(self, loss_dict, **kwargs):
        dataset = kwargs['dataset_name']
        self.log_parameters(title='test results', **
                            dict({'dataset': dataset}, **loss_dict))

        self.logger.info('Results from {}, {}, {}, {}, {}, K={}, sigma={}'.format(
            self.args.loada,
            self.args.loadb,
            self.args.loadc,
            dataset,
            loss_dict,
            self.args.K,
            self.args.sigma))