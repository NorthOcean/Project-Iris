"""
@Author: Conghao Wong
@Date: 2021-06-24 09:14:08
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-20 11:00:12
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from modules.models.helpmethods import BatchIndex
from tensorflow import keras as keras
from tqdm import tqdm

from ._alpha import MSNAlpha, MSNAlphaModel
from ._args import MSNArgs
from ._beta_D import MSNBeta_D
from ._beta_G import MSNBeta_G


class _MSNAlphaModelPlus(MSNAlphaModel):
    """
    A specific `MSNAlpha` model that guides `MSNBeta_G` as the second stage model.
    """

    def __init__(self, Args, training_structure=None, *args, **kwargs):
        super().__init__(Args, training_structure=training_structure, *args, **kwargs)

    def post_process(self, outputs: Tuple[tf.Tensor],
                     training=None,
                     **kwargs) -> Tuple[tf.Tensor]:

        # shape = [(batch, K, 2)]
        outputs = super().post_process(outputs, training=training, **kwargs)

        if training:
            return outputs

        # shape = [(batch, obs, 2), (batch, 100, 100), None]
        model_inputs = kwargs['model_inputs']
        batch, Kc = outputs[0].shape[:2]
        pred = self.args.pred_frames
        K = self.args.K
        des = tf.reshape(outputs[0], [-1, 1, 2])

        # prepare new inputs into beta model
        # new batch_size (total) is batch*Kc
        beta_results = []
        batch_size = self.args.max_batch_size // Kc
        batch_index = BatchIndex(batch_size, batch)

        for index in tqdm(batch_index.index):
            [k0, k1, k] = index
            beta_inputs = [tf.repeat(inp[k0:k1], Kc, axis=0)
                           for inp in model_inputs]
            beta_inputs.append(des[k0*Kc: k1*Kc])

            beta_results.append(self.training_structure.beta(
                beta_inputs,
                return_numpy=False)[0][:, :, -1:, :])

        # choose beta destinations
        beta_des = tf.concat(beta_results, axis=0)
        beta_des = tf.reshape(beta_des, [-1, 1, 2])  # (batch*Kc*K, 1, 2)

        if self.args.linear or self.args.loadc.startswith('l'):
            # prepare new inputs into linear model

            start = model_inputs[0][:, -1, :][:, tf.newaxis, :]
            stop = tf.reshape(beta_des, [-1, Kc*K, 2])
            pred = tf.linspace(start, stop, pred+1, axis=-2)[:, :, 1:, :]

            # shape = (batch, K*K_c, pred, 2)
            final_results = pred

        else:
            # prepare new inputs into gamma model
            # new batch_size is batch*Kc*K
            batch_size = self.args.max_batch_size // (K * Kc)
            batch_index = BatchIndex(batch_size, batch)

            final_results = []
            for index in tqdm(batch_index.index):
                [k0, k1, k] = index
                gamma_inputs = [tf.repeat(inp[k0:k1], K*Kc, axis=0)
                                for inp in model_inputs]
                gamma_inputs.append(beta_des[k0*K*Kc:k1*K*Kc])

                # feed into gamma model
                # output shape = (batch*Kc*K, pred, 2)
                gamma_results = self.training_structure.gamma(
                    gamma_inputs,
                    return_numpy=False)[0]

                final_results.append(gamma_results)

            # re-organize outputs
            # shape = (batch, K*K_c, pred, 2)
            final_results = tf.reshape(
                tf.concat(final_results, axis=0),
                [-1, K*Kc, self.args.pred_frames, 2])

        # check failure cases
        if self.args.check:
            final_results = angle_check(pred=final_results,
                                        obs=model_inputs[0],
                                        max_angle=135)

        return (final_results,)


class MSN_G(MSNAlpha):
    """
    Structure for MSN_G prediction
    -----------------------------
    Please train `MSNAlphaModel`, `MSBBeta_DModel`, `MSNBeta_GModel`, 
    and pass their paths with args `--loada`, `--loadb` ,and `--loadc`.
    """

    alpha_model = _MSNAlphaModelPlus
    beta_model = MSNBeta_G
    gamma_model = MSNBeta_D

    def __init__(self, Args: List[str], *args, **kwargs):
        super().__init__(Args, *args, **kwargs)

        # set inputs and groundtruths
        self.set_model_inputs('trajs', 'maps', 'map_paras')
        self.set_model_groundtruths('gt')

        # set metrics
        self.set_metrics('ade', 'fde')
        self.set_metrics_weights(1.0, 0.0)

        # assign models
        self.alpha = self
        self.beta = self.beta_model(Args)
        self.gamma = self.gamma_model(Args)
        self.linear_predict = False

        # load gamma weights
        if ('null' in [self.args.loada, self.args.loadb]) and \
                (self.args.loadc == 'null' and self.args.linear == 0):
            self.logger.error(e := ('`MSNAlpha` or `MSNBeta_G` or `MSNBeta_D` not' +
                              ' found! Please specific their paths via' +
                                    ' `--loada` or `--loadb`.'))
            raise FileNotFoundError(e)

        if self.args.loadc.startswith('l'):
            self.linear_predict = True
        else:
            self.gamma.args = MSNArgs(
                self.gamma.load_args(Args, self.args.loadc))
            self.gamma.model = self.gamma.load_from_checkpoint(self.args.loadc)

        # load other weights
        self.alpha.args = MSNArgs(
            args=self.alpha.load_args(Args, self.args.loada),
            default_args=self.args._args
        )
        self.beta.args = MSNArgs(self.beta.load_args(Args, self.args.loadb))

        self.alpha.model = self.alpha.load_from_checkpoint(self.args.loada)
        self.beta.model = self.beta.load_from_checkpoint(self.args.loadb)

    def run_train_or_test(self):
        self.log(
            'Start test model from `{}` and `{}`'.format(
                self.args.loada, self.args.loadb))

        self.run_test()

    def create_model(self, model_type=None):
        return super().create_model(model_type=self.alpha_model)

    def print_test_results(self, loss_dict, **kwargs):
        dataset = kwargs['dataset_name']
        self.print_parameters(title='test results', **
                            dict({'dataset': dataset}, **loss_dict))

        self.log('Results from {}, {}, {}, {}, {}, K={}, sigma={}'.format(
            self.args.loada,
            self.args.loadb,
            self.args.loadc,
            dataset,
            loss_dict,
            self.args.K,
            self.args.sigma))


def angle_check(pred: tf.Tensor, obs: tf.Tensor, max_angle=135):
    """
    Check angle of predictions, and remove error ones.

    :param pred: predictions, shape = (batch, K, pred, 2)
    :param obs: observations, shape = (batch, obs, 2)
    :return pred_checked: predictions without wrong ones,
        shape = (batch, K, pred, 2)
    """
    obs = obs[:, tf.newaxis, :, :]

    obs_vec = obs[:, :, -1, :] - obs[:, :, 0, :]
    pred_vec = pred[:, :, -1, :] - pred[:, :, 0, :]

    dot = tf.reduce_sum(obs_vec * pred_vec, axis=-1)
    len_obs = tf.linalg.norm(obs_vec, axis=-1)
    len_pred = tf.linalg.norm(pred_vec, axis=-1)

    cosine = (dot + 0.0001) / ((len_obs * len_pred) + 0.0001)

    mask = tf.cast(cosine > tf.cos(max_angle/180 * 3.1415926),
                   tf.float32)   # (batch, K)
    true_item = tf.gather_nd(
        pred,
        tf.transpose([tf.range(0, obs.shape[0]),
                      tf.argsort(mask, axis=-1)[:, -1]])
    )[:, tf.newaxis, :, :]

    mask = mask[:, :, tf.newaxis, tf.newaxis]
    pred_checked = pred * mask + true_item * (1.0 - mask)
    return pred_checked
