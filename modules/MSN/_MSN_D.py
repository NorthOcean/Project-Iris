"""
@Author: Conghao Wong
@Date: 2021-06-22 16:45:24
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-20 10:05:54
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
from ._beta_D import MSNBeta_D, MSNBeta_DModel


class _MSNAlphaModelPlus(MSNAlphaModel):
    """
    A specific `MSNAlpha` model that guides `MSNBeta_D` as the second stage model.
    """

    def __init__(self, Args: MSNArgs,
                 training_structure,
                 linear_prediction=False,
                 *args, **kwargs):
        """
        :param Args: args used in this model
        :param training_structure: the parent structure of this model
        :param linear_prediction: controls if use the linear prediction as the second stage model
        """

        super().__init__(Args, training_structure,
                         *args, **kwargs)

        self.linear = linear_prediction

    def post_process(self, outputs: List[tf.Tensor],
                     training=None,
                     *args, **kwargs) -> List[tf.Tensor]:

        # shape = [(batch, Kc, 2)]
        outputs = super().post_process(outputs, training, *args, **kwargs)

        if training:
            return outputs

        # obtain shape parameters
        batch, Kc = outputs[0].shape[:2]
        pred = self.args.pred_frames

        # shape = (batch, Kc, 2)
        proposals = outputs[0]
        current_inputs = kwargs['model_inputs']

        if self.linear:
            # Linear interpolation

            # (batch, 1, 2)
            start = current_inputs[0][:, -1, :][:, tf.newaxis, :]  
            stop = proposals   # (batch, Kc, 2)
            pred = tf.linspace(start, stop, pred+1, axis=-2)[:, :, 1:, :]

            return (pred,)  # (batch, Kc, pred, 2)

        else:
            # prepare new inputs into beta model
            # new batch_size (total) is batch*Kc
            batch_size = self.args.max_batch_size // Kc
            batch_index = BatchIndex(batch_size, batch)

            # Flatten inputs
            proposals = tf.reshape(proposals, [batch*Kc, 1, 2])

            beta_results = []
            for index in tqdm(batch_index.index):
                [start, end, length] = index

                # prepare new batch inputs
                beta_inputs = [tf.repeat(inp[start:end], Kc, axis=0)
                               for inp in current_inputs]
                beta_inputs.append(proposals[start*Kc: end*Kc])

                # beta outputs shape = (batch*Kc, pred, 2)
                beta_results.append(self.training_structure.beta(
                    beta_inputs,
                    return_numpy=False)[0])

            beta_results = tf.concat(beta_results, axis=0)
            beta_results = tf.reshape(beta_results, [batch, Kc, pred, 2])
            return (beta_results,)  # (batch, Kc, pred, 2)


class MSN_D(MSNAlpha):
    """
    Structure for MSN_D prediction
    -----------------------------
    Please train `MSNAlphaModel` and `MSNBeta_DModel`, and pass their
    model paths with args `--loada` and `--loadb` to use them together.
    """

    def __init__(self, Args: List[str], *args, **kwargs):
        super().__init__(Args, *args, **kwargs)

        self.args = MSNArgs(Args)

        # set inputs and groundtruths
        self.set_model_inputs('trajs', 'maps', 'map_paras')
        self.set_model_groundtruths('gt')

        # set metrics
        self.set_metrics('ade', 'fde')
        self.set_metrics_weights(1.0, 0.0)

        # assign alpha model and beta model containers
        self.alpha = self
        self.beta = MSNBeta_D(Args)
        self.linear_predict = False

        # load weights
        if 'null' in [self.args.loada, self.args.loadb]:
            raise ('`MSNAlpha` or `MSNBeta_D` not found!' +
                   ' Please specific their paths via `--loada` or `--loadb`.')

        self.alpha.args = MSNArgs(self.alpha.load_args(Args, self.args.loada),
                                  default_args=self.args._args)

        if self.args.loadb.startswith('l'):
            self.linear_predict = True
        
        else:
            self.beta.args = MSNArgs(self.beta.load_args(Args, self.args.loadb))
            self.beta.model = self.beta.load_from_checkpoint(self.args.loadb)

        self.alpha.model = self.alpha.load_from_checkpoint(
            self.args.loada,
            linear_prediction=self.linear_predict
        )

    def run_train_or_test(self):
        self.log('Start test model from `{}` and `{}`'.format(
            self.args.loada, self.args.loadb))
        self.run_test()

    def create_model(self, model_type=None, *args, **kwargs):
        return super().create_model(_MSNAlphaModelPlus, *args, **kwargs)

    def print_test_result_info(self, loss_dict, **kwargs):
        dataset = kwargs['dataset_name']
        self.print_parameters(title='test results', **
                            dict({'dataset': dataset}, **loss_dict))

        self.log('Results: {}, {}, {}, {}, K={}, sigma={}'.format(
            self.args.loada,
            self.args.loadb,
            dataset,
            loss_dict,
            self.args.K,
            self.args.sigma))
