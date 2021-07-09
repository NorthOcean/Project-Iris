"""
@Author: Conghao Wong
@Date: 2021-05-08 15:50:39
@LastEditors: Conghao Wong
@LastEditTime: 2021-06-25 09:51:50
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import Dict, List, Tuple

import modules.models as M
import numpy as np
import tensorflow as tf
from tensorflow import keras as keras

from ._alpha_transformer import SatoshiAlphaTransformer, SatoshiAlphaTransformerModel
from ._beta_transformer import SatoshiBetaTransformer
from ._args import SatoshiArgs


class _SatoshiAlphaTransformerGenerativeModel(SatoshiAlphaTransformerModel):
    def __init__(self, Args, training_structure=None, *args, **kwargs):
        super().__init__(Args, training_structure=training_structure, *args, **kwargs)

    def post_process(self, outputs, training=False, **kwargs):
        # shape = (batch, K_c, 2)
        outputs = super().post_process(outputs, training=training, **kwargs)
        K = outputs[0].shape[1]

        if True: # training or K <= self.args.K:
            return outputs

        else:
            random_index = tf.argsort(tf.random.uniform([K]))[:self.args.K]
            outputs_sampled = tf.gather(outputs[0], random_index, axis=1)
            return M.prediction.Process.update((outputs_sampled,), outputs)


class _SatoshiAlphaTransformerModelPlus(_SatoshiAlphaTransformerGenerativeModel):
    def __init__(self, Args, training_structure=None, *args, **kwargs):
        super().__init__(Args, training_structure=training_structure, *args, **kwargs)

    def post_process(self, outputs, training=False, **kwargs):
        outputs = super().post_process(outputs, training=training, **kwargs)

        if training:
            return outputs
        else:
            model_inputs = kwargs['model_inputs']
            K = outputs[0].shape[1]

            new_inputs = tuple([tf.reshape(tf.repeat(tf.expand_dims(inputs, axis=1), K, axis=1), [
                inputs.shape[0] * K, inputs.shape[1], inputs.shape[2]]) for inputs in model_inputs]) + (tf.reshape(outputs[0], [-1, 1, 2]),)
            stack_results = self.training_structure.beta(
                new_inputs, return_numpy=False)[0]
            final_predictions = tf.reshape(
                stack_results, [-1, K, self.args.pred_frames, 2])
            return M.prediction.Process.update((final_predictions,), outputs)


class _SatoshiAlphaTransformerModelLinear(_SatoshiAlphaTransformerGenerativeModel):
    def __init__(self, Args, training_structure=None, *args, **kwargs):
        super().__init__(Args, training_structure=training_structure, *args, **kwargs)

    def post_process(self, outputs, training=False, **kwargs):
        outputs = super().post_process(outputs, training=training, **kwargs)

        if training:
            return outputs
        else:
            """
            Post process: Linear
            """
            model_inputs = kwargs['model_inputs']

            current_positions = model_inputs[0][:, -1, :]   # [batch, 2]
            intentions = outputs[0]

            final_predictions = []
            for pred in range(1, self.args.pred_frames+1):
                final_pred = (intentions - tf.expand_dims(current_positions, 1)) * \
                    pred / self.args.pred_frames + \
                    tf.expand_dims(current_positions, 1)
                final_predictions.append(final_pred)
            final_predictions = tf.transpose(final_predictions, [1, 2, 0, 3])

            return M.prediction.Process.update((final_predictions,), outputs)


class SatoshiTransformer(SatoshiAlphaTransformer):
    def __init__(self, args, arg_type=SatoshiArgs):
        super().__init__(args, arg_type=arg_type)

        # model inputs and groundtruths
        self.set_model_inputs('trajs', 'maps', 'map_paras')
        self.set_model_groundtruths('gt')

        # load model network and weights
        if self.args.loada == 'null' or self.args.loadb == 'null':
            raise 'Model(s) are not found!'

        # alpha model (intention)
        self.alpha = self
        self.linear_prediction = False

        # beta model (interpolation)
        if not self.args.loadb.startswith('l'):
            self.beta = SatoshiBetaTransformer(args)
            self.beta.load_args(args, args.loadb, arg_type=SatoshiArgs)
            self.beta.model = self.beta.load_from_checkpoint(
                self.args.loadb)
        else:
            self.linear_prediction = True

        # alpha model: load networks and weights
        self.alpha.load_args(args, args.loada, arg_type=SatoshiArgs)
        self.alpha.model = self.alpha.load_from_checkpoint(self.args.loada)
        self.alpha.set_metrics('ade', 'fde')
        self.alpha.set_metrics_weights(1.0, 0.0)

    def run_train_or_test(self):
        self.run_test()

    def create_model(self, **kwargs):
        model_type = _SatoshiAlphaTransformerModelLinear if self.linear_prediction else _SatoshiAlphaTransformerModelPlus
        return super().create_model(model_type=model_type)

    def print_test_result_info(self, loss_dict, **kwargs):
        dataset = kwargs['dataset_name']
        self.log_parameters(title='test results', **
                            dict({'dataset': dataset}, **loss_dict))
        self.logger.info('Results from {}, {}, {}, {}'.format(
            self.args.loada,
            self.args.loadb,
            dataset,
            loss_dict))