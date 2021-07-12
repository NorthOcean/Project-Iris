'''
Author: Conghao Wong
Date: 2021-03-18 19:13:26
LastEditors: Conghao Wong
LastEditTime: 2021-04-19 11:19:17
Description: file content
'''

from typing import Dict, List, Tuple

import modules.models as M
import numpy as np
import tensorflow as tf
from tensorflow import keras as keras

from ._alpha import SatoshiAlpha, SatoshiAlphaModel
from ._args import SatoshiArgs
from ._beta import SatoshiBeta, SatoshiBetaModel


class _SatoshiAlphaModelPlus(SatoshiAlphaModel):
    def __init__(self, Args, 
                 training_structure=None, 
                 gcn_layer_count=2, 
                 intention_count=10, 
                 *args, **kwargs):
        
        super().__init__(Args, training_structure=training_structure,
                         gcn_layer_count=gcn_layer_count,
                         intention_count=intention_count,
                         *args, **kwargs)

    def post_process(self, outputs: Tuple[tf.Tensor],
                     training=False,
                     **kwargs) -> Tuple[tf.Tensor]:

        # shape = [(batch, K, 2)]
        outputs = super().post_process(outputs, training=training, **kwargs)

        if training:
            return outputs
        else:
            model_inputs = kwargs['model_inputs']   # shape = (batch, obs, 2)
            K = outputs[0].shape[1]

            new_inputs = tuple(
                [tf.reshape(tf.repeat(tf.expand_dims(inputs, axis=1), K, axis=1), [inputs.shape[0] * K, inputs.shape[1], inputs.shape[2]]) 
                for inputs in model_inputs]) + (tf.reshape(outputs[0], [-1, 2]),)
            stack_results = self.training_structure.beta(
                new_inputs, return_numpy=False)[0]
            final_predictions = tf.reshape(
                stack_results, [-1, K, self.args.pred_frames, 2])
            return M.prediction.Process.update((final_predictions,), outputs)


class _SatoshiAlphaModelLinear(SatoshiAlphaModel):
    def __init__(self, Args, training_structure=None, gcn_layer_count=2, intention_count=10, *args, **kwargs):
        super().__init__(Args,
                         training_structure=training_structure,
                         gcn_layer_count=gcn_layer_count,
                         intention_count=intention_count,
                         *args, **kwargs)

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


class Satoshi(SatoshiAlpha):
    """
    Structure for Satoshi Model
    ---------------------------
    This model cannot be trained, please specify satoshi alpha and beta
    models via args `args.loada` and `args.loadb`.
    Model alpha will be rewritten in this structure, and post-processing 
    operations will be added to apply beta model or linear model.
    """

    def __init__(self, args: SatoshiArgs):
        super().__init__(args)

        # model inputs and groundtruths
        self.set_model_inputs('trajs', 'maps', 'map_paras')
        self.set_model_groundtruths('gt')

        # alpha model (intention)
        self.__alpha = self
        self.linear_prediction = False

        # beta model (interpolation: linear or beta)
        self.__beta = SatoshiBeta(args, SatoshiArgs)

        # load model network and weights
        if self.args.loada == 'null' or self.args.loadb == 'null':
            raise 'Model(s) are not found!'

        if not self.args.loadb.startswith('l'):
            self.__beta.load_args(args, args.loadb, arg_type=SatoshiArgs)
            self.__beta._model = self.beta.load_from_checkpoint(
                self.args.loadb)
        else:
            self.linear_prediction = True

        # alpha model: load networks and weights
        self.__alpha.load_args(args, args.loada, arg_type=SatoshiArgs)
        self.__alpha._model = self.alpha.load_from_checkpoint(self.args.loada)
        self.__alpha.set_metrics('ade', 'fde')
        self.__alpha.set_metrics_weights(1.0, 0.0)

    @property
    def alpha(self) -> SatoshiAlpha:
        return self.__alpha

    @property
    def beta(self) -> SatoshiBeta:
        return self.__beta

    def metrics(self, outputs, labels, loss_name_list=['ADE', 'FDE'], **kwargs):
        return M.prediction.Structure.metrics(self, outputs, labels)

    def run_train_or_test(self):
        self.run_test()

    def create_model(self, **kwargs):
        model_type = _SatoshiAlphaModelLinear if self.linear_prediction else _SatoshiAlphaModelPlus
        return super().create_model(model_type=model_type)

    def print_test_result_info(self, loss_dict, **kwargs):
        dataset = kwargs['dataset_name']
        self.log_parameters(title='test results', **
                            dict({'dataset': dataset}, **loss_dict))
        with open('./test_log.txt', 'a') as f:
            f.write('{}, {}, {}, {}, {}\n'.format(
                'Satoshi',
                self.args.loada,
                self.args.loadb,
                dataset,
                loss_dict))
