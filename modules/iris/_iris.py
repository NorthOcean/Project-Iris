"""
@Author: Conghao Wong
@Date: 2021-06-22 16:45:24
@LastEditors: Conghao Wong
@LastEditTime: 2021-06-24 09:51:49
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import Dict, List, Tuple

import modules.models as M
import numpy as np
import tensorflow as tf
from tensorflow import keras as keras

from ..satoshi._args import SatoshiArgs
from ._alpha import IrisAlpha, IrisAlphaModel
from ._beta import IrisBeta, IrisBetaModel
from ._beta_cvae import IrisBetaCVAE, IrisBetaCVAEModel


class _IrisAlphaModelPlus(IrisAlphaModel):
    """
    A specific `IrisAlpha` model that guides `IrisBeta` to post process outputs.
    """

    def __init__(self, Args, 
                 training_structure=None, 
                 *args, **kwargs):
        
        super().__init__(Args, training_structure=training_structure, 
                         *args, **kwargs)

    def post_process(self, outputs: Tuple[tf.Tensor],
                     training=False,
                     **kwargs) -> Tuple[tf.Tensor]:
        
        # shape = [(batch, K, 2)]
        outputs = super().post_process(outputs, training=training, **kwargs)

        if training:
            return outputs

        else:
            # shape = [(batch, obs, 2), (batch, 100, 100), None]
            model_inputs = kwargs['model_inputs']
            K = outputs[0].shape[1]

            # prepare new inputs into beta model
            # new batch_size is batch*K
            beta_inputs = [tf.repeat(inp, K, axis=0) for inp in model_inputs]
            beta_inputs.append(tf.reshape(outputs[0], [-1, 2]))

            # run beta model here
            # output shape = (batch*K, K', pred, 2)
            beta_results = self.training_structure.beta(
                beta_inputs,
                return_numpy=False,
            )[0]

            # re-organize outputs
            K2 = beta_results.shape[1]
            final_results = tf.reshape(beta_results, 
                                       [-1, K*K2, self.args.pred_frames, 2])

            return M.prediction.Process.update((final_results,), outputs)


class _IrisAlphaModelLinear(IrisAlphaModel):
    """
    A specific `IrisAlpha` model that guides linear prediction to post process outputs.
    """
    def __init__(self, Args, 
                 training_structure=None, 
                 *args, **kwargs):
                 
        super().__init__(Args, training_structure=training_structure, 
                         *args, **kwargs)

    def post_process(self, outputs: Tuple[tf.Tensor],
                     training=False,
                     **kwargs) -> Tuple[tf.Tensor]:
        
        # shape = [(batch, K, 2)]
        outputs = super().post_process(outputs, training=training, **kwargs)

        if training:
            return outputs

        else:
            # shape = [(batch, obs, 2), (batch, 100, 100), None]
            model_inputs = kwargs['model_inputs']

            obs_pos = model_inputs[0][:, -1, :][:, tf.newaxis, :] # (batch, 1, 2)
            des_pos = outputs[0]    # (batch, K, 2)

            final_results = []
            p = self.args.pred_frames
            for step in range(1, p+1):
                pred = (des_pos - obs_pos) * step/p + obs_pos   # (batch, K, 2)
                final_results.append(pred)

            # shape = (batch, K, pred, 2)
            final_results = tf.transpose(final_results, [1, 2, 0, 3])
            return M.prediction.Process.update((final_results,), outputs)
            
            
class Iris(IrisAlpha):
    """
    Structure for IRIS prediction
    -----------------------------
    Please train `IrisAlphaModel` and `IrisBetaModel`, and pass their
    model paths with args `--loada` and `--loadb` to use them together.
    """

    def __init__(self, args, beta_model=IrisBeta):
        super().__init__(args)

        # set inputs and groundtruths
        self.set_model_inputs('trajs', 'maps', 'map_paras')
        self.set_model_groundtruths('gt')

        # set metrics
        self.set_metrics('ade', 'fde')
        self.set_metrics_weights(1.0, 0.0)
        
        # assign alpha model and beta model containers
        self.alpha = self
        self.beta = beta_model(args)
        self.linear_predict = False

        # load weights
        if 'null' in [self.args.loada, self.args.loadb]:
            raise ('`IrisAlpha` or `IrisBeta` not found!' +
                   ' Please specific their paths via `--loada` or `--loadb`.')
        
        if self.args.loadb.startswith('l'):
            self.linear_predict = True
        else:
            self.beta.load_args(args, args.loadb, SatoshiArgs)
            self.beta.model = self.beta.load_from_checkpoint(args.loadb)

        self.alpha.load_args(args, args.loada, SatoshiArgs)
        self.alpha.model = self.alpha.load_from_checkpoint(args.loada)
    
    def run_train_or_test(self):
        self.logger.info('Start test model from `{}` and `{}`'.format(self.args.loada, self.args.loadb))
        self.run_test()

    def create_model(self, model_type=None):
        model_type = \
            _IrisAlphaModelLinear if self.linear_predict \
            else _IrisAlphaModelPlus

        return super().create_model(model_type)

    def print_test_result_info(self, loss_dict, **kwargs):
        dataset = kwargs['dataset_name']
        self.log_parameters(title='test results', **
                            dict({'dataset': dataset}, **loss_dict))
        # with open('./test_log.txt', 'a') as f:
        #     f.write('{}, {}, {}, {}, {}, K={}, sigma={}\n'.format(
        #         'Iris',
        #         self.args.loada,
        #         self.args.loadb,
        #         dataset,
        #         loss_dict,
        #         self.args.K,
        #         self.args.sigma))

        self.logger.info('Results: {}, {}, {}, {}, K={}, sigma={}'.format(
            self.args.loada,
            self.args.loadb,
            dataset,
            loss_dict,
            self.args.K,
            self.args.sigma))


class IrisCVAE(Iris):
    def __init__(self, args, beta_model=IrisBetaCVAE):
        super().__init__(args, beta_model=beta_model)
