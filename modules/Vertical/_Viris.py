"""
@Author: Conghao Wong
@Date: 2021-07-09 09:50:49
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-28 20:19:58
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from argparse import Namespace
from typing import List, Tuple

import tensorflow as tf
from modules.models.helpmethods import BatchIndex
from modules.MSN._MSN_G import angle_check
from tqdm import tqdm

from ._args import VArgs
from ._utils import Utils as U
from ._VirisAlpha import VIrisAlpha, VIrisAlphaModel
from ._VirisBeta import VIrisBeta, VIrisBetaModel


class _VIrisAlphaModelPlus(VIrisAlphaModel):
    def __init__(self, Args: VArgs,
                 pred_number: int,
                 linear_prediction=False,
                 training_structure=None,
                 *args, **kwargs):

        super().__init__(Args, pred_number,
                         training_structure,
                         *args, **kwargs)

        self.linear = linear_prediction

    def post_process(self, outputs: List[tf.Tensor],
                     training=None,
                     *args, **kwargs) -> List[tf.Tensor]:

        # shape = ((batch, Kc, n, 2))
        outputs = VIrisAlphaModel.post_process(self, outputs, training, **kwargs)

        if training:
            return outputs

        # obtain shape parameters
        batch, Kc = outputs[0].shape[:2]
        n = self.n_pred
        pos = self.training_structure.p_index
        pred = self.args.pred_frames

        # shape = (batch, Kc, n, 2)
        proposals = outputs[0]
        current_inputs = kwargs['model_inputs']

        if self.linear:
            # Piecewise linear interpolation
            pos = tf.cast(pos, tf.float32)
            pos = tf.concat([[-1], pos], axis=0)
            obs = current_inputs[0][:, tf.newaxis, -1:, :]
            proposals = tf.concat([tf.repeat(obs, Kc, 1), proposals], axis=-2)

            final_results = U.LinearInterpolation(x=pos, y=proposals)

        else:
            beta_inputs = [inp for inp in current_inputs]
            beta_inputs.append(proposals)
            final_results = self.training_structure.beta(
                beta_inputs, return_numpy=False)[0]
        
        # check failure cases
        if self.args.check:
            final_results = angle_check(pred=final_results,
                                        obs=kwargs['model_inputs'][0],
                                        max_angle=135)

        return (final_results,)


class VIris(VIrisAlpha):
    """
    Structure for Vertical prediction
    ---------------------------------

    """
    
    alpha_model = _VIrisAlphaModelPlus
    beta_structure = VIrisBeta

    def __init__(self, Args: List[str], *args, **kwargs):
        super().__init__(Args, *args, **kwargs)

        self.args = VArgs(Args)

        # set inputs and groundtruths
        self.set_model_inputs('trajs', 'maps', 'map_paras')
        self.set_model_groundtruths('gt')

        # set metrics
        self.set_metrics('ade', 'fde')
        self.set_metrics_weights(1.0, 0.0)

        # assign alpha model and beta model containers
        self.alpha = self
        self.beta = self.beta_structure(Args)
        self.linear_predict = False

        # load weights
        if 'null' in [self.args.loada, self.args.loadb]:
            raise ('`IrisAlpha` or `IrisBeta` not found!' +
                   ' Please specific their paths via `--loada` or `--loadb`.')

        self.alpha.args = VArgs(self.alpha.load_args(Args, self.args.loada),
                                default_args=self.args._args)

        if self.args.loadb.startswith('l'):
            self.linear_predict = True
        
        else:
            self.beta.args = VArgs(self.beta.load_args(Args, self.args.loadb),
                                   default_args=self.args)
            self.beta.model = self.beta.load_from_checkpoint(
                self.args.loadb,
                asSecondStage=True,
                p_index=self.alpha.args.p_index)

        self.alpha.model = self.alpha.load_from_checkpoint(
            self.args.loada,
            linear_prediction=self.linear_predict
        )

    def run_train_or_test(self):
        self.run_test()

    def create_model(self, *args, **kwargs):
        return super().create_model(model_type=self.alpha_model,
                                    *args, **kwargs)

    def print_test_result_info(self, loss_dict, **kwargs):
        dataset = kwargs['dataset_name']
        self.print_parameters(title='test results', **
                            dict({'dataset': dataset}, **loss_dict))
        self.log('Results from {}, {}, {}, {}, {}'.format(
            self.args.loada,
            self.args.loadb,
            self.args.p_index,
            dataset,
            loss_dict))

