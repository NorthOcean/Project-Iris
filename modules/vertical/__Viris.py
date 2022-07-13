"""
@Author: Conghao Wong
@Date: 2021-07-09 09:50:49
@LastEditors: Conghao Wong
@LastEditTime: 2022-04-21 11:02:46
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf
from modules.msn.__MSN_G import angle_check

from .__args import VArgs
from .__utils import Utils as U
from .__VirisAlpha import VIrisAlpha, VIrisAlphaModel
from .__VirisBeta import VIrisBeta, VIrisBetaModel


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

    @property
    def beta_model(self) -> VIrisBetaModel:
        try:
            return self.training_structure.beta.model
        except:
            raise ValueError('Structure object (id {}) has no `model` item.'.format(
                id(self.training_structure)))

    def post_process(self, outputs: list[tf.Tensor],
                     training=None,
                     *args, **kwargs) -> list[tf.Tensor]:

        # shape = ((batch, Kc, n, 2))
        outputs = VIrisAlphaModel.post_process(
            self, outputs, training, **kwargs)

        if training:
            return outputs

        # obtain shape parameters
        batch, Kc = outputs[0].shape[:2]
        pos = self.training_structure.p_index

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
            # call the second stage model
            beta_inputs = [inp for inp in current_inputs]
            beta_inputs.append(proposals)
            final_results = self.beta_model.forward(beta_inputs)[0]

        # check failure cases
        if self.args.check:
            final_results = angle_check(pred=final_results,
                                        obs=kwargs['model_inputs'][0],
                                        max_angle=135)

        return (final_results,)


class VIris(VIrisAlpha):
    """
    Structure for the deterministic `Vertical`
    """

    alpha_model = _VIrisAlphaModelPlus
    beta_structure = VIrisBeta

    def __init__(self, Args: list[str], *args, **kwargs):
        super().__init__(Args, *args, **kwargs)

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

    def print_test_results(self, loss_dict, **kwargs):
        dataset = kwargs['dataset_name']
        self.print_parameters(title='test results', **
                              dict({'dataset': dataset}, **loss_dict))
        self.log('Results from {}, {}, {}, {}, {}'.format(
            self.args.loada,
            self.args.loadb,
            self.args.p_index,
            dataset,
            loss_dict))
