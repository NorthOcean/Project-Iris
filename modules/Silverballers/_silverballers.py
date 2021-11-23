"""
@Author: Conghao Wong
@Date: 2021-11-23 16:15:34
@LastEditors: Conghao Wong
@LastEditTime: 2021-11-23 19:46:51
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import List

import tensorflow as tf

from .. import models as M
from ..Vertical._utils import Utils as U
from ._agent import Agent, AgentModel
from ._args import AgentArgs, HandlerArgs, SilverballersArgs
from ._handler import Handler, HandlerModel


class _AgentModelPlus(AgentModel):
    def __init__(self, Args: SilverballersArgs,
                 feature_dim: int = 128,
                 id_depth: int = 64,
                 keypoints_number: int = 3,
                 linear_prediction=False,
                 training_structure=None,
                 *args, **kwargs):

        super().__init__(Args, feature_dim=feature_dim,
                         id_depth=id_depth,
                         keypoints_number=keypoints_number,
                         training_structure=training_structure,
                         *args, **kwargs)

        self.linear = linear_prediction

    @property
    def handler_model(self) -> HandlerModel:
        try:
            return self.training_structure.association.handler.model
        except:
            raise ValueError('Structure object (id {}) has no `model` item.'.format(
                id(self.training_structure.association)))

    def post_process(self, outputs: List[tf.Tensor],
                     training=None,
                     *args, **kwargs) -> List[tf.Tensor]:

        # shape = ((batch, Kc, n, 2))
        outputs = AgentModel.post_process(
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
            final_results = self.handler_model.forward(beta_inputs)[0]

        return (final_results,)


class Silverballers(M.prediction.Structure):

    handler_structure = Handler
    handler_model = HandlerModel
    agent_structure = Agent
    agent_model = _AgentModelPlus

    def __init__(self, Args: List[str], *args, **kwargs):
        super().__init__(Args, *args, **kwargs)

        # set args
        self.args = SilverballersArgs(Args)
        self.important_args += ['K']

        # set inputs and outputs
        self.set_model_inputs('trajs', 'maps', 'paras', 'gt')
        self.set_model_groundtruths('gt')

        # set metrics
        self.set_metrics('ade', 'fde')
        self.set_metrics_weights(1.0, 0.0)

        # assign models
        self.agent = self.agent_structure(Args, association=self)
        self.agent.set_model_type(_AgentModelPlus)
        self.handler = self.handler_structure(Args, association=self)
        self.linear_predict = False

        # load weights
        if 'null' in [self.args.loada, self.args.loadb]:
            raise ('`Agent` or `Handler` not found!' +
                   ' Please specific their paths via `--loada` or `--loadb`.')

        self.agent.args = AgentArgs(self.agent.load_args(Args, self.args.loada),
                                    default_args=self.args._args)

        if self.args.loadb.startswith('l'):
            self.linear_predict = True

        else:
            self.handler.args = HandlerArgs(self.handler.load_args(Args, self.args.loadb),
                                            default_args=self.args)
            self.handler.model = self.handler.load_from_checkpoint(
                self.args.loadb,
                asSecondStage=True,
                p_index=self.agent.args.key_points)

        self.agent.model = self.agent.load_from_checkpoint(
            self.args.loada,
            linear_prediction=self.linear_predict
        )

        self.model = self.agent.model

    def run_train_or_test(self):
        self.run_test()

    def create_model(self, *args, **kwargs):
        return self.agent.create_model(model_type=self.agent_model,
                                       *args, **kwargs)

    def print_test_results(self, loss_dict, **kwargs):
        dataset = kwargs['dataset_name']
        self.print_parameters(title='test results', **
                              dict({'dataset': dataset}, **loss_dict))
        self.log('Results from {}, {}, {}, {}'.format(
            self.args.loada,
            self.args.loadb,
            dataset,
            loss_dict))
