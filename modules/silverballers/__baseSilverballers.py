"""
@Author: Conghao Wong
@Date: 2021-12-22 11:20:08
@LastEditors: Conghao Wong
@LastEditTime: 2021-12-30 10:12:40
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import List

import tensorflow as tf
from tensorflow import keras

from .. import applications as A
from .. import models as M
from .__args import AgentArgs, HandlerArgs, SilverballersArgs
from .__baseAgent import BaseAgentStructure
from .__baseHandler import BaseHandlerStructure


class BaseSilverballersModel(M.prediction.Model):

    def __init__(self, Args: SilverballersArgs,
                 agentModel: M.prediction.Model,
                 handlerModel: M.prediction.Model = None,
                 training_structure=None,
                 *args, **kwargs):

        super().__init__(Args, training_structure, *args, **kwargs)

        self.set_preprocess()

        self.agent = agentModel
        self.handler = handlerModel
        self.linear = not self.handler

        if self.linear:
            self.linear_layer = A.layers.LinearInterpolation()

    def call(self, inputs: List[tf.Tensor],
             training=None, mask=None,
             *args, **kwargs):

        outputs = self.agent.forward(inputs)

        # obtain shape parameters
        batch, Kc = outputs[0].shape[:2]
        pos = self.agent.training_structure.p_index

        # shape = (batch, Kc, n, 2)
        proposals = outputs[0]
        current_inputs = inputs

        if self.linear:
            # Piecewise linear interpolation
            pos = tf.cast(pos, tf.float32)
            pos = tf.concat([[-1], pos], axis=0)
            obs = current_inputs[0][:, tf.newaxis, -1:, :]
            proposals = tf.concat([tf.repeat(obs, Kc, 1), proposals], axis=-2)

            final_results = self.linear_layer.call(index=pos, value=proposals)

        else:
            # call the second stage model
            handler_inputs = [inp for inp in current_inputs]
            handler_inputs.append(proposals)
            final_results = self.handler.forward(handler_inputs)[0]

        return (final_results,)


class Silverballers(M.prediction.Structure):

    """
    Basic structure to run the `agent-handler` based silverballers model.
    Please set agent model and handler model used in this silverballers by
    subclassing this class, and call the `set_models` method *before*
    the `super().__init__()` method.
    """

    # Structures
    agent_structure = BaseAgentStructure
    handler_structure = BaseHandlerStructure

    # Models
    agent_model = None
    handler_model = None
    silverballer_model = BaseSilverballersModel

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

        # check weights
        if 'null' in [self.args.loada, self.args.loadb]:
            raise ('`Agent` or `Handler` not found!' +
                   ' Please specific their paths via `--loada` or `--loadb`.')

        # load args
        agent_args_list = self.load_args(Args, self.args.loada)
        agent_args = AgentArgs(agent_args_list, default_args=self.args)._args

        # assign models
        self.agent = self.agent_structure(agent_args, association=self)
        self.agent.set_model_type(self.agent_model)

        if self.args.loadb.startswith('l'):
            self.linear_predict = True

        else:
            self.linear_predict = False

            handler_args_list = self.load_args(Args, self.args.loadb)
            handler_args = HandlerArgs(
                handler_args_list, default_args=self.args)._args

            self.handler = self.handler_structure(
                handler_args, association=self)
            self.handler.set_model_type(self.handler_model)
            self.handler.model = self.handler.load_from_checkpoint(
                self.args.loadb,
                asHandler=True)

        self.agent.model = self.agent.load_from_checkpoint(self.args.loada)

        if self.args.batch_size > self.agent.args.batch_size:
            self.args._set('batch_size', self.agent.args.batch_size)
        self.args._set('test_set', self.agent.args.test_set)
        
        self.agent.args._set('K', self.args.K)

    def set_models(self, agentModel,
                   handlerModel,
                   agentStructure=None,
                   handlerStructure=None):
        """
        Set models and structures used in this silverballers instance.
        Please call this method before the `__init__` method when subclassing.
        You should better set `agentModel` and `handlerModel` rather than
        their training structures if you do not subclass these structures.
        """
        if agentModel:
            self.agent_model = agentModel

        if agentStructure:
            self.agent_structure = agentStructure

        if handlerModel:
            self.handler_model = handlerModel

        if handlerStructure:
            self.handler_structure = handlerStructure

    def run_train_or_test(self):
        self.model, _ = self.create_model()
        self.run_test()

    def create_model(self, *args, **kwargs):
        model = self.silverballer_model(
            self.args,
            agentModel=self.agent.model,
            handlerModel=None if self.linear_predict else self.handler.model,
            training_structure=self,
            *args, **kwargs)

        opt = keras.optimizers.Adam(self.args.lr)
        return model, opt

    def print_test_results(self, loss_dict, **kwargs):
        dataset = kwargs['dataset_name']
        self.print_parameters(title='test results', **
                              dict({'dataset': dataset}, **loss_dict))
        self.log('Results from {}, {}, {}, {}'.format(
            self.args.loada,
            self.args.loadb,
            dataset,
            loss_dict))
