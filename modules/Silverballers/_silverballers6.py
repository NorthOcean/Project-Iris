"""
@Author: Conghao Wong
@Date: 2021-12-20 09:6:55
@LastEditors: Conghao Wong
@LastEditTime: 2021-12-20 09:50:06
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import List

import tensorflow as tf

from ..Vertical._utils import Utils as U
from ._agent6 import Agent6Model
from ._args import SilverballersArgs
from ._silverballers import Silverballers


class _Agent6ModelPlus(Agent6Model):
    def __init__(self, Args: SilverballersArgs,
                 feature_dim: int = 128,
                 id_depth: int = 16,
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
    def handler_model(self):
        try:
            return self.training_structure.association.handler.model
        except:
            raise ValueError('Structure object (id {}) has no `model` item.'.format(
                id(self.training_structure.association)))

    def post_process(self, outputs: List[tf.Tensor],
                     training=None,
                     *args, **kwargs) -> List[tf.Tensor]:

        # shape = ((batch, Kc, n, 2))
        outputs = Agent6Model.post_process(
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


class Silverballers6(Silverballers):

    agent_model = _Agent6ModelPlus

    def __init__(self, Args: List[str], *args, **kwargs):
        super().__init__(Args, *args, **kwargs)
