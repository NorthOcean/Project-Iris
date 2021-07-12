"""
@Author: Conghao Wong
@Date: 2021-07-09 10:40:52
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-09 18:07:38
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import List
import tensorflow as tf
from .. import models as M


class Utils():

    @staticmethod
    def forward(self: M.prediction.Model,
                model_inputs: List[tf.Tensor],
                training=None,
                *args, **kwargs) -> List[tf.Tensor]:
        
        """
        Run a forward implementation.

        :param model_inputs: input tensor (or a list of tensors)
        :param mode: choose forward type, can be `'test'` or `'train'`
        :return output: model's output. type=`List[tf.Tensor]`
        """

        model_inputs_processed = self.pre_process(model_inputs, training)
        destination_processed = self.pre_process([model_inputs[-1]],
                                                 training,
                                                 use_new_para_dict=False)

        model_inputs_processed = (model_inputs_processed[0],
                                  model_inputs_processed[1],
                                  model_inputs_processed[2],
                                  destination_processed[0])

        if training:
            gt_processed = self.pre_process([kwargs['gt']],
                                            use_new_para_dict=False)

        # use `self.call()` to debug
        output = self.call(model_inputs_processed,
                           gt_processed[0] if training else None,
                           training=training)

        if not (type(output) == list or type(output) == tuple):
            output = [output]

        return self.post_process(output, training, model_inputs=model_inputs)