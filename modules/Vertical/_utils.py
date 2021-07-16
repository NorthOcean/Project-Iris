"""
@Author: Conghao Wong
@Date: 2021-07-09 10:40:52
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-15 11:17:22
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

    @staticmethod
    def LinearInterpolation(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """
        Piecewise linear interpolation
        (Results do not contain the start point)

        :param x: index, shape = `(n)`, where `m = x[-1] - x[0]`
        :param y: values, shape = `(..., n, 2)`
        :return yp: linear interpolations, shape = `(..., m, 2)`
        """
        linear_results = []
        for output_index in range(x.shape[0] - 1):
            p_start = x[output_index]
            p_end = x[output_index+1]

            # shape = (..., 2)
            start = tf.gather(y, output_index, axis=-2)
            end = tf.gather(y, output_index+1, axis=-2)

            for p in tf.range(p_start+1, p_end+1):
                linear_results.append(tf.expand_dims(
                    (end - start) * (p - p_start) / (p_end - p_start)
                    + start
                , axis=-2))   # (..., 1, 2)

        # shape = (..., n, 2)
        return tf.concat(linear_results, axis=-2)