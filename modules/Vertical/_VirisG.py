"""
@Author: Conghao Wong
@Date: 2021-07-22 20:33:09
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-27 21:08:55
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import List

import tensorflow as tf

from ._args import VArgs
from ._Viris import VIris, _VIrisAlphaModelPlus
from ._VirisAlphaG import VIrisAlphaGModel


class _VIrisAlphaModelGPuls(VIrisAlphaGModel):

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

        return _VIrisAlphaModelPlus.post_process(self, outputs,
                                                 training,
                                                 *args, **kwargs)


class VIrisG(VIris):

    alpha_model = _VIrisAlphaModelGPuls

    def __init__(self, Args: List[str], *args, **kwargs):
        super().__init__(Args, *args, **kwargs)

    def print_test_result_info(self, loss_dict, **kwargs):
        dataset = kwargs['dataset_name']
        self.print_parameters(title='test results', **
                              dict({'dataset': dataset}, **loss_dict))
        self.log('Results from {}, {}, {}, {}, {}, K={}, sigma={}'.format(
            self.args.loada,
            self.args.loadb,
            self.args.p_index,
            dataset,
            loss_dict,
            self.args.K,
            self.args.sigma))
