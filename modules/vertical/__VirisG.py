"""
@Author: Conghao Wong
@Date: 2021-07-22 20:33:09
@LastEditors: Conghao Wong
@LastEditTime: 2022-04-21 11:03:06
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from .__args import VArgs
from .__Viris import VIris, _VIrisAlphaModelPlus
from .__VirisAlphaG import VIrisAlphaGModel
from .__VirisBeta import VIrisBetaModel


class _VIrisAlphaModelGPuls(VIrisAlphaGModel):
    """
    A special version of the generative first stage `Vertical` model
    that guides the second stage `Vertical` model to process outputs.
    """

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
            raise ValueError('Structure object (id {}) has no attribute `model`.'.format(
                id(self.training_structure)))

    def post_process(self, outputs: list[tf.Tensor],
                     training=None,
                     *args, **kwargs) -> list[tf.Tensor]:

        return _VIrisAlphaModelPlus.post_process(self, outputs,
                                                 training,
                                                 *args, **kwargs)


class VIrisG(VIris):
    """
    Structure for the generative `Vertical`
    """
    
    alpha_model = _VIrisAlphaModelGPuls

    def __init__(self, Args: list[str], *args, **kwargs):
        super().__init__(Args, *args, **kwargs)

        self.important_args += ['K']

    def print_test_results(self, loss_dict, **kwargs):
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
