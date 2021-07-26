"""
@Author: Conghao Wong
@Date: 2021-07-22 20:33:09
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-23 09:31:11
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""


from typing import List

from ._Viris import VIris
from ._VirisBetaG import VIrisBetaG


class VIrisG(VIris):

    beta_structure = VIrisBetaG

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