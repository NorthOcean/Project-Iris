"""
@Author: Conghao Wong
@Date: 2021-07-09 10:50:39
@LastEditors: Conghao Wong
@LastEditTime: 2021-08-05 16:21:26
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""


from argparse import Namespace
from typing import List, Union

import modules.models as M


class VArgs(M.prediction.PredictionArgs):
    def __init__(self, args: Union[Namespace, List[str]], 
                 default_args: Union[Namespace, dict] = None):
                 
        super().__init__(args, default_args)

    @property
    def p_index(self) -> str:
        """
        Index of predicted points at the first stage.
        Split with `_`.
        For example, `'0_4_8_11'`.
        """
        return self._get('p_index', '11', changeable=False)

    @property
    def Kc(self) -> int:
        """
        Number of hidden categories used in alpha model.
        """
        return self._get('Kc', 10, changeable=False)
        
    @property
    def loada(self) -> str:
        """
        Path for alpha model.
        """
        return self._get('loada', 'null', changeable=True)

    @property
    def loadb(self) -> str:
        """
        Path for beta model.
        """
        return self._get('loadb', 'null', changeable=True)

    @property
    def check(self) -> int:
        """
        Controls whether apply the results choosing strategy.
        """
        return self._get('check', 0, changeable=True)

    @property
    def points(self) -> int:
        """
        Controls number of points (representative time steps) input to the beta model.
        """
        return self._get('points', 1, changeable=False)
