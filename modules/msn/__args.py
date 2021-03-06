"""
@Author: Conghao Wong
@Date: 2021-04-01 20:28:00
@LastEditors: Conghao Wong
@LastEditTime: 2022-04-21 11:01:07
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from argparse import Namespace
from typing import Union

import modules.models as M


class MSNArgs(M.prediction.PredictionArgs):
    def __init__(self, args: Union[Namespace, list[str]],
                 default_args: Union[Namespace, dict]=None):
                
        super().__init__(args, default_args=default_args)

    @property
    def loada(self) -> str:
        """
        Path for the first stage style hypothesis networks.
        """
        return self._get('loada', 'null', changeable=True)

    @property
    def loadb(self) -> str:
        """
        Path for the second stage stylized prediction networks.
        """
        return self._get('loadb', 'null', changeable=True)

    @property
    def loadc(self) -> str:
        """
        Path for the third stage model (Preserved).
        """
        return self._get('loadc', 'null', changeable=True)

    @property
    def check(self) -> int:
        """
        Controls whether apply the results choosing strategy
        """
        return self._get('check', 0, changeable=True)

    @property
    def K_train(self) -> int:
        """
        The number of hidden behavior categories in `AlphaModel`,
        or the number of multiple generations when training in `BetaModel`.
        """
        return self._get('K_train', 10, changeable=False)

