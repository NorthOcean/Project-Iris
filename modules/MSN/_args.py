"""
@Author: Conghao Wong
@Date: 2021-04-01 20:28:00
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-20 10:53:22
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from argparse import Namespace
from typing import List, Union

import modules.models as M


class MSNArgs(M.prediction.PredictionArgs):
    def __init__(self, args: Union[Namespace, List[str]],
                 default_args: Union[Namespace, dict]=None):
                
        super().__init__(args, default_args=default_args)

    @property
    def loada(self) -> str:
        """
        Path for the first stage Destination Transformer
        """
        return self._get('loada', 'null', changeable=True)

    @property
    def loadb(self) -> str:
        """
        Path for the second stage Interaction Transformer
        """
        return self._get('loadb', 'null', changeable=True)

    @property
    def loadc(self) -> str:
        """
        Path for the third stage model (Preserved)
        """
        return self._get('loadc', 'null', changeable=True)

    @property
    def linear(self) -> int:
        """
        Controls whether use linear prediction in the last stage
        """
        return self._get('linear', 0, changeable=False)

    @property
    def check(self) -> int:
        """
        Controls whether apply the results choosing strategy
        """
        return self._get('check', 0, changeable=True)

