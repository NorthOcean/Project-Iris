"""
@Author: Conghao Wong
@Date: 2021-04-01 20:28:00
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-14 10:58:26
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from argparse import Namespace
from typing import List, Union

import modules.models as M


class SatoshiArgs(M.prediction.TrainArgs):
    def __init__(self, args: List[str],
                 default_args: Union[Namespace, dict]=None):
                
        super().__init__(args, default_args=default_args)

        if self._args.force_pred_frames != -1:
            self._args.pred_frames = self._args.force_pred_frames

    @property
    def loada(self) -> str:
        """
        Path for Satoshi Alpha model
        """
        return self._get('loada', 'null', changeable=True)

    @property
    def loadb(self) -> str:
        """
        Path for Satoshi Beta model
        """
        return self._get('loadb', 'null', changeable=True)

    @property
    def loadc(self) -> str:
        """
        Path for Satoshi Gamma model
        """
        return self._get('loadc', 'null', changeable=True)

    @property
    def linear(self) -> int:
        """
        Controls whether use linear prediction in the last stage
        """
        return self._get('linear', 0, changeable=False)

    @property
    def H(self) -> int:
        """
        number of observed trajectories used
        """
        return self._get('H', 3, changeable=False)

    @property
    def force_pred_frames(self) -> int:
        """
        force setting of predict frames when test
        """
        return self._get('force_pred_frames', -1, changeable=True)

    @property
    def check(self) -> int:
        """
        Controls whether apply the results choosing strategy
        """
        return self._get('check', 0, changeable=True)


class SatoshiOnlineArgs(SatoshiArgs, M.prediction.OnlineArgs):
    def __init__(self, args: List[str]):
        SatoshiArgs.__init__(self, args)
        M.prediction.OnlineArgs.__init__(self, args)
