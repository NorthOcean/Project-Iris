"""
@Author: Conghao Wong
@Date: 2021-07-09 10:50:39
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-16 11:19:01
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

        if self._args.force_pred_frames != -1:
            self._args.pred_frames = self._args.force_pred_frames

    @property
    def p_index(self) -> str:
        """
        Index of predicted points at the first stage.
        Split with `_`.
        For example, `'0_4_8_11'`.
        """
        return self._get('p_index', '11', changeable=False)

    @property
    def K_train(self) -> int:
        """
        Number of hidden categories used in alpha model.
        """
        return self._get('K_train', 10, changeable=False)
        
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
    def loadc(self) -> str:
        """
        Path for gamma model. (Unused)
        """
        return self._get('loadc', 'null', changeable=True)

    @property
    def linear(self) -> int:
        """
        Controls whether use linear prediction instead of beta model.
        """
        return self._get('linear', 0, changeable=False)

    @property
    def force_pred_frames(self) -> int:
        """
        force setting of predict frames when test.
        """
        return self._get('force_pred_frames', -1, changeable=True)

    @property
    def check(self) -> int:
        """
        Controls whether apply the results choosing strategy.
        """
        return self._get('check', 0, changeable=True)

    @property
    def points(self) -> int:
        """
        Controls number of points input to the beta model.
        """
        return self._get('points', 1, changeable=False)
