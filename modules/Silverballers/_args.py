"""
@Author: Conghao Wong
@Date: 2021-10-28 19:48:56
@LastEditors: Conghao Wong
@LastEditTime: 2021-11-22 19:32:23
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from argparse import Namespace
from typing import List, Union

import modules.models as M

class SArgs(M.prediction.PredictionArgs):
    def __init__(self, args: Union[Namespace, List[str]],
                 default_args: Union[Namespace, dict] = None):
                 
        super().__init__(args, default_args=default_args)

    @property
    def K(self) -> int:
        """
        Number of multiple generations when test.
        This arg only works for `Generative Models`.
        """
        return self._get('K', 1, changeable=True)

    @property
    def Kc(self) -> int:
        """
        Number of styles in `Agent` model.
        """
        return self._get('Kc', 20, changeable=False)

    @property
    def key_points(self) -> str:
        """
        A list of keypoints' index to be predicted in `Agent` model.
        For example, `'0_6_11'`.
        """
        return self._get('key_points', '0_6_11', changeable=False)

    @property
    def use_maps(self) -> int:
        """
        Controls if uses the trajectory maps or the social maps in the model.
        """
        return self._get('use_maps', 0, changeable=False)