"""
@Author: Conghao Wong
@Date: 2021-10-28 19:48:56
@LastEditors: Conghao Wong
@LastEditTime: 2021-12-24 16:15:08
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from argparse import Namespace
from typing import List, Union

import modules.models as M


class AgentArgs(M.prediction.PredictionArgs):

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
    def depth(self) -> int:
        """
        Depth of the random contract id
        """
        return self._get('depth', 16, changeable=False)

    @property
    def use_maps(self) -> int:
        """
        Controls if uses the trajectory maps or the social maps in the model.
        """
        return self._get('use_maps', 0, changeable=False)

    @property
    def preprocess(self) -> str:
        """
        Controls if run any preprocess before model inference.
        Accept a 3-bit-like string value (like `'111'`):
        - the first bit indicates if `MOVE` trajectories to (0, 0);
        - the second bit indicates if re-`SCALE` trajectories;
        - the third bit indicates if `ROTATE` trajectories.
        """
        return self._get('preprocess', '111', changeable=False)

    @property
    def metric(self) -> str:
        """
        Controls the metric used to save best models when training.
        Accept either `'ade'` or `'fde'`.
        """
        return self._get('metric', 'fde', changeable=False)


class HandlerArgs(M.prediction.PredictionArgs):

    def __init__(self, args: Union[Namespace, List[str]],
                 default_args: Union[Namespace, dict] = None):

        super().__init__(args, default_args=default_args)

    @property
    def use_maps(self) -> int:
        """
        Controls if uses the trajectory maps or the social maps in the model.
        """
        return self._get('use_maps', 1, changeable=False)

    @property
    def points(self) -> int:
        """
        Controls number of points (representative time steps) input to the beta model.
        """
        return self._get('points', 1, changeable=False)

    @property
    def key_points(self) -> str:
        """
        A list of keypoints' index to be predicted in `Agent` model.
        For example, `'0_6_11'`.
        """
        return self._get('key_points', 'null', changeable=False)


class SilverballersArgs(M.prediction.PredictionArgs):

    def __init__(self, args: Union[Namespace, List[str]],
                 default_args: Union[Namespace, dict] = None):

        super().__init__(args, default_args=default_args)

    @property
    def use_maps(self) -> int:
        """
        Controls if uses the trajectory maps or the social maps in the model.
        """
        return self._get('use_maps', 1, changeable=False)

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
    def K(self) -> int:
        """
        Number of multiple generations when test.
        This arg only works for `Generative Models`.
        """
        return self._get('K', 1, changeable=True)
