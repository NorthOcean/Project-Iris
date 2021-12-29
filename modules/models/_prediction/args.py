"""
@Author: Conghao Wong
@Date: 2020-11-20 09:11:33
@LastEditors: Conghao Wong
@LastEditTime: 2021-12-29 15:49:28
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from argparse import Namespace
from typing import List, Union

from .. import base


class PredictionArgs(base.Args):
    """
    PredictionArgs
    --------------
    A set of args used in training universal prediction models
    """

    def __init__(self, args: Union[Namespace, List[str]],
                 default_args: Union[Namespace, dict] = None):

        super().__init__(args, default_args=default_args)

    @property
    def obs_frames(self) -> int:
        """
        Observation frames for prediction.
        """
        return self._get('obs_frames', 8, changeable=False)

    @property
    def pred_frames(self) -> int:
        """
        Prediction frames.
        """
        return self._get('pred_frames', 12, changeable=False)

    @property
    def draw_results(self) -> int:
        """
        Controls if draw visualized results on video frames.
        Make sure that you have put video files into `./videos`
        according to the specific name way.
        """
        return self._get('draw_results', 0, changeable=True)

    @property
    def step(self) -> int:
        """
        Frame interval for sampling training data.
        """
        return self._get('step', 1, changeable=True)

    @property
    def add_noise(self) -> int:
        """
        Controls if add noise to training data.
        *This arg is not used in the current training structure.*
        """
        return self._get('add_noise', 0, changeable=False)

    @property
    def rotate(self) -> int:
        """
        Rotate dataset to obtain more available training data.
        This arg is the time of rotation, for example set to 1 will 
        rotatetraining data 180 degree once; 
        set to 2 will rotate them 120 degreeand 240 degree.
        *This arg is not used in the current training structure.*
        """
        return self._get('rotate', 0, changeable=False)

    @property
    def test_mode(self) -> str:
        """
        Test settings, canbe `'one'` or `'all'` or `'mix'`.
        When set it to `one`, it will test the model on the `args.force_set` only;
        When set it to `all`, it will test on each of the test dataset in `args.test_set`;
        When set it to `mix`, it will test on all test dataset in `args.test_set` together.
        """
        return self._get('test_mode', 'mix', changeable=True)

    @property
    def max_batch_size(self) -> int:
        """
        Maximun batch size.
        """
        return self._get('max_batch_size', 20000, changeable=True)

    @property
    def lr(self) -> float:
        """
        Learning rate.
        """
        return self._get('lr', 0.001, changeable=False)

    @property
    def init_position(self) -> int:
        """
        ***DO NOT CHANGE THIS***.
        """
        return self._get('init_position', 10000, changeable=False)

    @property
    def window_size_expand_meter(self) -> float:
        """
        ***DO NOT CHANGE THIS***.
        """
        return self._get('window_size_expand_meter', 10.0, changeable=False)

    @property
    def window_size_guidance_map(self) -> int:
        """
        Resolution of map (grids per meter).
        """
        return self._get('window_size_guidance_map', 10, changeable=False)

    @property
    def avoid_size(self) -> int:
        """
        Avoid size in grid cells when modeling social interaction.
        """
        return self._get('avoid_size', 15, changeable=False)

    @property
    def interest_size(self) -> int:
        """
        Interest size in grid cells when modeling social interaction.
        """
        return self._get('interest_size', 20, changeable=False)

    @property
    def map_half_size(self) -> int:
        """
        Local map's half size.
        """
        return self._get('map_half_size', 50, changeable=False)

    @property
    def K(self) -> int:
        """
        Number of multiple generations when test.
        This arg only works for `Generative Models`.
        """
        return self._get('K', 20, changeable=True)

    @property
    def K_train(self) -> int:
        """
        Number of multiple generations when training.
        This arg only works for `Generative Models`.
        """
        return self._get('K_train', 10, changeable=False)

    @property
    def sigma(self) -> float:
        """
        Sigma of noise.
        This arg only works for `Generative Models`.
        """
        return self._get('sigma', 1.0, changeable=True)

    @property
    def draw_distribution(self) -> int:
        """
        Conrtols if draw distributions of predictions instead of points.
        """
        return self._get('draw_distribution', 0, changeable=True)

    @property
    def use_maps(self) -> int:
        """
        Controls if uses the trajectory maps or the social maps in the model.
        """
        return self._get('use_maps', 1, changeable=False)

    @property
    def use_extra_maps(self) -> int:
        """
        Controls if uses the calculated trajectory maps or the given trajectory maps. 
        The model will load maps from `./dataset_npz/.../agent1_maps/trajMap.png`
        if set it to `0`, and load from `./dataset_npz/.../agent1_maps/trajMap_load.png` 
        if set this item to `1`.
        """
        return self._get('use_extra_maps', 0, changeable=True)
