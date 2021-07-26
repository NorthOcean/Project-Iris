"""
@Author: Conghao Wong
@Date: 2020-11-20 09:11:33
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-22 10:59:50
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
        Controls if draw visualized results on videoframes.
        Make sure that you have put video files.
        """
        return self._get('draw_results', 0, changeable=True)

    @property
    def dataset(self) -> str:
        """
        Dataset. Can be `ethucy` or `sdd`.
        """
        return self._get('dataset', 'ethucy', changeable=True)

    @property
    def train_percent(self) -> str:
        """
        Percent of training data used in training datasets.
        Split with `_` if you want to specify each dataset,
        for example `0.5_0.9_0.1`.
        """
        return self._get('train_percent', '1.0_', changeable=False)

    @property
    def step(self) -> int:
        """
        Frame step for obtaining training data.
        """
        return self._get('step', 1, changeable=True)

    @property
    def add_noise(self) -> int:
        """
        Controls if add noise to training data
        """
        return self._get('add_noise', 0, changeable=False)

    @property
    def rotate(self) -> int:
        """
        Rotate dataset to obtain more available training data.
        This arg is the time of rotation, for example set to 1 will 
        rotatetraining data 180 degree once; 
        set to 2 will rotate them 120 degreeand 240 degree.
        """
        return self._get('rotate', 0, changeable=False)

    @property
    def test(self) -> int:
        """
        Controls if run test.
        """
        return self._get('test', 1, changeable=False)

    @property
    def start_test_percent(self) -> float:
        """
        Set when to start val during training.
        Range of this arg is [0.0, 1.0]. 
        The val will start at epoch = args.epochs * args.start_test_percent.
        """
        return self._get('start_test_percent', 0.0, changeable=False)

    @property
    def test_step(self) -> int:
        """
        Val step in epochs.
        """
        return self._get('test_step', 3, changeable=False)

    @property
    def test_mode(self) -> str:
        """
        Test settings, canbe `one` or `all` or `mix`.
        When set to `one`, it will test the test_set only;
        When set to `all`, it will test on all test datasets of this dataset;
        When set to `mix`, it will test on one mix dataset 
        that made up of alltest datasets of this dataset.
        """
        return self._get('test_mode', 'one', changeable=True)

    @property
    def epochs(self) -> int:
        """
        Training epochs.
        """
        return self._get('epochs', 500, changeable=False)

    @property
    def batch_size(self) -> int:
        """
        Training batch_size.
        """
        return self._get('batch_size', 5000, changeable=False)

    @property
    def max_batch_size(self) -> int:
        """
        Maximun batch_size.
        """
        return self._get('max_batch_size', 20000, changeable=True)

    @property
    def dropout(self) -> float:
        """
        Dropout rate.
        """
        return self._get('dropout', 0.5, changeable=False)

    @property
    def lr(self) -> float:
        """
        Learning rate.
        """
        return self._get('lr', 0.001, changeable=False)

    @property
    def save_model(self) -> int:
        """
        Controls if save the model.
        """
        return self._get('save_model', 1, changeable=False)

    @property
    def save_best(self) -> int:
        """
        Controls if save the best model when val.
        """
        return self._get('save_best', 1, changeable=False)

    @property
    def diff_weights(self) -> float:
        """
        Parameter of linera prediction.
        """
        return self._get('diff_weights', 0.95, changeable=False)

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
        Resolution of map.(grids per meter)
        """
        return self._get('window_size_guidance_map', 10, changeable=False)

    @property
    def avoid_size(self) -> int:
        """
        Avoid size in grids.
        """
        return self._get('avoid_size', 15, changeable=False)

    @property
    def interest_size(self) -> int:
        """
        Interest size in grids.
        """
        return self._get('interest_size', 20, changeable=False)

    @property
    def map_half_size(self) -> int:
        """
        Local map's size.
        """
        return self._get('map_half_size', 50, changeable=False)

    @property
    def K(self) -> int:
        """
        Number of multiple generation when test.
        """
        return self._get('K', 20, changeable=True)

    @property
    def K_train(self) -> int:
        """
        Number of multiple generation when training.
        """
        return self._get('K_train', 10, changeable=False)

    @property
    def sigma(self) -> float:
        """
        Sigma of noise.
        """
        return self._get('sigma', 1.0, changeable=True)

    @property
    def draw_distribution(self) -> int:
        """
        Conrtols if draw distributions ofpredictions instead of points.
        """
        return self._get('draw_distribution', 0, changeable=True)

    @property
    def prepare_type(self) -> str:
        """
        Prepare argument. Do Not Change it.
        """
        return self._get('prepare_type', 'test', changeable=True)

    @property
    def use_maps(self) -> int:
        """
        Controls if uses the trajectory maps in models.
        Do not change it when test or training.
        """
        return self._get('use_maps', 1, changeable=False)
