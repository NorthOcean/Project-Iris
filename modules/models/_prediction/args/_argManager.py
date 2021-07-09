"""
@Author: Conghao Wong
@Date: 2020-11-20 09:11:33
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-09 15:56:50
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import argparse
from ... import base


class BasePredictArgs(base.Args):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

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


class TrainArgsManager(BasePredictArgs):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        
    @property
    def draw_results(self) -> int:
        """
        Controls if draw visualized results on videoframes. Make sure that you have put video files.
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
        Percent of training data used in training datasets. Split with `_` if you want to specify each dataset, for example `0.5_0.9_0.1`.
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
        Rotate dataset to obtain more available training data.This arg is the time of rotation, for example set to 1 will rotatetraining data 180 degree once; set to 2 will rotate them 120 degreeand 240 degree.
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
        Set when to start val during training.Range of this arg is [0.0, 1.0]. The val will start at epoch = args.epochs * args.start_test_percent.
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
        Test settings, canbe `one` or `all` or `mix`.When set to `one`, it will test the test_set only;When set to `all`, it will test on all test datasets of this dataset;When set to `mix`, it will test on one mix dataset that made up of alltest datasets of this dataset.
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
    def gcn_layers(self) -> int:
        """
        Number of GCN layers used in GAN model.
        """
        return self._get('gcn_layers', 3, changeable=False)

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
    def spring_number(self) -> int:
        """
        Experimental.
        """
        return self._get('spring_number', 4, changeable=False)

    @property
    def focus_mode(self) -> int:
        """
        Experimental.
        """
        return self._get('focus_mode', 0, changeable=False)


class OnlineArgsManager(TrainArgsManager):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    @property
    def wait_frames(self) -> int:
        """
        None
        """
        return self._get('wait_frames', 4, changeable=False)

    @property
    def guidance_map_limit(self) -> int:
        """
        None
        """
        return self._get('guidance_map_limit', 10000, changeable=False)

    @property
    def order(self) -> list:
        """
        None
        """
        return self._get('order', [0, 1], changeable=True)

    @property
    def draw_future(self) -> int:
        """
        None
        """
        return self._get('draw_future', 0, changeable=False)

    @property
    def vis(self) -> str:
        """
        None
        """
        return self._get('vis', 'show', changeable=True)

    @property
    def img_save_base_path(self) -> str:
        """
        None
        """
        return self._get('img_save_base_path', './online_vis', changeable=False)

    @property
    def focus_mode(self) -> int:
        """
        None
        """
        return self._get('focus_mode', 0, changeable=False)

    @property
    def run_frames(self) -> int:
        """
        None
        """
        return self._get('run_frames', 1, changeable=False)
