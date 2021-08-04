"""
@Author: Conghao Wong
@Date: 2020-11-10 09:31:24
@LastEditors: Conghao Wong
@LastEditTime: 2021-08-04 14:52:59
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import Dict, List, Tuple

import numpy as np

from ... import base


class PredictionDatasetInfo(base.DatasetInfo):
    """
    PredictionDatasetInfo
    ---------------------
    Manager for all prediction datasets in ETH-UCY and SDD.

    Usage
    -----
    You can new a `PredictionDatasetInfo` object, 
    and then call it to access dataset info.
    For example,
    ```python
    >>> myManager = PredictionDatasetInfo()
    >>> dataset = myManager('eth')
    ```

    Properties
    ----------
    ```python
    # A dict of `Dataset`s
    >>> self.datasets -> Dict[str, Dataset]

    # A dict of names of all datasets
    >>> self.dataset_list -> Dict[str, List[str]]
    ```
    """

    def __init__(self):
        super().__init__()

        self._add_eth_ucy()
        self._add_sdd()
        self.datasets['eth'].dataset

    def _add_eth_ucy(self):
        self.datasets['eth'] = base.Dataset(
            dataset='eth',
            dataset_dir='./data/eth/univ',
            order=[1, 0],
            paras=[6, 25],
            video_path='./videos/eth.mp4',
            weights=[np.array([
                [2.8128700e-02, 2.0091900e-03, -4.6693600e+00],
                [8.0625700e-04, 2.5195500e-02, -5.0608800e+00],
                [3.4555400e-04, 9.2512200e-05, 4.6255300e-01],
            ]), 0.65, 225, 0.6, 160],
            scale=1,
        )

        self.datasets['hotel'] = base.Dataset(
            dataset='hotel',
            dataset_dir='./data/eth/hotel',
            order=[0, 1],
            paras=[10, 25],
            video_path='./videos/hotel.mp4',
            weights=[np.array([
                [-1.5966000e-03, 1.1632400e-02, -5.3951400e+00],
                [1.1048200e-02, 6.6958900e-04, -3.3295300e+00],
                [1.1190700e-04, 1.3617400e-05, 5.4276600e-01],
            ]), 0.54, 470, 0.54, 300],
            scale=1,
        )

        self.datasets['zara1'] = base.Dataset(
            dataset='zara1',
            dataset_dir='./data/ucy/zara/zara01',
            order=[1, 0],
            paras=[10, 25],
            video_path='./videos/zara1.mp4',
            weights=[-42.54748107, 580.5664891, 47.29369894, 3.196071003],
            scale=1,
        )

        self.datasets['zara2'] = base.Dataset(
            dataset='zara2',
            dataset_dir='./data/ucy/zara/zara02',
            order=[1, 0],
            paras=[10, 25],
            video_path='./videos/zara2.mp4',
            weights=[-42.54748107, 630.5664891, 47.29369894, 3.196071003],
            scale=1,
        )

        self.datasets['univ'] = base.Dataset(
            dataset='univ',
            dataset_dir='./data/ucy/univ/students001',
            order=[1, 0],
            paras=[10, 25],
            video_path='./videos/students003.mp4',
            weights=[-41.1428, 576, 48, 0],
            scale=1,
        )

        self.datasets['zara3'] = base.Dataset(
            dataset='zara3',
            dataset_dir='./data/ucy/zara/zara03',
            order=[1, 0],
            paras=[10, 25],
            video_path='./videos/zara2.mp4',
            weights=[-42.54748107, 630.5664891, 47.29369894, 3.196071003],
            scale=1,
        )

        self.datasets['univ3'] = base.Dataset(
            dataset='univ3',
            dataset_dir='./data/ucy/univ/students003',
            order=[1, 0],
            paras=[10, 25],
            video_path='./videos/students003.mp4',
            weights=[-41.1428, 576, 48, 0],
            scale=1,
        )

        self.datasets['unive'] = base.Dataset(
            dataset='unive',
            dataset_dir='./data/ucy/univ/uni_examples',
            order=[1, 0],
            paras=[10, 25],
            video_path='./videos/students003.mp4',
            weights=[-41.1428, 576, 48, 0],
            scale=1,
        )

        self.ethucy_testsets = ['eth', 'hotel', 'zara1', 'zara2', 'univ']
        self.dataset_list['ethucy'] = list(self.datasets.keys())
        self.dataset_list['ethucytest'] = list(self.datasets.keys())[:5]

    def _add_sdd(self):
        # SDD Datasets
        self._sdd_sets = {
            'quad':   [[0, 1, 2, 3], 100.0],
            'little':   [[0, 1, 2, 3], 100.0],
            'deathCircle':   [[0, 1, 2, 3, 4], 100.0],
            'hyang':   [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 100.0],
            'nexus':   [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 100.0],
            'coupa':   [[0, 1, 2, 3], 100.0],
            'bookstore':   [[0, 1, 2, 3, 4, 5, 6], 100.0],
            'gates':   [[0, 1, 2, 3, 4, 5, 6, 7, 8], 100.0],
        }

        self.sdd_test_sets = [
            'hyang7',
            'hyang11',
            'bookstore6',
            'nexus3',
            'deathCircle4',
            'hyang6',
            'hyang3',
            'little1',
            'hyang13',
            'gates8',
            'gates7',
            'hyang2',
        ]

        self.sdd_val_sets = [
            'nexus7',
            'coupa1',
            'gates4',
            'little2',
            'bookstore3',
            'little3',
            'nexus4',
            'hyang4',
            'gates3',
            'quad2',
            'gates1',
            'hyang9',
        ]

        for base_set in self._sdd_sets:
            for index in self._sdd_sets[base_set][0]:
                self.datasets['{}{}'.format(base_set, index)] = base.Dataset(
                    dataset='{}{}'.format(base_set, index),
                    dataset_dir='./data/sdd/{}/video{}'.format(
                        base_set, index),
                    order=[1, 0],
                    paras=[1, 30],
                    video_path='./videos/sdd_{}_{}.mov'.format(
                        base_set, index),
                    weights=[self._sdd_sets[base_set][1], 0.0,
                             self._sdd_sets[base_set][1], 0.0],
                    scale=2,
                )

        self.dataset_list['sdd'] = [name for name in self.datasets.keys(
        ) if not name in self.dataset_list['ethucy']]
        self.dataset_list['sdd'].sort()
        self.dataset_list['sddtest'] = self.sdd_test_sets
