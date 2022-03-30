"""
@Author: Conghao Wong
@Date: 2020-11-10 09:31:24
@LastEditors: Conghao Wong
@LastEditTime: 2022-03-30 19:26:12
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import os

from typing import Dict, List, Tuple


def load_from_plist(path: str) -> dict:
    """
    Load plist files into python `dict` object.
    It is used to fix error when loading plist files through
    `biplist.readPlist()` in python 3.9 or newer.

    :param path: path of the plist file
    :return dat: a `dict` object loaded from the file
    """

    import plistlib
    import sys
    import biplist

    v = sys.version
    if int(v.split('.')[1]) >= 9:
        with open(path, 'rb') as f:
            dat = plistlib.load(f)
    else:
        dat = biplist.readPlist(path)

    return dat


class Dataset():
    """
    Dataset
    -------
    Base structure for controlling each video dataset.

    Properties
    -----------------
    ```python
    >>> self.dataset        # dataset name
    >>> self.dataset_dir    # dataset folder
    >>> self.order          # X, Y order
    >>> self.paras          # [sample_step, frame_rate]
    >>> self.video_path     # video path
    >>> self.weights        # transfer weights from real scales to pixels
    >>> self.scale          # video scales
    ```
    """

    def __init__(self, dataset: str,
                 dataset_dir: str,
                 order: List[int],
                 paras: List[int],
                 video_path: str,
                 weights: list,
                 scale: float):

        self._dataset = dataset
        self._dataset_dir = dataset_dir
        self._order = order
        self._paras = paras
        self._video_path = video_path
        self._weights = weights
        self._scale = scale

    @staticmethod
    def get(dataset: str, root_dir='./datasets/subsets'):
        plist_path = os.path.join(root_dir, '{}.plist'.format(dataset))
        try:
            dic = load_from_plist(plist_path)
        except:
            raise FileNotFoundError(
                'Dataset file `{}`.plist NOT FOUND.'.format(dataset))

        return Dataset(**dic)

    @property
    def dataset(self):
        return self._dataset

    @property
    def dataset_dir(self):
        """
        Dataset folder, which contains a `*.txt` or `*.csv` dataset file, and a scene image `reference.jpg`
        """
        return self._dataset_dir

    @property
    def order(self):
        """
        order for coordinates, (x, y) -> `[0, 1]`, (y, x) -> `[1, 0]`
        """
        return self._order

    @property
    def paras(self):
        """
        [sample_step, frame_rate]
        """
        return self._paras

    @property
    def video_path(self):
        return self._video_path

    @property
    def weights(self):
        return self._weights

    @property
    def scale(self):
        return self._scale


class DatasetsInfo():
    def __init__(self, dataset: str, root_dir='./datasets'):
        plist_path = os.path.join(root_dir, '{}.plist'.format(dataset))
        try:
            dic = load_from_plist(plist_path)
        except:
            raise FileNotFoundError(
                'Dataset file `{}`.plist NOT FOUND.'.format(dataset))

        self.train_sets = dic['train']
        self.test_sets = dic['test']
        self.val_sets = dic['val']
