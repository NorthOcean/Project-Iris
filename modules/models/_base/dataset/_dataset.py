'''
Author: Conghao Wong
Date: 2020-11-10 09:31:24
LastEditors: Conghao Wong
LastEditTime: 2021-04-01 20:16:35
Description: file content
'''

from typing import Dict, List, Tuple

import numpy as np


class Dataset():
    """
    Dataset
    -------
    Base structure for controlling each *prediction* dataset.

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
    def __init__(self,
        dataset:str,
        dataset_dir:str,
        order:List[int],
        paras:List[int],
        video_path:str,
        weights:list,
        scale:float,
    ):
        self._dataset        =   dataset
        self._dataset_dir    =   dataset_dir
        self._order          =   order
        self._paras          =   paras
        self._video_path     =   video_path
        self._weights        =   weights
        self._scale          =   scale

    @property
    def dataset(self):
        return self._dataset

    @property
    def dataset_dir(self):
        return self._dataset_dir

    @property
    def order(self):
        return self._order

    @property
    def paras(self):
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
