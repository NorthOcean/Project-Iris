"""
@Author: Conghao Wong
@Date: 2021-04-13 12:52:39
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-20 15:45:21
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import Dict, List, Tuple

from ._dataset import Dataset


class DatasetInfo():
    """
    DatasetInfo
    -----------
    Manage `Dataset` object of several video datasets.

    Usage
    -----
    You can new a `DatasetInfo` object, and then call it to access dataset info.
    For example,
    ```python
    >>> di = DatasetInfo()
    >>> di('eth')
    ```

    Properties
    ----------
    ```python
    # dataset dict
    (property) datasets: (self: DatasetInfo) -> Dict[str, Dataset]

    # dataset list
    (property) dataset_list: (self: DatasetInfo) -> Dict[str, List[str]]
    ```
    """
    def __init__(self):
        super().__init__()

        self._datasets = {}
        self._dataset_list = {}

    @property
    def datasets(self) -> Dict[str, Dataset]:
        return self._datasets

    @property
    def dataset_list(self) -> Dict[str, List[str]]:
        return self._dataset_list

    def __call__(self, dataset: str) -> Dataset:
        if type(dataset) == str and len(dataset) == 1:
            dataset = int(dataset)

        if type(dataset) == int and dataset in range(0, len(self.datasets)):
            name_list = [key for key in self.datasets]
            dataset = name_list[dataset]

        if dataset.endswith('_R') or dataset.endswith('_Rxy'):
            dataset = dataset.split('_')[0]

        return self.datasets[dataset]
