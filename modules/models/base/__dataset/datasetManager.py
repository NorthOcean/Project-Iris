"""
@Author: Conghao Wong
@Date: 2021-04-13 12:03:47
@LastEditors: Conghao Wong
@LastEditTime: 2021-12-31 10:17:00
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import List, Tuple, Union

from ...helpmethods import dir_check
from ..__agent import Agent
from ..__args.args import BaseTrainArgs as Args
from ..__baseObject import BaseObject
from .dataset import Dataset, DatasetsInfo


class DatasetManager(BaseObject):
    """
    Dataset Manager
    --------------
    Manage all training data from one dataset.

    Properties
    ----------
    ```python
    >>> self.args   # args
    >>> self.dataset_name # name
    >>> self.dataset_info # dataset info
    ```

    Public Methods
    --------------
    ```python
    # Sample train data from dataset
    (method) sample_train_data: (self: DatasetManager) -> List[agent_type]

    # Load dataset files
    (method) load_data: (self: DatasetManager) -> Any
    ```
    """

    arg_type = Args
    agent_type = Agent

    def __init__(self, args: arg_type, dataset_name: str):
        super().__init__()
        self._args = args
        self._dataset_name = dataset_name

        self._dataset_info = None
        self.agent_count = None

    @property
    def args(self) -> arg_type:
        return self._args

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def dataset_info(self) -> Dataset:
        return self._dataset_info

    def load_data(self):
        raise NotImplementedError('Please rewrite this method.')

    def sample_train_data(self) -> List[agent_type]:
        raise NotImplementedError('Please rewrite this method.')


class DatasetsManager(BaseObject):

    arg_type = Args
    agent_type = Agent
    datasetManager_type = DatasetManager

    def __init__(self, args: Args, **kwargs):
        super().__init__()

        self._args = args
        self.dataset_info = DatasetsInfo(args.test_set)

    @property
    def args(self) -> arg_type:
        return self._args

    def load_fromManagers(self, dataset_managers: List[datasetManager_type],
                          mode='test') -> List[agent_type]:

        raise NotImplementedError('Please rewrite this method.')

    @classmethod
    def load(cls, args: arg_type, dataset: Union[str, List[str]], mode: str):
        """
        Load train samples in sub-dataset(s).

        :param args: args used
        :param dataset: dataset to load. Set it to `'auto'` to load train agents
        :param mode: load mode, canbe `'test'` or `'train'`
        :return agents: loaded agents. It returns a list of `[train_agents, test_agents]` when `mode` is `'train'`.
        """
        dir_check('./dataset_npz')
        Dm = cls(args)

        if dataset == 'auto':
            train_sets = Dm.dataset_info.train_sets
            test_sets = Dm.dataset_info.test_sets

            train_agents = cls.load(args, train_sets, mode='train')
            test_agents = cls.load(args, test_sets, mode='test')

            return train_agents, test_agents

        else:
            if type(dataset) == str:
                dataset = [dataset]

            dms = [cls.datasetManager_type(args, d) for d in dataset]
            return Dm.load_fromManagers(dms, mode=mode)
