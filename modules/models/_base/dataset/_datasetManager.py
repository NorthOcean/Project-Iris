'''
Author: Conghao Wong
Date: 2021-04-13 12:03:47
LastEditors: Conghao Wong
LastEditTime: 2021-04-15 09:38:34
Description: file content
'''

from typing import List, Tuple

from ...helpmethods import dir_check
from ..baseObject import BaseObject
from ..agent import Agent
from ..args.args import BaseTrainArgs as Args
from ._dataset import Dataset
from ._datasetInfo import DatasetInfo


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
    datasetInfo_type = DatasetInfo

    def __init__(self, args: Args, **kwargs):
        super().__init__()

        self._args = args
        self._dataset_info = self.datasetInfo_type()

    @property
    def args(self) -> arg_type:
        return self._args

    @property
    def dataset_info(self) -> datasetInfo_type:
        return self._dataset_info

    def get_train_and_test_agents(self) \
        -> Tuple[List[agent_type],
                 List[agent_type],
                 int,
                 int]:
        """
        Get training data
        """
        datasets = self.dataset_info.dataset_list[self.args.dataset]
        self.train_percent = dict(
            zip(datasets, [1.0 for _ in range(len(datasets))]))

        percents = [
            item for item in self.args.train_percent.split('_') if len(item)]
        if len(percents) == 1:
            percent = min(max(0.0, float(percents[0])), 1.0)
            self.train_percent = dict(
                zip(datasets, [percent for _ in range(len(datasets))]))

        else:
            for index, percent in enumerate(percents):
                percent = min(max(0.0, float(percent)), 1.0)
                dataset_name = self.dataset_info.dataset_list[self.args.dataset][index]
                self.train_percent[dataset_name] = percent

        # Prepare train agents
        data_managers_train = []
        sample_number_original = 0
        sample_time = 1
        for dataset in self.train_list:
            dm = self.datasetManager_type(self.args, dataset)
            data_managers_train.append(dm)
            dm.load_data()
            sample_number_original += dm.agent_count

        # Prepare test agents
        data_managers_test = []
        for dataset in self.val_list:
            data_managers_test.append(self.datasetManager_type(self.args, dataset))

        # Prepare test and train data
        test_agents = self.prepare_train_files(data_managers_test)
        train_agents = self.prepare_train_files(
            data_managers_train, mode='train')
        sample_time = len(train_agents) // sample_number_original

        return (train_agents,
                test_agents,
                len(train_agents),
                sample_time)

    def prepare_train_files(
            self,
            dataset_managers: List[datasetManager_type],
            mode='test') -> List[agent_type]:
        raise NotImplementedError('Please rewrite this method.')

    @classmethod
    def load_dataset_files(
        cls,
        args: arg_type,
        dataset: str
    ) -> List[agent_type]:

        dir_check('./dataset_npz')
        dm = cls(args, prepare_type='noprepare')
        return dm.prepare_train_files(
            dataset_managers=[cls.datasetManager_type(
                args, dataset_name=dataset)],
            mode='test')
