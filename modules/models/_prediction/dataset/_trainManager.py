'''
Author: Conghao Wong
Date: 2021-01-08 09:52:34
LastEditors: Conghao Wong
LastEditTime: 2021-04-15 11:15:54
Description: file content
'''

import os
import random
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from ... import base
from ..._helpmethods import dir_check
from ..agent._agentManager import MapManager, TrainAgentManager
from ..args._argManager import TrainArgsManager as Args
from ..training._entireTraj import EntireTrajectory
from ._datasetManager import PredictionDatasetManager


class DatasetManager(base.DatasetManager):

    arg_type = Args
    agent_type = TrainAgentManager

    def __init__(self, args: Args, dataset_name: str, custom_list=[]):
        """
        init parameters:
        :param args: train args, type = `NameSpace`.
            See detail args in `./modules/models/_managers/_argManager.py`.
        :param dataset_name: name for this dataset
        :param custom_list: (optional) 
        """
        super().__init__(args, dataset_name)

        self._dataset_info = PredictionDatasetManager()(dataset_name)
        self._custom_list = custom_list

    @property
    def args(self) -> arg_type:
        return self._args

    def load_data(self):
        if len(self._custom_list) == 3:
            self.video_neighbor_list, self.video_matrix, self.frame_list = self._custom_list
        else:
            self.video_neighbor_list, self.video_matrix, self.frame_list = self._load_data()

        self.all_entire_trajectories = self._prepare_agent_data()
        return self

    def sample_train_data(self) -> List[TrainAgentManager]:
        """
        Read Dataset, load data, and make tain data
        """
        self.load_data()
        return self._sample_train_data()

    def _load_data(self):
        """
        Load (or make) dataset data

        :return video_neighbor_list: a `np.ndarray` that contains all neighbor info
        :return video_matrix: a `np.ndarray` that contains all trajectories, axises are time and agent index
        :return frame_list: a list of agent index
        """
        dir_check('./dataset_npz')
        base_path = dir_check('./dataset_npz/{}'.format(self.dataset_name))
        npy_path = './dataset_npz/{}/data.npz'.format(self.dataset_name)

        if os.path.exists(npy_path):
            all_data = np.load(npy_path, allow_pickle=True)
            video_neighbor_list = all_data['video_neighbor_list']
            video_matrix = all_data['video_matrix']
            frame_list = all_data['frame_list']

        else:
            person_data, frame_list = self._load_csv(self.dataset_name)
            person_list = list(person_data.keys())

            count_p = len(person_list)
            count_f = len(frame_list)

            video_matrix = self.args.init_position * \
                np.ones([count_f, count_p, 2])

            # true_person_id -> person_index
            person_dict = dict(zip(person_list, np.arange(count_p)))

            # true_frame_id -> frame_index
            frame_dict = dict(zip(frame_list, np.arange(count_f)))

            for person, person_index in self.log_timebar(
                    person_dict.items(),
                    return_enumerate=False):

                frame_list_current \
                    = (person_data[person]).T[0].astype(np.int32).tolist()
                frame_index_current \
                    = np.array([frame_dict[frame_current]
                                for frame_current in frame_list_current])
                video_matrix[frame_index_current, person_index, :] \
                    = person_data[person][:, 1:]

            video_neighbor_list = np.array([
                np.where(np.not_equal(data.T[0], self.args.init_position))[0]
                for data in video_matrix], dtype=object)

            np.savez(npy_path,
                     video_neighbor_list=video_neighbor_list,
                     video_matrix=video_matrix,
                     frame_list=frame_list)

        return video_neighbor_list, video_matrix, frame_list

    def _load_csv(self, dataset_name) -> Tuple[Dict[int, np.ndarray], list]:
        """
        Read trajectory data from csv file.

        :param dataset_name: name of the dataset. See Details in `datasetManager.py`
        :return person_data: data sorted by person ids. type = `Dict[int, np.ndarray]`
        :return frame_list: a list of all frame indexs
        """
        dataset_dir_current = self.dataset_info.dataset_dir
        order = self.dataset_info.order

        csv_file_path = os.path.join(dataset_dir_current, 'true_pos_.csv')
        data = np.genfromtxt(csv_file_path, delimiter=',').T

        # Load data, and sort by person id
        person_data = {}
        person_list = set(data.T[1].astype(np.int32))
        for person in person_list:
            index_current = np.where(data.T[1] == person)[0]
            person_data[person] = np.column_stack([
                data[index_current, 0],
                data[index_current, 2 + order[0]],
                data[index_current, 2 + order[1]]])

        frame_list = list(set(data.T[0].astype(np.int32)))
        frame_list.sort()

        self.logger.info('Load dataset {} done.'.format(csv_file_path))
        return person_data, frame_list

    def _prepare_agent_data(self) -> List[EntireTrajectory]:
        """
        Get data of type `EntireTrajectory` from video matrix for each agent in dataset.
        """
        self.frame_number, self.agent_count, _ = self.video_matrix.shape
        all_entire_trajectories = []
        for person in range(self.agent_count):
            all_entire_trajectories.append(
                EntireTrajectory(person,
                                 self.video_neighbor_list,
                                 self.video_matrix,
                                 self.frame_list,
                                 self.args.init_position))

        return all_entire_trajectories

    def _get_trajectory(self, agent_index, start_frame, obs_frame, end_frame, frame_step=1, add_noise=False):
        """
        Sample single part of one specific agent's trajectory from `EntireTrajectory`.

        :return agent: agent manager, type = `TrainAgentManager`
        """
        trajecotry_current = self.all_entire_trajectories[agent_index]
        frame_list = trajecotry_current.frame_list
        neighbor_list = trajecotry_current.video_neighbor_list[obs_frame - frame_step].tolist(
        )
        neighbor_list = set(neighbor_list) - set([agent_index])
        neighbor_agents = [self.all_entire_trajectories[nei]
                           for nei in neighbor_list]

        return TrainAgentManager().init_data(trajecotry_current,
                                             neighbor_agents,
                                             frame_list,
                                             start_frame,
                                             obs_frame,
                                             end_frame,
                                             frame_step=frame_step,
                                             add_noise=add_noise)

    def _sample_train_data(self) -> List[TrainAgentManager]:
        """
        Sample all train data (type = `TrainAgentManager`) from all `EntireTrajectory`.
        """
        sample_rate, frame_rate = self.dataset_info.paras
        frame_step = int(0.4 / (sample_rate / frame_rate))

        # Sample all train agents
        train_agents = []

        for agent_id, _ in self.log_timebar(range(self.agent_count),
                                            'Prepare train data...'):
            trajecotry_current = self.all_entire_trajectories[agent_id]
            start_frame = trajecotry_current.start_frame
            end_frame = trajecotry_current.end_frame

            for frame_point in range(start_frame, end_frame, self.args.step * frame_step):
                # Normal mode
                if self.args.pred_frames > 0:
                    if frame_point + (self.args.obs_frames + self.args.pred_frames) * frame_step > end_frame:
                        break

                    obs_frame_current = frame_point + self.args.obs_frames * frame_step
                    end_frame_current = frame_point + \
                        (self.args.obs_frames+self.args.pred_frames) * frame_step

                # Infinity mode, only works for destination models
                elif self.args.pred_frames == -1:
                    if frame_point + (self.args.obs_frames + 1) * frame_step > end_frame:
                        break

                    obs_frame_current = frame_point + self.args.obs_frames * frame_step
                    end_frame_current = end_frame

                else:
                    raise ValueError(
                        '`pred_frames` should be a positive integer or -1.')

                train_agents.append(self._get_trajectory(agent_id,
                                                         start_frame=frame_point,
                                                         obs_frame=obs_frame_current,
                                                         end_frame=end_frame_current,
                                                         frame_step=frame_step,
                                                         add_noise=False))

        # Write Maps
        map_manager = MapManager(self.args, train_agents)
        map_manager.build_guidance_map(agents=train_agents)
        np.save('./dataset_npz/{}/gm.npy'.format(self.dataset_name),
                np.array([map_manager], dtype=object))

        for index, agent in self.log_timebar(train_agents,
                                             'Building Social Map...'):
            train_agents[index].trajMap = map_manager
            train_agents[index].socialMap = map_manager.build_social_map(
                target_agent=train_agents[index],
                traj_neighbors=train_agents[index].get_pred_traj_neighbor_linear())

        return train_agents


class DatasetsManager(base.DatasetsManager):
    """
    Train Data Manager
    ------------------
    Manage all training data.

    Public Methods
    --------------
    ```python
    # Prepare train agents from `DatasetManager`s
    >>> self.prepare_train_files(
            dataset_managers:List[DatasetManager],
            mode='test'
        ) -> List[TrainAgentManager]

    # Save agents' data
    >>> DatasetsManager.zip_and_save(
            save_dir, 
            agents:List[TrainAgentManager]
        )

    # Load agents' data
    >>> DatasetsManager.load_and_unzip(save_dir) -> List[TrainAgentManager]
    ```
    """

    arg_type = Args
    datasetInfo_type = PredictionDatasetManager
    agent_type = TrainAgentManager
    datasetManager_type = DatasetManager

    def __init__(self, args: Args, prepare_type='all'):
        super().__init__(args)
        self.prepare_datasets(prepare_type)

    def prepare_datasets(self, prepare_type='all'):
        self.dataset_list = self.dataset_info.dataset_list[self.args.dataset]
        test_list = self.dataset_info.dataset_list[self.args.dataset + 'test']

        if self.args.dataset == 'ethucy':
            self.train_list = [
                i for i in self.dataset_list if not i == self.args.test_set]
            self.val_list = [self.args.test_set]

        elif self.args.dataset == 'sdd':
            self.train_list = [
                i for i in self.dataset_list if not i in self.dataset_info.sdd_test_sets + self.dataset_info.sdd_val_sets]
            self.val_list = self.dataset_info.sdd_test_sets

        if prepare_type == 'all':
            self.train_info = self.get_train_and_test_agents()

        elif prepare_type == 'test':
            for index, dataset in enumerate(test_list):
                self.logger.info('Preparing {}/{}...'.format(index+1, len(test_list)))
                self.prepare_train_files([DatasetManager(self.args, dataset)])

        elif '_' in prepare_type:
            set_list = prepare_type.split('_')
            for index, dataset in enumerate(set_list):
                self.logger.info('Preparing {}/{}...'.format(index+1, len(set_list)))
                self.prepare_train_files([DatasetManager(self.args, dataset)])
        else:
            pass

    @property
    def args(self) -> arg_type:
        return self._args

    @property
    def dataset_info(self) -> datasetInfo_type:
        return self._dataset_info

    def prepare_train_files(self,
                            dataset_managers: List[datasetManager_type],
                            mode='test') -> List[agent_type]:
        """
        Make or load train files to get train agents.
        (a list of agent managers, type = `TrainAgentManager`)

        :param dataset_managers: a list of dataset managers (`DatasetManager`)
        :return all_agents: a list of train agents (`TrainAgentManager`)
        """
        all_agents = []
        count = 1
        dir_check('./dataset_npz/')

        for dm in dataset_managers:
            self.log('({}/{})  Prepare test data in `{}`...'.format(count,
                                                                    len(dataset_managers),
                                                                    dm.dataset_name))

            data_path = './dataset_npz/{}/agent'.format(dm.dataset_name) if (self.args.obs_frames == 8 and self.args.pred_frames == 12) \
                else './dataset_npz/{}/agent_{}to{}'.format(dm.dataset_name, self.args.obs_frames, self.args.pred_frames)
            
            endstring = '' if self.args.step == 4 else '{}'.format(self.args.step)
            data_path += '{}.npz'.format(endstring)

            if not os.path.exists(data_path):
                agents = dm.sample_train_data()
                self.zip_and_save(data_path, agents)
            else:
                agents = self.load_and_unzip(data_path)

            if mode == 'train':
                if (train_percent := self.train_percent[dm.dataset_name]) < 1.0:
                    agents = random.sample(
                        agents, int(train_percent * len(agents)))
                if (self.args.rotate > 0) and (length := len(agents)):
                    rotate_step = 360//(self.args.rotate + 1)
                    for time in tqdm(range(1, self.args.rotate + 1), desc='Making rotation data', file=self.log_function if 'write' in dir(self.log_function) else None):
                        rotate_angle = rotate_step * time
                        new_agents = [agent.copy().rotate(rotate_angle)
                                      for agent in agents[:length]]
                        agents += new_agents

            all_agents += agents
            count += 1
        return all_agents

    @staticmethod
    def zip_and_save(save_dir, agents: List[TrainAgentManager]):
        save_dict = {}
        for index, agent in enumerate(agents):
            save_dict[str(index)] = agent.zip_data()
        np.savez(save_dir, **save_dict)

    @classmethod
    def load_and_unzip(cls, save_dir) -> List[TrainAgentManager]:
        save_dict = np.load(save_dir, allow_pickle=True)

        if save_dict['0'].tolist()['__version__'] < TrainAgentManager.__version__:
            cls.log(('[Warning] Saved agent managers\' version is {}, ' +
                     'which is lower than current {}. Please delete ' +
                     'them and re-run this program, or there could ' +
                     'happen something wrong.').format(save_dict['0'].tolist()['__version__'],
                                                       TrainAgentManager.__version__))

        return [TrainAgentManager().load_data(save_dict[key].tolist()) for key in save_dict.keys()]
