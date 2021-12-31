"""
@Author: Conghao Wong
@Date: 2021-01-08 09:52:34
@LastEditors: Conghao Wong
@LastEditTime: 2021-12-29 15:51:57
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import os
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np

from ... import base
from ...helpmethods import dir_check
from ..__agent import PredictionAgent
from ..__args import PredictionArgs
from ..__maps import MapManager
from ..__traj import EntireTrajectory
from ..__utils import calculate_length


class TrajMapNotFoundError(FileNotFoundError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class DatasetManager(base.DatasetManager):
    """
    DatasetManager
    --------------
    Manage all training data from one prediction dataset (subdataset).

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
    # Sample train data (a list of `PredictionAgent` objects) from dataset
    (method) sample_train_data: (self: DatasetManager) -> List[PredictionAgent]

    # Load dataset files
    (method) load_data: (self: DatasetManager) -> DatasetManager
    """

    arg_type = PredictionArgs
    agent_type = PredictionAgent

    def __init__(self, args: PredictionArgs, dataset_name: str, custom_list=[]):
        """
        init parameters:
        :param args: train args, type = `NameSpace` or a subclass of `PredictionArgs`.
        :param dataset_name: name for this dataset
        :param custom_list: (optional) 
        """
        super().__init__(args, dataset_name)

        self._dataset_info = base.Dataset.get(dataset_name)
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

    def sample_train_data(self) -> List[PredictionAgent]:
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

        self.log('Load dataset {} done.'.format(csv_file_path))
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

    def _get_trajectory(self, agent_index, start_frame, obs_frame, end_frame, frame_step=1, max_neighbor=15, add_noise=False):
        """
        Sample single part of one specific agent's trajectory from `EntireTrajectory`.

        :return agent: agent manager, type = `PredictionAgent`
        """
        trajecotry_current = self.all_entire_trajectories[agent_index]
        frame_list = trajecotry_current.frame_list

        neighbor_list = self.video_neighbor_list[obs_frame - frame_step]

        if len(neighbor_list) > max_neighbor + 1:
            neighbor_pos = self.video_matrix[obs_frame -
                                             frame_step, neighbor_list, :]
            target_pos = self.video_matrix[obs_frame -
                                           frame_step, agent_index:agent_index+1, :]
            dis = calculate_length(neighbor_pos - target_pos)
            neighbor_list = neighbor_list[np.argsort(dis)[1:max_neighbor+1]]

        neighbor_agents = [self.all_entire_trajectories[nei]
                           for nei in neighbor_list]

        return PredictionAgent().init_data(trajecotry_current,
                                           neighbor_agents,
                                           frame_list,
                                           start_frame,
                                           obs_frame,
                                           end_frame,
                                           frame_step=frame_step,
                                           add_noise=add_noise)

    def _sample_train_data(self) -> List[PredictionAgent]:
        """
        Sample all train data (type = `PredictionAgent`) from all `EntireTrajectory`.
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

            for p in range(start_frame, end_frame, self.args.step * frame_step):
                # Normal mode
                if self.args.pred_frames > 0:
                    if p + (self.args.obs_frames + self.args.pred_frames) * frame_step > end_frame:
                        break

                    obs = p + self.args.obs_frames * frame_step
                    end = p + (self.args.obs_frames +
                               self.args.pred_frames) * frame_step

                # Infinity mode, only works for destination models
                elif self.args.pred_frames == -1:
                    if p + (self.args.obs_frames + 1) * frame_step > end_frame:
                        break

                    obs = p + self.args.obs_frames * frame_step
                    end = end_frame

                else:
                    raise ValueError(
                        '`pred_frames` should be a positive integer or -1.')

                train_agents.append(self._get_trajectory(agent_id,
                                                         start_frame=p,
                                                         obs_frame=obs,
                                                         end_frame=end,
                                                         frame_step=frame_step,
                                                         add_noise=False))
        return train_agents

    def make_maps(self, agents: List[PredictionAgent],
                  base_path: str,
                  save_map_file: str = None,
                  save_social_file: str = 'socialMap.npy',
                  save_para_file: str = 'para.txt',
                  save_centers_file: str = 'centers.txt'):
        """
        Make maps for input agents, and save them in the numpy format.

        :param agents: a list of agents that ready to calculate maps
        :param base_path: base folder to save the map and map parameters
        :param load_map_file: file name for the saved trajectory map (`.jpg` or `.png`).
        default is `None`. When this item is `None`, MapManager will build
        trajectory maps according to trajectories of the input agents.
        :param save_map_file: file name to save the built traj map
        :param save_social_file: file name to save the social map (already cut)
        :param save_para_file: file name to save the map parameters
        :param save_centers_file: path to save the centers
        """

        map_manager = MapManager(self.args, agents)

        if save_map_file:
            traj_map = map_manager.build_guidance_map(
                agents=agents,
                save=os.path.join(base_path, save_map_file))

        social_maps = []
        centers = []
        for agent in self.log_timebar(agents,
                                      'Build maps...',
                                      return_enumerate=False):

            centers.append(agent.traj[-1:, :])
            social_maps.append(map_manager.build_social_map(
                target_agent=agent,
                traj_neighbors=agent.get_pred_traj_neighbor_linear()))

        social_maps = np.array(social_maps)  # (batch, a, b)

        centers = np.concatenate(centers, axis=0)
        centers = map_manager.real2grid(centers)
        cuts = map_manager.cut_map(social_maps,
                                   centers,
                                   self.args.map_half_size)
        paras = map_manager.real2grid_paras

        np.savetxt(os.path.join(base_path, save_centers_file), centers)
        np.savetxt(os.path.join(base_path, save_para_file), paras)
        np.save(os.path.join(base_path, save_social_file), cuts)


class DatasetsManager(base.DatasetsManager):
    """
    DatasetsManager
    ---------------
    Manage all prediction training data.

    Public Methods
    --------------
    ```python
    # Prepare train agents from `DatasetManager`s
    (method) load_fromManagers: (self: DatasetsManager, dataset_managers: List[DatasetManager], mode='test') -> List[PredictionAgent]

    # Save and load agents' data
    (method) zip_and_save: (save_dir, agents: List[PredictionAgent]) -> None
    (method) load_and_unzip: (cls: Type[DatasetsManager], save_dir) -> List[PredictionAgent]
    ```
    """

    arg_type = PredictionArgs
    agent_type = PredictionAgent
    datasetManager_type = DatasetManager

    def __init__(self, args: PredictionArgs):
        super().__init__(args)

    @property
    def args(self) -> arg_type:
        return self._args

    def load_fromManagers(self, dataset_managers: List[DatasetManager],
                          mode='test') -> List[PredictionAgent]:
        """
        Make or load train files to get train agents.
        (a list of agent managers, type = `PredictionAgent`)

        :param dataset_managers: a list of dataset managers (`DatasetManager`)
        :return all_agents: a list of train agents (`PredictionAgent`)
        """
        all_agents = []
        count = 1
        dir_check('./dataset_npz/')

        for dm in dataset_managers:
            print('({}/{})  Prepare test data in `{}`...'.format(
                count, len(dataset_managers), dm.dataset_name))

            if (self.args.obs_frames, self.args.pred_frames) == (8, 12):
                data_path = './dataset_npz/{}/agent'.format(dm.dataset_name)
            else:
                data_path = './dataset_npz/{}/agent_{}to{}'.format(dm.dataset_name,
                                                                   self.args.obs_frames,
                                                                   self.args.pred_frames)

            endstring = '' if self.args.step == 4 else self.args.step
            data_path += '{}.npz'.format(endstring)

            if not os.path.exists(data_path):
                agents = dm.sample_train_data()
                self.zip_and_save(data_path, agents)
            else:
                agents = self.load_and_unzip(data_path)
            self.log('Successfully load train agents from `{}`'.format(data_path))

            if self.args.use_maps:
                map_path = dir_check(data_path.split('.np')[0] + '_maps')
                map_file = ('trajMap.png' if not self.args.use_extra_maps
                            else 'trajMap_load.png')

                try:
                    agents = self.load_maps(map_path, agents,
                                            map_file=map_file,
                                            social_file='socialMap.npy',
                                            para_file='para.txt',
                                            centers_file='centers.txt')

                except TrajMapNotFoundError:
                    path = os.path.join(map_path, map_file)
                    self.log(s := ('Trajectory map `{}`'.format(path) +
                                   ' not found, stop running...'),
                             level='error')
                    exit()

                except:
                    self.log('Load maps failed, start re-making...')

                    dm.make_maps(agents, map_path,
                                 save_map_file='trajMap.png',
                                 save_social_file='socialMap.npy',
                                 save_para_file='para.txt',
                                 save_centers_file='centers.txt')

                    agents = self.load_maps(map_path, agents,
                                            map_file=map_file,
                                            social_file='socialMap.npy',
                                            para_file='para.txt',
                                            centers_file='centers.txt')

                self.log('Successfully load maps from `{}`.'.format(map_path))

            all_agents += agents
            count += 1
        return all_agents

    def zip_and_save(self, save_dir, agents: List[PredictionAgent]):
        save_dict = {}
        for index, agent in enumerate(agents):
            save_dict[str(index)] = agent.zip_data()
        np.savez(save_dir, **save_dict)

    def load_and_unzip(self, save_dir) -> List[PredictionAgent]:
        save_dict = np.load(save_dir, allow_pickle=True)

        if save_dict['0'].tolist()['__version__'] < PredictionAgent.__version__:
            self.log(('Saved agent managers\' version is {}, ' +
                      'which is lower than current {}. Please delete ' +
                      'them and re-run this program, or there could ' +
                      'happen something wrong.').format(save_dict['0'].tolist()['__version__'],
                                                        PredictionAgent.__version__),
                     level='error')

        return [PredictionAgent().load_data(save_dict[key].tolist()) for key in save_dict.keys()]

    def load_maps(self, base_path: str,
                  agents: List[PredictionAgent],
                  map_file: str,
                  social_file: str,
                  para_file: str,
                  centers_file: str) -> List[PredictionAgent]:
        """
        Load maps from the base folder

        :param base_path: base save folder
        :param agents: agents to assign maps
        :param map_file: file name for traj maps, support `.jpg` or `.png`
        :param social_file: file name for social maps, support `.npy`
        :param para_file: file name for map parameters, support `.txt`
        :param centers_file: file name for centers, support `.txt`

        :return agents: agents with maps
        """
        traj_map = cv2.imread(os.path.join(base_path, map_file))

        if traj_map is None:
            if self.args.use_extra_maps:
                raise TrajMapNotFoundError
            else:
                raise FileNotFoundError

        traj_map = (traj_map[:, :, 0]).astype(np.float32)/255.0

        social_map = np.load(os.path.join(
            base_path, social_file), allow_pickle=True)
        para = np.loadtxt(os.path.join(base_path, para_file))
        centers = np.loadtxt(os.path.join(base_path, centers_file))

        batch_size = len(social_map)
        traj_map = np.repeat(traj_map[np.newaxis, :, :], batch_size, axis=0)
        traj_map_cut = MapManager.cut_map(traj_map,
                                          centers,
                                          self.args.map_half_size)

        for agent, t_map, s_map in zip(agents, traj_map_cut, social_map):
            PredictionAgent.set_map(agent, 0.5*t_map + 0.5*s_map, para)

        return agents
