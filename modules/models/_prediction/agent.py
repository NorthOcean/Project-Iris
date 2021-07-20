'''
Author: Conghao Wong
Date: 2020-11-20 12:28:31
LastEditors: Conghao Wong
LastEditTime: 2021-04-16 14:57:06
Description: file content
'''

import copy
import os
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np

from .. import base
from ..helpmethods import dir_check, predict_linear_for_person
from .args import PredictionArgs
from .traj import EntireTrajectory

MASK = cv2.imread('./mask_circle.png')[:, :, 0]/50
MASKS = {}


def prepare_rotate_matrix(min_angel=1, base_path='./.cache', save_name='./rotate_matrix.npy'):
    need_to_re_calculate = True
    dir_check(base_path)
    save_path = os.path.join(base_path, save_name)
    if os.path.exists(save_path):
        rotate_matrix = np.load(save_path)
        if rotate_matrix.shape[0] == 360//min_angel:
            need_to_re_calculate = False

    if need_to_re_calculate:
        angles = np.arange(0, 2 * np.pi, min_angel * np.pi / 180)
        sin = np.sin(angles)
        cos = np.cos(angles)

        rotate_matrix = np.empty((angles.shape[0], 2, 2))
        rotate_matrix[..., 0, 0] = cos
        rotate_matrix[..., 0, 1] = -sin
        rotate_matrix[..., 1, 0] = sin
        rotate_matrix[..., 1, 1] = cos
        np.save(save_path, rotate_matrix)

    return rotate_matrix


ROTATE_MATRIX = prepare_rotate_matrix()


class BasePredictionAgent(base.Agent):
    """
    BasePredictionManager
    ---------------------
    Agent manager for trajectory prediction, activity analysis (TODO).
    One agent manager contains these items for one specific agent:
    - historical trajectory: `traj`;
    - context map: `socialMap` and `trajMap`;
    - future works: activity label;
    - future works: agent category;
    - future works: agent preference items

    Properties
    ----------
    ```python
    self.traj -> np.ndarray     # historical trajectory
    self.pred -> np.ndarray     # predicted (future) trajectory
    self.frame_list -> List[int]    # a list of frame index when this agent appeared
    self.frame_list_future -> List[int]     # agent's future frame index
    self.pred_linear -> np.ndarray  # agent's linear prediction
    self.groundtruth -> np.ndarray  # agent's future trajectory (when available)

    self.fusionMap  -> np.ndarray   # agent's context map
    self.loss -> Dict[str, np.ndarray]  # loss of agent's prediction
    ```

    Public Methods
    --------------
    ```python
    # copy this manager to a new address
    >>> self.copy() -> BasePredictionAgent

    # rotate context maps for data strengthen
    >>> BasePredictionAgent.rotate(mapp:np.ndarray, rotate_angle:float) -> np.ndarray
    ```
    """

    __version__ = 2.0

    _save_items = ['_traj', '_traj_future',
                   '_traj_pred', '_traj_pred_linear',
                   '_frame_list', '_frame_list_future',
                   '_traj_map', '_social_map', 'real2grid',
                   '__version__']

    def __init__(self):
        self._traj = []
        self._traj_future = []

        self._traj_pred = None
        self._traj_pred_linear = None

        self._traj_map = None
        self._social_map = None
        self.real2grid = None

        self._frame_list = []
        self._frame_list_future = []

    def copy(self):
        return copy.deepcopy(self)

    @property
    def traj(self) -> np.ndarray:
        """
        historical trajectory, shape = (obs, 2)
        """
        return self._traj

    @traj.setter
    def traj(self, value):
        self._traj = np.array(value).astype(np.float32)

    @property
    def pred(self) -> np.ndarray:
        """
        predicted trajectory, shape = (pred, 2)
        """
        return self._traj_pred

    @pred.setter
    def pred(self, value):
        self._traj_pred = np.array(value).astype(np.float32)

    @property
    def frame_list(self) -> list:
        """
        a list of frame index during observation and prediction time.
        shape = (obs + pred, 2)
        """
        return self._frame_list + self._frame_list_future

    @frame_list.setter
    def frame_list(self, value):
        self._frame_list = value if isinstance(value, list) else value.tolist()

    @property
    def frame_list_future(self) -> list:
        """
        a list of frame index during prediction time.
        shape = (pred, 2)
        """
        return self._frame_list_future

    @frame_list_future.setter
    def frame_list_future(self, value):
        if isinstance(value, list):
            self._frame_list_future = value
        elif isinstance(value, np.ndarray):
            self._frame_list_future = value.tolist()

    @property
    def pred_linear(self) -> np.ndarray:
        """
        linear prediction.
        shape = (pred, 2)
        """
        return self._traj_pred_linear

    @pred_linear.setter
    def pred_linear(self, value):
        self._traj_pred_linear = np.array(value).astype(np.float32)

    @property
    def groundtruth(self) -> np.ndarray:
        """
        ground truth future trajectory.
        shape = (pred, 2)
        """
        return self._traj_future

    @groundtruth.setter
    def groundtruth(self, value):
        self._traj_future = np.array(value).astype(np.float32)

    @property
    def trajMap(self) -> np.ndarray:
        """
        trajectory map, shape = (100, 100)
        """
        return self._traj_map

    @trajMap.setter
    def trajMap(self, trajmap):
        full_map = trajmap.guidance_map
        half_size = trajmap.args.map_half_size

        center_pos = trajmap.real2grid(self.traj[-1])
        original_map = cv2.resize(full_map[
            np.maximum(center_pos[0]-2*half_size, 0):np.minimum(center_pos[0]+2*half_size, full_map.shape[0]),
            np.maximum(center_pos[1]-2*half_size, 0):np.minimum(center_pos[1]+2*half_size, full_map.shape[1]),
        ], (4*half_size, 4*half_size))

        final_map = original_map[half_size:3*half_size, half_size:3*half_size]
        self._traj_map = final_map.astype(np.float32)
        self.real2grid = trajmap.real2grid_paras

    # 6. Social Map
    @property
    def socialMap(self) -> np.ndarray:
        """
        social map, shape = (100, 100)
        """
        return self._social_map

    @socialMap.setter
    def socialMap(self, trajmap):
        half_size = trajmap.args.map_half_size
        center_pos = trajmap.real2grid(self.traj[-1])
        full_map = trajmap.full_map

        original_map = cv2.resize(full_map[
            np.maximum(center_pos[0]-2*half_size, 0):np.minimum(center_pos[0]+2*half_size, full_map.shape[0]),
            np.maximum(center_pos[1]-2*half_size, 0):np.minimum(center_pos[1]+2*half_size, full_map.shape[1]),
        ], (4*half_size, 4*half_size))

        final_map = original_map[half_size:3*half_size, half_size:3*half_size]
        self._social_map = final_map.astype(np.float32)
        self.real2grid = trajmap.real2grid_paras

    # 7. Fusion Map
    @property
    def fusionMap(self):
        """
        fusion map that combines trajectory map and social map
        """
        if (not type(self._traj_map) == type(None)) and (not type(self._social_map) == type(None)):
            return 0.5 * self._traj_map + 0.5 * self._social_map
        elif (not type(self._traj_map) == type(None)):
            return self._traj_map
        else:
            raise

    @staticmethod
    def rotate_map(mapp: np.ndarray, rotate_angle):
        map_shape = mapp.shape
        final_map = cv2.warpAffine(
            mapp,
            cv2.getRotationMatrix2D(
                (map_shape[0]//2, map_shape[1]//2),
                rotate_angle,
                1,
            ),
            (map_shape[0], map_shape[1]),
        )
        return final_map

    def zip_data(self) -> Dict[str, object]:
        zipped = {}
        for item in self._save_items:
            zipped[item] = getattr(self, item)
        return zipped

    def load_data(self, zipped_data: Dict[str, object]):
        for item in self._save_items:
            if not item in zipped_data.keys():
                continue
            else:
                setattr(self, item, zipped_data[item])
        return self


class MapManager(base.BaseObject):
    """
    Map Manager
    -----------
    Map manager that manages agent's context map.

    Usage
    -----
    ```python
    # build guidanceMap
    >>> MapManager.build_guidance_map(
            self:MapManager,
            agents:List[BasePredictionAgent],
            source=None,
            regulation=True
        ) -> np.ndarray

    # build socialMap (Attention: return `self`)
    >>> MapManager.build_social_map(
            self:MapManager,
            target_agent:BasePredictionAgent,
            traj_neighbors=[],
            source=None,
            regulation=True
        ) -> MapManager    
    ```
    """

    def __init__(self, args: PredictionArgs,
                 agents: List[BasePredictionAgent],
                 init_manager=None):
        """
        init map manager

        :param args: args to init this manager
        :param agents: a list of `BasePredictionAgent` object to init the map
        :init_manager: a map manager to init this (available)
        """

        super().__init__()

        self.args = args
        self.agents = agents

        if init_manager:
            self.void_map, self.W, self.b = [
                init_manager.void_map, init_manager.W, init_manager.b]
        else:
            self.void_map, self.W, self.b = self._init_guidance_map(agents)

    @property
    def real2grid_paras(self) -> np.ndarray:
        return np.stack([self.W, self.b])   # (2, 2)

    def _init_guidance_map(self, agents: List[BasePredictionAgent]):
        if issubclass(type(agents[0]), BasePredictionAgent):
            traj = get_trajectories(agents)
        else:
            traj = agents

        traj = np.array(traj)
        # shape of `traj` should be [*, *, 2] or [*, 2]

        if len(traj.shape) == 3:
            traj = np.reshape(traj, [-1, 2])

        x_max = np.max(traj[:, 0])
        x_min = np.min(traj[:, 0])
        y_max = np.max(traj[:, 1])
        y_min = np.min(traj[:, 1])

        guidance_map = np.zeros([
            int((x_max - x_min + 2*self.args.window_size_expand_meter)
                * self.args.window_size_guidance_map) + 1,
            int((y_max - y_min + 2*self.args.window_size_expand_meter)
                * self.args.window_size_guidance_map) + 1,
        ])
        W = np.array([self.args.window_size_guidance_map,
                      self.args.window_size_guidance_map])
        b = np.array([x_min - self.args.window_size_expand_meter,
                      y_min - self.args.window_size_expand_meter])
        self.map_coe = [x_max, x_min, y_max, y_min]
        return guidance_map.astype(np.float32), W, b

    def build_guidance_map(self, agents: Union[List[BasePredictionAgent], np.ndarray],
                           source: np.ndarray = None,
                           regulation=True) -> np.ndarray:
        """
        Build guidance map

        :param agents: a list of agents or trajectories to calculate the map
        :param source: source map, default are zeros
        :param regulation: controls if scale the map into [0, 1]
        """

        self.log('Building Guidance Map...')

        if type(source) == type(None):
            source = self.void_map

        source = source.copy()
        if issubclass(type(agents[0]), BasePredictionAgent):
            trajs = get_trajectories(agents)
        else:
            trajs = agents

        source = self._add_to_map(source,
                                  self.real2grid(trajs),
                                  amplitude=1,
                                  radius=7,
                                  add_mask=MASK,
                                  decay=False,
                                  max_limit=False)

        source = np.minimum(source, 30)
        if regulation:
            source = 1 - source / np.max(source)

        self.log('Done.')
        self.guidance_map = source
        return source

    def build_social_map(self, target_agent: BasePredictionAgent,
                         traj_neighbors: np.ndarray = [],
                         source: np.ndarray = None,
                         regulation=True,
                         max_neighbor=15):
        """
        Build social map

        :param target_agent: target `BasePredictionAgent` object to calculate the map
        :param traj_neighbor: neighbors' predictions
        :param source: source map, default are zeros
        :param regulation: controls if scale the map into [0, 1]
        """

        if type(source) == type(None):
            source = self.void_map

        if not type(traj_neighbors) == np.ndarray:
            traj_neighbors = np.array(traj_neighbors)

        source = source.copy()
        pred_frames = self.args.pred_frames

        trajs = []
        amps = []
        rads = []

        # Destination
        trajs.append(target_agent.pred_linear)
        amps.append(-2)
        rads.append(self.args.interest_size)

        # Interplay
        amp_neighbors = []
        rads_neighbors = self.args.avoid_size * np.ones(len(traj_neighbors))

        vec_target = target_agent.pred_linear[-1] - target_agent.pred_linear[0]
        len_target = calculate_length(vec_target)

        vec_neighbor = traj_neighbors[:, -1] - traj_neighbors[:, 0]

        if len_target >= 0.05:
            cosine = activation(
                calculate_cosine(vec_target[np.newaxis, :], vec_neighbor),
                a=1.0,
                b=0.2)
            velocity = (calculate_length(vec_neighbor) /
                        calculate_length(vec_target[np.newaxis, :]))

        else:
            cosine = np.ones(len(traj_neighbors))
            velocity = 2

        amp_neighbors = - cosine * velocity

        amps += amp_neighbors.tolist()
        trajs += traj_neighbors.tolist()
        rads += rads_neighbors.tolist()

        if len(trajs) > max_neighbor + 1:
            trajs = np.array(trajs)
            dis = calculate_length(trajs[:1, 0, :] - trajs[:, 0, :])
            index = np.argsort(dis)
            trajs = trajs[index[:max_neighbor+1]]

        source = self._add_to_map(target_map=source,
                                  grid_trajs=self.real2grid(trajs),
                                  amplitude=amps,
                                  radius=rads,
                                  add_mask=MASK,
                                  max_limit=False,
                                  decay=True)

        if regulation:
            if (np.max(source) - np.min(source)) <= 0.01:
                source = 0.5 * np.ones_like(source)
            else:
                source = (source - np.min(source)) / \
                    (np.max(source) - np.min(source))

        self.full_map = source
        return self

    def _add_to_map(self, target_map: np.ndarray,
                    grid_trajs: np.ndarray,
                    amplitude: np.ndarray = 1,
                    radius: np.ndarray = 0,
                    add_mask=None,
                    max_limit=False,
                    decay=True):
        """
        `amplitude`: Value of each add point. Accept both `float` and `np.array` types.
        `radius`: Raduis of each add point. Accept both `float` and `np.array` types.
        """

        if len(grid_trajs.shape) == 2:
            grid_trajs = grid_trajs[np.newaxis, :, :]

        n_traj = grid_trajs.shape[0]
        amplitude = np.array(amplitude)
        if not len(amplitude.shape):
            amplitude = amplitude * \
                np.ones([n_traj, grid_trajs.shape[-2]], dtype=np.int32)
            radius = radius * np.ones(n_traj, dtype=np.int32)

        target_map = target_map.copy()

        if type(add_mask) == type(None):
            add_mask = np.ones([1, 1], dtype=np.int32)

        for traj, a, r in zip(grid_trajs, amplitude, radius):
            r = int(r)
            if not r in MASKS.keys():
                MASKS[r] = cv2.resize(add_mask, (r*2+1, r*2+1))

            add_mask = MASKS[r]
            target_map = self._add_one_traj(target_map,
                                            traj, a, r,
                                            add_mask,
                                            max_limit=max_limit,
                                            amplitude_decay=decay)

        return target_map

    def real2grid(self, traj: np.ndarray) -> np.ndarray:
        if not type(traj) == np.ndarray:
            traj = np.array(traj)

        return ((traj - self.b) * self.W).astype(np.int32)

    def _add_one_traj(self, source_map: np.ndarray,
                      traj: np.ndarray,
                      amplitude: float,
                      radius: int,
                      add_mask: np.ndarray,
                      max_limit=True,
                      amplitude_decay=False,
                      amplitude_decay_p=np.array([[0.0, 0.7, 1.0], [1.0, 1.0, 0.5]])):

        if amplitude_decay:
            amplitude = amplitude * np.interp(np.linspace(0, 1, len(traj)),
                                              amplitude_decay_p[0],
                                              amplitude_decay_p[1])

        new_map = np.zeros_like(source_map)
        for pos, a in zip(traj, amplitude):
            if (pos[0]-radius >= 0 and 
                pos[1]-radius >= 0 and 
                pos[0]+radius+1 < new_map.shape[0] and 
                pos[1]+radius+1 < new_map.shape[1]):

                new_map[pos[0]-radius:pos[0]+radius+1, pos[1]-radius:pos[1]+radius+1] = \
                    a * add_mask + \
                    new_map[pos[0]-radius:pos[0]+radius+1, pos[1]-radius:pos[1]+radius+1]

        if max_limit:
            new_map = np.sign(new_map)

        return new_map + source_map


class PredictionAgent(BasePredictionAgent):
    """
    PredictionAgent
    ---------------
    Agent manager used to train prediction models.

    Additional Public Methods
    -------------------------
    ```python
    # rotate (to strengthen data)
    >>> self.rotate(rotate_angle)

    # get neighbors' trajs -> List[np.ndarray]
    >>> self.get_neighbor_traj()

    # get neighbors' linear predictions
    >>> self.get_pred_traj_neighbor_linear() -> List[np.ndarray]
    ```
    """

    _save_items = BasePredictionAgent._save_items + [
        'linear_predict',
        'neighbor_number',
        'neighbor_traj', 'neighbor_traj_linear_pred',
        'obs_length', 'total_frame']

    def __init__(self):
        super().__init__()

        self.linear_predict = False
        self.obs_length = 0
        self.total_frame = 0

        self.neighbor_number = 0
        self.neighbor_traj = []
        self.neighbor_traj_linear_pred = []

    def init_data(self, target_agent, neighbor_agents,
                  frame_list, start_frame,
                  obs_frame, end_frame,
                  frame_step=1,
                  add_noise=False,
                  linear_predict=True):

        self.linear_predict = linear_predict

        # Trajectory info
        self.obs_length = (obs_frame - start_frame) // frame_step
        self.total_frame = (end_frame - start_frame) // frame_step

        # Trajectory
        whole_traj = target_agent.traj[start_frame:end_frame:frame_step]
        frame_list_current = frame_list[start_frame:end_frame:frame_step]

        # data strengthen: noise
        if add_noise:
            noise_curr = np.random.normal(0, 0.1, size=self.traj.shape)
            whole_traj += noise_curr

        self.frame_list = frame_list_current[:self.obs_length]
        self.traj = whole_traj[:self.obs_length]
        self.groundtruth = whole_traj[self.obs_length:]
        self.frame_list_future = frame_list_current[self.obs_length:]

        if linear_predict:
            self.pred_linear = predict_linear_for_person(
                self.traj, time_pred=self.total_frame)[self.obs_length:]

        # Neighbor info
        self.neighbor_traj = []
        self.neighbor_traj_linear_pred = []
        for neighbor in neighbor_agents:
            neighbor_traj = neighbor.traj[start_frame:obs_frame:frame_step]
            if neighbor_traj.max() >= 5000:
                available_index = np.where(neighbor_traj.T[0] <= 5000)[0]
                neighbor_traj[:available_index[0],
                              :] = neighbor_traj[available_index[0]]
                neighbor_traj[available_index[-1]:,
                              :] = neighbor_traj[available_index[-1]]
            self.neighbor_traj.append(neighbor_traj)

            if linear_predict:
                pred = predict_linear_for_person(neighbor_traj, time_pred=self.total_frame)[
                    self.obs_length:]
                self.neighbor_traj_linear_pred.append(pred)

        self.neighbor_number = len(neighbor_agents)
        return self

    def rotate(self, rotate_angle):
        traj_current = self.traj
        future_traj_current = self.groundtruth
        neighbor_traj = self.get_neighbor_traj()

        center = traj_current[-1]
        traj_current_r = rotate_trajectory(traj_current, center, rotate_angle)
        future_traj_current_r = rotate_trajectory(
            future_traj_current, center, rotate_angle)
        neighbor_traj_r = [rotate_trajectory(
            traj, center, rotate_angle) for traj in neighbor_traj]

        self.traj = traj_current_r
        self.groundtruth = future_traj_current_r
        self.neighbor_traj = neighbor_traj_r

        if True:
            self.pred_linear = predict_linear_for_person(
                self.traj, time_pred=self.total_frame)[self.obs_length:]
            self.neighbor_traj_linear_pred = []
            for neighbor_traj in neighbor_traj_r:
                pred = predict_linear_for_person(neighbor_traj, time_pred=self.total_frame)[
                    self.obs_length:]
                self.neighbor_traj_linear_pred.append(pred)

        if not type(self._traj_map) == type(None):
            self._traj_map = self.rotate_map(self._traj_map, rotate_angle)
        if not type(self._social_map) == type(None):
            self._social_map = self.rotate_map(self._social_map, rotate_angle)

        return self

    def get_neighbor_traj(self):
        return self.neighbor_traj

    def clear_all_neighbor_info(self):
        self.neighbor_traj = []
        self.neighbor_traj_linear_pred = []

    def get_pred_traj_neighbor_linear(self) -> list:
        return self.neighbor_traj_linear_pred


def calculate_cosine(vec1: np.ndarray,
                     vec2: np.ndarray):

    length1 = np.linalg.norm(vec1, axis=-1)
    length2 = np.linalg.norm(vec2, axis=-1)

    return (np.sum(vec1 * vec2, axis=-1) + 0.0001) / ((length1 * length2) + 0.0001)


def calculate_length(vec1):
    return np.linalg.norm(vec1, axis=-1)


def activation(x: np.ndarray, a=1, b=1):
    return np.less_equal(x, 0) * a * x + np.greater(x, 0) * b * x


def get_trajectories(agents: List[BasePredictionAgent],
                     return_movement=False,
                     return_destination=False,
                     destination_steps=3) -> list:
    """
    Get trajectories from input structures.

    :param agents: trajectory manager, support both `BasePredictionAgent` and `EntireTrajectory`
    :param return_movement: controls if return move flag
    :return trajs: a list of all trajectories from inputs
    """
    all_trajs = []
    movement = []
    for agent in agents:
        if issubclass(type(agent), BasePredictionAgent):
            trajs = agent.traj
        elif issubclass(type(agent), EntireTrajectory):
            trajs = agent.traj[agent.start_frame:agent.end_frame]

            if return_destination:
                trajs = trajs[-destination_steps:]

        if return_movement:
            flag = True if (
                (trajs.shape[0] == 0) or
                (calculate_length(trajs[-1]-trajs[0]) >= return_movement)
            ) else False
            movement += [flag for _ in range(len(trajs))]

        if type(trajs) == np.ndarray:
            trajs = trajs.tolist()
        all_trajs += trajs

    return (all_trajs, movement) if return_movement else all_trajs


def rotate_trajectory(traj, center, rotate_angle):
    """
    rotate trajecotry according to the center point

    :param traj: trajectory, shape = `[m, 2]`
    :param center: rotation center, shape = `[2]`
    :param rotate_angle: rotation angle, in degree
    """
    R = ROTATE_MATRIX[rotate_angle, :, :]
    traj_R = center + np.matmul(traj-center, R)
    return traj_R
