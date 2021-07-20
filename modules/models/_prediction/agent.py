'''
Author: Conghao Wong
Date: 2020-11-20 12:28:31
LastEditors: Conghao Wong
LastEditTime: 2021-04-16 14:57:06
Description: file content
'''

import copy
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from .. import base
from ..helpmethods import dir_check, predict_linear_for_person
from .args import PredictionArgs
from .traj import EntireTrajectory


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


class BaseAgentManager(base.Agent):
    """
    Base Agent Manager
    ------------------
    Agent manager for trajectory prediction, activity analysis (TODO).
    One agent manager contains these items for one specific agent:
    - historical trajectory: `traj`;
    - context map: `socialMap` and `trajMap`;
    - TODO: activity label;
    - TODO: agent category;
    - TODO: agent preference items

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
    >>> self.copy() -> BaseAgentManager

    # rotate context maps for data strengthen
    >>> BaseAgentManager.rotate(mapp:np.ndarray, rotate_angle:float) -> np.ndarray
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

    # 1. Historical Trajectory
    @property
    def traj(self) -> np.ndarray:
        return self._traj

    @traj.setter
    def traj(self, value):
        self._traj = np.array(value).astype(np.float32)

    # 2. Prediction Trajectory
    @property
    def pred(self) -> np.ndarray:
        return self._traj_pred

    @pred.setter
    def pred(self, value):
        self._traj_pred = np.array(value).astype(np.float32)

    # Frame List
    @property
    def frame_list(self) -> list:
        return self._frame_list + self._frame_list_future

    @frame_list.setter
    def frame_list(self, value):
        self._frame_list = value if isinstance(value, list) else value.tolist()

    # Future Frame List
    @property
    def frame_list_future(self) -> list:
        return self._frame_list_future

    @frame_list_future.setter
    def frame_list_future(self, value):
        if isinstance(value, list):
            self._frame_list_future = value
        elif isinstance(value, np.ndarray):
            self._frame_list_future = value.tolist()

    # 3. Linear Prediction
    @property
    def pred_linear(self) -> np.ndarray:
        return self._traj_pred_linear

    @pred_linear.setter
    def pred_linear(self, value):
        self._traj_pred_linear = np.array(value).astype(np.float32)

    # 4. Future Ground Truth
    @property
    def groundtruth(self) -> np.ndarray:
        return self._traj_future

    @groundtruth.setter
    def groundtruth(self, value):
        self._traj_future = np.array(value).astype(np.float32)

    # 5. Trajectory Map
    @property
    def trajMap(self) -> np.ndarray:
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
        if (not type(self._traj_map) == type(None)) and (not type(self._social_map) == type(None)):
            return 0.5 * self._traj_map + 0.5 * self._social_map
        elif (not type(self._traj_map) == type(None)):
            return self._traj_map
        else:
            raise

    # 8. Loss
    @property
    def loss(self):
        return self._loss_dict

    @loss.setter
    def loss(self, dic: dict):
        self._loss_dict = dic

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
            agents:List[BaseAgentManager],
            source=None,
            regulation=True
        ) -> np.ndarray

    # build socialMap (Attention: return `self`)
    >>> MapManager.build_social_map(
            self:MapManager,
            target_agent:BaseAgentManager,
            traj_neighbors=[],
            source=None,
            regulation=True
        ) -> MapManager    
    ```
    """
    def __init__(self, args: PredictionArgs,
                 agents: List[BaseAgentManager],
                 init_manager=None):
        """
        init map manager

        :param args: args to init this manager
        :param agents: a list of `BaseAgentManager` in init the map
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

    def _init_guidance_map(self, agents: List[BaseAgentManager]):
        if issubclass(type(agents[0]), BaseAgentManager):
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
        return guidance_map, W, b

    def build_guidance_map(self, agents: List[BaseAgentManager], source=None, regulation=True) -> np.ndarray:
        self.logger.info('Building Guidance Map...')

        if type(source) == type(None):
            source = self.void_map

        source = source.copy()
        if issubclass(type(agents[0]), BaseAgentManager):
            trajs = get_trajectories(agents)
        else:
            trajs = agents

        source = self._add_to_map(
            source,
            trajs,
            self.real2grid,
            amplitude=1,
            radius=7,
            add_mask=(cv2.imread('./mask_circle.png')[:, :, 0])/50,
            decay=False,
            max_limit=False,
        )
        source = np.minimum(source, 30)
        if regulation:
            source = 1 - source / np.max(source)

        self.logger.info('Done.')
        self.guidance_map = source
        return source

    def build_social_map(self, target_agent: BaseAgentManager, traj_neighbors=[], source=None, regulation=True):
        if type(source) == type(None):
            source = self.void_map

        source = source.copy()
        add_mask = (cv2.imread('./mask_circle.png')[:, :, 0])
        pred_frames = target_agent.total_frame - target_agent.obs_length

        trajs = []
        amps = []
        rads = []

        # Destination
        trajs.append(target_agent.pred_linear.tolist())
        amps.append([-2 for _ in range(pred_frames)])
        rads.append(self.args.interest_size)

        # Interplay
        amp_neighbors = []
        rads_neighbors = [
            self.args.avoid_size for _ in range(len(traj_neighbors))]

        vec_target = target_agent.pred_linear[-1] - target_agent.pred_linear[0]
        len_target = calculate_length(vec_target)
        for pred in traj_neighbors:
            vec_neighbor = pred[-1] - pred[0]
            cosine = activation(
                calculate_cosine(vec_target, vec_neighbor),
                a=1.0,
                b=0.2,
            ) if len_target >= 0.05 else 1.0

            velocity = (calculate_length(vec_neighbor) /
                        calculate_length(vec_target)) if len_target >= 0.05 else 2.0
            amp_neighbors.append(
                [-cosine*velocity for _ in range(pred_frames)])

        amps += amp_neighbors
        trajs += traj_neighbors
        rads += rads_neighbors

        source = self._add_to_map(target_map=source,
                                  trajs=trajs,
                                  map_function=self.real2grid,
                                  amplitude=amps,
                                  radius=rads,
                                  add_mask=add_mask)

        if regulation:
            if (np.max(source) - np.min(source)) <= 0.01:
                source = 0.5 * np.ones_like(source)
            else:
                source = (source - np.min(source)) / \
                    (np.max(source) - np.min(source))

        self.full_map = source
        return self

    def _add_to_map(self, target_map, trajs: np.array, map_function, amplitude=1, radius=0, add_mask=None, interp=False, max_limit=False, decay=True):
        """
        `amplitude`: Value of each add point. Accept both `float` and `np.array` types.
        `radius`: Raduis of each add point. Accept both `float` and `np.array` types.
        """
        if not type(trajs) == np.array:
            trajs = np.array(trajs)

        if len(trajs.shape) == 2:
            trajs = np.reshape(trajs, [1, trajs.shape[0], trajs.shape[1]])

        n_traj = trajs.shape[0]
        amplitude = np.array(amplitude)
        if not len(amplitude.shape):
            amplitude = amplitude * \
                np.ones([n_traj, trajs.shape[-2]], dtype=np.int32)
            radius = radius * np.ones(n_traj, dtype=np.int32)

        target_map = target_map.copy()

        if type(add_mask) == type(None):
            add_mask = np.ones([1, 1], dtype=np.int32)

        if interp:
            trajs_grid = [interp_2d(map_function(traj), step=1)
                          for traj in trajs]
        else:
            trajs_grid = [map_function(traj) for traj in trajs]

        for traj, a, r in zip(trajs_grid, amplitude, radius):
            add_mask = cv2.resize(add_mask, (r*2+1, r*2+1))
            target_map = self._add_one_traj(
                target_map, traj, a, r, add_mask, max_limit=max_limit, amplitude_decay=decay)

        return target_map

    def real2grid(self, traj: np.array):
        return ((traj - self.b) * self.W).astype(np.int32)

    def _add_one_traj(
        self, source_map, traj, amplitude, radius, add_mask,
        max_limit=True, amplitude_decay=False,
        amplitude_decay_p=np.array([[0.0, 0.7, 1.0], [1.0, 1.0, 0.5]])
    ):
        """
        `amplitude` is a ndarray, shape = [len(traj)]
        """
        if amplitude_decay:
            amplitude = amplitude * np.interp(
                np.arange(0, len(traj))/len(traj),
                amplitude_decay_p[0],
                amplitude_decay_p[1],
            )

        new_map = np.zeros_like(source_map)
        for pos, a in zip(traj, amplitude):
            if pos[0]-radius >= 0 and pos[1]-radius >= 0 and pos[0]+radius+1 < new_map.shape[0] and pos[1]+radius+1 < new_map.shape[1]:
                new_map[pos[0]-radius:pos[0]+radius+1, pos[1]-radius:pos[1]+radius+1] = a * \
                    add_mask + new_map[pos[0]-radius:pos[0] +
                                       radius+1, pos[1]-radius:pos[1]+radius+1]

        if max_limit:
            new_map = np.sign(new_map)

        return new_map + source_map


class TrainAgentManager(BaseAgentManager):
    """
    Train Agent Manager
    -------------------
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

    _save_items = BaseAgentManager._save_items + [
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



def interp_2d(traj: np.array, step=1):
    """
    shape(traj) should be [m, 2].
    """
    x = traj
    if type(step) == int:
        step = step * np.ones(2).astype(np.int32)

    x_p = []
    index = 0
    while True:
        if len(x_p):
            x_last = x_p[-1]
            if np.linalg.norm(x[index] - x_last, ord=1) >= np.min(step):
                coe = np.sign(x[index] - x_last)
                coe_mask = (abs(x[index] - x_last) ==
                            np.max(abs(x[index] - x_last)))
                x_p.append(x_last + coe * coe_mask * step)
                continue

        if len(x_p) and np.linalg.norm(x[index] - x_last, ord=1) > 0:
            x_p.append(x[index])
        elif len(x_p) == 0:
            x_p.append(x[index])

        index += 1
        if index >= len(x):
            break

    return np.array(x_p)


def calculate_cosine(vec1, vec2):
    """
    两个输入均为表示方向的向量, shape=[2]
    """
    length1 = np.linalg.norm(vec1)
    length2 = np.linalg.norm(vec2)

    if length2 == 0:
        return -1.0
    else:
        return np.sum(vec1 * vec2) / (length1 * length2)


def calculate_length(vec1):
    """
    表示方向的向量, shape=[2]
    """
    length1 = np.linalg.norm(vec1)
    return length1


def activation(x: np.array, a=1, b=1):
    return (x <= 0) * a * x + (x > 0) * b * x


def get_trajectories(agents: List[BaseAgentManager],
                     return_movement=False,
                     return_destination=False,
                     destination_steps=3) -> list:
    """
    Get trajectories from input structures.

    :param agents: trajectory manager, support both `BaseAgentManager` and `EntireTrajectory`
    :param return_movement: controls if return move flag
    :return trajs: a list of all trajectories from inputs
    """
    all_trajs = []
    movement = []
    for agent in agents:
        if issubclass(type(agent), BaseAgentManager):
            trajs = agent.traj
        elif issubclass(type(agent), EntireTrajectory):
            trajs = agent.traj[agent.start_frame:agent.end_frame]
            
            if return_destination:
                trajs = trajs[-destination_steps:]

        if return_movement:
            # FIXME start_frame > end_frame on SDD
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
