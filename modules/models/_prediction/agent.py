"""
@Author: Conghao Wong
@Date: 2020-11-20 12:28:31
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-22 11:43:09
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import copy
from typing import Dict, List, Tuple, Union

import numpy as np

from .. import base
from ..helpmethods import predict_linear_for_person


class PredictionAgent(base.Agent):
    """
    PredictionManager
    -----------------
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

    self.Map  -> np.ndarray   # agent's context map
    self.loss -> Dict[str, np.ndarray]  # loss of agent's prediction
    ```

    Public Methods
    --------------
    ```python
    # copy this manager to a new address
    >>> self.copy() -> BasePredictionAgent

    # get neighbors' trajs -> List[np.ndarray]
    >>> self.get_neighbor_traj()

    # get neighbors' linear predictions
    >>> self.get_pred_traj_neighbor_linear() -> List[np.ndarray]
    ```
    """

    __version__ = 3.0

    _save_items = ['_traj', '_traj_future',
                   '_traj_pred', '_traj_pred_linear',
                   '_frame_list', '_frame_list_future',
                   'real2grid', '__version__',
                   'linear_predict',
                   'neighbor_number',
                   'neighbor_traj',
                   'neighbor_traj_linear_pred',
                   'obs_length', 'total_frame']

    def __init__(self):
        self._traj = []
        self._traj_future = []

        self._traj_pred = None
        self._traj_pred_linear = None

        self._map = None
        self.real2grid = None

        self._frame_list = []
        self._frame_list_future = []

        self.linear_predict = False
        self.obs_length = 0
        self.total_frame = 0

        self.neighbor_number = 0
        self.neighbor_traj = []
        self.neighbor_traj_linear_pred = []

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
    def Map(self) -> np.ndarray:
        """
        context map
        """
        return self._map

    def set_map(self, Map: np.ndarray, paras: np.ndarray):
        self._map = Map
        self.real2grid = paras

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

    def get_neighbor_traj(self):
        return self.neighbor_traj

    def clear_all_neighbor_info(self):
        self.neighbor_traj = []
        self.neighbor_traj_linear_pred = []

    def get_pred_traj_neighbor_linear(self) -> list:
        return self.neighbor_traj_linear_pred
