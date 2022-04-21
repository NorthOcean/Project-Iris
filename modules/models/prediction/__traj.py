"""
@Author: Conghao Wong
@Date: 2021-04-12 11:18:35
@LastEditors: Conghao Wong
@LastEditTime: 2022-04-21 11:00:33
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import numpy as np


class EntireTrajectory():
    """
    Entire Trajectory
    -----------------
    Manage one agent's entire trajectory in datasets.

    Properties
    ----------
    ```python
    >>> self.agent_index
    >>> self.traj
    >>> self.video_neighbor_list
    >>> self.frame_list
    >>> self.start_frame
    >>> self.end_frame
    ```
    """

    def __init__(self, agent_index: int,
                 video_neighbor_list: list[int],
                 video_matrix: np.ndarray,
                 frame_list: list[int],
                 init_position: float):

        self._agent_index = agent_index
        self._traj = video_matrix[:, agent_index, :]
        self._video_neighbor_list = video_neighbor_list
        self._frame_list = frame_list

        base = self.traj.T[0]
        diff = base[:-1] - base[1:]

        appear = np.where(diff > init_position/2)[0]
        # disappear in next step
        disappear = np.where(diff < -init_position/2)[0]

        self._start_frame = appear[0] + 1 if len(appear) else 0
        self._end_frame = disappear[0] + 1 if len(disappear) else len(base)

    @property
    def agent_index(self):
        return self._agent_index

    @property
    def traj(self):
        return self._traj

    @property
    def video_neighbor_list(self):
        return self._video_neighbor_list

    @property
    def frame_list(self):
        return self._frame_list

    @property
    def start_frame(self):
        return self._start_frame

    @property
    def end_frame(self):
        return self._end_frame
