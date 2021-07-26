"""
@Author: Conghao Wong
@Date: 2021-07-22 11:29:36
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-22 11:38:43
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import Dict, List, Tuple, Union

import cv2
import numpy as np

from .. import base
from .agent import PredictionAgent
from .args import PredictionArgs
from .traj import EntireTrajectory
from .utils import activation, calculate_cosine, calculate_length

MASK = cv2.imread('./mask_circle.png')[:, :, 0]/50
MASKS = {}


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
            agents:List[PredictionAgent],
            source=None,
            regulation=True
        ) -> np.ndarray

    # build socialMap (Attention: return `self`)
    >>> MapManager.build_social_map(
            self:MapManager,
            target_agent:PredictionAgent,
            traj_neighbors=[],
            source=None,
            regulation=True
        ) -> MapManager    
    ```
    """

    def __init__(self, args: PredictionArgs,
                 agents: List[PredictionAgent],
                 init_manager=None):
        """
        init map manager

        :param args: args to init this manager
        :param agents: a list of `PredictionAgent` object to init the map
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

    def _init_guidance_map(self, agents: List[PredictionAgent]):
        if issubclass(type(agents[0]), PredictionAgent):
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

    def build_guidance_map(self, agents: Union[List[PredictionAgent], np.ndarray],
                           source: np.ndarray = None,
                           regulation=True) -> np.ndarray:
        """
        Build guidance map

        :param agents: a list of agents or trajectories to calculate the map
        :param source: source map, default are zeros
        :param regulation: controls if scale the map into [0, 1]
        """

        if type(source) == type(None):
            source = self.void_map

        source = source.copy()
        if issubclass(type(agents[0]), PredictionAgent):
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

        return source

    def build_social_map(self, target_agent: PredictionAgent,
                         traj_neighbors: np.ndarray = [],
                         source: np.ndarray = None,
                         regulation=True,
                         max_neighbor=15) -> np.ndarray:
        """
        Build social map

        :param target_agent: target `PredictionAgent` object to calculate the map
        :param traj_neighbor: neighbors' predictions
        :param source: source map, default are zeros
        :param regulation: controls if scale the map into [0, 1]
        """

        if type(source) == type(None):
            source = self.void_map

        if not type(traj_neighbors) == np.ndarray:
            traj_neighbors = np.array(traj_neighbors)

        source = source.copy()

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

        return source

    def cut_map(self, maps: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Cut original maps into small local maps

        :param maps: maps, shape = (batch, a, b)
        :param centers: center positions (in grids), shape = (batch, 2)
        """
        batch, a, b = maps.shape[-3:]
        half_size = self.args.map_half_size

        centers = np.maximum(centers, half_size)
        centers = np.array([np.minimum(centers[:, 0], a - half_size),
                            np.minimum(centers[:, 1], b - half_size)]).T
        
        cuts = []
        for m, c in zip(maps, centers):
            cuts.append(m[c[0] - half_size: c[0] + half_size,
                          c[1] - half_size: c[1] + half_size])
                    
        return np.array(cuts)

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


def get_trajectories(agents: List[PredictionAgent],
                     return_movement=False,
                     return_destination=False,
                     destination_steps=3) -> list:
    """
    Get trajectories from input structures.

    :param agents: trajectory manager, support both `PredictionAgent` and `EntireTrajectory`
    :param return_movement: controls if return move flag
    :return trajs: a list of all trajectories from inputs
    """
    all_trajs = []
    movement = []
    for agent in agents:
        if issubclass(type(agent), PredictionAgent):
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
