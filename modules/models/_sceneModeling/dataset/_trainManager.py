'''
Author: Conghao Wong
Date: 2021-04-08 19:56:34
LastEditors: Conghao Wong
LastEditTime: 2021-04-19 19:55:30
Description: file content
'''

import os
import shutil
from typing import Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from ... import base, prediction
from ..._helpmethods import dir_check
from ..agent._agent import Agent


class DatasetManager(base.DatasetManager):

    agent_type = Agent

    def __init__(self, args, dataset_name):
        super().__init__(args, dataset_name)

        pdm = prediction.PredictionDatasetManager()
        self.coe = [1.0, 1.0] if dataset_name in pdm.dataset_list['ethucy'] \
            else [2.0, 2.0]
        self._dataset_info = pdm(self.dataset_name)

        self.mean_scene = None
        self.semantic_label = None
        self.sample_step = 5.0

        self.agent_shapes = [np.array([10, 8])]
        self.agent_names = ['fine']
        self.agent_strides = [[1.0, 1.0]]

        self.agent_count = len(self.agent_shapes)
        self.dataset_path = './dataset_npz/{}/{}'.format(dataset_name, '{}')

    @property
    def video_capture(self) -> cv2.VideoCapture:
        return self._vc

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def frames(self) -> int:
        return self._frames

    @property
    def shape(self) -> List[int]:
        return self._shape

    @property
    def ref_image(self) -> np.ndarray:
        img_path = os.path.join(self.dataset_info.dataset_dir, 'reference.jpg')
        return cv2.imread(img_path)

    def load_data(self):
        self._vc, self._fps, self._frames, self._shape = self._read_video()
        self.calculate_mean_scene(self.sample_step)
        return self

    def sample_train_data(self) -> List[Agent]:
        train_agents = []
        for name, shape, stride in zip(self.agent_names,
                                       self.agent_shapes,
                                       self.agent_strides):
            train_agents.append(self.agent_type(
                dataset_name=self.dataset_name,
                grid_shape=shape,
                grid_stride=stride,
                grid_coe=self.coe,
                dataset_path=self.dataset_path.format(name),
                local_name=name))

        return self.init_agents(train_agents)

    def _read_video(self) -> Tuple[cv2.VideoCapture, int, int, list]:
        if os.path.exists(self.dataset_info.video_path):
            vc = cv2.VideoCapture(self.dataset_info.video_path)
            fps = vc.get(cv2.CAP_PROP_FPS)
            frames = vc.get(cv2.CAP_PROP_FRAME_COUNT)
            shape = [vc.get(cv2.CAP_PROP_FRAME_WIDTH),
                     vc.get(cv2.CAP_PROP_FRAME_HEIGHT)]
            return vc, fps, frames, shape

        else:
            return None, None, None, self.ref_image.shape[:2]

    def real2pixel(self, real_pos, pixel_limit: Tuple[int, int] = None):
        weights = self.dataset_info.weights

        if type(real_pos) == list:
            real_pos = np.array(real_pos)

        if len(real_pos.shape) == 3:
            real_pos = real_pos.reshape([-1, 2])

        if len(weights) == 4:
            results = np.column_stack([
                weights[2] * real_pos.T[1] + weights[3],
                weights[0] * real_pos.T[0] + weights[1],
            ]).astype(np.int32)
        else:
            H = weights[0]
            real = np.ones([real_pos.shape[0], 3])
            real[:, :2] = real_pos
            pixel = np.matmul(real, np.linalg.inv(H))
            pixel = pixel[:, :2].astype(np.int32)
            results = np.column_stack([
                weights[1] * pixel.T[0] + weights[2],
                weights[3] * pixel.T[1] + weights[4],
            ]).astype(np.int32)

        if pixel_limit:
            results[:, 0] = np.minimum(np.maximum(
                0.5, results[:, 0]), pixel_limit[0] - 0.5)
            results[:, 1] = np.minimum(np.maximum(
                0.5, results[:, 1]), pixel_limit[1] - 0.5)
        return results

    def pixel2grid(self, pixel, grid_shape: Tuple[int, int], pixel_shape: Tuple[int, int]):
        if type(pixel) == list:
            pixel = np.array(pixel)

        W = np.array([[grid_shape[0]/pixel_shape[0],
                       grid_shape[1]/pixel_shape[1]]])
        return (pixel * W).astype(np.int32)

    def _sample_video(self, sample_step: float = 5.0) -> np.ndarray:
        time = 0.0
        total_time = self.frames/self.fps
        sample_frames = []
        while time < total_time:
            self.video_capture.set(cv2.CAP_PROP_POS_MSEC, 1000*time)
            _, f = self.video_capture.read()
            sample_frames.append(f)
            time += sample_step

        return np.array(sample_frames)

    def calculate_mean_scene(self, sample_step=5.0):
        if (image := self.ref_image) is None:
            frames = self._sample_video(sample_step=sample_step)
            mean_frames = np.mean(frames, axis=0).astype(np.uint8)
            self.mean_scene = mean_frames
        else:
            self.mean_scene = image
        return self

    def init_agents(self, agents: List[Agent]) -> List[Agent]:
        dataset_status = [agent.dataset_status for agent in agents]
        
        if False in dataset_status:
            self.load_data()
            traj_dataset_manager = prediction.DatasetManager(
                self.args, self.dataset_name).load_data()

        for agent, status in zip(agents, dataset_status):
            if not status:
                agent.init_data(trajs=traj_dataset_manager.all_entire_trajectories,
                                video_manager=self)
                agent.make_dataset()

            agent.load_data()
        return agents


class DatasetsManager(prediction.DatasetsManager):

    agent_type = Agent
    datasetManager_type = DatasetManager

    def __init__(self, args, prepare_type='all', **kwargs):
        super().__init__(args, prepare_type=prepare_type)

    def prepare_train_files(self, dataset_managers: List[DatasetManager], mode='test') -> List[Agent]:
        """

        """
        all_agents = []
        count = 1

        for dm in dataset_managers:
            self.log('({}/{})  Prepare training images in dataset `{}`...'.format(
                count,
                len(dataset_managers),
                dm.dataset_name))

            agents = dm.sample_train_data()

            if mode == 'train':
                # TODO dataset process
                if balance := True:
                    self.logger.info('Start balancing train images...\n')
                    for agent in agents:
                        agent.balance()

            all_agents += agents
            count += 1
            
        return all_agents


def _remove_all_scene_datasets(base_path='./dataset_npz/'):
    """
    ***DANGEROUS***
    This function will remove all save scene datasets
    """
    dir_list = os.listdir(base_path)
    all_dataset_list = prediction.PredictionDatasetManager().datasets
    for item in dir_list:
        if item in all_dataset_list and os.path.exists(base_path + '{}/images'.format(item)):
            shutil.rmtree(base_path + '{}/images'.format(item))
