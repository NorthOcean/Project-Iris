'''
Author: Conghao Wong
Date: 2021-04-13 10:36:03
LastEditors: Conghao Wong
LastEditTime: 2021-04-19 21:02:20
Description: file content
'''

import os
import re
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .. import base, prediction
from ..helpmethods import dir_check


class Agent(base.Agent):
    """
    Agent
    -----
    Agent manager for scene modeling and activity analysis.
    `Agent` indicates a video. One agent manager contains these
    items for a specific scene:
    - sub-images' size
    - sub-images' path
    - semantic labels

    TODO
    """

    def __init__(self, dataset_name: str,
                 dataset_path: str,
                 grid_shape: Tuple[int, int],
                 grid_stride: Tuple[float, float],
                 grid_coe: Tuple[float, float] = [1.0, 1.0],
                 local_name: str = 'GridDataset0'):

        super().__init__()
        self._dataset_name = dataset_name
        self._local_name = local_name
        self._grid_shape = [int(grid_shape[0] * grid_coe[0]),
                            int(grid_shape[1] * grid_coe[1])]
        self._grid_stride = grid_stride
        self._path = dataset_path

        self.train_index = None

        self._trajs = None
        self._mean_scene = None
        self._grid_length = None
        self.semantic_label = None
        self._file_path = []
        self._label = []
        self.pred = None
        self._hist = [None, None]

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def local_name(self) -> str:
        return self._local_name

    @property
    def grid_shape(self) -> Tuple[int, int]:
        return self._grid_shape

    @property
    def grid_stride(self) -> Tuple[float, float]:
        return self._grid_stride

    @property
    def pixel_shape(self) -> Tuple[int, int]:
        return (self._grid_shape[0] * self.grid_length[0],
                self._grid_shape[1] * self.grid_length[1])

    @property
    def grid_number(self) -> int:
        ii = len(np.arange(0, self.grid_shape[0], self.grid_stride[0]))
        jj = len(np.arange(0, self.grid_shape[1], self.grid_stride[1]))
        return ii * jj

    @property
    def grid_length(self) -> Tuple[int, int]:
        if self._grid_length is None:
            self._grid_length = [self.image_shape[0] / self.grid_shape[0],
                                 self.image_shape[1] / self.grid_shape[1]]
        return self._grid_length

    @property
    def scene_image(self) -> np.ndarray:
        return self._mean_scene

    @property
    def image_shape(self) -> Tuple[int, int]:
        return [self.scene_image.shape[1], self.scene_image.shape[0]]

    @property
    def trajs(self) -> List[prediction.EntireTrajectory]:
        return self._trajs

    @property
    def path(self) -> str:
        return self._path

    @property
    def file_path(self) -> List[str]:
        if self.train_index is None:
            return self._file_path
        else:
            return [self._file_path[index] for index in self.train_index]

    @property
    def label(self) -> List[float]:
        if self.train_index is None:
            return self._label
        else:
            return [self._label[index] for index in self.train_index]

    @property
    def histogram(self) -> List[np.ndarray]:
        return self._hist

    @property
    def dataset_status(self) -> bool:
        save_format = os.path.join(self.path, '{}.{}')
        label_path = save_format.format('labels', 'txt')
        check = False

        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()

            if len(lines) < self.grid_number:
                check = False

            for line in lines:
                file_name = line.split(' ')[0]
                if not os.path.exists(save_format.format(file_name, 'jpg')):
                    check = False
            check = True

        except:
            check = False

        if check:
            log_format = 'Scene dataset `{}/{}` checked!'
        else:
            log_format = 'Dataset `{}/{}` do not exist, start making...'

        self.log(log_format.format(self.dataset_name,
                                   self.local_name))

        return check

    def get_grid(self, index: Tuple[float, float]) -> np.ndarray:
        i, j = index
        si, sj = self.grid_stride
        return self.scene_image[
            int(j * sj * self.grid_length[1]):int((j+1) * sj * self.grid_length[1]),
            int(i * si * self.grid_length[0]):int((i+1) * si * self.grid_length[0])]

    def get_label(self, index: Tuple[float, float]) -> float:
        i, j = index
        i_min = np.floor(i).astype(int)
        i_max = np.floor(i+1).astype(int)
        j_min = np.floor(j).astype(int)
        j_max = np.floor(j+1).astype(int)

        single_square = 1.0
        l = np.zeros((self.grid_shape[0] + 1, self.grid_shape[1] + 1))
        l[:self.grid_shape[0], :self.grid_shape[1]] = self.semantic_label

        return (l[i_min, j_min] * (i_max - i) * (j_max - j) / single_square +
                l[i_min, j_max] * (i_max - i) * (j + 1 - j_max) / single_square +
                l[i_max, j_min] * (i + 1 - i_max) * (j_max - j) / single_square +
                l[i_max, j_max] * (i + 1 - i_max) * (j + 1 - j_max) / single_square)

    def pixel2grid(self, pixel):
        if type(pixel) == list:
            pixel = np.array(pixel)

        W = np.array([[self.grid_shape[0]/self.pixel_shape[0],
                       self.grid_shape[1]/self.pixel_shape[1]]])
        return (pixel * W).astype(np.int32)

    def grid2pixel(self, grid):
        if type(grid) == list:
            grid = np.expand_dims(np.array(grid), -1)

        W = np.array([[self.pixel_shape[0]/self.grid_shape[0]],
                      [self.pixel_shape[1]/self.grid_shape[1]]])
        return (grid * W).astype(np.int32)

    def summary(self):
        values, ranges = np.histogram(self.label, bins=5, range=(0, 1))
        length = len(self.label)
        prob_dict = {}
        for index, value in enumerate(values):
            item_name = '{:.2f}~{:.2f}'.format(ranges[index], ranges[index+1])
            percent = value / length
            prob_dict[item_name] = '{:.2f} '.format(
                percent) + self.log_bar(percent)

        self._hist = [values/length, ranges]
        self.print_parameters(
            'Score distribution of `{}/{}`'.format(
                self.dataset_name,
                self.local_name), **prob_dict)

    def _calculate_semantic_label(self,
                                  trajectories: List[prediction.EntireTrajectory],
                                  real2pixel) -> np.ndarray:
        # get trajectories
        trajs, movement_flag = prediction.get_trajectories(trajectories,
                                                           return_movement=0.2)
        true_index = np.where(np.array(movement_flag))[0]

        # real to pixel to grid
        trajs_pixel = real2pixel(np.array(trajs)[true_index],
                                 pixel_limit=self.image_shape)
        trajs_grid = self.pixel2grid(trajs_pixel)

        # make labels
        label = np.zeros(self.grid_shape)
        for traj in trajs_grid:
            label[traj[0], traj[1]] += 1
        label = label / np.max(label)

        return label

    def init_data(self,
                  trajs: List[prediction.EntireTrajectory],
                  video_manager):
        """
        Calculate dataset.
        Use this method when there is no existing dataset saves.
        """
        self._trajs = trajs
        self._mean_scene = video_manager.mean_scene
        self.semantic_label = self._calculate_semantic_label(
            trajectories=trajs,
            real2pixel=video_manager.real2pixel)

    def load_data(self):
        """
        Load dataset from existing files.
        This method can be used only when `agent.dataset_status` is `True`.
        """
        save_format = os.path.join(self.path, '{}.{}')
        label_path = save_format.format('labels', 'txt')
        image_path = save_format.format('-1', 'jpg')

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines[1:]:
            img_name, label = line[:-1].split(' ')
            self._file_path.append(save_format.format(img_name, 'jpg'))
            self._label.append(float(label))

        self._mean_scene = cv2.imread(image_path)
        self.summary()

    def make_dataset(self):
        """
        Save dataset files on local disk.
        """
        dir_check(self.path)
        save_format = os.path.join(self.path, '{}.{}')
        label_path = save_format.format('labels', 'txt')

        labels = []

        # save scene image
        cv2.imwrite(save_format.format(-1, 'jpg'), self.scene_image)
        labels.append('-1.jpg null\n')

        # save grids and labels
        for i in np.arange(0, self.grid_shape[0], self.grid_stride[0]):
            for j in np.arange(0, self.grid_shape[1], self.grid_stride[1]):
                file_name = '{},{}'.format(i, j)
                cv2.imwrite(save_format.format(file_name, 'jpg'),
                            self.get_grid((i, j)))
                labels.append(file_name
                              + ' '
                              + str(self.get_label((i, j)))
                              + '\n')

        with open(save_format.format('labels', 'txt'), 'w+') as f:
            f.writelines(labels)

    def balance(self, mode='avg'):
        num, thr = self.histogram
        label_dict = {}
        for index, label in enumerate(self.label):
            stage = np.where(label - thr[1:] <= 0)[0][0]

            if not stage in label_dict.keys():
                label_dict[stage] = []

            label_dict[stage].append(index)

        # sample images to average count
        if mode == 'avg':
            standard_count = int(np.mean([len(c)
                                          for c in label_dict.values()]))

        for key, value in label_dict.items():
            if len(value):
                label_dict[key] = np.random.choice(value, size=standard_count)
        self.train_index = np.stack(list(label_dict.values())).reshape([-1])
        self.summary()

    def save_results(self, save_folder: str):
        lines = []
        for img_path, predict in zip(self.file_path, self.pred):
            grid_pos = [float(item) for item in
                        re.findall('[0-9]+\.\d*[0-9]', img_path)]
            pixel_pos = self.grid2pixel(grid_pos)[:, 0]
            lines.append('{} {} {} {} {}\n'.format(pixel_pos[0], pixel_pos[1],
                                                   self.grid_length[0],
                                                   self.grid_length[1],
                                                   predict[0]))

        file_path = os.path.join(save_folder,
                                 '{}_{}_predict.txt'.format(self.dataset_name,
                                                            self.local_name))

        with open(file_path, 'w+') as f:
            f.writelines(lines)
