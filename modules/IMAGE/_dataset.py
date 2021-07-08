"""
@Author: Conghao Wong
@Date: 2021-07-05 16:35:51
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-07 19:29:43
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import re
from typing import Dict, List, Tuple, Union
import numpy as np
import cv2
import os

import tensorflow as tf
from .. import models as M


class DatasetLoader(M.datasetProcess.SampleLoader):
    def __init__(self, grid_shape: Tuple[int, int],
                 distribution_radius: int,
                 distribution_value_range: Tuple[float, float],
                 root_dataset_dir='./dataset_json',
                 root_sample_dir='./samples'):

        super().__init__(root_dataset_dir=root_dataset_dir,
                         root_sample_dir=root_sample_dir)

        self.dataset_info = M.prediction.PredictionDatasetManager()
        self.save_format = (
            '{}_{}_'            # input/label + sample_index, like `x_100_`, `y_20_`
            + '{}_to_{}'        # length, like `8_to_20`
            + '_f{}'            # frame index in the video, like `_f101`
            + '.{}')            # file format, like `.jpg`

        self.grid_shape = grid_shape
        self.radius = distribution_radius
        self.range = distribution_value_range

        d_file = cv2.imread('./mask_circle.png')[:, :, 0]/50
        self.d_file = cv2.resize(d_file,
                                 (self.radius*2 + 1, self.radius*2 + 1))

    def check_status(self, dataset: str,
                     obs_length: int,
                     pred_length: int,
                     save_path: str = None) -> bool:
        """
        Check if dataset files (images) exist

        :param dataset: name of dataset
        :param obs_length: length of observed frames
        :param pred_length: length of predictions
        :param save_path: dataset folder
        """

        checked = False

        if save_path is None:
            save_path = os.path.join(self.sample_path, dataset)

        try:
            file_list = os.listdir(save_path)

            if not self.get_filename(
                    None, 'scene', -1, obs_length, pred_length, -1, 'jpg') in file_list:

                raise FileNotFoundError

            train_id = set(np.loadtxt(self.get_filename(
                save_path, 'id', -1, obs_length, pred_length, -1, 'txt'))[:, 0].astype(np.int))

            x_file_list = set([int(p[2]) for f in file_list if (p := re.match(self.get_filename(
                None, '(x', ')([0-9]+)(', obs_length, pred_length, '[0-9]+', 'jpg)'), f))])
            y_file_list = set([int(p[2]) for f in file_list if (p := re.match(self.get_filename(
                None, '(y', ')([0-9]+)(', obs_length, pred_length, '[0-9]+', 'jpg)'), f))])

            if (train_id.issubset(x_file_list)
                    and train_id.issubset(y_file_list)):

                checked = True

        except:
            pass

        return checked

    def check_trajs(self, dataset: str,
                    obs_length: int,
                    pred_length: int,
                    save_path: str = None) -> bool:

        checked = False

        if save_path is None:
            save_path = os.path.join(self.sample_path, dataset)

        try:
            image_index = np.loadtxt(self.get_filename(save_path, 'id', -1,
                                                       obs_length, pred_length,
                                                       -1, 'txt'))

            train_id = set(image_index[:, 0].astype(np.int))
            train_frames = set(image_index[:, 1].astype(np.int))

            traj_data = np.load(self.get_filename(save_path, 'traj', -1,
                                                  obs_length, pred_length,
                                                  -1, 'npy'),
                                allow_pickle=True).item()

            traj_id_list = set([int(p[2]) for f in traj_data.keys()
                                if (p := re.match(
                                    self.get_filename(None, '(traj', ')([0-9]+)(',
                                                      obs_length, pred_length,
                                                      '[0-9]+)(', 'total)'), f))])

            frame_id_list = set([int(p[4]) for f in traj_data.keys()
                                 if (p := re.match(
                                     self.get_filename(None, '(traj', ')([0-9]+)(',
                                                       obs_length, pred_length,
                                                       ')([0-9]+)(', 'total)'), f))])

            if (train_id.issubset(traj_id_list)
                    and train_frames.issubset(frame_id_list)):

                checked = True

        except:
            pass

        return checked

    def restore_dataset(self, dataset: Union[str, List[str]],
                        obs_length: int,
                        pred_length: int,
                        sample_step: int) -> tf.data.Dataset:
        """
        Restore dataset images (if exist) and make into `tf.data.Dataset`

        :param dataset: dataset name (or a list of names)
        :param obs_length: length of observed frames
        :param pred_length: length of predictions
        :param sample_step: steps between each two samples
        """
        if not '__len__' in dataset.__dir__():
            dataset = [dataset]

        x_list = [[], [], []]
        y_list = [[], [], []]

        for ds in dataset:
            image_state = self.check_status(ds, obs_length, pred_length)
            traj_state = self.check_trajs(ds, obs_length, pred_length)

            if (not image_state) or (not traj_state):
                dataset_info = self.dataset_info(ds)
                ds_step, ds_rate = dataset_info.paras
                if (ds_rate * 0.4) % ds_step == 0:
                    frame_rate = ds_rate * 0.4
                else:
                    frame_rate = ds_step

                samples_file = self.restore_samples(ds,
                                                    sample_step * frame_rate,
                                                    obs_length,
                                                    pred_length,
                                                    frame_rate,
                                                    strategy='sampleFromPerson')

                if not image_state:
                    self.make_dataset_images(samples_file,
                                             H='auto',
                                             obs_length=obs_length,
                                             pred_length=pred_length,
                                             dataset_info=dataset_info)

                if not traj_state:
                    self.make_dataset_trajs(samples_file,
                                            H='auto',
                                            obs_length=obs_length,
                                            pred_length=pred_length,
                                            dataset_info=dataset_info)

            x, y = self.load_dataset(ds, obs_length, pred_length)

            for index in range(len(x)):
                x_list[index] += x[index]
            for index in range(len(y)):
                y_list[index] += y[index]

        return tf.data.Dataset.from_tensor_slices((tuple(x_list) + (y_list[-1],)))

    def make_dataset_images(self, samples: List[Dict[int, List[float]]],
                            H: Union[np.ndarray, str],
                            obs_length: int,
                            pred_length: int,
                            save_path: str = None,
                            scene_image: str = None,
                            dataset_info: M.base.Dataset = None,
                            *args, **kwargs):
        """
        Make dataset images

        :param samples: a list of dictionary, whose keys are frame index, and values are trajectories
        :param H: a 2*2 matrix that transfer real positions to grid positions. (Y_g = Y_r H). Set it to `auto` to obtain it automatically
        :param obs_length: length of observations
        :param pred_length: length of predictions
        :param save_path: image save path. Set to `None` to save in default path
        :param scene_image: path for the 2D scene image (or segmentation map). Set to `None` to load default scene RGB image
        :param dataset_info: type = `base.Dataset`. Must give values when setting `save_path` or `scene_image` to `None`
        """
        if scene_image is None:
            if dataset_info is None:
                raise ValueError

            scene_image = os.path.join(
                dataset_info.dataset_dir, 'reference.jpg')

        scene_image = cv2.imread(scene_image)
        original_shape = scene_image.shape[:-1]

        scene_image = cv2.resize(scene_image, self.grid_shape)
        neighbor_map = np.zeros(scene_image.shape[:-1], np.float)
        target_map = np.zeros(scene_image.shape[:-1], np.float)

        if save_path is None:
            if dataset_info is None:
                raise ValueError

            save_path = os.path.join(self.sample_path, dataset_info.dataset)

        M.helpMethods.dir_check(save_path)

        cv2.imwrite(self.get_filename(
            save_path, 'scene', -1,
            obs_length, pred_length, -1, 'jpg'), scene_image)

        if H == 'auto':
            if dataset_info is None:
                raise ValueError

            order = dataset_info.order
            X = self.grid_shape[order[0]] / original_shape[1]
            Y = self.grid_shape[order[1]] / original_shape[0]
            H = np.array([[0, X], [Y, 0]])

            if dataset_info.dataset in self.dataset_info.dataset_list['ethucy']:
                H = [H, dataset_info]

        train_id = []
        for sample_index, sample in self.log_timebar(samples):
            frame_number = len(sample.keys())
            amps = np.linspace(self.range[0], self.range[1], frame_number)

            n_map = neighbor_map.copy()
            t_map = target_map.copy()

            for frame_index, (frame_data, amp) in enumerate(zip(sample.values(), amps)):
                frame_data = np.array(frame_data)
                t_map = self.add_to_grid(
                    t_map, real2grid(frame_data[:1, 1:], H), amp)

                if len(frame_data) == 1:
                    continue

                n_map = self.add_to_grid(
                    n_map, real2grid(frame_data[1:, 1:], H), amp)

                if frame_index == obs_length - 1:
                    n_map_x = n_map.copy()
                    t_map_x = t_map.copy()

            if n_map.max() * t_map.max() * n_map_x.max() * t_map_x.max() == 0:
                continue

            frame_id = list(sample.keys())[obs_length]
            train_id.append([sample_index, frame_id])
            n_map /= np.max(n_map)
            t_map /= np.max(t_map)
            n_map_x /= np.max(n_map_x)
            t_map_x /= np.max(t_map_x)

            cv2.imwrite(self.get_filename(
                save_path, 'x', sample_index,
                obs_length, pred_length, frame_id, 'jpg'),
                np.concatenate([255*n_map_x[:, :, np.newaxis],
                                255*t_map_x[:, :, np.newaxis],
                                np.zeros_like(scene_image[:, :, :1])], axis=-1))

            cv2.imwrite(self.get_filename(
                save_path, 'y', sample_index,
                obs_length, pred_length, frame_id, 'jpg'),
                np.concatenate([255*n_map[:, :, np.newaxis],
                                255*t_map[:, :, np.newaxis],
                                np.zeros_like(scene_image[:, :, :1])], axis=-1))

        np.savetxt(self.get_filename(save_path, 'id', -1, obs_length,
                   pred_length, -1, 'txt'), np.array(train_id))
        self.logger.info('Dataset samples saved at `{}`.'.format(save_path))

    def make_dataset_trajs(self, samples: List[Dict[int, List[float]]],
                           H: Union[np.ndarray, str],
                           obs_length: int, pred_length: int,
                           save_path: str = None,
                           scene_image: str = None,
                           dataset_info: M.base.Dataset = None):
        """
        Make dataset trajectories
        """
        if scene_image is None:
            if dataset_info is None:
                raise ValueError

            scene_image = os.path.join(
                dataset_info.dataset_dir, 'reference.jpg')

        scene_image = cv2.imread(scene_image)
        original_shape = scene_image.shape[:-1]

        if save_path is None:
            if dataset_info is None:
                raise ValueError

            save_path = os.path.join(self.sample_path, dataset_info.dataset)

        M.helpMethods.dir_check(save_path)

        if H == 'auto':
            if dataset_info is None:
                raise ValueError

            order = dataset_info.order
            X = self.grid_shape[order[0]] / original_shape[1]
            Y = self.grid_shape[order[1]] / original_shape[0]
            H = np.array([[0, X], [Y, 0]])

            if dataset_info.dataset in self.dataset_info.dataset_list['ethucy']:
                H = [H, dataset_info]

        all_trajs = {}
        for sample_index, sample in self.log_timebar(samples):
            frame_id = list(sample.keys())[obs_length]
            key = self.get_filename(None, 'traj', sample_index,
                                    obs_length, pred_length, frame_id, 'total')
            real_traj = np.array([sample[k][0][1:] for k in sample.keys()])
            all_trajs[key] = real2grid(real_traj, H).astype(np.float32)

        np.save(self.get_filename(save_path, 'traj', -1,
                                  obs_length, pred_length,
                                  -1, 'npy'),
                all_trajs)

        self.logger.info('Trajectores saved at `{}`.'.format(save_path))

    def load_dataset(self, dataset: str,
                     obs_length: int,
                     pred_length: int,
                     save_path: str = None) -> Tuple[list, list]:

        if save_path is None:
            save_path = os.path.join(self.sample_path, dataset)

        train_id = np.loadtxt(self.get_filename(
            save_path, 'id', -1, obs_length, pred_length, -1, 'txt')).astype(np.int)
        traj_data = np.load(self.get_filename(save_path, 'traj', -1,
                                              obs_length, pred_length,
                                              -1, 'npy'),
                            allow_pickle=True).item()

        inputs = [[], [], []]
        labels = [[], [], []]
        for s_id, f_id in train_id:
            inputs[0].append(self.get_filename(save_path, 'x', s_id,
                                               obs_length, pred_length,
                                               f_id, 'jpg'))
            inputs[1].append(self.get_filename(save_path, 'scene', -1,
                                               obs_length, pred_length,
                                               -1, 'jpg'))
            inputs[2].append(traj_data[self.get_filename(None, 'traj', s_id,
                                                         obs_length, pred_length,
                                                         f_id, 'total')][:obs_length].astype(np.float32))

            labels[0].append(self.get_filename(save_path, 'y', s_id,
                                               obs_length, pred_length,
                                               f_id, 'jpg'))
            labels[1].append(self.get_filename(save_path, 'scene', -1,
                                               obs_length, pred_length,
                                               -1, 'jpg'))
            labels[2].append(traj_data[self.get_filename(None, 'traj', s_id,
                                                         obs_length, pred_length,
                                                         f_id, 'total')][obs_length:].astype(np.float32))

        return inputs, labels

    def get_filename(self, save_path: str,
                     filetype: str,
                     index: int,
                     obs_length: int, pred_length: int,
                     frame_id: int,
                     end: str) -> str:

        if not type(index) is str:
            index = str(int(index))

        if not type(frame_id) is str:
            frame_id = str(int(frame_id))

        if not type(obs_length) is str:
            obs_length = str(int(obs_length))

        if not type(pred_length) is str:
            pred_length = str(int(pred_length))

        local_path = self.save_format.format(filetype,
                                             index,
                                             obs_length,
                                             pred_length,
                                             frame_id,
                                             end)

        if save_path:
            return os.path.join(save_path, local_path)
        else:
            return local_path

    def add_to_grid(self, grid: np.ndarray,
                    grid_pos: np.ndarray,
                    amplitude: Union[float, List[float]] = 1):

        if len(grid_pos.shape) == 1:
            grid_pos = grid_pos[np.newaxis, :]

        if amplitude is not list:
            amplitude = [amplitude for _ in range(grid_pos.shape[0])]

        for pos, amp in zip(grid_pos, amplitude):
            [x, y] = pos
            if (x - self.radius < 0 or
                y - self.radius < 0 or
                x + self.radius + 1 > self.grid_shape[0] or
                    y + self.radius + 1 > self.grid_shape[1]):
                continue

            grid[x-self.radius: x+self.radius+1,
                 y-self.radius: y+self.radius+1] += self.d_file * amp

        return grid

    def transfer_dataset(self, dataset: str):
        dm = self.dataset_info(dataset)

        if dataset in self.dataset_info.dataset_list['sdd']:
            label_file = 'annotations.txt'
        elif dataset in self.dataset_info.dataset_list['ethucy']:
            label_file = 'true_pos_.csv'
        else:
            raise

        self._transfer_dataset(os.path.join(dm.dataset_dir, label_file),
                               os.path.join(self.ds_path, dataset+'.json'))

    def get_dataset_list(self, args):
        dataset_list = self.dataset_info.dataset_list[args.dataset]

        if args.dataset == 'ethucy':
            train_list = [i for i in dataset_list if not i == args.test_set]
            val_list = [args.test_set]

        elif args.dataset == 'sdd':
            train_list = [i for i in dataset_list if not i in
                          self.dataset_info.sdd_test_sets + self.dataset_info.sdd_val_sets]
            val_list = self.dataset_info.sdd_test_sets

        return train_list, val_list


def real2grid(real: np.ndarray, H: np.ndarray):
    if (type(H) is list) and (type(H[1]) is M.base.Dataset):
        real = real2pixel(real, H[1].weights)[:, 0, :]
        H = H[0]

    if len(real.shape) == 1:
        real = real[np.newaxis, :]  # shape = (batch, 2)

    return np.matmul(real, H).astype(np.int)   # output_shape = (batch, 2)


def real2pixel(real_pos, weights):
    """
    Transfer coordinates from real scale to pixels.

    :param real_pos: coordinates, shape = (n, 2) or (k, n, 2)
    :return pixel_pos: coordinates in pixels
    """
    if type(real_pos) == list:
        real_pos = np.array(real_pos)

    if len(real_pos.shape) == 2:
        real_pos = real_pos[np.newaxis, :, :]

    all_results = []
    for step in range(real_pos.shape[1]):
        r = real_pos[:, step, :]
        if len(weights) == 4:
            result = np.column_stack([
                weights[2] * r.T[0] + weights[3],
                weights[0] * r.T[1] + weights[1],
            ]).astype(np.int32)
        else:
            H = weights[0]
            real = np.ones([r.shape[0], 3])
            real[:, :2] = r
            pixel = np.matmul(real, np.linalg.inv(H))
            pixel = pixel[:, :2].astype(np.int32)
            result = np.column_stack([
                weights[1] * pixel.T[1] + weights[2],
                weights[3] * pixel.T[0] + weights[4],
            ]).astype(np.int32)

        all_results.append(result)

    return np.array(all_results)
