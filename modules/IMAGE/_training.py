"""
@Author: Conghao Wong
@Date: 2021-07-06 16:10:10
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-07 21:06:16
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import os
from typing import Any, List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .. import models as M
from ..satoshi._args import SatoshiArgs
from ._dataset import DatasetLoader


class IMAGEEncoderLayer(keras.layers.Layer):
    def __init__(self, filters: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv1 = keras.layers.Conv2D(filters, [3, 3],
                                         activation=tf.nn.relu,
                                         padding='same')
        self.conv2 = keras.layers.Conv2D(filters, [3, 3],
                                         activation=tf.nn.relu,
                                         padding='same')
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
        self.maxpooling = keras.layers.MaxPooling2D((2, 2))

    def call(self, inputs, **kwargs):
        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1)
        conv2 = self.conv2(bn1)
        bn2 = self.bn2(conv2)
        return self.maxpooling(bn2), bn2


class IMAGEDecoderLayer(keras.layers.Layer):
    def __init__(self, filters: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.convT = keras.layers.Conv2DTranspose(filters, [2, 2],
                                                  strides=[2, 2],
                                                  padding='valid')
        self.concat = keras.layers.Concatenate()
        self.conv1 = keras.layers.Conv2D(filters, [3, 3], padding='same')
        self.conv2 = keras.layers.Conv2D(filters, [3, 3], padding='same')

    def call(self, f_last, f_res, **kwargs):
        convT = self.convT(f_last)
        concat = self.concat([convT, f_res])
        conv1 = self.conv1(concat)
        conv2 = self.conv2(conv1)
        return conv2


class IMAGEModel(M.prediction.Model):
    def __init__(self, Args, training_structure=None, *args, **kwargs):
        super().__init__(Args, training_structure=training_structure,
                         *args, **kwargs)

        self.num_encoder_layers = 3
        self.num_decoder_layers = 3
        self.init_filters = 16

        if self.args.draw_results:
            self.image_dir = M.helpMethods.dir_check(
                os.path.join(self.args.log_dir,
                             'predictions'))

        self.layer_list = []
        for i in range(self.num_encoder_layers):
            self.layer_list.append(IMAGEEncoderLayer(
                self.init_filters * (2 ** i)
            ))

        for i in range(self.num_encoder_layers - 1):
            self.layer_list.append(IMAGEDecoderLayer(
                self.init_filters * (2 ** (self.num_encoder_layers - i - 1))
            ))

        self.conv = keras.layers.Conv2D(self.args.pred_frames, (128, 127))

    def call(self, inputs: tf.Tensor,
             training=None, mask=None):

        scene_img, traj_img, traj = inputs[:3]

        res_features = []
        f_last = tf.concat([scene_img, traj_img], axis=-1)
        for i in range(self.num_encoder_layers):
            [f_last, f_res] = self.layer_list[i](f_last)
            res_features.append(f_res)

        for i in range(self.num_encoder_layers - 1):
            f_last = self.layer_list[self.num_encoder_layers + i](
                f_last,
                res_features[-1-i])

        return tf.transpose(self.conv(f_last)[:, 0, :, :], [0, 2, 1])

    def pre_process(self, tensors: Tuple[tf.Tensor],
                    training=None,
                    **kwargs) -> Tuple[tf.Tensor]:

        [img_s_paths, img_t_paths, trajs] = tensors[:3]
        scene = tf.stack([load_image(img) for img in img_s_paths])
        traj_imgs = tf.stack([load_image(img)[:, :, :2]
                             for img in img_t_paths])

        self.file_names = img_t_paths
        return (scene, traj_imgs, trajs)


class IMAGEStructure(M.prediction.Structure):
    def __init__(self, args, arg_type=SatoshiArgs):
        super().__init__(args, arg_type=arg_type)

        self.grid_shape = (200, 200)
        self.radius = 10
        self.range = [0.5, 1.0]
        self.root_ds_dir = './dataset_json'
        self.root_sample_dir = './samples'

        self.set_loss('ade')
        self.set_loss_weights(1.0)

        self.set_metrics('ade', 'fde')
        self.set_metrics_weights(0.7, 0.3)

        self.dl = DatasetLoader(grid_shape=self.grid_shape,
                                distribution_radius=self.radius,
                                distribution_value_range=self.range,
                                root_dataset_dir=self.root_ds_dir,
                                root_sample_dir=self.root_sample_dir)

    def create_model(self) -> Tuple[Any, keras.optimizers.Optimizer]:
        model = IMAGEModel(self.args)
        opt = keras.optimizers.Adam(self.args.lr)

        model.build([[None, 256, 256, 3],
                     [None, 256, 256, 2],
                     [None, self.args.obs_frames, 2]])
        model.summary()
        return model, opt

    def run_test(self):
        """
        Run test of trajectory prediction on ETH-UCY or SDD dataset.
        """
        if self.args.test:
            if self.args.test_mode == 'all':
                with open('./test_log.txt', 'a') as f:
                    f.write('-'*40 + '\n')
                    f.write(
                        '- K = {}, sigma = {} -\n'.format(self.args.K, self.args.sigma))
                for dataset in M.prediction.PredictionDatasetManager().sdd_test_sets if self.args.dataset == 'sdd' else M.prediction.PredictionDatasetManager().ethucy_testsets:
                    self.test(datasets=[dataset], dataset_name=dataset)

            elif self.args.test_mode == 'mix':
                agents = []
                dataset = ''
                for dataset_c in M.prediction.PredictionDatasetManager().sdd_test_sets if self.args.dataset == 'sdd' else M.prediction.PredictionDatasetManager().ethucy_testsets:
                    agents += [dataset_c]
                    dataset += '{}; '.format(dataset_c)

                self.test(datasets=agents, dataset_name='mix: '+dataset)

            elif self.args.test_mode == 'one':
                datasets = [self.args.test_set]
                self.test(datasets=datasets, dataset_name=self.args.test_set)

    def load_dataset(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Load training and val dataset.

        :return dataset_train: train dataset, type = `tf.data.Dataset`
        :return dataset_val: val dataset, type = `tf.data.Dataset`
        """
        train_list, test_list = self.dl.get_dataset_list(self.args)
        dataset_train = self.dl.restore_dataset(train_list, self.args.obs_frames,
                                                self.args.pred_frames,
                                                self.args.step)
        dataset_test = self.dl.restore_dataset(test_list, self.args.obs_frames,
                                               self.args.pred_frames,
                                               self.args.step)

        dataset_train = dataset_train.shuffle(len(dataset_train),
                                              reshuffle_each_iteration=True)

        return dataset_train, dataset_test

    def load_test_dataset(self, **kwargs) -> tf.data.Dataset:
        """
        Load test dataset.

        :return dataset_train: test dataset, type = `tf.data.Dataset`
        """
        datasets = kwargs['datasets']
        return self.dl.restore_dataset(datasets, self.args.obs_frames,
                                       self.args.pred_frames,
                                       self.args.step)

    def write_test_results(self,
                           model_outputs: List[tf.Tensor],
                           model_inputs: List[List[Any]],
                           labels: List[tf.Tensor],
                           datasets: List[str],
                           dataset_name: str,
                           *args, **kwargs):

        if (not self.args.draw_results) or (dataset_name.startswith('mix')):
            return

        scene_image = cv2.imread(model_inputs[0][1].decode())
        tv = M.prediction.TrajVisualization(dataset=None)
        save_base_path = M.helpMethods.dir_check(self.args.log_dir) \
            if self.args.load == 'null' \
            else self.args.load

        save_format = os.path.join((base_path := M.helpMethods.dir_check(os.path.join(
            save_base_path, 'VisualTrajs_{}'.format(dataset_name)))), '{}.jpg')

        for predictions, inputs in self.log_timebar(
                inputs=zip(model_outputs[0].numpy(), model_inputs),
                text='Saving...',
                return_enumerate=False):

            file_name = inputs[0].decode().split('/')[-1]

            obs = inputs[2].astype(np.int)[:, ::-1]
            gt = inputs[3].astype(np.int)[:, ::-1]
            pred = predictions.astype(np.int)[:, ::-1]

            scene = tv._visualization(scene_image, obs, gt, pred,
                                      self.args.draw_distribution)

            cv2.imwrite(save_format.format(file_name), scene)

        self.logger.info(
            'Prediction result images are saved at {}'.format(base_path))


def load_image(image_file: str,
               reshape=True,
               reshape_size=(256, 256)) -> tf.Tensor:
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    if reshape:
        image = tf.image.resize(image, reshape_size)
    return image


def save_image(image_file: tf.Tensor, path: str):
    if path.endswith('jpg'):
        image = tf.image.encode_jpeg(image_file)
    elif path.endswith('png'):
        image = tf.image.encode_png(image_file)
    else:
        raise NotImplementedError

    tf.io.write_file(path, image)
