"""
@Author: Conghao Wong
@Date: 2021-07-06 16:10:10
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-06 21:45:03
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import Any, List, Tuple

import tensorflow as tf
from tensorflow import keras

from .. import models as M
from ._dataset import DatasetLoader
from ..satoshi._args import SatoshiArgs


class IMAGEEncoderLayer(keras.layers.Layer):
    def __init__(self, filters: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv1 = keras.layers.Conv2D(
            filters, [3, 3], activation=tf.nn.relu, padding='same')
        self.conv2 = keras.layers.Conv2D(
            filters, [3, 3], activation=tf.nn.relu, padding='same')
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

    def call(self, inputs, **kwargs):
        f_last = inputs[0]
        f_res = inputs[1]

        convT = self.convT(f_last)
        concat = self.concat([convT, f_res[:, :convT.shape[1], :convT.shape[2], :]])
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

        self.layer_list = []
        for i in range(self.num_encoder_layers):
            self.layer_list.append(IMAGEEncoderLayer(
                self.init_filters * (2 ** i)
            ))
        
        for i in range(self.num_decoder_layers):
            self.layer_list.append(IMAGEDecoderLayer(
                self.init_filters * (2 ** (self.num_decoder_layers - i - 1))
            ))

        self.dense = keras.layers.Dense(1)

    def call(self, inputs: tf.Tensor,
             training=None, mask=None):

        res_features = []
        f_last = inputs
        for i in range(self.num_encoder_layers):
            [f_last, f_res] = self.layer_list[i](f_last)
            res_features.append(f_res)    

        for i in range(self.num_decoder_layers):
            f_last = self.layer_list[self.num_encoder_layers + i]([
                f_last,
                res_features[-1-i],
            ])

        return self.dense(f_last)

    def pre_process(self, tensors: Tuple[tf.Tensor],
                    training=False,
                    **kwargs) -> Tuple[tf.Tensor]:
        """
        Pre-processing before inputting to the model
        """
        all_images = []

        scene = tf.stack([self.load_image(img) for img in tensors[0][:, 0]])
        trajs = tf.stack([self.load_image(img) for img in tensors[0][:, 1]])
        return tf.concat([scene, trajs], axis=-1)

    def load_image(self, image_file, 
                   reshape=True, 
                   reshape_size=(200, 200)) -> tf.Tensor:
        image = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image)
        if reshape:
            image = tf.image.resize(image, reshape_size)
        return image


class IMAGEStructure(M.prediction.Structure):
    def __init__(self, args, arg_type=SatoshiArgs):
        super().__init__(args, arg_type=arg_type)

        self.grid_shape = (200, 200)
        self.radius = 10
        self.range = [0.5, 1.0]
        self.root_ds_dir = './dataset_json'
        self.root_sample_dir = './samples'

        self.dl = DatasetLoader(grid_shape=self.grid_shape,
                                distribution_radius=self.radius,
                                distribution_value_range=self.range,
                                root_dataset_dir=self.root_ds_dir,
                                root_sample_dir=self.root_sample_dir)
    
    def create_model(self) -> Tuple[Any, keras.optimizers.Optimizer]:
        model = IMAGEModel(self.args)
        opt = keras.optimizers.Adam(self.args.lr)
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

    