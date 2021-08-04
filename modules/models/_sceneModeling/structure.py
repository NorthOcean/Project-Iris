"""
@Author: Conghao Wong
@Date: 2020-12-24 11:09:35
@LastEditors: Conghao Wong
@LastEditTime: 2021-08-04 14:53:23
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import os
import re
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .. import base, prediction
from ..helpmethods import dir_check
from .agent import Agent
from .trainManager import DatasetsManager
from .utils import Data
from .vis import Visualization


class Model(base.Model):
    def __init__(self, Args, training_structure=None, *args, **kwargs):
        super().__init__(Args, training_structure=training_structure, *args, **kwargs)

        self.pre_process_list = ['REGULATION']

    def set_pre_process(self, *args):
        """
        Set pre-process operations to apply on images.
        Accept keywords:
        ```python
        regulation = ['reg']
        random_brightness = ['brig']
        random_flip = ['flip']
        random_contrast = ['cont']
        random_hue = ['hue']
        random_saturation = ['satu']
        random_quality = ['qua']
        ```
        """
        self.pre_process_list = []
        for item in args:
            if 'reg' in item:
                self.pre_process_list.append('REGULATION')

            elif 'brig' in item:
                self.pre_process_list.append('RANDOM_BRIGHTNESS')

            elif 'flip' in item:
                self.pre_process_list.append('RANDOM_FLIP')

            elif 'cont' in item:
                self.pre_process_list.append('RANDOM_CONTRAST')

            elif 'hue' in item:
                self.pre_process_list.append('RANDOM_HUE')

            elif 'satu' in item:
                self.pre_process_list.append('RANDOM_SATURATION')

            elif 'qua' in item:
                self.pre_process_list.append('RANDOM_QUALITY')

    def pre_process(self, model_inputs: List[tf.Tensor],
                    training=None,
                    **kwargs) -> List[tf.Tensor]:
        """
        Pre-processing before inputting to the model
        """
        all_images = []
        for img_path in model_inputs[0]:
            img = self._load_image(img_path)

            if training:
                img = Data.process(img, self.pre_process_list)

            all_images.append(img)

        imgs = tf.stack([self._load_image(img) for img in model_inputs[0]])
        return [imgs, model_inputs[1:]]

    def _load_image(self, image_file, reshape=True, reshape_size=(224, 224)) -> tf.Tensor:
        image = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image)
        if reshape:
            image = tf.image.resize(image, reshape_size)
        return image


class Structure(base.Structure):

    agent_type = Agent
    datasetsManager_type = DatasetsManager

    def __init__(self, Args: List[str], *args, **kwargs):
        super().__init__(Args, *args, **kwargs)

        self.args = prediction.PredictionArgs

    def run_test(self):
        """
        Run test of trajectory prediction on ETH-UCY or SDD dataset.
        """
        if self.args.test:
            if self.args.test_mode == 'all':
                with open('./test_log.txt', 'a') as f:
                    f.write('-'*40 + '\n')

                for dataset in prediction.PredictionDatasetInfo().sdd_test_sets if self.args.dataset == 'sdd' else prediction.PredictionDatasetInfo().ethucy_testsets:
                    agents = self.datasetsManager_type.load_dataset_files(
                        self.args, dataset)
                    self.test(agents=agents, dataset_name=dataset)

                with open('./test_log.txt', 'a') as f:
                    f.write('-'*40 + '\n')

            elif self.args.test_mode == 'one':
                agents = self.datasetsManager_type.load_dataset_files(
                    self.args, self.args.test_set)
                self.test(agents=agents, dataset_name=self.args.test_set)

    def get_inputs_from_agents(self, input_agents: List[Agent]) -> Tuple[tf.Tensor, tf.Tensor]:
        model_inputs = []
        labels = []

        for agent in input_agents:
            model_inputs += agent.file_path
            labels += agent.label

        return (tf.cast(model_inputs, tf.string),
                tf.cast(labels, tf.float32))

    def load_forward_dataset(self, model_inputs: List[str], **kwargs) -> tf.data.Dataset:
        """
        Load forward dataset.

        :return dataset_train: test dataset, type = `tf.data.Dataset`
        """
        model_inputs_all = []

        for dataset_name in model_inputs:
            model_inputs_all += model_inputs[dataset_name]

        model_inputs = tf.cast(model_inputs_all, tf.float32)
        test_dataset = tf.data.Dataset.from_tensor_slices(model_inputs)
        return test_dataset

    def loss(self, outputs, labels, loss_name_list: List[str] = ['CrossEntropy'], **kwargs) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Train loss, using cross-entropy loss (with class weights) by default.

        :param outputs: model's outputs (A list of logits, shape=[batch, 2])
        :param labels: groundtruth labels
        :param loss_name_list: a list of name of used loss functions

        :return loss: sum of all single loss functions
        :return loss_dict: a dict of all losses
        """
        labels_onehot = tf.one_hot(
            tf.cast(labels > 0, tf.int32), depth=2)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels_onehot, outputs[0])

        weights = (
            1.0 * tf.cast(labels == 0, tf.float32) +
            1.0 * tf.cast(labels > 0, tf.float32))
        loss = tf.reduce_mean(cross_entropy * weights)
        loss_dict = dict(zip(loss_name_list, [loss]))
        return loss, loss_dict

    def metrics(self, outputs, labels, loss_name_list=['acc', 'acc+', 'acc-'], **kwargs) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Metrics
        """
        output_classes = tf.argmax(outputs[0], axis=-1, output_type=tf.int32)
        right = tf.reduce_sum(tf.cast(
            output_classes == tf.cast(labels > 0, tf.int32),
            tf.int32))

        acc_pos = tf.gather(output_classes, tf.where(labels > 0))
        acc_pos = tf.reduce_sum(acc_pos)/len(acc_pos)

        acc_neg = tf.abs(1 - tf.gather(output_classes,
                                       tf.where(labels == 0)))
        acc_neg = tf.reduce_sum(acc_neg)/len(acc_neg)

        loss = right/len(output_classes)
        loss_dict = dict(zip(loss_name_list, [loss, acc_pos, acc_neg]))
        return 1.0-loss, loss_dict

    def print_dataset_info(self):
        self.print_parameters(title='dataset options',
                            rotate=self.args.rotate,
                            add_noise=self.args.add_noise)

    def write_test_results(self,
                           model_outputs: List[tf.Tensor],
                           agents: Dict[str, List[Agent]],
                           **kwargs):

        testset_name = kwargs['dataset_name']

        if self.args.draw_results:
            sv = Visualization()
            save_base_path = dir_check(
                self.args.log_dir) if self.args.load == 'null' else self.args.load
            save_base_path = dir_check(
                os.path.join(save_base_path, 'VisualScenes'))

            self.log('Start saving images at {}'.format(save_base_path))

            img_count = 0
            for index, agent in self.log_timebar(agents, 'Saving images...'):

                # write results
                output = model_outputs[0][img_count:img_count +
                                          agent.grid_number]
                img_count += agent.grid_number
                agent.pred = output
                agent.save_results(save_base_path)

                # draw as one image
                for draw_gt in [False, True]:
                    sv.draw(agent=agent,
                            save_base_path=save_base_path,
                            file_name='auto',
                            draw_groundtruth=draw_gt,
                            regulation=False,
                            draw_heatmap=True)
