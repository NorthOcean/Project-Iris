'''
Author: Conghao Wong
Date: 2021-04-09 09:50:19
LastEditors: Conghao Wong
LastEditTime: 2021-04-14 10:35:46
Description: file content
'''

import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from ... import base, prediction
from ..agent._agent import Agent


class Visualization(base.Visualization):
    def __init__(self):
        super().__init__()

        # color bar in RGB format
        # 0 -> 127 -> 255
        # rgb(0, 0, 178) -> rgb(252, 0, 0) -> rgb(255, 255, 10)
        self.color_bar = np.column_stack([
            np.interp(np.arange(256),
                      np.array([0, 127, 255]),
                      np.array([0, 252, 255])),
            np.interp(np.arange(256),
                      np.array([0, 127, 255]),
                      np.array([0, 0, 255])),
            np.interp(np.arange(256),
                      np.array([0, 127, 255]),
                      np.array([178, 0, 10])),
        ])

    def grid2pixel(self, grid_pos):
        x = grid_pos[0] * self.shape[0]/self.grid_shape[0]
        y = grid_pos[1] * self.shape[1]/self.grid_shape[1]
        return np.array([x, y])

    def get_heatmap_color(self, score: float, alpha=0.5):
        score_int = int(min(max(score, 0.0), 1.0) * 255)
        color_rgb = self.color_bar[score_int]/255.0
        color_rgba = np.zeros(4)
        color_rgba[:3] = color_rgb
        color_rgba[-1] = alpha
        return color_rgba

    def regulation(self, logits: List[tf.Tensor]) -> tf.Tensor:
        """
        Regulation logits into range of [0, 1]
        """
        logits = tf.stack(logits)
        logits = tf.minimum(tf.maximum(logits, 0.0), 1.0)
        max_value = tf.reduce_max(logits, axis=0)
        min_value = tf.reduce_min(logits, axis=0)
        return (logits - min_value)/(max_value - min_value)

    def draw(self, agent: Agent,
             save_base_path,
             file_name='auto',
             draw_groundtruth=False,
             regulation=True,
             draw_heatmap=True):
        """
        Draw scene modeling results.

        :param dataset: name of the dataset
        :param logits: model outputs, a list of `tf.Tensor`
        :param save_base_path: folder to save result images
        :param file_name: string, set save file's name
        :param regulation: controls if regular logits into [0, 1]
        :param draw_heatmap: controls if draw as heatmap on original scene
        """
        dataset = agent.dataset_name
        mean_image = agent.scene_image
        self.shape = agent.image_shape
        self.grid_shape = agent.grid_shape

        pos_index = [[i, j]
                     for i in np.arange(0, agent.grid_shape[0], agent.grid_stride[0])
                     for j in np.arange(0, agent.grid_shape[1], agent.grid_stride[1])]
        logits = agent.pred if not draw_groundtruth else agent.label

        plt.figure(figsize=(16, 12))
        plt.imshow(np.stack([mean_image[:, :, 2],
                             mean_image[:, :, 1],
                             mean_image[:, :, 0], ],
                            axis=-1))    # BGR -> RGB

        logits = tf.stack(logits)
        if regulation:
            logits = self.regulation(logits)

        for index, pos in enumerate(pos_index):
            pixel_pos = self.grid2pixel(pos)
            _logits = logits[index]

            if len(_logits.shape) >= 2:
                score = _logits[1] if _logits[1] > _logits[0] else _logits[0]
                text = '{} ({:.2f})'.format(
                    1 if _logits[1] > _logits[0] else 0,
                    score)

            else:
                score = _logits if len(_logits.shape) == 0 else _logits[0]
                text = '{:.2f}'.format(score)

            plt.gca().add_patch(plt.Rectangle(
                xy=pixel_pos,
                width=self.shape[0] / self.grid_shape[0],
                height=self.shape[1] / self.grid_shape[1],
                edgecolor='r',
                fill=draw_heatmap,
                facecolor=self.get_heatmap_color(score),
                linewidth=2))

            plt.text(pixel_pos[0],
                     pixel_pos[1],
                     text,
                     bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))

        plus_name = agent.local_name if file_name == 'auto' else file_name
        file_name = dataset + '_' + plus_name + \
            ('_gt' if draw_groundtruth else '')
        plt.savefig(os.path.join(save_base_path, '{}.jpg'.format(file_name)))
