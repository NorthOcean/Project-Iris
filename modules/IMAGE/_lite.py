"""
@Author: Conghao Wong
@Date: 2021-07-08 09:17:23
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-08 10:35:45
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import Any, List, Tuple

import tensorflow as tf
from tensorflow import keras

from .. import models as M
from ..satoshi._args import SatoshiArgs
from ._training import IMAGEStructure, load_image


class IMAGELiteModel(M.prediction.Model):

    def __init__(self, Args, training_structure=None, *args, **kwargs):
        super().__init__(Args, training_structure, *args, **kwargs)

        self.img_shape = (256, 256)
        self.Kc = 10

        self.concat = keras.layers.Concatenate()
        self.pooling = keras.layers.MaxPool2D((4, 4))
        self.flatten = keras.layers.Flatten()

        self.fc1 = keras.layers.Dense(512, tf.nn.relu)
        self.fc2 = keras.layers.Dense(512, tf.nn.relu)
        self.fc3 = keras.layers.Dense(2 * self.Kc)
        self.reshape = keras.layers.Reshape((self.Kc, 2))

    def call(self, inputs: List[tf.Tensor],
             training=None, mask=None):

        # input shape = ((batch, imgshape, imgshape, 3),
        #                (batch, imgshape, imgshape, 2),
        #                (batch, obs, 2))

        scene_img, traj_img, traj = inputs[:3]
        image = self.concat([scene_img, traj_img])

        pooling = self.pooling(image)
        flatten = self.flatten(pooling)
        fc = self.fc3(self.fc2(self.fc1(flatten)))
        return self.reshape(fc)

    def pre_process(self, tensors: Tuple[tf.Tensor],
                    training=None,
                    **kwargs) -> Tuple[tf.Tensor]:

        [img_s_paths, img_t_paths, trajs] = tensors[:3]
        scene = tf.stack([load_image(img) for img in img_s_paths])
        traj_imgs = tf.stack([load_image(img)[:, :, :2]
                             for img in img_t_paths])

        self.file_names = img_t_paths
        return (scene, traj_imgs, trajs)


class IMAGELite(IMAGEStructure):
    def __init__(self, args):
        super().__init__(args)

        self.set_loss(self.min_FDE)
        self.set_loss_weights(1.0)

        self.set_metrics(self.min_FDE)
        self.set_metrics_weights(1.0)

    def create_model(self) -> Tuple[Any, keras.optimizers.Optimizer]:
        model = IMAGELiteModel(self.args)
        opt = keras.optimizers.Adam(self.args.lr)

        model.build([[None, 256, 256, 3],
                     [None, 256, 256, 2],
                     [None, self.args.obs_frames, 2]])
        model.summary()
        return model, opt

    def min_FDE(self, outputs: List[tf.Tensor], labels: tf.Tensor) -> tf.Tensor:
        """
        Calculate min FDE

        :param outputs: a list of outputs, where `output[0]` is the predictions with shape `(None, pred, 2)`
        :param labels: groundtruth trajectories, shape = `(None, pred, 2)`
        """
        distance = tf.linalg.norm(outputs[0] - labels[:, -1:, :],
                                  ord=2, axis=-1)
        return tf.reduce_mean(tf.reduce_min(distance, axis=-1))
