"""
@Author: Conghao Wong
@Date: 2021-09-16 19:53:44
@LastEditors: Conghao Wong
@LastEditTime: 2021-12-30 10:39:44
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import List
from .. import applications as A
from .. import models as M

import tensorflow as tf
from tensorflow import keras


class LinearModel(M.prediction.Model):
    def __init__(self, Args: M.prediction.PredictionArgs,
                 training_structure=None,
                 *args, **kwargs):

        super().__init__(Args, training_structure, *args, **kwargs)

        self.linear_layer = A.layers.LinearLayer(
            obs_frames=Args.obs_frames,
            pred_frames=Args.pred_frames,
            diff=0.95)

    def call(self, inputs, training=None, mask=None, *args, **kwargs):
        return self.linear_layer.call(inputs[0])


class LinearStructure(M.prediction.Structure):
    def __init__(self, Args: List[str], *args, **kwargs):
        super().__init__(Args, *args, **kwargs)
        self.args._set('epochs', 5)

    def create_model(self, *args, **kwargs):
        model = LinearModel(self.args)
        opt = keras.optimizers.Adam(self.args.lr)
        return model, opt
