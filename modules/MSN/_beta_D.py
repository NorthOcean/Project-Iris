"""
@Author: Conghao Wong
@Date: 2021-05-07 09:12:57
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-16 16:25:08
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import Dict, List, Tuple

import modules.applications as A
import modules.models as M
import numpy as np
import tensorflow as tf
from tensorflow import keras as keras

from ._args import MSNArgs


class MSNBeta_DModel(M.prediction.Model):
    def __init__(self, Args: MSNArgs, 
                 training_structure=None,
                 *args, **kwargs):

        super().__init__(Args, training_structure, *args, **kwargs)

        self.set_preprocess('move')
        self.set_preprocess_parameters(move=0)

        # context feature
        self.average_pooling = keras.layers.AveragePooling2D([5, 5],
                                                             input_shape=[100, 100, 1])
        self.flatten = keras.layers.Flatten()
        self.context_dense1 = keras.layers.Dense((self.args.obs_frames+1) * 64,
                                                 activation=tf.nn.tanh)

        # traj embedding
        self.pos_embedding = keras.layers.Dense(64, tf.nn.tanh)
        self.concat = keras.layers.Concatenate()

        # trajectory transformer
        self.transformer = A.Transformer(num_layers=4,
                                         d_model=128,
                                         num_heads=8,
                                         dff=512,
                                         input_vocab_size=None,
                                         target_vocab_size=2,
                                         pe_input=Args.obs_frames + 1,
                                         pe_target=Args.pred_frames)

    def call(self, inputs: List[tf.Tensor],
             training=None, mask=None):

        positions_ = inputs[0]
        maps = inputs[1]
        destinations = inputs[3]

        # concat positions and destinations
        positions = tf.concat([positions_, destinations], axis=1)

        # traj embedding, shape == (batch, obs+1, 64)
        positions_embedding = self.pos_embedding(positions)

        # context feature, shape == (batch, obs+1, 64)
        average_pooling = self.average_pooling(maps[:, :, :, tf.newaxis])
        flatten = self.flatten(average_pooling)
        context_feature = self.context_dense1(flatten)
        context_feature = tf.reshape(context_feature,
                                     [-1, self.args.obs_frames+1, 64])

        # concat, shape == (batch, obs+1, 128)
        concat_feature = self.concat([positions_embedding, context_feature])

        t_inputs = concat_feature
        t_outputs = linear_prediction(positions[:, -2:, :],
                                      self.args.pred_frames,
                                      return_zeros=False)
        me, mc, md = A.create_transformer_masks(t_inputs, t_outputs)
        predictions, _ = self.transformer(t_inputs, t_outputs, True,
                                          me, mc, md)

        return predictions

    # @tf.function
    def forward(self, model_inputs: Tuple[tf.Tensor],
                training=False,
                *args, **kwargs):
        """
        Run a forward implementation.

        :param model_inputs: input tensor (or a list of tensors)
        :param mode: choose forward type, can be `'test'` or `'train'`
        :return output: model's output. type=`List[tf.Tensor]`
        """
        model_inputs_processed = self.pre_process(model_inputs, training)
        destination_processed = self.pre_process([model_inputs[-1]],
                                                 training,
                                                 use_new_para_dict=False)

        model_inputs_processed = (model_inputs_processed[0],
                                  model_inputs_processed[1],
                                  model_inputs_processed[2],
                                  destination_processed[0])

        if training:
            gt_processed = self.pre_process([kwargs['gt']],
                                            use_new_para_dict=False)

        output = self.call(model_inputs_processed,
                           training=training)   # use `self.call()` to debug

        if not (type(output) == list or type(output) == tuple):
            output = [output]

        return self.post_process(output, training, model_inputs=model_inputs)


class MSNBeta_D(M.prediction.Structure):
    def __init__(self, Args: List[str], *args, **kwargs):
        super().__init__(Args, *args, **kwargs)

        self.args = MSNArgs(Args)

        self.set_model_inputs('trajs', 'maps', 'paras', 'destinations')
        self.set_model_groundtruths('gt')

        self.set_loss('ade', 'diff')
        self.set_loss_weights(0.8, 0.2)

        self.set_metrics('ade', 'fde')
        self.set_metrics_weights(1.0, 0.0)

    def create_model(self, model_type=MSNBeta_DModel):
        model = model_type(self.args, training_structure=self)
        opt = keras.optimizers.Adam(self.args.lr)
        return model, opt

    def load_forward_dataset(self, model_inputs: Tuple[tf.Tensor], **kwargs):
        trajs = model_inputs[0]
        maps = model_inputs[1]
        paras = model_inputs[2]
        proposals = model_inputs[3]
        return tf.data.Dataset.from_tensor_slices((trajs, maps, paras, proposals))


def linear_prediction(end_points: tf.Tensor, number, return_zeros=None):
    """
    Linear prediction from start points (not contain) to end points.

    :param end_points: start points and end points, shape == (batch, 2, 2)
    :param number: number of prediction points, DO NOT contain start point
    """
    if return_zeros:
        return tf.pad(end_points[:, -1:, :], [0, 0], [number-1, 0], [0, 0])

    start = end_points[:, :1, :]
    end = end_points[:, -1:, :]

    r = []
    for n in range(1, number):
        p = n / number
        r_c = (end - start) * p + start  # shape = (batch, 1, 2)
        r.append(r_c)

    r.append(end)
    return tf.concat(r, axis=1)
