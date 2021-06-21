"""
@Author: Conghao Wong
@Date: 2021-03-24 09:01:15
@LastEditors: Conghao Wong
@LastEditTime: 2021-05-06 11:07:04
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

from ._args import SatoshiArgs


class SatoshiAlphaModel(M.base.Model):
    def __init__(self, Args,
                 training_structure=None,
                 gcn_layer_count=2,
                 intention_count=10,
                 *args, **kwargs):

        super().__init__(Args, training_structure=training_structure, *args, **kwargs)
        self.gcn_layer_count = gcn_layer_count
        self.intention_count = intention_count

        # GCN layers
        self.pos_embadding = keras.layers.Dense(
            64, activation=tf.nn.tanh, input_shape=[self.args.obs_frames, 2])
        self.adj_dense = keras.layers.Dense(
            self.args.obs_frames, activation=tf.nn.tanh)
        self.gcn_layers = {}
        for count in range(self.gcn_layer_count):
            self.gcn_layers[str(count)] = M.helpMethods.GraphConv_layer(
                64, activation=tf.nn.relu if count < self.gcn_layer_count - 1 else tf.nn.tanh)

        self.gcn_dropout = keras.layers.Dropout(self.args.dropout)
        self.gcn_bn = keras.layers.BatchNormalization()

        self.adj_dense2 = keras.layers.Dense(
            self.intention_count, activation=tf.nn.tanh)
        self.gcn_transfer = M.helpMethods.GraphConv_layer(64, tf.nn.tanh)

        # context feature
        self.average_pooling = keras.layers.AveragePooling2D(
            [5, 5], input_shape=[100, 100, 1])
        self.flatten = keras.layers.Flatten()
        self.context_dense1 = keras.layers.Dense(
            self.intention_count * 64, activation=tf.nn.tanh)

        # decoder
        self.concat = keras.layers.Concatenate()
        self.decoder = keras.layers.Dense(2)

    def call(self, inputs: List[tf.Tensor], training=None, mask=None):
        positions = inputs[0]
        maps = inputs[1]

        # historical GCN -> historical feature ([batch, obs, 64])
        positions_embadding = self.pos_embadding(positions)
        adj_matrix = self.adj_dense(positions_embadding)

        gcn_input = positions_embadding
        for repeat in range(self.gcn_layer_count):
            gcn_output = M.helpMethods.GraphConv_func(
                gcn_input,
                adj_matrix,
                layer=self.gcn_layers[str(repeat)])
            gcn_input = gcn_output

        dropout = self.gcn_dropout(gcn_output, training=training)
        historical_feature = self.gcn_bn(dropout)

        # context feature -> context feature ([batch, K, 64])
        maps_r = tf.expand_dims(maps, -1)
        average_pooling = self.average_pooling(maps_r)
        flatten = self.flatten(average_pooling)
        context_feature = self.context_dense1(flatten)
        context_feature = tf.reshape(
            context_feature, [-1, self.intention_count, 64])

        # transfer GCN
        adj_matrix_transfer_T = self.adj_dense2(
            positions_embadding)   # shape = [batch, obs, pred]
        adj_matrix_transfer = tf.transpose(adj_matrix_transfer_T, [0, 2, 1])
        future_feature = M.helpMethods.GraphConv_func(
            historical_feature, adj_matrix_transfer, layer=self.gcn_transfer)

        # decoder
        concat_feature = self.concat([future_feature, context_feature])
        predictions = self.decoder(concat_feature)

        return predictions

    def pre_process(self,
                    model_inputs: Tuple[tf.Tensor],
                    training=False,
                    **kwargs) -> Tuple[tf.Tensor]:
        trajs = model_inputs[0]
        trajs, self.move_dict = M.prediction.Process.move(trajs)
        # trajs, self.rotate_dict = M.prediction.Process.rotate(trajs)
        # trajs, self.scale_dict = M.prediction.Process.scale(trajs)
        return M.prediction.Process.update((trajs,), model_inputs)

    def post_process(self,
                     outputs: Tuple[tf.Tensor],
                     training=False,
                     **kwargs) -> Tuple[tf.Tensor]:
        trajs = outputs[0]
        # trajs = M.prediction.Process.scale_back(trajs, self.scale_dict)
        # trajs = M.prediction.Process.rotate_back(trajs, self.rotate_dict)
        trajs = M.prediction.Process.move_back(trajs, self.move_dict)
        return M.prediction.Process.update((trajs,), outputs)


class SatoshiAlpha(M.prediction.Structure):
    """
    Training Structure for Satoshi-alpha model
    """

    def __init__(self, args, arg_type=SatoshiArgs):
        super().__init__(args, arg_type=arg_type)
        self.gcn_layer_count = self.args.gcn_layers

        self.set_model_inputs('traj', 'maps')
        self.set_model_groundtruths('destination')

        self.set_loss(self.min_FDE)
        self.set_loss_weights(1.0)

        self.set_metrics(self.min_FDE)
        self.set_metrics_weights(1.0)

    def create_model(self, model_type=SatoshiAlphaModel):
        model = model_type(self.args,
                           gcn_layer_count=2,
                           intention_count=self.args.K_train,
                           training_structure=self)
        model.build([[None, self.args.obs_frames, 2],
                     [None, 100, 100], [None, 2, 2]])
        # model.summary()
        opt = keras.optimizers.Adam(self.args.lr)
        return model, opt

    def min_FDE(self, outputs, labels) -> tf.Tensor:
        distance = tf.linalg.norm(
            outputs[0] - tf.expand_dims(labels[:, -1, :], 1), axis=-1)   # shape = [batch, K]
        return tf.reduce_mean(tf.reduce_min(distance, axis=-1))
