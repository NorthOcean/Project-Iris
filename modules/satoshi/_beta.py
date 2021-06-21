'''
Author: Conghao Wong
Date: 2021-03-24 09:01:23
LastEditors: Conghao Wong
LastEditTime: 2021-04-19 11:20:54
Description: file content
'''

from typing import Dict, List, Tuple

import modules.models as M
import numpy as np
import tensorflow as tf
from tensorflow import keras as keras

from ._args import SatoshiArgs


class SatoshiBetaModel(M.base.Model):
    def __init__(self, Args, training_structure=None, gcn_layer_count=2, obs_count=3, *args, **kwargs):
        super().__init__(Args, training_structure=training_structure, *args, **kwargs)
        self.gcn_layer_count = gcn_layer_count
        self.obs_count = obs_count

        # GCN layers
        self.pos_embadding = keras.layers.Dense(
            64, activation=tf.nn.tanh, input_shape=[self.obs_count + 1, 2])
        self.adj_dense = keras.layers.Dense(
            self.obs_count + 1, activation=tf.nn.tanh)
        self.gcn_layers = {}
        for count in range(self.gcn_layer_count):
            self.gcn_layers[str(count)] = M.helpMethods.GraphConv_layer(
                64, activation=tf.nn.relu if count < self.gcn_layer_count - 1 else tf.nn.tanh)

        self.gcn_dropout = keras.layers.Dropout(self.args.dropout)
        self.gcn_bn = keras.layers.BatchNormalization()

        self.adj_dense2 = keras.layers.Dense(
            self.args.pred_frames, activation=tf.nn.tanh)
        self.gcn_transfer = M.helpMethods.GraphConv_layer(64, tf.nn.tanh)

        # context feature
        self.average_pooling = keras.layers.AveragePooling2D(
            [5, 5], input_shape=[100, 100, 1])
        self.flatten = keras.layers.Flatten()
        self.context_dense1 = keras.layers.Dense(
            self.args.pred_frames * 64, activation=tf.nn.tanh)

        # # intention feature
        # self.int_embadding = keras.layers.Dense(64, tf.nn.tanh)

        # decoder
        self.concat = keras.layers.Concatenate()
        self.decoder = keras.layers.Dense(2)

    def call(self, inputs, training=None, mask=None):
        # shape = [batch, obs, 2]
        positions = inputs[0][:, -self.obs_count:, :]
        maps = inputs[1]
        paras = inputs[2]
        intentions = inputs[3]  # shape = [batch, 2]

        intentions = tf.expand_dims(intentions, 1)
        # shape = [batch, obs+1, 2]
        all_positions = tf.concat([positions, intentions], axis=1)

        positions_embadding = self.pos_embadding(all_positions)
        adj_matrix = self.adj_dense(positions_embadding)

        # historical GCN -> historical feature ([batch, obs, 64])
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
        maps_r = tf.reshape(maps, [-1, 100, 100, 1])
        average_pooling = self.average_pooling(maps_r)
        flatten = self.flatten(average_pooling)
        context_feature = self.context_dense1(flatten)
        context_feature = tf.reshape(
            context_feature, [-1, self.args.pred_frames, 64])

        # transfer GCN
        adj_matrix_transfer_T = self.adj_dense2(
            positions_embadding)   # shape = [batch, obs, pred]
        adj_matrix_transfer = tf.transpose(adj_matrix_transfer_T, [0, 2, 1])
        future_feature = M.helpMethods.GraphConv_func(
            historical_feature, adj_matrix_transfer, layer=self.gcn_transfer)

        # # UNMANED -> intention conditioned feature ([batch, pred, 64])
        # intentions_embadding = self.int_embadding(intentions)
        # intention_feature = tf.repeat(
        #     tf.expand_dims(intentions_embadding, 1),
        #     self.args.pred_frames,
        #     axis=1
        # ) # [batch, pred, 64]

        # decoder -> trajs ([batch, pred, 2])
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


class SatoshiBeta(M.prediction.Structure):
    """
    Training structure for Satoshi-beta model
    """

    def __init__(self, args, arg_type=SatoshiArgs):
        super().__init__(args, arg_type=arg_type)
        self.gcn_layer_count = self.args.gcn_layers

        self.set_model_inputs('trajs', 'maps', 'paras', 'destinations')
        self.set_model_groundtruths('gt')

    def create_model(self, model_type=SatoshiBetaModel):
        model = model_type(self.args,
                           gcn_layer_count=2,
                           obs_count=self.args.H,
                           training_structure=self)
        model.build([[None, self.args.obs_frames, 2], [
                    None, 100, 100], [None, 2, 2], [None, 2]])
        # model.summary()
        opt = keras.optimizers.Adam(self.args.lr)
        return model, opt

    def loss(self, outputs, labels, loss_name_list=['ADE'], **kwargs):
        loss_diff = M.prediction.Loss.diff(outputs[0], labels, ord=1)
        loss_fde = M.prediction.Loss.FDE(outputs[0], labels)

        loss = 0.1 * loss_fde + 0.8 * loss_diff[0] + 0.1 * loss_diff[1]
        loss_dict = dict(zip(['fde', 'l2(0)', 'l2(1)'], [
                         loss_fde, loss_diff[0], loss_diff[1]]))
        return loss, loss_dict

    def load_forward_dataset(self, model_inputs: Tuple[tf.Tensor], **kwargs):
        trajs = model_inputs[0]
        maps = model_inputs[1]
        paras = model_inputs[2]
        proposals = model_inputs[3]
        return tf.data.Dataset.from_tensor_slices((trajs, maps, paras, proposals))
