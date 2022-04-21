"""
@Author: Conghao Wong
@Date: 2021-06-21 15:01:50
@LastEditors: Conghao Wong
@LastEditTime: 2022-04-21 11:01:03
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import modules.applications as A
import modules.models as M
import numpy as np
import tensorflow as tf

from .__args import MSNArgs


class MSNAlphaModel(M.prediction.Model):
    """
    First stage model, i.e., the Destination Transformer
    """
    def __init__(self, Args: MSNArgs,
                 training_structure=None,
                 *args, **kwargs):

        super().__init__(Args, training_structure, *args, **kwargs)

        self.set_preprocess('move')
        self.set_preprocess_parameters(move=0)

        # context feature
        self.average_pooling = tf.keras.layers.AveragePooling2D([5, 5],
                                                             input_shape=[100, 100, 1])
        self.flatten = tf.keras.layers.Flatten()
        self.context_dense1 = tf.keras.layers.Dense(self.args.obs_frames * 64,
                                                 activation=tf.nn.tanh)

        # traj embedding
        self.pos_embedding = tf.keras.layers.Dense(64, tf.nn.tanh)
        self.concat = tf.keras.layers.Concatenate()

        # trajectory transformer
        self.T1 = A.Transformer(num_layers=4,
                                d_model=128,
                                num_heads=8,
                                dff=512,
                                input_vocab_size=None,
                                target_vocab_size=None,
                                pe_input=Args.obs_frames,
                                pe_target=Args.obs_frames,
                                include_top=False)

        # transfer GCN
        self.adj_dense2 = tf.keras.layers.Dense(self.args.K_train,
                                             activation=tf.nn.tanh)
        self.gcn_transfer = M.helpmethods.GraphConv_layer(128, tf.nn.tanh)

        # decoder
        self.decoder = tf.keras.layers.Dense(2)

    def call(self, inputs: list[tf.Tensor], training=None, mask=None):
        positions = inputs[0]
        maps = inputs[1]

        # traj embedding, shape == (batch, obs, 64)
        positions_embedding = self.pos_embedding(positions)

        # context feature, shape == (batch, obs, 64)
        average_pooling = self.average_pooling(maps[:, :, :, tf.newaxis])
        flatten = self.flatten(average_pooling)
        context_feature = self.context_dense1(flatten)
        context_feature = tf.reshape(context_feature,
                                     [-1, self.args.obs_frames, 64])

        # concat, shape == (batch, obs, 128)
        concat_feature = self.concat([positions_embedding, context_feature])

        # transformer
        t_inputs = concat_feature
        t_outputs = positions

        # shape == (batch, obs, 128)
        t_features, _ = self.T1.call(t_inputs, 
                                     t_outputs,
                                     training=training)

        # transfer GCN
        adj_matrix_transfer_T = self.adj_dense2(
            concat_feature)   # shape = [batch, obs, pred]
        adj_matrix_transfer = tf.transpose(adj_matrix_transfer_T, [0, 2, 1])
        future_feature = M.helpmethods.GraphConv_func(
            t_features, adj_matrix_transfer, layer=self.gcn_transfer)

        # decoder
        predictions = self.decoder(future_feature)

        return predictions


class MSNAlpha(M.prediction.Structure):
    """
    Structure for the first stage Destination Transformer.
    """
    def __init__(self, Args: list[str], *args, **kwargs):
        super().__init__(Args, *args, **kwargs)

        self.args = MSNArgs(Args)

        self.set_model_inputs('traj', 'maps')
        self.set_model_groundtruths('destination')

        self.set_loss(self.min_FDE)
        self.set_loss_weights(1.0)

        self.set_metrics(self.min_FDE)
        self.set_metrics_weights(1.0)

    def create_model(self, model_type=MSNAlphaModel, *args, **kwargs):
        model = model_type(self.args, 
                           training_structure=self,
                           *args, **kwargs)
        opt = tf.keras.optimizers.Adam(self.args.lr)
        return model, opt

    def min_FDE(self, outputs, labels) -> tf.Tensor:
        distance = tf.linalg.norm(
            outputs[0] - tf.expand_dims(labels[:, -1, :], 1), axis=-1)   # shape = [batch, K]
        return tf.reduce_mean(tf.reduce_min(distance, axis=-1))
