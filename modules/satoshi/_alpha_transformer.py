"""
@Author: Conghao Wong
@Date: 2021-05-06 11:06:15
@LastEditors: Conghao Wong
@LastEditTime: 2021-05-10 11:26:30
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


class SatoshiAlphaTransformerModel(M.prediction.Model):
    def __init__(self, Args: SatoshiArgs, training_structure=None, *args, **kwargs):
        super().__init__(Args, training_structure=training_structure, *args, **kwargs)

        self.set_preprocess('move')
        self.set_preprocess_parameters(move=0)

        # context feature
        self.average_pooling = keras.layers.AveragePooling2D([5, 5],
                                                             input_shape=[100, 100, 1])
        self.flatten = keras.layers.Flatten()
        self.context_dense1 = keras.layers.Dense(self.args.obs_frames * 64,
                                                 activation=tf.nn.tanh)

        # traj embedding
        self.pos_embedding = keras.layers.Dense(64, tf.nn.tanh)
        self.concat = keras.layers.Concatenate()

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
        self.adj_dense2 = keras.layers.Dense(self.args.K_train,
                                             activation=tf.nn.tanh)
        self.gcn_transfer = M.helpMethods.GraphConv_layer(128, tf.nn.tanh)

        # decoder
        self.decoder = keras.layers.Dense(2)

    def call(self, inputs: List[tf.Tensor], outputs: tf.Tensor = None, training=None, mask=None):
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
        me, mc, md = A.create_transformer_masks(t_inputs, t_outputs)

        # shape == (batch, obs, 128)
        t_features, _ = self.T1(t_inputs, t_outputs,
                                training,
                                me, mc, md)

        # transfer GCN
        adj_matrix_transfer_T = self.adj_dense2(
            concat_feature)   # shape = [batch, obs, pred]
        adj_matrix_transfer = tf.transpose(adj_matrix_transfer_T, [0, 2, 1])
        future_feature = M.helpMethods.GraphConv_func(
            t_features, adj_matrix_transfer, layer=self.gcn_transfer)

        # decoder
        predictions = self.decoder(future_feature)

        return predictions

    # @tf.function
    def forward(self, model_inputs: Tuple[tf.Tensor], training=False, *args, **kwargs):
        """
        Run a forward implementation.

        :param model_inputs: input tensor (or a list of tensors)
        :param mode: choose forward type, can be `'test'` or `'train'`
        :return output: model's output. type=`List[tf.Tensor]`
        """
        model_inputs_processed = self.pre_process(model_inputs, training)

        if training:
            gt_processed = self.pre_process([kwargs['gt']],
                                            use_new_para_dict=False)

        output = self.call(model_inputs_processed,
                           gt_processed[0] if training else None,
                           training=training)   # use `self.call()` to debug

        if not (type(output) == list or type(output) == tuple):
            output = [output]

        return self.post_process(output, training, model_inputs=model_inputs)


class SatoshiAlphaTransformer(M.prediction.Structure):
    def __init__(self, args, arg_type=SatoshiArgs):
        super().__init__(args, arg_type=arg_type)

        self.set_model_inputs('traj', 'maps')
        self.set_model_groundtruths('destination')

        self.set_loss(self.min_FDE)
        self.set_loss_weights(1.0)

        self.set_metrics(self.min_FDE)
        self.set_metrics_weights(1.0)

    def create_model(self, model_type=SatoshiAlphaTransformerModel):
        model = model_type(self.args, training_structure=self)
        # model.build([[None, self.args.obs_frames, 2], [None, None, 2]])
        # model.summary()
        opt = keras.optimizers.Adam(self.args.lr)
        return model, opt

    def min_FDE(self, outputs, labels) -> tf.Tensor:
        distance = tf.linalg.norm(
            outputs[0] - tf.expand_dims(labels[:, -1, :], 1), axis=-1)   # shape = [batch, K]
        return tf.reduce_mean(tf.reduce_min(distance, axis=-1))
