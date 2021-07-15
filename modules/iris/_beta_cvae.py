"""
@Author: Conghao Wong
@Date: 2021-06-23 16:21:48
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-14 10:43:03
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

from ._beta import Encoder, Generator
from ..satoshi._args import SatoshiArgs
from ..satoshi._beta_transformer import linear_prediction


class IrisBetaCVAEModel(M.prediction.Model):
    def __init__(self, Args, training_structure=None, *args, **kwargs):
        super().__init__(Args, training_structure=training_structure, *args, **kwargs)

        self.set_preprocess('move')
        self.set_preprocess_parameters(move=0)

        self.E = Encoder(Args)
        self.G = Generator()

    def call(self, inputs: List[tf.Tensor],
             outputs: tf.Tensor = None,
             training=None, mask=None):
        
        obs = inputs[0]
        features = self.E(inputs)

        K = self.args.K_train if training else self.args.K
        sigma = 1.0 if training else self.args.sigma

        all_outputs = []
        for repeat in range(K):
            z = tf.random.normal(features.shape, 0.0, sigma)
            all_outputs.append(self.G([features, z]))

        # shape = (batch, K, pred, 2)
        pred = tf.transpose(tf.stack(all_outputs), [1, 0, 2, 3])

        return (pred, features)

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

        d = model_inputs[-1]
        des = d if len(d.shape) == 3 else d[:, tf.newaxis, :]
        destination_processed = self.pre_process([des],
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
                           gt_processed[0] if training else None,
                           training=training)   # use `self.call()` to debug

        if not (type(output) == list or type(output) == tuple):
            output = [output]

        return self.post_process(output, training, model_inputs=model_inputs)


class IrisBetaCVAE(M.prediction.Structure):
    def __init__(self, Args: List[str], *args, **kwargs):
        super().__init__(Args, *args, **kwargs)

        self.args = SatoshiArgs(Args)

        self.set_model_inputs('trajs', 'maps', 'paras', 'destinations')
        self.set_model_groundtruths('gt')

        self.set_loss('ade', 'diff', self.p_loss)
        self.set_loss_weights(0.8, 0.2, 1.0)

        self.set_metrics('ade', 'fde')
        self.set_metrics_weights(0.3, 0.7)

    def create_model(self, model_type=IrisBetaCVAEModel):
        model = model_type(self.args, training_structure=self)
        opt = keras.optimizers.Adam(self.args.lr)
        return model, opt

    def load_forward_dataset(self, model_inputs: Tuple[tf.Tensor], **kwargs):
        trajs = model_inputs[0]
        maps = model_inputs[1]
        paras = model_inputs[2]
        proposals = model_inputs[3]
        return tf.data.Dataset.from_tensor_slices((trajs, maps, paras, proposals))

    def p_loss(self, model_outputs: Tuple[tf.Tensor], labels=None):
        features = tf.reshape(model_outputs[1], [-1, 128])
        
        mu_real = tf.reduce_mean(features, axis=0)  # (128)
        std_real = tf.math.reduce_std(features, axis=0) # (128)

        return tf.reduce_mean(tf.abs(mu_real - 0)) + tf.reduce_mean(tf.abs(std_real - 1))