"""
@Author: Conghao Wong
@Date: 2021-06-21 15:05:18
@LastEditors: Conghao Wong
@LastEditTime: 2021-06-23 16:29:50
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

from ..satoshi._args import SatoshiArgs
from ..satoshi._beta_transformer import linear_prediction


class Encoder(keras.Model):
    def __init__(self, Args: SatoshiArgs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = Args

        # context feature
        self.average_pooling = keras.layers.AveragePooling2D([5, 5],
                                                             input_shape=[100, 100, 1])
        self.flatten = keras.layers.Flatten()
        self.context_dense1 = keras.layers.Dense((Args.obs_frames+1) * 64,
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
                                         pe_target=Args.pred_frames,
                                         include_top=False)

    def call(self, inputs: List[tf.Tensor], training=None, mask=None):
        """
        Encode inputs and destinations into features.
        Output shape = (batch, pred, 128)

        :param inputs: a list of input tensors, where
            inputs[0] is the trajectory input, shape = (batch, obs, 2);
            inputs[1] is the map input, shape = (batch, 100, 100);
            inputs[2] (leave None now)
            inputs[3] is the destination input, shape = (batch, 2).
        """
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
        features, _ = self.transformer(t_inputs, t_outputs, True,
                                       me, mc, md)
        return features


class Generator(keras.Model):
    """
    Generator in IrisBeta model.

    :param inputs: a list of tensors, which contains
        `inputs[0]`: features from encoder, shape = (batch, pred, 128)
        `inputs[1]`: noise vector z, shape = (batch, pred, 128)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Generator layers
        self.g1 = keras.layers.Dense(128, activation=tf.nn.relu)
        self.g2 = keras.layers.Dense(2)

    def call(self, inputs: List[tf.Tensor], training=None, mask=None):
        features = inputs[0]
        z = inputs[1]

        new_feature = features + z
        g1 = self.g1(new_feature)
        g2 = self.g2(g1 + new_feature)
        return g2


class Discriminator(keras.Model):
    """
    Discriminator in IrisBeta model.
    
    :param inputs: a list of tensors, which contains
        `inputs[0]`: predictions, shape = (batch, pred, 2)
        `inputs[1]`: observations, shape = (batch, obs, 2)
    """
    def __init__(self, Args: SatoshiArgs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = Args

        # Discriminator layers
        self.d1 = keras.layers.Dense(128, activation=tf.nn.relu)
        self.d2 = keras.layers.Conv2D(4, (Args.pred_frames + Args.obs_frames, 1), activation=tf.nn.relu)
        self.d_flatten = keras.layers.Flatten()
        self.d3 = keras.layers.Dense(2, activation=tf.nn.sigmoid)

    def call(self, inputs: List[tf.Tensor], training=None, mask=None):
        pred = inputs[0]
        obs = inputs[1]
        
        traj = tf.concat([obs, pred], axis=1)   # (batch, pred+obs, 2)
        d1 = self.d1(traj)   # (batch, pred+obs, 128)
        d2 = self.d2(d1[:, :, :, tf.newaxis])[:, 0, :, :]   # (batch, 128, 4)
        df = self.d_flatten(d2) # (batch, 512)
        logits = self.d3(df)
        return logits


class IrisBetaModel(M.prediction.Model):
    def __init__(self, Args, training_structure=None, *args, **kwargs):
        super().__init__(Args, training_structure=training_structure, *args, **kwargs)

        self.set_preprocess('move')
        self.set_preprocess_parameters(move=0)

        self.E = Encoder(Args)
        self.G = Generator()
        self.D = Discriminator(Args)

    def call(self, inputs: List[tf.Tensor],
             outputs: tf.Tensor = None,
             training=None, mask=None):
        
        obs = inputs[0]
        features = self.E(inputs)

        if training:
            fake_outputs = []
            fake_logits = []
            real_logits = []

            for repeat in range(self.args.K_train):
                z = tf.random.normal(features.shape)
                fake_outputs.append(fo := self.G([features, z]))
                fake_logits.append(self.D([fo, obs]))
                real_logits.append(self.D([outputs, obs]))
            
            fake_outputs = tf.transpose(tf.stack(fake_outputs), [1, 0, 2, 3])
            fake_logits = tf.transpose(tf.stack(fake_logits), [1, 0, 2])
            real_logits = tf.transpose(tf.stack(real_logits), [1, 0, 2])

            return fake_outputs, fake_logits, real_logits, features
        
        else:
            all_outputs = []
            for repeat in range(self.args.K):
                z = tf.random.normal(features.shape, mean=0, stddev=self.args.sigma)
                outputs = self.G([features, z])
                all_outputs.append(outputs)

            # shape = (batch, K, pred, 2)
            return tf.transpose(tf.stack(all_outputs), [1, 0, 2, 3])

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


class IrisBeta(M.prediction.Structure):
    def __init__(self, args, arg_type=SatoshiArgs):
        super().__init__(args, arg_type=arg_type)

        self.set_model_inputs('trajs', 'maps', 'paras', 'destinations')
        self.set_model_groundtruths('gt')

        self.set_loss('ade', 'diff', self.p_loss)
        self.set_loss_weights(0.8, 0.2, 1.0)

        self.set_metrics('ade', 'fde')
        self.set_metrics_weights(0.3, 0.7)

    def create_model(self, model_type=IrisBetaModel):
        model = model_type(self.args, training_structure=self)
        opt = keras.optimizers.Adam(self.args.lr)
        return model, opt

    def load_forward_dataset(self, model_inputs: Tuple[tf.Tensor], **kwargs):
        trajs = model_inputs[0]
        maps = model_inputs[1]
        paras = model_inputs[2]
        proposals = model_inputs[3]
        return tf.data.Dataset.from_tensor_slices((trajs, maps, paras, proposals))

    def gradient_operations(self, model_inputs,
                            gt,
                            loss_move_average: tf.Variable,
                            **kwargs) -> Tuple[tf.Tensor, Dict[str, tf.Tensor], tf.Tensor]:

        with tf.GradientTape(persistent=True) as tape:
            model_output = self.model_forward(model_inputs, training=True, gt=gt)
            loss, loss_dict = self.loss(model_output,
                                        gt,
                                        model_inputs=model_inputs,
                                        **kwargs)

            d_loss = self.gan_loss(model_output)
            g_loss = -1.0 * d_loss
            
            loss_move_average = 0.7 * loss + 0.3 * loss_move_average + g_loss

        trainable_variables = self.model.E.trainable_variables + self.model.G.trainable_variables
        grads = tape.gradient(loss_move_average, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))
        
        d_grads = tape.gradient(d_loss, self.model.D.trainable_variables)
        self.optimizer.apply_gradients(zip(d_grads, self.model.D.trainable_variables))

        loss_dict['D_loss'] = d_loss

        return loss, loss_dict, loss_move_average

    def gan_loss(self, model_outputs: Tuple[tf.Tensor], labels: tf.Tensor=None):
        fake_logits = model_outputs[1]
        real_logits = model_outputs[2]

        fake_labels = tf.one_hot(tf.zeros_like(fake_logits, tf.int32)[:, :, 0], depth=2)
        real_labels = tf.one_hot(tf.ones_like(real_logits, tf.int32)[:, :, 0], depth=2)

        loss_d_fake = tf.nn.softmax_cross_entropy_with_logits(fake_labels, fake_logits)
        loss_d_real = tf.nn.softmax_cross_entropy_with_logits(real_labels, real_logits)

        return tf.reduce_mean(tf.concat([loss_d_fake, loss_d_real], axis=0))

    def p_loss(self, model_outputs: Tuple[tf.Tensor], labels=None):
        features = tf.reshape(model_outputs[3], [-1, 128])
        
        mu_real = tf.reduce_mean(features, axis=0)  # (128)
        std_real = tf.math.reduce_std(features, axis=0) # (128)

        return tf.reduce_mean(tf.abs(mu_real - 0)) + tf.reduce_mean(tf.abs(std_real - 1))

