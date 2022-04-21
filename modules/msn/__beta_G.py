"""
@Author: Conghao Wong
@Date: 2021-06-21 15:05:18
@LastEditors: Conghao Wong
@LastEditTime: 2022-04-21 11:01:19
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import modules.applications as A
import modules.models as M
import tensorflow as tf

from .__args import MSNArgs
from .__beta_D import MSNBeta_DModel


class Encoder(tf.keras.Model):
    """
    Encoder in the CVAE structure in generative Interaction Transformer
    """

    def __init__(self, Args: MSNArgs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = Args

        # context feature
        self.average_pooling = tf.keras.layers.AveragePooling2D([5, 5],
                                                             input_shape=[100, 100, 1])
        self.flatten = tf.keras.layers.Flatten()
        self.context_dense1 = tf.keras.layers.Dense((Args.obs_frames+1) * 64,
                                                 activation=tf.nn.tanh)

        # traj embedding
        self.pos_embedding = tf.keras.layers.Dense(64, tf.nn.tanh)
        self.concat = tf.keras.layers.Concatenate()

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

    def call(self, inputs: list[tf.Tensor], training=None, mask=None):
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
        t_outputs = tf.linspace(start=positions[:, -2, :],
                                stop=positions[:, -1, :],
                                num=self.args.pred_frames + 1,
                                axis=-2)[:, 1:, :]

        features, _ = self.transformer.call(t_inputs, 
                                            t_outputs,
                                            training=training)
        return features


class Generator(tf.keras.Model):
    """
    Generator (Decoder) in the CVAE structure.

    :param inputs: a list of tensors, which contains
        `inputs[0]`: features from encoder, shape = (batch, pred, 128)
        `inputs[1]`: noise vector z, shape = (batch, pred, 128)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Generator layers
        self.g1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.g2 = tf.keras.layers.Dense(2)

    def call(self, inputs: list[tf.Tensor], training=None, mask=None):
        features = inputs[0]
        z = inputs[1]

        new_feature = features + z
        g1 = self.g1(new_feature)
        g2 = self.g2(g1 + new_feature)
        return g2


class MSNBeta_GModel(M.prediction.Model):
    """
    Second stage generative Interaction Transformer model
    """

    def __init__(self, Args: MSNArgs,
                 training_structure=None,
                 *args, **kwargs):

        super().__init__(Args, training_structure, *args, **kwargs)

        self.set_preprocess('move')
        self.set_preprocess_parameters(move=0)

        self.E = Encoder(Args)
        self.G = Generator()

    def call(self, inputs: list[tf.Tensor],
             training=None, mask=None):

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
    def forward(self, model_inputs: tuple[tf.Tensor],
                training=None,
                *args, **kwargs):

        return MSNBeta_DModel.forward(self, model_inputs,
                                      training,
                                      *args, **kwargs)


class MSNBeta_G(M.prediction.Structure):
    """
    Structure for the second stage generative Interaction Transformer
    """

    def __init__(self, Args: list[str], *args, **kwargs):
        super().__init__(Args, *args, **kwargs)

        self.args = MSNArgs(Args)

        self.set_model_inputs('trajs', 'maps', 'paras', 'destinations')
        self.set_model_groundtruths('gt')

        self.set_loss('ade', 'diff', self.p_loss)
        self.set_loss_weights(0.8, 0.2, 1.0)

        self.set_metrics('ade', 'fde')
        self.set_metrics_weights(0.3, 0.7)

    def create_model(self, model_type=MSNBeta_GModel):
        model = model_type(self.args, training_structure=self)
        opt = tf.keras.optimizers.Adam(self.args.lr)
        return model, opt

    def load_forward_dataset(self, model_inputs: list[tf.Tensor], **kwargs):
        trajs = model_inputs[0]
        maps = model_inputs[1]
        paras = model_inputs[2]
        proposals = model_inputs[3]
        return tf.data.Dataset.from_tensor_slices((trajs, maps, paras, proposals))

    def p_loss(self, model_outputs: tuple[tf.Tensor], labels=None):
        features = tf.reshape(model_outputs[1], [-1, 128])

        mu_real = tf.reduce_mean(features, axis=0)  # (128)
        std_real = tf.math.reduce_std(features, axis=0)  # (128)

        return (tf.reduce_mean(tf.abs(mu_real - 0)) + 
                tf.reduce_mean(tf.abs(std_real - 1)))
