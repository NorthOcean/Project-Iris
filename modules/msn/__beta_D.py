"""
@Author: Conghao Wong
@Date: 2021-05-07 09:12:57
@LastEditors: Conghao Wong
@LastEditTime: 2022-04-21 11:01:12
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import modules.applications as A
import modules.models as M
import tensorflow as tf

from .__args import MSNArgs


class MSNBeta_DModel(M.prediction.Model):
    """
    Second stage deterministic model, i.e., the Interaction Transformer
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
        self.context_dense1 = tf.keras.layers.Dense((self.args.obs_frames+1) * 64,
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
                                         pe_target=Args.pred_frames)

    def call(self, inputs: list[tf.Tensor],
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
        t_outputs = tf.linspace(start=positions[:, -2, :],
                                stop=positions[:, -1, :],
                                num=self.args.pred_frames + 1,
                                axis=-2)[:, 1:, :]

        predictions, _ = self.transformer.call(t_inputs, 
                                              t_outputs,
                                              training=training)

        return predictions

    # @tf.function
    def forward(self, model_inputs: list[tf.Tensor],
                training=None,
                *args, **kwargs):
        """
        Run a forward implementation.

        :param model_inputs: input tensor (or a list of tensors)
        :param mode: choose forward type, can be `'test'` or `'train'`
        :return output: model's output. type=`list[tf.Tensor]`
        """
        model_inputs_processed = self.pre_process(model_inputs, training)
        destination_processed = self.pre_process([model_inputs[-1]],
                                                 training,
                                                 use_new_para_dict=False)

        model_inputs_processed = [model_inputs_processed[0],    # traj
                                  model_inputs_processed[1],    # map
                                  model_inputs_processed[2],    # map para
                                  destination_processed[0]]     # destination

        # use `self.call()` to debug
        output = self(model_inputs_processed, training)

        if not (type(output) == list or type(output) == tuple):
            output = [output]

        return self.post_process(output, training, model_inputs=model_inputs)


class MSNBeta_D(M.prediction.Structure):
    """
    Structure for the second stage deterministic Interaction Transformer
    """

    def __init__(self, Args: list[str], *args, **kwargs):
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
        opt = tf.keras.optimizers.Adam(self.args.lr)
        return model, opt

    def load_forward_dataset(self, model_inputs: list[tf.Tensor], **kwargs):
        trajs = model_inputs[0]
        maps = model_inputs[1]
        paras = model_inputs[2]
        proposals = model_inputs[3]
        return tf.data.Dataset.from_tensor_slices((trajs, maps, paras, proposals))
