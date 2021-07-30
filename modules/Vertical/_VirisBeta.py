"""
@Author: Conghao Wong
@Date: 2021-07-08 15:45:53
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-30 16:55:08
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from .. import applications as A
from .. import models as M
from ._args import VArgs
from ._layers import ContextEncoding, FFTlayer, IFFTlayer, TrajEncoding
from ._utils import Utils as U


class VIrisBetaModel(M.prediction.Model):
    """
    Second stage `Vertical` model.

    It can be applyed on both first stage generative `Vertical-G`
    and first stage deterministic `Vertical-D`
    """

    def __init__(self, Args: VArgs,
                 points: int,
                 asSecondStage=False,
                 p_index: str = None,
                 training_structure=None,
                 *args, **kwargs):
        """
        Init a second stage `Vertical` model

        :param Args: args used in this model
        :param points: number of predicted steps accept
        :param asSecondStage: controls if use the model as the second stage `Vertical`
        :param p_index: (Only works when `asSecondStage is not None`) 
            time step of predicted points
        """

        super().__init__(Args, training_structure,
                         *args, **kwargs)

        # Preprocess
        self.set_preprocess('move')
        self.set_preprocess_parameters(move=0)

        # Args
        self.n_pred = points
        self.asSecondStage = asSecondStage

        if self.asSecondStage and p_index:
            pi = [int(i) for i in p_index.split('_')]
            self.points_index = tf.cast(pi, tf.float32)

        # Layers
        self.concat = keras.layers.Concatenate(axis=-1)

        self.fft = FFTlayer()

        self.te = TrajEncoding(units=64,
                               activation=tf.nn.tanh,
                               useFFT=True)

        self.ce = ContextEncoding(units=64,
                                  output_channels=self.args.obs_frames,
                                  activation=tf.nn.tanh)

        self.transformer = A.Transformer(num_layers=4,
                                         d_model=128,
                                         num_heads=8,
                                         dff=512,
                                         input_vocab_size=None,
                                         target_vocab_size=4,
                                         pe_input=Args.obs_frames,
                                         pe_target=Args.obs_frames + Args.pred_frames,
                                         include_top=True)

        self.decoder = IFFTlayer()

    def call(self, inputs: List[tf.Tensor],
             points: tf.Tensor,
             points_index: tf.Tensor,
             training=None, mask=None) -> tf.Tensor:
        """
        Run the second stage `Vertical` model
        
        :param inputs: a list of tensors, which includes `trajs` and `maps`
            - trajs, shape = `(batch, obs, 2)`
            - maps, shape = `(batch, a, a)`
            
        :param points: pred points, shape = `(batch, n, 2)`
        :param points_index: pred time steps, shape = `(n)`
        :param training: controls run as the training mode or the test mode

        :return predictions: predictions, shape = `(batch, pred, 2)`
        """

        # unpack inputs
        trajs, maps = inputs[:2]

        traj_feature = self.te.call(trajs)
        context_feature = self.ce.call(maps)

        # transformer inputs shape = (batch, obs, 128)
        t_inputs = self.concat([traj_feature, context_feature])

        # transformer target shape = (batch, obs+pred, 4)
        points_index = tf.concat([[-1], points_index], axis=0)
        points = tf.concat([trajs[:, -1:, :], points], axis=1)

        # add the last obs point to finish linear interpolation
        linear_pred = U.LinearInterpolation(points_index, points)
        traj = tf.concat([trajs, linear_pred], axis=-2)
        lfft_r, lfft_i = self.fft.call(traj)
        t_outputs = self.concat([lfft_r, lfft_i])

        # transformer output shape = (batch, obs+pred, 4)
        p_fft, _ = self.transformer.call(t_inputs, 
                                         t_outputs,
                                         training=training)

        # decode
        p = self.decoder.call(real=p_fft[:, :, :2],
                              imag=p_fft[:, :, 2:])

        return p[:, self.args.obs_frames:, :]

    def call_secondStage(self, inputs: List[tf.Tensor],
                         points: tf.Tensor,
                         points_index: tf.Tensor,
                         training=None, mask=None):
        """
        Call as the second stage model.
        Do not call this method if the model is not trained.
        
        :param inputs: a list of trajs, maps
        :param points: pred points, shape = `(batch, K, n, 2)`
        :param points_index: pred index, shape = `(n)`
        """
        
        # unpack inputs
        K = points.shape[1]
        trajs, maps = inputs[:2]

        traj_feature = self.te.call(trajs)
        context_feature = self.ce.call(maps)

        # transformer inputs shape = (batch, obs, 128)
        t_inputs = self.concat([traj_feature, context_feature])
        t_inputs_index = tf.range([t_inputs.shape[0]])
        t_inputs_index = tf.repeat(t_inputs_index, K, axis=0)

        # transformer target shape = (batch, obs+pred, 4)
        
        points_index = tf.concat([[-1], points_index], axis=0)
        trajs = tf.repeat(trajs[:, tf.newaxis], K, axis=1)
        points = tf.concat([trajs[:, :, -1:, :], points], axis=-2)

        # add the last obs point to finish linear interpolation
        linear_pred = U.LinearInterpolation(points_index, points)
        traj = tf.concat([trajs, linear_pred], axis=-2)
        lfft_r, lfft_i = self.fft.call(traj)
        t_outputs = self.concat([lfft_r, lfft_i])
        t_outputs = tf.reshape(t_outputs, [-1]+[s for s in t_outputs.shape[-2:]])

        # prepare new inputs into transformer
        # new batch_size (total) is batch*K
        ds = tf.data.Dataset.from_tensor_slices((t_inputs_index, t_outputs))
        ds = ds.batch(self.args.batch_size)
        
        predictions = []
        for t_ii, t_o in tqdm(ds):
            t_i = tf.gather(t_inputs, t_ii, axis=0)
            p_fft, _ = self.transformer.call(t_i, t_o, training=training)

            # decode
            p = self.decoder.call(real=p_fft[:, :, :2],
                                  imag=p_fft[:, :, 2:])
            
            predictions.append(p[:, self.args.obs_frames:, :])
        
        p = tf.concat(predictions, axis=0)
        p = tf.reshape(p, [-1, K, self.args.pred_frames, 2])
        return p

    def forward(self, model_inputs: List[tf.Tensor],
                training=None,
                *args, **kwargs):

        model_inputs_processed = self.pre_process(model_inputs, training)
        destination_processed = self.pre_process([model_inputs[-1]],
                                                 training,
                                                 use_new_para_dict=False)

        # only when training the single model
        if not self.asSecondStage:
            gt_processed = destination_processed[0]

            index = np.random.choice(np.arange(self.args.pred_frames-1),
                                     self.n_pred-1)
            index = tf.concat([np.sort(index),
                               [self.args.pred_frames-1]], axis=0)

            points = tf.gather(gt_processed, index, axis=1)
            index = tf.cast(index, tf.float32)

            outputs = self.call(model_inputs_processed,
                                points=points,
                                points_index=index,
                                training=True)

        # use as the second stage model
        else:
            outputs = self.call_secondStage(model_inputs_processed,
                                            points=destination_processed[0],
                                            points_index=self.points_index,
                                            training=None)

        if not type(outputs) in [list, tuple]:
            outputs = [outputs]

        return self.post_process(outputs, training, model_inputs=model_inputs)


class VIrisBeta(M.prediction.Structure):
    """
    Training structure for the second stage `Vertical`
    """
    
    def __init__(self, Args: List[str], *args, **kwargs):
        super().__init__(Args, *args, **kwargs)

        self.args = VArgs(Args)

        self.important_args += ['points']

        self.set_model_inputs('trajs', 'maps', 'paras', 'gt')
        self.set_model_groundtruths('gt')

        self.set_loss('ade', 'diff')
        self.set_loss_weights(0.8, 0.2)

        self.set_metrics('ade', 'fde')
        self.set_metrics_weights(1.0, 0.0)

    def create_model(self, *args, **kwargs):
        model = VIrisBetaModel(self.args,
                               points=self.args.points,
                               training_structure=self,
                               *args, **kwargs)

        opt = keras.optimizers.Adam(self.args.lr)
        return model, opt

    def load_forward_dataset(self, model_inputs: Tuple[tf.Tensor], **kwargs):
        trajs = model_inputs[0]
        maps = model_inputs[1]
        proposals = model_inputs[-1]
        return tf.data.Dataset.from_tensor_slices((trajs, maps, proposals))

    def print_test_result_info(self, loss_dict, dataset_name, **kwargs):
        self.print_parameters(title='rest results',
                              **dict({'dataset': dataset_name}, **loss_dict))

        self.log('Results: {}, {}, {}.'.format(
            self.args.load,
            dataset_name,
            loss_dict
        ))
