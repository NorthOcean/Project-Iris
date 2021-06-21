'''
Author: Conghao Wong
Date: 2021-01-08 09:21:29
LastEditors: Conghao Wong
LastEditTime: 2021-04-02 16:41:02
Description: file content
'''

from typing import Dict, List, Tuple

import modules.models as M
import numpy as np
import tensorflow as tf
from tensorflow import keras as keras


class LinearModel(M.base.Model):
    def __init__(self, Args, training_structure=None, *args, **kwargs):
        super().__init__(Args, training_structure=training_structure, *args, **kwargs)

        if self.args.diff_weights == 0:
            P = tf.linalg.diag(tf.ones(self.args.obs_frames))
        else:
            P = tf.linalg.diag(tf.nn.softmax([(i+1)**self.args.diff_weights for i in range(self.args.obs_frames)]))

        self.x = tf.cast(np.arange(self.args.obs_frames), tf.float32)
        self.x_p = tf.cast(np.arange(self.args.pred_frames) + self.args.obs_frames, tf.float32)
        A = tf.transpose(tf.stack([
            tf.ones([self.args.obs_frames]),
            self.x
        ]))
        self.A_p = tf.transpose(tf.stack([
            tf.ones([self.args.pred_frames]),
            self.x_p
        ]))
        self.W = tf.linalg.inv(tf.transpose(A) @ P @ A) @ tf.transpose(A) @ P

    def call(self, inputs, training=None, mask=None):
        inputs = inputs[0]
        
        x = inputs[:, :, 0:1]
        y = inputs[:, :, 1:2]
        
        Bx = self.W @ x
        By = self.W @ y

        results = tf.stack([
            self.A_p @ Bx,
            self.A_p @ By,
        ])

        results = tf.transpose(results[:, :, :, 0], [1, 2, 0])
        return results
        

class Linear(M.prediction.Structure):
    def __init__(self, args, arg_type=M.prediction.TrainArgs):
        args.load = 'linear'
        super().__init__(args, arg_type=arg_type)
        self.x_obs = np.array([i for i in range(self.args.obs_frames)])/(self.args.obs_frames - 1)

    def load_args(self, current_args, load_path, arg_type=M.prediction.TrainArgs):
        return self.args
        
    def load_from_checkpoint(self, model_path=None):
        self._model, _ = self.create_model()
        return self.model

    def create_model(self):
        return [LinearModel(self.args), None]

    def load_forward_dataset(self, model_inputs:List[M.prediction.TrainAgentManager], **kwargs) -> tf.data.Dataset:
        """
        Load forward dataset.

        :return dataset_train: test dataset, type = `tf.data.Dataset`
        :return dataset_labels: a list of datasets' names where inputs come from
        """
        agents = model_inputs
        trajs = []
        for agent in agents:
            traj = agent.traj
            if len(traj) < self.args.obs_frames:
                length = len(traj)
                x_p = np.array([i for i in range(length)])/(length - 1)
                traj_fix = np.column_stack([
                    np.interp(self.x_obs, x_p, traj.T[0]),
                    np.interp(self.x_obs, x_p, traj.T[1]),
                ])
                trajs.append(traj_fix)
            else:
                trajs.append(traj)

        trajs = tf.cast(trajs, tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices((trajs))
        return dataset
