"""
@Author: Conghao Wong
@Date: 2021-01-08 15:08:07
@LastEditors: Conghao Wong
@LastEditTime: 2021-08-04 14:50:15
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import re
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf

from .. import base
from .agent import PredictionAgent as Agent
from tqdm import tqdm


class Loss():
    """
    Loss
    ----
    Loss functions and metrics used in trajectory prediction models.

    Methods
    -------
    ```python
    # Average Displacement Error, ADE (or minADE for multiple-prediction models)
    >>> Loss.ADE(pred, GT) -> tf.Tensor

    # Final Displacement Error, FDE (or minFDE for multiple-prediction models)
    >>> Loss.FDE(pred, GT) -> tf.Tensor
    ```
    """
    @classmethod
    def Apply(cls, loss_list: List[Union[str, Any]],
              model_outputs: List[tf.Tensor],
              labels: tf.Tensor,
              loss_weights: List[float] = None,
              mode='loss',
              *args, **kwargs) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:

        loss_dict = {}
        for loss in loss_list:
            if type(loss) == str:
                if re.match('[Aa][Dd][Ee]', loss):
                    loss_dict['ADE({})'.format(mode)] = cls.ADE(
                        model_outputs[0], labels)

                elif re.match('[Ff][Dd][Ee]', loss):
                    loss_dict['FDE({})'.format(mode)] = cls.FDE(
                        model_outputs[0], labels)

                elif re.match('[Dd][Ii][Ff]', loss):
                    order = 2 if not 'diff_order' in kwargs.keys() \
                        else kwargs['diff_order']
                    weights = [min(1.0, 5 * 10 ** -o) for o in range(order+1)] \
                        if not 'diff_weights' in kwargs.keys() \
                        else kwargs['diff_weights']
                    loss_dict['Diff'] = tf.reduce_sum(
                        tf.stack(weights) *
                        tf.stack(cls.diff(model_outputs[0], labels, order)))

            elif callable(loss):
                loss_dict[loss.__name__ + '({})'.format(mode)] = loss(model_outputs, labels,
                                                                      *args, **kwargs)

        if loss_weights is None:
            loss_weights = tf.ones(len(loss_dict))

        if len(loss_weights) != len(loss_dict):
            raise ValueError('Incorrect weights')

        summary = tf.matmul(tf.expand_dims(list(loss_dict.values()), 0),
                            tf.expand_dims(loss_weights, 1))
        return summary, loss_dict

    @staticmethod
    @tf.function
    def ADE(pred, GT) -> tf.Tensor:
        """
        Calculate `ADE` or `minADE`.

        :param pred: pred traj, shape = `[batch, pred, 2]`
        :param GT: ground truth future traj, shape = `[batch, pred, 2]`
        :return loss_ade:
            Return `ADE` when input_shape = [batch, pred_frames, 2];
            Return `minADE` when input_shape = [batch, K, pred_frames, 2].
        """

        pred = tf.cast(pred, tf.float32)
        GT = tf.cast(GT, tf.float32)

        if len(pred.shape) == 3:  # [batch, K, pred, 2]
            pred = pred[:, tf.newaxis, :, :]

        all_ade = tf.reduce_mean(tf.linalg.norm(
            pred - GT[:, tf.newaxis, :, :], ord=2, axis=-1), axis=-1)
        best_ade = tf.reduce_min(all_ade, axis=1)
        return tf.reduce_mean(best_ade)

    @classmethod
    def FDE(cls, pred, GT) -> tf.Tensor:
        """
        Calculate `FDE` or `minFDE`

        :param pred: pred traj, shape = `[batch, pred, 2]`
        :param GT: ground truth future traj, shape = `[batch, pred, 2]`
        :return fde:
            Return `FDE` when input_shape = [batch, pred_frames, 2];
            Return `minFDE` when input_shape = [batch, K, pred_frames, 2].
        """
        pred = tf.cast(pred, tf.float32)
        GT = tf.cast(GT, tf.float32)

        t = pred.shape[-2]
        f = tf.gather(pred, [t-1], axis=-2)
        f_gt = tf.gather(GT, [t-1], axis=-2)
        return cls.ADE(f, f_gt)

    @staticmethod
    @tf.function
    def context(pred, maps, paras, pred_bias=None) -> tf.Tensor:
        """
        Context loss for Energy Maps by `tensorflow`.

        :param pred: pred traj, shape = `[batch, pred, 2] or [batch, K, pred, 2]`
        :param maps: energy map, shape = `[batch, h, w]`
        :param paras: mapping function paras [[Wx, Wy], [bx, by]]
        :param pred_bias: bias for prediction, shape = `[batch, 2]`
        :return loss_context: context loss
        """

        if type(pred_bias) == type(None):
            pred_bias = tf.zeros([pred.shape[0], 2], dtype=tf.float32)
        if len(pred_bias.shape) == 2:
            pred_bias = tf.expand_dims(pred_bias, axis=1)

        if len(pred.shape) == 3:
            W = tf.reshape(paras[:, 0, :], [-1, 1, 2])
            b = tf.reshape(paras[:, 1, :], [-1, 1, 2])
        elif len(pred.shape) == 4:
            W = tf.expand_dims(tf.reshape(paras[:, 0, :], [-1, 1, 2]), axis=1)
            b = tf.expand_dims(tf.reshape(paras[:, 1, :], [-1, 1, 2]), axis=1)

        # from real positions to grid positions, shape = [batch, pred, 2]
        pred_grid = tf.cast((pred - b) * W, tf.int32)

        if len(pred.shape) == 3:
            center_grid = tf.cast((pred_bias - b) * W, tf.int32)
        elif len(pred.shape) == 4:
            center_grid = tf.cast(
                (tf.expand_dims(pred_bias, axis=1) - b) * W, tf.int32)
        final_grid = pred_grid - center_grid + \
            tf.cast(maps.shape[-1]/2, tf.int32)

        final_grid = tf.maximum(final_grid, tf.zeros_like(final_grid))
        final_grid = tf.minimum(
            final_grid, (maps.shape[-1]-1) * tf.ones_like(final_grid))

        if len(pred.shape) == 3:
            sel = tf.gather_nd(maps, final_grid, batch_dims=1)
        elif len(pred.shape) == 4:
            sel = tf.gather_nd(tf.repeat(tf.expand_dims(
                maps, axis=1), pred.shape[1], axis=1), final_grid, batch_dims=2)
        context_loss_mean = tf.reduce_mean(sel)
        return context_loss_mean

    @staticmethod
    def diff(pred, GT, ordd=2) -> List[tf.Tensor]:
        """
        loss_functions with diference limit

        :param pred: pred traj, shape = `[(K,) batch, pred, 2]`
        :param GT: ground truth future traj, shape = `[batch, pred, 2]`
        :return loss: a list of Tensor, `len(loss) = ord + 1`
        """
        pred = tf.cast(pred, tf.float32)
        GT = tf.cast(GT, tf.float32)

        pred_diff = difference(pred, ordd=ordd)
        GT_diff = difference(GT, ordd=ordd)

        loss = []
        for pred_, gt_ in zip(pred_diff, GT_diff):
            loss.append(Loss.ADE(pred_, gt_))

        return loss


class Process():
    @staticmethod
    def move(trajs: tf.Tensor,
             para_dict: Dict[str, tf.Tensor],
             ref: int = -1,
             use_new_para_dict=True) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Move a specific point to (0, 0) according to the reference time step.
        Default reference time step is the last obsetvation step.

        :param trajs: observations, shape = `[(batch,) obs, 2]`
        :param ref: reference point, default is `-1`

        :return traj_moved: moved trajectories
        :return para_dict: a dict of used parameters
        """
        if use_new_para_dict:
            ref_point = trajs[:, ref, :] if len(trajs.shape) == 3\
                else trajs[ref, :]
            
            # shape is [batch, 1, 2] or [1, 2]
            ref_point = tf.expand_dims(ref_point, -2)
            para_dict['MOVE'] = ref_point

        else:
            ref_point = para_dict['MOVE']

        if len(trajs.shape) == 4:   # (batch, K, n, 2)
            ref_point = ref_point[:, tf.newaxis, :, :]

        traj_moved = trajs - ref_point

        return traj_moved, para_dict

    @staticmethod
    def move_back(trajs: tf.Tensor,
                  para_dict: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Move trajectories back to their original positions.

        :param trajs: trajectories moved to (0, 0) with reference point, shape = `[(batch,) (K,) pred, 2]`
        :param para_dict: a dict of used parameters, which contains `'ref_point': tf.Tensor`
        
        :return traj_moved: moved trajectories
        """
        try:
            ref_point = para_dict['MOVE']  # shape = [(batch,) 1, 2]
            if len(ref_point.shape) == len(trajs.shape):
                traj_moved = trajs + ref_point
            else:   # [(batch,) K, pred, 2]
                traj_moved = trajs + tf.expand_dims(ref_point, -3)
            return traj_moved

        except:
            return trajs

    @staticmethod
    def rotate(trajs: tf.Tensor,
               para_dict: Dict[str, tf.Tensor],
               ref: int = 0,
               use_new_para_dict=True) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Rotate trajectories to the referce angle.

        :param trajs: observations, shape = `[(batch,) obs, 2]`
        :param ref: reference angle, default is `0`

        :return traj_rotated: moved trajectories
        :return para_dict: a dict of used parameters, `'rotate_angle': tf.Tensor`
        """
        if use_new_para_dict:
            vector_x = (trajs[:, -1, 0] - trajs[:, 0, 0]) if len(trajs.shape) == 3 else (
                trajs[-1, 0] - trajs[0, 0])  # shape is [batch] or []
            vector_y = (trajs[:, -1, 1] - trajs[:, 0, 1]) if len(trajs.shape) == 3 else (
                trajs[-1, 1] - trajs[0, 1])  # shape is [batch] or []

            main_angle = tf.atan((vector_y + 0.01)/(vector_x + 0.01))
            angle = ref - main_angle
            para_dict['ROTATE'] = angle

        else:
            angle = para_dict['ROTATE']

        rotate_matrix = tf.stack([
            [tf.cos(angle), tf.sin(angle)],
            [-tf.sin(angle), tf.cos(angle)]
        ])  # shape = [2, 2, batch] or [2, 2]

        if len(trajs.shape) == 3:
            rotate_matrix = tf.transpose(rotate_matrix, [2, 0, 1])

        traj_rotated = trajs @ rotate_matrix

        return traj_rotated, para_dict

    @staticmethod
    def rotate_back(trajs: tf.Tensor,
                    para_dict: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Rotate trajectories back to their original angles.

        :param trajs: trajectories, shape = `[(batch, ) pred, 2]`
        :param para_dict: a dict of used parameters, `'rotate_matrix': tf.Tensor`
        
        :return traj_rotated: rotated trajectories
        """
        angle = -1 * para_dict['ROTATE']
        rotate_matrix = tf.stack([
            [tf.cos(angle), tf.sin(angle)],
            [-tf.sin(angle), tf.cos(angle)]
        ])
        if len(trajs.shape) == 3:
            rotate_matrix = tf.transpose(rotate_matrix, [2, 0, 1])

        traj_rotated = trajs @ rotate_matrix
        return traj_rotated

    @staticmethod
    def scale(trajs: tf.Tensor,
              para_dict: Dict[str, tf.Tensor],
              ref: float = 1,
              use_new_para_dict=True) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Scale trajectories' direction vector into (x, y), where |x| <= 1, |y| <= 1.
        Reference point when scale is the `last` observation point.

        :param trajs: input trajectories, shape = `[(batch,) obs, 2]`
        :param ref: reference length, default is `1`
        :return traj_scaled: scaled trajectories
        :return para_dict: a dict of used parameters, contains `scale:tf.Tensor`
        """
        change_flag = False
        if len(trajs.shape) == 2:
            trajs = tf.expand_dims(trajs, 0)    # change into [batch, obs, 2]
            change_flag = True

        if use_new_para_dict:
            x = trajs[:, :, 0]  # shape = [batch, obs]
            y = trajs[:, :, 1]

            scale = tf.linalg.norm(
                trajs[:, -1, :] - trajs[:, 0, :], axis=-1)  # [batch]
            scale = tf.maximum(0.05, scale)
            scale = tf.expand_dims(scale, -1)   # [batch, 1]
            para_dict['SCALE'] = scale

        else:
            scale = para_dict['SCALE']

        # shape = [batch, obs]
        new_x = (x - tf.expand_dims(x[:, -1], -1)) / \
            scale + tf.expand_dims(x[:, -1], -1)
        new_y = (y - tf.expand_dims(y[:, -1], -1)) / \
            scale + tf.expand_dims(y[:, -1], -1)

        traj_scaled = tf.stack([new_x, new_y])  # shape = [2, batch, obs]
        traj_scaled = tf.transpose(traj_scaled, [1, 2, 0])

        if change_flag:
            traj_scaled = traj_scaled[0, :, :]

        return traj_scaled, para_dict

    @staticmethod
    def scale_back(trajs: tf.Tensor,
                   para_dict: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Scale trajectories back to their original.
        Reference point is the `first` prediction point.

        :param trajs: trajectories, shape = `[(batch,) (K,) pred, 2]`
        :param para_dict: a dict of used parameters, contains `scale:tf.Tensor`
        :return traj_scaled: scaled trajectories
        """
        original_dim = len(trajs.shape)
        if original_dim < 4:
            for repeat in range(4 - original_dim):
                # change into [batch, K, pred, 2]
                trajs = tf.expand_dims(trajs, -3)

        x = trajs[:, :, :, 0]   # [batch, K, pred]
        y = trajs[:, :, :, 1]

        scale = para_dict['SCALE']  # [batch, 1]
        scale = tf.expand_dims(scale, 1)    # [batch, 1, 1]

        # shape = [batch, K, obs]
        new_x = (x - tf.expand_dims(x[:, :, 0], -1)) * \
            scale + tf.expand_dims(x[:, :, 0], -1)
        new_y = (y - tf.expand_dims(y[:, :, 0], -1)) * \
            scale + tf.expand_dims(y[:, :, 0], -1)

        traj_scaled = tf.stack([new_x, new_y])  # [2, batch, K, pred]
        traj_scaled = tf.transpose(traj_scaled, [1, 2, 3, 0])

        if original_dim < 4:
            for repeat in range(4 - original_dim):
                traj_scaled = traj_scaled[0]

        return traj_scaled

    @staticmethod
    def upSampling(trajs: tf.Tensor,
                   para_dict: Dict[str, tf.Tensor],
                   sample_time: int,
                   use_new_para_dict=True):

        if use_new_para_dict:
            para_dict['UPSAMPLING'] = sample_time
        else:
            sample_time = para_dict['UPSAMPLING']

        original_number = trajs.shape[-2]
        sample_number = sample_time * original_number

        if len(trajs.shape) == 3:   # (batch, n, 2)
            return tf.image.resize(trajs[:, :, :, tf.newaxis], [sample_number, 2])[:, :, :, 0], para_dict

        elif len(trajs.shape) == 4:   # (batch, K, n, 2)
            K = trajs.shape[1]
            results = []
            for k in range(K):
                results.append(tf.image.resize(
                    trajs[:, k, :, :, tf.newaxis],
                    [sample_number, 2])[:, :, :, 0])

            return tf.transpose(tf.stack(results), [1, 0, 2, 3]), para_dict

    @staticmethod
    def upSampling_back(trajs: tf.Tensor,
                        para_dict: Dict[str, tf.Tensor]):
        sample_time = para_dict['UPSAMPLING']
        sample_number = trajs.shape[-2]
        original_number = sample_number // sample_time
        original_index = tf.range(original_number) * sample_time

        return tf.gather(trajs, original_index, axis=-2)

    @staticmethod
    def update(new: Union[tuple, list],
               old: Union[tuple, list]) -> tuple:

        if type(old) == list:
            old = tuple(old)
        if type(new) == list:
            new = tuple(new)

        if len(new) < len(old):
            return new + old[len(new):]
        else:
            return new


class IO(base.BaseObject):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_inputs_by_type(cls, input_agents: List[Agent], type_name: str) -> tf.Tensor:
        if type_name == 'TRAJ':
            call = cls._get_obs_traj
        elif type_name == 'MAP':
            call = cls._get_context_map
        elif type_name == 'MAPPARA':
            call = cls._get_context_map_paras
        elif type_name == 'DEST':
            call = cls._get_dest_traj
        elif type_name == 'GT':
            call = cls._get_gt_traj
        return call(input_agents)

    @classmethod
    def _get_obs_traj(cls, input_agents: List[Agent]) -> tf.Tensor:
        """
        Get observed trajectories from agents.

        :param input_agents: a list of input agents, type = `List[Agent]`
        :return inputs: a Tensor of observed trajectories
        """
        inputs = []
        for agent in tqdm(input_agents, 'Prepare trajectories...'):
            inputs.append(agent.traj)
        return tf.cast(inputs, tf.float32)

    @classmethod
    def _get_gt_traj(cls, input_agents: List[Agent], destination=False) -> tf.Tensor:
        """
        Get groundtruth trajectories from agents.

        :param input_agents: a list of input agents, type = `List[Agent]`
        :return inputs: a Tensor of gt trajectories
        """
        inputs = []
        for agent in tqdm(input_agents, 'Prepare groundtruth...'):
            if destination:
                inputs.append(np.expand_dims(agent.groundtruth[-1], 0))
            else:
                inputs.append(agent.groundtruth)

        return tf.cast(inputs, tf.float32)

    @classmethod
    def _get_dest_traj(cls, input_agents: List[Agent]) -> tf.Tensor:
        return cls._get_gt_traj(input_agents, destination=True)

    @classmethod
    def _get_context_map(cls, input_agents: List[Agent]) -> tf.Tensor:
        """
        Get context map from agents.

        :param input_agents: a list of input agents, type = `List[Agent]`
        :return inputs: a Tensor of maps
        """
        inputs = []
        for agent in tqdm(input_agents, 'Prepare maps...'):
            inputs.append(agent.Map)
        return tf.cast(inputs, tf.float32)

    @classmethod
    def _get_context_map_paras(cls, input_agents: List[Agent]) -> tf.Tensor:
        """
        Get parameters of context map from agents.

        :param input_agents: a list of input agents, type = `List[Agent]`
        :return inputs: a Tensor of map paras
        """
        inputs = []
        for agent in tqdm(input_agents, 'Prepare maps...'):
            inputs.append(agent.real2grid)
        return tf.cast(inputs, tf.float32)


def difference(trajs: tf.Tensor, direction='back', ordd=1) -> List[tf.Tensor]:
    """
    :param trajs: trajectories, shape = `[(K,) batch, pred, 2]`
    :param direction: string, canbe `'back'` or `'forward'`
    :param ord: repeat times

    :return result: results list, `len(results) = ord + 1`
    """
    outputs = [trajs]
    for repeat in range(ordd):
        outputs_current = \
            outputs[-1][:, :, 1:, :] - outputs[-1][:, :, :-1, :] if len(trajs.shape) == 4 else \
            outputs[-1][:, 1:, :] - outputs[-1][:, :-1, :]
        outputs.append(outputs_current)
    return outputs


def calculate_cosine(vec1: np.ndarray,
                     vec2: np.ndarray):

    length1 = np.linalg.norm(vec1, axis=-1)
    length2 = np.linalg.norm(vec2, axis=-1)

    return (np.sum(vec1 * vec2, axis=-1) + 0.0001) / ((length1 * length2) + 0.0001)


def calculate_length(vec1):
    return np.linalg.norm(vec1, axis=-1)


def activation(x: np.ndarray, a=1, b=1):
    return np.less_equal(x, 0) * a * x + np.greater(x, 0) * b * x
