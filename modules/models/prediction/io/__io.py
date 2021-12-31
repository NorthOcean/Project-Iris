"""
@Author: Conghao Wong
@Date: 2021-12-31 09:05:00
@LastEditors: Conghao Wong
@LastEditTime: 2021-12-31 09:59:48
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import List

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ..__agent import PredictionAgent as Agent


def get_inputs_by_type(input_agents: List[Agent],
                       type_name: str) -> tf.Tensor:
    """
    Get model inputs from a list of `Agent`-like objects.

    :param input_agents: a list of `Agent` objects or their subclass-objects
    :param type_name: inputs names, accept `'TRAJ'`, `'MAP'`, `'MAPPARA'`,
        `'DEST'`, and `'GT'`
    :return inputs: a tensor of stacked inputs
    """
    if type_name == 'TRAJ':
        call = _get_obs_traj
    elif type_name == 'MAP':
        call = _get_context_map
    elif type_name == 'MAPPARA':
        call = _get_context_map_paras
    elif type_name == 'DEST':
        call = _get_dest_traj
    elif type_name == 'GT':
        call = _get_gt_traj
    return call(input_agents)


def _get_obs_traj(input_agents: List[Agent]) -> tf.Tensor:
    """
    Get observed trajectories from agents.

    :param input_agents: a list of input agents, type = `List[Agent]`
    :return inputs: a Tensor of observed trajectories
    """
    inputs = []
    for agent in tqdm(input_agents, 'Prepare trajectories...'):
        inputs.append(agent.traj)
    return tf.cast(inputs, tf.float32)


def _get_gt_traj(input_agents: List[Agent],
                 destination=False) -> tf.Tensor:
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


def _get_dest_traj(input_agents: List[Agent]) -> tf.Tensor:
    return _get_gt_traj(input_agents, destination=True)


def _get_context_map(input_agents: List[Agent]) -> tf.Tensor:
    """
    Get context map from agents.

    :param input_agents: a list of input agents, type = `List[Agent]`
    :return inputs: a Tensor of maps
    """
    inputs = []
    for agent in tqdm(input_agents, 'Prepare maps...'):
        inputs.append(agent.Map)
    return tf.cast(inputs, tf.float32)


def _get_context_map_paras(input_agents: List[Agent]) -> tf.Tensor:
    """
    Get parameters of context map from agents.

    :param input_agents: a list of input agents, type = `List[Agent]`
    :return inputs: a Tensor of map paras
    """
    inputs = []
    for agent in tqdm(input_agents, 'Prepare maps...'):
        inputs.append(agent.real2grid)
    return tf.cast(inputs, tf.float32)
