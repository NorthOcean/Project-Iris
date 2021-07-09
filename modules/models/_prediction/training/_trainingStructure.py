"""
@Author: Conghao Wong
@Date: 2019-12-20 09:39:34
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-09 15:41:58
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import os
import re
from argparse import Namespace
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ... import base
from ..._helpmethods._helpmethods import dir_check
from .._utils import IO, Loss, Process
from ..agent._agentManager import TrainAgentManager as Agent
from ..args._argManager import TrainArgsManager as PredictionArgs
from ..dataset._datasetManager import PredictionDatasetManager
from ..dataset._trainManager import DatasetsManager
from ..vis._trajVisual import TrajVisualization

MOVE = 'MOVE'
ROTATE = 'ROTATE'
SCALE = 'SCALE'
UPSAMPLING = 'UPSAMPLING'


class Model(base.Model):

    arg_type = PredictionArgs

    def __init__(self, Args, training_structure=None, *args, **kwargs):
        super().__init__(Args, training_structure=training_structure, *args, **kwargs)

        self._preprocess_list = []
        self._preprocess_para = {
            MOVE: -1,
            ROTATE: 0,
            SCALE: 1,
            UPSAMPLING: 4,
        }
        self._preprocess_variables = {}

    @property
    def args(self) -> PredictionArgs:
        return self._args

    def set_preprocess(self, *args):
        self._preprocess_list = []
        for item in args:
            if not issubclass(type(item), str):
                raise TypeError

            if re.match('.*[Mm][Oo][Vv][Ee].*', item):
                self._preprocess_list.append(MOVE)

            elif re.match('.*[Rr][Oo][Tt].*', item):
                self._preprocess_list.append(ROTATE)

            elif re.match('.*[Ss][Cc][Aa].*', item):
                self._preprocess_list.append(SCALE)

            elif re.match('.*[Uu][Pp].*[Ss][Aa][Mm].*', item):
                self._preprocess_list.append(UPSAMPLING)
    
    def set_preprocess_parameters(self, **kwargs):
        for item in kwargs.keys():
            if not issubclass(type(item), str):
                raise TypeError

            if re.match('.*[Mm][Oo][Vv][Ee].*', item):
                self._preprocess_para[MOVE] = kwargs[item]

            elif re.match('.*[Rr][Oo][Tt].*', item):
                self._preprocess_para[ROTATE] = kwargs[item]

            elif re.match('.*[Ss][Cc][Aa].*', item):
                self._preprocess_para[SCALE] = kwargs[item]

            elif re.match('.*[Uu][Pp].*[Ss][Aa][Mm].*', item):
                self._preprocess_para[UPSAMPLING] = kwargs[item]

    def pre_process(self, tensors: Tuple[tf.Tensor],
                    training=False,
                    use_new_para_dict=True,
                    **kwargs) -> Tuple[tf.Tensor]:

        trajs = tensors[0]
        items = [MOVE, ROTATE, SCALE, UPSAMPLING]
        funcs = [Process.move, Process.rotate,
                 Process.scale, Process.upSampling]

        for item, func in zip(items, funcs):
            if item in self._preprocess_list:
                trajs, self._preprocess_variables = func(
                    trajs, self._preprocess_variables,
                    self._preprocess_para[item],
                    use_new_para_dict)

        return Process.update((trajs,), tensors)

    def post_process(self, outputs: Tuple[tf.Tensor],
                     training=False,
                     **kwargs) -> Tuple[tf.Tensor]:

        trajs = outputs[0]
        items = [MOVE, ROTATE, SCALE, UPSAMPLING]
        funcs = [Process.move_back, Process.rotate_back,
                 Process.scale_back, Process.upSampling_back]
        
        for item, func in zip(items[::-1], funcs[::-1]):
            if item in self._preprocess_list:
                trajs = func(trajs, self._preprocess_variables)
        
        return Process.update((trajs,), outputs)


class Structure(base.Structure):
    """
    Introduction
    ------------
    Basic training structure for training and test ***Prediction*** 
    models based on `tensorflow v2`.

    Usage
    -----
    When training or test new models, please subclass this class like below,
    and specific your model's inputs and groundtruths.
    ```python
    class MyPredictionStructure(Structure):
        def __init__(self, args, arg_type=Args):
            super().__init__(args, arg_type=arg_type)

            self.set_model_inputs('traj')
            self.set_model_groundtruths('gt')
    ```

    These methods must be rewritten:
    ```python
    def self.create_model(self) -> Tuple[Model, keras.optimizers.Optimizer]:
        # create new model
        pass
    ```

    Public Methods
    --------------
    ```python
    # Load args
    >>> self.load_args(current_args, load_path, arg_type=Args)

    # ----Models----
    # Load model from check point
    >>> self.load_from_checkpoint(model_path)

    # Save model
    >>> self.save_model(save_path:str)

    # Assign model's inputs and outputs
    >>> set_model_inputs(*args)
    >>> set_model_groundtruths(*args)

    # Create model
    >>> self.create_model() -> Tuple[Model, keras.optimizers.Optimizer]

    # Forward process
    >>> self.model_forward(model_inputs:Tuple[tf.Tensor], mode='test') -> Tuple[tf.Tensor]:

    # ----Datasets----
    # Load train/test/forward dataset
    >>> self.load_dataset() -> Tuple[tf.data.Dataset, tf.data.Dataset]
    >>> self.load_test_dataset() -> Tuple[tf.data.Dataset, tf.data.Dataset]
    >>> self.load_forward_dataset() -> Tuple[tf.data.Dataset, tf.data.Dataset]

    # ----Training and Test----
    # Loss Function
    >>> self.loss(outputs, labels, loss_name_list:List[str]=['L2']) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]

    # Metrics
    >>> self.metrics(outputs, labels, loss_name_list=['L2_val']) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]

    # Run gradient densest once
    >>> self.gradient_operations(model_inputs, gt, loss_move_average:tf.Variable) -> Tuple[tf.Tensor, Dict[str, tf.Tensor], tf.Tensor]

    # Entrance of train or test
    >>> self.run_train_or_test()

    # Train
    >>> self.train()

    # Run Test
    >>> self.run_test()
    >>> self.test()

    # Forward
    >>> self.forward(dataset:tf.data.Dataset, return_numpy=True) -> np.ndarray
    ```
    """

    arg_type = PredictionArgs
    agent_type = Agent
    datasetsManager_type = DatasetsManager
    
    def __init__(self, Args: List[str], *args, **kwargs):
        super().__init__(Args, *args, **kwargs)

        self.args = PredictionArgs(Args)

        self.model_inputs = ['TRAJ']
        self.model_groundtruths = ['GT']

        self.loss_list = ['ade']
        self.loss_weights = [1.0]
        self.metrics_list = ['ade', 'fde']
        self.metrics_weights = [1.0, 0.0]

    def set_model_inputs(self, *args):
        """
        Set variables to input to the model.
        Accept keywords:
        ```python
        historical_trajectory = ['traj', 'obs']
        groundtruth_trajectory = ['pred', 'gt']
        context_map = ['map']
        context_map_paras = ['para', 'map_para']
        destination = ['des', 'inten']
        ```

        :param input_names: type = `str`, accept several keywords
        """
        self.model_inputs = []
        for item in args:
            if 'traj' in item or \
                    'obs' in item:
                self.model_inputs.append('TRAJ')

            elif 'para' in item or \
                    'map_para' in item:
                self.model_inputs.append('MAPPARA')

            elif 'context' in item or \
                    'map' in item:
                self.model_inputs.append('MAP')

            elif 'des' in item or \
                    'inten' in item:
                self.model_inputs.append('DEST')

            elif 'gt' in item or \
                    'pred' in item:
                self.model_inputs.append('GT')

    def set_model_groundtruths(self, *args):
        """
        Set ground truths of the model
        Accept keywords:
        ```python
        groundtruth_trajectory = ['traj', 'pred', 'gt']
        destination = ['des', 'inten']

        :param input_names: type = `str`, accept several keywords
        """
        self.model_groundtruths = []
        for item in args:
            if 'traj' in item or \
                'gt' in item or \
                    'pred' in item:
                self.model_groundtruths.append('GT')

            elif 'des' in item or \
                    'inten' in item:
                self.model_groundtruths.append('DEST')

    def set_loss(self, *args: Tuple[Any]):
        self.loss_list = [arg for arg in args]

    def set_loss_weights(self, *args: Tuple[float]):
        self.loss_weights = [arg for arg in args]

    def set_metrics(self, *args: Tuple[Any]):
        self.metrics_list = [arg for arg in args]

    def set_metrics_weights(self, *args: Tuple[float]):
        self.metrics_weights = [arg for arg in args]

    def run_test(self):
        """
        Run test of trajectory prediction on ETH-UCY or SDD dataset.
        """
        if self.args.test:
            if self.args.test_mode == 'all':
                with open('./test_log.txt', 'a') as f:
                    f.write('-'*40 + '\n')
                    f.write(
                        '- K = {}, sigma = {} -\n'.format(self.args.K, self.args.sigma))
                for dataset in PredictionDatasetManager().sdd_test_sets if self.args.dataset == 'sdd' else PredictionDatasetManager().ethucy_testsets:
                    agents = DatasetsManager.load_dataset_files(
                        self.args, dataset)
                    self.test(agents=agents, dataset_name=dataset)
                with open('./test_log.txt', 'a') as f:
                    f.write('-'*40 + '\n')

            elif self.args.test_mode == 'mix':
                agents = []
                dataset = ''
                for dataset_c in PredictionDatasetManager().sdd_test_sets if self.args.dataset == 'sdd' else PredictionDatasetManager().ethucy_testsets:
                    agents_c = DatasetsManager.load_dataset_files(
                        self.args, dataset_c)
                    agents += agents_c
                    dataset += '{}; '.format(dataset_c)

                self.test(agents=agents, dataset_name='mix: '+dataset)

            elif self.args.test_mode == 'one':
                agents = DatasetsManager.load_dataset_files(
                    self.args, self.args.test_set)
                self.test(agents=agents, dataset_name=self.args.test_set)

    def get_inputs_from_agents(self, input_agents: List[agent_type]) -> Tuple[List[tf.Tensor], tf.Tensor]:
        """
        Get inputs for models who only takes `obs_traj` as input.

        :param input_agents: a list of input agents, type = `List[agent_type]`
        :return model_inputs: a list of traj tensor, `len(model_inputs) = 1`
        :return gt: ground truth trajs, type = `tf.Tensor`
        """
        model_inputs = [IO.get_inputs_by_type(input_agents, type_name)
                        for type_name in self.model_inputs]
        gt = [IO.get_inputs_by_type(input_agents, type_name)
              for type_name in self.model_groundtruths][0]
        # TODO accept more than one gt

        model_inputs.append(gt)
        return tuple(model_inputs)

    def load_forward_dataset(self, model_inputs: List[agent_type], **kwargs) -> tf.data.Dataset:
        """
        Load forward dataset.

        :param model_inputs: inputs to the model
        :return dataset_train: test dataset, type = `tf.data.Dataset`
        """
        model_inputs = [IO.get_inputs_by_type(model_inputs, type_name)
                        for type_name in self.model_inputs]
        return tf.data.Dataset.from_tensor_slices(tuple(model_inputs))

    def loss(self, outputs: List[tf.Tensor],
             labels: tf.Tensor,
             **kwargs) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Train loss, use ADE by default.

        :param outputs: model's outputs
        :param labels: groundtruth labels

        :return loss: sum of all single loss functions
        :return loss_dict: a dict of all losses
        """
        return Loss.Apply(self.loss_list,
                          outputs,
                          labels,
                          self.loss_weights,
                          mode='loss')

    def metrics(self, outputs: List[tf.Tensor],
                labels: tf.Tensor,
                **kwargs) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Metrics, use [ADE, FDE] by default.
        Use ADE as the validation item.

        :param outputs: model's outputs, a list of tensor
        :param labels: groundtruth labels

        :return loss: sum of all single loss functions
        :return loss_dict: a dict of all losses
        """
        return Loss.Apply(self.metrics_list,
                          outputs,
                          labels,
                          self.metrics_weights,
                          mode='m')

    def print_dataset_info(self):
        self.log_parameters(title='dataset options',
                            rotate_times=self.args.rotate,
                            add_noise=self.args.add_noise)

    def print_training_info(self):
        self.log_parameters(
            title='training options',
            model_name=self.args.model_name,
            test_set=self.args.test_set,
            batch_number=int(
                np.ceil(self.train_number / self.args.batch_size)),
            batch_size=self.args.batch_size,
            lr=self.args.lr,
            train_number=self.train_number)

    def write_test_results(self,
                           model_outputs: List[tf.Tensor],
                           agents: Dict[str, List[agent_type]],
                           **kwargs):

        testset_name = kwargs['dataset_name']

        if self.args.draw_results and (not testset_name.startswith('mix')):
            # draw results on video frames
            tv = TrajVisualization(dataset=testset_name)
            save_base_path = dir_check(self.args.log_dir) \
                if self.args.load == 'null' \
                else self.args.load

            save_format = os.path.join(dir_check(os.path.join(
                save_base_path, 'VisualTrajs')), '{}_{}.{}')

            self.logger.info('Start saving images at {}'.format(
                os.path.join(save_base_path, 'VisualTrajs')))

            for index, agent in self.log_timebar(agents, 'Saving...'):
                # write traj
                output = model_outputs[0][index].numpy()
                agents[index].pred = output
                # agents[index].calculate_loss()

                # draw as one image
                tv.draw(agents=[agent],
                        frame_name=float(agent.frame_list[self.args.obs_frames]),
                        save_path=save_format.format(
                            testset_name, index, 'jpg'),
                        show_img=False,
                        draw_distribution=self.args.draw_distribution)

                # # draw as one video
                # tv.draw_video(
                #     agent,
                #     save_path=save_format.format(index, 'avi'),
                #     draw_distribution=self.args.draw_distribution,
                # )

            self.logger.info('Prediction result images are saved at {}'.format(
                os.path.join(save_base_path, 'VisualTrajs')))
