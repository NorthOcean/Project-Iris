"""
@Author: Conghao Wong
@Date: 2019-12-20 09:39:34
@LastEditors: Conghao Wong
@LastEditTime: 2021-09-10 09:45:33
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import os
import re
from argparse import Namespace
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .. import base
from ..helpmethods import dir_check
from .agent import PredictionAgent as Agent
from .args import PredictionArgs
from .dataset._trainManager import DatasetsManager
from .utils import IO, Loss, Process
from .vis import TrajVisualization

MOVE = 'MOVE'
ROTATE = 'ROTATE'
SCALE = 'SCALE'
UPSAMPLING = 'UPSAMPLING'


class Model(base.Model):
    """
    Model
    -----
    An expanded class for `base.Model`.

    Usage
    -----
    When training or test new models, please subclass this class, and clarify
    model layers used in your model.
    ```python
    class MyModel(Model):
        def __init__(self, Args, training_structure, *args, **kwargs):
            super().__init__(Args, training_structure, *args, **kwargs)

            self.fc = keras.layers.Dense(64, tf.nn.relu)
            self.fc1 = keras.layers.Dense(2)
    ```

    Then define your model's pipeline in `call` method:
    ```python
        def call(self, inputs, training=None, mask=None):
            y = self.fc(inputs)
            return self.fc1(y)
    ```

    Public Methods
    --------------
    ```python
    # forward model with pre-process and post-process
    (method) forward: (self: Model, model_inputs: List[Tensor], training=None, *args, **kwargs) -> List[Tensor]

    # Pre/Post-processes
    (method) pre_process: (self: Model, tensors: List[Tensor], training=None, use_new_para_dict=True, *args, **kwargs) -> List[Tensor]
    (method) post_process: (self: Model, outputs: List[Tensor], training=None, *args, **kwargs) -> List[Tensor]
    ```
    """

    def __init__(self, Args: PredictionArgs,
                 training_structure, *args, **kwargs):

        super().__init__(Args, training_structure, *args, **kwargs)

        self.args = Args
        self._preprocess_list = []
        self._preprocess_para = {MOVE: -1,
                                 ROTATE: 0,
                                 SCALE: 1,
                                 UPSAMPLING: 4}

        self._preprocess_variables = {}

    def set_preprocess(self, *args):
        """
        Set pre-process methods used before training.

        args: pre-process methods.
            - Move positions on the observation step to (0, 0):
                args in `['MOVE', 'move']`

            - TODO
        """
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

    def pre_process(self, tensors: List[tf.Tensor],
                    training=None,
                    use_new_para_dict=True,
                    *args, **kwargs) -> List[tf.Tensor]:

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

    def post_process(self, outputs: List[tf.Tensor],
                     training=None,
                     *args, **kwargs) -> List[tf.Tensor]:

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
    Structure
    ---------
    Basic training structure for training and test ***Prediction*** models.

    Usage
    -----
    When training or test new models, please subclass this class,
    and specific your model's inputs, groundtruths, loss functions and
    metrics in the `__init__` method:
    ```python
    class MyPredictionStructure(Structure):
        def __init__(self, Args: List[str], *args, **kwargs):
            super().__init__(Args, *args, **kwargs)

            # model inputs and groundtruths
            self.set_model_inputs('traj')
            self.set_model_groundtruths('gt')

            # loss functions and their weights
            self.set_loss('ade')
            self.set_loss_weights('1.0')

            # metrics and their weights
            self.set_metrics('ade', 'fde')
            self.set_metrics_weights(1.0, 0.0)
    ```

    These methods must be rewritten:
    ```python
    # create new model
    def create_model(self, *args, **kwargs) -> Tuple[Model, keras.optimizers.Optimizer]:
        ...
    ```

    Public Methods
    --------------
    ```python
    # Load args
    (method) load_args: (self: Structure, current_args: List[str], load_path: str) -> (Namespace | List[str])

    # ----Models----
    # Load model from check point
    (method) load_from_checkpoint: (self: Structure, model_path, *args, **kwargs) -> Model

    # Save model
    (method) save_model: (self: Structure, save_path: str) -> None

    # Assign model's inputs and outputs
    (method) set_model_inputs: (self: Structure, *args) -> None
    (method) set_model_groundtruths: (self: Structure, *args) -> None

    # Assign loss functions and metrics
    (method) set_loss: (self: Structure, *args) -> None
    (method) set_loss_weights: (self: Structure, *args: List[float]) -> None
    (method) set_metrics: (self: Structure, *args) -> None
    (method) set_metrics_weights: (self: Structure, *args: List[float]) -> None

    # Create model
    (method) create_model: (self: Structure, *args, **kwargs) -> Tuple[Model, OptimizerV2]

    # Forward process
    (method) model_forward: (self: Structure, model_inputs: Tuple[Tensor], training=None, *args, **kwargs) -> Tuple[Tensor]

    # ----Datasets----
    # Load train/test/forward dataset
    (method) load_dataset: (self: Structure, *args, **kwargs) -> Tuple[DatasetV2, DatasetV2]
    (method) load_test_dataset: (self: Structure, *args, **kwargs) -> DatasetV2
    (method) load_forward_dataset: (self: Structure, *args, **kwargs) -> DatasetV2

    # ----Training and Test----
    # Loss Functions and Metrics
    (method) loss: (self: Structure, outputs, labels, loss_name_list: List[str] = ['L2'], *args, **kwargs) -> Tuple[Tensor, Dict[str, Tensor]]
    (method) metrics: (self: Structure, outputs, labels, loss_name_list=['L2_val'], *args, **kwargs) -> Tuple[Tensor, Dict[str, Tensor]]

    # Gradient densest operation
    (method) gradient_operations: (self: Structure, model_inputs, gt, loss_move_average: Variable, **kwargs) -> Tuple[Tensor, Dict[str, Tensor], Tensor]

    # Entrance of train or test
    (method) run_train_or_test: (self: Structure) -> None

    # Train
    (method) train: (self: Structure) -> None

    # Run Test
    (method) test: (self: Structure, *args, **kwargs) -> None

    # Forward
    (method) forward: (self: Structure, dataset: DatasetV2, return_numpy=True, **kwargs) -> ndarray

    # Call
    (method) __call__: (self: Structure, model_inputs, return_numpy=True) -> Tuple[ndarray, list]
    ```
    """

    agent_type = Agent
    DM_type = DatasetsManager

    def __init__(self, Args: List[str], *args, **kwargs):
        super().__init__(Args, *args, **kwargs)

        self.args = PredictionArgs(Args)

        self.important_args += ['test_set']

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

    def set_loss(self, *args):
        self.loss_list = [arg for arg in args]

    def set_loss_weights(self, *args: List[float]):
        self.loss_weights = [arg for arg in args]

    def set_metrics(self, *args):
        self.metrics_list = [arg for arg in args]

    def set_metrics_weights(self, *args: List[float]):
        self.metrics_weights = [arg for arg in args]

    def run_test(self):
        """
        Run test of trajectory prediction on ETH-UCY or SDD dataset.
        """
        if True:
            info = base.DatasetsInfo(self.args.test_set)
            
            if self.args.test_mode == 'all':
                for dataset in info.test_sets:
                    agents = self.DM_type.load(self.args, dataset, mode='test')
                    self.test(agents=agents, dataset_name=dataset)

            elif self.args.test_mode == 'mix':
                agents = []
                for dataset_c in info.test_sets:
                    agents += self.DM_type.load(self.args, dataset_c, mode='test')

                self.test(agents=agents, dataset_name=self.args.test_set)

            elif self.args.test_mode == 'one':
                ds = self.args.force_set
                agents = self.DM_type.load(self.args, ds, mode='test')
                self.test(agents=agents, dataset_name=ds)

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

        model_inputs.append(gt)
        return tuple(model_inputs)

    def load_dataset(self, *args, **kwargs) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Load training and val dataset.

        :return dataset_train: train dataset, type = `tf.data.Dataset`
        :return dataset_val: val dataset, type = `tf.data.Dataset`
        """
        agents_train, agents_test = self.DM_type.load(self.args, 'auto', mode='train')

        train_data = self.get_inputs_from_agents(agents_train)
        test_data = self.get_inputs_from_agents(agents_test)

        dataset_train = tf.data.Dataset.from_tensor_slices(train_data)
        dataset_train = dataset_train.shuffle(len(dataset_train),
                                              reshuffle_each_iteration=True)
        dataset_test = tf.data.Dataset.from_tensor_slices(test_data)
        return dataset_train, dataset_test

    def load_test_dataset(self, *args, **kwargs) -> tf.data.Dataset:
        """
        Load test dataset.

        :return dataset_train: test dataset, type = `tf.data.Dataset`
        """
        agents = kwargs['agents']
        test_tensor = self.get_inputs_from_agents(agents)
        dataset_test = tf.data.Dataset.from_tensor_slices(test_tensor)
        return dataset_test

    def load_forward_dataset(self, model_inputs: List[agent_type],
                             *args, **kwargs) -> tf.data.Dataset:
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
             *args, **kwargs) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
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
                          mode='loss',
                          *args, **kwargs)

    def metrics(self, outputs: List[tf.Tensor],
                labels: tf.Tensor,
                *args, **kwargs) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
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
        self.print_parameters(title='dataset options',
                              rotate_times=self.args.rotate,
                              add_noise=self.args.add_noise)

    def write_test_results(self, model_outputs: List[tf.Tensor],
                           agents: Dict[str, List[agent_type]],
                           *args, **kwargs):

        testset_name = kwargs['dataset_name']

        if self.args.draw_results and (not testset_name.startswith('mix')):
            # draw results on video frames
            tv = TrajVisualization(dataset=testset_name)
            save_base_path = dir_check(self.args.log_dir) \
                if self.args.load == 'null' \
                else self.args.load

            save_format = os.path.join(dir_check(os.path.join(
                save_base_path, 'VisualTrajs')), '{}_{}.{}')

            self.log('Start saving images at {}'.format(
                os.path.join(save_base_path, 'VisualTrajs')))

            for index, agent in self.log_timebar(agents, 'Saving...'):
                # write traj
                output = model_outputs[0][index].numpy()
                agents[index].pred = output

                # draw as one image
                tv.draw(agents=[agent],
                        frame_name=float(
                            agent.frame_list[self.args.obs_frames]),
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

            self.log('Prediction result images are saved at {}'.format(
                os.path.join(save_base_path, 'VisualTrajs')))
