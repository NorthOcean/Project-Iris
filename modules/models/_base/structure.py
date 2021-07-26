'''
Author: Conghao Wong
Date: 2020-12-24 18:20:20
LastEditors: Conghao Wong
LastEditTime: 2021-04-19 19:43:42
Description: file content
'''

import os
import re
from argparse import Namespace
from typing import Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras

from ..helpmethods import dir_check
from .agent import Agent
from .args import argParse as ArgParse
from .args.args import BaseTrainArgs as ArgType
from .baseObject import BaseObject


class Model(keras.Model):
    """
    Base Model
    ----------
    An expanded class for `keras.Model`.

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
    (method) pre_process: (self: Model, model_inputs: List[Tensor], training=None, *args, **kwargs) -> List[Tensor]
    (method) post_process: (self: Model, outputs: List[Tensor], training=None, *args, **kwargs) -> List[Tensor]
    ```
    """

    def __init__(self, Args: Union[Namespace, ArgType],
                 training_structure=None,
                 *args, **kwargs):
        """
        init the model class

        :param Args: args used in the model
        :param training_structure: structure for training the model. type = `Structure`
        """
        super().__init__(*args, **kwargs)
        self.args = Args
        self.training_structure = training_structure

    def call(self, inputs, training=None, mask=None, *args, **kwargs):
        raise NotImplementedError

    # @tf.function
    def forward(self, model_inputs: List[tf.Tensor],
                training=None,
                *args, **kwargs) -> List[tf.Tensor]:
        """
        Run a forward implementation.

        :param model_inputs: input tensor (or a `list` of tensors)
        :param training: config training or test mode
        :return output: model's output. type=`List[tf.Tensor]`
        """
        model_inputs_processed = self.pre_process(model_inputs, training)

        # use `self.call()` to debug
        output = self(model_inputs_processed)

        # make sure the output is a list or a tuple
        if not type(output) in [list, tuple]:
            output = [output]

        return self.post_process(output, training,
                                 model_inputs=model_inputs)

    def pre_process(self, model_inputs: List[tf.Tensor],
                    training=None,
                    *args, **kwargs) -> List[tf.Tensor]:
        """
        Pre-processing before inputting to the model
        """
        return model_inputs

    def post_process(self, outputs: List[tf.Tensor],
                     training=None,
                     *args, **kwargs) -> List[tf.Tensor]:
        """
        Post-processing of model's output when model's inferencing.
        """
        return outputs


class Structure(BaseObject):
    """
    Introduction
    ------------
    Basic training structure for training and test models based on `tensorflow v2`.
    It contains methods for loading datasets and training or evaluating the model.

    Usage
    -----
    When training or test new models, please subclass this class, and specify args used
    in the structure. You can do like:
    ```python
    class MyTrainingStructure(Structure):
        def __init__(self, Args: List[str], *args, **kwargs):
            super().__init__(Args, *args, **kwargs)

            self.args = PredictionArgs(Args)
    ```
    Then, please specify loss functions and metrics:
    ```python
        def loss(self, outputs, labels, 
                loss_name_list: List[str] = ['L2'],
                *args, **kwargs):
            ...

        def metrics(self, outputs, labels, 
                    loss_name_list=['L2_val'],
                    *args, **kwargs):
            ...
    ```

    These methods must be rewritten before training:
    ```python
    # create new model
    def create_model(self, *args, **kwargs) -> Tuple[Model, keras.optimizers.Optimizer]:
        ...

    # load training and val dataset from original dataset files
    def load_dataset(self, *args, **kwargs) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        ...

    # load test dataset from original dataset files
    def load_test_dataset(self, *args, **kwargs) -> tf.data.Dataset:
        ...

    # load data when use models for test new data
    def load_forward_dataset(self, *args, **kwargs) -> tf.data.Dataset:
        ...
    ```

    Public methods
    --------------
    ```python
    # Load args
    (method) load_args: (self: Structure, current_args: List[str], load_path: str) -> (Namespace | List[str])

    # ----Models----
    # Load model
    (method) load_model: (self: Structure, model_path: str, *args, **kwargs) -> Model

    # Save model
    (method) save_model: (self: Structure, save_path: str) -> None

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

    def __init__(self, Args: List[str], *args, **kwargs):
        """
        :param Args: args used when training. (got from `sys.argv`)
        """
        super().__init__()

        self.model = None
        self.args = ArgType(Args)
        self.gpu_config()

    def gpu_config(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu.replace('_', ',')
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    def load_args(self, current_args: List[str],
                  load_path: str) -> Union[Namespace, List[str]]:
        """
        Parse args from `load_path`.

        :param current_args: default args from `sys.argv`
        :param load_path: path of new args to load
        :return save_args: loaded args
        """
        try:
            arg_paths = [os.path.join(load_path, item) for item in os.listdir(
                load_path) if (item.endswith('args.npy') or item.endswith('args.json'))]
            save_args = ArgParse.load(arg_paths)
        except:
            save_args = current_args

        return save_args

    def load_model(self, model_path: str, *args, **kwargs) -> Model:
        """
        Load already trained models from checkpoint files.

        :param model_path: path of model
        :return model: loaded model
        """
        model, _ = self.create_model(*args, **kwargs)
        model.load_weights(model_path)
        return model

    def load_from_checkpoint(self, model_path, *args, **kwargs) -> Model:
        """
        Load already trained models from `.h5` or `.tf` files according to args.

        :param model_path: target dir where your model puts in
        :return model: model loaded
        """
        if model_path == 'null':
            return self.model

        dir_list = os.listdir(model_path)
        save_format = '.' + self.args.save_format
        try:
            name_list = [item.split(save_format)[0].split(
                '_epoch')[0] for item in dir_list if save_format in item]
            if not len(name_list):
                raise IndexError

        except IndexError as e:
            save_format = '.h5' if save_format == '.tf' else '.tf'
            name_list = [item.split(save_format)[0].split(
                '_epoch')[0] for item in dir_list if item.endswith(save_format)]

        try:
            model_name = max(name_list, key=name_list.count)
            base_path = os.path.join(model_path, model_name + '{}')

            if self.args.save_best and ('best_ade_epoch.txt' in dir_list):
                best_epoch = np.loadtxt(os.path.join(model_path, 'best_ade_epoch.txt'))[
                    1].astype(int)
                model = self.load_model(base_path.format(
                    '_epoch{}{}'.format(best_epoch, save_format)),
                    *args, **kwargs)
            else:
                model = self.load_model(base_path.format(save_format),
                                        *args, **kwargs)

        except:
            model_name = name_list[0]
            base_path = os.path.join(model_path, model_name + save_format)
            model = self.load_model(base_path, *args, **kwargs)

        return model

    def save_model(self, save_path: str):
        """
        Save trained model to `save_path`.

        :param save_path: where model saved.
        """
        self.model.save_weights(save_path)

    def create_model(self, *args, **kwargs) -> Tuple[Model, keras.optimizers.Optimizer]:
        """
        Create models.
        Please *rewrite* this when training new models.

        :return model: created model
        :return optimizer: training optimizer
        """
        raise NotImplementedError('MODEL is not defined!')

    def model_forward(self, model_inputs: List[tf.Tensor], training=None, *args, **kwargs) -> Tuple[tf.Tensor]:
        """
        Entire forward process of this model.

        :param model_inputs: a list (or tuple) of tensor to input to model(s)
        :param mode: forward type, canbe `'test'` or `'train'`
        :return outputs: a list (or tuple) of tensor
        """
        return self.model.forward(model_inputs, training, **kwargs)

    def loss(self, outputs, labels, loss_name_list: List[str] = ['L2'], *args, **kwargs) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Train loss, using L2 loss by default.

        :param outputs: model's outputs
        :param labels: groundtruth labels
        :param loss_name_list: a list of name of used loss functions

        :return loss: sum of all single loss functions
        :return loss_dict: a dict of all losses
        """
        loss = tf.reduce_mean(tf.linalg.norm(outputs - labels, axis=-1))
        loss_dict = dict(zip(loss_name_list, [loss]))
        return loss, loss_dict

    def metrics(self, outputs, labels, loss_name_list=['L2_val'], *args, **kwargs) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Metrics, using L2 loss by default.

        :param outputs: model's outputs
        :param labels: groundtruth labels
        :param loss_name_list: a list of name of used loss functions

        :return loss: sum of all single loss functions
        :return loss_dict: a dict of all losses
        """
        return self.loss(outputs, labels, loss_name_list, **kwargs)

    def _run_one_step(
            self,
            model_inputs,
            gt,
            training=None) -> Tuple[List[tf.Tensor], tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Run one step of forward and calculate loss.

        :param test_tensor: `[model_inputs, gt]`
        :param mode: run mode, canbe in `['train', 'test']`
        :return model_output: model output
        :return metrics: weighted sum of all loss 
        :return loss_dict: a dict contains all loss
        """

        model_output = self.model_forward(model_inputs, training)
        metrics, loss_dict = self.metrics(
            model_output, gt, mode='train' if training else 'test')
        return model_output, metrics, loss_dict

    def gradient_operations(self, model_inputs,
                            gt,
                            loss_move_average: tf.Variable,
                            **kwargs) -> Tuple[tf.Tensor, Dict[str, tf.Tensor], tf.Tensor]:
        """
        Run gradient dencent once during training.

        :param model_inputs: model inputs
        :param gt :ground truth
        :param loss_move_average: Moving average loss

        :return loss: sum of all single loss functions
        :return loss_dict: a dict of all loss functions
        :return loss_move_average: Moving average loss
        """
        with tf.GradientTape() as tape:
            model_output = self.model_forward(
                model_inputs, training=True, gt=gt)
            loss, loss_dict = self.loss(model_output,
                                        gt,
                                        model_inputs=model_inputs,
                                        **kwargs)
            loss_move_average = 0.7 * loss + 0.3 * loss_move_average

        grads = tape.gradient(loss_move_average,
                              self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,
                                           self.model.trainable_variables))

        return loss, loss_dict, loss_move_average

    def get_inputs_from_agents(self, input_agents: List[Agent]) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError

    def load_dataset(self, *args, **kwargs) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Load training and val dataset.

        :return dataset_train: train dataset, type = `tf.data.Dataset`
        :return dataset_val: val dataset, type = `tf.data.Dataset`
        """
        raise NotImplementedError('DATASET is not defined!')

    def load_test_dataset(self, *args, **kwargs) -> tf.data.Dataset:
        """
        Load test dataset.

        :return dataset_train: test dataset, type = `tf.data.Dataset`
        """
        raise NotImplementedError('DATASET is not defined!')

    def load_forward_dataset(self, *args, **kwargs) -> tf.data.Dataset:
        """
        Load forward dataset.

        :return dataset_train: test dataset, type = `tf.data.Dataset`
        """
        raise NotImplementedError('DATASET is not defined!')

    def print_dataset_info(self):
        self.print_parameters(title='dataset options')

    def print_training_info(self):
        self.print_parameters(title='training options')

    def print_test_result_info(self, loss_dict, **kwargs):
        dataset = kwargs['dataset_name']
        self.print_parameters(title='test results',
                            dataset=dataset,
                            **loss_dict)
        with open('./test_log.txt', 'a') as f:
            f.write('{}, {}, {}\n'.format(
                self.args.load,
                dataset,
                loss_dict))

    def run_train_or_test(self):
        """
        Load args, load datasets, and start training or test
        """
        # prepare training
        if self.args.load == 'null':
            self.model, self.optimizer = self.create_model()

            if self.args.restore != 'null':
                # restore weights from files
                self.model = self.load_from_checkpoint(self.args.restore)

            self.log('Start training with args = {}'.format(self.args))
            self.train()

        # prepare test
        else:
            self.log(
                'Start test model from `{}`'.format(self.args.load))
            self.model = self.load_from_checkpoint(self.args.load)
            self.run_test()

    def train(self):
        """
        Train model `self.model`.
        """
        # Load dataset
        dataset_train, dataset_val = self.load_dataset()
        self.train_number = len(dataset_train)

        self.print_dataset_info()
        self.print_training_info()

        if self.args.save_model:
            arg_save_path = os.path.join(dir_check(self.args.log_dir),
                                         'args.json')
            ArgParse.save(arg_save_path, self.args)

        # Prepare for training
        summary_writer = tf.summary.create_file_writer(self.args.log_dir)
        best_metric = 10000.0
        best_epoch = 0
        loss_move_average = tf.Variable(0.0, dtype=tf.float32)
        loss_dict = {'-': 'null'}
        test_epochs = []

        # Start training
        dataset_train = dataset_train.repeat(self.args.epochs   # epochs
                                             ).batch(self.args.batch_size)   # batchs
        time_bar = self.log_timebar(dataset_train,
                                    text='Training...',
                                    return_enumerate=False)
        for batch_id, train_data in enumerate(time_bar):
            # Run training
            epoch = (batch_id * self.args.batch_size) // self.train_number
            loss, loss_dict, loss_move_average = self.gradient_operations(
                model_inputs=train_data[:-1],
                gt=train_data[-1],
                loss_move_average=loss_move_average,
                epoch=epoch)

            # Run eval
            if ((epoch >= self.args.start_test_percent * self.args.epochs)
                    and ((epoch - 1) % self.args.test_step == 0)
                    and (not epoch in test_epochs)
                    and (epoch > 0)):

                metrics_all = []
                loss_dict_all = {}
                test_epochs.append(epoch)
                for val_data in dataset_val.batch(self.args.batch_size):
                    _, metrics, loss_dict_eval = self._run_one_step(val_data[:-1],
                                                                    val_data[-1])
                    metrics_all.append(metrics)
                    for key in loss_dict_eval:
                        if not key in loss_dict_all:
                            loss_dict_all[key] = []
                        loss_dict_all[key].append(loss_dict_eval[key])

                # Calculate loss
                metrics_all = tf.reduce_mean(tf.stack(metrics_all)).numpy()
                for key in loss_dict_all:
                    loss_dict_all[key] = tf.reduce_mean(
                        tf.stack(loss_dict_all[key])).numpy()

                # Save model
                if metrics_all <= best_metric:
                    best_metric = metrics_all
                    best_epoch = epoch

                    if self.args.save_best:
                        self.save_model(os.path.join(self.args.log_dir, '{}_epoch{}.{}'.format(
                            self.args.model_name,
                            epoch,
                            self.args.save_format)))

                        np.savetxt(os.path.join(self.args.log_dir, 'best_ade_epoch.txt'),
                                   np.array([best_metric, best_epoch]))

            # Update time bar
            step_dict = dict(zip(['epoch', 'best'], [epoch, best_metric]))
            try:
                loss_dict = dict(
                    step_dict, **dict(loss_dict, **loss_dict_eval))  # 拼接字典
            except UnboundLocalError as e:
                loss_dict = dict(step_dict, **loss_dict)

            for key in loss_dict:
                if issubclass(type(loss_dict[key]), tf.Tensor):
                    loss_dict[key] = loss_dict[key].numpy()
            time_bar.set_postfix(loss_dict)

            # Write tfboard
            with summary_writer.as_default():
                for loss_name in loss_dict:
                    value = loss_dict[loss_name]
                    tf.summary.scalar(loss_name, value, step=epoch)

        self.print_training_done_info()
        if self.args.save_model:
            model_save_path = os.path.join(
                self.args.log_dir,
                '{}.{}'.format(self.args.model_name, self.args.save_format))

            self.save_model(model_save_path)
            self.log(('Trained model is saved at `{}`.\n' +
                              'To re-test this model, please use ' +
                              '`python main.py --load {}`.').format(model_save_path,
                                                                    self.args.log_dir))

    def print_training_done_info(self, **kwargs):
        self.log(('Training done.' +
                          'Tensorboard training log file is saved at `{}`' +
                          'To open this log file, please use `tensorboard ' +
                          '--logdir {}`').format(self.args.log_dir, self.args.log_dir))

    def run_test(self):
        self.test()

    def test(self, *args, **kwargs):
        """
        Run test
        """
        # Load dataset
        dataset_test = self.load_test_dataset(*args, **kwargs)

        # Start test
        # model_inputs_all = []
        model_outputs_all = []
        label_all = []
        loss_dict_all = {}
        for batch_id, test_data in self.log_timebar(
                dataset_test.batch(self.args.batch_size),
                'Test...'):

            model_outputs, loss, loss_dict = self._run_one_step(
                test_data[:-1], test_data[-1])

            # model_inputs_all = append_results_to_list(
            # test_data[:-1], model_inputs_all)
            model_outputs_all = append_results_to_list(
                model_outputs, model_outputs_all)
            label_all = append_results_to_list(test_data[-1:], label_all)

            for key in loss_dict:
                if not key in loss_dict_all:
                    loss_dict_all[key] = []
                loss_dict_all[key].append(loss_dict[key])

            # self.log(loss_dict)

        for key in loss_dict_all:
            loss_dict_all[key] = tf.reduce_mean(
                tf.stack(loss_dict_all[key])).numpy()

        # Write test results
        self.print_test_result_info(loss_dict_all, **kwargs)

        model_inputs_all = list(dataset_test.as_numpy_iterator())
        model_outputs_all = stack_results(model_outputs_all)
        label_all = stack_results(label_all)

        self.write_test_results(model_outputs_all,
                                model_inputs=model_inputs_all,
                                labels=label_all,
                                **kwargs)

    def write_test_results(self, model_outputs: List[tf.Tensor], **kwargs):
        pass

    def forward(self, dataset: tf.data.Dataset, return_numpy=True, **kwargs) -> np.ndarray:
        """
        Forward model on one dataset and return outputs.

        :param dataset: dataset to forward, type = `tf.data.Dataset`
        :param return_numpy: controls if returns `numpy.ndarray` or `tf.Tensor`
        :return model_outputs: model outputs, canbe numpy array or tensors
        """
        model_outputs_all = []
        for model_inputs in dataset.batch(self.args.batch_size):
            if issubclass(type(model_inputs), tf.Tensor):
                model_inputs = (model_inputs,)

            model_outputs = self.model_forward(model_inputs, training=None)

            if not len(model_outputs_all):
                [model_outputs_all.append([])
                 for _ in range(len(model_outputs))]

            [model_outputs_all[index].append(
                model_outputs[index]) for index in range(len(model_outputs))]

        if return_numpy:
            return [tf.concat(model_outputs, axis=0).numpy() for model_outputs in model_outputs_all]
        else:
            return [tf.concat(model_outputs, axis=0) for model_outputs in model_outputs_all]

    def __call__(self, model_inputs, return_numpy=True) -> Tuple[np.ndarray, list]:
        test_dataset = self.load_forward_dataset(model_inputs=model_inputs)
        results = self.forward(test_dataset, return_numpy)
        return results


def append_results_to_list(results: List[tf.Tensor], target: list):
    if not len(target):
        [target.append([]) for _ in range(len(results))]
    [target[index].append(results[index]) for index in range(len(results))]
    return target


def stack_results(results: List[tf.Tensor]):
    for index, tensor in enumerate(results):
        results[index] = tf.concat(tensor, axis=0)
    return results
