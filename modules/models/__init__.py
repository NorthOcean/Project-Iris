"""
Models
======
'''
@Author: Conghao Wong
@Date: 2021-03-15 10:49:31
@LastEditors: Conghao Wong
@LastEditTime: 2021-08-04 14:51:21
@Description: Description: A framework for training, eval, and test on models based on `tensorflow 2`.
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
'''

Usage
-----
You can design your new models and training it based on this module.
- Design a new model.

For designing a new model, you can do as the following example.
```python
import modules.models as M
import tensorflow as tf

class MyModel(M.base.Model):
    def __init__(self, Args):
        super().__init__(Args)

        # please clearfy all layers used in the model here like these:
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(2, activation=tf.nn.sigmoid)

    def call(self, inputs, training=None, mask=None):
        # put your model's implemention process here like these:
        output1 = self.dense1(inputs)
        output2 = self.dense2(output1)
        return output2
```

- Training, eval, and test your model.

For training your new model, like above `MyModel`, you should subclass some training
structure (like `base.Structure` for general models or `prediction.Structure` for 
trajectory prediction models), and **REWRITE THESE METHODS**:

1. For general models, you can do like this:
```python
class MyTrainingStructure(M.base.Structure):
    def __init__(self, args, arg_type=M.base.Args):
        super().__init__(args, arg_type=arg_type)
    
    # rewrite #1
    def self.create_model(self) -> Tuple[Model, keras.optimizers.Optimizer]:
        # create a instance of your model
        model = MyModel(self.args)
        opt = tf.keras.optimizers.Adam(self.args.lr)
        return model, opt
    
    # rewrite #2
    def self.load_dataset(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        # load training and val dataset from original files
        file_path = 'SOME_DATASET_PATH_X'
        label_path = 'SOME_DATASET_PATH_Y'
        
        files_train, files_eval = SOME_LOAD_FUNCTION(file_path)
        labels_train, labels_eval = SOME_LOAD_FUNCTION(label_path)

        dataset_train = tf.data.Dataset.from_tensor_slices(
            (files_train, labels_train))
        dataset_eval = tf.data.Dataset.from_tensor_slices(
            (files_eval, labels_eval))
        return dataset_train, dataset_eval

    # rewrite #3
    def self.load_test_dataset(self, **kwargs) -> tf.data.Dataset:
        # load test dataset from original files
        files_test = SOME_OTHER_LOAD_FUNCTION('TEST_PATH_X')
        labels_test = SOME_OTHER_LOAD_FUNCTION('TEST_PATH_Y')
        return tf.data.Dataset.from_tensor_slices(
            (files_test, labels_test))
    
    # (Optional) rewrite #4
    def self.load_forward_dataset(self, **kwargs) -> tf.data.Dataset:
        # load data when use models for online test
        pass
```
Besides, you can also rewrite `base.Structure.loss` and `base.Structure.metrics`
to change your loss functions and metrics.

2. For trajectory prediction models, what your should do is to rewrite the
`prediction.Structure.create_model` method.
Default dataset used for training is `ETH-UCY` or `SDD`, which can be changed by
subclassing `prediction.PredictionDatasetInfo`.

```python
class MyPredictionTrainingStructure(M.prediction.Structure):
    def __init__(self, args, arg_type=M.prediction.TrainArgs):
        super().__init__(args, arg_type=arg_type)

    # rewrite #1
    def create_model(self):
        # create a instance of your model
        model = MyModel(self.args)
        opt = tf.keras.optimizers.Adam(self.args.lr)
        return model, opt
```
Besides, you can also rewrite `prediction.Structure.loss` and
`prediction.Structure.metrics` to change your loss functions and metrics.

- Run training or test.

You can run training or test on your model by:
```python
>>> MyTrainingStructure.run_commands()
```

Packages
--------
- `base`
    A package which is designed for general training process.
    It contains these public classes:
```python
base.Args       # Manage args
base.Dataset    # Manage single dataset's info
base.Model      # a subclass of `keras.Model` that contains data processing methods
base.Structure  # a base structure for training, eval, and test models
```
   For more details, please refer to each class's doc.

- `prediction`
    A package designed for general prediction models.
    It contains these public classes:
    1. Agent managers
```python
prediction.PredictionAgent    # for training, eval, and test
prediction.OnlineAgentManager   # for online implemention
```

    2. Args managers
```python
prediction.TrainArgs    # training, eval, and test args
prediction.OnlineArgs   # online implemention args
```

    3. Training, evaluation, and test
```python
prediction.Loss         # basic loss functions or metrics for trajectory prediction
prediction.Structure    # basic structure for training, eval, and test for prediction models
```

    4. Others
```python
prediction.Process      # basic process methods for trajectory data
```

"""

from . import _base as base
from . import _prediction as prediction
from . import _sceneModeling as sceneModeling
from . import helpmethods
