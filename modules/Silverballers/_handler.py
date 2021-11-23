"""
@Author: Conghao Wong
@Date: 2021-10-28 19:42:20
@LastEditors: Conghao Wong
@LastEditTime: 2021-11-23 19:15:44
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import List, Tuple

import tensorflow as tf
from tensorflow import keras

from .. import models as M
from ..Vertical._VirisBeta import VIrisBetaModel
from ._args import HandlerArgs


class HandlerModel(VIrisBetaModel):
    def __init__(self, Args: HandlerArgs,
                 points: int,
                 asSecondStage=False,
                 p_index: str = None,
                 training_structure=None,
                 *args, **kwargs):

        super().__init__(Args, points,
                         asSecondStage=asSecondStage,
                         p_index=p_index,
                         training_structure=training_structure,
                         *args, **kwargs)


class Handler(M.prediction.Structure):

    def __init__(self, Args: List[str],
                 association: M.prediction.Structure = None,
                 *args, **kwargs):

        super().__init__(Args, *args, **kwargs)

        self.args = HandlerArgs(Args)

        self.important_args += ['points']

        self.set_model_inputs('trajs', 'maps', 'paras', 'gt')
        self.set_model_groundtruths('gt')

        self.set_loss('ade', 'diff')
        self.set_loss_weights(0.8, 0.2)

        self.set_metrics('ade', 'fde')
        self.set_metrics_weights(1.0, 0.0)

        self.association = association

    def create_model(self, *args, **kwargs):
        model = HandlerModel(self.args,
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

    def print_test_results(self, loss_dict, dataset_name, **kwargs):
        self.print_parameters(title='rest results',
                              **dict({'dataset': dataset_name}, **loss_dict))

        self.log('Results: {}, {}, {}.'.format(
            self.args.load,
            dataset_name,
            loss_dict
        ))
