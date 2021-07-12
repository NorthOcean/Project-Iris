"""
@Author: Conghao Wong
@Date: 2021-07-09 09:50:49
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-12 15:52:26
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from argparse import Namespace
from typing import List, Tuple

import tensorflow as tf

from ._args import VArgs
from ..satoshi._alpha_transformer import SatoshiAlphaTransformerModel as VIrisAlphaModel
from ._VirisAlpha import VIrisAlpha #, VIrisAlphaModel
from ._VirisBeta import VIrisBeta, VIrisBetaModel


class _VIrisAlphaModelPlus(VIrisAlphaModel):
    def __init__(self, Args,
                 linear_prediction=False,
                 training_structure=None,
                 *args, **kwargs):

        super().__init__(Args, training_structure,
                         *args, **kwargs)

        self.linear = linear_prediction

    def post_process(self, outputs: Tuple[tf.Tensor],
                     training=None,
                     **kwargs) -> Tuple[tf.Tensor]:

        # shape = ((batch, Kc, 2))
        outputs = super().post_process(outputs, training, **kwargs)

        if training:
            return outputs

        batch, Kc = outputs[0].shape[:2]
        pred = self.args.pred_frames
        K = self.args.K

        # shape = (batch*Kc, 1, 2)
        proposals = tf.reshape(outputs[0], [-1, 1, 2])
        current_inputs = kwargs['model_inputs']

        if self.linear:
            start = current_inputs[0][:, -1:, :]    # (batch, 1, 2)
            end = outputs[0]    # (batch, Kc, 2)

            linear_results = []
            for p in range(1, pred + 1):
                linear_results.append(
                    (end - start) * p / pred
                    + start
                )   # (batch, Kc, 2)

            return (tf.transpose(linear_results, [1, 2, 0, 3]),)

        else:
            # prepare new inputs into beta model
            # new batch_size (total) is batch*Kc
            batch_size = self.args.max_batch_size // Kc
            batch_index = BatchIndex(batch_size, batch)

            beta_results = []
            while (index := batch_index.get_new()) is not None:
                [start, end, length] = index
                beta_inputs = [tf.repeat(inp[start:end], Kc, axis=0)
                               for inp in current_inputs]
                beta_inputs.append(proposals[start*Kc: end*Kc])

                # beta outputs shape = (batch*Kc, pred, 2)
                beta_results.append(self.training_structure.beta(
                    beta_inputs,
                    return_numpy=False)[0])

            beta_results = tf.concat(beta_results, axis=0)
            beta_results = tf.reshape(beta_results, [batch, Kc, pred, 2])
            return (beta_results,)


class VIris(VIrisAlpha):
    """
    Structure for Vertical prediction
    ---------------------------------

    """

    def __init__(self, Args: List[str], *args, **kwargs):
        super().__init__(Args, *args, **kwargs)
        
        self.args = VArgs(Args)

        # set inputs and groundtruths
        self.set_model_inputs('trajs', 'maps', 'map_paras')
        self.set_model_groundtruths('gt')

        # set metrics
        self.set_metrics('ade', 'fde')
        self.set_metrics_weights(1.0, 0.0)

        # assign alpha model and beta model containers
        self.alpha = self
        self.beta = VIrisBeta(Args)
        self.linear_predict = False

        # load weights
        if 'null' in [self.args.loada, self.args.loadb]:
            raise ('`IrisAlpha` or `IrisBeta` not found!' +
                   ' Please specific their paths via `--loada` or `--loadb`.')

        if self.args.loadb.startswith('l'):
            self.linear_predict = True
        else:
            self.beta.args = VArgs(self.beta.load_args(Args, self.args.loadb))
            self.beta._model = self.beta.load_from_checkpoint(self.args.loadb)

        self.alpha.args = VArgs(
            args=self.alpha.load_args(Args, self.args.loada),
            default_args=self.args._args
        )
        
        self.alpha._model = self.alpha.load_from_checkpoint(
            self.args.loada,
            linear_prediction=self.linear_predict
        )

    def run_train_or_test(self):
        self.run_test()

    def create_model(self, *args, **kwargs):
        return super().create_model(model_type=_VIrisAlphaModelPlus,
                                    *args, **kwargs)

    def print_test_result_info(self, loss_dict, **kwargs):
        dataset = kwargs['dataset_name']
        self.log_parameters(title='test results', **
                            dict({'dataset': dataset}, **loss_dict))
        self.logger.info('Results from {}, {}, {}, {}'.format(
            self.args.loada,
            self.args.loadb,
            dataset,
            loss_dict))


class BatchIndex():
    def __init__(self, batch_size, length):
        super().__init__()

        self.bs = batch_size
        self.l = length

        self.start = 0
        self.end = 0

    def init(self):
        self.start = 0
        self.end = 0

    def get_new(self):
        """
        Get batch index

        :return index: (start, end, length)
        """
        if self.start >= self.l:
            return None

        start = self.start
        self.end = self.start + self.bs
        if self.end > self.l:
            self.end = self.l

        self.start += self.bs

        return [start, self.end, self.end - self.start]
