"""
@Author: Conghao Wong
@Date: 2021-09-16 20:00:49
@LastEditors: Conghao Wong
@LastEditTime: 2021-09-17 09:46:17
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import os
import shutil
import sys
from typing import List

import pytest

from main import train_or_test
from modules import models as M


class TestClass():

    def setup_class(self):
        if os.path.exists(p := './.test'):
            shutil.rmtree(p)

        if not os.path.exists(p := './test.log'):
            with open(p, 'w+') as f:
                f.writelines(['-----Start Test-----'])

    def teardown_class(self):
        pass

    def test_train_linear(self):
        self.run_with_args(['--model', 'test', '--epochs', '5',
                            '--use_maps', '0', '--step', '4',
                            '--save_base_dir', './.test'])

    def test_evaluate_linear(self):
        p = './.test'
        dir_name = os.listdir(p)[0]
        self.run_with_args(['--load', os.path.join(p, dir_name)])

    def test_evaluate_msn(self):
        self.run_with_args(['--model', 'msn',
                            '--loada', './.github/workflows/test_weights/msna_zara1',
                            '--loadb', 'l'])

    def run_with_args(self, args: List[str]):
        _arg = M.prediction.PredictionArgs(['null.py'] + args)
        train_or_test(_arg, force_args=args)


if __name__ == '__main__':
    a = TestClass()

    a.setup_class()
    a.test_train_linear()
    a.test_evaluate_linear()
    a.test_evaluate_msn()
    a.teardown_class()