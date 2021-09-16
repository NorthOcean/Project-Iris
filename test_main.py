"""
@Author: Conghao Wong
@Date: 2021-09-16 20:00:49
@LastEditors: Conghao Wong
@LastEditTime: 2021-09-16 21:51:59
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import os
import shutil
import sys

import pytest

from main import train_or_test
from modules import models as M


def train():
    fa = [sys.argv[0]] + ['--model', 'test', '--epochs', '5',
                          '--use_maps', '0', '--step', '4',
                          '--save_base_dir', './.test']
    args = M.prediction.PredictionArgs(fa)

    train_or_test(args, force_args=fa)

def evaluate():

    p = './.test'
    dir_name = os.listdir(p)[0]
    fa = [sys.argv[0]] + ['--load', os.path.join(p, dir_name)]
    args = M.prediction.PredictionArgs(fa)

    train_or_test(args, force_args=fa)

def test_pred():
    if os.path.exists(p := './.test'):
        shutil.rmtree(p)

    train()
    evaluate()


if __name__ == '__main__':
    test_pred()
