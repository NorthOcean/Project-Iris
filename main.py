"""
@Author: Conghao Wong
@Date: 2019-12-20 09:38:24
@LastEditors: Conghao Wong
@LastEditTime: 2021-06-25 09:41:01
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'        # 去除TF输出
import time

import numpy as np
import modules as M

Args = M.satoshi.SatoshiArgs

TIME = time.strftime('%Y%m%d-%H%M%S',time.localtime(time.time()))


def train_or_test(args:Args):
    if not args.load == 'null':
        current_args = args
        try:
            arg_paths = [os.path.join(current_args.load, item) for item in os.listdir(current_args.load) if (item.endswith('args.npy') or item.endswith('args.json'))]
            save_args = M.models.base.ArgParse.load(arg_paths)
        except IndexError as e:
            save_args = current_args
        model = save_args.model
    else:
        model = args.model

    if model == 'l':
        trainingStructure = M.linear.LinearPrediction

    elif model == 'test':
        trainingStructure = M.satoshi.Satoshi

    elif model == 'sat':
        trainingStructure = M.satoshi.sat
    
    elif model == 'sbt':
        trainingStructure = M.satoshi.sbt
    
    elif model == 'st':
        trainingStructure = M.satoshi.SatoshiTransformer
    
    elif model == 'ia':
        trainingStructure = M.iris.IrisAlpha

    elif model == 'ib':
        trainingStructure = M.iris.IrisBeta

    elif model == 'iris':
        trainingStructure = M.iris.Iris

    elif model == 'ibc':
        trainingStructure = M.iris.IrisBetaCVAE

    elif model == 'irisc':
        trainingStructure = M.iris.IrisCVAE

    elif model == 'iris3':
        trainingStructure = M.iris.Iris3

    trainingStructure(args).run_train_or_test()
    
def main():
    args = Args().args()
    train_or_test(args)


if __name__ == "__main__":
    main()
