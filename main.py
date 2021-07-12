"""
@Author: Conghao Wong
@Date: 2019-12-20 09:38:24
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-12 14:34:54
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import os
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'        # 去除TF输出
import time

import numpy as np
import modules as M

import argparse
import sys

def get_args() -> argparse.Namespace:
    arg_str = ''
    for s in sys.argv[1:]:
        arg_str += s + ' '

    accept_str = ''
    load = re.findall('(.*)(--load [^-^ ]+)', arg_str)
    model = re.findall('(.*)(--model [^-^ ]+)', arg_str)
    for l in [load, model]:
        for s in l:
            accept_str += s[1]

    parser = argparse.ArgumentParser(description='Main args', )
    parser.add_argument('--load', type=str, default='null')
    parser.add_argument('--model', type=str, default='null')

    return parser.parse_args(accept_str.split(' '))


def train_or_test(args: argparse.Namespace):
    if args.load != 'null':
        current_args = args
        try:
            arg_paths = [os.path.join(current_args.load, item) for item in os.listdir(current_args.load) if (item.endswith('args.npy') or item.endswith('args.json'))]
            save_args = M.models.base.ArgParse.load(arg_paths)
        except IndexError as e:
            save_args = current_args
        model = save_args.model
        args = save_args
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

    elif model == 'image':
        trainingStructure = M.IMAGE.IMAGEStructure
    
    # elif model == 'imagelite':
    #     trainingStructure = M.IMAGE.IMAGELite

    elif model == 'vb':
        trainingStructure = M.Vertical.VIrisBeta

    elif model == 'viris':
        trainingStructure = M.Vertical.VIris

    trainingStructure(sys.argv).run_train_or_test()
    
def main():
    args = get_args()
    train_or_test(args)


if __name__ == "__main__":
    main()
