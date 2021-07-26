"""
@Author: Conghao Wong
@Date: 2019-12-20 09:38:24
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-22 20:43:16
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
    for name in ['load', 'model']:
        if  p := re.match('(.*)(--{} -?[^-^ ]+[^ ]*)( .*)'.format(name), 
                            arg_str):
            accept_str += p[2] + ''

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

    if model == 'msn':
        structure = M.MSN.MSN_D

    elif model == 'msng':
        structure = M.MSN.MSN_G
    
    elif model == 'msna':
        structure = M.MSN.MSNAlpha

    elif model == 'msnb':
        structure = M.MSN.MSNBeta_D

    elif model == 'msnc':
        structure = M.MSN.MSNBeta_G

    elif model == 'va':
        structure = M.Vertical.VIrisAlpha

    elif model == 'vb':
        structure = M.Vertical.VIrisBeta
    
    elif model == 'vc':
        structure = M.Vertical.VIrisBetaG

    elif model == 'viris':
        structure = M.Vertical.VIris
    
    elif model == 'virisg':
        structure = M.Vertical.VIrisG

    structure(sys.argv).run_train_or_test()
    
def main():
    args = get_args()
    train_or_test(args)


if __name__ == "__main__":
    main()
