"""
@Author: Conghao Wong
@Date: 2021-06-11 10:01:50
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-12 19:13:27
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import argparse
import json
import re
from argparse import Namespace
from typing import Any, Dict, List, Union, overload

import numpy as np

from ._base import BaseArgs

    
def load(path: Union[str, List[str]]) -> Namespace:
    if type(path) == str:
        path = [path]

    if len([p := item for item in path if item.endswith('.json')]):
        with open(p, 'r') as f:
            args_dict = json.load(f)

        args = Namespace()
        args.__dict__ = args_dict
        return args

    elif len([p := item for item in path if item.endswith('.npy')]):
        return np.load(p, allow_pickle=True).item()

    else:
        raise ValueError(
            'Path {} invalid. Please input a vaild arg file path.'.format(path))

def save(path:str, args: Union[dict, Namespace, BaseArgs]):
    d = type(args)

    if d == dict:
        write_json(path, args)

    elif d == Namespace:
        write_json(path, args.__dict__)
    
    elif issubclass(BaseArgs, d):
        keys = args._arg_list
        values = [getattr(args, k) for k in keys]
        write_json(path, dict(zip(keys, values)))


def parse(argv: List[str], names: List[str], values: List[Any]) -> Namespace:
    parser = argparse.ArgumentParser(description='args', )

    argv_current = ''
    for s in argv:
        argv_current += s + ' '
    
    argv_filt = ''
    for name in names:
        if p := re.match('(.*)(--{} -?[^-^ ]+[^ ]*)( .*)'.format(name), 
                            argv_current):
            argv_filt += p[2] + ' '

    for name, value in zip(names, values):
        parser.add_argument('--' + name, 
                            type=type(value),
                            default=value)
    
    return parser.parse_args(argv_filt[:-1].split(' '))


def write_json(path:str, dic: dict):
    if path.endswith('json'):
        with open(path, 'w+') as f:
            json.dump(dic, f, separators=(',\n', ':'))
    
    else:
        raise NotImplementedError
