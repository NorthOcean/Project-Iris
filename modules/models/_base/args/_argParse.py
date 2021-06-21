"""
@Author: Conghao Wong
@Date: 2021-06-11 10:01:50
@LastEditors: Conghao Wong
@LastEditTime: 2021-06-11 15:26:12
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import json
from typing import Any, Dict, List, Union

import numpy as np


class Args():
    def __init__(self, init_args: Dict[str, Any]):
        super().__init__()

        for key in init_args.keys():
            setattr(self, key, init_args[key])


class ArgParse():
    @staticmethod
    def load(path: Union[str, List[str]]) -> Args:
        if type(path) == str:
            path = [path]

        if len([p := item for item in path if item.endswith('.json')]):
            with open(p, 'r') as f:
                args_dict = json.load(f)

            args = Args(args_dict)
            return args

        elif len([p := item for item in path if item.endswith('.npy')]):
            return np.load(p, allow_pickle=True).item()

        else:
            raise ValueError(
                'Path {} invalid. Please input a vaild arg file path.'.format(path))

    @staticmethod
    def save(path: str, args: Any):
        if path.endswith('json'):
            with open(path, 'w+') as f:
                json.dump(args.__dict__, f, separators=(',\n', ':'))
        
        elif path.endswith('npy'):
            np.save(path, args)
        
        else:
            raise NotImplementedError
