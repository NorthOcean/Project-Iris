"""
@Author: Conghao Wong
@Date: 2021-04-15 09:26:41
@LastEditors: Conghao Wong
@LastEditTime: 2021-05-10 11:29:42
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from ._writefunction import LogFunction
from tqdm import tqdm


class BaseObject():

    log_function = LogFunction

    def __init__(self):
        super().__init__()

    @classmethod
    def log(cls, s: str, end='\n'):
        cls.log_function.log(s, end)

    @classmethod
    def log_timebar(cls, inputs, text='', return_enumerate=True):
        log_function = cls.log_function
        try:
            itera = tqdm(inputs, desc=text, file=log_function)
        except:
            itera = tqdm(inputs, desc=text)

        if return_enumerate:
            return enumerate(itera)
        else:
            return itera

    @classmethod
    def log_parameters(cls, title='null', **kwargs):
        cls.log('>>> ' + title + ':')
        for key in kwargs:
            cls.log('    - {} is {}.'.format(
                key,
                kwargs[key].numpy() if type(kwargs[key]) == tf.Tensor
                else kwargs[key]
            ))
        cls.log('\n')

    @classmethod
    def log_bar(cls, percent, total_length=30):
        
        bar = (''.join('=' * (int(percent * total_length) - 1))
               + '>')
        return bar
