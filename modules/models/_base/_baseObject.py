"""
@Author: Conghao Wong
@Date: 2021-04-15 09:26:41
@LastEditors: Conghao Wong
@LastEditTime: 2021-06-23 19:09:04
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf
import logging
# logging.basicConfig(level=logging.DEBUG)

from ._writefunction import LogFunction
from tqdm import tqdm


class BaseObject():

    log_function = LogFunction

    def __init__(self):
        super().__init__()
        
        # create a logger
        logger = logging.getLogger(name=type(self).__name__)
        logger.setLevel(logging.INFO)
        
        # add file handler
        fhandler = logging.FileHandler(filename='./test.log', mode='a')
        fhandler.setLevel(logging.INFO)

        # add terminal handler
        thandler = logging.StreamHandler()
        thandler.setLevel(logging.INFO)

        # add formatter
        fformatter = logging.Formatter('[%(asctime)s][%(levelname)s] `%(name)s`: %(message)s')
        fhandler.setFormatter(fformatter)
        
        tformatter = logging.Formatter('[%(levelname)s] `%(name)s`: %(message)s')
        thandler.setFormatter(tformatter)

        logger.addHandler(fhandler)
        logger.addHandler(thandler)

        self.logger = logger
        

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
