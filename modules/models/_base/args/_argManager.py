"""
@Author: Conghao Wong
@Date: 2020-11-20 09:11:33
@LastEditors: Conghao Wong
@LastEditTime: 2021-05-11 09:41:37
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import argparse
import copy


class BaseArgsManager():
    def __init__(self):
        # args end with '_C' means they can be changed when load arg files

        # environment settings
        self.gpu_C = ['0', 'Speed up training or test if you have at least one ' + \
            'nvidia GPU. Use `_` to separate if you want to use more than one ' + \
            'gpus. If you have no GPUs or want to run the code on your CPU, ' + \
            'please set it to `-1`.', 'g']
        self.verbose_C = [1, 'Set if print logs', 'v']

        # save and load settings
        self.save_base_dir = ['./logs', 'Base saving dir of logs.']
        self.save_format = ['tf', 'Model save format, canbe `tf` or `h5`.']
        self.log_dir = ['null', 'Log dir for saving logs. If set to `null`,' + \
            'logs will save at save_base_dir/current_model.']
        self.load_C = ['null', 'Log dir to load model. If set to `null`,' + \
            'it will start training new models according to arguments.', 'l']
        self.restore_C = ['null', 'Restore pre-trained models before training.']
        
        
    def args(self):
        self._do_not_update_list = []
        name_list = [name for name in dir(self) if (not name.startswith('_')) and (not name == 'args')]
        parser = argparse.ArgumentParser(description='Online Video Predict')
        for name in name_list:
            default_value = getattr(self, name)
            
            if name.endswith('_C'):
                name = name[:-2]
                self._do_not_update_list.append(name)

            if type(default_value) == list and len(default_value) >= 3:
                kargs = ('--' + name, '-' + default_value[2])
            else:
                kargs = ('--' + name,)

            if type(default_value) == list and len(default_value) >= 2:
                parser.add_argument(*kargs, type=type(default_value[0]), default=default_value[0], help=default_value[1])
            else:
                parser.add_argument(*kargs, type=type(default_value), default=default_value)

        return copy.copy(parser.parse_args())

    def _get_do_not_update_list(self):
        return [name[:-2] for name in dir(self) if name.endswith('_C')]

    def _get_arg_list(self):
        return [name[:-2] for name in dir(self) if name.endswith('_C')] + [name for name in dir(self) if not name.endswith('_C')]
        