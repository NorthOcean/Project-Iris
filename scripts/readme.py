'''
Author: Conghao Wong
Date: 2021-01-13 11:50:08
LastEditors: Conghao Wong
LastEditTime: 2021-01-13 12:05:42
Description: file content
'''

import modules.models as M

arg_manager = M.managers.TrainArgsManager()

for item in dir(arg_manager):
    if ((not item.startswith('_')) and (not item == 'args')):
        value = getattr(arg_manager, item)
        print('- `--{}`{}:'.format(
            item.split('_C')[0],
            ' *or* `-{}`'.format(value[-1]) if len(value) >= 3 else '',
        ))
        print(value[1])
        print('Default is `{}`.\n'.format(value[0]))