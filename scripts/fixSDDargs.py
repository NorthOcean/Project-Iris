"""
@Author: Conghao Wong
@Date: 2021-06-11 09:25:43
@LastEditors: Conghao Wong
@LastEditTime: 2021-08-24 15:48:40
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import numpy as np
import json
import os

BASE_DIR = './VLogs'

for d in os.listdir(BASE_DIR):
    try:
        with open(os.path.join(BASE_DIR, d, 'args.json'), 'r') as f:
            args = json.load(f)

        if args['dataset'] == 'sdd':
            args['test_set'] = 'sdd'

            with open(os.path.join(BASE_DIR, d, 'args.json'), 'w+') as f:
                json.dump(args, f, separators=(',\n', ':'))

            print('Convert to json in {} done.'.format(d))
        
    except:
        continue