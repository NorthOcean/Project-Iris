'''
@Author: ConghaoWong
@Date: 2019-12-20 09:39:02
LastEditors: Conghao Wong
LastEditTime: 2021-04-02 16:49:29
@Description: file content
'''

import modules as mod


if __name__ == '__main__':
    args = mod.models.prediction.TrainArgs().args()
    if '_' in args.test_set:
        args.prepare_type = args.test_set

    dm = mod.models.prediction.TrainDataManager(args, prepare_type=args.prepare_type)
