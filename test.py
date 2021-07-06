"""
@Author: Conghao Wong
@Date: 2021-07-04 10:43:15
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-06 20:30:58
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import numpy as np
import modules.models as M
import modules as mod

if __name__ == '__main__':
    # p = M.datasetProcess.FormatProcess()
    # npydate = p.process(
    #     # './data/ucy/zara/zara01/true_pos_.csv',
    #     # './dataset_json/zara1.json')
    #     './data/sdd/little/video0/annotations.txt',
    #     './dataset_json/little0.json')

    # p = M.datasetProcess.JsonLoader()
    # # a = p.load('./dataset_json/zara1.json')
    # s = p.restore('zara1', sample_step=10, 
    #          obs_length=8, 
    #          pred_length=12, 
    #          frame_rate=10, 
    #          strategy='sampleFromPerson')
    #         #  strategy='sampleFromBeginning')
    # # p.save(s, './test.json')


    p = mod.IMAGE.DatasetLoader([200, 200], 10, [0.5, 1],
        './dataset_json/', './samples/')

    ds = p.restore_dataset(['hotel'], 8, 12, 1)

    print('!')