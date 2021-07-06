"""
@Author: Conghao Wong
@Date: 2021-07-04 10:29:31
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-05 09:18:40
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""


from typing import Any, Dict, List
import numpy as np
import json

from .. import base


class FormatProcess(base.BaseObject):
    """
    Process dataset files, and turn them into json format. 
    """
    def __init__(self):
        super().__init__()

    def load_csv(self, path: str) -> np.ndarray:
        """
        load csv trajectory data.
        csv contains four lines, and they are
            - frame id
            - agent id
            - x position
            - y position

        :param path: csv path
        :return dat: numpy dataset, shape = (n, 4)
        """
        return np.genfromtxt(path, delimiter=',').T

    def load_sdd_txt(self, path: str) -> List[List[float]]:
        """
        load sdd text dataset.
        each line contains:
            - agent id
            - x0 (bounding box)
            - y0 (bounding box)
            - x1 (bounding box)
            - y1 (bounding box)
            - frame id
            - true or fake data (`1` means fake)
            - not used
            - not used
            - class (not used)
        """

        results = []
        with open(path, 'r') as f:
            while data := f.readline():
                data = data.split(' ')
                if data[6] == '1':
                    continue
                    
                results.append([
                    float(data[5]),
                    float(data[0]),
                    (float(data[1]) + float(data[3]))/2,
                    (float(data[2]) + float(data[4]))/2,
                ])
        
        return results

    def transfer_to_dict(self, dat) -> Dict[str, list]:
        """
        Transfer np array dataset into dict form.
        Dict key = frame index, and dict value = trajectories.

        :param dat: dataset file, shape = (n_record, 4)
        :return dic: dic dataset, key = frame_index.
        """
        dic = dict()
        for item in dat:
            frame_id = str(int(item[0]))
            if not frame_id in dic.keys():
                dic[frame_id] = []

            dic[frame_id].append([item[1], item[2], item[3]])
        return dic

    def save_as_json(self, dataset: Dict[str, list], path: str):
        """
        Save dict dataset into json files.

        :param dataset: dict dataset, key = frame_index, and value is a list of [agent_id, pos_x, pos_y].
        :param path: save path
        """
        with open(path, 'w+') as f:
            json.dump(dataset, f, separators=(', ', ':'))

    def process(self, dataset_path: str, json_path: str):     
        if 'sdd' in dataset_path:
            dat = self.load_sdd_txt(dataset_path)
        elif 'eth' in dataset_path or 'ucy' in dataset_path:
            dat = self.load_csv(dataset_path)
        else:
            dat = None

        dic = self.transfer_to_dict(dat)
        self.save_as_json(dic, json_path)
        self.logger.info('Dataset file {} is transferred into {}.'.format(
            dataset_path, json_path
        ))
