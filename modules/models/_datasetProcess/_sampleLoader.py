"""
@Author: Conghao Wong
@Date: 2021-07-05 09:15:58
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-06 15:59:21
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import json
import os
from typing import Dict, List

import numpy as np
import tensorflow as tf

from .. import base
from .._helpmethods import dir_check
from ._format import FormatProcess


class SampleLoader(base.BaseObject):
    """
    Load json datasets and make them into `tf.Dataset` 
    """

    def __init__(self, root_dataset_dir='./dataset_json/',
                 root_sample_dir='./samples/'):
        super().__init__()

        dir_check(root_dataset_dir)
        dir_check(root_sample_dir)

        self.ds_path = root_dataset_dir
        self.sample_path = root_sample_dir

    def load(self, path: str):
        """
        load csv dataset.

        :param path: path for csv file, endswith `.csv` or `.npy`
        :return dic: a dict of dataset, keys are frame index
        """
        if path.endswith('.json'):
            with open(path, 'r') as f:
                dic = json.load(f)

        elif path.endswith('.npy'):
            dic = np.load(path, allow_pickle=True)
        
        else:
            raise NotImplementedError

        return dic

    def sample(self, dic: Dict[str, List[List[float]]],
               sample_step: int,
               obs_length: int,
               pred_length: int,
               frame_rate: int,
               strategy='sampleFromPerson') -> List[Dict[int, List[float]]]:
        """
        Sample training data from json dict

        :param dic: json dict, keys are frame index
        :param sample_step: frame step of samples (in real frame index)
        :param obs_length: length of observation
        :param pred_length: length of prediction
        :param frame_rate: step between each frame
        :param strategy: sample strategy, canbe `'sampleFromBeginning'` or `sampleFromPerson`
        :return samples: a list of training trajectories
        """
        T = obs_length + pred_length

        samples = []
        if strategy == 'sampleFromBeginning':
            samples = self.sample_from_frames(
                frames=list(dic.keys()),
                dic=dic,
                sample_step=sample_step,
                sample_length=T,
                frame_rate=frame_rate
            )

        elif strategy == 'sampleFromPerson':
            person_appera = {}
            for frame, frame_data in dic.items():
                person_list = [int(f[0]) for f in frame_data]
                for person_index, person_id in enumerate(person_list):
                    if not person_id in person_appera.keys():
                        person_appera[person_id] = []

                    person_appera[person_id].append([frame, person_index])

            for person_id, frames in self.log_timebar(person_appera.items(), 
                                                      return_enumerate=False):
                frames, person_index = zip(*frames)
                samples_new = self.sample_from_frames(
                    frames, dic,
                    sample_step, T,
                    frame_rate, person_index)
                samples += samples_new

        return samples

    def sample_from_frames(self, frames: List[str],
                           dic: Dict[str, List[List[float]]],
                           sample_step: int,
                           sample_length: int,
                           frame_rate: int,
                           person_index: List[int] = None) -> List[Dict[int, List[float]]]:
        """
        sample trajectories from frame data via frame list and person list
        """
        samples = []
        if person_index:
            person_index_dict = dict(zip(frames, person_index))

        frames = np.sort(np.array(frames).astype(np.int))

        for frame in range(frames[0],
                           frames[-1] - sample_length*int(frame_rate),
                           int(sample_step)):

            t_list = list(range(
                frame,
                frame + sample_length*int(frame_rate),
                int(frame_rate)))

            trust = True
            values = []
            for t in t_list:
                if not t in frames:
                    trust = False
                    break
                else:
                    if person_index is None:
                        values.append(dic[str(t)])
                    else:
                        p_index = person_index_dict[str(t)]
                        value_original = dic[str(t)].copy()
                        value = [value_original.pop(p_index)]
                        values.append(value + value_original)

            if not trust:
                continue

            samples.append(dict(zip(t_list, values)))

        return samples

    def save(self, samples: List[Dict[int, List[List[float]]]], path: str):
        if path.endswith('.json'):
            with open(path, 'w+') as f:
                json.dump(samples, f, separators=(', ', ':'))
                
        elif path.endswith('.npy'):
            np.save(path, samples)
        
        else:
            raise NotImplementedError

    def restore_samples(self, dataset: str,
                        sample_step: int,
                        obs_length: int,
                        pred_length: int,
                        frame_rate: int,
                        strategy='sampleFromBeginning') -> List[Dict[int, List[float]]]:

        try:
            sample_path = os.path.join(self.sample_path,
                                       '{}.npy'.format(dataset))
            samples = self.load(sample_path)
            self.logger.info('Load dataset file from `{}` done.'.format(sample_path))

        except:
            self.logger.info('Samples from dataset {}'.format(dataset) + 
                             ' does not exist, start making...')
            ds_path = os.path.join(self.ds_path, '{}.json'.format(dataset))
            
            if not os.path.exists(ds_path):
                self.transfer_dataset(dataset)
                
            dataset_dict = self.load(ds_path)
            samples = self.sample(dataset_dict,
                                  sample_step,
                                  obs_length,
                                  pred_length,
                                  frame_rate,
                                  strategy)
            self.save(samples, sample_path)
            self.logger.info('Samples file saved at `{}`.'.format(sample_path))

        return samples

    def _transfer_dataset(self, label_path, target_path):
        p = FormatProcess()
        p.process(label_path, target_path)

    def transfer_dataset(self, dataset: str):
        raise NotImplementedError


    def make_dataset(self, samples: List[Dict[int, List[float]]],
                     *args, **kwargs) -> tf.data.Dataset:
        raise NotImplementedError
