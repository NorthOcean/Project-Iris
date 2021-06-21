'''
Author: Conghao Wong
Date: 2021-03-25 14:49:24
LastEditors: Conghao Wong
LastEditTime: 2021-03-26 14:37:53
Description: file content
'''

import numpy as np
import re
from typing import Tuple, List, Dict

LOG_PATH = './test_log.txt'


class SatoshiResultsManager():
    def __init__(self, file_path):
        super().__init__()
        self.start_flag = 'Satoshi,'
        self.all_lines = read_log_file(file_path, self.start_flag)
        self.results = self.sort_lines()
        print('!')
    
    def sort_lines(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        results = {}
        for line in self.all_lines:
            K = int(re.findall('_K[0-9]*[0-9]_', line)[0].split('_')[1][1:])
            H = int(re.findall('_H[0-9]*[0-9]_', line)[0].split('_')[1][1:])
            dataset = line.split(', ')[3]
            if not dataset in results.keys():
                results[dataset] = {}
            results[dataset]['{},{}'.format(K, H)] = eval(re.findall('{.*}', line)[0])
        return results

    def __call__(self, dataset, K, H, key='ADE'):
        return self.results[dataset]['{},{}'.format(K, H)][key]


def read_log_file(file_path, start_flag) -> List[str]:
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    results = [line for line in lines if line.startswith(start_flag)]
    return results


def sort_to_matrix(manager:SatoshiResultsManager, dataset, K_list, H_list):
    ade = np.zeros([len(K_list), len(H_list)])
    fde = ade.copy()

    for k, K in enumerate(K_list):
        for h, H in enumerate(H_list):
            ade[k, h] = manager(dataset, K, H, 'ADE')
            fde[k, h] = manager(dataset, K, H, 'FDE')
    
    return ade, fde


a = SatoshiResultsManager(LOG_PATH)
ade, fde = sort_to_matrix(a, 'univ', [5, 8, 10, 15, 20, 30], [0, 1, 3, 5, 8])
np.savetxt('./logresults.txt', np.concatenate([ade, fde]))
print('!')
    