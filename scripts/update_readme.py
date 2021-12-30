"""
@Author: Conghao Wong
@Date: 2021-08-05 15:26:57
@LastEditors: Conghao Wong
@LastEditTime: 2021-12-30 14:54:51
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import re
from typing import List

from read_comments import read_comments

FLAG = '<!-- DO NOT CHANGE THIS LINE -->'
TARGET_FILE = './docs/{}/README.md'


def update(md_file, files: List[str], titles: List[str]):

    new_lines = []
    for f, title in zip(files, titles):
        new_lines += ['\n### {}\n\n'.format(title)]
        c = read_comments(f)
        c.sort()
        new_lines += c

    with open(md_file, 'r') as f:
        lines = f.readlines()
    lines = ''.join(lines)

    try:
        pattern = re.findall(
            '([\s\S]*)({})([\s\S]*)({})([\s\S]*)'.format(FLAG, FLAG), lines)[0]
        all_lines = list(pattern[:2]) + new_lines + list(pattern[-2:])

    except:
        flag_line = '{}\n'.format(FLAG)
        all_lines = [lines, flag_line] + new_lines + [flag_line]

    with open(md_file, 'w+') as f:
        f.writelines(all_lines)


if __name__ == '__main__':
    for model in ['MSN', 'Vertical', 'Silverballers']:
        files = ['./modules/models/_base/args/args.py',
                 './modules/models/_prediction/args.py',
                 './modules/{}/_args.py'.format(model)]
        titles = ['Basic args',
                  'Prediction args',
                  '{} args'.format(model)]
        update(TARGET_FILE.format(model), files, titles)
