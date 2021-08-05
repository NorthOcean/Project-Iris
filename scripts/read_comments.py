"""
@Author: Conghao Wong
@Date: 2021-08-05 14:56:26
@LastEditors: Conghao Wong
@LastEditTime: 2021-08-05 15:38:16
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""


import re
from typing import List

# FILE = './modules/models/_base/args/args.py'
# FILE = './modules/models/_prediction/args.py'
FILE = './modules/MSN/_args.py'
MAX_SPACE = 20


def read_comments(file) -> List[str]:
    with open(file, 'r') as f:
        lines = f.readlines()

    lines = ''.join(lines)
    args = re.findall('@property[^@]*', lines)
    
    results = []
    for arg in args:
        name = re.findall('(def )(.+)(\()', arg)[0][1]
        dtype = re.findall('(-> )(.*)(:)', arg)[0][1]
        changable = re.findall('(changeable=)(.*)(\))', arg)[0][1]
        default = re.findall('(, )(.*)(, ch)', arg)[0][1]
        comments = re.findall('(""")([\S\s]+)(""")', arg)[0][1]
        comments = comments.replace('\n', ' ')
        for _ in range(MAX_SPACE):
            comments = comments.replace('  ', ' ')
        
        comments = re.findall('( *)(.*)( *)', comments)[0][1]
        
        s = '- `--{}`, type=`{}`, changeable=`{}`. {} Default value is `{}`.'.format(name, dtype, changable, comments, default)
        results.append(s + '\n')
        print(s)
    
    return results


if __name__ == '__main__':
    read_comments(FILE)