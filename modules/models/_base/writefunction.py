"""
@Author: Conghao Wong
@Date: 2020-12-30 16:43:41
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-16 10:32:51
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

class LogFunction():
    
    v = True
    log_function = print
    string = []
    max_line = 500

    @classmethod
    def write(cls, s, end='\n'):
        if (not len(s)) or (not cls.v):
            return
        if not '|' in s:
            s += end
        cls.log_function(s, end='')
        cls.string.append(s)
        if len(cls.string) >= cls.max_line:
            cls.string = cls.string[cls.max_line//2:]

    @classmethod
    def log(cls, s, end='\n'):
        cls.write(s, end)

