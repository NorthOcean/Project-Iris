"""
Log Function
============

'''
Author: Conghao Wong
Date: 2020-12-30 16:43:41
LastEditors: Conghao Wong
LastEditTime: 2021-04-02 17:18:29
Description: file content
'''

Methods that use LogFunction
----------------------------
```python

import modules as mod
import online

mod.models.base.Structure.log_function = LogFunction
mod.models.prediction.MapManager.log_function = LogFunction
mod.models.prediction.DatasetManager.log_function = LogFunction
mod.models.prediction.DatasetsManager.log_function = LogFunction
online.FrameManager.log_function = LogFunction
online.UIVisualization.log_function = logFunction
online.RealTimeManager.log_function = logFunction
```
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

