"""
@Author: Conghao Wong
@Date: 2021-04-13 12:18:05
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-20 15:22:55
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from .baseObject import BaseObject


class Agent(BaseObject):
    """
    Agent
    -----
    One sample for training or evolution.
    It manages all model inputs and groundtruths.
    """

    def __init__(self):
        super().__init__()
