"""
@Author: Conghao Wong
@Date: 2021-07-09 10:50:39
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-09 16:33:29
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""


from typing import List

from ..satoshi._args import SatoshiArgs


class VArgs(SatoshiArgs):
    def __init__(self, args: List[str]):
        super().__init__(args)
