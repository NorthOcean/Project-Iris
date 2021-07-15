"""
@Author: Conghao Wong
@Date: 2021-06-21 15:01:50
@LastEditors: Conghao Wong
@LastEditTime: 2021-07-14 10:38:43
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import List

from ..satoshi._alpha_transformer import SatoshiAlphaTransformer as SAT
from ..satoshi._alpha_transformer import SatoshiAlphaTransformerModel as SATM


class IrisAlphaModel(SATM):
    def __init__(self, Args, training_structure=None, *args, **kwargs):
        super().__init__(Args, training_structure=training_structure, *args, **kwargs)


class IrisAlpha(SAT):
    def __init__(self, Args: List[str], *args, **kwargs):
        super().__init__(Args, *args, **kwargs)

