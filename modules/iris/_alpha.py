"""
@Author: Conghao Wong
@Date: 2021-06-21 15:01:50
@LastEditors: Conghao Wong
@LastEditTime: 2021-06-21 19:03:19
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from ..satoshi._alpha_transformer import SatoshiAlphaTransformer as SAT
from ..satoshi._alpha_transformer import SatoshiAlphaTransformerModel as SATM
from ..satoshi._args import SatoshiArgs


class IrisAlphaModel(SATM):
    def __init__(self, Args, training_structure=None, *args, **kwargs):
        super().__init__(Args, training_structure=training_structure, *args, **kwargs)


class IrisAlpha(SAT):
    def __init__(self, args, arg_type=SatoshiArgs):
        super().__init__(args, arg_type=arg_type)

