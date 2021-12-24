"""
@Author: Conghao Wong
@Date: 2021-12-22 11:30:19
@LastEditors: Conghao Wong
@LastEditTime: 2021-12-23 10:40:00
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import List

from ._baseSilverballers import Silverballers
from .agents._agent6 import Agent6Model
from .agents._agent47 import Agent47Model
from .agents._agent47C import Agent47CModel
from .handlers._burnwood import BurnwoodModel
from .handlers._burnwoodC import BurnwoodCModel


class Silverballers47(Silverballers):

    def __init__(self, Args: List[str], *args, **kwargs):

        self.set_models(agentModel=Agent47Model,
                        handlerModel=BurnwoodCModel)

        super().__init__(Args, *args, **kwargs)


class Silverballers47C(Silverballers):

    def __init__(self, Args: List[str], *args, **kwargs):

        self.set_models(agentModel=Agent47CModel,
                        handlerModel=BurnwoodCModel)

        super().__init__(Args, *args, **kwargs)


class Silverballers6(Silverballers):

    def __init__(self, Args: List[str], *args, **kwargs):

        self.set_models(agentModel=Agent6Model,
                        handlerModel=BurnwoodModel)

        super().__init__(Args, *args, **kwargs)
