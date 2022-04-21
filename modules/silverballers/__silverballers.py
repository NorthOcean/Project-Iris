"""
@Author: Conghao Wong
@Date: 2021-12-22 11:30:19
@LastEditors: Conghao Wong
@LastEditTime: 2022-04-21 11:01:47
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from .__baseSilverballers import Silverballers
from .agents import Agent47Model
from .agents import Agent47CModel
from .handlers.__burnwood import BurnwoodModel
from .handlers.__burnwoodC import BurnwoodCModel


class Silverballers47(Silverballers):

    def __init__(self, Args: list[str], *args, **kwargs):

        self.set_models(agentModel=Agent47Model,
                        handlerModel=BurnwoodCModel)

        super().__init__(Args, *args, **kwargs)


class Silverballers47C(Silverballers):

    def __init__(self, Args: list[str], *args, **kwargs):

        self.set_models(agentModel=Agent47CModel,
                        handlerModel=BurnwoodCModel)

        super().__init__(Args, *args, **kwargs)
