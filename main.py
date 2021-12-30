"""
@Author: Conghao Wong
@Date: 2019-12-20 09:38:24
@LastEditors: Conghao Wong
@LastEditTime: 2021-12-30 14:40:14
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import os
import sys

import modules as M

# Remove tensorflow log info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train_or_test(args: M.models.base.Args, 
                  force_args=None):
                  
    model = args.model

    if model == 'test':
        s = M.linear.LinearStructure

    elif model == 'msn':
        s = M.msn.MSN_D

    elif model == 'msng':
        s = M.msn.MSN_G

    elif model in ['msna', 'sat']:
        s = M.msn.MSNAlpha

    elif model in ['msnb', 'sbt']:
        s = M.msn.MSNBeta_D

    elif model in ['msnc', 'ibc']:
        s = M.msn.MSNBeta_G

    elif model == 'va':
        s = M.vertical.VIrisAlpha

    elif model == 'vb':
        s = M.vertical.VIrisBeta

    elif model == 'vc':
        s = M.vertical.VIrisBetaG

    elif model == 'vag':
        s = M.vertical.VIrisAlphaG

    elif model in ['viris', 'vertical']:
        s = M.vertical.VIris

    elif model == 'virisg':
        s = M.vertical.VIrisG

    elif model == 'agent47':
        s = M.silverballers.agents.Agent47
    
    elif model == 'agent47C':
        s = M.silverballers.agents.Agent47C

    elif model == 'burnwood':
        s = M.silverballers.handlers.Burnwood

    elif model == 'burnwoodC':
        s = M.silverballers.handlers.BurnwoodC

    elif model == 'sb47':
        s = M.silverballers.Silverballers47

    elif model == 'sb47C':
        s = M.silverballers.Silverballers47C

    else:
        raise NotImplementedError(
            'model type `{}` is not supported.'.format(model))

    if not force_args:
        _args = sys.argv
    else:
        _args = force_args

    s(_args).run_train_or_test()
    


if __name__ == "__main__":
    args = M.models.base.Args(sys.argv)
    train_or_test(args)
