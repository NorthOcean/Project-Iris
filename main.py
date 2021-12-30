"""
@Author: Conghao Wong
@Date: 2019-12-20 09:38:24
@LastEditors: Conghao Wong
@LastEditTime: 2021-12-30 10:15:30
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
        s = M.Linear.LinearStructure

    elif model == 'msn':
        s = M.MSN.MSN_D

    elif model == 'msng':
        s = M.MSN.MSN_G

    elif model in ['msna', 'sat']:
        s = M.MSN.MSNAlpha

    elif model in ['msnb', 'sbt']:
        s = M.MSN.MSNBeta_D

    elif model in ['msnc', 'ibc']:
        s = M.MSN.MSNBeta_G

    elif model == 'va':
        s = M.Vertical.VIrisAlpha

    elif model == 'vb':
        s = M.Vertical.VIrisBeta

    elif model == 'vc':
        s = M.Vertical.VIrisBetaG

    elif model == 'vag':
        s = M.Vertical.VIrisAlphaG

    elif model in ['viris', 'vertical']:
        s = M.Vertical.VIris

    elif model == 'virisg':
        s = M.Vertical.VIrisG

    elif model == 'agent47':
        s = M.Silverballers.agents.Agent47
    
    elif model == 'agent47C':
        s = M.Silverballers.agents.Agent47C

    elif model == 'burnwood':
        s = M.Silverballers.handlers.Burnwood

    elif model == 'burnwoodC':
        s = M.Silverballers.handlers.BurnwoodC

    elif model == 'sb47':
        s = M.Silverballers.Silverballers47

    elif model == 'sb47C':
        s = M.Silverballers.Silverballers47C

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
