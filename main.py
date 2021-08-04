"""
@Author: Conghao Wong
@Date: 2019-12-20 09:38:24
@LastEditors: Conghao Wong
@LastEditTime: 2021-08-02 11:17:17
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import os
import sys

import modules as M

# Remove tensorflow log info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train_or_test(args: M.models.base.Args):
    model = args.model

    if model == 'msn':
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

    elif model == 'viris':
        s = M.Vertical.VIris

    elif model == 'virisg':
        s = M.Vertical.VIrisG

    else:
        raise NotImplementedError(
            'model type `{}` is not supported.'.format(model))

    s(sys.argv).run_train_or_test()


if __name__ == "__main__":
    args = M.models.base.Args(sys.argv)
    train_or_test(args)
