'''
Author: Conghao Wong
Date: 2021-04-01 20:28:00
LastEditors: Conghao Wong
LastEditTime: 2021-04-12 10:33:51
Description: file content
'''

import modules.models as M


class SatoshiArgs(M.prediction.TrainArgs):
    def __init__(self):
        super().__init__()

        self.loada_C = ['null', 'Path for Satoshi Alpha model', 'a']
        self.loadb_C = ['null', 'Path for Satoshi Beta model', 'b']
        self.H = [3, 'number of observed trajectories used']
        self.force_pred_frames_C = [-1,
                                    'force setting of predict frames when test']

    def args(self):
        _args = super().args()
        if _args.force_pred_frames != -1:
            _args.pred_frames = _args.force_pred_frames

        return _args


class SatoshiOnlineArgs(SatoshiArgs, M.prediction.OnlineArgs):
    def __init__(self):
        SatoshiArgs.__init__(self)
        M.prediction.OnlineArgs.__init__(self)
