"""
@Author: Conghao Wong
@Date: 2020-11-20 09:11:33
@LastEditors: Conghao Wong
@LastEditTime: 2021-06-24 14:35:31
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from ... import base


class BasePredictArgs(base.Args):
    def __init__(self):
        super().__init__()

        # prediction settings
        self.obs_frames = [8, 'Observation frames for prediction.']
        self.pred_frames = [12, 'Prediction frames.']


class TrainArgsManager(BasePredictArgs):
    def __init__(self):
        super().__init__()
        # environment settrings and test options

        # model settings
        self.draw_results_C = [0, 'Controls if draw visualized results on video' +
                               'frames. Make sure that you have put video files.']

        # dataset base settings
        self.dataset_C = ['ethucy', 'Dataset. Can be `ethucy` or `sdd`.']
        self.test_set = [
            'zara1', 'Test dataset. Only works on ETH-UCY dataset.', 's']
        self.force_set_C = ['null', 'Force test dataset. Only works on ' +
                            'ETH-UCY dataset when arg `load` is not `null`.', 'fs']

        # dataset training settings
        self.train_percent = ['1.0_', 'Percent of training data used in ' +
                              'training datasets. Split with `_` if you want to specify ' +
                              'each dataset, for example `0.5_0.9_0.1`.', 'tp']
        self.step = [1, 'Frame step for obtaining training data.']

        self.add_noise = [0, 'Controls if add noise to training data']
        self.rotate = [0, 'Rotate dataset to obtain more available training data.' +
                       'This arg is the time of rotation, for example set to 1 will rotate' +
                       'training data 180 degree once; set to 2 will rotate them 120 degree' +
                       'and 240 degree.']

        # test settings when training
        self.test = [1, 'Controls if run test.']
        self.start_test_percent = [0.0, 'Set when to start val during training.' +
                                   'Range of this arg is [0.0, 1.0]. The val will start at epoch = ' +
                                   'args.epochs * args.start_test_percent.']
        self.test_step = [3, 'Val step in epochs.']

        # test settings
        self.test_mode_C = ['one', 'Test settings, canbe `one` or `all` or `mix`.' +
                            'When set to `one`, it will test the test_set only;' +
                            'When set to `all`, it will test on all test datasets of this dataset;' +
                            'When set to `mix`, it will test on one mix dataset that made up of all' +
                            'test datasets of this dataset.']

        # training settings
        self.epochs = [500, 'Training epochs.']
        self.batch_size = [5000, 'Training batch_size.']
        self.max_batch_size_C = [20000, 'Maximun batch_size.']
        self.dropout = [0.5, 'Dropout rate.']
        self.lr = [1e-3, 'Learning rate.']

        # save/load settings
        self.model_name = ['model', 'Model\'s name when saving.']
        self.save_model = [1, 'Controls if save the model.']
        self.save_best = [1, 'Controls if save the best model when val.']

        # Linear args
        self.diff_weights = [0.95, 'Parameter of linera prediction.']

        # prediction model args
        self.model = ['gan', 'Model used to train. Canbe `l` or `bgm`.']

        # Social args
        self.init_position = [10000, '***DO NOT CHANGE THIS***.']
        self.window_size_expand_meter = [10.0, '***DO NOT CHANGE THIS***.']
        self.window_size_guidance_map = [10, 'Resolution of map.' +
                                         '(grids per meter)']
        self.avoid_size = [15, 'Avoid size in grids.']
        self.interest_size = [20, 'Interest size in grids.']

        # Guidance Map args
        self.map_half_size = [50, 'Local map\'s size.']

        # GCN args
        self.gcn_layers = [3, 'Number of GCN layers used in GAN model.']

        # GAN args
        self.K_C = [20, 'Number of multiple generation when test.']
        self.K_train = [10, 'Number of multiple generation when training.']
        self.sigma_C = [1.0, 'Sigma of noise.']
        self.draw_distribution_C = [0, 'Conrtols if draw distributions of' +
                                    'predictions instead of points.']

        # dataset test
        self.prepare_type_C = ['test', 'Prepare argument. Do Not Change it.']

        # Spring args
        self.spring_number = [4, 'Experimental.']
        self.focus_mode = [0, 'Experimental.']

        # Scene args


class OnlineArgsManager(TrainArgsManager):
    def __init__(self):
        super().__init__()

        self.wait_frames = 4

        self.guidance_map_limit = 10000
        self.order_C = [0, 1]     # Do Not Change

        self.draw_future = 0
        self.vis_C = 'show'

        self.img_save_base_path = './online_vis'

        self.focus_mode = 0
        self.run_frames = 1
