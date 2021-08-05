<!--
 * @Author: Conghao Wong
 * @Date: 2021-04-24 00:39:31
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-08-05 15:48:44
 * @Description: file content
 * @Github: https://github.com/conghaowoooong
 * Copyright 2021 Conghao Wong, All Rights Reserved.
-->

# Project Iris

## Args Used

<!-- DO NOT CHANGE THIS LINE -->
### Basic args

- `--batch_size`, type=`int`, changeable=`False`. Batch size when implementation.  Default value is `5000`.
- `--epochs`, type=`int`, changeable=`False`. Maximum training epochs.  Default value is `500`.
- `--force_set`, type=`str`, changeable=`True`. Force test dataset. Only works on ETH-UCY dataset when arg `load` is not `null`.  Default value is `'null'`.
- `--gpu`, type=`str`, changeable=`True`. Speed up training or test if you have at least one nvidia GPU. Use `_` to separate if you want to use more than one gpus. If you have no GPUs or want to run the code on your CPU, please set it to `-1`.  Default value is `'0'`.
- `--verbose`, type=`int`, changeable=`True`. Set if print logs  Default value is `1`.
- `--save_base_dir`, type=`str`, changeable=`False`. Base saving dir of logs.  Default value is `'./logs'`.
- `--save_best`, type=`int`, changeable=`False`. Controls if save model with the best val results when training.  Default value is `1`.
- `--save_format`, type=`str`, changeable=`False`. Model save format, canbe `tf` or `h5`. (Current useless)  Default value is `'tf'`.
- `--save_model`, type=`int`, changeable=`False`. Controls if save the final model at the end of training.  Default value is `1`.
- `--start_test_percent`, type=`float`, changeable=`False`. Set when to start val during training. Range of this arg is [0.0, 1.0]. The val will start at epoch = args.epochs * args.start_test_percent.  Default value is `0.0`.
- `--log_dir`, type=`str`, changeable=`False`. Log dir for saving logs. If set to `null`, logs will save at `save_base_dir/current_model`.  Default value is `'null'`.
- `--load`, type=`str`, changeable=`True`. Log folder to load model. If set to `null`, it will start training new models according to other args.  Default value is `'null'`.
- `--model`, type=`str`, changeable=`False`. Model used to train.  Default value is `'none'`.
- `--model_name`, type=`str`, changeable=`False`. Customized model name.  Default value is `'model'`.
- `--restore`, type=`str`, changeable=`True`. Path to the pre-trained models before training.  Default value is `'null'`.
- `--test_set`, type=`str`, changeable=`False`. Test dataset. Only works on ETH-UCY dataset.  Default value is `'zara1'`.
- `--test_step`, type=`int`, changeable=`False`. Epoch interval to run validation during training.  Default value is `3`.

### Prediction args

- `--obs_frames`, type=`int`, changeable=`False`. Observation frames for prediction.  Default value is `8`.
- `--pred_frames`, type=`int`, changeable=`False`. Prediction frames.  Default value is `12`.
- `--draw_results`, type=`int`, changeable=`True`. Controls if draw visualized results on videoframes. Make sure that you have put video files.  Default value is `0`.
- `--dataset`, type=`str`, changeable=`True`. Dataset. Can be `ethucy` or `sdd`.  Default value is `'ethucy'`.
- `--train_percent`, type=`str`, changeable=`False`. Percent of training data used in training datasets. Split with `_` if you want to specify each dataset, for example `0.5_0.9_0.1`.  Default value is `'1.0_'`.
- `--step`, type=`int`, changeable=`True`. Frame step for obtaining training data.  Default value is `1`.
- `--add_noise`, type=`int`, changeable=`False`. Controls if add noise to training data  Default value is `0`.
- `--rotate`, type=`int`, changeable=`False`. Rotate dataset to obtain more available training data. This arg is the time of rotation, for example set to 1 will rotatetraining data 180 degree once; set to 2 will rotate them 120 degreeand 240 degree.  Default value is `0`.
- `--test`, type=`int`, changeable=`False`. Controls if run test.  Default value is `1`.
- `--test_mode`, type=`str`, changeable=`True`. Test settings, canbe `one` or `all` or `mix`. When set to `one`, it will test the test_set only; When set to `all`, it will test on all test datasets of this dataset; When set to `mix`, it will test on one mix dataset that made up of alltest datasets of this dataset.  Default value is `'one'`.
- `--max_batch_size`, type=`int`, changeable=`True`. Maximun batch_size.  Default value is `20000`.
- `--dropout`, type=`float`, changeable=`False`. Dropout rate.  Default value is `0.5`.
- `--lr`, type=`float`, changeable=`False`. Learning rate.  Default value is `0.001`.
- `--diff_weights`, type=`float`, changeable=`False`. Parameter of linera prediction.  Default value is `0.95`.
- `--init_position`, type=`int`, changeable=`False`. ***DO NOT CHANGE THIS***.  Default value is `10000`.
- `--window_size_expand_meter`, type=`float`, changeable=`False`. ***DO NOT CHANGE THIS***.  Default value is `10.0`.
- `--window_size_guidance_map`, type=`int`, changeable=`False`. Resolution of map.(grids per meter)  Default value is `10`.
- `--avoid_size`, type=`int`, changeable=`False`. Avoid size in grids.  Default value is `15`.
- `--interest_size`, type=`int`, changeable=`False`. Interest size in grids.  Default value is `20`.
- `--map_half_size`, type=`int`, changeable=`False`. Local map's size.  Default value is `50`.
- `--K`, type=`int`, changeable=`True`. Number of multiple generations when test.  Default value is `20`.
- `--K_train`, type=`int`, changeable=`False`. Number of multiple generations when training.  Default value is `10`.
- `--sigma`, type=`float`, changeable=`True`. Sigma of noise.  Default value is `1.0`.
- `--draw_distribution`, type=`int`, changeable=`True`. Conrtols if draw distributions ofpredictions instead of points.  Default value is `0`.
- `--prepare_type`, type=`str`, changeable=`True`. Prepare argument. Do Not Change it.  Default value is `'test'`.
- `--use_maps`, type=`int`, changeable=`False`. Controls if uses the trajectory maps in models. Do not change it when test or training.  Default value is `1`.
- `--use_extra_maps`, type=`int`, changeable=`True`. Controls if uses the calculated trajectory maps or the given trajectory maps. The model will load maps from `./dataset_npz/.../agent1_maps/trajMap.png` if set it to `0`, and load from `./dataset_npz/.../agent1_maps/trajMap_load.png` if set this item to `1`.  Default value is `0`.

### MSN args

- `--loada`, type=`str`, changeable=`True`. Path for the first stage Destination Transformer  Default value is `'null'`.
- `--loadb`, type=`str`, changeable=`True`. Path for the second stage Interaction Transformer  Default value is `'null'`.
- `--loadc`, type=`str`, changeable=`True`. Path for the third stage model (Preserved)  Default value is `'null'`.
- `--linear`, type=`int`, changeable=`False`. Controls whether use linear prediction in the last stage  Default value is `0`.
- `--check`, type=`int`, changeable=`True`. Controls whether apply the results choosing strategy  Default value is `0`.
- `--K_train`, type=`int`, changeable=`False`. The number of hidden behavior categories in `AlphaModel`, or the number of multiple generations when training in `BetaModel`.  Default value is `10`.
<!-- DO NOT CHANGE THIS LINE -->
