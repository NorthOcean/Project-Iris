<!--
 * @Author: Conghao Wong
 * @Date: 2021-04-24 00:39:31
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-08-05 17:20:35
 * @Description: file content
 * @Github: https://github.com/conghaowoooong
 * Copyright 2021 Conghao Wong, All Rights Reserved.
-->

# Codes for Multi-Style Network for Trajectory Prediction

## Introduction

## Training

## Evaluation

## Pre-Trained Models

## Args Used

Please specific your customized args when training or test your model through the following way:

```bash
python main.py --ARG_KEY1 ARG_VALUE2 --ARG_KEY2 ARG_VALUE2 --ARG_KEY3 ARG_VALUE3 ...
```

where `ARG_KEY` is the name of args, and `ARG_VALUE` is the corresponding value. All args and their usages when training and test `MSN` are listed below. Args with `changable=True` means that their values can be changed after training.

<!-- DO NOT CHANGE THIS LINE -->
### Basic args

- `--batch_size`, type=`int`, changeable=`False`. Batch size when implementation.  Default value is `5000`.
- `--epochs`, type=`int`, changeable=`False`. Maximum training epochs.  Default value is `500`.
- `--force_set`, type=`str`, changeable=`True`. Force test dataset. Only works when `args.load` is not `'null'`. It is used to test models on their test datasets.  Default value is `'null'`.
- `--gpu`, type=`str`, changeable=`True`. Speed up training or test if you have at least one nvidia GPU. If you have no GPUs or want to run the code on your CPU, please set it to `-1`.  Default value is `'0'`.
- `--load`, type=`str`, changeable=`True`. Folder to load model. If set to `null`, it will start training new models according to other args.  Default value is `'null'`.
- `--log_dir`, type=`str`, changeable=`False`. Folder to save training logs and models. If set to `null`, logs will save at `args.save_base_dir/current_model`.  Default value is `'null'`.
- `--model_name`, type=`str`, changeable=`False`. Customized model name.  Default value is `'model'`.
- `--model`, type=`str`, changeable=`False`. Model type used to train or test.  Default value is `'none'`.
- `--restore`, type=`str`, changeable=`True`. Path to restore the pre-trained weights before training. It will not restore any weights if `args.restore == 'null'`  Default value is `'null'`.
- `--save_base_dir`, type=`str`, changeable=`False`. Base folder to save all running logs.  Default value is `'./logs'`.
- `--save_best`, type=`int`, changeable=`False`. Controls if save model with the best validation results when training.  Default value is `1`.
- `--save_format`, type=`str`, changeable=`False`. Model save format, canbe `tf` or `h5`. *This arg is now useless.*  Default value is `'tf'`.
- `--save_model`, type=`int`, changeable=`False`. Controls if save the final model at the end of training.  Default value is `1`.
- `--start_test_percent`, type=`float`, changeable=`False`. Set when to start validation during training. Range of this arg is `0 <= x <= 1`. Validation will start at `epoch = args.epochs * args.start_test_percent`.  Default value is `0.0`.
- `--test_set`, type=`str`, changeable=`False`. Test dataset. Only works on ETH-UCY dataset when training. (The `leave-one-out` training strategy.)  Default value is `'zara1'`.
- `--test_step`, type=`int`, changeable=`False`. Epoch interval to run validation during training.  Default value is `3`.

### Prediction args

- `--K_train`, type=`int`, changeable=`False`. Number of multiple generations when training. This arg only works for `Generative Models`.  Default value is `10`.
- `--K`, type=`int`, changeable=`True`. Number of multiple generations when test. This arg only works for `Generative Models`.  Default value is `20`.
- `--add_noise`, type=`int`, changeable=`False`. Controls if add noise to training data. *This arg is not used in the current training structure.*  Default value is `0`.
- `--avoid_size`, type=`int`, changeable=`False`. Avoid size in grid cells when modeling social interaction.  Default value is `15`.
- `--dataset`, type=`str`, changeable=`True`. Prediction dataset. Accept both `'ethucy'` and `'sdd'`.  Default value is `'ethucy'`.
- `--draw_distribution`, type=`int`, changeable=`True`. Conrtols if draw distributions of predictions instead of points.  Default value is `0`.
- `--draw_results`, type=`int`, changeable=`True`. Controls if draw visualized results on video frames. Make sure that you have put video files into `./videos` according to the specific name way.  Default value is `0`.
- `--init_position`, type=`int`, changeable=`False`. ***DO NOT CHANGE THIS***.  Default value is `10000`.
- `--interest_size`, type=`int`, changeable=`False`. Interest size in grid cells when modeling social interaction.  Default value is `20`.
- `--lr`, type=`float`, changeable=`False`. Learning rate.  Default value is `0.001`.
- `--map_half_size`, type=`int`, changeable=`False`. Local map's half size.  Default value is `50`.
- `--max_batch_size`, type=`int`, changeable=`True`. Maximun batch size.  Default value is `20000`.
- `--obs_frames`, type=`int`, changeable=`False`. Observation frames for prediction.  Default value is `8`.
- `--pred_frames`, type=`int`, changeable=`False`. Prediction frames.  Default value is `12`.
- `--prepare_type`, type=`str`, changeable=`True`. Prepare argument. ***Do Not Change it***.  Default value is `'test'`.
- `--rotate`, type=`int`, changeable=`False`. Rotate dataset to obtain more available training data. This arg is the time of rotation, for example set to 1 will rotatetraining data 180 degree once; set to 2 will rotate them 120 degreeand 240 degree. *This arg is not used in the current training structure.*  Default value is `0`.
- `--sigma`, type=`float`, changeable=`True`. Sigma of noise. This arg only works for `Generative Models`.  Default value is `1.0`.
- `--step`, type=`int`, changeable=`True`. Frame interval for sampling training data.  Default value is `1`.
- `--test_mode`, type=`str`, changeable=`True`. Test settings, canbe `'one'` or `'all'` or `'mix'`. When set it to `one`, it will test the model on the `args.test_set` only; When set it to `all`, it will test on each of the test dataset in `args.dataset`; When set it to `mix`, it will test on all test dataset in `args.dataset` together.  Default value is `'one'`.
- `--train_percent`, type=`str`, changeable=`False`. Percent of training samples used in each training dataset when training. Split with `_` if you want to specify the train percent on each dataset, for example `0.5_0.9_0.1`.  Default value is `'1.0_'`.
- `--use_extra_maps`, type=`int`, changeable=`True`. Controls if uses the calculated trajectory maps or the given trajectory maps. The model will load maps from `./dataset_npz/.../agent1_maps/trajMap.png` if set it to `0`, and load from `./dataset_npz/.../agent1_maps/trajMap_load.png` if set this item to `1`.  Default value is `0`.
- `--use_maps`, type=`int`, changeable=`False`. Controls if uses the trajectory maps or the social maps in the model.  Default value is `1`.
- `--window_size_expand_meter`, type=`float`, changeable=`False`. ***DO NOT CHANGE THIS***.  Default value is `10.0`.
- `--window_size_guidance_map`, type=`int`, changeable=`False`. Resolution of map (grids per meter).  Default value is `10`.

### MSN args

- `--K_train`, type=`int`, changeable=`False`. The number of hidden behavior categories in `AlphaModel`, or the number of multiple generations when training in `BetaModel`.  Default value is `10`.
- `--check`, type=`int`, changeable=`True`. Controls whether apply the results choosing strategy  Default value is `0`.
- `--loada`, type=`str`, changeable=`True`. Path for the first stage Destination Transformer  Default value is `'null'`.
- `--loadb`, type=`str`, changeable=`True`. Path for the second stage Interaction Transformer  Default value is `'null'`.
- `--loadc`, type=`str`, changeable=`True`. Path for the third stage model (Preserved)  Default value is `'null'`.
<!-- DO NOT CHANGE THIS LINE -->
