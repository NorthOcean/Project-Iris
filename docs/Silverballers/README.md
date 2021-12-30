<!--
 * @Author: Conghao Wong
 * @Date: 2021-12-30 14:44:38
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-12-30 14:55:30
 * @Description: file content
 * @Github: https://github.com/conghaowoooong
 * Copyright 2021 Conghao Wong, All Rights Reserved.
-->

# Codes for Project-Silverballers

## Abstract

## Requirements

These versions of packages may be required:

- tqdm==4.60.0
- biplist==1.0.3
- pytest==6.2.5
- numpy==1.19.3
- matplotlib==3.4.1
- tensorflow==2.5.0
- opencv_python

We recommend you install the above versions of the python packages in a virtual environment (like the `conda` environment), otherwise there *COULD* be other problems due to version conflicts.

Please run the following command to install these required packages:

```bash
pip install -r requirements.txt
```

## Training On Your Datasets

## Evaluation

## Pre-Trained Weights

## Model Args

Please specific your customized args when training or testing your model by:

```bash
python main.py --ARG_KEY1 ARG_VALUE2 --ARG_KEY2 ARG_VALUE2 --ARG_KEY3 ARG_VALUE3 ...
```

where `ARG_KEY` is the name of args, and `ARG_VALUE` is the corresponding value.
All args and their usages when training and testing are listed below.
Args with `changable=False` means that their values can not be changed once the model has beed trained.

<!-- DO NOT CHANGE THIS LINE -->
### Basic args

- `--batch_size`, type=`int`, changeable=`False`.
  Batch size when implementation.
  The default value is `5000`.
- `--epochs`, type=`int`, changeable=`False`.
  Maximum training epochs.
  The default value is `500`.
- `--force_set`, type=`str`, changeable=`True`.
  Force test dataset. Only works when evaluating when `test_mode` is `one`.
  The default value is `'null'`.
- `--gpu`, type=`str`, changeable=`True`.
  Speed up training or test if you have at least one nvidia GPU. If you have no GPUs or want to run the code on your CPU, please set it to `-1`.
  The default value is `'0'`.
- `--load`, type=`str`, changeable=`True`.
  Folder to load model. If set to `null`, it will start training new models according to other args.
  The default value is `'null'`.
- `--log_dir`, type=`str`, changeable=`False`.
  Folder to save training logs and models. If set to `null`, logs will save at `args.save_base_dir/current_model`.
  The default value is `'null'`.
- `--model_name`, type=`str`, changeable=`False`.
  Customized model name.
  The default value is `'model'`.
- `--model`, type=`str`, changeable=`False`.
  Model type used to train or test.
  The default value is `'none'`.
- `--restore`, type=`str`, changeable=`True`.
  Path to restore the pre-trained weights before training. It will not restore any weights if `args.restore == 'null'`.
  The default value is `'null'`.
- `--save_base_dir`, type=`str`, changeable=`False`.
  Base folder to save all running logs.
  The default value is `'./logs'`.
- `--save_best`, type=`int`, changeable=`False`.
  Controls if save model with the best validation results when training.
  The default value is `1`.
- `--save_model`, type=`int`, changeable=`False`.
  Controls if save the final model at the end of training.
  The default value is `1`.
- `--start_test_percent`, type=`float`, changeable=`False`.
  Set when to start validation during training. Range of this arg is `0 <= x <= 1`. Validation will start at `epoch = args.epochs * args.start_test_percent`.
  The default value is `0.0`.
- `--test_set`, type=`str`, changeable=`False`.
  Dataset used when training or evaluating.
  The default value is `'zara1'`.
- `--test_step`, type=`int`, changeable=`False`.
  Epoch interval to run validation during training.
  The default value is `3`.

### Prediction args

- `--K_train`, type=`int`, changeable=`False`.
  Number of multiple generations when training. This arg only works for `Generative Models`.
  The default value is `10`.
- `--K`, type=`int`, changeable=`True`.
  Number of multiple generations when test. This arg only works for `Generative Models`.
  The default value is `20`.
- `--avoid_size`, type=`int`, changeable=`False`.
  Avoid size in grid cells when modeling social interaction.
  The default value is `15`.
- `--draw_distribution`, type=`int`, changeable=`True`.
  Conrtols if draw distributions of predictions instead of points.
  The default value is `0`.
- `--draw_results`, type=`int`, changeable=`True`.
  Controls if draw visualized results on video frames. Make sure that you have put video files into `./videos` according to the specific name way.
  The default value is `0`.
- `--init_position`, type=`int`, changeable=`False`.
  ***DO NOT CHANGE THIS***.
  The default value is `10000`.
- `--interest_size`, type=`int`, changeable=`False`.
  Interest size in grid cells when modeling social interaction.
  The default value is `20`.
- `--lr`, type=`float`, changeable=`False`.
  Learning rate.
  The default value is `0.001`.
- `--map_half_size`, type=`int`, changeable=`False`.
  Local map's half size.
  The default value is `50`.
- `--max_batch_size`, type=`int`, changeable=`True`.
  Maximun batch size.
  The default value is `20000`.
- `--obs_frames`, type=`int`, changeable=`False`.
  Observation frames for prediction.
  The default value is `8`.
- `--pred_frames`, type=`int`, changeable=`False`.
  Prediction frames.
  The default value is `12`.
- `--sigma`, type=`float`, changeable=`True`.
  Sigma of noise. This arg only works for `Generative Models`.
  The default value is `1.0`.
- `--step`, type=`int`, changeable=`True`.
  Frame interval for sampling training data.
  The default value is `1`.
- `--test_mode`, type=`str`, changeable=`True`.
  Test settings, canbe `'one'` or `'all'` or `'mix'`. When set it to `one`, it will test the model on the `args.force_set` only; When set it to `all`, it will test on each of the test dataset in `args.test_set`; When set it to `mix`, it will test on all test dataset in `args.test_set` together.
  The default value is `'mix'`.
- `--use_extra_maps`, type=`int`, changeable=`True`.
  Controls if uses the calculated trajectory maps or the given trajectory maps. The model will load maps from `./dataset_npz/.../agent1_maps/trajMap.png` if set it to `0`, and load from `./dataset_npz/.../agent1_maps/trajMap_load.png` if set this item to `1`.
  The default value is `0`.
- `--use_maps`, type=`int`, changeable=`False`.
  Controls if uses the trajectory maps or the social maps in the model.
  The default value is `1`.
- `--window_size_expand_meter`, type=`float`, changeable=`False`.
  ***DO NOT CHANGE THIS***.
  The default value is `10.0`.
- `--window_size_guidance_map`, type=`int`, changeable=`False`.
  Resolution of map (grids per meter).
  The default value is `10`.

### Silverballers args

- `--K`, type=`int`, changeable=`True`.
  Number of multiple generations when evaluating. The number of trajectories outputed for one agent is calculated by `N = args.K * Kc`, where `Kc` is the number of style channels (given by args in the agent model).
  The default value is `1`.
- `--K`, type=`int`, changeable=`True`.
  Number of multiple generations when evaluating. The number of trajectories predicted for one agent is calculated by `N = args.K * args.Kc`, where `Kc` is the number of style channels.
  The default value is `1`.
- `--Kc`, type=`int`, changeable=`False`.
  Number of style channels in `Agent` model.
  The default value is `20`.
- `--depth`, type=`int`, changeable=`False`.
  Depth of the random contract id.
  The default value is `16`.
- `--key_points`, type=`str`, changeable=`False`.
  A list of key-time-steps to be predicted in the agent model. For example, `'0_6_11'`.
  The default value is `'0_6_11'`.
- `--key_points`, type=`str`, changeable=`False`.
  A list of key-time-steps to be predicted in the handler model. For example, `'0_6_11'`. When setting it to `'null'`, it will start training by randomly sampling keypoints from all future moments.
  The default value is `'null'`.
- `--loada`, type=`str`, changeable=`True`.
  Path for agent model.
  The default value is `'null'`.
- `--loadb`, type=`str`, changeable=`True`.
  Path for handler model.
  The default value is `'null'`.
- `--metric`, type=`str`, changeable=`False`.
  Controls the metric used to save model weights when training. Accept either `'ade'` or `'fde'`.
  The default value is `'fde'`.
- `--points`, type=`int`, changeable=`False`.
  Controls the number of keypoints accepted in the handler model.
  The default value is `1`.
- `--preprocess`, type=`str`, changeable=`False`.
  Controls if running any preprocess before model inference. Accept a 3-bit-like string value (like `'111'`): - the first bit: `MOVE` trajectories to (0, 0); - the second bit: re-`SCALE` trajectories; - the third bit: `ROTATE` trajectories.
  The default value is `'111'`.
- `--use_maps`, type=`int`, changeable=`False`.
  Controls if using the trajectory maps and the social maps in the model. DO NOT change this arg manually.
  The default value is `1`.
<!-- DO NOT CHANGE THIS LINE -->
