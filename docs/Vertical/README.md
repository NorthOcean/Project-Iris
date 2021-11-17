<!--
 * @Author: Conghao Wong
 * @Date: 2021-08-05 15:51:15
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-11-17 11:27:35
 * @Description: file content
 * @Github: https://github.com/conghaowoooong
 * Copyright 2021 Conghao Wong, All Rights Reserved.
-->

# Codes for View Vertically: A Hierarchical Network for Trajectory Prediction via Fourier Spectrums

![$V^2$-Net](../../figs/vmethod.png)

## Abstract

Understanding and forecasting future trajectories of agents are critical for behavior analysis, robot navigation, autonomous cars, and other related applications.
Previous methods mostly treat trajectory prediction as time sequences generation.
In this work, we try for the first time to focus on agents' trajectories in a vertical view, i.e., the spectral domain for the trajectory prediction.
Different frequency bands in the trajectory spectrum can reflect different preferences hierarchically.
The low-frequency and high-frequency portions represent the coarse motion trends and the fine motion variations, respectively.
Accordingly, we propose a hierarchical network named $V^2$-Net containing two sub-networks to model and predict agents' behaviors with trajectory spectrums hierarchically.
The coarse-level keypoints estimation sub-network infers trajectory keypoints on the trajectory spectrum to present agents' motion trends.
The fine-level spectrum interpolation sub-network reconstructs trajectories from the spectrum of keypoints considering the detailed motion variations.
Experimental results show that $V^2$-Net improves the state-of-the-art performance by 14.2\% on ETH-UCY benchmark and by 17.2\% on the Stanford Drone Dataset.

## Requirements

The packages and versions used in our experiments include:

- tqdm==4.60.0
- biplist==1.0.3
- pytest==6.2.5
- numpy==1.19.3
- matplotlib==3.4.1
- tensorflow==2.5.0
- opencv_python

We recommend you install the above versions of the python packages in a virtual environment (like the `conda` environment), otherwise there *COULD* be other problems due to version conflicts.

Please run the following command to install the required packages:

```bash
pip install -r requirements.txt
```

## Training On Your Datasets

The `V^2-Net` contains two main sub-networks, the coarse-level keypoints estimation sub-network and the fine-level spectrum interpolation sub-network.
`V^2-Net` forecast agents' multiple stochastic trajectories end-to-end.
Considering that most of the loss function terms used to optimize the model work within one sub-network alone, we divide `V^2-Net` into `V^2-Net-a` and `V^2-Net-b`, and apply gradient descent separately for easier training.
You can train your own `V^2-Net` weights on your datasets by training each of these two sub-networks.
But don't worry, you can use it as a normal end-to-end model after training.

### Dataset

Before training `V^2-Net` on your own dataset, you can add your dataset information to the `datasets` directory.

- Dataset Splits File:

  It contains the dataset splits used for training and evaluation.
  For example, you can save the following python `dict` object as the `MyDataset.plist` (Maybe a python package like `biplist` is needed):

  ```python
  my_dataset = {
    'test': ['test_subset1'],
    'train': ['train_subset1', 'train_subset2', 'train_subset3'],
    'val': ['val_subset1', 'val_subset2'],
  }
  ```

- Sub-Dataset File:

  You should edit and put information about all sub-dataset, which you have written into the dataset splits file, into the `/datasets/subsets` directory.
  For example, you can save the following python `dict` object as the `test_subset1.plist`:

  ```python
  test_subset1 = {
    'dataset': 'test_subset1',    # name of that sub-dataset
    'dataset_dir': '....',        # root dir for your dataset csv file
    'order': [1, 0],              # x-y order in your csv file
    'paras': [1, 30],             # [your data fps, your video fps]
    'scale': 1,                   # scale when save visualization figs
    'video_path': '....',         # path for the corresponding video file 
  }
  ```

  Besides, all trajectories should be saved in the following `true_pos_.csv` format:

  - Size of the matrix is 4 x numTrajectoryPoints
  - The first row contains all the frame numbers
  - The second row contains all the pedestrian IDs
  - The third row contains all the y-coordinates
  - The fourth row contains all the x-coordinates

### `V^2-Net-a`

It is actually the coarse-level keypoints estimation sub-network.
To train the `V^2-Net-a`, you can pass the `--model va` argument to run the `main.py`.
You should also specify the indexes of the temporal keypoints in the predicted period.
For example, when you want to train a model that predicts future 12 frames of trajectories, and you would like to set $N_{key} = 3$ (which is the same as the basic settings in our paper), you can pass the `--p_index 3_7_11` argument when training.
Please note that indexes are start with `0`.
You can also try any other keypoints settings or combinations to train and obtain the `V^2-Net-a` that best fits your datasets.
Please refer to section `Args Used` to learn how other args work when training and evaluating.
Note that do not pass any value to `--load` when training, or it will start *evaluating* the loaded model.

For example, you can train the `V^2-Net-a` via the following minimum arguments:

```bash
cd REPO_ROOT_DIR
python main.py --model va --p_index 3_7_11 --test_set MyDataset
```

### `V^2-Net-b`

It is the fine-level spectrum interpolation sub-network.
You can pass the `--model vb` to run the training.
Please note that you should specify the number of temporal keypoints.
For example, you can pass the `--points 3` to train the corresponding sub-network that takes 3 temporal keypoints or their spectrums as the input.
Similar to the above `V^2-Net-a`, you can train the `V^2-Net-b` with the following minimum arguments:

```bash
python main.py --model vb --points 3 --test_set MyDataset
```

## Evaluation

You can use the following command to evaluate the `V^2-Net` performance end-to-end:

```bash
python main.py \
  --model vertical \
  --loada A_MODEL_PATH \
  --loadb B_MODEL_PATH
```

Where `A_MODEL_PATH` and `B_MODEL_PATH` are two sub-networks' weights.

## Pre-Trained Models

We have provided our pre-trained model weights to help you quickly evaluate the `V^2-Net` performance.
Click [here](drive.google.com) to download the zipped weights file.
Please unzip it to the project's root folder.
It contains model weights trained on `ETH-UCY` by the `leave-one-out` stragety, and on `SDD` via the dataset split method from SimAug.

```null
REPO_ROOT_DIR
  - pretrained_models
    - vertical
      - a_eth
      - a_hotel
      - a_sdd
      - a_univ
      - a_zara1
      - a_zara2
      - b_eth
      - b_hotel
      - b_sdd
      - b_univ
      - b_zara1
      - b_zara2
```

You can start the quick evaluation via the following commands:

```bash
for dataset in eth hotel univ zara1 zara2 sdd
  python main.py \
    --model vertical \
    --loada ./pretrained_models/vertical/a_${dataset} \
    --loadb ./pretrained_models/vertical/b_${dataset}
```

## Args Used

Please specific your customized args when training or testing your model through the following way:

```bash
python main.py --ARG_KEY1 ARG_VALUE2 --ARG_KEY2 ARG_VALUE2 --ARG_KEY3 ARG_VALUE3 ...
```

where `ARG_KEY` is the name of args, and `ARG_VALUE` is the corresponding value.
All args and their usages when training and testing are listed below. Args with `changable=True` means that their values can be changed after training.

<!-- DO NOT CHANGE THIS LINE -->
### Basic args

- `--batch_size`, type=`int`, changeable=`False`. Batch size when implementation.  Default value is `5000`.
- `--epochs`, type=`int`, changeable=`False`. Maximum training epochs.  Default value is `500`.
- `--force_set`, type=`str`, changeable=`True`. Force test dataset. Only works when evaluating when `test_mode` is `one`.  Default value is `'null'`.
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
- `--test_set`, type=`str`, changeable=`False`. Dataset used when training or evaluating.  Default value is `'zara1'`.
- `--test_step`, type=`int`, changeable=`False`. Epoch interval to run validation during training.  Default value is `3`.

### Prediction args

- `--K_train`, type=`int`, changeable=`False`. Number of multiple generations when training. This arg only works for `Generative Models`.  Default value is `10`.
- `--K`, type=`int`, changeable=`True`. Number of multiple generations when test. This arg only works for `Generative Models`.  Default value is `20`.
- `--add_noise`, type=`int`, changeable=`False`. Controls if add noise to training data. *This arg is not used in the current training structure.*  Default value is `0`.
- `--avoid_size`, type=`int`, changeable=`False`. Avoid size in grid cells when modeling social interaction.  Default value is `15`.
- `--draw_distribution`, type=`int`, changeable=`True`. Conrtols if draw distributions of predictions instead of points.  Default value is `0`.
- `--draw_results`, type=`int`, changeable=`True`. Controls if draw visualized results on video frames. Make sure that you have put video files into `./videos` according to the specific name way.  Default value is `0`.
- `--init_position`, type=`int`, changeable=`False`. ***DO NOT CHANGE THIS***.  Default value is `10000`.
- `--interest_size`, type=`int`, changeable=`False`. Interest size in grid cells when modeling social interaction.  Default value is `20`.
- `--lr`, type=`float`, changeable=`False`. Learning rate.  Default value is `0.001`.
- `--map_half_size`, type=`int`, changeable=`False`. Local map's half size.  Default value is `50`.
- `--max_batch_size`, type=`int`, changeable=`True`. Maximun batch size.  Default value is `20000`.
- `--obs_frames`, type=`int`, changeable=`False`. Observation frames for prediction.  Default value is `8`.
- `--pred_frames`, type=`int`, changeable=`False`. Prediction frames.  Default value is `12`.
- `--rotate`, type=`int`, changeable=`False`. Rotate dataset to obtain more available training data. This arg is the time of rotation, for example set to 1 will rotatetraining data 180 degree once; set to 2 will rotate them 120 degreeand 240 degree. *This arg is not used in the current training structure.*  Default value is `0`.
- `--sigma`, type=`float`, changeable=`True`. Sigma of noise. This arg only works for `Generative Models`.  Default value is `1.0`.
- `--step`, type=`int`, changeable=`True`. Frame interval for sampling training data.  Default value is `1`.
- `--test_mode`, type=`str`, changeable=`True`. Test settings, canbe `'one'` or `'all'` or `'mix'`. When set it to `one`, it will test the model on the `args.force_set` only; When set it to `all`, it will test on each of the test dataset in `args.test_set`; When set it to `mix`, it will test on all test dataset in `args.test_set` together.  Default value is `'mix'`.
- `--train_percent`, type=`str`, changeable=`False`. Percent of training samples used in each training dataset when training. Split with `_` if you want to specify the train percent on each dataset, for example `0.5_0.9_0.1`.  Default value is `'1.0_'`.
- `--use_extra_maps`, type=`int`, changeable=`True`. Controls if uses the calculated trajectory maps or the given trajectory maps. The model will load maps from `./dataset_npz/.../agent1_maps/trajMap.png` if set it to `0`, and load from `./dataset_npz/.../agent1_maps/trajMap_load.png` if set this item to `1`.  Default value is `0`.
- `--use_maps`, type=`int`, changeable=`False`. Controls if uses the trajectory maps or the social maps in the model.  Default value is `1`.
- `--window_size_expand_meter`, type=`float`, changeable=`False`. ***DO NOT CHANGE THIS***.  Default value is `10.0`.
- `--window_size_guidance_map`, type=`int`, changeable=`False`. Resolution of map (grids per meter).  Default value is `10`.

### Vertical args

- `--Kc`, type=`int`, changeable=`False`. Number of hidden categories used in alpha model.  Default value is `20`.
- `--check`, type=`int`, changeable=`True`. Controls whether apply the results choosing strategy.  Default value is `0`.
- `--loada`, type=`str`, changeable=`True`. Path for alpha model.  Default value is `'null'`.
- `--loadb`, type=`str`, changeable=`True`. Path for beta model.  Default value is `'null'`.
- `--p_index`, type=`str`, changeable=`False`. Index of predicted points at the first stage. Split with `_`. For example, `'0_4_8_11'`.  Default value is `'11'`.
- `--points`, type=`int`, changeable=`False`. Controls number of points (representative time steps) input to the beta model.  Default value is `1`.
<!-- DO NOT CHANGE THIS LINE -->
