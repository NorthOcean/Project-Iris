'''
Author: Conghao Wong
Date: 2020-08-20 23:05:05
LastEditors: Conghao Wong
LastEditTime: 2021-04-13 15:08:24
Description: file content
'''

from typing import Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf

from ... import base
from ..agent._agentManager import TrainAgentManager
from ..dataset._trainManager import PredictionDatasetManager


SMALL_POINTS = True
OBS_IMAGE = './vis_pngs/obs_small.png' if SMALL_POINTS else './vis_pngs/obs.png'
GT_IMAGE = './vis_pngs/gt_small.png' if SMALL_POINTS else './vis_pngs/gt.png'
PRED_IMAGE = './vis_pngs/pred_small.png' if SMALL_POINTS else './vis_pngs/pred.png'
DISTRIBUTION_IMAGE = './vis_pngs/dis.png'

    
class TrajVisualization(base.Visualization):
    def __init__(self, dataset):
        super().__init__()
        
        if dataset:
            self.DM = PredictionDatasetManager()(dataset)
            self.set_video(video_capture=cv2.VideoCapture(self.DM.video_path),
                        video_paras=self.DM.paras,
                        video_weights=self.DM.weights)

        self.obs_file = cv2.imread(OBS_IMAGE, -1)
        self.pred_file = cv2.imread(PRED_IMAGE, -1)
        self.gt_file = cv2.imread(GT_IMAGE, -1)
        self.dis_file = cv2.imread(DISTRIBUTION_IMAGE, -1)

        self.conv_layer = tf.keras.layers.Conv2D(
            1, (20, 20), (1, 1), 'same', 
            kernel_initializer=tf.initializers.constant(1/(20*20)))

        # color bar in BGR format
        # rgb(0, 0, 178) -> rgb(252, 0, 0) -> rgb(255, 255, 10)
        self.color_bar = np.column_stack([
            np.interp(np.arange(256),
                      np.array([0, 127, 255]),
                      np.array([178, 0, 10])),
            np.interp(np.arange(256),
                      np.array([0, 127, 255]),
                      np.array([0, 0, 255])),
            np.interp(np.arange(256),
                      np.array([0, 127, 255]),
                      np.array([0, 252, 255])),
        ])

    def draw(self, agents: List[TrainAgentManager],
             frame_name,
             save_path='null',
             show_img=False,
             draw_distribution=False,
             focus_mode=False,
             draw_relations=False):
        """
        Draw trajecotries on images.

        :param agents: a list of agent managers (`TrainAgentManager`)
        :param frame_name: name of the frame to draw on
        :param save_path: save path
        :param show_img: controls if show results in opencv window
        :draw_distrubution: controls if draw as distribution for generative models
        :focus_model: controls if highlight someone, canbe `False` or a one-hot array
        :draw_relations: controls if draw relations between the focused agent and others
        """
        obs_frame = frame_name
        time = 1000 * obs_frame / self.video_paras[1]
        self.video_capture.set(cv2.CAP_PROP_POS_MSEC, time - 1)
        _, f = self.video_capture.read()

        if type(focus_mode) == bool and (not focus_mode):
            for agent in agents:
                obs = self.real2pixel(agent.traj)
                pred = self.real2pixel(agent.pred)
                gt = self.real2pixel(agent.groundtruth) if len(
                    agent.groundtruth) else None
                f = self._visualization(
                    f, obs, gt, pred, draw_distribution, alpha=1.0)
        else:
            for agent, label in zip(agents, focus_mode):
                obs = self.real2pixel(agent.traj)
                pred = self.real2pixel(agent.pred)
                gt = self.real2pixel(agent.groundtruth) if len(
                    agent.groundtruth) else None
                f = self._visualization(
                    f, obs, gt, pred, draw_distribution, alpha=0.3 if label == 0 else 1.0)

        if not type(draw_relations) == bool:
            for point in draw_relations:
                f = draw_relation(f, self.real2pixel(
                    [agents[np.where(np.array(focus_mode) == 1)[0][0]].traj[-1], point]), self.gt_file)

        f = cv2.putText(f, self.DM.dataset + ' ' + str(int(frame_name)).zfill(6),
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        if not type(focus_mode) == bool:
            f = cv2.putText(f, 'Focus Mode', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        if self.DM.scale > 1:
            original_shape = f.shape
            f = cv2.resize(
                f, (int(original_shape[1]/self.DM.scale), int(original_shape[0]/self.DM.scale)))

        if show_img:
            cv2.namedWindow(self.DM.dataset, cv2.WINDOW_NORMAL |
                            cv2.WINDOW_KEEPRATIO)
            f = f.astype(np.uint8)
            cv2.imshow(self.DM.dataset, f)
            cv2.waitKey(80)

        else:
            cv2.imwrite(save_path, f)

    def draw_video(self, agent: TrainAgentManager, save_path, interp=True, indexx=0, draw_distribution=False):
        _, f = self.video_capture.read()
        video_shape = (f.shape[1], f.shape[0])

        frame_list = (np.array(agent.frame_list).astype(
            np.float32)).astype(np.int32)
        frame_list_original = frame_list

        if interp:
            frame_list = np.array(
                [index for index in range(frame_list[0], frame_list[-1]+1)])

        video_list = []
        times = 1000 * frame_list / self.video_paras[1]

        obs = self.real2pixel(agent.traj)
        gt = self.real2pixel(agent.groundtruth)
        pred = self.real2pixel(agent.pred)

        # # load from npy file
        # pred = np.load('./figures/hotel_{}_stgcnn.npy'.format(indexx)).reshape([-1, 2])
        # pred = self.real2pixel(np.column_stack([
        #     pred.T[0],  # sr: 0,1; sgan: 1,0; stgcnn: 1,0
        #     pred.T[1],
        # ]), traj_weights)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        VideoWriter = cv2.VideoWriter(
            save_path, fourcc, self.video_paras[1], video_shape)

        for time, frame in zip(times, frame_list):
            self.video_capture.set(cv2.CAP_PROP_POS_MSEC, time - 1)
            _, f = self.video_capture.read()

            # draw observations
            for obs_step in range(agent.obs_length):
                if frame >= frame_list_original[obs_step]:
                    f = base.Visualization.add_png_to_source(
                        f, self.obs_file, obs[obs_step])

            # draw predictions
            if frame >= frame_list_original[agent.obs_length]:
                f = self._visualization(
                    f, pred=pred, draw_distribution=draw_distribution)

            # draw GTs
            for gt_step in range(agent.obs_length, agent.total_frame):
                if frame >= frame_list_original[gt_step]:
                    f = base.Visualization.add_png_to_source(
                        f, self.gt_file, gt[gt_step - agent.obs_length])

            video_list.append(f)
            VideoWriter.write(f)

    def _visualization(self, f: np.ndarray, obs=None, gt=None, pred=None, draw_distribution: int =None, alpha=1.0):
        """
        Draw one agent's observations, predictions, and groundtruths.

        :param f: image file
        :param obs: (optional) observations in *pixel* scale
        :param gt: (optional) ground truth in *pixel* scale
        :param pred: (optional) predictions in *pixel* scale
        :param draw_distribution: controls if draw as a distribution
        :param alpha: alpha channel coefficient
        """
        f_original = f.copy()
        f = np.zeros([f.shape[0], f.shape[1], 4])
        if obs is not None:
            f = draw_traj(f, obs, self.obs_file, 
                          color=(255, 255, 255), 
                          width=3, alpha=alpha)

        if gt is not None:
            f = draw_traj(f, gt, self.gt_file,
                          color=(255, 255, 255), 
                          width=3, alpha=alpha)

        f = base.Visualization.add_png_to_source(f_original, f, 
                                                 [f.shape[1]//2, f.shape[0]//2], 
                                                 alpha)
        if pred is not None:
            background = np.zeros(f.shape[:2] + (4,))
            if draw_distribution == 1:
                f1 = draw_dis(background, pred.reshape([-1, 2]), 
                              self.dis_file, self.color_bar, 
                              alpha=0.5)
            
            elif draw_distribution == 2:
                all_steps = pred.shape[0]
                for index, step in enumerate(pred):
                    f1 = draw_dis(background, step, self.dis_file, 
                                  index/all_steps * self.color_bar,
                                  alpha=1.0)
            
            if draw_distribution > 0:
                f_smooth = self.conv_layer(np.transpose(f1.astype(np.float32), [2, 0, 1])[:, :, :, np.newaxis]).numpy()
                f_smooth = np.transpose(f_smooth[:, :, :, 0], [1, 2, 0])
                f1 = f_smooth

            else:
                for p in pred.reshape([-1, 2]):
                    f1 = base.Visualization.add_png_to_source(
                        f, self.pred_file, p, alpha=1.0)

        
        return base.Visualization.add_png_to_source(
            f, f1, 
            [f.shape[1]//2, f.shape[0]//2], 
            alpha=0.8)


def draw_traj(source, trajs, png_file, color=(255, 255, 255), width=3, alpha=1.0):
    """
    Draw lines and points.
    `color` in (B, G, R)
    """
    if (len(s := trajs.shape) == 3) and (s[1] == 1):
        trajs = trajs[:, 0, :]

    if len(trajs) >= 2:
        for left, right in zip(trajs[:-1, :], trajs[1:, :]):
            cv2.line(source, (left[0], left[1]),
                     (right[0], right[1]), color, width)
            source[:, :, 3] = alpha * 255 * source[:, :, 0]/color[0]
            source = base.Visualization.add_png_to_source(
                source, png_file, left)

        source = base.Visualization.add_png_to_source(source, png_file, right)
    else:
        source = base.Visualization.add_png_to_source(
            source, png_file, trajs[0])

    return source


def draw_dis(source, trajs, png_file, color_bar: np.ndarray, alpha=1.0):
    dis = np.zeros([source.shape[0], source.shape[1], 3])
    for p in trajs:
        dis = base.Visualization.add_png_value(dis, png_file, p, alpha)
    dis = dis[:, :, -1]

    if not dis.max() == 0:
        dis = dis ** 0.2
        alpha_channel = (255 * dis/dis.max()).astype(np.int32)
        color_map = color_bar[alpha_channel]
        distribution = np.concatenate(
            [color_map, np.expand_dims(alpha_channel, -1)], axis=-1)
        source = base.Visualization.add_png_to_source(
            source, distribution, [source.shape[1]//2, source.shape[0]//2], alpha)
    
    return source


def draw_relation(source, points, png_file, color=(255, 255, 255), width=2):
    left = points[0]
    right = points[1]
    cv2.line(source, (left[0], left[1]), (right[0], right[1]), color, width)
    source = base.Visualization.add_png_to_source(source, png_file, left)
    source = base.Visualization.add_png_to_source(source, png_file, right)
    return source
