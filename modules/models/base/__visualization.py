"""
@Author: Conghao Wong
@Date: 2021-04-09 09:10:39
@LastEditors: Conghao Wong
@LastEditTime: 2022-04-21 10:59:28
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

import cv2
import numpy as np


class Visualization():
    """
    Visualization
    -------------
    Visualize results on video datasets

    Properties
    ----------
    ```python
    >>> self.video_capture  # Video capture
    >>> self.video_paras    # a list of [sample_step, frame_rate]
    >>> self.video_weights  # weights to tansfer real scale to pixel
    ```

    Methods
    -------
    ```python
    # setup video parameters
    >>> self.set_video(video_capture, video_paras, video_weights)

    # transfer coordinates from real scale to pixels
    >>> self.real2pixel(real_pos)

    # Add a png file to another file
    >>> Visualization.add_png_to_source(
            source:np.ndarray,
            png:np.ndarray,
            position,
            alpha=1.0)
    ```
    """

    def __init__(self):
        self._vc = None
        self._paras = None
        self._weights = None

    @property
    def video_capture(self) -> cv2.VideoCapture:
        return self._vc

    @property
    def video_paras(self):
        return self._paras

    @property
    def video_weights(self):
        return self._weights

    def set_video(self, video_capture: cv2.VideoCapture,
                  video_paras: list[int],
                  video_weights: list):

        self._vc = video_capture
        self._paras = video_paras
        self._weights = video_weights

    def real2pixel(self, real_pos):
        """
        Transfer coordinates from real scale to pixels.

        :param real_pos: coordinates, shape = (n, 2) or (k, n, 2)
        :return pixel_pos: coordinates in pixels
        """
        weights = self.video_weights

        if type(real_pos) == list:
            real_pos = np.array(real_pos)

        if len(real_pos.shape) == 2:
            real_pos = real_pos[np.newaxis, :, :]

        all_results = []
        for step in range(real_pos.shape[1]):
            r = real_pos[:, step, :]
            if len(weights) == 4:
                result = np.column_stack([
                    weights[2] * r.T[1] + weights[3],
                    weights[0] * r.T[0] + weights[1],
                ]).astype(np.int32)
            else:
                H = weights[0]
                real = np.ones([r.shape[0], 3])
                real[:, :2] = r
                pixel = np.matmul(real, np.linalg.inv(H))
                pixel = pixel[:, :2].astype(np.int32)
                result = np.column_stack([
                    weights[1] * pixel.T[0] + weights[2],
                    weights[3] * pixel.T[1] + weights[4],
                ]).astype(np.int32)
            
            all_results.append(result)
        
        return np.array(all_results)

    def draw(self, *args, **kwargs):
        raise NotImplementedError(
            'Please rewrite this method in your subclass')

    @staticmethod
    def add_png_to_source(source: np.ndarray, 
                          png: np.ndarray, 
                          position, 
                          alpha=1.0):

        yc, xc = position
        xp, yp, _ = png.shape
        xs, ys, _ = source.shape
        x0, y0 = [xc-xp//2, yc-yp//2]

        if png.shape[-1] == 4:
            png_mask = png[:, :, 3:4]/255
            png_file = png[:, :, :3]
        else:
            png_mask = np.ones_like(png)
            png_file = png

        if x0 >= 0 and y0 >= 0 and x0 + xp <= xs and y0 + yp <= ys:
            source[x0:x0+xp, y0:y0+yp, :3] = \
                (1.0 - alpha * png_mask) * source[x0:x0+xp, y0:y0+yp, :3] + \
                alpha * png_mask * png_file

            if source.shape[-1] == 4:
                source[x0:x0+xp, y0:y0+yp, 3:4] = \
                    np.minimum(source[x0:x0+xp, y0:y0+yp, 3:4] +
                               255 * alpha * png_mask, 255)
        return source

    @staticmethod
    def add_png_value(source, png, position, alpha=1.0):
        yc, xc = position
        xp, yp, _ = png.shape
        xs, ys, _ = source.shape
        x0, y0 = [xc-xp//2, yc-yp//2]

        if x0 >= 0 and y0 >= 0 and x0 + xp <= xs and y0 + yp <= ys:
            source[x0:x0+xp, y0:y0+yp, :3] = \
                source[x0:x0+xp, y0:y0+yp, :3] + png[:, :, :3] * alpha * png[:, :, 3:]/255
        
        return source

