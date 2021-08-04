"""
@Author: Conghao Wong
@Date: 2021-04-15 19:17:06
@LastEditors: Conghao Wong
@LastEditTime: 2021-08-04 14:54:24
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2021 Conghao Wong, All Rights Reserved.
"""

from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf


class Data():

    @classmethod
    @tf.function
    def process(cls, img: tf.Tensor, process_list: List[str]):
        """
        Apply preprocessings on images according to `process_list`.
        Accept key words:
        ```python
        ['REGULATION',
         'RANDOM_BRIGHTNESS',
         'RANDOM_FLIP',
         'RANDOM_CONTRAST',
         'RANDOM_HUE',
         'RANDOM_SATURATION',
         'RANDOM_QUALITY']
        ```

        :param img: a Tensor of img file, shape = `[x, y, 3]`
        :param process_list: a list of string to indicate process type.
        """
        for item in process_list:
            if item == 'REGULATION':
                img = cls.regulation(img)

            elif item == 'RANDOM_BRIGHTNESS':
                img = cls.random_brightness(img)
            
            elif item == 'RANDOM_FLIP':
                img = cls.random_flip(img)

            elif item == 'RANDOM_CONTRAST':
                img = cls.random_contrast(img)
            
            elif item == 'RANDOM_HUE':
                img = cls.random_hue(img)

            elif item == 'RANDOM_SATURATION':
                img = cls.random_saturation(img)

            elif item == 'RANDOM_QUALITY':
                img = cls.random_quality(img)

        return img

    @staticmethod
    def regulation(img: tf.Tensor) -> tf.Tensor:
        return img/255.0

    @staticmethod
    def random_brightness(img: tf.Tensor) -> tf.Tensor:
        return tf.image.random_brightness(img, max_delta=0.3)

    @staticmethod
    def random_flip(img: tf.Tensor) -> tf.Tensor:
        lr = tf.image.random_flip_left_right(img)
        return tf.image.random_flip_up_down(lr)

    @staticmethod
    def random_contrast(img: tf.Tensor) -> tf.Tensor:
        return tf.image.random_contrast(img, lower=0.5, upper=1.5)

    @staticmethod
    def random_hue(img: tf.Tensor) -> tf.Tensor:
        return tf.image.random_hue(img, max_delta=0.2)

    @staticmethod
    def random_saturation(img: tf.Tensor) -> tf.Tensor:
        return tf.image.random_saturation(img, lower=0.5, upper=1.0)

    @staticmethod
    def random_quality(img: tf.Tensor) -> tf.Tensor:
        return tf.image.random_jpeg_quality(img, 5, 100)
