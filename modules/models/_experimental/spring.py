'''
Author: Conghao Wong
Date: 2020-12-15 14:47:41
LastEditors: Conghao Wong
LastEditTime: 2021-04-08 21:46:51
Description: file content
'''

import cv2
import numpy as np
import scipy
import scipy.signal

from .._prediction import TrainArgs


class SpringManager():
    def __init__(self, args:TrainArgs):
        self.args = args

    def get_hook_from_guidance_map(self, guidance_map:np.ndarray, paras:np.ndarray):
        # kernel_size = 10
        # kernel = np.ones([kernel_size, kernel_size])/(kernel_size **2)
        # guidance_map_smooth = scipy.signal.convolve2d(guidance_map, kernel, mode='same')
        det, edg = compute_harris_response(guidance_map, sigma=5)
        det = (det - np.min(det))/(np.max(det) - np.min(det))

        cv2.imwrite('./test.jpg', 255*(det >= 0.3))
        cv2.imwrite('./test.jpg', 255*edg/np.max(edg))
        det = det >= 0.3
        
        dots = np.array(np.where(det)).T
        labels = k_nearest_neighbor(dots, K=5)

        import matplotlib.pyplot as plt
        plt.figure()
        for index in range(np.max(labels).astype(int) + 1):
            dots_index = np.where(labels == index)[0]
            d = dots[dots_index]
            plt.plot(d.T[1], d.T[0], 'o')
        
        plt.savefig('./test.jpg')

        hooks = []
        for index in range(np.max(labels).astype(int) + 1):
            dots_index = np.where(labels == index)[0]
            d = dots[dots_index]
            center = np.mean(d, axis=0)
            hooks.append(center)

        hooks = np.array(hooks)/paras[0] + paras[1]
        return hooks.tolist()



def compute_harris_response(im, sigma=3):
    # 在一幅灰度图像中，对每个像素计算Harris角点检测器响应函数
    
    # 计算导数
    imx = np.zeros(im.shape)
    scipy.ndimage.filters.gaussian_filter(im, (sigma,sigma), (0,1), imx)
    imy = np.zeros(im.shape)
    scipy.ndimage.filters.gaussian_filter(im, (sigma,sigma), (1,0), imy)
    
    # 计算harris矩阵分量
    Wxx = scipy.ndimage.filters.gaussian_filter(imx*imx, sigma)
    Wxy = scipy.ndimage.filters.gaussian_filter(imx*imy, sigma)
    Wyy = scipy.ndimage.filters.gaussian_filter(imy*imy, sigma)
    
    # 计算矩阵的特征值和迹
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy

    # 返回像素值为 Harris 响应函数值的一幅图像
    return Wdet, Wtr

def k_nearest_neighbor(data, K=3):
    distance_matrix = np.array([np.linalg.norm(data - data_current, axis=1) for data_current in data])
    nearest = np.argsort(distance_matrix)[:, :K]
    
    label = -1 * np.ones(len(data))
    for index, d in enumerate(data):
        if not label[index] == -1:
            continue
            
        new_label = np.max(label) + 1
        buffer = [index]
        while len(buffer):
            neighbor_index = buffer.pop()
            label[neighbor_index] = new_label
            
            for nn in nearest[neighbor_index]:
                if label[nn] == -1:
                    buffer.append(nn)

    return label

if __name__ == "__main__":
    args = TrainArgs().args()
    sm = SpringManager(args)
    map_manager = np.load('./dataset_npz/{}/gm.npy'.format(args.test_set), allow_pickle=True)[0]
    hooks = sm.get_hook_from_guidance_map(map_manager.guidance_map, paras=map_manager.real2grid_paras())
    print('!')
