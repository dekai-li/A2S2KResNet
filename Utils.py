import numpy as np
from sklearn import metrics, preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import spectral
import torch
import cv2
from operator import truediv


def sampling(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = max(ground_truth)
    for i in range(m):
        indexes = [
            j for j, x in enumerate(ground_truth.ravel().tolist())
            if x == i + 1
        ]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)
        else:
            nb_val = 0
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes


def set_figsize(figsize=(3.5, 2.5)):
    display.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = figsize


def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi,
                        ground_truth.shape[0] * 2.0 / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)
    return 0


PALETTE = np.array([
   # [0,   0,   0],
    [255, 0,   0],
    [0,   255, 0],
    [0,   0,   255],
    [255, 255, 0],
    [255, 0,   255],
    [0,   255, 255],
    [200, 100, 0],
    [0,   200, 100],
    [100, 0,   200],
    [200, 0,   100],
    [100, 200, 0],
    [0,   100, 200],
    [150, 75,  75],
    [75,  150, 75],
    [75,  75,  150],
    [255, 100, 100],
    [0,0,0],
    [100, 255, 100],
    [100, 100, 255],
    [255, 150, 75],
    [75,  255, 150],
    [150, 75,  255],
    [50,  50,  50],
    [100, 100, 100],
    [150, 150, 150],
    [200, 200, 200],
    [250, 250, 250],
    [100, 0,   0],
    [200, 0,   0],
    [0,   100, 0],
    [0,   200, 0],
    [0,   0,   100],
    [0,   0,   200],
    [100, 100, 0],
    [200, 200, 0],
    [100, 0,   100],
    [200, 0,   200],
    [0,   100, 100],
    [0,   200, 200],
], dtype=np.float32) / 255.0


def list_to_colormap(x_list):
    """
    将 x_list 中的每个整数值映射到 PALETTE 中对应的 RGB 颜色。
    超出索引范围（包括 -1）统一返回黑色 [0,0,0]。
    """
    x = np.asarray(x_list, dtype=int)
    n = x.shape[0]
    y = np.zeros((n, 3), dtype=np.float32)

    # 只对 0 <= x < PALETTE.shape[0] 的元素进行映射
    valid = (x >= 0) & (x < PALETTE.shape[0])
    y[valid] = PALETTE[x[valid]]
    # 其余元素（如 -1 或超范围）保持为 [0,0,0]
    return y


def generate_png(all_iter, net, gt_hsi, Dataset, device, total_indices, path):
    pred_test = []
    for X, y in all_iter:
        #X = X.permute(0, 3, 1, 2)
        X = X.to(device)
        net.eval()
        pred_test.extend(net(X).cpu().argmax(axis=1).detach().numpy())
    gt = gt_hsi.flatten()
    x_label = np.zeros(gt.shape)
    for i in range(len(gt)):
        if gt[i] == 0:
            gt[i] = 17
            x_label[i] = 16
    gt = gt[:] - 1
    x_label[total_indices] = pred_test
    x = np.ravel(x_label)
    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)
    y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    gt_re = np.reshape(y_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    classification_map(y_re, gt_hsi, 300,
                       path + '.png')
    classification_map(gt_re, gt_hsi, 300,
                       path + '_gt.png')
    print('------Get classification maps successful-------')
