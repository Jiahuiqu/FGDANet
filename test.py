import heapq
import math
import os
import numpy as np
import scipy.io as sio
import hdf5storage
import matplotlib.pyplot as plt
import random
from itertools import zip_longest
from itertools import cycle
from matplotlib import cm
import spectral as spy
from sklearn import metrics
import time
import scipy.sparse as sp
import torch.utils.data as dataf
from sklearn import preprocessing
from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F

# A1 = np.array([[0, 1, 1, 0, 1],
#               [1, 0, 0, 1, 0],
#               [1, 0, 0, 1, 1],
#               [0, 1, 1, 0, 1],
#               [1, 0, 1, 1, 0]])
#
# A2 = np.array([[0, 1, 1, 0, 1],
#               [0, 1, 1, 0, 1],
#               [1, 1, 0, 1, 1],
#               [0, 1, 1, 0, 1],
#               [1, 0, 1, 1, 0]])
#
# n = A1.shape[0]
# pseudo_label = np.zeros(n)
#
# for i in range(n):
#
#     index_T1 = np.where(A1[i] != 0)[0]
#     index_T2 = np.where(A2[i] != 0)[0]
#     index = np.concatenate([index_T1, index_T2], axis=0)
#     num_index_T1 = len(index_T1)
#     num_index_T2 = len(index_T2)
#     num_set = len(set(index))
#     if num_index_T1 == num_index_T2:
#         if index_T1.all() == index_T2.all():
#             pseudo_label[i] = 1     # 是没变化的点
#
#     if num_set == (num_index_T1 + num_index_T2):
#         pseudo_label[i] = 2     # 是变化的点，因为源域和目标域的index完全不一样
#
# print('3423')

# import warnings
# warnings.filterwarnings('ignore')
#
# A = np.array([[0, 3, 0, 1],
#               [1, 0, 2, 0],
#               [0, 1, 0, 0],
#               [1, 0, 0, 0]])
#
# edge_index_temp = sp.coo_matrix(A)
# print(edge_index_temp)
#
# values = edge_index_temp.data  # 边上对应权重值weight
# indices = np.vstack((edge_index_temp.row, edge_index_temp.col))  # 我们真正需要的coo形式
# edge_index_A = torch.LongTensor(indices)  # 我们真正需要的coo形式
# print(edge_index_A)
#
# i = torch.LongTensor(indices)  # 转tensor
# v = torch.FloatTensor(values)  # 转tensor
# edge_index = torch.sparse_coo_tensor(i, v, edge_index_temp.shape)
# print(edge_index)
#
# X = np.array([[1, 0, 1, 1, 1, 1],
#               [0, 1, 1, 0, 0, 1],
#               [-1, 1, 0, 1, -1, 1],
#               [-1, 0, 0, 1, 1, 1]])
# X = torch.FloatTensor(X)
#
# Y = np.array([1, 0, 1, 1])
# Y = torch.LongTensor(Y)
#
#


# ##### 构造图结构的测试代码 ######
def circle_graph_build():
    k = 5
    sum_src = 100
    sum_tar = 300
    begin_num_tar = 205000      # tar开始的节点
    n = 500000     # 总共的节点数

    symbol_src = 0      # 若等于1，说明要对这个域的点接上邻居
    symbol_tar = 1      # 若等于1，说明要对这个域的点接上邻居
    index_src = [0]     # 即将做运算的几个点
    index_tar = np.array(range(begin_num_tar, begin_num_tar + k))       # 即将做运算的几个点
    index_used_src = 0      # 表示已经用到src的第几个点了(真实index)
    index_used_tar = begin_num_tar + k      # 表示已经用到tar的第几个点了（真实index）
    # 初始化最终的邻接矩阵，以coo形式表示
    A_final_change = np.zeros((2, k))
    A_final_change[0, :] = 0
    A_final_change[1, :] = np.array(index_tar)

    while 1:
        if symbol_tar == 1:
            initial_index_src = index_used_src
            for i in range(len(index_tar)):

                if index_used_src >= begin_num_tar:      # 如果源域已经放完了，就跳出当前循环
                    break

                if index_used_src + 1 + (k-1) > begin_num_tar:      # 即这一轮，源域的节点用到中途就完了
                    temp_end_point = begin_num_tar
                else:
                    temp_end_point = index_used_src + 1 + (k-1)

                temp_length = temp_end_point - (index_used_src + 1)
                temp_A_change = np.zeros((2, temp_length))
                temp_A_change[0, :] = index_tar[i]
                temp_A_change[1, :] = np.array(range(index_used_src + 1, temp_end_point))
                A_final_change = np.concatenate((A_final_change, temp_A_change), axis=1)

                index_used_src = index_used_src + (k-1)

            index_src = np.array(range(initial_index_src + 1, index_used_src + 1))

            symbol_src = 1
            symbol_tar = 0

        if symbol_src == 1:
            initial_index_tar = index_used_tar
            for i in range(len(index_src)):

                if index_used_tar >= n:      # 如果目标域已经放完了，就跳出当前循环
                    break

                if index_used_tar + 1 + (k-1) > (n-1):      # 即这一轮，目标域的节点用到最后一个了
                    temp_end_point = n
                else:
                    temp_end_point = index_used_tar + 1 + (k-1)

                temp_length = temp_end_point - (index_used_tar + 1)
                temp_A_change = np.zeros((2, temp_length))
                temp_A_change[0, :] = index_src[i]
                temp_A_change[1, :] = np.array(range(index_used_tar + 1, temp_end_point))
                A_final_change = np.concatenate((A_final_change, temp_A_change), axis=1)

                index_used_tar = index_used_tar + (k - 1)

            index_tar = np.array(range(initial_index_tar + 1, index_used_tar + 1))

            symbol_src = 0
            symbol_tar = 1

        if index_used_src >= begin_num_tar:
            if index_used_tar >= n:
                break

    # 将邻接矩阵按照第一行排序
    AAA = A_final_change
    A_final_change = A_final_change[:, np.argsort(A_final_change[0, :])]
    print('131313')


def sort():
    a = np.array([[1, 1, 5, 5, 2, 3, 1],
                  [12, 45, 78, 68, 45, 21, 20]])
    b = a[:, np.argsort(a[0, :])]
    print('131313')


def statistic():
    a = np.array([[1, 1, 5, 5, 2, 3, 1],
                  [12, 12, 78, 68, 45, 12, 20]])
    b = np.sum(a == 12)
    print('qewqw')

if __name__ == '__main__':
    # start = time.time()
    # circle_graph_build()
    # end = time.time()
    # print('运行时间：', end-start)
    # sort()
    # statistic()
    # a = np.array([1, 1, 5, 5, 2, 3, 1])
    # b = torch.from_numpy(a)
    # b = b.unsqueeze(dim=1)
    # a = np.array([[1, 1, 5, 5, 2, 3, 1],
    #              [2, 3, 4, 2, 3, 5, 3]])
    # b = np.array([[1, 1, 5, 5, 2, 3, 1],
    #              [2, 3, 4, 2, 3, 5, 3],
    #              [3, 4, 6, 3, 5, 3, 2]])
    # a = torch.from_numpy(a)
    # b = torch.from_numpy(b)
    #
    # c = torch.cat([a, b], dim=0)

    pseudo_label = np.array([1, 1, 2, 2, 2, 1, 1, 0])
    num_changed = np.where(pseudo_label == 1)[0]
    # similar_sp = np.array([0.95, 0.23, 0.12, 0.89, 0.11, 0.94, 0.9, 0.56])
    # aaa = np.ones([728160, 4200])
    # hdf5storage.savemat(os.path.join('debug/', 'Q_' + 'dataset_name_src' + '.mat'), {'Q_' + 'dataset_name_src': aaa},
    #                     format='7.3')

    # index_unchanged = np.where(pseudo_label == 1)[0]
    # similar_sp_unchanged = similar_sp[index_unchanged]
    # temp_unchanged = np.where(similar_sp_unchanged < 0.4)[0]
    # index_unchanged = index_unchanged[temp_unchanged]
    #
    # index_changed = np.where(pseudo_label == 2)[0]
    # similar_sp_changed = similar_sp[index_changed]
    # temp_changed = np.where(similar_sp_changed > 0.9)[0]
    # index_changed = index_changed[temp_changed]
    #
    # pseudo_label = np.zeros(8)
    # pseudo_label[index_unchanged] = 1
    # pseudo_label[index_changed] = 2

    print('131313')
