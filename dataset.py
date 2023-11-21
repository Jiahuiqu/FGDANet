import random
import numpy as np
import networkx as nx
import time
import torch
import torch.utils
import scipy.io as sio
import h5py
import os
import utils
from sklearn import preprocessing
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = '/media/xidian/55bc9b72-e29e-4dfa-b83e-0fbd0d5a7677/xd132/YT/data/change detection/regular dataset/'


def print_data(gt, class_count):
    gt_reshape = np.reshape(gt, [-1])
    for i in range(class_count):
        idx = np.where(gt_reshape == i + 1)[-1]
        samplesCount = len(idx)
        print('第' + str(i + 1) + '类的个数为' + str(samplesCount))


def get_dataset(dataset):

    data_T1 = []
    data_T2 = []
    gt = []
    val_ratio = 0
    class_count = 0
    learning_rate = 0
    max_epoch = 0
    trainloss_result = []
    dataset_name = ''

    if dataset == 1:
        dataset_name = 'China'
        data_T1 = sio.loadmat(path + 'China/T1.mat')['T1']
        data_T2 = sio.loadmat(path + 'China/T2.mat')['T2']
        gt = sio.loadmat(path + 'China/label.mat')['label']
        class_count = 2

        val_ratio = 0.01
        learning_rate = 2e-4
        max_epoch = 600
        trainloss_result = np.zeros([max_epoch + 1, 1])
        pass

    if dataset == 2:
        dataset_name = 'USA'
        data_T1 = sio.loadmat(path + 'USA/T1.mat')['T1']
        data_T2 = sio.loadmat(path + 'USA/T2.mat')['T2']
        gt = sio.loadmat(path + 'USA/label.mat')['label']
        class_count = 2

        val_ratio = 0.01
        learning_rate = 2e-4
        max_epoch = 600
        trainloss_result = np.zeros([max_epoch + 1, 1])
        pass

    return [data_T1, data_T2, gt, val_ratio, class_count,
            learning_rate, max_epoch, dataset_name, trainloss_result]


def data_standard(data_T1, dataT2):

    height, width, bands = data_T1.shape

    data_T1 = np.reshape(data_T1, [height * width, bands])
    minMax = preprocessing.StandardScaler()
    data_T1 = minMax.fit_transform(data_T1)
    data_T1 = np.reshape(data_T1, [height, width, bands])

    dataT2 = np.reshape(dataT2, [height * width, bands])
    minMax = preprocessing.StandardScaler()
    dataT2 = minMax.fit_transform(dataT2)
    dataT2 = np.reshape(dataT2, [height, width, bands])
    return [data_T1, dataT2]


def data_partition(class_count, gt, train_ratio, height, width):

    train_rand_idx = []

    gt_reshape = np.reshape(gt, [-1])

    for i in range(class_count):
        idx = np.where(gt_reshape == i + 1)[-1]
        samplesCount = len(idx)
        rand_list = [i for i in range(samplesCount)]
        rand_idx = random.sample(rand_list,
                                 np.ceil(samplesCount * train_ratio).astype('int32'))
        rand_real_idx_per_class = idx[rand_idx]
        train_rand_idx.append(rand_real_idx_per_class)
    train_rand_idx = np.array(train_rand_idx)
    train_data_index = []
    for c in range(train_rand_idx.shape[0]):
        a = train_rand_idx[c]
        for j in range(a.shape[0]):
            train_data_index.append(a[j])
    train_data_index = np.array(train_data_index)

    train_data_index = set(train_data_index)
    train_data_index = list(train_data_index)

    train_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(train_data_index)):
        train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
        pass
    train_label = np.reshape(train_samples_gt, [height, width])

    return train_label


def target_data_partition(gt, height, width):
    gt_reshape = np.reshape(gt, [-1])

    background_idx = np.where(gt_reshape == 0)[-1]
    background_idx = set(background_idx)
    all_data_index = [i for i in range(len(gt_reshape))]
    all_data_index = set(all_data_index)
    test_data_index = all_data_index - background_idx
    test_data_index = list(test_data_index)


    test_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(test_data_index)):
        test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
        pass
    test_label = np.reshape(test_samples_gt, [height, width])  # 测试样本图
    test_label_tar = test_label
    return test_label


def get_A_k(data_T1, k):
    k = k + 1

    data_T1 = torch.from_numpy(data_T1.astype(np.float32)).to(device)

    n = data_T1.shape[0]


    A_k = torch.zeros(n, n).to(device)

    T1_rel = utils.SAM(data_T1, data_T1)

    index = torch.argsort(T1_rel)
    for i in range(n):
        for j in range(k):
            A_k[i, index[i, j]] = 1

    return A_k


def get_A_k_2num(data_T1, data_T2, k):
    k = k + 1

    data_T1 = torch.from_numpy(data_T1.astype(np.float32)).to(device)
    data_T2 = torch.from_numpy(data_T2.astype(np.float32)).to(device)

    n = data_T1.shape[0]

    A_k_T1 = torch.zeros(n, n).to(device)
    A_k_T2 = torch.zeros(n, n).to(device)


    T1_rel = utils.SAM(data_T1, data_T1)
    T2_rel = utils.SAM(data_T2, data_T2)

    index = torch.argsort(T1_rel)
    for i in range(n):
        for j in range(k):
            A_k_T1[i, index[i, j]] = 1

    index = torch.argsort(T2_rel)
    for i in range(n):
        for j in range(k):
            A_k_T2[i, index[i, j]] = 1

    return A_k_T1, A_k_T2


def pixel2superpixel(Q, data_HSI):
    height, width, bands = data_HSI.shape
    data_HSI = np.reshape(data_HSI, (height*width, bands))

    norm_col_Q = Q / (np.sum(Q, axis=0))
    norm_col_Q = norm_col_Q.T

    data_HSI = torch.from_numpy(data_HSI.astype(np.float32)).to(device)
    norm_col_Q = torch.from_numpy(norm_col_Q.astype(np.float32)).to(device)
    norm_col_Q = norm_col_Q.to_sparse()

    superpixels_flatten = torch.spmm(norm_col_Q, data_HSI)
    superpixels_flatten = superpixels_flatten.cpu().numpy()
    return superpixels_flatten


def gen_cnn_data(data_HSI, data_LiDAR, patchsize_HSI, patchsize_LiDAR, train_label, test_label, batchsize):
    height, width, bands = data_HSI.shape
    temp = data_HSI[:, :, 0]
    pad_width = np.floor(patchsize_HSI / 2)
    pad_width = np.int(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    [h_pad, w_pad] = temp2.shape
    data_HSI_pad = np.empty((h_pad, w_pad, bands), dtype='float32')

    for i in range(bands):
        temp = data_HSI[:, :, i]
        pad_width = np.floor(patchsize_HSI / 2)
        pad_width = np.int(pad_width)
        temp2 = np.pad(temp, pad_width, 'symmetric')
        data_HSI_pad[:, :, i] = temp2

    data_LiDAR_pad = data_LiDAR
    pad_width2 = np.floor(patchsize_LiDAR / 2)
    pad_width2 = np.int(pad_width2)
    temp = np.pad(data_LiDAR_pad, pad_width2, 'symmetric')
    data_LiDAR_pad = temp

    [ind1, ind2] = np.where(train_label != 0)
    TrainNum = len(ind1)
    TrainPatch_HSI = np.empty((TrainNum, bands, patchsize_HSI, patchsize_HSI), dtype='float32')
    TrainLabel_HSI = np.empty(TrainNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch = data_HSI_pad[(ind3[i] - pad_width):(ind3[i] + pad_width + 1),
                             (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
        patch = np.reshape(patch, (patchsize_HSI * patchsize_HSI, bands))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (bands, patchsize_HSI, patchsize_HSI))
        TrainPatch_HSI[i, :, :, :] = patch
        patchlabel_HSI = train_label[ind1[i], ind2[i]]
        TrainLabel_HSI[i] = patchlabel_HSI

    [ind1, ind2] = np.where(test_label != 0)
    TestNum = len(ind1)
    TestPatch_HSI = np.empty((TestNum, bands, patchsize_HSI, patchsize_HSI), dtype='float32')
    TestLabel_HSI = np.empty(TestNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch = data_HSI_pad[(ind3[i] - pad_width):(ind3[i] + pad_width + 1),
                             (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
        patch = np.reshape(patch, (patchsize_HSI * patchsize_HSI, bands))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (bands, patchsize_HSI, patchsize_HSI))
        TestPatch_HSI[i, :, :, :] = patch
        patchlabel_HSI = test_label[ind1[i], ind2[i]]
        TestLabel_HSI[i] = patchlabel_HSI

    print('Training size and testing size of HSI are:', TrainPatch_HSI.shape, 'and', TestPatch_HSI.shape)

    [ind1, ind2] = np.where(train_label != 0)
    TrainNum = len(ind1)
    TrainPatch_LiDAR = np.empty((TrainNum, 1, patchsize_LiDAR, patchsize_LiDAR), dtype='float32')
    TrainLabel_LiDAR = np.empty(TrainNum)
    ind3 = ind1 + pad_width2
    ind4 = ind2 + pad_width2
    for i in range(len(ind1)):
        patch = data_LiDAR_pad[(ind3[i] - pad_width2):(ind3[i] + pad_width2 + 1),
                               (ind4[i] - pad_width2):(ind4[i] + pad_width2 + 1)]
        patch = np.reshape(patch, (patchsize_LiDAR * patchsize_LiDAR, 1))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (1, patchsize_LiDAR, patchsize_LiDAR))
        TrainPatch_LiDAR[i, :, :, :] = patch
        patchlabel_LiDAR = train_label[ind1[i], ind2[i]]
        TrainLabel_LiDAR[i] = patchlabel_LiDAR

    [ind1, ind2] = np.where(test_label != 0)
    TestNum = len(ind1)
    TestPatch_LIDAR = np.empty((TestNum, 1, patchsize_LiDAR, patchsize_LiDAR), dtype='float32')
    TestLabel_LiDAR = np.empty(TestNum)
    ind3 = ind1 + pad_width2
    ind4 = ind2 + pad_width2
    for i in range(len(ind1)):
        patch = data_LiDAR_pad[(ind3[i] - pad_width2):(ind3[i] + pad_width2 + 1), (ind4[i] - pad_width2):(ind4[i] + pad_width2 + 1)]
        patch = np.reshape(patch, (patchsize_LiDAR * patchsize_LiDAR, 1))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (1, patchsize_LiDAR, patchsize_LiDAR))
        TestPatch_LIDAR[i, :, :, :] = patch
        patchlabel_LiDAR = test_label[ind1[i], ind2[i]]
        TestLabel_LiDAR[i] = patchlabel_LiDAR

    print('Training size and testing size of LiDAR are:', TrainPatch_LiDAR.shape, 'and', TestPatch_LIDAR.shape)

    TrainPatch_HSI = torch.from_numpy(TrainPatch_HSI).to(device)
    TrainLabel_HSI = torch.from_numpy(TrainLabel_HSI) - 1
    TrainLabel_HSI = TrainLabel_HSI.long().to(device)

    TestPatch_HSI = torch.from_numpy(TestPatch_HSI).to(device)
    TestLabel_HSI = torch.from_numpy(TestLabel_HSI) - 1
    TestLabel_HSI = TestLabel_HSI.long().to(device)

    TrainPatch_LiDAR = torch.from_numpy(TrainPatch_LiDAR).to(device)
    TrainLabel_LiDAR = torch.from_numpy(TrainLabel_LiDAR) - 1
    TrainLabel_LiDAR = TrainLabel_LiDAR.long().to(device)

    TestPatch_LIDAR = torch.from_numpy(TestPatch_LIDAR).to(device)
    TestLabel_LiDAR = torch.from_numpy(TestLabel_LiDAR) - 1
    TestLabel_LiDAR = TestLabel_LiDAR.long().to(device)

    return TrainPatch_HSI, TrainPatch_LiDAR, TrainLabel_HSI, TestPatch_HSI, TestPatch_LIDAR, TestLabel_HSI


def get_all_graph(data_HSI_src, data_LiDAR_src, data_HSI_tar, data_LiDAR_tar, k):
    A_HSI_src = gen_A_coo(data_HSI_src, k, type='HSI')
    A_LiDAR_src = gen_A_coo(data_LiDAR_src, k, type='LiDAR')
    A_HSI_tar = gen_A_coo(data_HSI_tar, k, type='HSI')
    A_LiDAR_tar = gen_A_coo(data_LiDAR_tar, k, type='LiDAR')
    return A_HSI_src, A_LiDAR_src, A_HSI_tar, A_LiDAR_tar


def gen_A_coo(data, k, type):
    if type == 'LiDAR':
        data = np.expand_dims(data, 2)
    h, w, b = data.shape
    data = np.reshape(data, (h * w, b))
    data_list = []
    max_k_list = []
    all_index_value = np.zeros((data.shape[0], 2, k))
    ave_num = 3000
    number = data.shape[0] // ave_num

    time_start = time.time()

    for i in range(number):
        temp = data[i * ave_num:(i + 1) * ave_num, :]
        data_list.append(temp)
        del temp
    if (i + 1) * ave_num < data.shape[0]:
        temp = data[(i + 1) * ave_num: data.shape[0], :]
        data_list.append(temp)
        del temp

    for i in range(len(data_list)):
        data_1 = data_list[i]
        for j in range(len(data_list)):
            data_2 = data_list[j]
            if type == 'HSI':
                relation = utils.SAM(data_1, data_2)
            if type == 'LiDAR':
                relation = utils.euclidean_distances(data_1, data_2)
            index_and_value_batch = np.zeros((data_1.shape[0], 2, k))
            for m in range(data_1.shape[0]):
                relation_m = relation[m, :]
                temp_index = np.argsort(relation_m)
                index = temp_index[1:k + 1]
                value = relation_m[index]
                index = index + ave_num * j

                temp_result = np.zeros((2, k))
                temp_result[0] = index
                temp_result[1] = value
                index_and_value_batch[m] = temp_result

            if j == 0:
                index_and_value = index_and_value_batch
            else:
                index_and_value = np.concatenate((index_and_value, index_and_value_batch), axis=2)

        for m in range(data_1.shape[0]):
            temp_1 = index_and_value[m, 1, :]
            temp_index = np.argsort(temp_1)
            temp_index = temp_index[0: k]
            index = index_and_value[m, 0, temp_index]
            value = index_and_value[m, 1, temp_index]
            all_index_value[ave_num * i + m, 0, :] = index
            all_index_value[ave_num * i + m, 1, :] = value

    A_coo = np.zeros((3, (data.shape[0] * k)))
    for i in range(data.shape[0]):
        A_coo[0, k * i: k * (i + 1)] = i
        A_coo[1, k * i: k * (i + 1)] = all_index_value[i, 0, :]
        A_coo[2, k * i: k * (i + 1)] = all_index_value[i, 1, :]

    time_end = time.time()
    print('gen graph A cost', time_end - time_start, 's')
    return A_coo


def get_neighbors(train_label, A_coo):
    """
    获取输入训练集的一阶及二阶邻居
    :param train_label: 训练集
    :param A_coo: 以稀疏格式存储的邻接矩阵
    :return: 包含了原训练集及其 一阶和二阶 邻居的新训练集
    """
    height, width = train_label.shape
    num_node = np.max(A_coo[0]).astype(int)
    edge = []
    for i in range(0, A_coo.shape[1]):
        temp = (A_coo[0, i], A_coo[1, i])
        edge.append(temp)

    G = nx.Graph()
    G.add_nodes_from(list(range(num_node)))
    G.add_edges_from(edge)

    train_label = np.reshape(train_label, [height*width])
    new_train_index = []

    train_label_index = np.where(train_label)[0]
    for i in range(len(train_label_index)):
        out = utils.find_node_neighbor(G, train_label_index[i])
        temp1 = out[0]
        temp2 = out[1]
        temp = temp1 + temp2
        new_train_index = new_train_index + temp
    new_train_index = list(set(new_train_index))

    train_label_index = list(train_label_index)
    new_train_index = new_train_index + train_label_index
    new_train_index = list(set(new_train_index))
    new_train_index = np.array(new_train_index, dtype=int)
    train_neighbor = np.zeros((height * width))
    train_neighbor[new_train_index] = 1
    train_neighbor = np.reshape(train_neighbor, [height, width])
    return train_neighbor


def get_merge_neighbor(train_label, A_HSI, A_LiDAR):
    height, width = train_label.shape
    train_neighbor_HSI = get_neighbors(train_label, A_HSI)
    train_neighbor_LiDAR = get_neighbors(train_label, A_LiDAR)

    train_neighbor_HSI = np.reshape(train_neighbor_HSI, (height*width))
    train_neighbor_LiDAR = np.reshape(train_neighbor_LiDAR, (height * width))

    train_neighbor_HSI_index = list(np.where(train_neighbor_HSI)[0])
    train_neighbor_LiDAR_index = list(np.where(train_neighbor_LiDAR)[0])

    train_neighbor_index = train_neighbor_HSI_index + train_neighbor_LiDAR_index
    train_neighbor_index = list(set(train_neighbor_index))
    train_neighbor = np.zeros((height * width))
    train_neighbor[train_neighbor_index] = 1
    train_neighbor = np.reshape(train_neighbor, [height, width])

    return train_neighbor


def load_exist_graph():
    A_HSI_src_mat = sio.loadmat('graph structure/A_HSI_src.mat')
    A_HSI_src = A_HSI_src_mat['A_HSI_src']
    A_LiDAR_src_mat = sio.loadmat('graph structure/A_LiDAR_src.mat')
    A_LiDAR_src = A_LiDAR_src_mat['A_LiDAR_src']

    A_HSI_tar_mat = sio.loadmat('graph structure/A_HSI_tar.mat')
    A_HSI_tar = A_HSI_tar_mat['A_HSI_tar']
    A_LiDAR_tar_mat = sio.loadmat('graph structure/A_LiDAR_tar.mat')
    A_LiDAR_tar = A_LiDAR_tar_mat['A_LiDAR_tar']

    return A_HSI_src, A_LiDAR_src, A_HSI_tar, A_LiDAR_tar


def load_exist_pseudo_label(dataset, ratio):
    global pseudo_label
    if dataset == 1:
        dataset_name = 'China'
        pseudo_label = sio.loadmat(path + 'China/pseudo label/pseudo_label_' + dataset_name + '_' + str(ratio) + '.mat')['pseudo_label_' + dataset_name]
        pass
    if dataset == 2:
        dataset_name = 'USA'
        pseudo_label = sio.loadmat(path + 'USA/pseudo label/pseudo_label_' + dataset_name + '_' + str(ratio) + '.mat')['pseudo_label_' + dataset_name]
        pass
    if dataset == 3:
        dataset_name = 'river'
        pseudo_label = sio.loadmat(path + 'river/pseudo label/pseudo_label_' + dataset_name + '_' + str(ratio) + '.mat')['pseudo_label_' + dataset_name]
        pass
    if dataset == 4:
        dataset_name = 'Bay'
        pseudo_label = sio.loadmat(path + 'Bay/pseudo label/pseudo_label_' + dataset_name + '_' + str(ratio) + '.mat')['pseudo_label_' + dataset_name]
        pass
    if dataset == 5:
        dataset_name = 'Barbara'
        pseudo_label = sio.loadmat(path + 'Barbara/pseudo label/pseudo_label_' + dataset_name + '_' + str(ratio) + '.mat')['pseudo_label_' + dataset_name]
        pass
    return pseudo_label


def load_exist_pseudo_label_sp(dataset):
    global pseudo_label_sp
    if dataset == 1:
        dataset_name = 'China'
        pseudo_label_sp = sio.loadmat('debug/pseudo_label_sp_' + dataset_name + '.mat')['pseudo_label_sp_' + dataset_name]
        pass
    if dataset == 2:
        dataset_name = 'USA'
        pseudo_label_sp = sio.loadmat('debug/pseudo_label_sp_' + dataset_name + '.mat')['pseudo_label_sp_' + dataset_name]
        pass
    if dataset == 3:
        dataset_name = 'Bay'
        pseudo_label_sp = sio.loadmat('debug/pseudo_label_sp_' + dataset_name + '.mat')['pseudo_label_sp_' + dataset_name]
        pass
    if dataset == 4:
        dataset_name = 'Barbara'
        pseudo_label_sp = sio.loadmat('debug/pseudo_label_sp_' + dataset_name + '.mat')['pseudo_label_sp_' + dataset_name]
        pass
    return pseudo_label_sp


def load_exist_superpixel(dataset):
    global Q
    if dataset == 1:
        dataset_name = 'China'
        Q = h5py.File('debug/Q_' + dataset_name + '.mat')['Q_' + dataset_name][:]
        pass
    if dataset == 2:
        dataset_name = 'USA'
        Q = h5py.File('debug/Q_' + dataset_name + '.mat')['Q_' + dataset_name][:]
        pass
    if dataset == 3:
        dataset_name = 'river'
        Q = h5py.File('debug/Q_' + dataset_name + '.mat')['Q_' + dataset_name][:]
        pass
    if dataset == 4:
        dataset_name = 'Bay'
        Q = h5py.File('debug/Q_' + dataset_name + '.mat')['Q_' + dataset_name][:]
        pass
    if dataset == 5:
        dataset_name = 'Barbara'
        Q = h5py.File('debug/Q_' + dataset_name + '.mat')['Q_' + dataset_name][:]
        pass
    return Q.T


def load_exist_pixel2superpixel(dataset):
    global data_sp_T1, data_sp_T2
    if dataset == 1:
        dataset_name = 'China'
        data_sp_T1 = h5py.File('debug/data_sp_T1_' + dataset_name + '.mat')['data_sp_T1_' + dataset_name][:]
        data_sp_T2 = h5py.File('debug/data_sp_T2_' + dataset_name + '.mat')['data_sp_T2_' + dataset_name][:]
        pass
    if dataset == 2:
        dataset_name = 'USA'
        data_sp_T1 = h5py.File('debug/data_sp_T1_' + dataset_name + '.mat')['data_sp_T1_' + dataset_name][:]
        data_sp_T2 = h5py.File('debug/data_sp_T2_' + dataset_name + '.mat')['data_sp_T2_' + dataset_name][:]
        pass
    if dataset == 3:
        dataset_name = 'river'
        data_sp_T1 = h5py.File('debug/data_sp_T1_' + dataset_name + '.mat')['data_sp_T1_' + dataset_name][:]
        data_sp_T2 = h5py.File('debug/data_sp_T2_' + dataset_name + '.mat')['data_sp_T2_' + dataset_name][:]
        pass
    if dataset == 4:
        dataset_name = 'Bay'
        data_sp_T1 = h5py.File('debug/data_sp_T1_' + dataset_name + '.mat')['data_sp_T1_' + dataset_name][:]
        data_sp_T2 = h5py.File('debug/data_sp_T2_' + dataset_name + '.mat')['data_sp_T2_' + dataset_name][:]
        pass
    if dataset == 5:
        dataset_name = 'Barbara'
        data_sp_T1 = h5py.File('debug/data_sp_T1_' + dataset_name + '.mat')['data_sp_T1_' + dataset_name][:]
        data_sp_T2 = h5py.File('debug/data_sp_T2_' + dataset_name + '.mat')['data_sp_T2_' + dataset_name][:]
        pass
    return data_sp_T1.T, data_sp_T2.T


def load_exist_similar(dataset):
    global similar
    if dataset == 'China':
        dataset_name = 'China'
        similar = sio.loadmat('pseudo_label/similar_' + dataset_name + '.mat')['similar_' + dataset_name]
        pass
    if dataset == 'USA':
        dataset_name = 'USA'
        similar = sio.loadmat('pseudo_label/similar_' + dataset_name + '.mat')['similar_' + dataset_name]
        similar = similar[1:200, :]
        pass
    if dataset == 'river':
        dataset_name = 'river'
        similar = sio.loadmat('pseudo_label/similar_' + dataset_name + '.mat')['similar_' + dataset_name]
        pass
    if dataset == 'Bay':
        dataset_name = 'Bay'
        similar = sio.loadmat('pseudo_label/similar_' + dataset_name + '.mat')['similar_' + dataset_name]
        pass
    if dataset == 'Barbara':
        dataset_name = 'Barbara'
        similar = sio.loadmat('pseudo_label/similar_' + dataset_name + '.mat')['similar_' + dataset_name]
        pass
    return similar


def num_label_balance(pseudo_label):
    pseudo_label = np.reshape(pseudo_label, [-1])

    num_unchanged = np.where(pseudo_label == 1)[0].shape[0]
    num_changed = np.where(pseudo_label == 2)[0].shape[0]
    if num_unchanged > num_changed:
        ratio = num_unchanged / num_changed
    else:
        ratio = num_changed / num_unchanged


    if ratio > 1:

        if num_unchanged > num_changed:
            index_unchanged = np.where(pseudo_label == 1)[0]
            samplesCount = len(index_unchanged)
            rand_list = [i for i in range(samplesCount)]
            rand_idx = random.sample(rand_list, np.ceil(num_changed).astype('int32'))
            index_unchanged_new = index_unchanged[rand_idx]
            index_unchanged_new = np.array(index_unchanged_new)
            pseudo_label[index_unchanged] = 0
            pseudo_label[index_unchanged_new] = 1
        else:
            index_changed = np.where(pseudo_label == 2)[0]
            samplesCount = len(index_changed)
            rand_list = [i for i in range(samplesCount)]
            rand_idx = random.sample(rand_list, np.ceil(num_unchanged).astype('int32'))
            index_changed_new = index_changed[rand_idx]
            index_changed_new = np.array(index_changed_new)
            pseudo_label[index_changed] = 0
            pseudo_label[index_changed_new] = 2

    num_unchanged = np.where(pseudo_label == 1)[0].shape[0]
    num_changed = np.where(pseudo_label == 2)[0].shape[0]
    print(num_unchanged, num_changed)

    return pseudo_label
