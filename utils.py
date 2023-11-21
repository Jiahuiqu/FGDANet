import math
import numpy as np
from sklearn.decomposition import PCA
from scipy.sparse import coo_matrix
import networkx as nx
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.nn as nn
import data_SLIC
import os
import dataset

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pca_process(data, NC):
    [height, width, bands] = data.shape
    temp = np.reshape(data, (height * width, bands))
    pca = PCA(n_components=NC, copy=True, whiten=False)
    temp = pca.fit_transform(temp)
    temp = np.reshape(temp, (height, width, NC))
    return temp


def gt_to_one_hot(gt, class_count):
    """
    Convert Gt to one-hot labels
    : param gt:
    : param class_count:
    : return:
    """
    GT_One_Hot = []
    height = gt.shape[0]
    width = gt.shape[1]
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            temp = np.zeros(class_count, dtype=np.float32)
            if gt[i, j] != 0:
                temp[int(gt[i, j]) - 1] = 1
            GT_One_Hot.append(temp)
    GT_One_Hot = np.reshape(GT_One_Hot, [height, width, class_count])
    return GT_One_Hot


def get_superpixels(data, n_segments_init):
    # init seed
    n_segments_init = n_segments_init
    ls = data_SLIC.data_SLIC(data, 6)
    Q, S, A, Seg = ls.simple_superpixel(n_segments=n_segments_init)
    return Q, S, A, Seg


def merge_superpixel(seg_A, seg_B):
    height, width = seg_B.shape
    num_seg_B = seg_B.max()
    seg_B = seg_B + num_seg_B
    seg_B = seg_B * seg_B
    seg_m = seg_A + seg_B
    seg_m = seg_m.reshape(-1)
    num_seg_m_list = np.unique(seg_m)
    for i in range(len(num_seg_m_list)):
        seg_m[seg_m == num_seg_m_list[i]] = i
    seg_m = seg_m.reshape((height, width))

    superpixel_count = seg_m.max() + 1
    seg_m = np.reshape(seg_m, [-1])
    Q_merge = np.zeros([height * width, superpixel_count], dtype=np.float32)
    for i in range(superpixel_count):
        idx = np.where(seg_m == i)[0]
        Q_merge[idx, i] = 1
    seg_m = seg_m.reshape((height, width))
    return Q_merge, seg_m


def label_operation(x_label, class_count):
    samples_gt_onehot = gt_to_one_hot(x_label, class_count)
    samples_gt_onehot = np.reshape(samples_gt_onehot, [-1, class_count]).astype(int)
    return samples_gt_onehot


def mask_operation(x_label, class_count):
    [height, width] = x_label.shape
    label_mask = np.zeros([height * width, class_count])
    temp_ones = np.ones([class_count])
    samples_gt = np.reshape(x_label, [height * width])
    for i in range(height * width):
        if samples_gt[i] != 0:
            label_mask[i] = temp_ones
    label_mask = np.reshape(label_mask, [height * width, class_count])
    return label_mask


def ce_loss(feature, label):
    feature, label = extract_feature_label(feature, label)
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(feature, label)
    return loss


def extract_feature_label(feature, label):
    h, w = label.shape
    label = np.reshape(label, (h * w))
    index = np.where(label != 0)[0]
    label = label[index] - 1
    label = torch.from_numpy(label.astype(np.float32)).to(device)
    label = label.long()
    feature = feature[index]
    return feature, label


def ce_loss_sp(feature, label):
    feature, label = extract_feature_label_sp(feature, label)
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(feature, label)
    return loss


def extract_feature_label_sp(feature, label):
    index = np.where(label != 0)[0]
    label = label[index] - 1
    label = torch.from_numpy(label.astype(np.float32)).to(device)
    label = label.long()
    feature = feature[index]
    return feature, label


def coral_loss(source, target):
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)

    # source covariance
    tmp_s = torch.ones((1, ns)).to(device) @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)

    # target covariance
    tmp_t = torch.ones((1, nt)).to(device) @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

    # frobenius norm
    loss = (cs - ct).pow(2).sum().sqrt()
    loss = loss / (4 * d * d)

    return loss


def SAM(X, Y):
    feature1 = X
    feature2 = Y
    feature1 = F.normalize(feature1)
    feature2 = F.normalize(feature2)
    distance = feature1.mm(feature2.t())
    distance = torch.clamp(distance, -1, 1)
    SAM_value = torch.acos(distance)
    SAM_value = SAM_value.cpu().detach().numpy()
    SAM_value[np.isnan(SAM_value)] = 0
    SAM_value = torch.from_numpy(SAM_value.astype(np.float32)).to(device)
    return SAM_value


def euclidean_distances(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.mm(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def euclidean_distances_pixel(x, y):
    n = x.shape[0]
    dist = np.zeros(n)
    aaa = x[423]
    print('eqw')
    for i in range(n):
        dist[i] = np.linalg.norm(x[i] - y[i])

    return dist


class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        with torch.no_grad():
            delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
            loss = delta.dot(delta.T)
        torch.cuda.empty_cache()
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
                del XX, YY, XY, YX
            torch.cuda.empty_cache()
            return loss


def mmd_loss(source, target):
    mmd = MMD_loss(kernel_type='linear')
    source = mmd_data_standard(source)
    target = mmd_data_standard(target)
    loss = mmd(source, target)
    loss.requires_grad_(True)
    return loss


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """计算Gram/核矩阵
    source: sample_size_1 * feature_size 的数据
    target: sample_size_2 * feature_size 的数据
    kernel_mul: 这个概念不太清楚，感觉也是为了计算每个核的bandwith
    kernel_num: 表示的是多核的数量
    fix_sigma: 表示是否使用固定的标准差
    return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的矩阵，表达形式:
        [	K_ss K_st
            K_ts K_tt ]
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)  # 合并在一起

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)  # 计算高斯核中的|x-y|

    # 计算多核中每个核的bandwidth
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    # 高斯核的公式，exp(-|x-y|/bandwith)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val)  # 将多个核合并在一起


def mmd_data_standard(data):
    d_min = data.min()
    if d_min < 0:
        data = data + torch.abs(d_min)
        d_min = data.min()
    d_max = data.max()
    dst = d_max - d_min
    norm_data = (data - d_min).true_divide(dst)
    return norm_data


def mmd_early(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul,
                              kernel_num=kernel_num,
                              fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]  # Source<->Source
    YY = kernels[batch_size:, batch_size:]  # Target<->Target
    XY = kernels[:batch_size, batch_size:]  # Source<->Target
    YX = kernels[batch_size:, :batch_size]  # Target<->Source
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def accuracy_compute(pred_y, classes, TestLabel, model):

    height, width = TestLabel.shape
    TestLabel = np.reshape(TestLabel, height*width)
    index = np.where(TestLabel != 0)[0]
    TestLabel = TestLabel[index] - 1
    TestLabel = torch.from_numpy(TestLabel).to(device)
    pred_y = pred_y[index]
    OA = torch.eq(pred_y, TestLabel).sum().type(torch.FloatTensor) / TestLabel.size(0)
    print('The OA is: ', OA)
    return OA


def sparse2tensor(matrix):
    sparse_mx = matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to(device)


def pixel2superpixel(data, Q):
    height, width, bands = data.shape
    data = np.reshape(data, [height*width, bands])
    data = torch.from_numpy(data.astype(np.float32)).to(device)
    norm_col_Q = Q / (np.sum(Q, axis=0))
    norm_col_Q = norm_col_Q.T
    Q_sparse = coo_matrix(Q)
    Q = sparse2tensor(Q_sparse)
    norm_col_Q_sparse = coo_matrix(norm_col_Q)
    norm_col_Q = sparse2tensor(norm_col_Q_sparse)
    data_sp = torch.mm(norm_col_Q, data)
    data_sp = data_sp.detach().cpu().numpy()
    return data_sp


def pixel2superpixel_label(label_data, Q):
    data = label_data
    height, width = data.shape
    data = np.reshape(data, [height*width, ])
    temp_index = np.where(data == 2)[0]
    data[temp_index] = 10000
    data = torch.from_numpy(data.astype(np.float32)).to(device)
    norm_col_Q = Q / (np.sum(Q, axis=0))
    norm_col_Q = norm_col_Q.T
    Q_sparse = coo_matrix(Q)
    Q = sparse2tensor(Q_sparse)
    norm_col_Q_sparse = coo_matrix(norm_col_Q)
    norm_col_Q = sparse2tensor(norm_col_Q_sparse)
    data = data.unsqueeze(dim=1)
    data_sp = torch.mm(norm_col_Q, data)
    data_sp = data_sp.detach().cpu().numpy()
    temp_unchanged_index = np.where((data_sp != 0) & (data_sp < 20))[0]
    data_sp[temp_unchanged_index] = 1
    temp_changed_index = np.where((data_sp != 0) & (data_sp > 20))[0]
    data_sp[temp_changed_index] = 2
    return data_sp


def gen_pseudo_label(dataset_name_src, dataset_name_tar,
                     data_T1_src, data_T2_src, data_T1_tar, data_T2_tar,
                     data_sp_T1_src, data_sp_T2_src, data_sp_T1_tar, data_sp_T2_tar, Q_src, Q_tar,
                     height_src, width_src, height_tar, width_tar, k):
    similar_sp_src, similar_src = similarity_sp_sum_compute(data_sp_T1_src, data_sp_T2_src, Q_src)
    similar_sp_tar, similar_tar = similarity_sp_sum_compute(data_sp_T1_tar, data_sp_T2_tar, Q_tar)
    similar_src = np.reshape(similar_src, [height_src, width_src])
    similar_tar = np.reshape(similar_tar, [height_tar, width_tar])

    A_T1_src, A_T2_src = dataset.get_A_k_2num(data_sp_T1_src, data_sp_T2_src, k)
    A_T1_tar, A_T2_tar = dataset.get_A_k_2num(data_sp_T1_tar, data_sp_T2_tar, k)

    # 像素方案伪标签
    pseudo_label_src = gen_pseudo_label_from_graph_pixel_similarity(A_T1_src, A_T2_src, Q_src, similar_sp_src, similar_src)
    pseudo_label_tar = gen_pseudo_label_from_graph_pixel_similarity(A_T1_tar, A_T2_tar, Q_tar, similar_sp_tar, similar_tar)
    pseudo_label_src = np.reshape(pseudo_label_src, [height_src, width_src])
    pseudo_label_tar = np.reshape(pseudo_label_tar, [height_tar, width_tar])
    # 转到超像素上
    pseudo_label_src_sp = pixel2superpixel_label(pseudo_label_src.copy(), Q_src)
    pseudo_label_tar_sp = pixel2superpixel_label(pseudo_label_tar.copy(), Q_tar)

    return pseudo_label_src, pseudo_label_src_sp, pseudo_label_tar, pseudo_label_tar_sp


def gen_pseudo_label_from_graph(A_T1, A_T2, Q, similar_sp):
    A_T1 = A_T1.cpu().numpy()
    A_T2 = A_T2.cpu().numpy()
    n = A_T1.shape[0]
    identity_matrix = np.identity(n)
    A_T1 = A_T1 - identity_matrix
    A_T2 = A_T2 - identity_matrix

    pseudo_label = np.zeros(n)

    for i in range(n):
        index_T1 = np.where(A_T1[i] != 0)[0]
        index_T2 = np.where(A_T2[i] != 0)[0]
        index = np.concatenate([index_T1, index_T2], axis=0)
        num_index_T1 = len(index_T1)
        num_index_T2 = len(index_T2)
        num_set = len(set(index))
        if num_index_T1 == num_index_T2:
            if index_T1.all() == index_T2.all():
                pseudo_label[i] = 1

    index_unchanged = np.where(pseudo_label == 1)[0]

    index_small2big = np.argsort(similar_sp[:, 0])
    temp_count = np.ceil(similar_sp.shape[0] * 0.2).astype('int32')
    index_sp_small = index_small2big[0: temp_count]
    temp_list_1 = list(index_unchanged)
    temp_list_2 = list(index_sp_small)
    index_unchanged = np.array(list(set(temp_list_1) & set(temp_list_2)))

    pseudo_label = np.zeros(n)
    pseudo_label[index_unchanged] = 1
    index_background = np.where(pseudo_label != 1)[0]

    index_big2small = np.argsort(-similar_sp[:, 0])
    temp_count = np.ceil(similar_sp.shape[0] * 0.3).astype('int32')
    index_sp_big = index_big2small[0: temp_count]
    temp_list_1 = list(index_background)
    temp_list_2 = list(index_sp_big)
    index_changed = np.array(list(set(temp_list_1) & set(temp_list_2)))
    pseudo_label[index_changed] = 2

    pseudo_label_sp = pseudo_label
    pseudo_label = torch.from_numpy(pseudo_label.astype(np.float32)).to(device)
    pseudo_label = pseudo_label.unsqueeze(dim=1)
    Q_sparse = coo_matrix(Q)
    Q = sparse2tensor(Q_sparse)
    pseudo_label = torch.mm(Q, pseudo_label)
    pseudo_label = pseudo_label.detach().cpu().numpy()

    return pseudo_label, pseudo_label_sp


def gen_pseudo_label_from_graph_pixel_similarity(A_T1, A_T2, Q, similar_sp, similar_pixel):
    A_T1 = A_T1.cpu().numpy()
    A_T2 = A_T2.cpu().numpy()
    height, width = similar_pixel.shape
    similar_pixel = np.reshape(similar_pixel, [height * width])
    n = A_T1.shape[0]
    identity_matrix = np.identity(n)
    A_T1 = A_T1 - identity_matrix
    A_T2 = A_T2 - identity_matrix

    pseudo_label = np.zeros(n)

    for i in range(n):
        index_T1 = np.where(A_T1[i] != 0)[0]
        index_T2 = np.where(A_T2[i] != 0)[0]
        index = np.concatenate([index_T1, index_T2], axis=0)
        num_index_T1 = len(index_T1)
        num_index_T2 = len(index_T2)
        num_set = len(set(index))

        index_T1 = list(index_T1)
        index_T2 = list(index_T2)
        T1_T2 = set(index_T1).difference(set(index_T2))
        T2_T1 = set(index_T2).difference(set(index_T1))
        condition = len(T1_T2) + len(T2_T1)
        if condition == 0:
            pseudo_label[i] = 1

    index_unchanged = np.where(pseudo_label == 1)[0]
    index_small2big = np.argsort(similar_sp[:, 0])
    temp_count = np.ceil(similar_sp.shape[0] * 0.2).astype('int32')
    index_sp_small = index_small2big[0: temp_count]

    temp_list_1 = list(index_unchanged)
    temp_list_2 = list(index_sp_small)
    index_unchanged = np.array(list(set(temp_list_1) | set(temp_list_2)))

    pseudo_label = np.zeros(n)
    pseudo_label[index_unchanged] = 1

    pseudo_label = torch.from_numpy(pseudo_label.astype(np.float32)).to(device)
    pseudo_label = pseudo_label.unsqueeze(dim=1)
    Q_sparse = coo_matrix(Q)
    Q = sparse2tensor(Q_sparse)
    pseudo_label = torch.mm(Q, pseudo_label)
    pseudo_label = pseudo_label.detach().cpu().numpy()

    index_big2small = np.argsort(similar_pixel)
    temp_count = np.ceil(similar_pixel.shape[0] * 0.2).astype('int32')
    index_sp_big = index_big2small[0: temp_count]
    index_unchanged_from_sp = np.where(pseudo_label == 1)[0]
    temp_list_1 = list(index_unchanged_from_sp)
    temp_list_2 = list(index_sp_big)
    index_changed = np.array(list(set((set(temp_list_1) | set(temp_list_2)))))
    pseudo_label[index_changed] = 1

    index_background = np.where(pseudo_label != 1)[0]

    index_big2small = np.argsort(-similar_pixel)
    temp_count = np.ceil(similar_pixel.shape[0] * 0.2).astype('int32')
    index_sp_big = index_big2small[0: temp_count]
    temp_list_1 = list(index_background)
    temp_list_2 = list(index_sp_big)
    index_changed = np.array(list(set(temp_list_1) & set(temp_list_2)))
    pseudo_label[index_changed] = 2

    return pseudo_label


def class_average_spectral(data, label, num_class):
    [height, width, bands] = data.shape
    average_spectral = np.zeros((num_class, bands))
    data = np.reshape(data, (height*width, bands))
    label = np.reshape(label, -1)
    for i in range(num_class):
        index = np.where(label == i + 1)[0]
        data_temp = data[index]
        sum_temp = np.sum(data_temp, axis=0)
        average_spectral[i] = sum_temp / index.shape[0]
    return average_spectral


def conditional_mmd(output_src, output_tar, label_src, label_tar):
    mmd_value = torch.tensor(0).to(device)

    height_src, width_src = label_src.shape
    height_tar, width_tar = label_tar.shape
    label_src = np.reshape(label_src, (height_src * width_src))
    label_tar = np.reshape(label_tar, (height_tar * width_tar))
    label_src = torch.from_numpy(label_src).to(device)
    label_tar = torch.from_numpy(label_tar).to(device)

    class_src = torch.unique(label_src)
    class_src = class_src.cpu().numpy()
    class_src = set(class_src)
    class_tar = torch.unique(label_tar)
    class_tar = class_tar.cpu().numpy()
    class_tar = set(class_tar)
    common_class = class_src & class_tar
    class_index = np.array(list(common_class))
    class_index = torch.from_numpy(class_index).to(device)

    for i in range(len(class_index)):
        class_type = class_index[i]
        index_src = torch.where(label_src == class_type)
        data_class_src = output_src[index_src]
        index_tar = torch.where(label_tar == class_type)
        data_class_tar = output_tar[index_tar]
        mmd_class_loss = mmd_loss(data_class_src, data_class_tar)
        mmd_value = mmd_value + mmd_class_loss
        del index_src, index_tar
    return mmd_value


def conditional_coral(output_src, output_tar, label_src, label_tar):
    coral_value = torch.tensor(0).to(device)

    class_src = torch.unique(label_src)
    class_src = class_src.cpu().numpy()
    class_src = set(class_src)
    class_tar = torch.unique(label_tar)
    class_tar = class_tar.cpu().numpy()
    class_tar = set(class_tar)
    common_class = class_src & class_tar
    class_index = np.array(list(common_class))
    class_index = torch.from_numpy(class_index).to(device)

    for i in range(len(class_index)):
        class_type = class_index[i]
        index_src = torch.where(label_src == class_type)
        data_class_src = output_src[index_src]
        index_tar = torch.where(label_tar == class_type)
        data_class_tar = output_tar[index_tar]
        coral_class_loss = coral_loss(data_class_src, data_class_tar)
        coral_value = coral_value + coral_class_loss
        del index_src, index_tar
    return coral_value


def find_node_neighbor(G, node):
    nodes = list(nx.nodes(G))
    nei1_li = []
    nei2_li = []
    nei3_li = []
    for FNs in list(nx.neighbors(G, node)):  # find 1_th neighbors
        nei1_li.append(FNs)

    for n1 in nei1_li:
        for SNs in list(nx.neighbors(G, n1)):  # find 2_th neighbors
            nei2_li.append(SNs)
    nei2_li = list(set(nei2_li) - set(nei1_li))
    if node in nei2_li:
        nei2_li.remove(node)

    return nei1_li, nei2_li


def variable_substitution_HSI(data_original, data_CNN, train_neighbor):
    height, width = train_neighbor.shape
    original_bands = data_original.shape[2]
    needed_bands = data_CNN.shape[1]
    train_neighbor = np.reshape(train_neighbor, (height*width))
    index = np.where(train_neighbor)[0]
    data = torch.from_numpy(data_original.astype(np.float32)).to(device)
    data = torch.reshape(data, [height*width, original_bands])
    temp_i = needed_bands // original_bands
    for i in range(temp_i-1):
        data = torch.cat([data, data], dim=1)
    temp_difference = needed_bands - data.shape[1]
    temp_data = data[:, 0:temp_difference]
    data = torch.cat([data, temp_data], dim=1)
    data[index] = data_CNN
    return data


def variable_substitution_LiDAR(data_original, data_CNN, train_neighbor):
    height, width = train_neighbor.shape
    needed_bands = data_CNN.shape[1]
    train_neighbor = np.reshape(train_neighbor, (height*width))
    index = np.where(train_neighbor)[0]
    data = torch.from_numpy(data_original.astype(np.float32)).to(device)
    data = torch.reshape(data, [height*width, ])
    data = torch.reshape(data, [height * width, ])
    data = data.unsqueeze(dim=1)
    data = data.repeat(1, needed_bands)
    data[index] = data_CNN
    return data


def transform_A_to_coo(A):
    A = A.detach().cpu().numpy()

    edge_index_temp = sp.coo_matrix(A)

    values = edge_index_temp.data
    indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
    edge_index_A = torch.LongTensor(indices)

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    edge_index = torch.sparse_coo_tensor(i, v, edge_index_temp.shape)

    return edge_index_A


def circle_graph_build(num_src, num_tar, k):
    begin_num_tar = num_src
    n = num_src + num_tar

    symbol_src = 0
    symbol_tar = 1
    index_src = [0]
    index_tar = np.array(range(begin_num_tar, begin_num_tar + k))
    index_used_src = 0
    index_used_tar = begin_num_tar + k
    A_final_change = np.zeros((2, k))
    A_final_change[0, :] = 0
    A_final_change[1, :] = np.array(index_tar)

    flag_src = False
    flag_tar = False

    while 1:
        if symbol_tar == 1:
            initial_index_src = index_used_src
            for i in range(len(index_tar)):

                if index_used_src >= begin_num_tar:
                    flag_src = True
                    break

                if index_used_src + 1 + (k - 1) > begin_num_tar:
                    temp_end_point = begin_num_tar
                else:
                    temp_end_point = index_used_src + 1 + (k - 1)

                temp_length = temp_end_point - (index_used_src + 1)
                temp_A_change = np.zeros((2, temp_length))
                temp_A_change[0, :] = index_tar[i]
                temp_A_change[1, :] = np.array(range(index_used_src + 1, temp_end_point))
                A_final_change = np.concatenate((A_final_change, temp_A_change), axis=1)

                index_used_src = index_used_src + (k - 1)

            index_src = np.array(range(initial_index_src + 1, index_used_src + 1))

            if flag_tar == True:
                break

            symbol_src = 1
            symbol_tar = 0

        if symbol_src == 1:
            initial_index_tar = index_used_tar
            for i in range(len(index_src)):

                if index_used_tar >= n:
                    flag_tar = True
                    break

                if index_used_tar + 1 + (k - 1) > (n - 1):
                    temp_end_point = n
                else:
                    temp_end_point = index_used_tar + 1 + (k - 1)

                temp_length = temp_end_point - (index_used_tar + 1)
                temp_A_change = np.zeros((2, temp_length))
                temp_A_change[0, :] = index_src[i]
                temp_A_change[1, :] = np.array(range(index_used_tar + 1, temp_end_point))
                A_final_change = np.concatenate((A_final_change, temp_A_change), axis=1)

                index_used_tar = index_used_tar + (k - 1)

            index_tar = np.array(range(initial_index_tar + 1, index_used_tar + 1))

            if flag_src == True:
                break

            symbol_src = 0
            symbol_tar = 1

        if index_used_src >= begin_num_tar:
            if index_used_tar >= n:
                break

    A_final_change = A_final_change[:, np.argsort(A_final_change[0, :])]

    return A_final_change


def build_domain_differ_graph(pseudo_label_src,  pseudo_label_tar, k):

    num_changed_src = np.sum(pseudo_label_src == 2)
    num_unchanged_src = np.sum(pseudo_label_src == 1)
    num_changed_tar = np.sum(pseudo_label_tar == 2)
    num_unchanged_tar = np.sum(pseudo_label_tar == 1)


    A_changed = circle_graph_build(num_changed_src, num_changed_tar, k)
    A_unchanged = circle_graph_build(num_unchanged_src, num_unchanged_tar, k)

    return A_changed, A_unchanged


def get_pseudo_index(pseudo_label_sp):

    index_unchanged = np.where(pseudo_label_sp == 1)[0]
    index_changed = np.where(pseudo_label_sp == 2)[0]

    return index_unchanged, index_changed

def SAM_vector(H_i, H_j):
    SAM_value = math.sqrt(torch.dot(H_i, H_i)) * math.sqrt(torch.dot(H_j, H_j))
    SAM_value = torch.tensor(SAM_value)
    SAM_value = torch.dot(H_i, H_j) / SAM_value
    if SAM_value > 1 or SAM_value < -1:
        SAM_value = 1
    SAM_value = math.acos(SAM_value)
    SAM_value = torch.tensor(SAM_value)
    return SAM_value


def similarity_sp_sum_compute(data_T1_sp, data_T2_sp, Q):
    data_T1_sp = torch.from_numpy(data_T1_sp).to(device)
    data_T2_sp = torch.from_numpy(data_T2_sp).to(device)
    # similarity_sp = SAM(data_T1_sp, data_T2_sp)
    similarity_sp = euclidean_distances(data_T1_sp, data_T2_sp)
    similarity_sp = torch.diag(similarity_sp)

    dmax = similarity_sp.max()
    dmin = similarity_sp.min()
    dst = dmax - dmin
    similarity_sp = (similarity_sp - dmin) / dst

    similarity_sp = similarity_sp.unsqueeze(dim=1)
    Q_sparse = coo_matrix(Q)
    Q = sparse2tensor(Q_sparse)
    similarity = torch.mm(Q, similarity_sp)

    similarity = similarity.detach().cpu().numpy()
    similarity_sp = similarity_sp.cpu().detach().numpy()

    return similarity_sp, similarity


def similarity_sum_compute(data_T1, data_T2):
    height, width, bands = data_T1.shape
    data_T1 = np.reshape(data_T1, [height * width, bands])
    data_T2 = np.reshape(data_T2, [height * width, bands])
    similarity = euclidean_distances_pixel(data_T1, data_T2)

    dmax = similarity.max()
    dmin = similarity.min()
    dst = dmax - dmin
    similarity = (similarity - dmin) / dst

    return similarity


def pseudo_label_clean(pseudo_label, label):
    mask = label.copy()
    mask[mask > 0] = 1
    pseudo_label = pseudo_label * mask
    return pseudo_label
