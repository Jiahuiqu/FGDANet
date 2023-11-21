import numpy as np
import scipy.io as sio
import time
import torch
import os
# import NetWork
import NetWork
import dataset
import utils
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

print('\n')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Seed_List = [0]

# ###################### hyperparameter ######################
ratio = 0.2
k_1 = 300
k_2 = 30
k_3 = 30
# superpixel
n_segments_init_src = 5000
n_segments_init_tar = 2000
tip = 2
# network
max_epoch = 500
LR = 0.0002
FM = 32     # dim
BestLoss = 1000

# ###################### dataloader ######################
samples_type = ['ratio', 'same_num'][1]
# dataset
dataset_src = 1
dataset_tar = 2

def test(output, TestLabel, classes, model):
    output = torch.max(output, 1)[1].squeeze()
    OA = utils.accuracy_compute(output, classes, TestLabel, model)
    return OA


if __name__ == '__main__':
    # src data
    [data_T1_src, data_T2_src, All_label_src, val_ratio_src, class_count_src, learning_rate_src,
     max_epoch_src, dataset_name_src, trainloss_result_src] = dataset.get_dataset(dataset_src)

    # tar data
    [data_T1_tar, data_T2_tar, All_label_tar, val_ratio_tar, class_count_tar, learning_rate_tar,
     max_epoch_tar, dataset_name_tar, trainloss_result_tar] = dataset.get_dataset(dataset_tar)

    height_src, width_src, bands_src = data_T1_src.shape
    height_tar, width_tar, bands_tar = data_T1_tar.shape

    # standardization
    [data_T1_src, data_T2_src] = dataset.data_standard(data_T1_src, data_T2_src)
    [data_T1_tar, data_T2_tar] = dataset.data_standard(data_T1_tar, data_T2_tar)

    # ###################### Superpixel segmentation ######################
    data_cat_src = np.concatenate((data_T1_src, data_T2_src), axis=2)
    data_cat_tar = np.concatenate((data_T1_tar, data_T2_tar), axis=2)

    [Q_src, _, S_src, seg_src] = utils.get_superpixels(data_cat_src, n_segments_init_src)
    [Q_tar, _, S_tar, seg_tar] = utils.get_superpixels(data_cat_tar, n_segments_init_tar)

    # pixel2superpixel
    data_sp_T1_src = utils.pixel2superpixel(data_T1_src, Q_src)
    data_sp_T2_src = utils.pixel2superpixel(data_T2_src, Q_src)
    data_sp_T1_tar = utils.pixel2superpixel(data_T1_tar, Q_tar)
    data_sp_T2_tar = utils.pixel2superpixel(data_T2_tar, Q_tar)

    data_sp_cat_src = utils.pixel2superpixel(data_cat_src, Q_src)
    data_sp_cat_tar = utils.pixel2superpixel(data_cat_tar, Q_tar)

    # obtain the pseudo-label by graph
    [pseudo_label_src, pseudo_label_src_sp,
     pseudo_label_tar, pseudo_label_tar_sp] = utils.gen_pseudo_label(dataset_name_src, dataset_name_tar,
                                                                     data_T1_src, data_T2_src, data_T1_tar, data_T2_tar,
                                                                     data_sp_T1_src, data_sp_T2_src,
                                                                     data_sp_T1_tar, data_sp_T2_tar,
                                                                     Q_src, Q_tar,
                                                                     height_src, width_src, height_tar, width_tar, k_1)

    pseudo_label_src = dataset.num_label_balance(pseudo_label_src)
    pseudo_label_tar = dataset.num_label_balance(pseudo_label_tar)
    pseudo_label_src = dataset.data_partition(2, pseudo_label_src, ratio, height_src, width_src)
    pseudo_label_tar = dataset.data_partition(2, pseudo_label_tar, ratio, height_tar, width_tar)

    pseudo_label_src_sp = utils.pixel2superpixel_label(pseudo_label_src.copy(), Q_src)
    pseudo_label_tar_sp = utils.pixel2superpixel_label(pseudo_label_tar.copy(), Q_tar)
    pseudo_label_src_sp = np.squeeze(pseudo_label_src_sp)
    pseudo_label_tar_sp = np.squeeze(pseudo_label_tar_sp)

    sio.savemat(os.path.join('pseudo_label/', 'pseudo_label_' + dataset_name_src + '.mat'),
                {'pseudo_label_' + dataset_name_src: pseudo_label_src})
    sio.savemat(os.path.join('pseudo_label/', 'pseudo_label_' + dataset_name_tar + '.mat'),
                {'pseudo_label_' + dataset_name_tar: pseudo_label_tar})

    sio.savemat(os.path.join('debug/', 'pseudo_label_sp_' + dataset_name_src + '.mat'),
                {'pseudo_label_sp_' + dataset_name_src: pseudo_label_src_sp})
    sio.savemat(os.path.join('debug/', 'pseudo_label_sp_' + dataset_name_tar + '.mat'),
                {'pseudo_label_sp_' + dataset_name_tar: pseudo_label_tar_sp})
    print('pseudo label done')

    index_unchanged_src, index_changed_src = utils.get_pseudo_index(pseudo_label_src_sp)
    index_unchanged_tar, index_changed_tar = utils.get_pseudo_index(pseudo_label_tar_sp)

    A_changed, A_unchanged = utils.build_domain_differ_graph(pseudo_label_src_sp, pseudo_label_tar_sp, k_2)

    A_src = dataset.get_A_k(data_sp_cat_src, k_3)
    A_tar = dataset.get_A_k(data_sp_cat_tar, k_3)
    A_src = utils.transform_A_to_coo(A_src)
    A_tar = utils.transform_A_to_coo(A_tar)

    data_sp_T1_src = torch.from_numpy(data_sp_T1_src).to(device)
    data_sp_T2_src = torch.from_numpy(data_sp_T2_src).to(device)
    data_sp_T1_tar = torch.from_numpy(data_sp_T1_tar).to(device)
    data_sp_T2_tar = torch.from_numpy(data_sp_T2_tar).to(device)

    A_src = A_src.to(device)
    A_tar = A_tar.to(device)
    Q_src = torch.from_numpy(Q_src).to(device)
    Q_tar = torch.from_numpy(Q_tar).to(device)

    A_changed = torch.from_numpy(A_changed).to(device)
    A_unchanged = torch.from_numpy(A_unchanged).to(device)

    # Building the Network
    net_need_name = NetWork.net_need_name(bands_src, bands_tar, FM, A_src, A_tar, A_changed, A_unchanged, Q_src, Q_tar,
                                          index_unchanged_src, index_changed_src, index_unchanged_tar, index_changed_tar)
    print('net para：', net_need_name)
    net_need_name.to(device)

    # ###################### train ######################
    print('train begin')
    optimizer = torch.optim.Adam(net_need_name.parameters(), lr=LR)
    best_loss = 99999
    acc_dataset = 0
    net_need_name.train()
    loss_mmd = torch.tensor(0)
    w = [1, 1, 0.1, 0.1, 0.1]

    torch.cuda.synchronize()
    start = time.time()

    for epoch in range(max_epoch + 1):

        [final_feature_unchanged_labeled_src, final_feature_changed_labeled_src,
         final_feature_unchanged_labeled_tar, final_feature_changed_labeled_tar,
         final_feature_src, final_feature_tar,
         output_src_sp, output_tar_sp, output_src, output_tar] = net_need_name(data_sp_T1_src, data_sp_T2_src,
                                                                               data_sp_T1_tar, data_sp_T2_tar)

        loss_ce_src = utils.ce_loss(output_src, pseudo_label_src)
        loss_ce_tar = utils.ce_loss(output_tar, pseudo_label_tar)
        loss_cmmd_unchanged = utils.mmd_loss(final_feature_unchanged_labeled_src, final_feature_unchanged_labeled_tar)
        loss_cmmd_changed = utils.mmd_loss(final_feature_changed_labeled_src, final_feature_changed_labeled_tar)
        loss_mmd_feature = utils.mmd_loss(final_feature_src, final_feature_tar)

        loss = w[0]*loss_ce_src + w[1]*loss_ce_tar + w[2]*loss_cmmd_unchanged + w[3]*loss_cmmd_changed + w[4]*loss_mmd_feature

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch: ', epoch, '| train loss: %.4f' % loss,
              '| loss_ce_src: %.4f' % loss_ce_src,
              '| loss_ce_tar: %.4f' % loss_ce_tar,
              '| loss_cmmd_unchanged: %.4f' % loss_cmmd_unchanged,
              '| loss_cmmd_changed: %.4f' % loss_cmmd_changed,
              '| loss_mmd_feature: %.4f' % loss_mmd_feature)

        if epoch % 10 == 0:
            torch.cuda.empty_cache()
            with torch.no_grad():
                print(dataset_name_src, 'result：')
                OA_src = test(output_src, All_label_src, class_count_src, model='test_final')
                print(dataset_name_tar, 'result：')
                OA_tar = test(output_tar, All_label_tar, class_count_tar, model='test_final')

                if loss <= BestLoss:
                    torch.save(net_need_name.state_dict(), 'net_params.pkl')

                net_need_name.train()

    torch.cuda.synchronize()
    end = time.time()

    epoch_time = end - start
    print('train time：%.4f' % epoch_time)

    # ###################### test ######################
    print('test result')
    net_need_name.load_state_dict(torch.load('net_params.pkl'))
    net_need_name.eval()

    torch.cuda.synchronize()
    start = time.time()

    _, _, _, _, _, _, _, _, output_src, output_tar = net_need_name(data_sp_T1_src, data_sp_T2_src, data_sp_T1_tar, data_sp_T2_tar)

    print(dataset_name_src, 'result：')
    test(output_src, All_label_src, class_count_src, model='test_final')
    print(dataset_name_tar, 'result：')
    test(output_tar, All_label_tar, class_count_tar, model='test_final')

    print('save result：')
    output_src = torch.max(output_src, 1)[1].squeeze()
    output_tar = torch.max(output_tar, 1)[1].squeeze()
    output_src = output_src.detach().cpu().numpy()
    output_tar = output_tar.detach().cpu().numpy()
    output_src = np.reshape(output_src, (height_src, width_src))
    output_tar = np.reshape(output_tar, (height_tar, width_tar))

    torch.cuda.synchronize()
    end = time.time()
    ref_time = end-start
    print('test time：%.4f' % ref_time)

    sio.savemat(os.path.join('result/', 'result_' + dataset_name_src + '_' + str(tip) + '.mat'), {'result': output_src})
    sio.savemat(os.path.join('result/', 'result_' + dataset_name_tar + '_' + str(tip) + '.mat'), {'result': output_tar})
