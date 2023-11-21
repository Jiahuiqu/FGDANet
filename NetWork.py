import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch
import torch.nn as nn
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class GCN_raw_feature_extraction_net_src(torch.nn.Module):
    def __init__(self, input_bands, output_bands):
        super(GCN_raw_feature_extraction_net_src, self).__init__()
        self.GCN_1 = GCNConv(
                            in_channels=input_bands,
                            out_channels=input_bands*2)
        self.bn1 = nn.BatchNorm1d(input_bands*2)

        self.GCN_2 = SAGEConv(input_bands*2, output_bands)
        self.bn2 = nn.BatchNorm1d(output_bands)

        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.ablation_layer_1 = GCNConv(in_channels=input_bands, out_channels=output_bands),
        self.ablation_layer_1_after = nn.Sequential(
            nn.BatchNorm1d(output_bands),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.ablation_layer_2_1 = GCNConv(in_channels=input_bands, out_channels=input_bands * 2)
        self.ablation_layer_2_after_1 = nn.Sequential(
            nn.BatchNorm1d(input_bands * 2),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.ablation_layer_2_2 = SAGEConv(in_channels=input_bands * 2, out_channels=output_bands)
        self.ablation_layer_2_after_2 = nn.Sequential(
            nn.BatchNorm1d(output_bands),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.ablation_layer_3_1 = GCNConv(in_channels=input_bands, out_channels=input_bands*2)
        self.ablation_layer_3_after_1 = nn.Sequential(
            nn.BatchNorm1d(input_bands * 2),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.ablation_layer_3_2 = GCNConv(in_channels=input_bands*2, out_channels=input_bands * 2)
        self.ablation_layer_3_after_2 = nn.Sequential(
            nn.BatchNorm1d(output_bands),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.ablation_layer_3_3 = SAGEConv(in_channels=input_bands, out_channels=output_bands)
        self.ablation_layer_3_after_3 = nn.Sequential(
            nn.BatchNorm1d(output_bands),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.ablation_layer_4_1 = GCNConv(in_channels=input_bands, out_channels=input_bands * 2)
        self.ablation_layer_4_after_1 = nn.Sequential(
            nn.BatchNorm1d(input_bands * 2),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.ablation_layer_4_2 = GCNConv(in_channels=input_bands * 2, out_channels=input_bands * 2)
        self.ablation_layer_4_after_2 = nn.Sequential(
            nn.BatchNorm1d(output_bands),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.ablation_layer_4_3 = GCNConv(in_channels=input_bands * 2, out_channels=input_bands * 2)
        self.ablation_layer_4_after_3 = nn.Sequential(
            nn.BatchNorm1d(output_bands),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.ablation_layer_4_4 = SAGEConv(in_channels=input_bands, out_channels=output_bands)
        self.ablation_layer_4_after_4 = nn.Sequential(
            nn.BatchNorm1d(output_bands),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.ablation_layer_5_1 = GCNConv(in_channels=input_bands, out_channels=input_bands * 2)
        self.ablation_layer_5_after_1 = nn.Sequential(
            nn.BatchNorm1d(input_bands * 2),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.ablation_layer_5_2 = GCNConv(in_channels=input_bands * 2, out_channels=input_bands * 2)
        self.ablation_layer_5_after_2 = nn.Sequential(
            nn.BatchNorm1d(output_bands),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.ablation_layer_5_3 = GCNConv(in_channels=input_bands * 2, out_channels=input_bands * 2)
        self.ablation_layer_5_after_3 = nn.Sequential(
            nn.BatchNorm1d(output_bands),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.ablation_layer_5_4 = GCNConv(in_channels=input_bands * 2, out_channels=input_bands * 2)
        self.ablation_layer_5_after_4 = nn.Sequential(
            nn.BatchNorm1d(output_bands),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.ablation_layer_5_5 = SAGEConv(in_channels=input_bands, out_channels=output_bands)
        self.ablation_layer_5_after_5 = nn.Sequential(
            nn.BatchNorm1d(output_bands),
            nn.ReLU(),
            nn.Dropout(0.5)
        )



    def forward(self, data):

        data, A = data.x, data.edge_index
        data = self.ablation_layer_2_1(data, A)
        data = self.ablation_layer_2_after_1(data)
        data = self.ablation_layer_2_2(data, A)
        data = self.ablation_layer_2_after_2(data)

        return data


class GCN_raw_feature_extraction_net_tar(torch.nn.Module):
    def __init__(self, input_bands, output_bands):
        super(GCN_raw_feature_extraction_net_tar, self).__init__()
        self.GCN_1 = GCNConv(
                            in_channels=input_bands,
                            out_channels=input_bands*2)
        self.bn1 = nn.BatchNorm1d(input_bands*2)

        self.GCN_2 = SAGEConv(input_bands*2, output_bands)
        self.bn2 = nn.BatchNorm1d(output_bands)

        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.ablation_layer_1 = GCNConv(in_channels=input_bands, out_channels=output_bands),
        self.ablation_layer_1_after = nn.Sequential(
            nn.BatchNorm1d(output_bands),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.ablation_layer_2_1 = GCNConv(in_channels=input_bands, out_channels=input_bands * 2)
        self.ablation_layer_2_after_1 = nn.Sequential(
            nn.BatchNorm1d(input_bands * 2),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.ablation_layer_2_2 = GCNConv(in_channels=input_bands * 2, out_channels=output_bands)
        self.ablation_layer_2_after_2 = nn.Sequential(
            nn.BatchNorm1d(output_bands),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.ablation_layer_3_1 = GCNConv(in_channels=input_bands, out_channels=input_bands*2)
        self.ablation_layer_3_after_1 = nn.Sequential(
            nn.BatchNorm1d(input_bands * 2),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.ablation_layer_3_2 = GCNConv(in_channels=input_bands*2, out_channels=input_bands * 2)
        self.ablation_layer_3_after_2 = nn.Sequential(
            nn.BatchNorm1d(output_bands),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.ablation_layer_3_3 = GCNConv(in_channels=input_bands, out_channels=output_bands)
        self.ablation_layer_3_after_3 = nn.Sequential(
            nn.BatchNorm1d(output_bands),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.ablation_layer_4_1 = GCNConv(in_channels=input_bands, out_channels=input_bands * 2)
        self.ablation_layer_4_after_1 = nn.Sequential(
            nn.BatchNorm1d(input_bands * 2),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.ablation_layer_4_2 = GCNConv(in_channels=input_bands * 2, out_channels=input_bands * 2)
        self.ablation_layer_4_after_2 = nn.Sequential(
            nn.BatchNorm1d(output_bands),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.ablation_layer_4_3 = GCNConv(in_channels=input_bands * 2, out_channels=input_bands * 2)
        self.ablation_layer_4_after_3 = nn.Sequential(
            nn.BatchNorm1d(output_bands),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.ablation_layer_4_4 = GCNConv(in_channels=input_bands, out_channels=output_bands)
        self.ablation_layer_4_after_4 = nn.Sequential(
            nn.BatchNorm1d(output_bands),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.ablation_layer_5_1 = GCNConv(in_channels=input_bands, out_channels=input_bands * 2)
        self.ablation_layer_5_after_1 = nn.Sequential(
            nn.BatchNorm1d(input_bands * 2),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.ablation_layer_5_2 = GCNConv(in_channels=input_bands * 2, out_channels=input_bands * 2)
        self.ablation_layer_5_after_2 = nn.Sequential(
            nn.BatchNorm1d(output_bands),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.ablation_layer_5_3 = GCNConv(in_channels=input_bands * 2, out_channels=input_bands * 2)
        self.ablation_layer_5_after_3 = nn.Sequential(
            nn.BatchNorm1d(output_bands),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.ablation_layer_5_4 = GCNConv(in_channels=input_bands * 2, out_channels=input_bands * 2)
        self.ablation_layer_5_after_4 = nn.Sequential(
            nn.BatchNorm1d(output_bands),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.ablation_layer_5_5 = GCNConv(in_channels=input_bands, out_channels=output_bands)
        self.ablation_layer_5_after_5 = nn.Sequential(
            nn.BatchNorm1d(output_bands),
            nn.ReLU(),
            nn.Dropout(0.5)
        )



    def forward(self, data):

        data, A = data.x, data.edge_index

        data = self.ablation_layer_2_1(data, A)
        data = self.ablation_layer_2_after_1(data)
        data = self.ablation_layer_2_2(data, A)
        data = self.ablation_layer_2_after_2(data)

        return data


class linear_difference(nn.Module):
    def __init__(self, input_bands, output_bands):
        super(linear_difference, self).__init__()
        self.linear1 = nn.Linear(input_bands, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 128)

    def forward(self, feature):
        feature = F.relu(self.linear1(feature))
        return feature


class GCN_shared(torch.nn.Module):
    def __init__(self, input_bands, output_bands):
        super(GCN_shared, self).__init__()
        self.GCN_1 = GCNConv(
                            in_channels=input_bands,
                            out_channels=input_bands * 2)
        self.bn1 = nn.BatchNorm1d(input_bands * 2)

        self.GCN_2 = SAGEConv(input_bands * 2, output_bands)
        self.bn2 = nn.BatchNorm1d(output_bands)

        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, data, model):
        data, A = data.x, data.edge_index

        data = self.GCN_1(data, A)
        data = self.bn1(data)
        if model == 'feature extraction':
            data = self.ReLU(data)
            data = self.dropout(data)

        data = self.GCN_2(data, A)
        data = self.bn2(data)
        if model == 'feature extraction':
            data = self.ReLU(data)
            data = self.dropout(data)
        return data


class GCN_src(torch.nn.Module):
    def __init__(self, input_bands, output_bands):
        super(GCN_src, self).__init__()
        self.GCN_1 = GCNConv(
                            in_channels=input_bands,
                            out_channels=input_bands * 2)
        self.bn1 = nn.BatchNorm1d(input_bands * 2)

        self.GCN_2 = SAGEConv(input_bands * 2, output_bands)
        self.bn2 = nn.BatchNorm1d(output_bands)

        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):

        data, A = data.x, data.edge_index

        data = self.GCN_1(data, A)
        data = self.bn1(data)
        data = self.ReLU(data)
        data = self.dropout(data)

        data = self.GCN_2(data, A)
        data = self.bn2(data)
        data = self.ReLU(data)
        data = self.dropout(data)

        return data


class GCN_tar(torch.nn.Module):
    def __init__(self, input_bands, output_bands):
        super(GCN_tar, self).__init__()
        self.GCN_1 = GCNConv(
                            in_channels=input_bands,
                            out_channels=input_bands * 2)
        self.bn1 = nn.BatchNorm1d(input_bands * 2)

        self.GCN_2 = SAGEConv(input_bands * 2, output_bands)
        self.bn2 = nn.BatchNorm1d(output_bands)

        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):

        data, A = data.x, data.edge_index

        data = self.GCN_1(data, A)
        data = self.bn1(data)
        data = self.ReLU(data)
        data = self.dropout(data)

        data = self.GCN_2(data, A)
        data = self.bn2(data)
        data = self.ReLU(data)
        data = self.dropout(data)
        return data


class classifier_net_src(nn.Module):
    def __init__(self, input_bands, class_num):
        super(classifier_net_src, self).__init__()
        # input_bands = FM*16 = 32*16
        self.fully_connect_1 = nn.Linear(input_bands, 32*8)
        self.fully_connect_2 = nn.Linear(32*8, 32*4)
        self.fully_connect_3 = nn.Linear(32*4, 32*2)
        self.fully_connect_4 = nn.Linear(32*2, 16)
        self.fully_connect_5 = nn.Linear(16, class_num)

    def forward(self, data):
        output = F.relu(self.fully_connect_1(data))
        output = F.relu(self.fully_connect_2(output))
        output = F.relu(self.fully_connect_3(output))
        output = F.relu(self.fully_connect_4(output))
        output = self.fully_connect_5(output)
        return output


class classifier_net_tar(nn.Module):
    def __init__(self, input_bands, class_num):
        super(classifier_net_tar, self).__init__()
        # input_bands = FM*16 = 32*16
        self.fully_connect_1 = nn.Linear(input_bands, 32 * 8)
        self.fully_connect_2 = nn.Linear(32 * 8, 32 * 4)
        self.fully_connect_3 = nn.Linear(32 * 4, 32 * 2)
        self.fully_connect_4 = nn.Linear(32 * 2, 16)
        self.fully_connect_5 = nn.Linear(16, class_num)

    def forward(self, data):
        output = F.relu(self.fully_connect_1(data))
        output = F.relu(self.fully_connect_2(output))
        output = F.relu(self.fully_connect_3(output))
        output = F.relu(self.fully_connect_4(output))
        output = self.fully_connect_5(output)
        return output


class net_need_name(torch.nn.Module):
    def __init__(self, bands_src, bands_tar, FM, A_src, A_tar, A_changed, A_unchanged, Q_src, Q_tar,
                 index_unchanged_src, index_changed_src, index_unchanged_tar, index_changed_tar):
        super(net_need_name, self).__init__()

        self.A_src = A_src
        self.A_tar = A_tar
        self.A_changed = A_changed.long()
        self.A_unchanged = A_unchanged.long()
        self.Q_src = Q_src
        self.Q_tar = Q_tar
        self.index_unchanged_src = index_unchanged_src
        self.index_changed_src = index_changed_src
        self.index_unchanged_tar = index_unchanged_tar
        self.index_changed_tar = index_changed_tar

        self.dimension_conversion_T1_src = nn.Linear(bands_src, FM)
        self.dimension_conversion_T2_src = nn.Linear(bands_src, FM)
        self.dimension_conversion_T1_tar = nn.Linear(bands_tar, FM)
        self.dimension_conversion_T2_tar = nn.Linear(bands_tar, FM)

        self.Feature_extraction_T1_src = GCN_raw_feature_extraction_net_src(FM, FM * 4)
        self.Feature_extraction_T2_src = GCN_raw_feature_extraction_net_src(FM, FM * 4)
        self.Feature_extraction_T1_tar = GCN_raw_feature_extraction_net_tar(FM, FM * 4)
        self.Feature_extraction_T2_tar = GCN_raw_feature_extraction_net_tar(FM, FM * 4)

        self.linear_difference = linear_difference(FM * 8, FM * 4)

        self.GCN_shared = GCN_shared(FM * 8, FM * 16)

        self.GCN_src = GCN_src(FM * 4, FM * 16)
        self.GCN_tar = GCN_tar(FM * 4, FM * 16)

        self.classifier_src = classifier_net_src(FM * 16, 2)
        self.classifier_tar = classifier_net_tar(FM * 16, 2)

    def forward(self, data_T1_src, data_T2_src, data_T1_tar, data_T2_tar):

        data_T1_src = self.dimension_conversion_T1_src(data_T1_src)
        data_T2_src = self.dimension_conversion_T2_src(data_T2_src)
        data_T1_tar = self.dimension_conversion_T1_tar(data_T1_tar)
        data_T2_tar = self.dimension_conversion_T2_tar(data_T2_tar)

        data_T1_src = Data(x=data_T1_src, edge_index=self.A_src)
        data_T2_src = Data(x=data_T2_src, edge_index=self.A_src)
        data_T1_tar = Data(x=data_T1_tar, edge_index=self.A_tar)
        data_T2_tar = Data(x=data_T2_tar, edge_index=self.A_tar)

        data_T1_src = self.Feature_extraction_T1_src(data_T1_src)
        data_T2_src = self.Feature_extraction_T2_src(data_T2_src)
        data_T1_tar = self.Feature_extraction_T1_tar(data_T1_tar)
        data_T2_tar = self.Feature_extraction_T2_tar(data_T2_tar)


        difference_image_src = torch.concat([data_T1_src, data_T2_src], dim=1)
        difference_image_tar = torch.concat([data_T1_tar, data_T2_tar], dim=1)
        difference_image_unchanged_src = difference_image_src[self.index_unchanged_src]
        difference_image_changed_src = difference_image_src[self.index_changed_src]
        difference_image_unchanged_tar = difference_image_tar[self.index_unchanged_tar]
        difference_image_changed_tar = difference_image_tar[self.index_changed_tar]

        n_unchanged_src = difference_image_unchanged_src.shape[0]
        n_changed_src = difference_image_changed_src.shape[0]
        n_unchanged_tar = difference_image_unchanged_tar.shape[0]
        n_changed_tar = difference_image_changed_tar.shape[0]

        difference_image_unchanged = torch.cat([difference_image_unchanged_src, difference_image_unchanged_tar], dim=0)
        difference_image_changed = torch.cat([difference_image_changed_src, difference_image_changed_tar], dim=0)

        difference_image_unchanged = Data(x=difference_image_unchanged, edge_index=self.A_unchanged)
        difference_image_changed = Data(x=difference_image_changed, edge_index=self.A_changed)

        difference_image_unchanged = self.GCN_shared(difference_image_unchanged, 'represent unchanged')
        difference_image_changed = self.GCN_shared(difference_image_changed, 'represent changed')

        final_feature_unchanged_labeled_src = difference_image_unchanged[0:n_unchanged_src, :]
        final_feature_changed_labeled_src = difference_image_changed[0:n_changed_src, :]
        final_feature_unchanged_labeled_tar = difference_image_unchanged[n_unchanged_src:, :]
        final_feature_changed_labeled_tar = difference_image_changed[n_changed_src:, :]

        difference_image_src = Data(x=difference_image_src, edge_index=self.A_src)
        difference_image_tar = Data(x=difference_image_tar, edge_index=self.A_tar)
        #
        difference_image_src = self.GCN_shared(difference_image_src, 'feature extraction')
        difference_image_tar = self.GCN_shared(difference_image_tar, 'feature extraction')
        final_feature_src = difference_image_src
        final_feature_tar = difference_image_tar

        output_src_sp = self.classifier_src(difference_image_src)
        output_tar_sp = self.classifier_tar(difference_image_tar)

        output_src = torch.matmul(self.Q_src, output_src_sp)
        output_tar = torch.matmul(self.Q_tar, output_tar_sp)

        return [final_feature_unchanged_labeled_src, final_feature_changed_labeled_src,
                final_feature_unchanged_labeled_tar, final_feature_changed_labeled_tar,
                final_feature_src, final_feature_tar, output_src_sp, output_tar_sp, output_src, output_tar]



