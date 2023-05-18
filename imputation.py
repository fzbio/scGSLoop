import os.path
import time

import cooler
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.nn import VGAE, global_mean_pool
from torch_geometric.loader import NeighborLoader, DataLoader
from gnns import VariationalGraphSageEncoder, DenseDecoder
from schickit.data_reading import read_cool_as_sparse
from schickit.data_storage import save_sparse_to_scool, save_sparse_to_cool
from schickit.file_format_conversion import convert_cool_to_scool
from utils import hpc_celltype_parser
import random
from tqdm.auto import tqdm
# from torch_geometric.transforms import RandomLinkSplit
from train_utils import gnn_evaluate_all, gnn_train_batch, gnn_approx_evaluate_all, save_model, load_model, EarlyStopper
from torch_geometric import transforms as T
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix
from collections import OrderedDict
import multiprocessing as mp
from nn_data import ScoolDataset
from nn_data import get_split_dataset, easy_to_device
import tempfile
from nn_data import RemoveSelfLooping
from schickit.utils import coarsen_scool
from configs import DATA_SPLIT_SEED, DEVICE

# torch.manual_seed(SEED)
# random.seed(SEED)
# np.random.seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)


class Pooling(torch.nn.Module):
    def __init__(self, pooling_layer):
        super().__init__()
        self.pooling = pooling_layer

    def forward(self, z, batch):
        z = self.pooling(z, batch)
        return z


# def average_adj(coo_group):
#     # coo matrices in coo_list must be of the same shape. This is not checked in the function and
#     # must be taken care by the user.
#     shape = coo_group[0].shape
#     data_dict = dict()
#     count_dict = dict()
#     for m in coo_group:
#         coords = zip(m.row, m.col)
#         data = m.data
#         for c, d in zip(coords, data):
#             if c in data_dict:
#                 data_dict[c] += d
#                 count_dict[c] += 1
#             else:
#                 data_dict[c] = d
#                 count_dict[c] = 1
#     for key in data_dict:
#         data_dict[key] = data_dict[key] / count_dict[key]
#     coord_data_pairs = data_dict.items()
#     if len(coord_data_pairs) > 0:
#         coords, data = zip(*coord_data_pairs)
#         row, col = zip(*coords)
#         data_df = pd.DataFrame({'row': row, 'col': col, 'data': data})
#         data_df.sort_values(by=['row', 'col'], ascending=True)
#         row, col, data = data_df['row'], data_df['col'], data_df['data']
#         return coo_matrix((data, (row, col)), shape=shape)
#     else:
#         print(1)
#         return coo_matrix(([], ([], [])), shape=shape)


def average_adj(coo_group):
    # coo matrices in coo_list must be of the same shape. This is not checked in the function and
    # must be taken care by the user.
    for m in coo_group:
        m.sum_duplicates()
    sum_adj = sum(coo_group).tocoo()
    mean_adj = sum_adj / len(coo_group)
    return mean_adj


def impute_and_save_cell(cell_name, closest_cell_names, chrom_names, zoom_in_scool_path, bin_selector, tmp_dir):
    cell_mlist_dict = {}
    cell_mlist_dict[cell_name] = read_cool_as_sparse(
        zoom_in_scool_path + '::' + cell_name, chrom_names, weighted=True, sparse_format='coo'
    )
    assert len(cell_mlist_dict[cell_name]) == len(chrom_names)
    for c in closest_cell_names:
        cell_mlist_dict[c] = read_cool_as_sparse(
            zoom_in_scool_path + '::' + c, chrom_names, weighted=True, sparse_format='coo'
        )
    assert len(cell_mlist_dict) == len(closest_cell_names) + 1
    averaged_chrom_coos = []
    for i, chrom in enumerate(chrom_names):
        raw_coo_group = [cell_mlist_dict[c][i] for c in [cell_name] + closest_cell_names]
        averaged_chrom_coos.append(average_adj(raw_coo_group))
    save_sparse_to_cool(
        os.path.join(tmp_dir, cell_name.split('/')[-1] + '.cool'), chrom_names, averaged_chrom_coos, bin_selector
    )


def convert_dist_mat_to_closest_cells(cell_names, dist_matrix, k):
    closest_cells_list = []
    for i in range(len(dist_matrix)):
        dist_vect = dist_matrix[:, i]
        top_k_indices = np.argsort(dist_vect)[:k]
        closest_cells_list.append([cell_names[top_i] for top_i in top_k_indices])
    return closest_cells_list


def get_cell_names_of_ds(ds):
    cell_names = []
    unique_names = []
    loader = DataLoader(ds, shuffle=False, batch_size=1)
    for d in loader:
        cell_names.append(d.cell_name[0])
    # print(cell_names)
    for n in cell_names:
        if n not in unique_names:
            unique_names.append(n)
    return unique_names


class Imputer(object):
    def __init__(self, run_id, k, model_path, num_feature, train_coarse_set=None, val_coarse_set=None,
                 num_workers=0):
        self.k = k
        self.run_id = run_id
        self.train_coarse_set = train_coarse_set
        self.val_coarse_set = val_coarse_set
        self.model_path = model_path

        # Hyper-parameters
        self.epochs = 40
        self.in_channels, self.out_channels = num_feature, 64
        self.learning_rate = 0.001
        self.bs = 32
        self.kl_coef = None

        if self.train_coarse_set is None:
            assert self.val_coarse_set is None
        else:
            assert self.val_coarse_set is not None
        if self.train_coarse_set is not None:
            self.train_loader, self.val_loader = \
                DataLoader(self.train_coarse_set, self.bs, num_workers=0, shuffle=True), \
                DataLoader(self.val_coarse_set, self.bs, num_workers=0)

        self.model, self.optimizer = self.get_imputation_settings(
            self.in_channels, self.out_channels, self.learning_rate
        )

    def get_imputation_settings(self, in_channels, out_channels, lr):
        model = VGAE(VariationalGraphSageEncoder(in_channels, out_channels))
        model = model.to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        return model, optimizer

    def train(self):
        early_stopper = EarlyStopper()
        for epoch in range(1, self.epochs + 1):
            epoch_loss_list = []
            for batch in tqdm(self.train_loader, leave=False, position=0, desc='Epoch {}'.format(epoch)):
                loss = gnn_train_batch(batch, self.model, self.optimizer, DEVICE, 'contact', kl_coef=self.kl_coef)
                epoch_loss_list.append(loss)
            auc, ap, val_loss = gnn_approx_evaluate_all(self.val_loader, self.model, DEVICE, 'contact', kl_coef=self.kl_coef)
            mean_loss = np.array(epoch_loss_list).mean()
            print(
                f'\t Epoch: {epoch:03d}, train loss: {mean_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {auc:.4f}, Val AP: {ap:.4f}')
            if early_stopper.early_stop(val_loss):
                break
            # print(f'\t Epoch: {epoch:03d}, loss: {mean_loss:.4f}')
        save_model(epoch, self.model, self.optimizer, loss, self.model_path)

    def load_model(self):
        self.model, self.optimizer, *_ = load_model(self.model, self.optimizer, self.model_path)

    @torch.no_grad()
    def get_graph_embeddings(self, ds):
        self.model.eval()
        data_loader = DataLoader(ds, batch_size=len(ds.chrom_names), num_workers=0, shuffle=False)
        pooling_model = Pooling(global_mean_pool).to(DEVICE)
        pooling_model.eval()
        graph_embedding_list = []
        for d in data_loader:
            d = easy_to_device(d, DEVICE, ['chrom_name', 'cell_name', 'cell_type', 'edge_label_index', 'edge_weights'])
            node_embeddings = self.model.encode(d.x, d.edge_index)
            pooling_size = torch.zeros(d.num_nodes, dtype=torch.int64).to(DEVICE)
            graph_embedding = pooling_model(node_embeddings, pooling_size)
            graph_embedding_list.append(graph_embedding)
        graph_embeddings = torch.stack(graph_embedding_list, dim=0)
        graph_embeddings = graph_embeddings.detach().cpu().numpy()
        graph_embeddings = np.reshape(graph_embeddings, [-1, self.out_channels])
        return graph_embeddings

    @torch.no_grad()
    def calculate_dist_matrix(self, ds):
        graph_embeddings = self.get_graph_embeddings(ds)
        # print(graph_embeddings.shape)
        return cdist(graph_embeddings, graph_embeddings)

    def impute_dataset(self, ds, scool_of_dataset, out_finer_scool, out_coarse_scool):
        dist_matrix = self.calculate_dist_matrix(ds)
        cell_names = get_cell_names_of_ds(ds)
        chrom_names = ds.chrom_names
        assert len(ds) == dist_matrix.shape[0] * len(chrom_names)
        np.fill_diagonal(dist_matrix, 99999)
        closest_cells_list = convert_dist_mat_to_closest_cells(cell_names, dist_matrix, self.k)
        assert len(closest_cells_list) == len(cell_names)
        bins_selector = cooler.Cooler(scool_of_dataset + '::' + cell_names[0]).bins()[:]
        with tempfile.TemporaryDirectory() as temp_dir:
            # with mp.Pool(workers) as pool:
            #     param_list = [
            #         (cell_names[i], closest_cells_list[i], chrom_names, scool_of_dataset,
            #          bins_selector, temp_dir)
            #         for i in range(len(cell_names))
            #     ]
            #     print('Imputing...')
            #     pool.starmap(impute_and_save_cell, param_list)
            #     print('Done!')
            #     print('Creating .scool from imputed coolers...')
            #     convert_cool_to_scool(temp_dir, out_finer_scool, lambda s: s.rstrip('.cool'))
            print('Imputing...')
            for i in tqdm(list(range(len(cell_names)))):
                impute_and_save_cell(cell_names[i], closest_cells_list[i], chrom_names,
                                     scool_of_dataset, bins_selector, temp_dir)
            print('Done')
            print('Creating .scool from imputed coolers...')
            convert_cool_to_scool(temp_dir, out_finer_scool, lambda s: s.rstrip('.cool'))
        coarsen_scool(out_finer_scool, out_coarse_scool)


if __name__ == '__main__':
    chroms = ['chr'+str(i) for i in range(1, 20)]
    # chroms = ['chr21', 'chr22']
    dataset = ScoolDataset(
        'data/mES/imputation_dataset',
        'data/mES/nagano_100kb.scool',
        chroms, 100000, {
            # 'MG': 'data/human_prefrontal_cortex/MG.bedpe', 'Neuron': 'data/human_prefrontal_cortex/Neuron.bedpe',
            # 'ODC': 'data/human_prefrontal_cortex/ODC.bedpe'
            'ES': 'data/mES/ES.bedpe'
        },
        # hpc_celltype_parser,
        # ['MG', 'ODC', 'Neuron'],
        pre_transform=T.Compose([RemoveSelfLooping(), T.LocalDegreeProfile()])
    )


    K = 4



    # train(data_loader, epochs, model, optimizer, kl_coef=kl_coef)
    # dist_matrix = calculate_dist_matrix(
    #     model, DataLoader(dataset, batch_size=len(chroms), num_workers=8, shuffle=False),
    #     Pooling(global_mean_pool).to(DEVICE), out_channels
    # )
    # impute_dataset(
    #     dataset, 'data/mES/nagano_10kb_raw.scool',
    #     dist_matrix, K, 'data/mES/refined_data/nagano_10kb_raw_imputed{}.scool'.format(K), 10
    # )
# coarsen_scool(
#             'data/mES/refined_data/nagano_10kb_raw_imputed{}.scool'.format(K),
#             'data/mES/refined_data/nagano_100kb_raw_imputed{}.scool'.format(K)
#         )

