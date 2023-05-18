#!/usr/bin/env python3
import glob
import time

import pandas as pd

from loop_calling import GnnLoopCaller
import numpy as np
import random
from nn_data import StreamScoolDataset, expand_cell_ids_to_graph_ids, ScoolDataset, ReadKmerFeatures, ReadMotifFeatures, PositionalEncoding
from torch_geometric import transforms as T
from nn_data import RemoveSelfLooping, ImageDataset
from torch.utils.data import DataLoader as PlainDataLoader
from metrics import slack_metrics_df, slack_f1_df
import os
from nn_data import get_split_dataset
from utils import read_chrom_loopnum_json, remove_datasets
from post_process import PostProcessor, remove_short_distance_loops
import sys
from configs import DEVICE, SEED, DATA_SPLIT_SEED
from imputation import Imputer
from sklearn.preprocessing import minmax_scale
from matplotlib import pyplot as plt
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


def get_valid_test_cell_names():
    valid_names = []
    data_loader = PlainDataLoader(ImageDataset('image_data/mES_raw_k4_dense_test'), batch_size=1, shuffle=False)
    for data in data_loader:
        valid_names.append(data['cell_name'][0])
    return set(valid_names)


def get_test_cell_names(graph_dataset, chroms, seed, runtime):
    *_, test_set = get_split_dataset(graph_dataset, 10, runtime, chroms, 'loop', seed)
    return test_set


def predict_on_other_dataset(training_run_id, chroms, bedpe_dict, model_dir, fine_scool_path,
                             graph_dataset_path, thresh, pred_out_path, kmer_feature_path, motif_feature_path,
                             name_parser=None, desired_cell_type=None):
    gnn_path = f'{model_dir}/{training_run_id}.pt'
    graph_dataset = StreamScoolDataset(
        graph_dataset_path,
        fine_scool_path,
        chroms, 10000,
        bedpe_dict,
        name_parser, desired_cell_type,
        # hpc_celltype_parser, ['MG', 'ODC', 'Neuron'],
        pre_transform=T.Compose([
            RemoveSelfLooping(),
            ReadKmerFeatures(
                kmer_feature_path, chroms, False, os.path.join(model_dir, f'{run_id}_kmer_scaler_calling.pkl')
            ),
            ReadMotifFeatures(
                motif_feature_path, chroms, False, os.path.join(model_dir, f'{run_id}_motif_scaler_calling.pkl')
            ),
            PositionalEncoding()
        ])
    )
    gnn_caller = GnnLoopCaller(training_run_id, chroms, gnn_path, graph_dataset.num_features)
    gnn_caller.load_model()
    gnn_caller.predict(pred_out_path, graph_dataset, DEVICE, thresh)



def read_bedpe_as_df(bedpe_path):
    label_df = pd.read_csv(
        bedpe_path, header=None, index_col=False, sep='\t', dtype={0: 'str', 3: 'str'},
        names=['chrom1', 'x1', 'x2', 'chrom2', 'y1', 'y2']
    )
    label_df['chrom1'], label_df['chrom2'] = 'chr' + label_df['chrom1'], 'chr' + label_df['chrom2']
    return label_df


def visualize_embedding(embedding, centroid, outpath):
    embedding = embedding[:, :2]
    plt.scatter(embedding[:, 0], embedding[:, 1])
    plt.scatter(centroid[:, 0], centroid[:, 1], c='red')
    plt.savefig(outpath)
    plt.close()


def get_raw_average_pred_dfs(pred_dfs):
    df = pd.concat(pred_dfs)
    df1 = df.groupby(
        ['chrom1', 'x1', 'x2', 'chrom2', 'y1', 'y2'], as_index=False
    ).count()
    df2 = df.groupby(
        ['chrom1', 'x1', 'x2', 'chrom2', 'y1', 'y2'], as_index=False
    ).mean()
    proba = df2['proba']
    size = df1['proba']
    df2['proba'] = proba * size
    return df2


def normalize_dist(df):
    dist_vec = df['y1'] - df['x1']
    for dist in dist_vec.unique():
        df.loc[dist_vec == dist, 'proba'] /= np.mean(df[dist_vec == dist]['proba'])
    return df


def remove_low_confidence_matrices(dfs, alpha=1):
    proba_sums = []
    n_large = int(len(dfs) * alpha)
    for i, df in enumerate(dfs):
        df = df
        proba_sums.append(df['proba'].sum())
    proba_sums = np.array(proba_sums)
    threshold = np.partition(proba_sums.flatten(), -n_large)[-n_large]
    indices = (proba_sums > threshold).nonzero()[0]
    dfs = [dfs[i] for i in indices]
    return dfs


def evaluate_average_cells(cell_pred_paths, bedpe_path, resolution, loop_num=None, threshold=None, percentile=None):
    """
    Evaluate based on the average prediction of a cell type
    cell_pred_paths must be of the same cell type
    """
    pred_dfs = []
    label_df = read_bedpe_as_df(bedpe_path)
    for pred_path in cell_pred_paths:
        pred_dfs.append(pd.read_csv(pred_path, header=0, index_col=False, sep='\t'))
    pred_dfs = [df[df['proba'] > 0.5] for df in pred_dfs]
    for df in pred_dfs:
        df[['x1', 'x2', 'y1', 'y2']] = df[['x1', 'x2', 'y1', 'y2']].astype('int')
    average_pred = get_raw_average_pred_dfs(pred_dfs)
    proba_mat = average_pred['proba'].to_numpy()[..., np.newaxis]
    average_pred['proba'] = minmax_scale(proba_mat[:, 0], feature_range=(0, 1))
    # average_pred2 = average_pred_dfs_short_dist(pred_dfs, chrom_size_df, resolution)
    if threshold is not None:
        average_pred = average_pred[average_pred['proba'] >= threshold]
    elif percentile is not None:
        threshold = np.percentile(average_pred['proba'], percentile)
        average_pred = average_pred[average_pred['proba'] >= threshold]
    else:
        if len(average_pred) > loop_num:
            threshold = average_pred['proba'].nlargest(loop_num).iloc[-1]
            # print(threshold)
            original_len = len(average_pred)
            average_pred = average_pred[average_pred['proba'] >= threshold]
            new_len = len(average_pred)
            print(new_len / original_len)
    candidate_df = average_pred.drop('proba', axis=1)
    # print(len(candidate_df))
    return slack_metrics_df(label_df, candidate_df, resolution) + (len(candidate_df),) + (average_pred,)


def read_snap_excel_preds(excel_path, sheet_name):
    df = pd.read_excel(
        excel_path, sheet_name=sheet_name, header=0,
        dtype={'chr1': str, 'chr2': str}, engine='openpyxl'
    )
    df['chr1'] = 'chr' + df['chr1']
    df['chr2'] = 'chr' + df['chr2']
    df = df.rename(columns={'chr1': 'chrom1', 'chr2': 'chrom2'})
    return df


def random_select_cells_from_ds(ds, chrom_names, desired_cell_num, seed):
    np.random.seed(seed)
    random.seed(seed)
    assert len(ds) % len(chrom_names) == 0
    indices = np.random.choice(len(ds) // len(chrom_names), desired_cell_num, replace=False)
    indices = expand_cell_ids_to_graph_ids(indices * len(chrom_names), len(chrom_names))
    small_ds = ds.index_select(indices)
    assert len(small_ds) == len(chrom_names) * desired_cell_num
    return small_ds


# Predicting on the same schic dataset
# if __name__ == '__main__':
#     run_time = 0
#     K = 4
#     run_id = f'mES_k{K}_run{run_time}_FILTERED'
#     chroms = ['chr' + str(i) for i in range(1, 20)]
#     model_dir = 'models'
#     gnn_path = f'{model_dir}/{run_id}.pt'
#     cnn_path = f'{model_dir}/{run_id}_cnn.pt'
#     imputed_coarse_scool = f'data/mES/refined_data/nagano_100kb_filtered_imputed{K}.scool'
#     imputed_finer_scool = f'data/mES/refined_data/nagano_10kb_filtered_imputed{K}.scool'
#     graph_dir = 'graph_data'
#     image_dir = 'image_data'
#     graph_dataset_path = f'{graph_dir}/mES_k{K}_loop_calling_filtered_dataset'
#     filter_region_path = 'region_filter/mm10_filter_regions.txt'
#     # out_image_dir = f'tmp/{run_id}_test_imgs'
#     gnn_threshold = f'{model_dir}/{run_id}.json'
#     cnn_threshold = 0.18
#     pred_out_dir = f'preds/{run_id}_pred_18'
#     bedpe_dict = {
#             # 'MG': 'data/human_prefrontal_cortex/MG.bedpe', 'Neuron': 'data/human_prefrontal_cortex/Neuron.bedpe',
#             # 'ODC': 'data/human_prefrontal_cortex/ODC.bedpe'
#             'ES': 'data/mES/ES.bedpe'
#     }
#
#     predict_on_same_schic_dataset(
#         run_id, chroms, bedpe_dict, gnn_path, cnn_path, imputed_coarse_scool, imputed_finer_scool,
#         graph_dataset_path, gnn_threshold, cnn_threshold, pred_out_dir, SEED, run_time
#     )
#
#     processor = PostProcessor()
#     processor.read_filter_file(filter_region_path)
#     processor.remove_invalid_loops_in_dir(
#         pred_out_dir, pred_out_dir + '_filtered', proba_threshold=0.18
#     )


# Predicting on the another schic dataset
if __name__ == '__main__':
    run_time = int(sys.argv[1])
    # run_time = 0
    K = int(sys.argv[2])
    cell_num = int(sys.argv[3])
    eval_time = int(sys.argv[4])
    run_id = f'mES_k{K}_run{run_time}_GNNFINE'
    SEED = SEED + eval_time
    chroms = ['chr' + str(i) for i in range(1, 23)]
    model_dir = 'models'
    impute_model_path = f'{model_dir}/{run_id}_impute.pt'
    gnn_path = f'{model_dir}/{run_id}.pt'
    refined_dir = 'refined_scools'
    graph_dir = 'graph_data'
    raw_coarse_scool = 'data/human_prefrontal_cortex/luo_100kb_filtered.scool'
    raw_finer_scool = 'data/human_prefrontal_cortex/luo_10kb_filtered.scool'
    graph_dataset_path = f'{graph_dir}/{run_id}_graph_transfer{cell_num}_replicate{eval_time}'
    # filter_region_path = 'region_filter/mm10_filter_regions.txt'
    filter_region_path = 'region_filter/hg19_filter_regions.txt'

    pred_out_dir = f'preds/{run_id}_transfer{cell_num}_replicate{eval_time}'
    # motif_feature_path = f'data/graph_features/mouse/CTCF_mm10.10kb.input.csv'
    # kmer_feature_path = f'data/graph_features/mouse/mm10.10kb.kmer.csv'
    motif_feature_path = f'data/graph_features/human/CTCF_hg19.10kb.input.csv'
    kmer_feature_path = f'data/graph_features/human/hg19.10kb.kmer.csv'

    name_parser = None
    desired_cell_types = None
    bedpe_dict = {
            'HPC': 'data/placeholder.bedpe'
            # 'ES': 'data/mES/ES.bedpe'
    }
    do_impute = False
    if do_impute:
        imputed_coarse_scool = f'{refined_dir}/{run_id}_coarse_imputed.transfer{cell_num}.replicate{eval_time}.scool'
        imputed_finer_scool = f'{refined_dir}/{run_id}_finer_imputed.transfer{cell_num}.replicate{eval_time}.scool'
        # imputation_dataset_path = 'data/mES/filtered_imputation_dataset'
        imputation_dataset_path = 'data/human_prefrontal_cortex/filtered_imputation_dataset_no_label'
        raw_coarse_dataset = ScoolDataset(
            imputation_dataset_path,
            raw_coarse_scool,
            chroms, 100000, bedpe_dict,
            name_parser,
            desired_cell_types,
            pre_transform=T.Compose([RemoveSelfLooping(), T.LocalDegreeProfile()])
        )

        impute_ds = random_select_cells_from_ds(raw_coarse_dataset, chroms, cell_num, SEED)

        imputer = Imputer(
            run_id, K, impute_model_path, impute_ds.num_features
        )
        # imputer.load_model()
        imputer.impute_dataset(impute_ds, raw_finer_scool, imputed_finer_scool, imputed_coarse_scool)
    else:
        imputed_finer_scool = raw_finer_scool
    predict_on_other_dataset(
        run_id, chroms, bedpe_dict, 'models', imputed_finer_scool, graph_dataset_path, 0,
        pred_out_dir, kmer_feature_path, motif_feature_path, name_parser, desired_cell_types
    )

    processor = PostProcessor()
    processor.read_filter_file(filter_region_path)
    processor.remove_invalid_loops_in_dir(
        pred_out_dir, pred_out_dir + '_filtered', proba_threshold=0
    )
    remove_datasets([graph_dataset_path])


# Evaluating
# if __name__ == '__main__':
    # f1s = evaluate_preds('preds/filtered_mES_k4_run0_MEAN_pred_18', lambda x: 'ES', {'ES': 'data/mES/ES.bedpe'}, 10000, loop_num=260000)
    # with open('preds/f1s/filtered_mES_k4_run0_MEAN_pred_18_f1.json', 'w') as fp:
    #     json.dump(f1s, fp)
    # evaluate_preds('preds/mES_raw_k4_dense_pred_18', lambda x: 'ES', {'ES': 'data/mES/ES.bedpe'}, 10000)
    # evaluate_preds('preds/mES_raw_k4_dense_pred_21', lambda x: 'ES', {'ES': 'data/mES/ES.bedpe'}, 10000)
    # evaluate_preds('preds/mES_raw_k4_dense_pred_25', lambda x: 'ES', {'ES': 'data/mES/ES.bedpe'}, 10000)

    # for k in range(8):
    #     run_time = 0
    #     precision, recall, f1, loop_num = evaluate_average_cells(
    #         glob.glob(os.path.join('preds', f'hpc_k{k}_run{run_time}_REG_transfer10_mES_15_filtered', '*.csv')),
    #         'data/mES/ES.bedpe', 10000, 15000
    #     )
    #     print(f'K={k} -- Loop num: {loop_num}; Precision: {precision}; Recall: {recall}; F1: {f1}')

    # label_df = read_bedpe_as_df('data/mES/ES.bedpe')
    # snap_df = read_snap_excel_preds('data/snaphic_preds/41592_2021_1231_MOESM4_ESM.xlsx', 'Permu0_50')
    # snap_df = snap_df.rename(columns={'chr1': 'chrom1', 'chr2': 'chrom2'})
    # print(slack_f1_df(label_df, snap_df, 10000))

