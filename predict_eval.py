#!/usr/bin/env python3
import glob
import time

import pandas as pd
import tempfile
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

# Import user-defined variables from the config file.
from configs import IMPUTE as do_impute, MODEL_ID as run_id, K, CHROMOSOMES as chroms, MODEL_DIR as model_dir,\
    KMER_FEATURE_PATH as kmer_feature_path, MOTIF_FEATURE_PATH as motif_feature_path, SCOOL_10KB as raw_finer_scool,\
    SCOOL_100KB as raw_coarse_scool, OUT_DIR as pred_out_dir, GENOME_REGION_FILTER as filter_region_path, \
    THRESHOLD, IMPUTATION_DATASET_DIR as imputation_dataset_path, OUT_IMPUTED_SCOOL_100KB as imputed_coarse_scool,\
    OUT_IMPUTED_SCOOL_10KB as imputed_finer_scool

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


# Predicting on the another schic dataset
if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as graph_dir:
        impute_model_path = f'{model_dir}/{run_id}_impute.pt'
        gnn_path = f'{model_dir}/{run_id}.pt'
        graph_dataset_path = f'{graph_dir}/{run_id}_graph_transfer'
        name_parser = None
        desired_cell_types = None
        placeholder_path = 'data/placeholder.bedpe'
        if not os.path.exists(placeholder_path):
            open(placeholder_path, 'w').close()
        bedpe_dict = {'PLACEHOLDER': placeholder_path}
        if do_impute:
            raw_coarse_dataset = ScoolDataset(
                imputation_dataset_path,
                raw_coarse_scool,
                chroms, 100000, bedpe_dict,
                name_parser,
                desired_cell_types,
                pre_transform=T.Compose([RemoveSelfLooping(), T.LocalDegreeProfile()])
            )
            impute_ds = raw_coarse_dataset
            imputer = Imputer(
                run_id, K, impute_model_path, impute_ds.num_features
            )
            imputer.load_model()
            imputer.impute_dataset(impute_ds, raw_finer_scool, imputed_finer_scool, imputed_coarse_scool)
        else:
            imputed_finer_scool = raw_finer_scool
        predict_on_other_dataset(
            run_id, chroms, bedpe_dict, model_dir, imputed_finer_scool, graph_dataset_path, THRESHOLD,
            pred_out_dir, kmer_feature_path, motif_feature_path, name_parser, desired_cell_types
        )
        if filter_region_path is not None:
            processor = PostProcessor()
            processor.read_filter_file(filter_region_path)
            processor.remove_invalid_loops_in_dir(
                pred_out_dir, pred_out_dir + '_filtered', proba_threshold=THRESHOLD
            )
        remove_datasets([graph_dataset_path])

