#!/usr/bin/env python3
import glob
import os
import cooler
import numpy as np
from sklearn.preprocessing import minmax_scale
from predict_eval import get_raw_average_pred_dfs
import pandas as pd
import argparse


def average_cells(cell_pred_paths, chrom_sizes_path, resolution=10000, loop_num=None, threshold=None, percentile=None):
    pred_dfs = (pd.read_csv(pred_path, header=0, index_col=False, sep='\t') for pred_path in cell_pred_paths)
    average_pred = get_raw_average_pred_dfs(pred_dfs, chrom_sizes_path, len(cell_pred_paths), resolution)
    proba_mat = average_pred['proba'].to_numpy()[..., np.newaxis]
    average_pred['proba'] = minmax_scale(proba_mat[:, 0], feature_range=(0, 1))
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
    return average_pred


def get_chrom_df(scool_path):
    cell_names = cooler.fileops.list_scool_cells(scool_path)
    chrom_df = cooler.Cooler(scool_path + "::" + cell_names[0]).chroms()[:]
    return chrom_df


if __name__ == '__main__':
    # Create an argument parser object
    parser = argparse.ArgumentParser()

    # Add mutually exclusive options, each followed by a numerical value
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-p", "--percentile", type=float, help="Output the consensus loops with probability > percentile")
    group.add_argument("-n", "--num-loop", dest='num_loop', type=int, help="Output a fixed number of loops")

    # Add positional arguments
    parser.add_argument("raw_scool_path", type=str, help="Path to the raw 10kb scool file")
    parser.add_argument("pred_dir", type=str, help="Path to the directory storing single-cell level predictions")
    parser.add_argument("out_path", type=str, help="Path to the output file path")
    parser.add_argument("assembly_size", type=str, help="Path to the assembly sizes file (e.g. hg38.sizes)")

    args = parser.parse_args()

    raw_scool_path = args.raw_scool_path
    pred_dir = args.pred_dir
    out_path = args.out_path
    chrom_sizes_path = args.assembly_size

    chrom_size_df = get_chrom_df(raw_scool_path)
    if args.num_loop:
        result_df = average_cells(
            glob.glob(os.path.join(pred_dir, '*.csv')), chrom_sizes_path,
            loop_num=args.num_loop
        )
    elif args.percentile:
        result_df = average_cells(
            glob.glob(os.path.join(pred_dir, '*.csv')), chrom_sizes_path,
            percentile=args.percentile
        )
    else:
        raise Exception('Must specify --percentile or --num-loop')
    result_df.to_csv(out_path, sep='\t', index=False)
