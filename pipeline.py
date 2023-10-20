#!/usr/bin/env python3
import os.path

from imputation import Imputer
from torch_geometric import transforms as T
from nn_data import ScoolDataset
from nn_data import RemoveSelfLooping, PositionalEncoding, ReadKmerFeatures, ReadMotifFeatures, get_split_dataset
from loop_calling import GnnLoopCaller, estimate_chrom_loop_num_train
import numpy as np
import torch
import random
from utils import get_imputes_scool_paths, get_loop_calling_dataset_paths
from configs import DATA_SPLIT_SEED, SEED
import sys


if __name__ == '__main__':

    # Imputer configs
    run_time = int(sys.argv[1])
    K = int(sys.argv[2])
    SEED = SEED + run_time
    DATA_SPLIT_SEED = DATA_SPLIT_SEED + run_time
    if K == 3:
        SEED += 1
        DATA_SPLIT_SEED += 1
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # run_time = 0
    run_id = f'mES_k{K}_run{run_time}_GNNFINE'
    model_dir = 'models'
    imputation_dataset_path = 'data/mES/filtered_imputation_dataset'
    coarse_scool = 'data/mES/nagano_100kb_filtered.scool'
    finer_scool = 'data/mES/nagano_10kb_filtered.scool'
    kmer_feature_path = 'data/graph_features/mouse/mm10.10kb.kmer.csv'
    motif_feature_path = 'data/graph_features/mouse/CTCF_mm10.10kb.input.csv'

    refined_data_dir = 'refined_scools'
    imputed_coarse_scool_name = f'{run_id}_coarse_imputed'
    imputed_finer_scool_name = f'{run_id}_finer_imputed'


    # Loop calling configs
    graph_dir = 'graph_data'
    image_dir = 'image_data'
    loop_calling_dataset_name = f'{run_id}_graph'
    loop_calling_train, loop_calling_val, loop_calling_test = \
        get_loop_calling_dataset_paths(graph_dir, loop_calling_dataset_name)


    # Shared configs
    chroms = ['chr' + str(i) for i in range(1, 20)]
    bedpe_dict = {
            # 'MG': 'data/human_prefrontal_cortex/MG.bedpe', 'Neuron': 'data/human_prefrontal_cortex/Neuron.bedpe',
            # 'ODC': 'data/human_prefrontal_cortex/ODC.bedpe'
            'ES': 'data/mES/ES.bedpe'
        }
    # name_parser = hpc_celltype_parser
    name_parser = None
    # desired_cell_types = ['MG', 'ODC', 'Neuron']
    desired_cell_types = None


    # Imputing starts here

    imputed_finer_scool_train, imputed_finer_scool_val, imputed_finer_scool_test = get_imputes_scool_paths(
        refined_data_dir, imputed_finer_scool_name
    )
    imputed_coarse_scool_train, imputed_coarse_scool_val, imputed_coarse_scool_test = get_imputes_scool_paths(
        refined_data_dir, imputed_coarse_scool_name
    )

    # chroms = ['chr21', 'chr22']
    raw_coarse_dataset = ScoolDataset(
        imputation_dataset_path,
        coarse_scool,
        chroms, 100000, bedpe_dict,
        name_parser,
        desired_cell_types,
        pre_transform=T.Compose([RemoveSelfLooping(), T.LocalDegreeProfile()])
    )
    raw_coarse_train, raw_coarse_val, _ = get_split_dataset(
        raw_coarse_dataset, 10, run_time, raw_coarse_dataset.chrom_names, 'loop', DATA_SPLIT_SEED
    )
    imputer = Imputer(
        run_id, K, f'{model_dir}/{run_id}_impute.pt', raw_coarse_train.num_features, raw_coarse_train, raw_coarse_val
    )
    # imputer.train()
    imputer.load_model()
    # imputer.impute_dataset(raw_coarse_train, finer_scool, imputed_finer_scool_train, imputed_coarse_scool_train, 4)
    # imputer.impute_dataset(raw_coarse_val, finer_scool, imputed_finer_scool_val, imputed_coarse_scool_val, 4)

    # Loop calling starts here

    # chroms = ['chr21', 'chr22']
    train_chroms = chroms[:10]
    val_chroms = chroms[10:]
    gnn_train_set = ScoolDataset(
        loop_calling_train,
        imputed_finer_scool_train,
        train_chroms, 10000, bedpe_dict,
        name_parser, desired_cell_types,
        pre_transform=T.Compose([
            RemoveSelfLooping(),
            ReadKmerFeatures(
                kmer_feature_path, train_chroms, True, os.path.join(model_dir, f'{run_id}_kmer_scaler_calling.pkl')
            ),
            ReadMotifFeatures(
                motif_feature_path, train_chroms, True, os.path.join(model_dir, f'{run_id}_motif_scaler_calling.pkl')
            ),
            PositionalEncoding()
        ])
    )
    gnn_val_set = ScoolDataset(
        loop_calling_val,
        imputed_finer_scool_val,
        val_chroms, 10000, bedpe_dict,
        name_parser, desired_cell_types,
        pre_transform=T.Compose([
            RemoveSelfLooping(),
            ReadKmerFeatures(
                kmer_feature_path, val_chroms, False, os.path.join(model_dir, f'{run_id}_kmer_scaler_calling.pkl')
            ),
            ReadMotifFeatures(
                motif_feature_path, val_chroms, False, os.path.join(model_dir, f'{run_id}_motif_scaler_calling.pkl')
            ),
            PositionalEncoding()
        ])
    )
    assert len(gnn_train_set.chrom_names) == len(train_chroms)
    assert len(gnn_train_set) > len(gnn_val_set)
    gnn_caller = GnnLoopCaller(run_id, chroms, f'{model_dir}/{run_id}.pt', gnn_train_set.num_features, gnn_train_set, gnn_val_set)
    gnn_caller.train()
    # gnn_caller.load_model()

    train_chrom_num = 100
    chrom_loop_num_dict = estimate_chrom_loop_num_train(gnn_caller.train_set, run_id, model_dir)
    print(chrom_loop_num_dict)








    # remove_datasets([loop_calling_train, loop_calling_val])
