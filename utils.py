import json
import os
import shutil
import pandas as pd
import numpy as np
import glob


def threshold_pred_dirs_target_mES(target_parent_dir_name):
    """
    Apply thresholding to pred directories and output to another location.
    This function is only for backup purpose.
    """
    for cell_num in [10, 50, 100, 300, 500, 700, 742]:
    # for cell_num in [742]:
        if cell_num == 742:
            replicates = range(1)
        else:
            replicates = range(0, 6)
        for replicate in replicates:
            for suffix in ['', '_filtered']:
                pred_experiment_identifier = f'hpc_k3_run0_GNNFINE_transfer{cell_num}_replicate{replicate}{suffix}'
                target_dir = os.path.join(target_parent_dir_name, pred_experiment_identifier)
                os.makedirs(target_dir)
                csv_paths = glob.glob(os.path.join('preds', pred_experiment_identifier, '*.csv'))
                for pred_path in csv_paths:
                    name = pred_path.split('/')[-1]
                    df = pd.read_csv(pred_path, header=0, index_col=False, sep='\t')
                    df = df[df['proba'] >= 0.5]
                    df.to_csv(
                        os.path.join(target_dir, name), sep='\t', header=True, index=False
                    )


def threshold_pred_dirs_target_hpc(target_parent_dir_name):
    """
    Apply thresholding to pred directories and output to another location.
    This function is only for backup purpose.
    """
    cell_num = 2869
    replicate = 0

    for suffix in ['', '_filtered']:
        pred_experiment_identifier = f'mES_k3_run0_GNNFINE_transfer{cell_num}_replicate{replicate}{suffix}'
        target_dir = os.path.join(target_parent_dir_name, pred_experiment_identifier)
        os.makedirs(target_dir)
        csv_paths = glob.glob(os.path.join('preds', pred_experiment_identifier, '*.csv'))
        for pred_path in csv_paths:
            name = pred_path.split('/')[-1]
            df = pd.read_csv(pred_path, header=0, index_col=False, sep='\t')
            df = df[df['proba'] >= 0.5]
            df.to_csv(
                os.path.join(target_dir, name), sep='\t', header=True, index=False
            )


def kth_diag_indices(a, k):
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols


def observe_to_oe(chrom_mat):
    for k in list(range(chrom_mat.shape[0])) + list(range(-chrom_mat.shape[0] + 1, 0)):
        diag_indices = kth_diag_indices(chrom_mat, k)
        if chrom_mat[diag_indices].sum() > 0:
            chrom_mat[diag_indices] = chrom_mat[diag_indices] / chrom_mat[diag_indices].mean()
    return chrom_mat


def padding(mat, size):
    if mat.shape[0] != size or mat.shape[1] != size:
        mat = np.pad(mat, ((0, size - mat.shape[0]), (0, size - mat.shape[1])), mode='constant')
    return mat


def split_image_into_patches(image_mat, patch_size):
    patches = []
    coords = []
    for i in range(0, image_mat.shape[0], patch_size[0]):
        for j in range(0, image_mat.shape[1], patch_size[1]):
            i_end = i + patch_size[0]
            j_end = j + patch_size[1]
            padding_flag = False
            if i_end > image_mat.shape[0]:
                i_end = image_mat.shape[0]
                padding_flag = True
            if j_end > image_mat.shape[1]:
                j_end = image_mat.shape[1]
                padding_flag = True
            patch = image_mat[i:i_end, j:j_end]
            if padding_flag:
                patch = padding(patch, patch_size[0])
            assert patch.shape[0] == patch_size[0]
            assert patch.shape[1] == patch_size[1]
            patches.append(patch)
            coords.append((i, j))
    return patches, coords


def utria_df_to_symm_df(df):
    df2 = df.copy()
    df2['x1'] = df['y1']
    df2['x2'] = df['y2']
    df2['y1'] = df['x1']
    df2['y2'] = df['x2']
    return pd.concat([df, df2])


def remove_datasets(dir_list):
    for d in dir_list:
        shutil.rmtree(d)


def get_loop_calling_dataset_paths(graph_ds_dir, loop_calling_ds_name):
    train_name = loop_calling_ds_name + '_train'
    val_name = loop_calling_ds_name + '_val'
    test_name = loop_calling_ds_name + '_test'
    train_path = os.path.join(graph_ds_dir, train_name)
    val_path = os.path.join(graph_ds_dir, val_name)
    test_path = os.path.join(graph_ds_dir, test_name)
    return train_path, val_path, test_path


def get_imputes_scool_paths(refined_data_dir, imputed_scool_name):
    train_name = imputed_scool_name + '.train.scool'
    val_name = imputed_scool_name + '.val.scool'
    test_name = imputed_scool_name + '.test.scool'
    train_path = os.path.join(refined_data_dir, train_name)
    val_path = os.path.join(refined_data_dir, val_name)
    test_path = os.path.join(refined_data_dir, test_name)
    return train_path, val_path, test_path


def read_chrom_loopnum_json(json_path):
    with open(json_path) as fp:
        cl_dict = json.load(fp)
        cl_dict = {k: int(cl_dict[k]) for k in cl_dict}
    return cl_dict


def hpc_celltype_parser(cell_name, desired_ctypes):
    # print(cell_name)
    cell_type = cell_name.split('_')[-1]
    if cell_type == 'MG':
        ct = 'MG'
    elif cell_type == 'ODC':
        ct = 'ODC'
    elif cell_type in ['L23', 'L4', 'L5', 'L6', 'Sst', 'Vip', 'Ndnf', 'Pvalb']:
        ct = 'Neuron'
    else:
        return None

    if ct in desired_ctypes:
        return ct
    else:
        return None


def convert_pred_to_coarser_res(input_dir, output_dir, desired_res):
    csv_paths = glob.glob(os.path.join(input_dir, '*.csv'))
    for pred_path in csv_paths:
        name = pred_path.split('/')[-1]
        df = pd.read_csv(pred_path, header=0, index_col=False, sep='\t')
        df['x1'] = df['x1'] // desired_res * desired_res
        df['x2'] = df['x1'] + desired_res
        df['y1'] = df['y1'] // desired_res * desired_res
        df['y2'] = df['y1'] + desired_res
        df = df.drop_duplicates(keep='first', subset=['chrom1', 'x1', 'x2', 'chrom2', 'y1', 'y2'])
        df.to_csv(
            os.path.join(output_dir, name), sep='\t', header=True, index=False
        )


if __name__ == '__main__':
    # threshold_pred_dirs_target_mES('preds_backup')
    # threshold_pred_dirs_target_hpc('preds_backup')
    convert_pred_to_coarser_res('preds_backup/mES_k3_run0_GNNFINE_transfer2869_replicate0_filtered', 'plot_scripts/track_data/preds_40kb', 40000)
