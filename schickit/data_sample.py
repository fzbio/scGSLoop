import cooler
import scipy
import numpy as np
import pandas as pd
import os
from .data_storage import save_to_pickle
import multiprocessing as mp
import time
from .utils import get_bin_count





#################################################################

# TODO: Debug this script. These functions are buggy because of misuse of matrix selector. Change style to fetching.

#################################################################

human_chrom_names = ['chr' + str(i) for i in range(1, 23)] + ['chrX']
mouse_chrom_names = ['chr' + str(i) for i in range(1, 20)] + ['chrX']


def mp_sample_from_scool(scool_path, cell_list, chrom_list, output_dir, max_distance, patch_size, resolution, balance, workers=12):
    create_output_dir(output_dir)
    # h5file = open_file(
    #     os.path.join(output_dir, 'original_sparse_mat.h5'), mode="w",
    #     title="Original sparse matrices sampled from {}.".format(os.path.basename(os.path.normpath(scool_path)))
    # )

    print('Timer starts')
    start = time.time()


    for cell_name in cell_list:
        clr = cooler.Cooler(scool_path + '::' + cell_name)
        chrom_len_df = clr.chroms()

        partition_list_of_cell = []

        for chrom_name in chrom_list:
            chrom_size = chrom_len_df[chrom_len_df['name'] == chrom_name]['length'].iloc[0]
            tblr_list = get_patch_tblr_tuples(chrom_size, patch_size, max_distance, resolution)
            put_tblr_list_to_partition_list(partition_list_of_cell, tblr_list, chrom_name, cell_name)
        with mp.Pool(workers) as pool:
            matrix_list_of_cell = pool.map(
                sample_a_matrix_from_cell,
                ((p, clr, resolution, patch_size, balance) for p in partition_list_of_cell)
            )
        # write_cell_coo_matrices_to_h5(h5file, cell_name, matrix_list_of_cell)
        save_to_pickle(os.path.join(output_dir, 'original_sparse_mat'), cell_name, matrix_list_of_cell)

        output_to_csv(matrix_list_of_cell)
        break


    # h5file.close()
    end = time.time()
    print(end - start)


def output_to_csv(matrx_dict_list):
    the_list = []
    for e in matrx_dict_list:
        the_list.append([e['chrom_name'], e['start1'], e['end1'], e['start2'], e['end2']])

    df = pd.DataFrame(the_list, columns=['chrom_name', 'start1', 'end1', 'start2', 'end2'])
    df.to_csv('tmp/test_parallel_chrom_headers.csv')


def sample_a_matrix_from_cell(args_tuple):
    p, cool_obj, resolution, patch_size, balance = args_tuple
    matrix = slice_patch_from_cooler(cool_obj, p[-4:], resolution, patch_size, balance)
    return {
        'cell_name': p[0], 'chrom_name': p[1], 'index': p[2],
        'start1': p[3], 'end1': p[4], 'start2': p[5], 'end2': p[6], 'matrix': matrix
    }


def put_tblr_list_to_partition_list(the_list, tblr_list, chrom_name, cell_name):
    # Partition list includes cell and chrom and index information in addition to the original tblr list
    for i, partition in enumerate(tblr_list):
        the_list.append((cell_name, chrom_name, i, *partition))


def slice_patch_from_cooler(cool_obj, genomic_partition, resolution, patch_size, balance):
    p = genomic_partition
    t = p[0] // resolution
    b = p[1] // resolution
    l = p[2] // resolution
    r = p[3] // resolution
    patch = cool_obj.matrix(balance=balance, sparse=True)[t:b, l:r]
    if b - t < patch_size or r - l < patch_size:
        patch = padding(patch, patch_size)
    return patch


def padding(subgraph, subgraph_size):
    # padding_len = subgraph_size - subgraph.shape[0]
    # subgraph = np.pad(subgraph, [(0, padding_len), (0, padding_len)], mode='constant')
    # return subgraph
    subgraph.resize((subgraph_size, subgraph_size))
    return subgraph


def create_output_dir(output_dir):
    try:
        os.makedirs(output_dir)
    except FileExistsError as e:
        # directory already exists
        raise FileExistsError('Datasets must have unique paths.')
    os.makedirs(os.path.join(output_dir, 'original_sparse_mat'))
    os.makedirs(os.path.join(output_dir, 'deanomaly_sparse_mat'))
    os.makedirs(os.path.join(output_dir, 'imputed_dense_mat'))


def get_patch_tblr_tuples(chrom_size, patch_size, max_distance, resolution):
    # Return a list of (top, bottom, left, right) tuples
    # The returned values are original genomic ranges
    tblr_tuples = []
    patch_genome_span = patch_size * resolution
    tb_tuples = get_patch_tb_tuples(chrom_size, patch_size, resolution)
    for tb in tb_tuples:
        t, b = tb
        horizontal_offset = -patch_genome_span
        while horizontal_offset < max_distance and t + horizontal_offset + patch_genome_span < chrom_size:
            if horizontal_offset < 0:
                horizontal_offset = 0
            else:
                horizontal_offset += patch_genome_span
            l = t + horizontal_offset
            r = l + patch_genome_span
            if r > chrom_size:
                r = chrom_size
            tblr_tuples.append((t, b, l, r))
    return tblr_tuples


def get_patch_tb_tuples(chrom_size, patch_size, resolution):  # Return a list of (top, bottom)
    patch_lo_hi_list = list(cooler.util.partition(0, chrom_size, patch_size * resolution))
    return patch_lo_hi_list





# def sample_from_scool(scool_path, cell_list, chrom_list, output_dir, max_distance, patch_size, resolution, balance):
#     create_output_dir(output_dir)
#     h5file = open_file(
#         os.path.join(output_dir, 'original_sparse_mat.h5'), mode="w",
#         title="Original sparse matrices sampled from {}.".format(os.path.basename(os.path.normpath(scool_path)))
#     )
#     for cell_name in cell_list:
#         print('Timer starts')
#         start = time.time()
#
#         clr = cooler.Cooler(scool_path + '::' + cell_name)
#         chrom_len_df = clr.chroms()
#         sparse_matrices = []
#         matrix_headers_of_cell = []
#         for chrom_name in chrom_list:
#             chrom_size = chrom_len_df[chrom_len_df['name'] == chrom_name]['length'].iloc[0]
#             chrom_partition_list = get_patch_tblr_tuples(chrom_size, patch_size, max_distance, resolution)
#             for p in chrom_partition_list:
#                 current_patch = slice_patch_from_cooler(clr, p, resolution, patch_size, balance)
#                 sparse_matrices.append(current_patch)
#             matrix_headers_of_cell.append(chrom_plist_to_headers(chrom_name, chrom_partition_list))
#         matrix_headers_of_cell = pd.concat(matrix_headers_of_cell)
#         matrix_headers_of_cell = matrix_headers_of_cell.reset_index(drop=True)
#         assert not matrix_headers_of_cell.isnull().values.any()
#         assert len(matrix_headers_of_cell) == len(sparse_matrices)
#         # matrix_headers_of_cell.to_csv('tmp/test_chrom_headers.csv')
#
#         # write_chrom_coo_matrices_to_h5(h5file, sparse_matrices, cell_name, chrom_name, cell_partition_list)
#         end = time.time()
#         print(end - start)
#         break
#     h5file.close()
#
#
# def chrom_plist_to_headers(chrom_name, chrom_partition_list):
#     df = pd.DataFrame(chrom_partition_list, columns=['start1', 'end1', 'start2', 'end2'])
#     df.insert(0, 'chrom_name', [chrom_name] * len(chrom_partition_list))
#     return df


if __name__ == '__main__':
    # sample_from_scool('data/mES/nagano_10kb_raw.scool', 'dataset/nagano_test', 2000000)
    # test_cell_list = cooler.fileops.list_coolers('data/test_data/nagano_10kb_qc.scool')
    # print(get_patch_tb_tuples(249250621, 64, 10000))
    print(get_patch_tblr_tuples(249250621, 64, 2000000, 10000))
