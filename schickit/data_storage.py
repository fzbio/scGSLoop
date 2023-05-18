import cooler
import pandas as pd
import numpy as np
import pickle
import os
import multiprocessing as mp
from schickit.utils import grouplist
from scipy import sparse


def save_to_pickle(output_dir, cell_name, matrix_dicts):
    pickle.dump(
        matrix_dicts,
        open(os.path.join(output_dir, "{}.pkl".format(cell_name)), "wb")
    )


def create_pixel_df(matrices_of_cell, bins_selector, chroms):
    sanitizer = cooler.create.sanitize_pixels(bins_selector, sort=True, tril_action='raise')
    dfs = []
    for i, m in enumerate(matrices_of_cell):
        chrom = chroms[i]
        bin_1 = np.asarray(bins_selector[bins_selector['chrom'] == chrom].index.values[m.row], dtype='int32')
        bin_2 = np.asarray(bins_selector[bins_selector['chrom'] == chrom].index.values[m.col], dtype='int32')
        df = pd.DataFrame({'bin1_id': bin_1, 'bin2_id': bin_2, 'count': m.data})
        dfs.append(df)
    return sanitizer(pd.concat(dfs).reset_index(drop=True))


def save_sparse_to_scool(cells_dict, chroms, out_scool, workers=10):
    cell_names = cells_dict['cell_names']
    bins = cells_dict['bins_selector'][:]
    matrix_list = cells_dict['matrix_list']
    matrix_list = [sparse.triu(m, k=0, format='coo') for m in matrix_list]
    matrix_list = grouplist(matrix_list, len(chroms))
    with mp.Pool(workers) as pool:
        pixel_dfs = pool.starmap(create_pixel_df, [(group, bins, chroms) for group in matrix_list])
    cell_name_pixels_dict = {cell_names[i]: pixel_dfs[i] for i in range(len(cell_names))}
    cooler.create_scool(out_scool, bins, cell_name_pixels_dict, ordered=True, symmetric_upper=True, mode='w', triucheck=True, dupcheck=True, ensure_sorted=True)


def save_sparse_to_cool(cool_path, chroms, matrix_list, bins_selector):
    matrix_list = [sparse.triu(m, k=0, format='coo') for m in matrix_list]
    bins = bins_selector[:]
    pixel_df = create_pixel_df(matrix_list, bins, chroms)
    cooler.create_cooler(cool_path, bins, pixel_df, ordered=True, symmetric_upper=True, mode='w', triucheck=True, ensure_sorted=True)

# class MatrixAttributes(IsDescription):
#     chrom_name = StringCol(128)
#     mat_index = Int64Col()
#     start1 = Int64Col()
#     end1 = Int64Col()
#     start2 = Int64Col()
#     end2 = Int64Col()
#
#
# def write_cell_coo_matrices_to_h5(h5file, cell_name, matrix_dicts):
#     cell = h5file.create_group("/", cell_name, cell_name)
#     info_list = [None] * len(matrix_dicts)
#     for i, mat_dict in enumerate(matrix_dicts):
#         info_list[i] = [
#             mat_dict['chrom_name'], mat_dict['index'], mat_dict['start1'],
#             mat_dict['end1'], mat_dict['start2'], mat_dict['end2']
#         ]
#         current_mat = mat_dict['matrix']
#         h5file.create_array(cell, 'shape_{}'.format(i), np.array(current_mat.shape))
#         h5file.create_array(cell, 'row_{}'.format(i), np.array(current_mat.row))
#         h5file.create_array(cell, 'col_{}'.format(i), np.array(current_mat.col))
#         h5file.create_array(cell, 'data_{}'.format(i), np.array(current_mat.data))
#     table = h5file.create_table(cell, 'matrix_info', MatrixAttributes, 'Info of matrices of cell {}'.format(cell_name))
#     h5file.flush()


# def write_chrom_coo_matrices_to_h5(h5file, matrices, cell_name, chrom_name, partition_list):
#     try:
#         cell = getattr(h5file.root, cell_name)
#     except AttributeError:
#         cell = h5file.create_group("/", cell_name, cell_name)
#     # if chrom_name not in cell:
#     #     chrom = h5file.create_group(cell, chrom_name, chrom_name)
#     # else:
#     #     chrom = cell.chrom_name
#     chrom = h5file.create_group(cell, chrom_name, chrom_name)
#     for i, p in enumerate(partition_list):
#         current_mat = matrices[i]
#         mat_group = h5file.create_group(chrom, 'matrix' + str(i), 'matrix' + str(i))
#         table = h5file.create_table(mat_group, 'matrix_attrs', MatrixAttributes, 'matrix_attrs')
#         current_partition = partition_list[i]
#         table.append([(cell_name, chrom_name, *current_partition)])
#         the_shape = h5file.create_array(mat_group, 'matrix_shape', np.array(current_mat.shape))
#         the_rows = h5file.create_array(mat_group, 'matrix_rows', np.array(current_mat.row))
#         the_cols = h5file.create_array(mat_group, 'matrix_cols', np.array(current_mat.col))
#         the_data = h5file.create_array(mat_group, 'matrix_data', np.array(current_mat.data))
#     h5file.flush()
