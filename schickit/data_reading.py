import pandas as pd
from scipy import sparse
import numpy as np
import cooler
import multiprocessing as mp
import itertools
from collections import defaultdict
from .utils import get_bin_count, grouplist
from .matrix_manipulation import weighted_hic_to_unweighted_graph


def check_chrom_in_order(chrom_df, chrom_names):
    chrom_df = chrom_df.reset_index(drop=True)
    indices = [chrom_df[chrom_df['name'] == c].index[0] for c in chrom_names]
    tmp = -1
    for i in indices:
        if i > tmp:
            tmp = i
        else:
            return False
    return True


def read_scool_as_sparse(scool_path, chrom_names, name_parser=None, desired_celltypes=None, weighted=True, workers=8, sparse_format='csc'):
    cell_names = cooler.fileops.list_scool_cells(scool_path)
    if name_parser is None:
        cell_types = None
    else:
        assert desired_celltypes is not None and callable(name_parser)
        all_cn, cell_names, cell_types = cell_names, [], []
        for c in all_cn:
            ct = name_parser(c, desired_celltypes)
            if ct is not None:
                cell_types.append(ct)
                cell_names.append(c)

    with mp.Pool(workers) as pool:
        matrix_list = pool.starmap(
            read_cool_as_sparse,
            ((scool_path + "::" + cell_name, chrom_names, weighted, sparse_format) for cell_name in cell_names)
        )
    matrix_list = list(itertools.chain.from_iterable(matrix_list))

    # Assume all cells have the same bins dataframe
    bins_selector = cooler.Cooler(scool_path + "::" + cell_names[0]).bins()
    chrom_df = cooler.Cooler(scool_path + "::" + cell_names[0]).chroms()[:]
    assert check_chrom_in_order(chrom_df, chrom_names)
    chrom_df = chrom_df[chrom_df['name'].isin(chrom_names)]
    chrom_df = pd.DataFrame({'name': chrom_names}).merge(chrom_df, on='name')
    result_dict = {
        'cell_names': cell_names,
        'cell_types': cell_types,
        'chrom_df': chrom_df,
        'bins_selector': bins_selector,
        'matrix_list': matrix_list
    }
    if result_dict['cell_types'] is not None:
        assert len(result_dict['cell_types']) == len(result_dict['cell_names'])
    assert len(result_dict['matrix_list']) == len(result_dict['chrom_df']) * len(result_dict['cell_names'])
    return result_dict


def read_cool_as_sparse(cool_path, chrom_names, weighted=True, sparse_format='csc'):
    cool_obj = cooler.Cooler(cool_path)
    matrix_list = []
    for chrom_name in chrom_names:
        mat = cool_obj.matrix(balance=False, sparse=True).fetch(chrom_name)
        if not weighted:
            mat = weighted_hic_to_unweighted_graph(mat)
        if sparse_format == 'coo':
            pass
        elif sparse_format == 'csc':
            mat = mat.tocsc()
        else:
            raise NotImplementedError('Unsupported sparse format.')
        matrix_list.append(mat)
    return matrix_list


def read_multiple_cell_labels(cell_type_list, celltype_bedpe_dict, resolution, chrom_df, sparse_format='csc'):
    for ct in celltype_bedpe_dict:
        celltype_bedpe_dict[ct] = read_labels_as_sparse(
            celltype_bedpe_dict[ct], resolution, chrom_df, sparse_format
        )
    matrix_list = []
    for ct in cell_type_list:
        matrix_list.append(celltype_bedpe_dict[ct])
    return list(itertools.chain.from_iterable(matrix_list))


def read_labels_as_sparse(bedpe_path, resolution, chrom_df, sparse_format='csc'):
    chrom_names = chrom_df['name']
    loop_dict = parsebed(bedpe_path, resolution)
    loop_dict = convert_loop_dict_to_symmetric(loop_dict)
    label_sp_matrix_list = []
    for chrom_name in chrom_names:
        if len(loop_dict) != 0:
            assert chrom_name in loop_dict
            chrom_bin_count = get_bin_count(chrom_df[chrom_df['name']==chrom_name]['length'].iloc[0], resolution)
            current_coord_list = loop_dict[chrom_name]
            indexes_1, indexes_2 = list(zip(*current_coord_list))
            current_coo = sparse.coo_matrix(
                ([1]*len(indexes_1), (indexes_1, indexes_2)),
                shape=(chrom_bin_count, chrom_bin_count)
            )
            label_sp_matrix_list.append(current_coo)
        else:
            chrom_bin_count = get_bin_count(chrom_df[chrom_df['name'] == chrom_name]['length'].iloc[0], resolution)
            current_coo = sparse.coo_matrix(
                ([], ([], [])), shape=(chrom_bin_count, chrom_bin_count)
            )
            label_sp_matrix_list.append(current_coo)
    if sparse_format == 'coo':
        pass
    elif sparse_format == 'csc':
        label_sp_matrix_list = [mat.tocsc() for mat in label_sp_matrix_list]
    else:
        raise NotImplementedError('Unsupported sparse format.')
    return label_sp_matrix_list


def convert_loop_dict_to_symmetric(loop_dict):
    symmetric_dict = dict()
    for chrom_name in loop_dict:
        symmetric_list = []
        for coord in loop_dict[chrom_name]:
            assert coord[0] < coord[1]
            a, b = coord[0], coord[1]
            symmetric_list.append((a, b))
            symmetric_list.append((b, a))
        assert len(symmetric_list) == 2 * len(loop_dict[chrom_name])
        symmetric_dict[chrom_name] = symmetric_list
    return symmetric_dict


def parsebed(chiafile, res=10000, lower=0, upper=5000000, valid_threshold=1):
    """
    Read the reference bedpe file and generate a distionary of positive center points.
    """
    coords = defaultdict(list)
    upper = upper // res
    with open(chiafile) as o:
        for line in o:
            s = line.rstrip().split()
            a, b = float(s[1]), float(s[4])
            a, b = int(a), int(b)
            if a > b:
                a, b = b, a
            a = a // res
            b = b // res
            # all chromosomes including X and Y
            if (b - a > lower) and (b - a < upper) and 'M' not in s[0]:
                # always has prefix "chr", avoid potential bugs
                chrom = 'chr' + s[0].lstrip('chr')
                coords[chrom].append((a, b))
    valid_coords = dict()
    for c in coords:
        current_set = set(coords[c])
        valid_set = set()
        for coord in current_set:
            if coords[c].count(coord) >= valid_threshold:
                valid_set.add(coord)
        valid_coords[c] = valid_set
    return valid_coords


def align_cell_dict_orders(reference_cell_names, cells_dict):
    assert len(reference_cell_names) == len(cells_dict['cell_names'])
    grp_matrix_list = grouplist(cells_dict['matrix_list'], len(cells_dict['chrom_df']))
    reference_df = pd.DataFrame({'cell_names': reference_cell_names})
    right_df = pd.DataFrame({'cell_names': cells_dict['cell_names']})
    right_df['new_id'] = right_df.index
    df = reference_df.merge(right_df, on='cell_names')
    order = df['new_id']
    matrix_list = list(itertools.chain.from_iterable([grp_matrix_list[i] for i in order]))
    cell_names = reference_cell_names.copy()
    cell_types = [cells_dict['cell_types'][i] for i in order]
    return {
        'cell_names': cell_names,
        'cell_types': cell_types,
        'chrom_df': cells_dict['chrom_df'],
        'bins_selector': cells_dict['bins_selector'],
        'matrix_list': matrix_list
    }


