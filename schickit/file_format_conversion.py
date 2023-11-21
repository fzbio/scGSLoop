import cooler
import os
from tqdm.auto import tqdm
import pandas as pd
import random


def convert_mES_excel_to_bedpe(excel_path, out_path, sheet_names):
    dfs = [pd.read_excel(
        excel_path, sheet_name=sheet_name, header=0,
        dtype={'chr1': str, 'chr2': str}, engine='openpyxl')
        for sheet_name in sheet_names]
    df = pd.concat(dfs)
    df = df.drop_duplicates()
    df.to_csv(out_path, sep='\t', header=False, index=False)
    # return df


def convert_hpc_excel_to_bedpe(excel_path, out_path, sheet_name):
    df = pd.read_excel(
        excel_path, sheet_name=sheet_name, header=0,
        dtype={'chr1': str, 'chr2': str}, engine='openpyxl'
    )
    # df['chr1'] = df['chr1'].apply(lambda x: 'chr' + x)
    # df['chr2'] = df['chr2'].apply(lambda x: 'chr' + x)
    df.to_csv(out_path, sep='\t', header=False, index=False)


def convert_cool_to_scool(cool_dir, scool_path, parse_func):
    cool_paths = [os.path.join(cool_dir, e) for e in os.listdir(cool_dir)]
    # print(cool_paths)
    for cool_path in tqdm(cool_paths):
        cell_name = parse_func(cool_path)
        clr = cooler.Cooler(cool_path)
        bins = clr.bins()[:]
        pixels = clr.pixels()[:]
        cooler.create_scool(scool_path, {cell_name: bins}, {cell_name: pixels}, mode='a', symmetric_upper=True)
        # print(cell_name)


def aggregate_scool(scool_path, output_cool_path, name_determine_func=None):
    if name_determine_func is None:
        cell_names = cooler.fileops.list_scool_cells(scool_path)
    else:
        all_cell_names = cooler.fileops.list_scool_cells(scool_path)
        cell_names = []
        for cn in all_cell_names:
            if name_determine_func(cn):
                cell_names.append(cn)
    cool_uris = [scool_path + '::' + c for c in cell_names]
    cooler.merge_coolers(output_cool_path, cool_uris, 1000000)


def random_select_subset_scools(scool_path, output_path, cell_num, seed):
    random.seed(seed)
    cell_names = cooler.fileops.list_scool_cells(scool_path)
    random.shuffle(cell_names)
    cell_names = cell_names[:cell_num]
    for name in tqdm(cell_names):
        cool_path = scool_path + '::' + name
        clr = cooler.Cooler(cool_path)
        bins = clr.bins()[:]
        pixels = clr.pixels()[:]
        cooler.create_scool(output_path, {name: bins}, {name: pixels}, mode='a', symmetric_upper=True)


def select_types_of_cells_from_scool(scool_path, output_path, name_parser):
    """
    name_parser is a function that determines if a cell name is of our desired cell type. It returns a boolean value.
    """
    cell_names = cooler.fileops.list_scool_cells(scool_path)
    for name in tqdm(cell_names):
        if name_parser(name):
            cool_path = scool_path + '::' + name
            clr = cooler.Cooler(cool_path)
            bins = clr.bins()[:]
            pixels = clr.pixels()[:]
            cooler.create_scool(output_path, {name: bins}, {name: pixels}, mode='a', symmetric_upper=True)


def parse_human_prefrontal_cortex(cool_path):
    path_list = cool_path.split('/')
    file_name = path_list[-1]
    cell_name = '_'.join(file_name.split('_')[:-2])
    return cell_name


if __name__ == '__main__':
    # df0,  df1 = convert_mES_excel_to_bedpe('../data/mES/mES_bulk_loop.xlsx', None, ['Bulk_HiC_filter', 'H3K4me3_PLACseq_filter'])
    # print(len(df0), len(df1))
    pass

