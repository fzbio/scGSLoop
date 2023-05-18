import cooler
import numpy as np
from tqdm.auto import tqdm


def count_contacts_in_scool(scool_path):
    cell_names = cooler.fileops.list_scool_cells(scool_path)
    name_count_dict = {}
    for cell_name in cell_names:
        clr = cooler.Cooler(scool_path + "::" + cell_name)
        count = clr.info['sum']
        name_count_dict[cell_name] = count
    return name_count_dict


def filter_cells(scool_path, left_cell_num, out_path):
    name_count_dict = count_contacts_in_scool(scool_path)
    counts = np.array(list(name_count_dict.values()))
    threshold = np.partition(counts, -left_cell_num)[-left_cell_num]
    for name in tqdm(name_count_dict):
        if name_count_dict[name] >= threshold:
            cool_path = scool_path + '::' + name
            clr = cooler.Cooler(cool_path)
            bins = clr.bins()[:]
            pixels = clr.pixels()[:]
            cooler.create_scool(out_path, {name: bins}, {name: pixels}, mode='a', symmetric_upper=True)
