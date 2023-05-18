import cooler
import os
from .file_format_conversion import convert_cool_to_scool, parse_human_prefrontal_cortex
from tqdm.auto import tqdm
import time
from tempfile import TemporaryDirectory


def get_chrom_sizes(file_path):
    sizes = {}
    with open(file_path, 'r') as fp:
        for line in fp:
            line_split = line.split()
            line_split[0] = line_split[0]
            sizes[line_split[0]] = int(line_split[1])
    return sizes


def coarsen_scool(scool_path, out_scool_path):
    print('Coarsening data...')
    with TemporaryDirectory() as temp_dir:
        cell_list = cooler.fileops.list_scool_cells(scool_path)
        for cell_path in tqdm(cell_list):
            cell_uri = scool_path + '::' + cell_path
            # print(cell_uri)
            cell_name = cell_uri.split('/')[-1]
            cooler.coarsen_cooler(cell_uri, os.path.join(temp_dir, cell_name) + '_100kb_contacts.cool', 10, 1000000, 12)
        convert_cool_to_scool(temp_dir, out_scool_path, parse_human_prefrontal_cortex)
    print('Done!')


def get_bin_count(base_count, resolution):
    if base_count % resolution != 0:
        return int(base_count / resolution) + 1
    else:
        return int(base_count / resolution)


def grouplist(L, grp_size):
    starts = range(0, len(L), grp_size)
    stops = [x + grp_size for x in starts]
    groups = [L[start:stop] for start, stop in zip(starts, stops)]
    return groups


def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te-ts))
        return result
    return timed
