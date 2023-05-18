import numpy as np
import pandas as pd
import glob
import os
from schickit.utils import get_chrom_sizes
from scipy.sparse import coo_matrix
from tqdm.auto import tqdm
import json
from itertools import combinations


class HubDiscoverer(object):
    # In this class, a gene locus represents its index on the graph adjacency matrix, i.e., the i-th node
    def __init__(self, k_cliques, chroms, loop_threshold=0.5):
        self.k_cliques = k_cliques
        self.chroms = chroms
        self.gene_coords_df = None
        self.promoter_loci = None
        self.pred_dfs = None
        self.chrom_size_dict = None
        self.loop_threshold = loop_threshold
        self.resolution = 40000
        self.hubs = None
        self.aggr_pred_path = None
        self.aggr_df = None
        self.cell_pred_paths = None
        self.hub_belonging_to_sc_pred = None

    def compile(self, gene_coords_file_path, pred_dir, parse_func, chrom_size_path, aggr_pred_path=None):
        self.gene_coords_df = pd.read_csv(gene_coords_file_path, sep=',', header=0, index_col=False)
        promoter_coords_df = self.find_promoter_coords_from_gene_coords(self.gene_coords_df)
        self.promoter_loci = pd.DataFrame(
            {'chr': promoter_coords_df['chr'], 'loci': promoter_coords_df['start'] // self.resolution}
        )
        self.promoter_loci = self.promoter_loci.drop_duplicates(inplace=False).reset_index(drop=True)
        self.chrom_size_dict = get_chrom_sizes(chrom_size_path)
        self.aggr_pred_path = aggr_pred_path
        all_cell_pred_paths = glob.glob(os.path.join(pred_dir, '*.csv'))
        self.cell_pred_paths = []
        self.pred_dfs = []
        for pred_path in all_cell_pred_paths:
            if parse_func(pred_path):
                df = pd.read_csv(pred_path, header=0, index_col=False, sep='\t')
                df = df[df['proba'] > self.loop_threshold]
                self.pred_dfs.append(df)
                self.cell_pred_paths.append(pred_path)

    # def convert_pred_df_to_desired_resolution(self, df):
    #     df['x1'] = df['x1'] // self.resolution * self.resolution
    #     df['x2'] = df['x1'] + self.resolution
    #     df['y1'] = df['y1'] // self.resolution * self.resolution
    #     df['y2'] = df['y1'] + self.resolution
    #     df = df.drop_duplicates(keep=False).reset_index(drop=True)
    #     return df

    @staticmethod
    def find_promoter_coords_from_gene_coords(gene_coords_df):
        promoter_coords = gene_coords_df.copy()
        promoter_starts = []
        for start, end, strand in zip(gene_coords_df['start'], gene_coords_df['end'], gene_coords_df['strand']):
            if strand == '+':
                promoter_starts.append(start - 2000)
            else:
                promoter_starts.append(end)
        promoter_coords['start'] = promoter_starts
        promoter_coords['end'] = promoter_coords['start'] + 2000
        return promoter_coords

    def find_hubs(self):
        # A dictionary where the keys are the hub tuples and the values are the times the hubs appear in
        # all cells. Hub tuples should be the starts of genomic coordinates.
        hubs = dict()
        belongings = dict()
        for i, pred_df in enumerate(tqdm(self.pred_dfs)):
            for chrom in self.chroms:
                current_chrom_loci = self.promoter_loci[self.promoter_loci['chr'] == chrom]['loci']
                chrom_df = pred_df[pred_df['chrom1'] == chrom]
                adj_mat = self.create_coo_from_df(chrom, chrom_df)
                for locus in current_chrom_loci:
                    hub_loci = self.find_gene_centric_hub(locus, adj_mat)
                    if hub_loci is not None:
                        clique = (chrom,) + tuple([node * self.resolution for node in hub_loci])
                        hubs[clique] = hubs.get(clique, 0) + 1
                        belongings[clique] = self.cell_pred_paths[i]
        self.hubs = hubs
        self.hub_belonging_to_sc_pred = belongings

    def get_sub_combination_of_hub_loci(self, hub_loci):
        """
        The last element of hub loci must be the promoter locus.
        """
        sub_combinations = []
        combination_set = hub_loci[:-1]
        promoter_locus = hub_loci[-1]
        for k in np.array(self.k_cliques) - 1:
            sub_combinations = sub_combinations + list(combinations(combination_set, k))
        return [c + (promoter_locus,) for c in sub_combinations]

    def align_with_aggr_hubs(self):
        aggr_df = pd.read_csv(self.aggr_pred_path, header=0, index_col=False, sep='\t')
        aggr_hubs = dict()
        for chrom in self.chroms:
            current_chrom_loci = self.promoter_loci[self.promoter_loci['chr'] == chrom]['loci']
            chrom_df = aggr_df[aggr_df['chrom1'] == chrom]
            adj_mat = self.create_coo_from_df(chrom, chrom_df)
            for locus in current_chrom_loci:
                hub_loci = self.find_gene_centric_hub(locus, adj_mat)
                if hub_loci is not None:
                    clique = (chrom,) + tuple([node * self.resolution for node in hub_loci])
                    aggr_hubs[clique] = aggr_hubs.get(clique, 0) + 1
        common_keys = (aggr_hubs.keys() & self.hubs.keys())
        old_hubs = self.hubs
        self.hubs = {}
        for k in common_keys:
            print(f'{k} -> {self.hub_belonging_to_sc_pred[k]}')
            self.hubs[k] = old_hubs[k]
        # self.hubs = {k: self.hubs[k] for k in common_keys}

    def save_hubs_to_json(self, json_file_path):
        hub_list = []
        count_list = []
        for key in self.hubs:
            chrom = [key[0]]
            loci = [int(locus) for locus in key[1:]]
            hub_list.append(chrom + loci)
            count_list.append(self.hubs[key])
        assert len(hub_list) == len(count_list)
        with open(json_file_path, 'w') as fp:
            json.dump({'hub': hub_list, 'count': count_list}, fp)

    def get_coo_items(self, s):
        return set(zip(s.row, s.col))

    def find_gene_centric_hub(self, gene_locus, adj_mat):
        mask = (adj_mat.row == gene_locus)
        gene_interact_loci = adj_mat.col[mask]
        hub_loci = np.concatenate([gene_interact_loci, [gene_locus]])
        hub_loci.sort()
        # adj_csr = adj_mat.tocsr()
        # hub_graph = adj_csr[hub_loci, :][:, hub_loci]
        # hub_graph = hub_graph.toarray()
        hub_loci_len = len(hub_loci)
        if hub_loci_len in self.k_cliques:
            # if np.sum(hub_graph) > 0.8 * (hub_loci_len ** 2 - hub_loci_len):
            return hub_loci
        return None

    def create_coo_from_df(self, chrom_name, chrom_df):
        size = self.chrom_size_dict[chrom_name] // self.resolution + 1
        x, y = chrom_df['x1'] // self.resolution, chrom_df['y1'] // self.resolution
        x, y = np.concatenate([x, y]), np.concatenate([y, x])
        data = np.ones((len(x),), dtype='int')
        mat = coo_matrix((data, (x, y)), shape=(size, size))
        mat.sum_duplicates()
        mat.data = np.ones((len(mat.data),), dtype='int')
        return mat


if __name__ == '__main__':
    discoverer = HubDiscoverer([3, 4, 5, 6], ['chr' + str(i) for i in range(1, 23)])
    discoverer.compile(
        'external_annotations/gene_coords.csv', 'preds/mES_k3_run0_GNNFINE_transfer2869_replicate0_filtered',
        lambda x: 'Astro' in x, 'external_annotations/hg19.sizes', 'preds/aggr_preds_on_hpc/Astro.cell2869.rep0.csv'
    )
    discoverer.find_hubs()
    discoverer.align_with_aggr_hubs()
    discoverer.save_hubs_to_json('preds/hub_preds/astro.promoter_hubs.40kb.json.test')

    # discoverer = HubDiscoverer([2, 3, 4, 5, 6], ['chr' + str(i) for i in range(1, 23)], loop_threshold=0)
    # discoverer.compile(
    #     'external_annotations/gene_coords.csv', 'tmp/astro_aggr_pred',
    #     lambda x: 'Astro' in x, 'external_annotations/hg19.sizes', 'preds/aggr_preds_on_hpc/Astro.cell2869.rep0.csv'
    # )
    # discoverer.find_hubs()
    # discoverer.align_with_aggr_hubs()
    # discoverer.save_hubs_to_json('preds/hub_preds/astro.promoter_hubs.aggr.40kb.json')
