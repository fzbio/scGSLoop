import pandas as pd
import numpy as np
import kcounter
import pyfastx
import os
import sys
import cooler
from sklearn.preprocessing import StandardScaler


def get_all_k_length(elements, prefix, n, k, all_kmer):
    # Base case: k is 0,
    # print prefix
    if (k == 0):
        all_kmer.append(prefix)
        return

    # One by one add all characters
    # from set and recursively
    # call for k equals to k-1
    for i in range(n):
        # Next character of input added
        new_prefix = prefix + elements[i]

        # k is decreased, because
        # we have added a new character
        get_all_k_length(elements, new_prefix, n, k - 1, all_kmer)


def scan_kmer_in_each_locus(fa, full_headers: pd.DataFrame, res, k, relative_freq):
    kmer_names = []
    get_all_k_length('ATCG', '', 4, k, kmer_names)
    full_headers = full_headers.copy()

    def get_locus_kmer_counts(row):
        chrom_name = row['chrom']
        seq = fa[chrom_name][row['start']:row['end']]
        seq = seq.seq
        current_locus_kmer_counts = np.zeros((len(kmer_names),), dtype='float')
        kmer_dict = kcounter.count_kmers(seq, k, relative_frequencies=relative_freq)
        for i, name in enumerate(kmer_names):
            if name in kmer_dict:
                current_locus_kmer_counts[i] = kmer_dict[name]
        return pd.Series(current_locus_kmer_counts, dtype='float')

    the_mat = np.asarray(full_headers.apply(get_locus_kmer_counts, axis=1))
    assert the_mat.shape[0] == len(full_headers)
    assert the_mat.shape[1] == len(kmer_names)
    assert full_headers.values.shape[1] == 3
    feature_df = pd.DataFrame(data=the_mat, columns=kmer_names)
    result = pd.concat([full_headers.reset_index(drop=True), feature_df], axis=1)
    return result


def create_kmer_input_file(scool_file_path, assembly_path, out_csv_path, chroms, resolution):
    ks = [3, 4]
    fasta = pyfastx.Fasta(assembly_path)
    cell_names = cooler.fileops.list_scool_cells(scool_file_path)
    bins = cooler.Cooler(scool_file_path + "::" + cell_names[0]).bins()[:]
    genomic_coords = []
    for chrom in chroms:
        genomic_coords.append(bins[bins['chrom'] == chrom])
    genomic_coords = pd.concat(genomic_coords)
    kmer_dfs = []
    for k in ks:
        kmer_df = scan_kmer_in_each_locus(fasta, genomic_coords, resolution, k, False)
        kmer_dfs.append(kmer_df)
    if len(kmer_dfs) > 1:
        kmer_dfs = [kmer_dfs[0]] + [d.drop(columns=['chrom', 'start', 'end']) for d in kmer_dfs[1:]]
    result_df = pd.concat(kmer_dfs, axis=1)
    result_df.to_csv(out_csv_path, sep='\t', header=True, index=False)


def create_motif_input_file(scool_file_path, motif_bed_path, out_csv_path, chroms, resolution):
    motif_bed_df = pd.read_csv(motif_bed_path, sep='\t', header=0, index_col=False)
    cell_names = cooler.fileops.list_scool_cells(scool_file_path)
    bins = cooler.Cooler(scool_file_path + "::" + cell_names[0]).bins()[:]
    genomic_coords = []
    for chrom in chroms:
        genomic_coords.append(bins[bins['chrom'] == chrom])
    bins = pd.concat(genomic_coords).reset_index(drop=True)
    motif_bed_df['start'] = motif_bed_df['start'] // resolution * resolution
    motif_bed_df = motif_bed_df[['seqnames', 'start', 'strand']]
    motif_bed_df['count'] = np.zeros(len(motif_bed_df), dtype='float')
    motif_bed_df = motif_bed_df.groupby(['seqnames', 'start', 'strand'], as_index=False).count()
    motif_bed_df = motif_bed_df.rename(columns={'seqnames': 'chrom'})
    pos_df = motif_bed_df[motif_bed_df['strand'] == '+'].drop(columns=['strand'])
    neg_df = motif_bed_df[motif_bed_df['strand'] == '-'].drop(columns=['strand'])
    pos_df = pos_df.rename(columns={'count': 'pos_count'})
    neg_df = neg_df.rename(columns={'count': 'neg_count'})
    merged = bins.merge(pos_df, on=['chrom', 'start'], how='left')
    merged = merged.merge(neg_df, on=['chrom', 'start'], how='left')
    merged = merged.fillna(0)
    merged.to_csv(out_csv_path, sep='\t', header=True, index=False)





if __name__ == '__main__':
    create_kmer_input_file(
        'data/mES/nagano_10kb_filtered.scool',
        'data/graph_features/mouse/mm10.fa',
        'data/graph_features/mouse/mm10.10kb.more.kmer.csv',
        ['chr' + str(i) for i in range(1, 20)],
        # ['chr22'],
        10000
    )
    create_kmer_input_file(
        'data/human_prefrontal_cortex/luo_10kb_filtered.scool',
        'data/graph_features/human/hg19.fa',
        'data/graph_features/human/hg19.10kb.more.kmer.csv',
        ['chr' + str(i) for i in range(1, 23)],
        # ['chr22'],
        10000
    )
    # create_motif_input_file(
    #     'data/mES/nagano_10kb_filtered.scool',
    #     'data/graph_features/mouse/CTCF_mm10.bed',
    #     'data/graph_features/mouse/CTCF_mm10.10kb.input.csv',
    #     ['chr' + str(i) for i in range(1, 20)], 10000
    # )