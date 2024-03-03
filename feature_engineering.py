import pandas as pd
import numpy as np
import kcounter
import pyfastx
from schickit.utils import get_chrom_sizes
import argparse


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


def create_bin_df_from_chrom_sizes(chrom_sizes, chroms, res):
    chrom_col = []
    start_col = []
    end_col = []
    for c in chroms:
        size = chrom_sizes[c]
        locus_count = size // res + 1 if size % res != 0 else size // res
        chrom_col += [c] * locus_count
        start_col += [l * res for l in range(locus_count)]

        current_chrom_end_col = [l * res + res for l in range(locus_count)]
        current_chrom_end_col[-1] = size
        end_col += current_chrom_end_col
    gw_df = pd.DataFrame({'chrom': chrom_col, 'start': start_col, 'end': end_col})
    return gw_df


def create_kmer_input_file(chrom_size_path, assembly_path, out_csv_path, chroms, resolution):
    ks = [3, 4]
    fasta = pyfastx.Fasta(assembly_path)
    chrom_sizes = get_chrom_sizes(chrom_size_path)
    genomic_coords = create_bin_df_from_chrom_sizes(chrom_sizes, chroms, resolution)
    kmer_dfs = []
    for k in ks:
        kmer_df = scan_kmer_in_each_locus(fasta, genomic_coords, resolution, k, False)
        kmer_dfs.append(kmer_df)
    if len(kmer_dfs) > 1:
        kmer_dfs = [kmer_dfs[0]] + [d.drop(columns=['chrom', 'start', 'end']) for d in kmer_dfs[1:]]
    result_df = pd.concat(kmer_dfs, axis=1)
    result_df.to_csv(out_csv_path, sep='\t', header=True, index=False)


def convert_fimo_df(fimo_df):
    fimo_df = fimo_df.drop(columns=['motif_alt_id'])
    fimo_df = fimo_df.rename(columns={
        'motif_id': 'name', 'sequence_name': 'seqnames',
        'start': 'start', 'stop': 'end', 'p-value': 'pvalue', 'q-value': 'qvalue', 'matched_sequence': 'sequence'
    })
    fimo_df['width'] = fimo_df['end'] - fimo_df['start'] + 1
    fimo_df = fimo_df[fimo_df['pvalue'] <= 1e-6]
    fimo_df = fimo_df.sort_values(by=['seqnames', 'start'])
    return fimo_df


def create_motif_input_file(chrom_size_path, motif_bed_path, out_csv_path, chroms, resolution, fimo_format):
    motif_bed_df = pd.read_csv(motif_bed_path, sep='\t', header=0, index_col=False, comment='#')
    if fimo_format:
        motif_bed_df = convert_fimo_df(motif_bed_df)
    chrom_sizes = get_chrom_sizes(chrom_size_path)
    bins = create_bin_df_from_chrom_sizes(chrom_sizes, chroms, resolution)
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


def get_chrom_list(chrom_size_path):
    chrom_sizes = get_chrom_sizes(chrom_size_path)
    return list(chrom_sizes.keys())


# if __name__ == '__main__':
#
#     create_kmer_input_file(
#         'external_annotations/mm9.chrom.sizes',
#         'data/graph_features/mouse/mm9.fa',
#         'data/graph_features/mouse/mm9.10kb.kmer.csv',
#         ['chr' + str(i) for i in range(1, 20)],
#         10000
#     )
#     create_motif_input_file(
#         'external_annotations/mm9.chrom.sizes',
#         'data/graph_features/mouse/CTCF_mm9.bed',
#         'data/graph_features/mouse/CTCF_mm9.10kb.input.csv',
#         ['chr' + str(i) for i in range(1, 20)], 10000
#     )
#
#     create_kmer_input_file(
#         'external_annotations/hg38.chrom.sizes',
#         'data/graph_features/human/hg38.fa',
#         'data/graph_features/human/hg38.10kb.kmer.csv',
#         ['chr' + str(i) for i in range(1, 23)],
#         10000
#     )
#     create_motif_input_file(
#         'external_annotations/hg38.chrom.sizes',
#         'data/graph_features/human/CTCF_hg38.bed',
#         'data/graph_features/human/CTCF_hg38.10kb.input.csv',
#         ['chr' + str(i) for i in range(1, 23)], 10000
#     )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create kmer and motif input files for the model')
    parser.add_argument('chrom_size_path', type=str, help='Path to the chrom size file (e.g., hg19.sizes). Make sure the file only contains the desired chromosomes.')
    parser.add_argument('assembly_path', type=str, help='Path to the assembly file (e.g., hg19.fa)')
    parser.add_argument('motif_tsv_path', type=str, help='The .tsv output of a FIMO run (e.g., fimo.tsv)')
    parser.add_argument('out_kmer_path', type=str, help='Path to the output kmer feature file')
    parser.add_argument('out_motif_path', type=str, help='Path to the output motif feature file')
    args = parser.parse_args()

    chroms = get_chrom_list(args.chrom_size_path)

    create_kmer_input_file(
        args.chrom_size_path,
        args.assembly_path,
        args.out_kmer_path,
        chroms,
        10000
    )

    create_motif_input_file(
        args.chrom_size_path,
        args.motif_tsv_path,
        args.out_motif_path,
        chroms,
        10000, True
    )
    print('Done')