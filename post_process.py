import pandas as pd
import os
import glob


def remove_short_distance_loops(df, min_dist=100000, max_dist=1000000):
    df = df[df['y1'] - df['x1'] >= min_dist]
    df = df[df['y1'] - df['x1'] <= max_dist]
    return df


class PostProcessor(object):
    def __init__(self):
        self.filter_df = None

    def read_filter_file(self, filter_path):
        filter_df = pd.read_csv(filter_path, sep='\t', header=None, index_col=False, usecols=[0, 1, 2])
        self.filter_df = filter_df.drop_duplicates()

    def remove_invalid_loops(self, df, output_path=None, proba_threshold=None):
        if self.filter_df is None:
            raise Exception('Must call read_filter_file before calling this method.')
        print(f'{len(df)} raw loops.')
        df1 = df.merge(self.filter_df, how='left', left_on=['chrom1', 'x1'], right_on=[0, 1], indicator=True)
        df2 = df.merge(self.filter_df, how='left', left_on=['chrom2', 'y1'], right_on=[0, 1], indicator=True)
        mask1 = df1['_merge'] == 'left_only'
        mask2 = df2['_merge'] == 'left_only'
        mask = mask1 & mask2
        clean_df = df[mask]
        if proba_threshold is not None:
            clean_df = clean_df[clean_df['proba']>=proba_threshold]
        print(f'{len(clean_df)} loops after filtering.')
        if output_path is not None:
            if len(clean_df) > 0:
                clean_df.to_csv(output_path, sep='\t', header=True, index=False)
        return clean_df

    def remove_invalid_loops_in_dir(self, in_dir, out_dir, proba_threshold=None):
        # print(self.filter_df)
        os.makedirs(out_dir, exist_ok=True)
        cell_pred_paths = glob.glob(os.path.join(in_dir, '*.csv'))
        for pred_path in cell_pred_paths:
            print(f'Processing {pred_path}')
            df = pd.read_csv(pred_path, header=0, index_col=False, sep='\t')
            file_name = pred_path.split('/')[-1]
            df = self.remove_invalid_loops(
                df, output_path=os.path.join(out_dir, file_name), proba_threshold=proba_threshold
            )


if __name__ == '__main__':
    processor = PostProcessor()
    processor.read_filter_file('region_filter/mm10_filter_regions.txt')
    processor.remove_invalid_loops_in_dir(
        'preds/mES_k4_run0_MEAN_pred_18', 'preds/filtered_mES_k4_run0_MEAN_pred_18', proba_threshold=0.18
    )
