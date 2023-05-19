import torch

SCOOL_100KB = 'data/human_prefrontal_cortex/luo_100kb_filtered.scool'
SCOOL_10KB = 'data/human_prefrontal_cortex/luo_10kb_filtered.scool'

MODEL_ID = 'mES_k3_GNNFINE'
CHROMOSOMES = ['chr' + str(i) for i in range(1, 23)]

MODEL_DIR = 'models'
OUT_DIR = f'preds/demo_output'
THRESHOLD = 0.5

MOTIF_FEATURE_PATH = f'data/graph_features/human/CTCF_hg19.10kb.input.csv'
KMER_FEATURE_PATH = f'data/graph_features/human/hg19.10kb.kmer.csv'

IMPUTE = True
OUT_IMPUTED_SCOOL_100KB = f'refined_scools/demo_imputed_coarse.scool'
OUT_IMPUTED_SCOOL_10KB = f'refined_scools/demo_imputed_finer.scool'
IMPUTATION_DATASET_DIR = 'data/human_prefrontal_cortex/demo_imputation_dataset_no_label'

GENOME_REGION_FILTER = 'region_filter/hg19_filter_regions.txt'  # Can be None





# Variables below are shared across all scripts
LOADER_WORKER = 0
# =======================================================
# The user doesn't need to modify the following variables
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_SPLIT_SEED = 1111
SEED = 1111
K = 3
