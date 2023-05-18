import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_SPLIT_SEED = 1111
SEED = 1111
LOADER_WORKER = 0
