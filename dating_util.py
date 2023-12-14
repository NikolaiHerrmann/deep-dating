
import random
import numpy as np
import torch

FIGURE_PATH = "figs"
DATASETS_PATH = "../datasets"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
print("Seed set")