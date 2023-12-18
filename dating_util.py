
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

FIGURE_PATH = "figs"
DATASETS_PATH = "../datasets"
SEED = 42

def set_seed():
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    print(f"Seed set to {SEED}")

set_seed()

def save_figure(title, fig_dir=FIGURE_PATH, show=False):
    plt.savefig(os.path.join(fig_dir, title + ".png"), dpi=300)
    plt.savefig(os.path.join(fig_dir, title + ".pdf"))

    if show:
        plt.show()