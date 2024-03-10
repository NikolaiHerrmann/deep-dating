
import os
import random
import numpy as np
import torch
import pickle
import datetime
import calendar
import matplotlib.pyplot as plt


FIGURE_PATH = "../figs"
DATASETS_PATH = "../../datasets"
SEED = 43
VERBOSE = True


def set_seed():
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True
    if VERBOSE:
        print(f"Seed set to {SEED}")

set_seed()


def save_figure(title, fig=None, fig_dir=FIGURE_PATH, show=False, pdf=True, png=True):
    if not fig:
        fig = plt.gcf()
    if png:
        fig.savefig(os.path.join(fig_dir, title + ".png"), dpi=300, bbox_inches="tight")
    if pdf:
        fig.savefig(os.path.join(fig_dir, title + ".pdf"), bbox_inches="tight")
    if show:
        plt.show()


def plt_clear():
    plt.cla()
    plt.clf()
    plt.close()


def get_date_as_str():
    now = datetime.datetime.now()
    return f"{calendar.month_abbr[now.month]}{now.day}-{now.hour}-{now.minute}-{now.second}"


def get_torch_device(verbose=False):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if verbose:
        print(f"Running torch model on: {device}")
    return device


def to_index(y, verbose=True):
    unique_y = np.unique(y)

    if verbose:
        print("Found", len(unique_y), "classes!")

    idx_lookup = {date: idx for idx, date in enumerate(unique_y)}

    y_idx = np.zeros((y.shape[0], 1), dtype=np.longlong)

    for i, num in enumerate(y):
        y_idx[i] = idx_lookup[num]

    return y_idx, unique_y


def serialize(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def read_serialize(path):
    with open(path, "rb") as f:
        return pickle.load(f)