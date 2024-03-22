
import os
import cv2
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


def save_figure(title, fig=None, fig_dir=FIGURE_PATH, show=False, pdf=True, png=True, dpi=600):
    if not fig:
        fig = plt.gcf()
    if png:
        fig.savefig(os.path.join(fig_dir, title + ".png"), dpi=dpi, bbox_inches="tight")
    if pdf:
        fig.savefig(os.path.join(fig_dir, title + ".pdf"), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()


def plt_clear():
    plt.cla()
    plt.clf()
    plt.close()


def remove_ticks(ax):
    for x in ax:
        x.set_xticks([])
        x.set_yticks([])


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
    

def convert(img_path):
    img = cv2.imread(img_path)
    cv2.imwrite(img_path.rsplit(".", 1)[0] + ".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])