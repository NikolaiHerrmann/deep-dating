
import cv2
import numpy as np
from deep_dating.datasets import MPS, BinDataset, CLaMM, ScribbleLens
from tqdm import tqdm


def get_mean_std(img_arr, verbose=True):
    """
    Adapted from https://stackoverflow.com/a/73359050
    """

    means = []
    vars_ = []

    for path in tqdm(img_arr):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255
        means.append(np.mean(img))

    mu = np.mean(means)

    if verbose:
        print("Mean", mu)

    for path in tqdm(img_arr):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255
        var = np.mean((img - mu) ** 2)
        vars_.append(var)

    std = np.sqrt(np.mean(vars_))

    if verbose:
        print("Mean", mu, "Std", std)

    return mu, std


if __name__ == "__main__":

    X = BinDataset(augment=True).X_train
    get_mean_std(X)
    
    # 0.6144142615182454 0.2606306621135704

