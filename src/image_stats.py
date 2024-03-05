
import cv2
import numpy as np
from deep_dating.datasets import MPS, BinDataset, CLaMM, ScribbleLens
from tqdm import tqdm


X = ScribbleLens().X

#X = BinDataset().X_train
# 0.6144142615182454 0.2606306621135704

means = []
vars_ = []

for path in tqdm(X):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255
    means.append(np.mean(img))

mu = np.mean(means)
print(mu)

for path in tqdm(X):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255
    var = np.mean((img - mu) ** 2)
    vars_.append(var)

std = np.sqrt(np.mean(vars_))

print(mu, std)

