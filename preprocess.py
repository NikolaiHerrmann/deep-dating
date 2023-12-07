
import matplotlib.pyplot as plt
import cv2
from multiprocessing import Pool
import os
from torchvision import transforms
import numpy as np
from scipy.signal import find_peaks
import torch
import random


RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


DATA_PATH = "../MPS/Download"

# def preprocess(img_path):
#     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(256)])
#     # img = transform(img)
#     # img = img.permute(1, 2, 0) 

#     img_rev = 1 - (img / 255)
#     hist = img_rev.sum(axis=1)
#     thresh = np.mean(hist)

#     print(hist.shape)
#     peaks, _ = find_peaks(hist, distance=50)
#     print(peaks > thresh)
#     plt.axhline(thresh, color="r")
#     plt.plot(peaks, hist[peaks], "x")
#     plt.plot(hist)
#     plt.show()

#     # plt.imshow(img)
#     # plt.show()

def random_patch(img_path, num_patches=20, patch_size=550):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_shape = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transforms.ToTensor()(img)

    img_name = os.path.basename(img_path)
    img_name, img_type = tuple(img_name.split("."))

    for i in range(num_patches):
        size = min(min(img_shape), patch_size)
        print(size)
        patch = transforms.RandomCrop(size)(img)
        input = transforms.Resize(224)(patch)
        new_img_name = f"{img_name}_p{i}.{img_type}"
        # path = "~/Downloads/mps_test"
        # cv2.imwrite(os.path.join(path, new_img_name))
        plt.imshow(input.permute(1, 2, 0) )
        plt.show()


if __name__ == "__main__":
    paths = [os.path.join(DATA_PATH, "1375", "MPS1375_0375.ppm")]
    paths = [os.path.join(DATA_PATH, "1375", "MPS1375_0011.ppm")]
    # with Pool() as pool:
    #     pool.map(preprocess, paths)
    random_patch(paths[0])
