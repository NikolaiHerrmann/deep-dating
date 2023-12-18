
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola


def binarize_img(img, show=False):
    thresh = threshold_sauvola(img)
    bin_img = (img >= thresh).astype(np.uint8) # convert from bool to 0s and 1s

    if show:
        plt.imshow(bin_img, cmap="gray")
        plt.show()

    return bin_img