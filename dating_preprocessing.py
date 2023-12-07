
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import find_peaks


class DatingPreprocessing:

    def __init__(self, path):
        self.files = glob.glob(path, recursive=True)
        self.num_lines_per_patch = 5
        self.peak_distance = 50
        self.plot = True

    def show_img(self, img):
        plt.imshow(img, cmap="gray")
        plt.show()

    def read_img(self, path):
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    def random_patch(self, img, size):
        min_dim = min(img.shape)
        if min_dim < size:
            print("Warning, image smaller than wanted size! Image size", img.shape, "but wanted", (size, size))
            size = min_dim - 1

        max_width = img.shape[1] - size
        max_height = img.shape[0] - size

        x = np.random.randint(0, max_width)
        y = np.random.randint(0, max_height)

        return img[y:y+size, x:x+size], x, y
    
    def get_num_lines_in_img(self, img):
        img_ones = 1 - (img / 255)

        hist = img_ones.sum(axis=1)
        thresh = np.mean(hist)
        peaks, _ = find_peaks(hist, distance=self.peak_distance)
        peaks = peaks[hist[peaks] > thresh]
        num_lines = len(peaks)

        if self.plot:
            plt.imshow(img, cmap="gray")
            for x in peaks:
                plt.axhline(x, color="orangered", alpha=0.5)
            plt.title(f"{num_lines} lines")

        return num_lines, peaks
    
    def get_patch_size_based_on_lines(self, img, num_lines):
        patch_size = int((img.shape[0] / num_lines) * self.num_lines_per_patch)
        return min(patch_size, min(img.shape))

    def get_random_patches_based_on_lines(self, img, num_patches=20):
        patch_ls = []
        num_lines, _ = self.get_num_lines_in_img(img)
        patch_size = self.get_patch_size_based_on_lines(img, num_lines)

        for _ in range(num_patches):
            patch, x, y = self.random_patch(img, patch_size)
            patch_ls.append(patch)
        
            if self.plot:
                rect = patches.Rectangle((x, y), patch_size, patch_size, linewidth=2, edgecolor="blue", facecolor="none")
                plt.gca().add_patch(rect)

        return patch_ls
    
    def get_window_patches_based_on_lines(self, img):
        num_lines, _ = self.get_num_lines_in_img(img)
        patch_size = self.get_patch_size_based_on_lines(img, num_lines)

        spots = int(img.shape[1] / patch_size)
        y = 0
        x = 0

        for i in range(int(img.shape[0] / patch_size)):
            for j in range(spots):
                patch = img[y:y+patch_size, x:x+patch_size]
                if self.plot:
                    rect = patches.Rectangle((x, y), patch_size, patch_size, linewidth=2, edgecolor="blue", facecolor="none")
                    plt.gca().add_patch(rect)
            
                x += patch_size
            x = 0
            y += patch_size
        
        #print(img.shape[1] / patch_size, img.shape[1] % patch_size)


dp = DatingPreprocessing("t")
img = dp.read_img("../MPS/Download/1375/MPS1375_0001.ppm")

print(dp.get_window_patches_based_on_lines(img))
plt.show()