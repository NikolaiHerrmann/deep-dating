
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
from scipy.signal import find_peaks
from enum import Enum
from dating_datasets import MPS, CLaMM


class PatchMethod(Enum):
    RANDOM = 0
    RANDOM_LINES = 1
    SLIDING_WINDOW_LINES = 2


class PatchExtractor:

    def __init__(self, method=PatchMethod.RANDOM, num_lines_per_patch=5, 
                 line_peak_distance=50, num_patches=20, plot=True, patch_size=256):
        self.method = method
        self.num_lines_per_patch = num_lines_per_patch
        self.line_peak_distance = line_peak_distance
        self.num_patches = num_patches
        self.plot = plot
        self.patch_size = patch_size
        self.method_funcs = {PatchMethod.RANDOM: self._extract_random_patches,
                             PatchMethod.RANDOM_LINES: self._extract_random_patches_based_on_lines,
                             PatchMethod.SLIDING_WINDOW_LINES: self._extract_window_patches_based_on_lines}
    
    def extract_patches(self, img_path, method=None):
        self._read_img(img_path)
        if method:
            self.method = method
        self.patch_ls = []
        func = self.method_funcs.get(self.method)
        if not func:
            print("Unknown method!")
        else:
            func()
        return self.patch_ls
    
    def save_plot(self, title=None, show=False):
        if not self.plot:
            print("No plot! Plotting was not set during initialization!")
            return
        if not title:
            title = str(self.method)
        plt.savefig(title + ".png", dpi=300)
        plt.savefig(title + ".pdf")
        if show:
            plt.show()

    def _read_img(self, path):
        img = cv2.imread(path)
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.plot:
            plt.imshow(self.img, cmap="gray")

    def _get_random_patch(self, img, size):
        min_dim = min(img.shape)
        if min_dim < size:
            print("Warning, image smaller than wanted size! Image size",
                  img.shape, "but wanted", (size, size))
            size = min_dim - 1

        max_width = img.shape[1] - size
        max_height = img.shape[0] - size

        x = np.random.randint(0, max_width)
        y = np.random.randint(0, max_height)

        return img[y:y+size, x:x+size], x, y

    def _calc_num_lines_in_img(self):
        img_ones = 1 - (self.img / 255)

        hist = img_ones.sum(axis=1)
        thresh = np.mean(hist)
        peaks, _ = find_peaks(hist, distance=self.line_peak_distance)
        self.peaks = peaks[hist[peaks] > thresh]
        self.num_lines = len(peaks)

        if self.plot:
            for x in peaks:
                plt.axhline(x, color="orangered", alpha=0.5)

    def _calc_patch_size_based_on_lines(self):
        patch_size = int((self.img.shape[0] / self.num_lines) * self.num_lines_per_patch)
        self.patch_size = min(patch_size, min(self.img.shape))

    def _draw_rect(self, x, y):
        rect = plt_patches.Rectangle((x, y), self.patch_size, self.patch_size, 
                                     linewidth=2, edgecolor="blue", facecolor="none")
        plt.gca().add_patch(rect)

    def _extract_random_patches(self):
        for _ in range(self.num_patches):
            patch, x, y = self._get_random_patch(self.img, self.patch_size)
            self.patch_ls.append(patch)

            if self.plot:
                self._draw_rect(x, y)

        if self.plot:
            plt.title(f"{self.num_patches} Random Patches at Size: {(self.patch_size, self.patch_size)}")

    def _extract_random_patches_based_on_lines(self):
        self._calc_num_lines_in_img()
        self._calc_patch_size_based_on_lines()
        self._extract_random_patches()
        if self.plot:
            plt.title(f"{self.num_patches} Random Patches (Approx {self.num_lines_per_patch} Lines within each Patch)")

    def _extract_window_patches_based_on_lines(self):
        self._calc_num_lines_in_img()
        self._calc_patch_size_based_on_lines()

        x_n = int(self.img.shape[1] / self.patch_size)
        y_n = int(self.img.shape[0] / self.patch_size)

        x_init = int((self.img.shape[1] % self.patch_size) / 2)
        y_init = int((self.img.shape[0] % self.patch_size) / 2)
        x = x_init
        y = y_init

        for _ in range(y_n):
            for _ in range(x_n):
                patch = self.img[y:y+self.patch_size, x:x+self.patch_size]
                self.patch_ls.append(patch)
                if self.plot:
                    self._draw_rect(x, y)

                x += self.patch_size
            x = x_init
            y += self.patch_size

        if self.plot:
            plt.title(f"Sliding Window Method (Approx {self.num_lines_per_patch} Lines within each Patch)")


if __name__ == "__main__":
    files = MPS().img_names
    dp = PatchExtractor()
    for file in files:
        patches = dp.extract_patches(file, PatchMethod.SLIDING_WINDOW_LINES)
        dp.save_plot(show=True)
