
import cv2
import numpy as np
from deep_dating import dating_util
from deep_dating.preprocessing import binarize_img
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
from scipy.signal import find_peaks
from enum import Enum


class PatchMethod(Enum):
    RANDOM = 0
    RANDOM_LINES = 1
    SLIDING_WINDOW_LINES = 2


class PatchExtractor:

    def __init__(self, method=PatchMethod.SLIDING_WINDOW_LINES, num_lines_per_patch=4, 
                 line_peak_distance=50, num_random_patches=20, plot=True, 
                 patch_size_for_random=256, calc_pixel_overlap=True,
                 rm_white_pixel_ratio=0.98):
        self.method = method
        self.num_lines_per_patch = num_lines_per_patch
        self.line_peak_distance = line_peak_distance
        self.num_random_patches = num_random_patches
        self.plot = plot
        self.patch_size = patch_size_for_random
        self.calc_pixel_overlap = calc_pixel_overlap
        self.rm_white_pixel_ratio = rm_white_pixel_ratio
        self.num_pixel_overlap = 0
        self.method_funcs = {PatchMethod.RANDOM: self._extract_random,
                             PatchMethod.RANDOM_LINES: self._extract_random_lines,
                             PatchMethod.SLIDING_WINDOW_LINES: self._extract_sliding_window_lines}
    
    def extract_patches(self, img_path, method=None):
        self._read_img(img_path)
        if method:
            self.method = method

        self.patch_ls = []
        func = self.method_funcs.get(self.method)
        if not func:
            print("Unknown method! Returning empty patch list.")
        else:
            func()

        return self.patch_ls
    
    def save_plot(self, title=None, show=False):
        if not self.plot:
            print("No plot possible! Plotting was not set during initialization!")
            return
        if not title:
            title = str(self.method)
        dating_util.save_figure(title, show=show)

    def _read_img(self, path):
        self.img = cv2.imread(path)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.img.shape

        self.img_bin = binarize_img(self.img, show=False)

        if self.plot:
            plt.imshow(self.img, cmap="gray")

    def _get_random_patch(self, img, size):
        min_dim = min(img.shape)
        if min_dim < size:
            print("Warning, image smaller than wanted size! Image size",
                  img.shape, "but wanted", (size, size), "Returning same image.")
            return img, 0, 0

        max_width = img.shape[1] - size
        max_height = img.shape[0] - size

        x = np.random.randint(0, max_width)
        y = np.random.randint(0, max_height)

        return img[y:y+size, x:x+size], x, y

    def _calc_num_lines_in_img(self, extra_show=False):
        hist = (1 - self.img_bin).sum(axis=1)
        thresh = np.mean(hist)

        self.peaks, _ = find_peaks(hist, distance=self.line_peak_distance)
        self.peaks = self.peaks[hist[self.peaks] > thresh]
        self.num_lines = len(self.peaks)
        if self.calc_pixel_overlap:
            self.num_pixel_overlap = int(np.mean(np.diff(self.peaks)))

        if extra_show:
            print("Debug, destroying plot!")
            plt.clf()
            plt.plot(hist)
            plt.axhline(thresh, color="red", alpha=0.5)
            plt.plot(self.peaks, hist[self.peaks], "x", "orange")
            plt.show()

        if self.plot:
            for x in self.peaks:
                plt.axhline(x, color="orangered", alpha=0.5, zorder=5)

    def _calc_patch_size_based_on_lines(self):
        patch_size = int((self.height / self.num_lines) * self.num_lines_per_patch)
        self.patch_size = min(patch_size, self.height, self.width)

    def _draw_rect(self, x, y, color="blue"):
        rect = plt_patches.Rectangle((x, y), self.patch_size, self.patch_size, 
                                     linewidth=2, edgecolor=color, alpha=0.4, 
                                     facecolor=color, linestyle="dotted")
        plt.gca().add_patch(rect)

    def _append_patch(self, patch, x, y):
        bin_patch = self.img_bin[y:y+self.patch_size, x:x+self.patch_size]
        white_pixel_count = np.count_nonzero(bin_patch)
        if white_pixel_count / bin_patch.size > self.rm_white_pixel_ratio:
            return

        self.patch_ls.append(patch)

        if self.plot:
            self._draw_rect(x, y)

    def _extract_random(self):
        for _ in range(self.num_random_patches):
            patch, x, y = self._get_random_patch(self.img, self.patch_size)
            self._append_patch(patch, x, y)

        if self.plot:
            plt.title(f"{self.num_random_patches} Random Patches at Size: {(self.patch_size, self.patch_size)}")

    def _extract_random_lines(self):
        self._calc_num_lines_in_img()
        self._calc_patch_size_based_on_lines()
        self._extract_random()
        if self.plot:
            plt.title(f"{self.num_random_patches} Random Patches (Approx {self.num_lines_per_patch} Lines within each Patch)")

    def _calc_num_patches_fit_in_img(self, num_pixels):
        pixels_can_cover = self.patch_size
        pixels_covered = self.patch_size
        patch_count = 1

        while True:
            pixels_can_cover -= self.num_pixel_overlap
            pixels_can_cover += self.patch_size
            if pixels_can_cover >= num_pixels:
                break
            patch_count += 1
            pixels_covered = pixels_can_cover
    
        start_coor = int((num_pixels - pixels_covered) / 2)

        return patch_count, start_coor

    def _extract_sliding_window_lines(self):
        self._calc_num_lines_in_img()
        self._calc_patch_size_based_on_lines()

        self.overlap_patch_size = self.patch_size - self.num_pixel_overlap

        x_n, x_init = self._calc_num_patches_fit_in_img(self.width)
        y_n, y_init = self._calc_num_patches_fit_in_img(self.height)

        x = x_init
        y = y_init

        for _ in range(y_n):
            for _ in range(x_n):
                patch = self.img[y:y+self.patch_size, x:x+self.patch_size]
                self._append_patch(patch, x, y)

                x += self.overlap_patch_size
            x = x_init
            y += self.overlap_patch_size

        if self.plot:
            plt.title(f"Sliding Window Method (Approx {self.num_lines_per_patch} Lines within each Patch)")


if __name__ == "__main__":
    from src.deep_dating.datasets.dating_dataloader import MPS, CLaMM, ScribbleLens
    from tqdm import tqdm
    
    files = MPS().img_names
    dp = PatchExtractor(method=PatchMethod.SLIDING_WINDOW_LINES)
    for file in tqdm(files):
        patches = dp.extract_patches(file)
        dp.save_plot(show=True)
    
