
import os
# from deep_dating import util
import numpy as np
from multiprocessing import Pool


class ImageMorphRunner:

    def __init__(self, displacement_range=(1, 3), radius_range=(1, 10), c_program_name="imagemorph2", verbose=True):
        self.displacement_range = displacement_range
        self.radius_range = radius_range
        self.c_program_name = c_program_name
        self.verbose = verbose

    def run(self, img_path):
        rand_displacement = np.random.uniform(*self.displacement_range)
        rand_radius = np.random.uniform(*self.radius_range)
        img_output_path, f_ext = tuple(img_path.split("."))
        img_output_path += "_aug." + f_ext
        os.system(f"cat {img_path} | ./{self.c_program_name} {rand_displacement} {rand_radius} > {img_output_path}")

    def run_batch(self, img_ls):
        if self.verbose:
            print("Starting augmentation batch of size:", len(img_ls), end=" ")
        with Pool() as pool:
            pool.map(self.run, img_ls)
        if self.verbose:
            print("- completed -")


if __name__ == "__main__":
    runner = ImageMorphRunner()
    runner.run_batch(["MPS1300_0001.ppm"])  # run test
