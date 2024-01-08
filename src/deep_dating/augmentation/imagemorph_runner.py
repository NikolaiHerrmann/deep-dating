
import os
import numpy as np
from multiprocessing import Pool
from deep_dating.augmentation import ImageMorph


class ImageMorphRunner:

    def __init__(self, displacement_range=(1, 3), radius_range=(1, 10), c_program_name="imagemorph2", verbose=True):
        self.displacement_range = displacement_range
        self.radius_range = radius_range
        self.c_program_name = c_program_name
        self.verbose = verbose

    def run(self, arg):
        img_path, id_num = arg
        rand_displacement = np.random.uniform(*self.displacement_range)
        rand_radius = np.random.uniform(*self.radius_range)

        img_name = os.path.basename(img_path).split(".")[0] + f"_aug_{id_num}.ppm"
        output_img_path = os.path.join(self.output_dir, img_name)

        ImageMorph().apply_rubber_sheet(img_path, rand_displacement, rand_radius, output_img_path)

        return output_img_path

    def run_batch(self, img_path_ls, output_dir):
        self.output_dir = output_dir
        if self.verbose:
            print("Starting augmentation batch of size:", len(img_path_ls), end=" ")
        ids = np.arange(img_path_ls.shape[0])
        with Pool() as pool:
            img_aug_path_ls = pool.map(self.run, zip(img_path_ls, ids))
        if self.verbose:
            print("- completed -")
        return np.array(img_aug_path_ls)


if __name__ == "__main__":
    runner = ImageMorphRunner()
    new_imgs = runner.run_batch(["MPS1300_0001.ppm", "MPS1300_0001.ppm"])  # run test
    print(new_imgs)
