
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deep_dating.util import DATASETS_PATH
from multiprocessing import Pool


class Preprocessor:

    def __init__(self, dataset_name):
        self.save_path = os.path.join(DATASETS_PATH, str(dataset_name) + "_Set")
        self.csv_header_path = os.path.join(self.save_path, "split.csv")

    def _preprocess_img(self, arg):
        img_path, img_date = arg
        imgs = self.process_func(img_path)
        old_img_name = os.path.basename(img_path).split(".")[0]
        names = []
        for i, img in enumerate(imgs):
            new_image_name = old_img_name + f"__{int(img_date)}_p{i}.ppm"
            file_name = os.path.join(self.save_path, new_image_name)
            plt.imsave(file_name, img, cmap="gray")
            names.append((file_name, img_date))
        return names
        
    def run(self, X, y, set_type, preprocessing_func):
        self.process_func = preprocessing_func
        os.makedirs(self.save_path, exist_ok=True)
        
        with Pool() as pool:
            processed_imgs = pool.map(self._preprocess_img, zip(X, y))

        concat = []
        for x in processed_imgs:
            concat += x
        processed_imgs = np.array(concat)
        file_names, dates = processed_imgs[:, 0], processed_imgs[:, 1]

        df = pd.DataFrame({"name": file_names, "date": dates, "set": [set_type.value] * len(file_names)})
        df.to_csv(self.csv_header_path, mode="a", index=False, header=False)

    def read_preprocessing_header(self, set_type):
        df = pd.read_csv(self.csv_header_path, header=None)
        df.columns = ["name", "date", "set"]
        df = df[df["set"] == set_type.value]
        X = df["name"].to_numpy()
        y = df["date"].to_numpy().astype(np.float32)
        return X, y