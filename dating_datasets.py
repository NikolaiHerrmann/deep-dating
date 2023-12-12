
import os
import glob
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


DATASETS_PATH = "../datasets"


class DatingDataset(ABC):

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.name = self.__class__.__name__
        self._read_header_file()
        self.img_names = self._extract_img_names()
        self.img_dates = np.asarray(self._extract_img_dates(), dtype=np.int64)
        self.size = len(self.img_names)
        assert self.size == len(self.img_dates), "Image and date lists not of same length!"

    def __len__(self):
        return self.size

    def __str__(self):
        return f"{self.name}: {self.size} images"

    @abstractmethod
    def _read_header_file(self):
        pass

    @abstractmethod
    def _extract_img_names(self):
        pass

    @abstractmethod
    def _extract_img_dates(self):
        pass


class MPS(DatingDataset):

    def __init__(self, path=os.path.join(DATASETS_PATH, "MPS", "Download")):
        super().__init__(path)

    def _read_header_file(self):
        imgs_path = os.path.join(self.path, "*", "*")
        self.header_ls = glob.glob(imgs_path)

    def _extract_img_names(self):
        return self.header_ls

    def _extract_img_dates(self):
        def extract(file_name):
            return int(os.path.basename(file_name).split("_")[0][3:])
        return [extract(x) for x in self.header_ls]


class CLaMM(DatingDataset):

    def __init__(self, path=os.path.join(DATASETS_PATH, "ICDAR2017_CLaMM_Training")):
        self.class_range = {1: (0, 1000),
                            2: (1001, 1100),
                            3: (1101, 1200),
                            4: (1201, 1250),
                            5: (1251, 1300),
                            6: (1301, 1350),
                            7: (1351, 1400),
                            8: (1401, 1425),
                            9: (1426, 1450),
                            10: (1451, 1475),
                            11: (1476, 1500),
                            12: (1501, 1525),
                            13: (1526, 1550),
                            14: (1551, 1575),
                            15: (1576, 1600)}
        self.class_range_mean = {}
        for key, val in self.class_range.items():
            self.class_range_mean[key] = np.mean(val)
        super().__init__(path)

    def _read_header_file(self):
        header_path = os.path.join(self.path, "@ICDAR2017_CLaMM_Training.csv")
        self.header_df = pd.read_csv(header_path, sep=";")

    def _extract_img_names(self):
        imgs_path = os.path.join(self.path, "*.tif")
        imgs_found = glob.glob(imgs_path)
        imgs_listed = set(self.header_df["FILENAME"].to_list())

        for img_name in imgs_found:
            assert not img_name in imgs_listed, "Unknown image found in directory, not listed in header file!"

        return imgs_found

    def _extract_img_dates(self):
        self.img_classes = self.header_df["DATE_TYPE"].to_list()
        return [self.class_range_mean[x] for x in self.img_classes]


class ScribbleLens(DatingDataset):

    def __init__(self, path=os.path.join("scribblelens.supplement.original.pages"),
                 path_header=os.path.join("scribblelens.corpus.v1.2", "scribblelens.corpus.v1", "corpora")):
        self.path_header = path_header
        super().__init__(path)

    def _read_header_file(self):
        path = os.path.join(DATASETS_PATH, self.path_header, "nl.writers.scribes.v2.txt")
        self.header_df = pd.read_csv(path, delimiter=r"\s+", skiprows=10)
        self.header_df = self.header_df.dropna()
        self.header_df.columns = ["writer-name", "ID", "directory", "year",
                                  "slant", "curviness/complexity"]

    def _extract_img_names(self):
        img_ls = []
        self.date_ls = []

        for img_dir, date in zip(self.header_df["directory"], self.header_df["year"]):
            if "unsupervised" in img_dir:  # fix this
                continue
            img_dir_path = os.path.join(DATASETS_PATH, self.path, img_dir, "*", "*")
            imgs_in_dir = glob.glob(img_dir_path)
            img_ls += imgs_in_dir
            self.date_ls += [date] * len(imgs_in_dir)

        return img_ls

    def _extract_img_dates(self):
        return self.date_ls


def load_all_dating_datasets():
    return [MPS(), CLaMM(), ScribbleLens()]


if __name__ == "__main__":
    for dataset in load_all_dating_datasets():
        print(dataset)
