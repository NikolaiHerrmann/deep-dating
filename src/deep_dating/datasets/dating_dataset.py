
import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deep_dating import dating_util
from deep_dating.datasets import SetType
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod


class DatingDataset(ABC):

    def __init__(self, path, val_size=0.2, test_size=0.3, min_class_count=5, verbose=True):
        super().__init__()

        self.path = path
        self.test_size = test_size
        self.min_class_count = min_class_count
        self.verbose = verbose
        self.name = self.__class__.__name__
        self.save_path = os.path.join(dating_util.DATASETS_PATH, self.name + "_Set")
        self.csv_path = os.path.join(self.save_path, "split.csv")

        self._read_header_file()
        self.img_names = np.asarray(self._extract_img_names())
        self.img_dates = np.asarray(self._extract_img_dates(), dtype=np.float32)

        self._update_size()
        self._remove_low_count_samples()
        self._make_test_set()
        self.make_val_set(val_size)

    def _verify_header_matches_imgs_found(self, imgs_found, imgs_listed):
        if self.verbose:
            imgs_found_set, imgs_listed_set = set(imgs_found), set(imgs_listed)
            if imgs_found_set != imgs_listed_set or len(imgs_found) != len(imgs_listed):
                diff = imgs_found_set.symmetric_difference(imgs_listed_set)
                self._print_message(f"Warning! File list found in header does not entirely match images found in directory. {len(diff)} image inconsistencies found.\n\tDifferences found: {diff}")

    @abstractmethod
    def _read_header_file(self):
        pass

    @abstractmethod
    def _extract_img_names(self):
        pass

    @abstractmethod
    def _extract_img_dates(self):
        pass

    def _remove_low_count_samples(self):
        unique_dates, counts = np.unique(self.img_dates, return_counts=True)
        low_count_idxs = counts < self.min_class_count
        kick_out_dates = unique_dates[low_count_idxs]
        num_kick_out_dates = kick_out_dates.size

        if num_kick_out_dates > 0:
            idxs_to_remove = np.where(self.img_dates == kick_out_dates)
            self.img_dates = np.delete(self.img_dates, idxs_to_remove)
            self.img_names = np.delete(self.img_names, idxs_to_remove)
            self._update_size()
            if self.verbose:
                dict_ = dict(zip(kick_out_dates, counts[low_count_idxs]))
                self._print_message(f"Removed {num_kick_out_dates} sample(s). Hist: {dict_}")

    def _split_data(self, X, y, test_size_ratio):
        return train_test_split(X, y, test_size=test_size_ratio, shuffle=True, random_state=dating_util.SEED, stratify=y)

    def _make_test_set(self):
        self.X_train_org, self.X_test, self.y_train_org, self.y_test = self._split_data(self.img_names, self.img_dates, self.test_size)

    def make_val_set(self, val_size):
        self.X_train, self.X_val, self.y_train, self.y_val = self._split_data(self.X_train_org, self.y_train_org, val_size)

    def get_data(self, set_type):
        if set_type == SetType.TEST:
            return self.X_test, self.y_test
        elif set_type == SetType.VAL:
            return self.X_val, self.y_val
        elif set_type == SetType.TRAIN:
            return self.X_train, self.y_train
        else:
            raise Exception("Unknown dataset type!")
        
    def process_img(self, arg):
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
        
    def process_files(self, process_func, set_type):
        self.process_func = process_func
        os.makedirs(self.save_path, exist_ok=True)

        with Pool() as pool:
            processed_imgs = pool.map(self.process_img, zip(*self.get_data(set_type)))

        concat = []
        for x in processed_imgs:
            concat += x
        processed_imgs = np.array(concat)
        file_names, dates = processed_imgs[:, 0], processed_imgs[:, 1]

        df = pd.DataFrame({"name": file_names, "date": dates, "set": [set_type.value] * len(file_names)})
        df.to_csv(self.csv_path, mode="a", index=False, header=False)

    def read_split_header(self, set_type):
        df = pd.read_csv(self.csv_path, header=None)
        df.columns = ["name", "date", "set"]
        df = df[df["set"] == set_type.value]
        X = df["name"].to_numpy()
        y = df["date"].to_numpy().astype(np.float32)
        return X, y

    def _update_size(self):
        self.size = len(self.img_names)
        assert self.size == len(self.img_dates), "Image and date lists not of same length!"

    def _print_message(self, msg):
        print(f"{self.name}: {msg}")

    def __len__(self):
        return self.size

    def __str__(self):
        if self.verbose:
            self._print_message(f"Train: {len(self.X_train_org)}, Test: {len(self.X_test)}")
        return f"{self.name}: {self.size} images"


class MPS(DatingDataset):

    def __init__(self, path=os.path.join(dating_util.DATASETS_PATH, "MPS", "Download")):
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

    def __init__(self, path=os.path.join(dating_util.DATASETS_PATH, "ICDAR2017_CLaMM_Training")):
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
            self.class_range_mean[key] = int(np.mean(val))
        super().__init__(path)

    def _read_header_file(self):
        header_path = os.path.join(self.path, "@ICDAR2017_CLaMM_Training.csv")
        self.header_df = pd.read_csv(header_path, sep=";")

    def _extract_img_names(self):
        imgs_path = os.path.join(self.path, "*.tif")
        imgs_found = glob.glob(imgs_path)
        imgs_listed = self.header_df["FILENAME"].to_list()
        imgs_listed = [os.path.join(self.path, x) for x in imgs_listed]
        self._verify_header_matches_imgs_found(imgs_found, imgs_listed)
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
        path = os.path.join(dating_util.DATASETS_PATH, self.path_header, "nl.writers.scribes.v2.txt")
        self.header_df = pd.read_csv(path, delimiter=r"\s+", skiprows=10, header=None)
        self.header_df = self.header_df.dropna()
        self.header_df.columns = ["writer-name", "ID", "directory", "year",
                                  "slant", "curviness/complexity"]

    def _parse_owic_brieven(self, exp):
        # example (3-14)
        if exp[0] == "(": 
            ls = exp[1:-1].split("-") # remove start "(" and ending ")"
            ls = [int(x) for x in ls]
            ls = [str(x) for x in range(ls[0], ls[1] + 1)]
            return ls
        
        # example 21.[1368-13]
        ls = exp.split("[")
        assert len(ls) <= 2, "More brackets than expected!"
        if len(ls) == 2:
            start, addons = tuple(ls)
            addons = addons[:-1] # remove "]"
            
            addons_split = re.split("-|,", addons)
            addons = list(addons_split[0]) + addons_split[1:]
            ls = [start + end for end in addons]

        return ls
    
    def _parse_roggeveen(self, exp):
        min_, max_ = tuple(exp[1:-1].split("-")) # remove start "[" and ending "]"
        min_, max_ = float(min_), float(max_)
        
        new_ls = np.concatenate([np.arange(int(min_), max_ + 1), 
                                 np.arange(int(min_) + 0.1, max_ + 1)])
        new_ls = new_ls[((new_ls >= min_) & (new_ls <= max_))]
        new_ls = [str(x) for x in new_ls]

        return new_ls

    def _parse_dir(self, img_dir, writer_name):
        imgs_in_dir = []
        special_dirs = writer_name.split("/")

        if "owic.brieven" in writer_name:
            dir_nums = self._parse_owic_brieven(special_dirs[1])
        elif "roggeveen" in writer_name:
            dir_nums = self._parse_roggeveen(img_dir.split("/")[-1])
        else:
            img_dir_path = os.path.join(dating_util.DATASETS_PATH, self.path, img_dir, "**", "*.j*")
            ls = glob.glob(img_dir_path, recursive=True)
            return ls

        for num in dir_nums:
            path = "scribblelens.corpus.v1/nl/unsupervised/" + special_dirs[0] + "/" + num
            imgs_listed = glob.glob(os.path.join(dating_util.DATASETS_PATH, self.path, path, "*"))
            imgs_in_dir += imgs_listed
        
        return imgs_in_dir

    def _extract_img_names(self):
        imgs_listed = []
        self.date_ls = []

        # use *.j* as files are both .jpg and .jpeg
        imgs_found = glob.glob(os.path.join(dating_util.DATASETS_PATH, self.path, "**", "*.j*"), recursive=True)

        for img_dir, date, writer_name in zip(self.header_df["directory"], self.header_df["year"], self.header_df["writer-name"]):
            imgs_in_dir = self._parse_dir(img_dir, writer_name)
            imgs_listed += imgs_in_dir
            self.date_ls += [date] * len(imgs_in_dir)

        self._verify_header_matches_imgs_found(imgs_found, imgs_listed)

        return imgs_listed

    def _extract_img_dates(self):
        return self.date_ls


def load_all_dating_datasets():
    return [MPS(), CLaMM(), ScribbleLens()]