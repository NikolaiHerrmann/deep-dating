
import os
import re
import glob
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm
from deep_dating.datasets import DatasetName
from deep_dating.util import DATASETS_PATH
from abc import ABC, abstractmethod


class DatingDataset(ABC):

    def __init__(self, path, name, verbose=True):
        super().__init__()

        self.path = path
        self.name = name
        self.verbose = verbose
        self.idxs_to_remove = []

        self._read_header_file()
        self.X = np.asarray(self._extract_img_names())
        self.y = np.asarray(self._extract_img_dates(), dtype=np.float32)

        if self.verbose and self.idxs_to_remove:
            print("Removing:", self.X[self.idxs_to_remove])

        self.X = np.delete(self.X, self.idxs_to_remove)
        self.y = np.delete(self.y, self.idxs_to_remove)

        assert len(self.X) == len(self.y), "Images and dates not of equal length!"
        
        self.size = len(self.X)

    @abstractmethod
    def _read_header_file(self):
        pass

    @abstractmethod
    def _extract_img_names(self):
        pass

    @abstractmethod
    def _extract_img_dates(self):
        pass

    def _verify_header_matches_imgs_found(self, imgs_found, imgs_listed):
        imgs_found_set, imgs_listed_set = set(imgs_found), set(imgs_listed)

        if imgs_found_set != imgs_listed_set or len(imgs_found) != len(imgs_listed):

            diff = imgs_found_set.symmetric_difference(imgs_listed_set)
            print("Warning! File list found in header does not entirely",
                    f"match images found in directory. {len(diff)} image",
                    f"inconsistencies found.\n\tDifferences found: {diff}")
            
            for x in diff:
                if x in imgs_listed_set:
                    self.idxs_to_remove.append(imgs_listed.index(x))

    def save_to_dir(self, save_dir):
        unique_dates = np.unique(self.y)

        def save_date(date):
            date_dir_path = os.path.join(save_dir, str(date.astype(int)))
            os.mkdir(date_dir_path)

            idxs = np.where(self.y == date)[0]
            img_paths = self.X[idxs]

            for path in img_paths:
                img_name = os.path.basename(path)
                shutil.copy(path, os.path.join(date_dir_path, img_name))

        os.mkdir(save_dir)

        for date in tqdm(unique_dates):
            save_date(date)


class MPS(DatingDataset):

    def __init__(self, path=os.path.join(DATASETS_PATH, "MPS", "Download"), dir_depth=2):
        self.dir_depth = dir_depth
        super().__init__(path, DatasetName.MPS)

    def _read_header_file(self):
        imgs_path = self.path
        
        for _ in range(self.dir_depth):
            imgs_path = os.path.join(imgs_path, "*")

        # imgs_path = os.path.join(self.path, "*", "*")

        self.header_ls = glob.glob(imgs_path)

    def _extract_img_names(self):
        return self.header_ls

    def _extract_img_dates(self):
        def extract(file_name):
            return int(os.path.basename(file_name).split("_")[0][3:])
        return [extract(x) for x in self.header_ls]


class CLaMM(DatingDataset):

    def __init__(self, path=os.path.join(DATASETS_PATH, "ICDAR2017_CLaMM_Training"),
                 file_name_header="FILENAME", date_header="DATE_TYPE", 
                 script_header="SCRIPT_TYPE", img_exts=["tif", "jpg", "JPG"]):
        self.class_range = {1: (1000, 1000),
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
        self.file_name_header = file_name_header
        self.date_header = date_header
        self.script_header = script_header
        self.img_exts = img_exts
        super().__init__(path, DatasetName.CLAMM)

    def _read_header_file(self):
        csv_name = "@" + os.path.basename(self.path) + ".csv"
        header_path = os.path.join(self.path, csv_name)
        self.header_df = pd.read_csv(header_path, sep=";")

    def _extract_img_names(self):
        imgs_found = []
        for ext in self.img_exts:
            imgs_found += glob.glob(os.path.join(self.path, "*." + ext))

        imgs_listed = self.header_df[self.file_name_header].to_list()
        imgs_listed = [os.path.join(self.path, x) for x in imgs_listed]
        self._verify_header_matches_imgs_found(imgs_found, imgs_listed)
        return imgs_listed

    def _extract_img_dates(self):
        self.img_classes = self.header_df[self.date_header].to_list()

        self.script_labels = self.header_df[self.script_header].to_list()

        return [self.class_range_mean[x] for x in self.img_classes]


class CLaMM_Test_Task3(CLaMM):

    def __init__(self, path=os.path.join(DATASETS_PATH, "CLaMM_task1_task3_Clean"),
                 file_name_header="FILE_NAME", date_header="DATE_ICDAR", script_header="Script_type_ICDAR2017"):
        super().__init__(path, file_name_header, date_header, script_header)


class CLaMM_Test_Task4(CLaMM_Test_Task3):

    def __init__(self, path=os.path.join(DATASETS_PATH, "ICDAR2017_CLaMM_task2_task4")):
        super().__init__(path)


class ScribbleLens(DatingDataset):

    def __init__(self, path=os.path.join("scribblelens.supplement.original.pages"),
                 path_header=os.path.join("scribblelens.corpus.v1.2", "scribblelens.corpus.v1", "corpora")):
        self.path_header = path_header

        self.bins = {1: (1598, 1602),
                     2: (1611, 1631),
                     3: (1634, 1637),
                     4: (1641, 1642),
                     5: (1653, 1693),
                     6: (1721, 1726)}

        # start_bin_date=1595, bin_width=20
        # self.start_bin_date = start_bin_date
        # self.bin_width = bin_width
        self.class_range_mean = {}
        for key, val in self.bins.items():
            self.class_range_mean[key] = int(np.mean(val))
        self.writer_ids_per_date = {}
        super().__init__(path, DatasetName.SCRIBBLE)

    def _read_header_file(self):
        path = os.path.join(DATASETS_PATH, self.path_header, "nl.writers.scribes.v2.txt")
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
            img_dir_path = os.path.join(DATASETS_PATH, self.path, img_dir, "**", "*.j*")
            ls = glob.glob(img_dir_path, recursive=True)
            return ls

        for num in dir_nums:
            path = "scribblelens.corpus.v1/nl/unsupervised/" + special_dirs[0] + "/" + num
            imgs_listed = glob.glob(os.path.join(DATASETS_PATH, self.path, path, "*"))
            imgs_in_dir += imgs_listed
        
        return imgs_in_dir
    
    def _add_writer(self, date, writer_id):
        date = int(date)
        writer_id = int(writer_id)

        if not date in self.writer_ids_per_date:
            self.writer_ids_per_date[date] = set()
        self.writer_ids_per_date[date].add(writer_id)

    def _extract_img_names(self):
        imgs_listed = []
        self.date_ls = []

        # use *.j* as files are both .jpg and .jpeg
        imgs_found = glob.glob(os.path.join(DATASETS_PATH, self.path, "**", "*.j*"), recursive=True)

        for img_dir, date, writer_name, writer_id in zip(self.header_df["directory"], self.header_df["year"], 
                                                         self.header_df["writer-name"], self.header_df["ID"]):
            imgs_in_dir = self._parse_dir(img_dir, writer_name)
            imgs_listed += imgs_in_dir
            self.date_ls += [date] * len(imgs_in_dir)

            self._add_writer(date, writer_id)

        self._verify_header_matches_imgs_found(imgs_found, imgs_listed)

        return imgs_listed
    
    # def _calc_bins(self):
    #     self.bins = []
    #     self.bin_tokens = []
    #     start_date = self.start_bin_date
    #     max_date = np.max(self.date_ls)

    #     while start_date < max_date:
    #         end_date = start_date + self.bin_width
    #         self.bins.append((start_date, end_date))
    #         self.bin_tokens.append(np.mean([start_date, end_date]))
    #         start_date = end_date

    def _apply_bins(self):
        self.writer_ids_per_bin = {} #[date].add(writer_id)

        new_dates = np.zeros(self.date_ls.shape)
        date_np = np.array(self.date_ls)

        for token, (start_date, end_date) in self.bins.items():
            token = self.class_range_mean[token]

            condition = (date_np >= start_date) & (date_np <= end_date)
            idxs = np.where(condition)

            count = 0
            for date in np.unique(date_np[idxs]):
                count += len(self.writer_ids_per_date[date])
            self.writer_ids_per_bin[token] = count

            new_dates[idxs] = token

        return new_dates
    
    def _extract_img_dates(self):
        self.date_ls = np.array(self.date_ls)
        return self._apply_bins()


def load_all_dating_datasets():
    return [MPS(), CLaMM(), ScribbleLens()]