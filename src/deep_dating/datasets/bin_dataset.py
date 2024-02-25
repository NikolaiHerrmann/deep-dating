
import os
import glob
import numpy as np
from deep_dating.datasets import DatasetName
from deep_dating.util import DATASETS_PATH


class BinDataset:

    def __init__(self, path=os.path.join(DATASETS_PATH, "dibco"),
                 train_years=[2009, 2010, 2011, 2012, 2013, 2014, 2017, 2018, 2019],
                 test_years=[2016], augment=True, aug_path=DATASETS_PATH):
        self.name = DatasetName.DIBCO
        self.path = path
        self.aug_path = aug_path
        
        self.X_train, self.y_train = self._merge_years(train_years, add_augment=augment)
        self.X_test, self.y_test = self._merge_years(test_years, add_augment=False)

    def _merge_years(self, years, add_augment):
        X = []
        y = []

        for year in years:
            imgs_year, gts_year = self._get_year(year, self.path)
            X += imgs_year
            y += gts_year

        if add_augment:
            imgs_year, gts_year = self._get_year("aug", self.aug_path)
            X += imgs_year
            y += gts_year

        assert len(X) == len(y)

        return np.array(X), np.array(y)
    
    def _get_year(self, year, read_path):
        imgs = glob.glob(os.path.join(read_path, str(year) + "_img_*", "*.*"))
        gts = glob.glob(os.path.join(read_path, str(year) + "_gt_*", "*.*"))

        gts_ordered = []

        img_names = [os.path.basename(x.lower()).rsplit(".", 1)[0] for x in imgs]
        gt_names = [os.path.basename(x.lower()).rsplit("_", 1)[0] for x in gts]

        for x in img_names:
            idx = gt_names.index(x)
            gts_ordered.append(gts[idx])

        return imgs, gts_ordered
        