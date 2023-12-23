
import numpy as np
from deep_dating.util import dating_util
from deep_dating.datasets import SetType
from sklearn.model_selection import train_test_split


class DatasetSplitter:

    def __init__(self, dataset, min_class_count=5, test_size=0.3, val_size=0.2, verbose=True):
        self.X = dataset.X
        self.y = dataset.y
        self.min_class_count = min_class_count
        self.test_size = test_size
        self.val_size = val_size
        self.verbose = verbose

        self._remove_low_count_samples()
        self._make_split()

    def get_data(self, set_type):
        if set_type == SetType.TEST:
            return self.X_test, self.y_test
        elif set_type == SetType.VAL:
            return self.X_val, self.y_val
        elif set_type == SetType.TRAIN:
            return self.X_train, self.y_train
        else:
            raise Exception("Unknown dataset type!")

    def _remove_low_count_samples(self):
        unique_dates, counts = np.unique(self.y, return_counts=True)
        low_count_idxs = counts < self.min_class_count
        kick_out_dates = unique_dates[low_count_idxs]
        num_kick_out_dates = kick_out_dates.size

        if num_kick_out_dates > 0:
            idxs_to_remove = np.where(self.y == kick_out_dates)
            self.y = np.delete(self.y, idxs_to_remove)
            self.X = np.delete(self.X, idxs_to_remove)
            if self.verbose:
                dict_ = dict(zip(kick_out_dates, counts[low_count_idxs]))
                print(f"Removed {num_kick_out_dates} sample(s). Hist: {dict_}")

    def _make_split(self):
        (self.X_train_org, self.X_test,
         self.y_train_org, self.y_test) = self._split_data(self.X, self.y, self.test_size)
        
        # augment data

        (self.X_train, self.X_val,
         self.y_train, self.y_val) = self._split_data(self.X_train_org, self.y_train_org, self.val_size)
        
    def _split_data(self, X, y, test_size_ratio):
        return train_test_split(X, y, test_size=test_size_ratio, shuffle=True,
                                random_state=dating_util.SEED, stratify=y)
