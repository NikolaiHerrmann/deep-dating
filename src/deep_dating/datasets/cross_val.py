
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from deep_dating.preprocessing import PreprocessRunner
from deep_dating.datasets import SetType
from deep_dating.util import SEED


class CrossVal:

    def __init__(self, dataset_name, preprocess_ext="_Set"):
        self.preprocess_runner = PreprocessRunner(dataset_name, preprocess_ext)
        self._read_data()

    def _read_data(self):
        self.X_train, self.y_train = self.preprocess_runner.read_preprocessing_header(SetType.TRAIN)
        self.X_val, self.y_val = self.preprocess_runner.read_preprocessing_header(SetType.VAL)
        print("val size:", self.y_val.shape, self.y_train.shape)

        self.X = np.concatenate([self.X_train, self.X_val])
        self.y = np.concatenate([self.y_train, self.y_val])

        self.X, self.y = shuffle(self.X, self.y, random_state=SEED)

        assert self.X.shape == self.y.shape

    def get_split(self, n_splits):
        if n_splits > 1:
            self.skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

            for train_idxs, val_idx in self.skf.split(self.X, self.y):
                print("val k fold size:", self.y[val_idx].shape, self.y[train_idxs].shape)
                yield self.X[train_idxs], self.y[train_idxs], self.X[val_idx], self.y[val_idx]

        elif n_splits == 1:
            for x in [(self.X_train, self.y_train, self.X_val, self.y_val)]:
                yield x
        else:
            print("split number cannot be less than 1.")

