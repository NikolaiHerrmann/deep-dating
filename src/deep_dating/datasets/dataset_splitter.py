
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from deep_dating.util import SEED, DATASETS_PATH
from deep_dating.datasets import SetType
from deep_dating.augmentation import ImageMorphRunner
from sklearn.model_selection import train_test_split


class DatasetSplitter:

    def __init__(self, dataset, min_count=None, max_count=None, min_removal_count=5, 
                 test_size=0.3, val_size=0.2, test_run=False, verbose=True,
                 read_aug=None, binary=False):
        self.X = dataset.X
        self.y = dataset.y
        extra_path = "_Bin" if binary else ""
        self.aug_path = os.path.join(DATASETS_PATH, str(dataset.name) + "_Aug" + extra_path)
        self.aug_csv_path = os.path.join(self.aug_path, "aug.csv")
        self.min_count = min_count
        self.max_count = max_count
        self.min_removal_count = min_removal_count
        self.test_size = test_size
        self.val_size = val_size
        self.test_run = test_run
        self.read_aug = read_aug
        self.verbose = verbose

        self._remove_low_count_samples()
        self._make_split()

    def get_data(self, set_type):
        if set_type == SetType.TEST and self.test_size > 0:
            return self.X_test, self.y_test
        elif set_type == SetType.VAL and self.val_size > 0:
            return self.X_val, self.y_val
        elif set_type == SetType.TRAIN:
            return self.X_train, self.y_train
        
        if self.verbose:
            print(f"Dataset type: {set_type} not implemented.")

        return None, None
        
    def get_summary(self):
        return f"Train: {len(self.X_train)} \n Val: {len(self.X_val)} \n Test: {len(self.X_test)}"

    def _remove_low_count_samples(self):
        unique_dates, counts = np.unique(self.y, return_counts=True)
        low_count_idxs = counts < self.min_removal_count
        kick_out_dates = unique_dates[low_count_idxs]
        num_kick_out_dates = kick_out_dates.size

        if num_kick_out_dates > 0:
            for kick in kick_out_dates:
                idxs_to_remove = np.where(self.y == kick)
                self.y = np.delete(self.y, idxs_to_remove)
                self.X = np.delete(self.X, idxs_to_remove)
                if self.verbose:
                    dict_ = dict(zip(kick_out_dates, counts[low_count_idxs]))
                    print(f"Removed {num_kick_out_dates} sample(s). Hist: {dict_}")

    def _balance_data(self, X, y):
        unique_labels, counts = np.unique(y, return_counts=True)
        X_new = []
        y_new = []
        total_aug = 0
        total_img = 0
        augment_runner = ImageMorphRunner(test_run=self.test_run, verbose=self.verbose)

        all_X_aug = []
        all_y_aug = []

        for label, count in tqdm(zip(unique_labels, counts), disable = not self.verbose, total=len(counts)):
            label_idxs = np.where(y == label)[0]
            
            # undersample
            if self.max_count and (count > self.max_count):
                label_idxs = np.random.choice(label_idxs, size=self.max_count, replace=False)
                assert label_idxs.shape[0] == self.max_count

            X_keep = X[label_idxs]
            y_keep = y[label_idxs]
            total_img += label_idxs.shape[0]

            # oversample with augmentation
            if self.min_count and (count < self.min_count):
                n_missing = self.min_count - count
                
                # repeat each sample n times
                if n_missing > count:
                    aug_each_factor = int(n_missing / count)
                    aug_idxs = np.repeat(label_idxs, aug_each_factor)
                    n_missing -= aug_each_factor * count
                else:
                    aug_idxs = np.array([], dtype=label_idxs.dtype)

                # randomly choice missing samples
                if n_missing > 0:
                    aug_idxs_extra = np.random.choice(label_idxs, size=n_missing, replace=False)
                    aug_idxs = np.concatenate([aug_idxs, aug_idxs_extra])

                assert aug_idxs.shape[0] + X_keep.shape[0] == self.min_count
                total_aug += aug_idxs.shape[0]
                
                X_aug = augment_runner.run_batch(X[aug_idxs], self.aug_path)
                y_aug = np.full(shape=aug_idxs.shape, fill_value=label)
                assert y_aug.shape[0] + y_keep.shape[0] == self.min_count

                all_X_aug += X_aug.tolist()
                all_y_aug += y_aug.tolist()

                X_keep = np.concatenate([X_keep, X_aug])
                y_keep = np.concatenate([y_keep, y_aug])

            X_new.append(X_keep)
            y_new.append(y_keep)

        X_new = np.concatenate(X_new)
        y_new = np.concatenate(y_new)

        _, counts_new = np.unique(y_new, return_counts=True)

        if total_aug > 0:
            self._write_aug_csv(all_X_aug, all_y_aug)

        if self.verbose:
            print("Counts compare:", counts, counts_new)
            print(total_aug, "new images were made through augmentation. Non-aug =", total_img)

        return X_new, y_new

    def _make_split(self):
        if self.test_size > 0:
            (self.X_train_org, self.X_test,
            self.y_train_org, self.y_test) = self._split_data(self.X, self.y, self.test_size)
        else:
            if self.verbose:
                print("Making no test split!")
            self.X_train_org, self.y_train_org = self.X, self.y

        (self.X_train, self.X_val,
         self.y_train, self.y_val) = self._split_data(self.X_train_org, self.y_train_org, self.val_size)
        
        if self.read_aug:
            self._read_aug_csv()
        
        if self.min_count or self.max_count:
            os.makedirs(self.aug_path, exist_ok=True)
            self.X_train, self.y_train = self._balance_data(self.X_train, self.y_train)
        
    def _split_data(self, X, y, test_size_ratio):
        return train_test_split(X, y, test_size=test_size_ratio, shuffle=True,
                                random_state=SEED, stratify=y)
    
    def _write_aug_csv(self, all_X_aug, all_y_aug):
        df = pd.DataFrame({"name": all_X_aug, "date": all_y_aug})
        df.to_csv(self.aug_csv_path, index=False, header=False)

    def _read_aug_csv(self):
        df = pd.read_csv(self.aug_csv_path, header=None)
        df.columns = ["name", "date"]
        X = df["name"].to_numpy()
        y = df["date"].to_numpy().astype(np.float32)

        shape_before = self.X_train.shape

        self.X_train = np.concatenate([self.X_train, X])
        self.y_train = np.concatenate([self.y_train, y])

        shape_after = self.X_train.shape
        diff = shape_after[0] - shape_before[0]

        if self.verbose:
            print(f"Read aug data! From shape: {shape_before} to {shape_after}. Added {diff} images.")
