
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from deep_dating.preprocessing import PreprocessRunner
from deep_dating.datasets import SetType
from deep_dating.util import SEED


class CrossVal:

    def __init__(self, dataset_name, preprocess_ext="_Set", verbose=True):
        self.verbose = verbose
        self.preprocess_runner = PreprocessRunner(dataset_name, preprocess_ext)
        self._read_data()

    def _read_image_level_data(self, patch_names, patch_labels):
        self.img_level_patches = {}
        self.img_level_labels = {}

        for patch_name, patch_label in zip(patch_names, patch_labels):
            img_name = PreprocessRunner.get_base_img_name(patch_name)

            if img_name not in self.img_level_patches:
                self.img_level_patches[img_name] = [patch_name]
                self.img_level_labels[img_name] = patch_label
            else:
                assert self.img_level_labels[img_name] == patch_label, "wrong label for a patch"
                self.img_level_patches[img_name].append(patch_name)

        assert list(self.img_level_labels.keys()) == list(self.img_level_patches.keys())

        if self.verbose:
            print(f"Found {len(self.img_level_patches.keys())} images from all patches.")

    def _read_data(self):
        self.X_train, self.y_train = self.preprocess_runner.read_preprocessing_header(SetType.TRAIN)
        self.X_val, self.y_val = self.preprocess_runner.read_preprocessing_header(SetType.VAL)

        X = np.concatenate([self.X_train, self.X_val])
        y = np.concatenate([self.y_train, self.y_val])

        self._read_image_level_data(X, y)

    def _merge_patches(self, idxs):
        patch_imgs = []
        patch_labels = []

        for idx in idxs:
            img_name = self.X[idx]

            patches = self.img_level_patches[img_name]
            patch_imgs += patches

            assert self.img_level_labels[img_name] == self.y[idx]
            patch_labels += [self.img_level_labels[img_name]] * len(patches)

        return np.array(patch_imgs), np.array(patch_labels)

    def get_split(self, n_splits):
        if n_splits > 1:
            self.skf = StratifiedKFold(n_splits=n_splits, shuffle=False)

            self.X = list(self.img_level_labels.keys())
            self.y = list(self.img_level_labels.values())

            self.X, self.y = shuffle(self.X, self.y, random_state=SEED)

            for train_idxs, val_idxs in self.skf.split(self.X, self.y):
                
                X_train, y_train = self._merge_patches(train_idxs)
                X_val, y_val = self._merge_patches(val_idxs)

                yield X_train, y_train, X_val, y_val

        elif n_splits == 1:
            for x in [(self.X_train, self.y_train, self.X_val, self.y_val)]:
                yield x
        else:
            print("split number cannot be less than 1.")

