
import os
import glob
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from deep_dating.prediction import DatingPredictor
from deep_dating.metrics import DatingMetrics
from deep_dating.preprocessing import PreprocessRunner
from deep_dating.util import SEED


class DatingClassifier:

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.network_predictor = DatingPredictor(verbose=self.verbose)
        self.metrics = DatingMetrics(alphas=[0, 25, 50])

    def _merge_patches(self, labels, feats, img_names):
        preds = {}

        labels = labels.flatten()

        for i, img_name in enumerate(img_names):
            img_name = PreprocessRunner.get_base_img_name(img_name)

            if not img_name in preds:
                preds[img_name] = {"label": labels[i], "feat": [feats[i]]}
            else:
                assert preds[img_name]["label"] == labels[i] 
                preds[img_name]["feat"].append(feats[i])

        feat_arr = []
        label_arr = []

        for key in preds.keys():
            aggregated_feat = np.mean(preds[key]["feat"], axis=0)
            feat_arr.append(aggregated_feat)
            label_arr.append(preds[key]["label"])

        return np.array(label_arr), np.array(feat_arr), preds
    
    def cross_val(self, dir_, n_splits=5):
        for i in range(n_splits):
            feats_path_train = glob.glob(os.path.join(dir_, f"model*split_{i}*train*.pkl"))[0]
            feats_path_val = glob.glob(os.path.join(dir_, f"model*split_{i}*val*.pkl"))[0]

            self.train(feats_path_train, feats_path_val)

    def train(self, feats_path_train, feats_path_val, save=False):
        labels_train_patch, features_train_patch, img_names_train = self.network_predictor.load(feats_path_train)
        labels_train_img, features_train_img, _ = self._merge_patches(labels_train_patch, features_train_patch, img_names_train)

        labels_val_patch, features_val_patch, img_names_val = self.network_predictor.load(feats_path_val)
        labels_val_img, features_val_img, _ = self._merge_patches(labels_val_patch, features_val_patch, img_names_val)

        scaler = MinMaxScaler(feature_range=(-1, 1))
        features_train_img_scaled = scaler.fit_transform(features_train_img)
        features_val_img_scaled = scaler.transform(features_val_img)

        model = SVC(kernel="rbf", C=1, gamma="scale", random_state=SEED)

        model.fit(features_train_img_scaled, labels_train_img)
        labels_val_predict_img = model.predict(features_val_img_scaled)

        vals = self.metrics.calc(labels_val_img, labels_val_predict_img)

        mae, mse = tuple(vals[:2])
        cs_ = vals[2:]

        print(mae, mse)
        print([0, 25, 50])
        print(cs_)
        print("-----")

        if save:
            with open("model.pkl", "wb") as f:
                pickle.dump((scaler, model) , f)
        