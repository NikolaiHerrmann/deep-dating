
import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
from deep_dating.prediction import DatingPredictor
from deep_dating.metrics import DatingMetrics
from deep_dating.preprocessing import PreprocessRunner
from deep_dating.util import SEED
from tqdm import tqdm


class DatingClassifier:

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.network_predictor = DatingPredictor(verbose=self.verbose)
        self.metrics = DatingMetrics(alphas=[0, 25, 50], classification=True, average="macro")
        self.feature_range = (-1, 1)

    def _merge_patches(self, labels, feats, img_names, test_dict=None, read_dict=None):
        preds = {}

        labels = labels.flatten()

        for i, img_name in enumerate(img_names):
            img_name = PreprocessRunner.get_base_img_name(img_name)

            if test_dict:
                assert img_name not in test_dict, "data is leaking!"

            if not img_name in preds:
                preds[img_name] = {"label": labels[i], "feat": [feats[i]]}
            else:
                assert preds[img_name]["label"] == labels[i] 
                preds[img_name]["feat"].append(feats[i])

        feat_arr = []
        label_arr = []

        if read_dict is None:
            read_dict = preds

        for key in read_dict.keys():
            aggregated_feat = np.mean(preds[key]["feat"], axis=0)
            feat_arr.append(aggregated_feat)
            label_arr.append(preds[key]["label"])

        return np.array(label_arr), np.array(feat_arr), preds
    
    def _get_train_val_split(self, dir_, split_num, train_read_dict=None, val_read_dict=None):
        feats_path_train = glob.glob(os.path.join(dir_, f"model*split_{split_num}*train*.pkl"))[0]
        feats_path_val = glob.glob(os.path.join(dir_, f"model*split_{split_num}*val*.pkl"))[0]
        
        labels_train_patch, features_train_patch, img_names_train = self.network_predictor.load(feats_path_train)
        labels_train_img, features_train_img, train_dict = self._merge_patches(labels_train_patch, features_train_patch, img_names_train, read_dict=train_read_dict)

        labels_val_patch, features_val_patch, img_names_val = self.network_predictor.load(feats_path_val)
        labels_val_img, features_val_img, val_dict = self._merge_patches(labels_val_patch, features_val_patch, img_names_val, test_dict=train_dict, read_dict=val_read_dict)

        return [labels_train_img, features_train_img, labels_val_img, features_val_img], train_dict, val_dict
    
    def cross_val(self, dir_1, dir_2=None, n_splits=5):
        metric_data = []

        for i in tqdm(range(n_splits)):
            
            split_data_1, train_dict, val_dict = self._get_train_val_split(dir_1, i)
            if dir_2 is not None:
                split_data_2, _, _ = self._get_train_val_split(dir_2, i, train_dict, val_dict)
            else:
                split_data_2 = None

            #save_model = os.path.join(dir_, f"classifier_model_split_{i}.pkl") if i == 0 else None

            metrics_nums = self.train(split_data_1, split_data_2, save_path=None)
            metric_data.append(metrics_nums)

        df = pd.DataFrame(metric_data)
        df.columns = self.metrics.names

        mean_metrics = df.mean(axis=0).round(2)
        std_metrics = df.std(axis=0).round(2)

        # Make latex table string
        str_ = ""
        for i in range(len(mean_metrics)):
            str_ += f"${mean_metrics[i]} \pm {std_metrics[i]}$ & "
        str_ = str_[:len(str_)-3]

        # with open(os.path.join(dir_, "classifier_results.txt"), "w") as f:
        #     f.write(str_)

        if self.verbose:
            print(df)
            print(mean_metrics.to_frame().T)
            print(str_)

        return mean_metrics, std_metrics

    def train(self, split_data_1, split_data_2=None, save_path=None):
        scaler_ls = []

        labels_train_img, features_train_img, labels_val_img, features_val_img = split_data_1

        scaler1 = MinMaxScaler(feature_range=self.feature_range)
        scaler_ls.append(scaler1)
        features_train_img_scaled = scaler1.fit_transform(features_train_img)
        features_val_img_scaled = scaler1.transform(features_val_img)

        # select = SelectKBest(f_classif, k=400)
        # features_train_img_scaled = select.fit_transform(features_train_img_scaled, labels_train_img)
        # features_val_img_scaled = select.transform(features_val_img_scaled)

        if split_data_2 is not None:
            labels_train_img_2, features_train_img_2, labels_val_img_2, features_val_img_2 = split_data_2

            assert np.array_equal(labels_train_img, labels_train_img_2), "train labels do not match between pipelines"
            assert np.array_equal(labels_val_img, labels_val_img_2), "val labels do not match between pipelines"

            scaler2 = MinMaxScaler(feature_range=self.feature_range)
            scaler_ls.append(scaler2)
            features_train_img_scaled_2 = scaler2.fit_transform(features_train_img_2)
            features_val_img_scaled_2 = scaler2.transform(features_val_img_2)

            # select = SelectKBest(f_classif, k=400)
            # features_train_img_scaled_2 = select.fit_transform(features_train_img_scaled_2, labels_train_img)
            # features_val_img_scaled_2 = select.transform(features_val_img_scaled_2)

            features_train_img_scaled = np.hstack([features_train_img_scaled, features_train_img_scaled_2])
            features_val_img_scaled = np.hstack([features_val_img_scaled, features_val_img_scaled_2])

        model = SVC(kernel="rbf", C=1, gamma="scale", random_state=SEED)

        model.fit(features_train_img_scaled, labels_train_img)
        labels_val_predict_img = model.predict(features_val_img_scaled)

        metrics_nums = self.metrics.calc(labels_val_img, labels_val_predict_img)

        #self.conf_matrix(labels_val_img, labels_val_predict_img)

        if save_path is not None:
            with open(save_path, "wb") as f:
                pickle.dump((scaler_ls, model), f)
            self.load(save_path)

        return metrics_nums
    
    def conf_matrix(self, model, labels_true, labels_predicted, show=True):
        cm = confusion_matrix(labels_true, labels_predicted, labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot()

        if show:
            plt.show()
        
    def load(self, classifier_path):
        with open(classifier_path, "rb") as f:
            scaler_ls, model = pickle.load(f)

        if self.verbose:
            print("Model and scaler loading completed!")

        return scaler_ls, model