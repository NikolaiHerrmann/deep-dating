
import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras import layers
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
from deep_dating.prediction import DatingPredictor
from deep_dating.metrics import DatingMetrics
from deep_dating.preprocessing import PreprocessRunner
from deep_dating.util import SEED
from deep_dating.networks import Voter
from tqdm import tqdm


class DatingClassifier:

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.network_predictor = DatingPredictor(verbose=self.verbose)
        self.metrics = DatingMetrics(alphas=[0, 25, 50], classification=True, average="macro")
        self.voter = Voter()
        self.feature_range = (0, 1)

    def _to_latex(self, metrics, std_metrics=None, dec=2):
        """
        Make latex table string
        """

        metrics = np.round(metrics, dec)
        if std_metrics is not None:
            std_metrics = np.round(std_metrics, dec)
        
        str_ = ""
        for i in range(len(metrics)):
            std_str = f" \pm {std_metrics[i]}" if std_metrics is not None else ""
            str_ += f"${metrics[i]}{std_str}$ & "
        str_ = str_[:len(str_)-3]

        return str_

        # with open(os.path.join(dir_, "classifier_results.txt"), "w") as f:
        #     f.write(str_)

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
    
    def _get_test_split(self, dir_, split_num, read_dict=None, task=""):
        feats_path_test = glob.glob(os.path.join(dir_, f"model*split_{split_num}*test*{task}.pkl"))[0]
        
        labels_test_patch, features_test_patch, img_names_test = self.network_predictor.load(feats_path_test)
        labels_test_img, features_test_img, test_dict = self._merge_patches(labels_test_patch, features_test_patch, img_names_test, read_dict=read_dict)

        return [labels_test_img, features_test_img], test_dict

    def cross_val(self, dir_1, dir_2=None, n_splits=5, train=True, task=""):
        metric_data = []

        for i in tqdm(range(n_splits)):

            is_concat = "_concat" if dir_2 is not None else ""
            model_path = os.path.join(dir_1, f"classifier_model_split_{i}{is_concat}.pkl")

            if train:
                split_data_1, train_dict, val_dict = self._get_train_val_split(dir_1, i)
                split_data_2 = self._get_train_val_split(dir_2, i, train_dict, val_dict)[0] if dir_2 is not None else None

                metrics_nums = self.train(split_data_1, split_data_2, save_path=model_path)
            else:
                split_data_1, _dict = self._get_test_split(dir_1, i, task=task)
                split_data_2 = self._get_test_split(dir_2, i, _dict, task=task)[0] if dir_2 is not None else None

                metrics_nums = self.predict(model_path, split_data_1, split_data_2)

            metric_data.append(metrics_nums)

        df = pd.DataFrame(metric_data)
        df.columns = self.metrics.names

        mean_metrics = df.mean(axis=0)
        std_metrics = df.std(axis=0)

        if self.verbose:
            print(df)
            print(mean_metrics.to_frame().T)
            print(self._to_latex(mean_metrics, std_metrics))

        if not train:
            for agg_name, pred_labels in self.voter.predict():
                metrics_nums = self.metrics.calc(self.voter.get_labels(), pred_labels)
                print(agg_name, self._to_latex(metrics_nums))
            
        self.voter.reset()

        return mean_metrics, std_metrics

    def nn_model(self, features_train_img, labels_train_img, features_val_img, labels_val_img):
        label_encoder = LabelEncoder()

        y_train = label_encoder.fit_transform(labels_train_img)
        y_train = to_categorical(y_train)
        y_val = label_encoder.transform(labels_val_img)
        y_val = to_categorical(y_val)

        input_size = features_train_img.shape[1]
        n_classes = len(label_encoder.classes_)

        model = keras.Sequential([
            keras.Input(shape=(input_size)),
            layers.Dense(input_size, activation="relu"),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(n_classes, activation="softmax"),
        ])

        batch_size = 64
        epochs = 50

        early_stopping = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.00001), metrics=["accuracy"])
        model.fit(features_train_img, y_train, batch_size=batch_size, epochs=epochs, validation_data=(features_val_img, y_val), callbacks=[early_stopping])

        predictions = model.predict(features_val_img)
        predictions = predictions.argmax(axis=-1)
        predictions = label_encoder.inverse_transform(predictions)

        return predictions, model, label_encoder
    
    def predict(self, model_path, split_data_1, split_data_2=None):
        scaler_ls, model = self.load(model_path)

        labels_test_img, features_test_img = split_data_1

        features_test_img_scaled = scaler_ls[0].transform(features_test_img)

        if split_data_2 is not None:
            labels_test_img_2, features_test_img_2 = split_data_2
            assert np.array_equal(labels_test_img, labels_test_img_2), "test labels do not match between pipelines"

            features_test_img_scaled_2 = scaler_ls[1].transform(features_test_img_2)

            features_test_img_scaled = np.hstack([features_test_img_scaled, features_test_img_scaled_2])

        labels_test_predict_img = model.predict(features_test_img_scaled)
        metrics_nums = self.metrics.calc(labels_test_img, labels_test_predict_img)

        self.voter.set_labels(labels_test_img)
        self.voter.add_prediction(labels_test_predict_img)

        return metrics_nums

    def train(self, split_data_1, split_data_2=None, save_path=None):
        scaler_ls = []

        labels_train_img, features_train_img, labels_val_img, features_val_img = split_data_1

        scaler1 = MinMaxScaler(feature_range=self.feature_range)
        scaler_ls.append(scaler1)
        features_train_img_scaled = scaler1.fit_transform(features_train_img)
        features_val_img_scaled = scaler1.transform(features_val_img)

        if split_data_2 is not None:
            labels_train_img_2, features_train_img_2, labels_val_img_2, features_val_img_2 = split_data_2

            assert np.array_equal(labels_train_img, labels_train_img_2), "train labels do not match between pipelines"
            assert np.array_equal(labels_val_img, labels_val_img_2), "val labels do not match between pipelines"

            scaler2 = MinMaxScaler(feature_range=self.feature_range)
            scaler_ls.append(scaler2)
            features_train_img_scaled_2 = scaler2.fit_transform(features_train_img_2)
            features_val_img_scaled_2 = scaler2.transform(features_val_img_2)

            features_train_img_scaled = np.hstack([features_train_img_scaled, features_train_img_scaled_2])
            features_val_img_scaled = np.hstack([features_val_img_scaled, features_val_img_scaled_2])

        model = SVC(kernel="rbf", C=1, gamma="scale", random_state=SEED)

        model.fit(features_train_img_scaled, labels_train_img)
        labels_val_predict_img = model.predict(features_val_img_scaled)
            
        #labels_val_predict_img, _, _ = self.nn_model(features_train_img_scaled, labels_train_img, features_val_img_scaled, labels_val_img)

        metrics_nums = self.metrics.calc(labels_val_img, labels_val_predict_img)

        #self.conf_matrix(labels_val_img, labels_val_predict_img)

        if save_path is not None:
            with open(save_path, "wb") as f:
                pickle.dump((scaler_ls, model), f)

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