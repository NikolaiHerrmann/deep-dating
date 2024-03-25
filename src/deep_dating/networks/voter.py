
import numpy as np
from deep_dating.util import mode


class Voter:
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.agg_funcs = [("Median", np.median), 
                          ("Mode", mode),
                          ("Mean", np.mean)]
        self.reset()

    def reset(self):
        self.pred_labels = []
        self.true_labels = []
        self.pred_labels_dict = []

    def add_prediction(self, labels):
        self.pred_labels.append(labels)

    def add_prediction_dict(self, dict_):
        self.pred_labels_dict.append(dict_)

    def set_labels(self, labels):
        if len(self.true_labels) > 0:
            assert np.array_equal(self.true_labels, labels), "voter found non-matching labels"
        else:
            self.true_labels = labels

    def get_labels(self):
        return self.true_labels

    def predict(self):
        if not self.pred_labels:
            if self.verbose:
                print("No labels were added to voter yet")
            return []
        return [(func_name, np.round(func(self.pred_labels, axis=0))) for func_name, func in self.agg_funcs]
    
    def predict_dict(self):
        if not self.pred_labels_dict:
            if self.verbose:
                print("No dict labels were added to voter yet")
            return []

        large_dict = {}

        for dict_ in self.pred_labels_dict:
            for key in dict_.keys():

                if not key in large_dict:
                    large_dict[key] = {"label": dict_[key]["label"], "feat_or_preds": [dict_[key]["feat_or_preds"]]}
                else:
                    assert large_dict[key]["label"] == dict_[key]["label"]
                    large_dict[key]["feat_or_preds"].append(dict_[key]["feat_or_preds"])

        preds_per_func = []
        for func_name, func in self.agg_funcs:
            labels = []
            preds = []
            for key in large_dict.keys():
                labels.append(large_dict[key]["label"])
                x = np.array(large_dict[key]["feat_or_preds"]).flatten()
                pred = np.round(func(x, axis=0))
                preds.append(pred)
            preds_per_func.append((func_name, preds))

        self.true_labels = labels

        return preds_per_func