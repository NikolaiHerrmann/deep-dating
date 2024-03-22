
import numpy as np
from deep_dating.util import mode


class Voter:
    
    def __init__(self):
        self.agg_funcs = [("Median", np.median), 
                          ("Mode", mode),
                          ("Mean", np.mean)]
        self.reset()

    def reset(self):
        self.pred_labels = []
        self.true_labels = []

    def add_prediction(self, labels):
        self.pred_labels.append(labels)

    def set_labels(self, labels):
        if len(self.true_labels) > 0:
            assert np.array_equal(self.true_labels, labels), "voter found non-matching labels"
        else:
            self.true_labels = labels

    def get_labels(self):
        return self.true_labels

    def predict(self):
        return [(func_name, np.round(func(self.pred_labels, axis=0))) for func_name, func in self.agg_funcs]