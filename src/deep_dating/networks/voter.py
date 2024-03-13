
import numpy as np
import scipy as sp


class Voter:
    
    def __init__(self):
        self.agg_funcs = [np.mean, np.median, sp.stats.mode]
        self.reset()

    def reset(self):
        self.pred_labels = []
        self.true_labels = []

    def add_prediction(self, labels):
        self.pred_labels.append(labels)

    def set_labels(self, labels):
        if len(self.true_labels) > 0:
            assert np.array_equal(self.true_labels, labels)
        else:
            self.true_labels = labels

    def get_labels(self):
        return self.true_labels

    def predict(self):
        return [func(self.pred_labels, axis=0) for func in self.agg_funcs]