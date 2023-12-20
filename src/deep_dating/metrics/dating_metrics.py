
import numpy as np


class DatingMetrics:

    def __init__(self):
        self.metrics = {"mae": self._mae,
                        "mse": self._mse,
                        "cs_0": self._cs_0,
                        "cs_25": self._cs_25}
        self.names = self.metrics.keys()

    def _mae(self):
        return np.sum(np.abs(self.true - self.pred)) / self.n

    def _mse(self):
        return np.sum(np.square(self.true - self.pred)) / self.n

    def _cs_0(self):
        return (np.count_nonzero(self.diff <= 0) / self.n) * 100

    def _cs_25(self):
        return (np.count_nonzero(self.diff <= 25) / self.n) * 100

    def calc(self, true, preds):
        self.true = true
        self.pred = np.round(preds)

        self.diff = np.abs(self.true - self.pred)
        self.n = np.max(preds.shape)

        return [self.metrics[key]() for key in self.metrics.keys()]
