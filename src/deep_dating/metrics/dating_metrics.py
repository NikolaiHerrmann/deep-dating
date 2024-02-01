
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score


class DatingMetrics:

    def __init__(self, alphas=[0, 25], classification=False, average="micro"):
        self.alphas = alphas
        assert len(self.alphas) > 0, "Empty alpha value list!"
        self.alpha_counter = 0

        self.metrics = {"mae": self._mae, "mse": self._mse}
        for alpha in alphas:
            self.metrics[f"cs_{alpha}"] = self._cs

        if classification:
            self.average = average
            self.metrics["f1"] = self._f1
            self.metrics["accuracy"] = self._accuracy
            self.metrics["recall"] = self._recall
            self.metrics["precision"] = self._precision

        self.names = self.metrics.keys()

    def _f1(self):
        return f1_score(self.true, self.pred, average=self.average)
    
    def _precision(self):
        return precision_score(self.true, self.pred, average=self.average)
    
    def _recall(self):
        return recall_score(self.true, self.pred, average=self.average)
    
    def _accuracy(self):
        return accuracy_score(self.true, self.pred, normalize=True)

    def _mae(self):
        return np.sum(self.abs_diff) / self.n

    def _mse(self):
        return np.sum(np.square(self.diff)) / self.n

    def _cs(self):
        alpha = self.alphas[self.alpha_counter]
        self.alpha_counter += 1
        if self.alpha_counter == len(self.alphas):
            self.alpha_counter = 0

        return (np.count_nonzero(self.abs_diff <= alpha) / self.n) * 100

    def calc(self, true, pred):
        self.true = true
        self.pred = np.round(pred)

        self.diff = self.true - self.pred
        self.abs_diff = np.abs(self.diff)
        self.n = np.max(self.pred.shape)

        return [func() for func in self.metrics.values()]


if __name__ == "__main__":
    metrics = DatingMetrics(alphas=[0, 5, 25], classification=True)
    true = [12, 12, 24, 14]
    preds = [12, 13, 24, 15]
    vals = metrics.calc(true, preds)
    print(len(metrics.names), metrics.names)
    print(len(vals), vals)
