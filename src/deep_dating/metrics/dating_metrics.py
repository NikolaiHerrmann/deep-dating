
import numpy as np


class DatingMetrics:

    def __init__(self, alphas=[0, 25]):
        self.alphas = alphas
        assert len(self.alphas) > 0, "Empty alpha value list!"
        self.alpha_counter = 0

        self.metrics = {"mae": self._mae,
                        "mse": self._mse}
        for alpha in alphas:
            self.metrics[f"cs_{alpha}"] = self._cs

        self.names = self.metrics.keys()

    def _mae(self):
        return np.sum(np.abs(self.true - self.pred)) / self.n

    def _mse(self):
        return np.sum(np.square(self.true - self.pred)) / self.n

    def _cs(self):
        alpha = self.alphas[self.alpha_counter]
        self.alpha_counter += 1
        if self.alpha_counter == len(self.alphas):
            self.alpha_counter = 0

        return (np.count_nonzero(self.diff <= alpha) / self.n) * 100

    def calc(self, true, pred):
        self.true = true
        self.pred = np.round(pred)

        self.diff = np.abs(self.true - self.pred)
        self.n = np.max(self.pred.shape)

        return [func() for func in self.metrics.values()]


if __name__ == "__main__":
    metrics = DatingMetrics(alphas=[0, 5, 25])
    true = [12, 12, 24, 14]
    preds = [12, 13, 22, 15]
    vals = metrics.calc(true, preds)
    print(len(metrics.names), metrics.names)
    print(len(vals), vals)
