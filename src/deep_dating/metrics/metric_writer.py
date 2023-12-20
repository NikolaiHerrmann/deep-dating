
import os
import numpy as np
import pandas as pd
from deep_dating.metrics import DatingMetrics
from deep_dating.util import get_date_as_str


class MetricWriter:

    def __init__(self, path):
        file_name = "epoch_log_" + get_date_as_str() + ".csv"
        self.csv_path = os.path.join(path, file_name)

        self.metrics = DatingMetrics()

        cols = ["set_type", "epoch", "mean_loss",
                "std_loss", "min_loss", "max_loss"]
        cols += self.metrics.names
        self.write_row_to_file(cols)

    def _reset_epoch_stats(self):
        self.epoch_losses = []
        self.epoch_labels = []
        self.epoch_preds = []

    def train(self):
        self.state = "train"
        self._reset_epoch_stats()

    def eval(self):
        self.state = "eval"
        self._reset_epoch_stats()

    def write_row_to_file(self, row):
        df = pd.DataFrame([row])
        df.to_csv(self.csv_path, mode="a", index=False, header=False)

    def add_batch_outputs(self, losses, labels, preds):
        self.epoch_losses.append(losses)
        self.epoch_labels.append(labels)
        self.epoch_preds.append(preds)

    def mark_epoch(self, epoch):
        mean_loss = np.mean(self.epoch_losses)
        std_loss = np.std(self.epoch_losses)
        min_loss = np.min(self.epoch_losses)
        max_loss = np.max(self.epoch_losses)

        self.epoch_labels = np.concatenate(self.epoch_labels)
        self.epoch_preds = np.concatenate(self.epoch_preds)
        metric_vals = self.metrics.calc(self.epoch_labels, self.epoch_preds)

        row_to_write = [self.state, epoch, mean_loss,
                        std_loss, min_loss, max_loss] + metric_vals
        self.write_row_to_file(row_to_write)

        return mean_loss
