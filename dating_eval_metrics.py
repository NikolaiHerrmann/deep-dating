
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt


class DatingEvalMetricWriter:

    def __init__(self, alphas=[0, 25], test_mode=False, path=None):
        self.alphas = alphas
        self.test_mode = test_mode
        self.write_file_name = "test_"
        self.run_path = path if path else ""
        cols = []

        if not self.test_mode:
            self.writer = SummaryWriter()
            self.state = "train"
            self.write_file_name = "epoch_log_"
            self.run_path = self.writer.get_logdir()
            cols = ["set", "epoch", "mean_running_loss", "std_running_loss"]

        now = datetime.datetime.now()
        self.write_file_name += f"{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}.csv"
        self.csv_path = os.path.join(self.run_path, self.write_file_name)
        cols += ["mse", "mae"] + [f"cs_{a}" for a in self.alphas]
        self.write_to_file(cols)

    def __del__(self):
        if not self.test_mode:
            self.writer.flush()

    def mae(self, true, pred):
        pred = np.round(pred)
        n = np.max(pred.shape)
        return np.sum(np.abs(true - pred)) / n

    def mse(self, true, pred):
        pred = np.round(pred)
        n = np.max(pred.shape)
        return np.sum(np.square(true - pred)) / n

    def cs(self, true, pred):
        pred = np.round(pred)
        n = np.max(pred.shape)
        diff = np.abs(true - pred)
        return [(np.count_nonzero(diff <= a) / n) * 100 for a in self.alphas]
    
    def train(self):
        self.state = "train"

    def eval(self):
        self.state = "eval"

    def write_to_file(self, row):
        df = pd.DataFrame([row])
        df.to_csv(self.csv_path, mode="a", index=False, header=False)       

    def calc_metrics(self, labels, preds):
        mse = self.mse(labels, preds)
        mae = self.mae(labels, preds)
        cs_list = self.cs(labels, preds)
        return mse, mae, cs_list
        
    def epoch(self, epoch_losses, epoch_labels, epoch_preds, epoch):
        if self.test_mode:
            print("Warning in test mode!")
            return 

        mean_running_loss = np.mean(epoch_losses)
        std_running_loss = np.std(epoch_losses)
        mse, mae, cs_list = self.calc_metrics(epoch_labels, epoch_preds)

        row = [self.state, epoch, mean_running_loss, std_running_loss, mse, mae] + cs_list
        self.write_to_file(row)

        self.writer.add_scalar(f"{self.state} Running Loss", mean_running_loss, epoch)
        self.writer.add_scalar(f"{self.state} MAE", mae, epoch)
        self.writer.add_scalar(f"{self.state} MSE", mse, epoch)
        for cs_val, alpha in zip(cs_list, self.alphas):
            self.writer.add_scalar(f"{self.state} CS (alpha={alpha})", cs_val, epoch)




# if __name__ == "__main__":
#     DatingEvalMetricWriter.plot("runs/Dec06_11-04-06_p14s/epoch_log_12_6_11_4_6.csv")