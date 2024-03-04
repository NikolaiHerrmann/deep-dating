
import numpy as np


class EarlyStopper:
    """
    Early stopping class since PyTorch doesn't have one
    Class adapted from https://stackoverflow.com/a/73704579
    """

    def __init__(self, patience, min_val_loss=np.inf, patience_count=0):
        self.patience = patience
        self.min_val_loss = min_val_loss
        self.patience_count = patience_count
        
    def stop(self, current_val_loss):
        stop = False
        save_model = False

        if current_val_loss < self.min_val_loss:
            self.min_val_loss = current_val_loss
            self.patience_count = 0
            save_model = True
        elif current_val_loss > self.min_val_loss:
            self.patience_count += 1
            if self.patience_count >= self.patience:
                stop = True
        
        return stop, save_model