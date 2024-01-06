
import numpy as np


class EarlyStopper:

    def __init__(self, patience):
        self.patience = patience
        self.min_val_loss = np.inf
        
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