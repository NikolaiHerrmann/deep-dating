
import torch
import pickle
import numpy as np
from tqdm import tqdm
from deep_dating.util import get_torch_device


class DatingPredictor:

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.device = get_torch_device(verbose=verbose)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            predictions = pickle.load(f)
        return predictions

    def predict(self, model, data_loader):
        all_outputs = []
        all_labels = []
        all_paths = []

        with torch.no_grad():
            for inputs, labels, paths in tqdm(data_loader, disable = not self.verbose):

                inputs = inputs.to(self.device)
                outputs = model(inputs)
                outputs = outputs.detach().numpy()

                labels = labels.unsqueeze(1)

                all_outputs.append(outputs)
                all_labels.append(labels)
                all_paths += list(paths)

        all_labels = np.concatenate(all_labels)
        all_outputs = np.concatenate(all_outputs)

        save_ls = (all_labels, all_outputs, all_paths)
        with open("predictions.pkl", "wb") as f:
            pickle.dump(save_ls, f)

        with open("predictions.pkl", "rb") as f:
            (all_labels_, all_outputs_, all_paths_) = pickle.load(f)
            print(all_labels_.shape)
            print(len(all_paths_))
            print(all_outputs_.shape)
            print(all_outputs_)