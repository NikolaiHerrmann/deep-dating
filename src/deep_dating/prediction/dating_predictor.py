
import torch
import pickle
import numpy as np
from tqdm import tqdm
from deep_dating.util import get_torch_device, get_date_as_str
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


class DatingPredictor:

    def __init__(self, verbose=True):
        self.verbose = verbose

    def load(self, path):
        with open(path, "rb") as f:
            predictions = pickle.load(f)
        return predictions
    
    def _format_output_and_labels(self, model, loader, outputs, labels):
        outputs_detach = outputs.cpu().detach().numpy()

        if model.classification:
            if not model.feature_extractor:
                class_idxs = np.argmax(outputs_detach, axis=1)  # get the index of the max log-probability
                outputs_detach = loader.torch_dataset.decode_class(class_idxs)

            labels_detach = torch.flatten(labels)
            labels_detach = loader.torch_dataset.decode_class(labels_detach)
        else:
            labels_detach = labels.unsqueeze(1)

        return outputs_detach, labels_detach

    def predict(self, model, data_loader, save_path=None, check_loading=True):
        all_outputs = []
        all_labels = []
        all_paths = []

        self.device = get_torch_device(verbose=self.verbose)
        model.to(self.device)

        if not save_path:
            save_path = "pred_" + get_date_as_str() + ".pkl"
            if self.verbose:
                print("No save path was given using:", save_path)

        with torch.no_grad():
            for inputs, labels, paths in tqdm(data_loader, disable = not self.verbose):

                inputs = inputs.to(self.device)
                outputs = model(inputs)
                
                outputs, labels = self._format_output_and_labels(model, data_loader, outputs, labels)

                all_outputs.append(outputs)
                all_labels.append(labels)
                all_paths += list(paths)

        all_labels = np.concatenate(all_labels)
        all_outputs = np.concatenate(all_outputs)

        with open(save_path, "wb") as f:
            pickle.dump((all_labels, all_outputs, all_paths), f)

        if check_loading:
            all_labels_, all_outputs_, all_paths_ = self.load(save_path)
            assert len(all_labels_) == len(all_outputs_) == len(all_paths_), "Loading Test Failed!"