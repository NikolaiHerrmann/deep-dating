
import os
import torch
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from deep_dating.networks import EarlyStopper, ModelType
from deep_dating.metrics import MetricWriter
from deep_dating.util import get_date_as_str, get_torch_device, save_figure


class DatingTrainer:

    def __init__(self, num_epochs=200, patience=10, verbose=True):
        self.save_path = "runs"
        self._init_save_dir()
        self.num_epochs = num_epochs
        self.patience = patience
        self.verbose = verbose
        self.device = get_torch_device(verbose)
        self.early_stopper = EarlyStopper(patience)

    def _init_save_dir(self):
        os.makedirs(self.save_path, exist_ok=True)
        self.exp_path = os.path.join(self.save_path, get_date_as_str())
        os.mkdir(self.exp_path)

    def _write_training_settings(self, model, loader):
        settings = {}

        settings["max_num_epochs"] = self.num_epochs
        settings["patience"] = self.patience
        settings["model_name"] = model.model_name
        settings["img_input_size"] = model.input_size
        settings["learning_rate"] = model.learning_rate
        settings["batch_size"] = loader.model_batch_size
        settings["dataset"] = loader.dataset_name.value

        if model.model_type == ModelType.PATCH_CNN:
            settings["starting_weights"] = model.starting_weights
            settings["classification"] = model.classification
            settings["weight_decay"] = model.weight_decay
            if model.classification:
                settings["classes"] = loader.torch_dataset.get_class_dict()

        json_object = json.dumps(settings, indent=4)

        with open(os.path.join(self.exp_path, "settings.json"), "w") as f:
            f.write(json_object)

    def _write_architecture(self, model):
        stdout_old = sys.stdout

        with open(os.path.join(self.exp_path, "architecture.txt"), "w") as sys.stdout:
            print("Torch summary")
            model.summary()
            print("---- Direct Print ----")
            print(model)

        sys.stdout = stdout_old

    def _get_labels(self, model, labels, inputs):
        labels = labels.to(self.device)

        if model.model_type == ModelType.PATCH_CNN:
            labels = torch.flatten(labels) if model.classification else labels.unsqueeze(1)
        # elif model.model_type == ModelType.AUTOENCODER:
        #     return labels
            
        return labels
        
    def _save_example(self, epoch, state, model, inputs, outputs, labels):
        if model.model_type == ModelType.AUTOENCODER:
            fig, axs = plt.subplots(1, 3)

            batch_size = inputs.shape[0]
            rand_idx = np.random.randint(0, batch_size, size=1)

            axs[0].imshow(inputs[rand_idx, :][0, 0, :], cmap="gray")
            axs[0].set_title("Input")
            axs[1].imshow(labels[rand_idx, :][0, 0, :], cmap="gray")
            axs[1].set_title("Label")
            axs[2].imshow(outputs[rand_idx, :][0, 0, :], cmap="gray")
            axs[2].set_title("Reconstruction")

            fig.tight_layout()
            save_figure(f"example_{state}_epoch_{epoch}", fig=fig, fig_dir=self.exp_path, pdf=False)

    def _detach(self, loader, model, outputs, labels):
        labels_detach = labels.cpu().detach().numpy()
        outputs_detach = outputs.cpu().detach().numpy()
        
        if model.model_type == ModelType.PATCH_CNN:
            if model.classification:
                class_idxs = np.argmax(outputs_detach, axis=1)  # get the index of the max log-probability
                outputs_detach = loader.torch_dataset.decode_class(class_idxs)
                labels_detach = loader.torch_dataset.decode_class(labels_detach)

        return labels_detach, outputs_detach

    def train(self, model, train_loader, val_loader, split):
        self.metric_writer = MetricWriter(self.exp_path, model.metrics)

        if split == 0:
            self._write_training_settings(model, train_loader)
            self._write_architecture(model)
        
        model.to(self.device)

        for epoch in range(self.num_epochs):
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.num_epochs}")

            model.train()
            self.metric_writer.train()

            for inputs, labels, paths in tqdm(train_loader, disable = not self.verbose):

                labels = self._get_labels(model, labels, inputs) #labels.to(self.device).unsqueeze(1)
                inputs = inputs.to(self.device)

                model.optimizer.zero_grad()
                outputs = model(inputs)
                loss = model.criterion(outputs, labels)

                labels_detach, outputs_detach = self._detach(train_loader, model, outputs, labels)

                self.metric_writer.add_batch_outputs(loss.item(), labels_detach, outputs_detach)
                
                loss.backward()
                model.optimizer.step()

            self._save_example(epoch, "train", model, inputs.cpu().detach().numpy(), outputs_detach, labels_detach)
            mean_train_loss = self.metric_writer.mark_epoch(epoch)
            model.eval()
            self.metric_writer.eval()

            with torch.no_grad():
                for inputs, labels, paths in tqdm(val_loader, disable = not self.verbose):

                    inputs = inputs.to(self.device)
                    labels = self._get_labels(model, labels, inputs)
                    
                    outputs = model(inputs)
                    loss = model.criterion(outputs, labels)

                    labels_detach, outputs_detach = self._detach(val_loader, model, outputs, labels)

                    self.metric_writer.add_batch_outputs(loss.item(), labels_detach, outputs_detach)

            self._save_example(epoch, "val", model, inputs.cpu().detach().numpy(), outputs_detach, labels_detach)
            mean_val_loss = self.metric_writer.mark_epoch(epoch)
            if self.verbose:
                print(f"Train loss: {mean_train_loss} -- Val loss: {mean_val_loss}")

            stop, save_model = self.early_stopper.stop(mean_val_loss)
            if save_model:
                path = os.path.join(self.exp_path, f"model_epoch_{epoch}.pt")
                model.save(path)
            if stop:
                if self.verbose:
                    print("Stopping early!")
                break