
import os
import torch
from tqdm import tqdm
from deep_dating.networks import EarlyStopper
from deep_dating.metrics import MetricWriter
from deep_dating.datasets import DatingDataLoader, SetType
from deep_dating.util import get_date_as_str, get_torch_device

class DatingTrainer:

    def __init__(self, num_epochs=50, patience=10, verbose=True):
        self.save_path = "runs"
        self._init_save_dir()
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.device = get_torch_device(verbose)
        self.early_stopper = EarlyStopper(patience)

    def _init_save_dir(self):
        os.makedirs(self.save_path, exist_ok=True)
        self.exp_path = os.path.join(self.save_path, get_date_as_str())
        os.mkdir(self.exp_path)
        self.metric_writer = MetricWriter(self.exp_path)

    def train(self, model, dataset):
        train_loader = DatingDataLoader(dataset, SetType.TRAIN, model)
        val_loader = DatingDataLoader(dataset, SetType.VAL, model)
        model.to(self.device)

        for epoch in range(self.num_epochs):
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.num_epochs}")

            model.train()
            self.metric_writer.train()

            for inputs, labels, paths in tqdm(train_loader, disable = not self.verbose):

                inputs = inputs.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)

                model.optimizer.zero_grad()
                outputs = model(inputs)
                loss = model.criterion(outputs, labels)

                self.metric_writer.add_batch_outputs(loss.item(), labels.detach().numpy(), outputs.detach().numpy())
                
                loss.backward()
                model.optimizer.step()

            mean_train_loss = self.metric_writer.mark_epoch(epoch)
            model.eval()
            self.metric_writer.eval()

            with torch.no_grad():
                for inputs, labels, paths in tqdm(val_loader, disable = not self.verbose):

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device).unsqueeze(1)
                    
                    outputs = model(inputs)
                    loss = model.criterion(outputs, labels)

                    self.metric_writer.add_batch_outputs(loss.item(), labels.detach().numpy(), outputs.detach().numpy())

            mean_val_loss = self.metric_writer.mark_epoch(epoch)
            print(f"Train loss: {mean_train_loss} -- Val loss: {mean_val_loss}")

            stop, save_model = self.early_stopper.stop(mean_val_loss)
            if save_model:
                path = os.path.join(self.exp_path, f"model_epoch_{epoch}.pt")
                model.save(path)
            if stop:
                print("Stopping early!")
                break