
import torch
from tqdm import tqdm
from deep_dating.networks import EarlyStopper
from deep_dating.metrics import DatingMetricWriter
from deep_dating.datasets import DatingDataLoader, SetType


class DatingTrainer:

    def __init__(self, num_epochs=100, verbose=True):
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if self.verbose:
            print(f"Training on: {self.device}")
        self.metric_writer = DatingMetricWriter()
        self.early_stopper = EarlyStopper()

    def train(self, model, dataset):
        train_loader = DatingDataLoader(dataset, SetType.TRAIN, model)
        val_loader = DatingDataLoader(dataset, SetType.VAL, model)
        model.to(self.device)

        for epoch in range(self.num_epochs):
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.num_epochs}")

            model.train()
            self.metric_writer.train()

            for inputs, labels in tqdm(train_loader, disable = not self.verbose):
                
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)

                model.optimizer.zero_grad()
                outputs = model(inputs)
                loss = model.criterion(outputs, labels)

                self.metric_writer.add_batch_outputs(loss.item(), labels.detach().numpy(), outputs.detach().numpy())
                
                loss.backward()
                model.optimizer.step()

            self.metric_writer.mark_epoch(epoch)
            model.eval()
            self.metric_writer.eval()

            with torch.no_grad():
                for inputs, labels in tqdm(val_loader):

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device).unsqueeze(1)
                    
                    outputs = model(inputs)
                    loss = model.criterion(outputs, labels)

                    self.metric_writer.add_batch_outputs(loss.item(), labels.detach().numpy(), outputs.detach().numpy())

            mean_val_loss = self.metric_writer.mark_epoch(epoch)

            stop, save_model = self.early_stopper.stop(mean_val_loss)
            if save_model:
                model.save(f"model_epoch_{epoch}.pt")
            if stop:
                print("Stopping early!")
                break