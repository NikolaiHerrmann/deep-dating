
import torch
from deep_dating.datasets import DatingDataLoader, SetType
from deep_dating.util import get_torch_device
from tqdm import tqdm


class AutoencoderTrainer:

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.device = get_torch_device(verbose)
        self.num_epochs = 50

    def train(self, model, dataset):
        train_loader = DatingDataLoader(dataset, SetType.TRAIN, model, preprocess_ext="_Set_Auto")
        val_loader = DatingDataLoader(dataset, SetType.VAL, model, preprocess_ext="_Set_Auto")

        model.to(self.device)

        # train_loader.test_loading()
        # exit()

        for epoch in range(self.num_epochs):
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.num_epochs}")

            model.train()

            for inputs, labels, paths in tqdm(train_loader, disable = not self.verbose):

                inputs = inputs.to(self.device)

                model.optimizer.zero_grad()
                outputs = model(inputs)
                loss = model.criterion(outputs, inputs)
                print(loss)
                
                loss.backward()
                model.optimizer.step()

            model.eval()

            with torch.no_grad():
                for inputs, labels, paths in tqdm(val_loader, disable = not self.verbose):
                    
                    inputs = inputs.to(self.device)
                    
                    outputs = model(inputs)
                    loss = model.criterion(outputs, inputs)
                    print(loss)
