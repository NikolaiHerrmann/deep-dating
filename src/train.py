
from deep_dating.datasets import DatasetName
from deep_dating.preprocessing import Preprocessor
from deep_dating.networks import DatingCNN, DatingTrainer


if __name__ == "__main__":
    model = DatingCNN()
    trainer = DatingTrainer()
    trainer.train(model, DatasetName.MPS)
