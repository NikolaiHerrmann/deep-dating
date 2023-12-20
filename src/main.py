
from deep_dating.datasets import MPS, ScribbleLens, CLaMM
from deep_dating.networks import DatingCNN, DatingTrainer


if __name__ == "__main__":
    model = DatingCNN()
    dataset = MPS()
    trainer = DatingTrainer()
    trainer.train(model, dataset)
