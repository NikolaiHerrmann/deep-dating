
from deep_dating.datasets import DatasetName
from deep_dating.networks import DatingCNN, DatingTrainer


if __name__ == "__main__":
    model = DatingCNN(model_name="inception_resnet_v2")
    trainer = DatingTrainer()
    trainer.train(model, DatasetName.SCRIBBLE)
