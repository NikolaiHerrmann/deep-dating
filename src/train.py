
from deep_dating.datasets import DatasetName
from deep_dating.networks import DatingCNN, DatingTrainer


if __name__ == "__main__":
    model = DatingCNN(model_name="inception_resnet_v2")
    model.load("runs/Jan6-22-21-16/model_epoch_28.pt", continue_training=True)
    trainer = DatingTrainer()
    trainer.train(model, DatasetName.CLAMM)
