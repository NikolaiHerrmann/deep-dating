
from deep_dating.datasets import DatasetName
from deep_dating.networks import DatingCNN, DatingTrainer, Autoencoder, AutoencoderTrainer


def train_dating_cnn():
    model = DatingCNN(model_name="inception_resnet_v2")
    #model.load("runs/Jan6-22-21-16/model_epoch_28.pt", continue_training=True)
    trainer = DatingTrainer()
    trainer.train(model, DatasetName.CLAMM)

def train_autoencoder():
    model = Autoencoder()
    trainer = AutoencoderTrainer()
    trainer.train(model, DatasetName.CLAMM)


if __name__ == "__main__":
    # train_dating_cnn()
    train_autoencoder()
