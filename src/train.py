
from deep_dating.datasets import DatasetName, DatingDataLoader, SetType
from deep_dating.networks import DatingCNN, DatingTrainer, Autoencoder


def train_dating_cnn():
    model = DatingCNN(model_name="inception_resnet_v2")
    #model.load("runs/Jan6-22-21-16/model_epoch_28.pt", continue_training=True)
    trainer = DatingTrainer()

    dataset = DatasetName.CLAMM

    train_loader = DatingDataLoader(dataset, SetType.TRAIN, model)
    val_loader = DatingDataLoader(dataset, SetType.VAL, model)

    trainer.train(model, train_loader, val_loader)

def train_autoencoder():
    model = Autoencoder()
    trainer = DatingTrainer()

    dataset = DatasetName.CLAMM

    train_loader = DatingDataLoader(dataset, SetType.TRAIN, model, preprocess_ext="_Set_Auto")
    val_loader = DatingDataLoader(dataset, SetType.VAL, model, preprocess_ext="_Set_Auto")

    trainer.train(model, train_loader, val_loader)


if __name__ == "__main__":
    #train_dating_cnn()
    train_autoencoder()
