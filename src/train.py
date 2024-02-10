
import os
from deep_dating.datasets import DatasetName, DatingDataLoader, SetType, CrossVal, CLaMM, MPS
from deep_dating.networks import DatingCNN, DatingTrainer, Autoencoder
from deep_dating.util import DATASETS_PATH
from preprocessing import preprocess_autoencoder, preprocess_dating_cnn


def train_dating_cnn():
    dataset = DatasetName.MPS

    #preprocess_autoencoder(CLaMM(path=os.path.join(DATASETS_PATH, "CLaMM_Training_Clean")))
    #preprocess_dating_cnn(MPS())

    cross_val = CrossVal(dataset)
    trainer = DatingTrainer(num_epochs=100, patience=15)
    n_splits = 1

    for i, (X_train, y_train, X_val, y_val) in enumerate(cross_val.get_split(n_splits=n_splits)):
        print(f" -- Running split: {i+1}/{n_splits} -- ")

        model = DatingCNN(model_name="inception_resnet_v2", num_classes=11)
        #model.load("runs/Feb9-12-7-17/model_epoch_0.pt", continue_training=True)
        
        train_loader = DatingDataLoader(dataset, X_train, y_train, model)
        val_loader = DatingDataLoader(dataset, X_val, y_val, model)

        trainer.train(model, train_loader, val_loader, i)

def train_autoencoder():
    model = Autoencoder()
    trainer = DatingTrainer(patience=50)

    dataset = DatasetName.CLAMM

    train_loader = DatingDataLoader(dataset, SetType.TRAIN, model, preprocess_ext="_Set_Auto")
    val_loader = DatingDataLoader(dataset, SetType.VAL, model, preprocess_ext="_Set_Auto")

    trainer.train(model, train_loader, val_loader)


if __name__ == "__main__":
    train_dating_cnn()
    #train_autoencoder()
