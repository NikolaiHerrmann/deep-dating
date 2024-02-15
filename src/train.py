
import os
from deep_dating.datasets import DatasetName, DatingDataLoader, SetType, CrossVal, CLaMM, MPS
from deep_dating.networks import DatingCNN, DatingTrainer, Autoencoder
from deep_dating.util import DATASETS_PATH
from preprocessing import preprocess_autoencoder, preprocess_dating_cnn


def train_dating_cnn():
    dataset = DatasetName.CLAMM

    #preprocess_autoencoder(CLaMM(path=os.path.join(DATASETS_PATH, "CLaMM_Training_Clean")))
    #preprocess_dating_cnn(MPS())

    cross_val = CrossVal(dataset)
    trainer = DatingTrainer(num_epochs=100, patience=15)
    n_splits = 1

    for i, (X_train, y_train, X_val, y_val) in enumerate(cross_val.get_split(n_splits=n_splits)):
        print(f" -- Running split: {i+1}/{n_splits} -- ")

        model = DatingCNN(model_name="inception_resnet_v2", num_classes=15)
        #model.load("runs/Feb9-12-7-17/model_epoch_0.pt", continue_training=True)
        
        train_loader = DatingDataLoader(dataset, X_train, y_train, model)
        val_loader = DatingDataLoader(dataset, X_val, y_val, model)

        trainer.train(model, train_loader, val_loader, i)

def train_autoencoder():
    
    #preprocess_autoencoder()

    dataset = DatasetName.MPS

    #cross_val = CrossVal(dataset, preprocess_ext="_Set_Auto")
    cross_val_bin = CrossVal(dataset, preprocess_ext="_Set_Auto_Bin")
    trainer = DatingTrainer(patience=20)

    n_splits = 1

    for i in range(n_splits):
        print(f" -- Running split: {i+1}/{n_splits} -- ")

        #(X_train, y_train, X_val, y_val) = next(cross_val.get_split(n_splits=n_splits))
        (X_train_bin, y_train_bin, X_val_bin, y_val_bin) = next(cross_val_bin.get_split(n_splits=n_splits))      

        model = Autoencoder()
        train_loader = DatingDataLoader(dataset, X_train_bin, X_train_bin, model)
        val_loader = DatingDataLoader(dataset, X_val_bin, X_val_bin, model)

        trainer.train(model, train_loader, val_loader, i)


if __name__ == "__main__":
    train_dating_cnn()
    #train_autoencoder()
