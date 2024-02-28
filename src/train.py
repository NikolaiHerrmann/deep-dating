
import os
from deep_dating.datasets import DatasetName, DatingDataLoader, SetType, CrossVal, CLaMM, MPS
from deep_dating.networks import DatingCNN, DatingTrainer, Autoencoder
from deep_dating.util import DATASETS_PATH


def train_dating_cnn():
    dataset = DatasetName.CLAMM

    cross_val = CrossVal(dataset, preprocess_ext="_Set_Bin")
    trainer = DatingTrainer(num_epochs=100, patience=20)
    n_splits = 1

    for i, (X_train, y_train, X_val, y_val) in enumerate(cross_val.get_split(n_splits=n_splits)):
        print(f" -- Running split: {i+1}/{n_splits} -- ")

        model = DatingCNN(model_name="inception_resnet_v2", num_classes=15, dropout=True)
        #model.load("runs/Feb26-19-40-59/model_epoch_7.pt", continue_training=True)
        
        train_loader = DatingDataLoader(dataset, X_train, y_train, model)
        val_loader = DatingDataLoader(dataset, X_val, y_val, model)

        trainer.train(model, train_loader, val_loader, i)

def train_autoencoder():
    
    #preprocess_autoencoder()

    dataset = DatasetName.CLAMM

    cross_val = CrossVal(dataset, preprocess_ext="_Set_Auto")
    #cross_val_bin = CrossVal(dataset, preprocess_ext="_Set_Auto_Bin")
    trainer = DatingTrainer(patience=20)

    n_splits = 1

    for i in range(n_splits):
        print(f" -- Running split: {i+1}/{n_splits} -- ")

        (X_train, y_train, X_val, y_val) = next(cross_val.get_split(n_splits=n_splits))
        #(X_train_bin, y_train_bin, X_val_bin, y_val_bin) = next(cross_val.get_split(n_splits=n_splits))      

        model = Autoencoder()
        train_loader = DatingDataLoader(dataset, X_train, X_train, model)
        val_loader = DatingDataLoader(dataset, X_val, X_val, model)

        trainer.train(model, train_loader, val_loader, i)

def train_autoencoder2():

    dataset = DatasetName.DIBCO

    cross_val = CrossVal(dataset, preprocess_ext="_Set")
    cross_val_gt = CrossVal(dataset, preprocess_ext="_Set_GT")
    trainer = DatingTrainer(num_epochs=200, patience=50)

    n_splits = 1

    for i in range(n_splits):
        print(f" -- Running split: {i+1}/{n_splits} -- ")

        (X_train, y_train, X_val, y_val) = next(cross_val.get_split(n_splits=n_splits))
        (X_train_gt, y_train_gt, X_val_gt, y_val_gt) = next(cross_val_gt.get_split(n_splits=n_splits))      

        model = Autoencoder()
        model.load("runs/Feb25-16-43-57/model_epoch_195.pt", continue_training=True)
        train_loader = DatingDataLoader(dataset, X_train, X_train_gt, model)
        val_loader = DatingDataLoader(dataset, X_val, X_val_gt, model)

        trainer.train(model, train_loader, val_loader, i)



if __name__ == "__main__":
    train_dating_cnn()
    #train_autoencoder2()
