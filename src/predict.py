
import torch
import os
from deep_dating.datasets import DatasetName, DatingDataLoader, SetType, CrossVal
from deep_dating.networks import DatingCNN
from deep_dating.prediction import DatingPredictor
from deep_dating.util import get_torch_device


if __name__ == "__main__":
    #model_path = "runs/Dec21-16-31-47/model_epoch_8.pt"
    model_path = "runs/Feb2-13-25-11/model_epoch_4.pt" #"runs/Jan8-19-25-16/model_epoch_3.pt"

    model = DatingCNN("inception_resnet_v2", num_classes=11)
    model.load(model_path, continue_training=False)
    predictor = DatingPredictor()

    dataset = DatasetName.MPS

    cross_val = CrossVal(dataset)

    X_train, y_train, X_val, y_val = list(cross_val.get_split(n_splits=1))[0]

    val_loader = DatingDataLoader(dataset, X_val, y_val, model)

    model_file_name = os.path.basename(model_path).split(".")[0] + "_pred.pkl"
    dirs = os.path.dirname(model_path)
    save_path = os.path.join(dirs, model_file_name)
    
    predictor.predict(model, val_loader, save_path=save_path)