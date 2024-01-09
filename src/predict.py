
import torch
import os
from deep_dating.datasets import DatasetName, DatingDataLoader, SetType
from deep_dating.networks import DatingCNN
from deep_dating.prediction import DatingPredictor
from deep_dating.util import get_torch_device


if __name__ == "__main__":
    #model_path = "runs/Dec21-16-31-47/model_epoch_8.pt"
    model_path = "runs/Jan8-19-25-16/model_epoch_3.pt"

    model = DatingCNN("inception_resnet_v2")
    model.load(model_path, continue_training=False)
    predictor = DatingPredictor()

    val_loader = DatingDataLoader(DatasetName.CLAMM, SetType.VAL, model)

    model_file_name = os.path.basename(model_path).split(".")[0] + "_pred.pkl"
    dirs = os.path.dirname(model_path)
    save_path = os.path.join(dirs, model_file_name)
    
    predictor.predict(model, val_loader, save_path=save_path)