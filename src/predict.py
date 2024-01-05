
import torch
from deep_dating.datasets import DatasetName, DatingDataLoader, SetType
from deep_dating.networks import DatingCNN
from deep_dating.prediction import DatingPredictor


if __name__ == "__main__":
    model = DatingCNN()
    model.load_state_dict(torch.load("runs/Dec21-16-31-47/model_epoch_8.pt"))
    model.eval()
    predictor = DatingPredictor()

    val_loader = DatingDataLoader(DatasetName.MPS, SetType.VAL, model)
    predictor.predict(model, val_loader)