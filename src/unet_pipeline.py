
import random
from deep_dating.datasets import MPS, ScribbleLens, CLaMM, CLaMM_Test_Task4
from deep_dating.prediction import AutoencoderPredictor


if __name__ == "__main__":
    dataset = ScribbleLens()

    X = dataset.X
    #random.shuffle(X)

    predictor = AutoencoderPredictor(model_path="runs/unet1/model_epoch_328_split_0.pt")

    for x in X:
        x = "/home/nikolai/Downloads/datasets/ICDAR2017_CLaMM_task2_task4/B591226101_MS0164_0062.jpg"
        predictor.run(x, plot=True)
