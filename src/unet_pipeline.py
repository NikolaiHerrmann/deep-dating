
import random
from deep_dating.datasets import MPS, ScribbleLens, CLaMM, CLaMM_Test_Task4
from deep_dating.prediction import AutoencoderPredictor


if __name__ == "__main__":
    dataset = CLaMM_Test_Task4()

    X = dataset.X
    random.shuffle(X)

    predictor = AutoencoderPredictor(model_path="runs/unet1/model_epoch_273_split_0.pt")

    for x in X:
        predictor.run(x, plot=True)
