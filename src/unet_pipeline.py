
import random
from deep_dating.datasets import MPS, ScribbleLens, CLaMM, CLaMM_Test_Task4
from deep_dating.prediction import AutoencoderPredictor


if __name__ == "__main__":
    dataset = MPS()

    X = dataset.X
    random.shuffle(X)

    predictor = AutoencoderPredictor()

    for x in X:
        predictor.run(x, plot=True)
