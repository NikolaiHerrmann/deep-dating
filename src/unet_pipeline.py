
import random
from deep_dating.datasets import MPS, ScribbleLens, CLaMM, CLaMM_Test_Task4
from deep_dating.prediction import AutoencoderPredictor


if __name__ == "__main__":
    dataset = MPS()

    X = dataset.X
    # random.shuffle(X)
    random.shuffle(X)

    predictor = AutoencoderPredictor(model_path="runs_v2/binet_normal/model_epoch_275_split_0.pt")

    for x in X:
        x = "../../datasets/MPS/Download/1350/MPS1350_0070.ppm"
        #x ="/home/nikolai/Downloads/datasets/dibco/2018_img_1/10.bmp"
        #x = "/home/nikolai/Downloads/datasets/ICDAR2017_CLaMM_task2_task4/500256201_MS0058_0124.jpg"
        predictor.run(x, plot=True)
