
import random
from deep_dating.datasets import MPS, ScribbleLens, CLaMM, CLaMM_Test_Task4
from deep_dating.prediction import AutoencoderPredictor


if __name__ == "__main__":
    dataset = MPS()

    X = dataset.X
    random.shuffle(X)

    predictor = AutoencoderPredictor(normalize_per_img=False)

    for x in X:
        #x = "../../datasets/MPS/Download/1425/MPS1425_0101.ppm"
        #x ="/home/nikolai/Downloads/datasets/dibco/2018_img_1/10.bmp"
        #x = "/home/nikolai/Downloads/datasets/ICDAR2017_CLaMM_task2_task4/710146101_S_019_019bis_0097.jpg"
        x ="P166-Fg007-R-C01-R01-fused.jpg"
        predictor.run(x, plot=True)
