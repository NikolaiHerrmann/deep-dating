
import random
from deep_dating.datasets import MPS, ScribbleLens, CLaMM, CLaMM_Test_Task4, CLaMM_Test_Task3
from deep_dating.prediction import AutoencoderPredictor


if __name__ == "__main__":
    dataset = CLaMM_Test_Task3()

    X = dataset.X
    random.shuffle(X)

    predictor = AutoencoderPredictor(normalize_per_img=False, detect_black_text=True)

    for x in X:
        #x = "../../datasets/MPS/Download/1425/MPS1425_0101.ppm"
        #x ="/home/nikolai/Downloads/datasets/dibco/2018_img_1/10.bmp"
        #x = "/home/nikolai/Downloads/datasets/ICDAR2017_CLaMM_task1_task3/M601415401_MS0455_0012.tif"
        predictor.run(x, plot=True)
