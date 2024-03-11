
import random
from deep_dating.datasets import MPS, ScribbleLens, CLaMM, CLaMM_Test_Task4, CLaMM_Test_Task3
from deep_dating.prediction import AutoencoderPredictor


if __name__ == "__main__":
    dataset = MPS()

    X = dataset.X
    random.shuffle(X)
    random.shuffle(X)
    random.shuffle(X)
    random.shuffle(X)
    random.shuffle(X)

    predictor = AutoencoderPredictor(normalize_per_img=False)#,model_path="runs_v2/Binet_norm/model_epoch_329_split_0.pt", detect_black_text=True)

    for x in X:
        #x = "../../datasets/MPS/Download/1425/MPS1425_0101.ppm"
        #x ="/home/nikolai/Downloads/datasets/dibco/2018_img_1/10.bmp"
        #x = "/home/nikolai/Downloads/datasets/CLaMM_Training_Clean/IRHT_P_000455.tif"
        print(x)
        predictor.run(x, plot=True)
