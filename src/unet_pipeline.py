
import os
import shutil
import random
from deep_dating.datasets import MPS, ScribbleLens, CLaMM, CLaMM_Test_Task4, CLaMM_Test_Task3
from deep_dating.prediction import AutoencoderPredictor
from deep_dating.util import DATASETS_PATH
from tqdm import tqdm


def run_through_dataset(dataset):
    X = dataset.X
    random.shuffle(X)

    predictor = AutoencoderPredictor(normalize_per_img=True)

    for x in X:
        print("Running prediction for:", x)

        predictor.run(x, plot=True)


def run_special_images():

    # (img_name, is_black_text, normalize_per_img)
    imgs = [("M601415401_MS0455_0004.tif", False, False),
            ("M601415401_MS0455_0009.tif", False, False),
            ("M601415401_MS0455_0010.tif", False, False),
            ("M601415401_MS0455_0012.tif", False, False),
            ("M601415401_MS0455_0013.tif", False, False),
            ("btv1b9068411m_f5.tif", True, True),
            ("btv1b9068411m_f17.tif", True, True),
            ("btv1b9068411m_f28.tif", True, True)]
    
    path = os.path.join(DATASETS_PATH, "CLaMM_task1_task3_Clean")
    save_path = os.path.join(DATASETS_PATH, "CLAMM_task1_task3_Clean_Binet")

    # old_save_path = os.path.join(DATASETS_PATH, "Before_Fixing")
    # os.makedirs(old_save_path, exist_ok=True)

    for x, detect_black_text, normalize in tqdm(imgs):
        x_raw = os.path.join(path, x)
        x_bin = os.path.join(save_path, x.rsplit(".", 1)[0] + ".png")
        # shutil.copy(x_raw, old_save_path)
        # shutil.copy(x_bin, old_save_path)
        predictor = AutoencoderPredictor(normalize_per_img=normalize, detect_black_text=detect_black_text, save_path=save_path)
        predictor.run(x_raw)


def run_extra():
    arr = ["IRHT_P_005091.tif",
    "IRHT_P_009793.tif",
    "M601415401006_MS0018_0006.tif",
    "M601415401015_MS0018_0015.tif",
    "M601415401025_MS0018_0026.tif",
    "M601415401037_MS0018_0038.tif",
    "M601415401043_MS0018_0045.tif",
    "M601415401_MS0053_0007.tif",
    "M601415401_MS0053_0019.tif",
    "M601415401_MS0053_0025.tif",
    "M601415401_MS0053_0031.tif",
    "M601415401_MS0053_0036.tif",
    "M601415401_MS0053_0042.tif",
    "M601415401_MS0053_0046.tif",
    "M601415401_MS0053_0053.tif",
    "M601415401_MS0053_0060.tif",
    "M601415401_MS0053_0072.tif",
    "M601415401_MS0102_0007.tif",
    "M601415401_MS0102_0010.tif",
    "M601415401_MS0102_0012.tif",
    "M601415401_MS0102_0016.tif",
    "M601415401_MS0102_0020.tif",
    "M601415401_MS0102_0024.tif",
    "M601415401_MS0102_0029.tif",
    "M601415401_MS0102_0039.tif",
    "M601415401_MS0102_0049.tif",
    "M601415401_MS0102_0061.tif",
    "M601415401_MS0102_0068.tif",
    "M601415401_MS0124_0005.tif",
    "M601415401_MS0124_0011.tif",
    "M601415401_MS0124_0023.tif",
    "M601415401_MS0124_0027.tif"]

    path = os.path.join(DATASETS_PATH, "CLaMM_task1_task3_Clean")
    save_path = os.path.join(DATASETS_PATH, "CLAMM_task1_task3_Clean_Binet")
    predictor = AutoencoderPredictor(normalize_per_img=False, save_path=save_path)

    for x in tqdm(arr):
        x_raw = os.path.join(path, x)
        predictor.run(x_raw)

if __name__ == "__main__":
    #run_special_images()
    run_extra()
