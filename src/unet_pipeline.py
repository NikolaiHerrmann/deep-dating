
import os
import glob
import shutil
import random
from deep_dating.datasets import MPS, ScribbleLens, CLaMM, CLaMM_Test_Task4, CLaMM_Test_Task3
from deep_dating.prediction import BiNetPredictor
from deep_dating.util import DATASETS_PATH
from tqdm import tqdm


def run_through_dataset(dataset):
    X = dataset.X
    random.shuffle(X)
    random.shuffle(X)

    predictor = BiNetPredictor(normalize_per_img=False, resize_factor=1)

    for x in X:
        print("Running prediction for:", x)

        x = os.path.join(DATASETS_PATH, "CLaMM_task2_task4_Clean", "CMDF_1_74b.jpg")
        x = os.path.join(DATASETS_PATH, "ICDAR2017_CLaMM_Training", "IRHT_P_005976.tif")
        x = "/home/nikolai/Downloads/attempts/Guerin(1)/Guerin(1)/FRAN_0021_5064_A.jpg"
        img = predictor.run(x, plot=True)
        # import cv2
        # cv2.imwrite(os.path.join(DATASETS_PATH, "t.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def run_special_task4():
    path = os.path.join(DATASETS_PATH, "CLaMM_task2_task4_Clean")
    imgs = glob.glob(os.path.join(path, "CMDF*"))

    save_path = os.path.join(DATASETS_PATH, "CLAMM_task2_task4_Clean_Binet")
    predictor = BiNetPredictor(normalize_per_img=False, resize_factor=0.35, save_path=save_path)

    for x in tqdm(imgs):
        predictor.run(x)


def run_special_task3():

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
        predictor = BiNetPredictor(normalize_per_img=normalize, detect_black_text=detect_black_text, save_path=save_path)
        predictor.run(x_raw)


if __name__ == "__main__":
    #run_special_task4()
    #run_extra()
    run_through_dataset(CLaMM_Test_Task4())
