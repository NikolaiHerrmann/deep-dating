
import os
import cv2
import glob
from deep_dating.datasets import SetType, BinDataset, DatasetSplitter, MPS, CLaMM, ScribbleLens, CLaMM_Test_Task3, CLaMM_Test_Task4
from deep_dating.preprocessing import PatchExtractor, PatchMethod, PreprocessRunner, ImageSplitter
from deep_dating.augmentation import AugDoc
from deep_dating.util import DATASETS_PATH
from deep_dating.prediction import AutoencoderPredictor
import matplotlib.pyplot as plt
from tqdm import tqdm


def preprocess_dating_cnn():
    dataset = CLaMM(os.path.join(DATASETS_PATH, "CLaMM_Training_Binet")) #MPS(os.path.join(DATASETS_PATH, "MPS_Binet"), dir_depth=1)

    splitter = DatasetSplitter(dataset, None, None, test_size=0)
    exit()

    preprocessor = PreprocessRunner(dataset.name, ext="_Set_P1_Bin_299", resize=299)
    preprocessing_func = PatchExtractor(plot=False, method=PatchMethod.SLIDING_WINDOW_LINES, num_lines_per_patch=4, is_binary=True).extract_patches

    for set_type in [SetType.TRAIN, SetType.VAL]:
        X, y = splitter.get_data(set_type)
        if X is not None:
            preprocessor.run(X, y, set_type, preprocessing_func)
            print("Patch extraction done for", set_type)
        else:
            print("Skipping", set_type)


def preprocess_dating_cnn_test(dataset):
    print("Running patch extraction for ", dataset.name, "...")

    preprocessor = PreprocessRunner(dataset.name, ext="_Set_Test")
    preprocessing_func = PatchExtractor(plot=False, method=PatchMethod.SLIDING_WINDOW_LINES, num_lines_per_patch=4).extract_patches

    preprocessor.run(dataset.X, dataset.y, SetType.TEST, preprocessing_func)

   
def preprocess_pipeline2():
    dataset = CLaMM(os.path.join(DATASETS_PATH, "CLaMM_Training_Binet")) #CLaMM(path=os.path.join(DATASETS_PATH, "CLaMM_Training_Clean"))
    splitter = DatasetSplitter(dataset, None, None, test_size=0) #400, test_size=0)

    preprocessor = PreprocessRunner(dataset.name, ext="_Set_P2_299")
    preprocessor_func = ImageSplitter(patch_size=299, force_size=True, plot=False).split

    for set_type in [SetType.TRAIN, SetType.VAL]:
        X, y = splitter.get_data(set_type)
        preprocessor.run(X, y, set_type, preprocessor_func)
        print("Image splitting done for", set_type)


def preprocess_bin():
    dataset = BinDataset(augment=True)

    for X_train, X_val, ext, padding_color in [(dataset.X_train, dataset.X_test, "_Set_Aug", 0), (dataset.y_train, dataset.y_test, "_Set_GT_Aug", 255)]:

        preprocessor = PreprocessRunner(dataset.name, ext=ext, include_old_name=False)
        preprocessing_func = PatchExtractor(method=PatchMethod.SLIDING_WINDOW, plot=False, padding_color=padding_color).extract_patches

        y = [0] * len(X_train)
        preprocessor.run(X_train, y, SetType.TRAIN, preprocessing_func)

        y = [0] * len(X_val)
        preprocessor.run(X_val, y, SetType.VAL, preprocessing_func)


def run_binarization():
    dataset = CLaMM_Test_Task3(path=os.path.join(DATASETS_PATH, "CLaMM_task1_task3_Clean"))

    from torch.multiprocessing import Pool, set_start_method
    
    set_start_method('spawn', force=True)
    
    x = dataset.X
    save_path = os.path.join(DATASETS_PATH, str(dataset.name) + "_task1_task3_Clean_Binet")
    os.makedirs(save_path, exist_ok=False)
    
    predictor = AutoencoderPredictor(normalize_per_img=False, save_path=save_path)

    with Pool(8) as pool:
        pool.map(predictor.run, x)            


def test_patch_extraction():
    #X = CLaMM(path=os.path.join(DATASETS_PATH, "CLaMM_Training_Clean_Bin")).X
    dataset = MPS(os.path.join(DATASETS_PATH, "MPS_Binet"), dir_depth=1)
    import random
    random.seed(43)
    X = dataset.X
    random.shuffle(X)

    patch_extractor = PatchExtractor(plot=True, method=PatchMethod.SLIDING_WINDOW_LINES, num_lines_per_patch=4, show_lines_in_plot=False)

    for x in X:
        patch_extractor.extract_patches(x)
        patch_extractor.save_plot(show=True)


def run_aug_doc(n=75, test=False):
    
    if test:
        aug_doc = AugDoc(plot=True)
        while True:
            aug_doc.make_img("")

    img_aug_dir = os.path.join(DATASETS_PATH, "aug_img_1")
    gt_aug_dir = os.path.join(DATASETS_PATH, "aug_gt_1")
    os.makedirs(img_aug_dir, exist_ok=True)
    os.makedirs(gt_aug_dir, exist_ok=True)

    aug_doc = AugDoc(plot=False, save_img_dir=img_aug_dir, save_gt_dir=gt_aug_dir)

    for i in tqdm(range(n)):
        aug_doc.make_img(f"aug_{i}")


if __name__ == "__main__":
    #pass
    #run_aug_doc(test=False)
    #preprocess_pipeline2()
    #preprocess_dating_cnn()
    #preprocess_bin()
    
    #test_patch_extraction()
    run_binarization()


    # parser = argparse.ArgumentParser()
    # parser.add_argument("--test", help="", nargs='?', default=None)
    # args = parser.parse_args()

    # if args.test:
    #run_binarization()
    #test_patch_extraction()
    #test()
    #preprocess_dating_cnn(CLaMM(path=os.path.join(DATASETS_PATH, "CLaMM_Training_Clean")))
    #preprocess_autoencoder()
    # print(CLaMM_Test_Task3().size)
    # print(CLaMM_Test_Task4().size)
    #preprocess_dating_cnn_test()
    #preprocess_bin()
    #run_binarization()
    #preprocess_dating_cnn_test(CLaMM_Test_Task3())
    #test()
    #preprocess_dating_cnn(CLaMM(path=os.path.join(DATASETS_PATH, "CLaMM_Training_Clean")))
    #preprocess_autoencoder()
    #preprocess_dating_cnn(MPS())
    
    #preprocess_dating_cnn(CLaMM(path=os.path.join(DATASETS_PATH, "CLaMM_Training_Clean_Bin")))

