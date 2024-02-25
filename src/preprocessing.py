
import os
from deep_dating.datasets import load_all_dating_datasets, SetType, BinDataset, DatasetSplitter, MPS, CLaMM, ScribbleLens, CLaMM_Test_Task3, CLaMM_Test_Task4
from deep_dating.preprocessing import PatchExtractor, PatchMethod, PreprocessRunner, ImageSplitter
from deep_dating.augmentation import AugDoc
from deep_dating.util import DATASETS_PATH
import matplotlib.pyplot as plt
from tqdm import tqdm


def preprocess_dating_cnn(dataset):
    print("Running patch extraction for ", dataset.name, "...")

    splitter = DatasetSplitter(dataset, 80, 400, test_size=0) #150 # 50
    preprocessor = PreprocessRunner(dataset.name)
    preprocessing_func = PatchExtractor(plot=False, method=PatchMethod.SLIDING_WINDOW_LINES, num_lines_per_patch=4).extract_patches

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


def preprocess_autoencoder():
    dataset = CLaMM(path=os.path.join(DATASETS_PATH, "CLaMM_Training_Clean"))
    splitter = DatasetSplitter(dataset, 80, 400, test_size=0)

    for i in range(1):
        binarize = i == 1
        name_ext = "_Bin" if binarize else ""

        preprocessor = PreprocessRunner(dataset.name, ext=("_Set_Auto" + name_ext))
        preprocessor_func = ImageSplitter(plot=False, binarize=binarize).split

        for set_type in [SetType.TRAIN, SetType.VAL]:
            X, y = splitter.get_data(set_type)
            preprocessor.run(X, y, set_type, preprocessor_func)
            print("Image splitting done for", set_type)


def preprocess_bin():
    dataset = BinDataset()

    for X_train, X_val, ext, padding_color in [(dataset.X_train, dataset.X_test, "_Set", 0), (dataset.y_train, dataset.y_test, "_Set_GT", 255)]:

        preprocessor = PreprocessRunner(dataset.name, ext=ext, include_old_name=False)
        preprocessing_func = PatchExtractor(method=PatchMethod.SLIDING_WINDOW, plot=False, padding_color=padding_color).extract_patches

        y = [0] * len(X_train)
        preprocessor.run(X_train, y, SetType.TRAIN, preprocessing_func)

        y = [0] * len(X_val)
        preprocessor.run(X_val, y, SetType.VAL, preprocessing_func)


def test_patch_extraction():
    BinDataset()
    #exit()

    dp = PatchExtractor(method=PatchMethod.SLIDING_WINDOW, padding_color=0)

    #for dataset in load_all_dating_datasets():
    import random
    #x = CLaMM_Test_Task4().X #(path=os.path.join(DATASETS_PATH, "CLaMM_Training_Clean")).X
    #random.shuffle(x)
    #for file in x:
    file = os.path.join("/home/nikolai/Downloads/datasets/dibco/2018_img_1/5.bmp")
    file = os.path.join("/home/nikolai/Downloads/datasets/aug_img_1/aug_29.png")
    dp.extract_patches(file)
    # plt.imshow(dp.img_bin, cmap="gray")
    # plt.show()
    dp.save_plot(show=True)
        #break


def test():
    c = CLaMM()
    c.save_to_dir(os.path.join(DATASETS_PATH, "clamm_visual"))


def run_aug_doc(n=60, test=False):
    
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
    #run_aug_doc()


    # parser = argparse.ArgumentParser()
    # parser.add_argument("--test", help="", nargs='?', default=None)
    # args = parser.parse_args()

    # if args.test:
    #test_patch_extraction()
    #test()
    #preprocess_dating_cnn(CLaMM(path=os.path.join(DATASETS_PATH, "CLaMM_Training_Clean")))
    #preprocess_autoencoder()
    # print(CLaMM_Test_Task3().size)
    # print(CLaMM_Test_Task4().size)
    #preprocess_dating_cnn_test()
    preprocess_bin()
    #test_patch_extraction()
    #preprocess_dating_cnn_test(CLaMM_Test_Task3())
    #test()
    #preprocess_dating_cnn(CLaMM(path=os.path.join(DATASETS_PATH, "CLaMM_Training_Clean")))
    #preprocess_autoencoder()
    #preprocess_dating_cnn(MPS())

