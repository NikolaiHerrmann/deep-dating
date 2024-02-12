
import os
from deep_dating.datasets import load_all_dating_datasets, SetType, DatasetSplitter, MPS, CLaMM, ScribbleLens, CLaMM_Test_Task3, CLaMM_Test_Task4
from deep_dating.preprocessing import PatchExtractor, PatchMethod, PreprocessRunner, ImageSplitter
from deep_dating.util import DATASETS_PATH
import matplotlib.pyplot as plt

def preprocess_dating_cnn(dataset):
    print("Running patch extraction for ", dataset.name, "...")

    splitter = DatasetSplitter(dataset, None, None) #150 # 50
    preprocessor = PreprocessRunner(dataset.name)
    preprocessing_func = PatchExtractor(plot=False, method=PatchMethod.SLIDING_WINDOW_LINES, num_lines_per_patch=4).extract_patches

    for set_type in [SetType.TRAIN, SetType.VAL]:
        X, y = splitter.get_data(set_type)
        if X is not None:
            preprocessor.run(X, y, set_type, preprocessing_func)
            print("Patch extraction done for", set_type)
        else:
            print("Skipping", set_type)


def preprocess_autoencoder():
    dataset = CLaMM(path=os.path.join(DATASETS_PATH, "CLaMM_Training_Clean"))
    dataset = MPS()
    splitter = DatasetSplitter(dataset, None, None, test_size=0)

    for i in range(2):
        binarize = i == 1
        name_ext = "_Bin" if binarize else ""

        preprocessor = PreprocessRunner(dataset.name, ext=("_Set_Auto" + name_ext))
        preprocessor_func = ImageSplitter(plot=False, binarize=binarize).split

        for set_type in [SetType.TRAIN, SetType.VAL]:
            X, y = splitter.get_data(set_type)
            preprocessor.run(X, y, set_type, preprocessor_func)
            print("Image splitting done for", set_type)


def test_patch_extraction():
    dp = PatchExtractor(method=PatchMethod.SLIDING_WINDOW_LINES, plot=True, drop_out_rate=0, num_lines_per_patch=8)

    #for dataset in load_all_dating_datasets():
    import random
    x = CLaMM().X #(path=os.path.join(DATASETS_PATH, "CLaMM_Training_Clean")).X
    #random.shuffle(x)
    for file in x:
        dp.extract_patches(file)
        # plt.imshow(dp.img_bin, cmap="gray")
        # plt.show()
        dp.save_plot(show=True)
            #break


def test():
    c = CLaMM()
    c.save_to_dir(os.path.join(DATASETS_PATH, "clamm_visual"))


if __name__ == "__main__":
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
    #test_patch_extraction()
    #test()
    #preprocess_dating_cnn(CLaMM(path=os.path.join(DATASETS_PATH, "CLaMM_Training_Clean")))
    preprocess_autoencoder()
    #preprocess_dating_cnn(MPS())

