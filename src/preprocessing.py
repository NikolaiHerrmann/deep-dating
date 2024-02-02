
import argparse
from deep_dating.datasets import load_all_dating_datasets, SetType, DatasetSplitter, MPS, CLaMM, ScribbleLens, CLaMM_Test_Task3, CLaMM_Test_Task4
from deep_dating.preprocessing import PatchExtractor, PatchMethod, PreprocessRunner, ImageSplitter


def preprocess_dating_cnn(dataset):
    print("Running patch extraction for ", dataset.name, "...")

    splitter = DatasetSplitter(dataset, 166, 166) #150 # 50
    preprocessor = PreprocessRunner(dataset.name)
    preprocessing_func = PatchExtractor(plot=False, method=PatchMethod.SLIDING_WINDOW_LINES).extract_patches

    for set_type in [SetType.TRAIN, SetType.VAL, SetType.TEST]:
        X, y = splitter.get_data(set_type)
        preprocessor.run(X, y, set_type, preprocessing_func)
        print("Patch extraction done for", set_type)


def preprocess_autoencoder():
    dataset = CLaMM()
    
    splitter = DatasetSplitter(dataset, test_size=0, val_size=0.2)
    preprocessor = PreprocessRunner(dataset.name, ext="_Set_Auto")
    preprocessor_func = ImageSplitter(plot=False).split

    for set_type in [SetType.TRAIN, SetType.VAL]:
        X, y = splitter.get_data(set_type)
        preprocessor.run(X, y, set_type, preprocessor_func)
        print("Image splitting done for", set_type)


def test_patch_extraction():
    dp = PatchExtractor(method=PatchMethod.SLIDING_WINDOW_LINES)

    #for dataset in load_all_dating_datasets():
    import random
    x = CLaMM().X
    random.shuffle(x)
    for file in x:
        dp.extract_patches(file)
        dp.save_plot(show=True)
            #break


def test():
    d = ScribbleLens()
    print(d.writer_ids_per_date)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--test", help="", nargs='?', default=None)
    # args = parser.parse_args()

    # if args.test:
    #test_patch_extraction()
    test()
    preprocess_dating_cnn(ScribbleLens())
    #preprocess_autoencoder()
    # print(CLaMM_Test_Task3().size)
    # print(CLaMM_Test_Task4().size)

