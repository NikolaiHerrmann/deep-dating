
import argparse
from deep_dating.datasets import load_all_dating_datasets, SetType, DatasetSplitter, MPS, CLaMM, ScribbleLens
from deep_dating.preprocessing import PatchExtractor, PatchMethod, Preprocessor


def preprocess(dataset):
    print("Running patch extraction for ", dataset.name, "...")

    splitter = DatasetSplitter(dataset, 50, 50) #220
    preprocessor = Preprocessor(dataset.name)
    preprocessing_func = PatchExtractor(plot=False, method=PatchMethod.SLIDING_WINDOW_LINES).extract_patches

    for set_type in [SetType.TRAIN, SetType.VAL, SetType.TEST]:
        X, y = splitter.get_data(set_type)
        preprocessor.run(X, y, set_type, preprocessing_func)
        print("Patch extraction done for ", set_type)


def test_patch_extraction():
    dp = PatchExtractor(method=PatchMethod.SLIDING_WINDOW_LINES)

    #for dataset in load_all_dating_datasets():
    for file in ScribbleLens().X:
        dp.extract_patches(file)
        dp.save_plot(show=True)
            #break


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--test", help="", nargs='?', default=None)
    # args = parser.parse_args()

    # if args.test:
    #test_patch_extraction()
    preprocess(ScribbleLens())

