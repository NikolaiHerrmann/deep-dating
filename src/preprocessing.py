
import argparse
from deep_dating.datasets import load_all_dating_datasets, SetType, MPS, CLaMM, ScribbleLens
from deep_dating.preprocessing import PatchExtractor, PatchMethod


def extract_patches(dataset):
    print("Running patch extraction for ", dataset.name, "...")

    extractor = PatchExtractor(plot=False, method=PatchMethod.SLIDING_WINDOW_LINES)

    for set_type in [SetType.TRAIN, SetType.VAL]:
        dataset.process_files(extractor.extract_patches, set_type)
        print("Patch extraction done for ", set_type)


def test_patch_extraction():
    dp = PatchExtractor(method=PatchMethod.SLIDING_WINDOW_LINES)

    for dataset in load_all_dating_datasets():
        files = dataset.img_names
        for file in files:
            dp.extract_patches(file)
            dp.save_plot(show=True)
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="", nargs='?', default=None)
    args = parser.parse_args()

    if args.test:
        test_patch_extraction()

