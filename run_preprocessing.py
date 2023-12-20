
import argparse
from src.deep_dating.preprocessing.dating_patch_extraction import PatchExtractor, PatchMethod
from src.deep_dating.datasets.dating_dataloader import MPS, CLaMM, ScribbleLens, SetType


def extract_patches(dataset):
    print("Running patch extraction for ", dataset.name, "...")

    extractor = PatchExtractor(plot=False, method=PatchMethod.SLIDING_WINDOW_LINES)

    for set_type in [SetType.TRAIN, SetType.VAL]:
        dataset.process_files(extractor.extract_patches, set_type)
        print("Patch extraction done for ", set_type)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("dataset", help="mps, claam, scribblelens")
    # args = parser.parse_args()

    mps = MPS()
    extract_patches(mps)
