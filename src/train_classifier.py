
from deep_dating.datasets import DatasetName, DatingDataLoader, SetType
from deep_dating.networks import DatingCNN
from deep_dating.prediction import DatingPredictor


def get_features(model, dataset_name, model_timestamp, set_type):
    print("Getting features for", set_type)
    loader = DatingDataLoader(dataset_name, set_type, model)
    DatingPredictor().predict(model, loader, save_path=model_timestamp + f"_feats_{set_type.value}.pkl")


if __name__ == "__main__":
    model_timestamp = "runs/Jan6-22-21-16/model_epoch_28" #"runs/Dec21-16-31-47/model_epoch_8"#

    model = DatingCNN(model_name=DatingCNN.INCEPTION)
    model.load(model_timestamp + ".pt", continue_training=False, use_as_feat_extractor=True)

    for set_type in [SetType.TRAIN, SetType.VAL]:
        get_features(model, DatasetName.MPS, model_timestamp, set_type)
