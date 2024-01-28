
from deep_dating.datasets import DatasetName, DatingDataLoader, SetType
from deep_dating.networks import DatingCNN
from deep_dating.prediction import DatingPredictor
from sklearn.svm import SVC
import pickle


def get_features(model, dataset_name, model_timestamp, set_type):
    print("Getting features for", set_type)
    loader = DatingDataLoader(dataset_name, set_type, model)
    DatingPredictor().predict(model, loader, save_path=model_timestamp + f"_feats_{set_type.value}.pkl")


def run():
    model_timestamp = "runs/Jan6-22-21-16/model_epoch_28" #"runs/Dec21-16-31-47/model_epoch_8"#

    model = DatingCNN(model_name=DatingCNN.INCEPTION)
    model.load(model_timestamp + ".pt", continue_training=False, use_as_feat_extractor=True)

    for set_type in [SetType.TRAIN, SetType.VAL]:
        get_features(model, DatasetName.MPS, model_timestamp, set_type)


def train_output_classifier():
    svm = SVC()
    train_labels, train_outputs, train_paths = DatingPredictor().load("runs/Jan6-22-21-16/model_epoch_28_feats_train.pkl")
    val_labels, val_outputs, val_paths = DatingPredictor().load("runs/Jan6-22-21-16/model_epoch_28_feats_val.pkl")
    print(train_labels.shape)
    print(train_labels.shape)
    svm.fit(train_outputs, train_labels.flatten())

    #svm.predict(val_outputs)

if __name__ == "__main__":
    # run()
    train_output_classifier()