
import os
import glob
from deep_dating.datasets import DatasetName, DatingDataLoader, SetType, CrossVal
from deep_dating.networks import DatingCNN, DatingTrainer, Autoencoder, DatingClassifier
from deep_dating.prediction import DatingPredictor
from deep_dating.util import serialize


def run_dating_cnn_predictions(model, model_path, train_loader, val_loader, split):
    predictor = DatingPredictor()

    model.use_as_feature_extractor()

    if model_path is None:
        model_path = ""
        print("Warning model path was none!")

    for loader, set_type in [(train_loader, SetType.TRAIN), (val_loader, SetType.VAL)]:
        
        model_file_name = os.path.basename(model_path).rsplit(".", 1)[0] + f"_feats_{set_type.value}_split_{split}.pkl"
        save_path = os.path.join(os.path.dirname(model_path), model_file_name)
        
        predictor.predict(model, loader, save_path=save_path)


def train_dating_cnn():
    dataset = DatasetName.SCRIBBLE
    num_classes = 6

    cross_val = CrossVal(dataset, preprocess_ext="_Set_P2_299")
    trainer = DatingTrainer("P2 Scribble", num_epochs=50, patience=6)
    n_splits = 5
    batch_size = 32

    # leave empty, just in case program crashes and need to re-run
    avoid_splits = [] 

    for i, (X_train, y_train, X_val, y_val) in enumerate(cross_val.get_split(n_splits=n_splits)):
        
        if i in avoid_splits:
            print(f"Avoiding split: {i + 1}")
            continue

        print(f" -- Running split: {i+1}/{n_splits} -- ")

        model = DatingCNN(model_name=DatingCNN.INCEPTION, num_classes=num_classes, dropout=True)
        
        train_loader = DatingDataLoader(dataset, X_train, y_train, model, batch_size=batch_size)
        val_loader = DatingDataLoader(dataset, X_val, y_val, model, batch_size=batch_size)

        trainer.train(model, train_loader, val_loader, i)

        run_dating_cnn_predictions(model, trainer.best_model_path, train_loader, val_loader, i)


def test_dating_cnn():
    dataset_name = DatasetName.MPS
    pipeline = "P2"
    num_classes = 11
    
    n_splits = 5
    batch_size = 32
    run_path = "runs_v2"    

    path = os.path.join(run_path, f"{str(dataset_name)}_{pipeline}_Crossval")

    ext = "_Set_P1_Bin_299_Test" if pipeline == "P1" else "_Set_P2_299_Test"

    cross_val = CrossVal(dataset_name, test=True, preprocess_ext=ext)
    X_test, y_test = cross_val.get_test()

    for i in range(n_splits):
        print(f"Running test split: {i+1}/{n_splits}")
        
        model = DatingCNN(model_name=DatingCNN.INCEPTION, num_classes=num_classes, dropout=True)
        model_path = sorted(glob.glob(os.path.join(path, f"model_epoch_*_split_{i}.pt")), reverse=True)[0]
        model.load(model_path, continue_training=False, use_as_feat_extractor=True)

        test_loader = DatingDataLoader(dataset_name, X_test, y_test, model, batch_size=batch_size, shuffle=False)

        predictor = DatingPredictor()

        feat_save_path = model_path.rsplit(".", 1)[0] + f"_feats_{SetType.TEST.value}_split_{i}.pkl"
            
        predictor.predict(model, test_loader, save_path=feat_save_path)


def train_autoencoder():
    dataset = DatasetName.DIBCO

    cross_val = CrossVal(dataset, preprocess_ext="_Set_No_Aug")
    cross_val_gt = CrossVal(dataset, preprocess_ext="_Set_GT_No_Aug")
    trainer = DatingTrainer("train binet with NOOO aug and new normalization", num_epochs=400, patience=50)

    (X_train, y_train, X_val, y_val) = next(cross_val.get_split(n_splits=1))
    (X_train_gt, y_train_gt, X_val_gt, y_val_gt) = next(cross_val_gt.get_split(n_splits=1))      

    model = Autoencoder()

    train_loader = DatingDataLoader(dataset, X_train, X_train_gt, model)
    val_loader = DatingDataLoader(dataset, X_val, X_val_gt, model)

    trainer.train(model, train_loader, val_loader, 0)


def train_classifier():
    n_splits = 5
    dataset_name = DatasetName.MPS
    run_path = "runs_v2"

    p1_path = os.path.join(run_path, f"{str(dataset_name)}_P1_Crossval")
    p2_path = os.path.join(run_path, f"{str(dataset_name)}_P2_Crossval")

    #p1_metrics = DatingClassifier().cross_val(p1_path, n_splits=n_splits)
    p1_metrics = DatingClassifier().cross_val(p1_path, n_splits=n_splits, train=False)
    #p2_metrics = DatingClassifier().cross_val(p2_path, n_splits=n_splits)
    #p1p2_metrics = DatingClassifier().cross_val(p1_path, dir_2=p2_path, n_splits=n_splits)

    # pipelines = [p1_metrics, p2_metrics, p1p2_metrics]
    # serialize(pipelines, "runs_v2/graphs/pipeline_results.pkl")


if __name__ == "__main__":
    test_dating_cnn()
    #train_dating_cnn()
    #train_autoencoder()
    #train_classifier()
