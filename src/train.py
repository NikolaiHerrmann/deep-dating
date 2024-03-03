
import os
from deep_dating.datasets import DatasetName, DatingDataLoader, SetType, CrossVal
from deep_dating.networks import DatingCNN, DatingTrainer, Autoencoder, DatingClassifier
from deep_dating.prediction import DatingPredictor


def run_dating_cnn_predictions(model, model_path, train_loader, val_loader, split):
    predictor = DatingPredictor()

    model.use_as_feature_extractor()

    for loader, set_type in [(train_loader, SetType.TRAIN), (val_loader, SetType.VAL)]:
        
        model_file_name = os.path.basename(model_path).rsplit(".", 1)[0] + f"_feats_{set_type.value}_split_{split}.pkl"
        save_path = os.path.join(os.path.dirname(model_path), model_file_name)
        
        predictor.predict(model, loader, save_path=save_path)


def train_dating_cnn():
    dataset = DatasetName.MPS

    cross_val = CrossVal(dataset, preprocess_ext="_Set_P2_299")
    trainer = DatingTrainer("Inception for P2 for MPS with cross-val", num_epochs=50, patience=6, exp_name="Mar3-10-15-23")
    n_splits = 5
    batch_size = 32

    # leave empty, just in case program crashes and need to re-run
    avoid_splits = [0] 

    for i, (X_train, y_train, X_val, y_val) in enumerate(cross_val.get_split(n_splits=n_splits)):
        
        if i in avoid_splits:
            print(f"Avoiding split: {i + 1}")
            continue

        print(f" -- Running split: {i+1}/{n_splits} -- ")

        model = DatingCNN(model_name=DatingCNN.INCEPTION, num_classes=11, dropout=True)
        
        train_loader = DatingDataLoader(dataset, X_train, y_train, model, batch_size=batch_size)
        val_loader = DatingDataLoader(dataset, X_val, y_val, model, batch_size=batch_size)

        trainer.train(model, train_loader, val_loader, i)

        run_dating_cnn_predictions(model, trainer.best_model_path, train_loader, val_loader, i)


def train_autoencoder():
    dataset = DatasetName.DIBCO

    cross_val = CrossVal(dataset, preprocess_ext="_Set_Aug")
    cross_val_gt = CrossVal(dataset, preprocess_ext="_Set_GT_Aug")
    trainer = DatingTrainer("train binet without aug", num_epochs=400, patience=50)

    (X_train, y_train, X_val, y_val) = next(cross_val.get_split(n_splits=1))
    (X_train_gt, y_train_gt, X_val_gt, y_val_gt) = next(cross_val_gt.get_split(n_splits=1))      

    model = Autoencoder()

    train_loader = DatingDataLoader(dataset, X_train, X_train_gt, model)
    val_loader = DatingDataLoader(dataset, X_val, X_val_gt, model)

    trainer.train(model, train_loader, val_loader, 0)


def train_classifier():
    classifier = DatingClassifier()

    classifier.cross_val("runs_v2")

if __name__ == "__main__":
    #train_dating_cnn()
    train_autoencoder()
    #train_classifier()
