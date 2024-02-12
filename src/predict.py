
import torch
import os
from deep_dating.datasets import DatasetName, DatingDataLoader, SetType, CrossVal
from deep_dating.networks import DatingCNN, Autoencoder
from deep_dating.prediction import DatingPredictor
from deep_dating.util import get_torch_device


if __name__ == "__main__":
    #model_path = "runs/Dec21-16-31-47/model_epoch_8.pt"
    model_path = "runs/auto/model_epoch_24.pt" #"runs/Jan8-19-25-16/model_epoch_3.pt"

    model = Autoencoder()
    model.load(model_path, continue_training=False)
    predictor = DatingPredictor()

    dataset = DatasetName.MPS

    cross_val = CrossVal(dataset, preprocess_ext="_Set_Auto")

    X_train, y_train, X_val, y_val = list(cross_val.get_split(n_splits=1))[0]

    train_loader = DatingDataLoader(dataset, X_train, y_train, model)

    model_file_name = os.path.basename(model_path).split(".")[0] + "_feats_train_mps.pkl"
    dirs = os.path.dirname(model_path)
    save_path = os.path.join(dirs, model_file_name)
    
    predictor.predict(model, train_loader, save_path=save_path)


    val_loader = DatingDataLoader(dataset, X_val, y_val, model)

    model_file_name = os.path.basename(model_path).split(".")[0] + "_feats_val_mps.pkl"
    dirs = os.path.dirname(model_path)
    save_path = os.path.join(dirs, model_file_name)
    
    predictor.predict(model, val_loader, save_path=save_path)



# if __name__ == "__main__":
#     #model_path = "runs/Dec21-16-31-47/model_epoch_8.pt"
#     model_path = "runs/Feb2-13-25-11/model_epoch_4.pt" #"runs/Jan8-19-25-16/model_epoch_3.pt"

#     model = DatingCNN("inception_resnet_v2", num_classes=11)
#     model.load(model_path, continue_training=False, use_as_feat_extractor=True)
#     predictor = DatingPredictor()

#     dataset = DatasetName.MPS

#     cross_val = CrossVal(dataset)

#     X_train, y_train, X_val, y_val = list(cross_val.get_split(n_splits=1))[0]

#     train_loader = DatingDataLoader(dataset, X_train, y_train, model)

#     model_file_name = os.path.basename(model_path).split(".")[0] + "_feats_train.pkl"
#     dirs = os.path.dirname(model_path)
#     save_path = os.path.join(dirs, model_file_name)
    
#     predictor.predict(model, train_loader, save_path=save_path)


#     val_loader = DatingDataLoader(dataset, X_val, y_val, model)

#     model_file_name = os.path.basename(model_path).split(".")[0] + "_feats_val.pkl"
#     dirs = os.path.dirname(model_path)
#     save_path = os.path.join(dirs, model_file_name)
    
#     predictor.predict(model, val_loader, save_path=save_path)