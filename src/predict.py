
import torch
import os
from deep_dating.datasets import DatasetName, DatingDataLoader, SetType, CrossVal
from deep_dating.networks import DatingCNN, Autoencoder
from deep_dating.prediction import DatingPredictor
from deep_dating.util import get_torch_device


# if __name__ == "__main__":
#     #model_path = "runs/Dec21-16-31-47/model_epoch_8.pt"
#     model_path = "runs/auto_2/model_epoch_38.pt" #"runs/Jan8-19-25-16/model_epoch_3.pt"

#     model = Autoencoder()
#     model.load(model_path, continue_training=False)
#     predictor = DatingPredictor()

#     dataset = DatasetName.MPS

#     cross_val = CrossVal(dataset, preprocess_ext="_Set_Auto")

#     X_train, y_train, X_val, y_val = list(cross_val.get_split(n_splits=1))[0]

#     train_loader = DatingDataLoader(dataset, X_train, y_train, model)

#     model_file_name = os.path.basename(model_path).split(".")[0] + "_feats_train_mps.pkl"
#     dirs = os.path.dirname(model_path)
#     save_path = os.path.join(dirs, model_file_name)
    
#     predictor.predict(model, train_loader, save_path=save_path)


#     val_loader = DatingDataLoader(dataset, X_val, y_val, model)

#     model_file_name = os.path.basename(model_path).split(".")[0] + "_feats_val_mps.pkl"
#     dirs = os.path.dirname(model_path)
#     save_path = os.path.join(dirs, model_file_name)
    
#     predictor.predict(model, val_loader, save_path=save_path)



if __name__ == "__main__":
    #model_path = "runs/Dec21-16-31-47/model_epoch_8.pt"
    #model_path = "runs/Feb15-19-21-40/model_epoch_0.pt" #"runs/Jan8-19-25-16/model_epoch_3.pt"
    for model_path, set_ in [("runs/Feb15-19-21-40/model_epoch_0.pt", "_Set"), ("runs/Feb18-10-52-17/model_epoch_1.pt", "_Set_Auto")]:

        model = DatingCNN("inception_resnet_v2", num_classes=15)
        model.load(model_path, continue_training=False, use_as_feat_extractor=True)
        predictor = DatingPredictor()

        dataset = DatasetName.CLAMM

        cross_val = CrossVal(dataset, preprocess_ext=set_)

        X_train, y_train, X_val, y_val = list(cross_val.get_split(n_splits=1))[0]

        train_loader = DatingDataLoader(dataset, X_train, y_train, model)

        model_file_name = os.path.basename(model_path).split(".")[0] + "_feats_train.pkl"
        dirs = os.path.dirname(model_path)
        save_path = os.path.join(dirs, model_file_name)
        
        predictor.predict(model, train_loader, save_path=save_path)


        val_loader = DatingDataLoader(dataset, X_val, y_val, model)

        model_file_name = os.path.basename(model_path).split(".")[0] + "_feats_val.pkl"
        dirs = os.path.dirname(model_path)
        save_path = os.path.join(dirs, model_file_name)
        
        predictor.predict(model, val_loader, save_path=save_path)