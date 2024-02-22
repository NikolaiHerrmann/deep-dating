
import torch
import os
from deep_dating.datasets import DatasetName, DatingDataLoader, SetType, CrossVal
from deep_dating.networks import DatingCNN, Autoencoder
from deep_dating.prediction import DatingPredictor
from deep_dating.preprocessing import PreprocessRunner
from deep_dating.util import get_torch_device
import numpy as np
import matplotlib.pyplot as plt
import cv2


if __name__ == "__main__":
    #model_path = "runs/Dec21-16-31-47/model_epoch_8.pt"
    model_path = "/home/nikolai/Downloads/model_epoch_58.pt" #"runs/Jan8-19-25-16/model_epoch_3.pt"

    model = Autoencoder()
    model.load(model_path, continue_training=True)

    img_path = "/home/nikolai/Downloads/test.png"
    input = [model.transform_img(x) for x in [img_path]]
    input = np.array(input)
    print(input.shape)
    input = torch.from_numpy(input)
    output = model(input).detach().numpy()

    img = output[0, 0, :, :]
    # print(np.unique(img))
    # img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    plt.imshow(img, cmap="gray")
    plt.show()

    print(output.shape)
    # predictor = DatingPredictor()

    # dataset = DatasetName.CLAMM

    # cross_val = CrossVal(dataset, preprocess_ext="_Set_Auto")

    # X_train, y_train, X_val, y_val = list(cross_val.get_split(n_splits=1))[0]

    # train_loader = DatingDataLoader(dataset, X_train, y_train, model)

    # model_file_name = os.path.basename(model_path).split(".")[0] + "_feats_train.pkl"
    # dirs = os.path.dirname(model_path)
    # save_path = os.path.join(dirs, model_file_name)
    
    # predictor.predict(model, train_loader, save_path=save_path)


    # val_loader = DatingDataLoader(dataset, X_val, y_val, model)

    # model_file_name = os.path.basename(model_path).split(".")[0] + "_feats_val.pkl"
    # dirs = os.path.dirname(model_path)
    # save_path = os.path.join(dirs, model_file_name)
    
    # predictor.predict(model, val_loader, save_path=save_path)



# if __name__ == "__main__":
#     #model_path = "runs/Dec21-16-31-47/model_epoch_8.pt"
#     #model_path = "runs/Feb15-19-21-40/model_epoch_0.pt" #"runs/Jan8-19-25-16/model_epoch_3.pt"
#     for model_path, set_ in [("runs/Feb15-19-21-40/model_epoch_0.pt", "_Set")]:#, ("runs/Feb18-10-52-17/model_epoch_1.pt", "_Set_Auto")]:

#         model = DatingCNN("inception_resnet_v2", num_classes=15)
#         model.load(model_path, continue_training=False, use_as_feat_extractor=True)
#         predictor = DatingPredictor()

#         dataset = DatasetName.CLAMM

#         X_test, y_test = PreprocessRunner(dataset, ext="_Set_Test").read_preprocessing_header(SetType.TEST)

#         # 

#         # cross_val = CrossVal(dataset, preprocess_ext=set_)

#         #X_train, y_train, X_val, y_val = list(cross_val.get_split(n_splits=1))[0]

#         train_loader = DatingDataLoader(dataset, X_test, y_test, model)

#         model_file_name = os.path.basename(model_path).split(".")[0] + "_feats_test.pkl"
#         dirs = os.path.dirname(model_path)
#         save_path = os.path.join(dirs, model_file_name)
        
#         predictor.predict(model, train_loader, save_path=save_path)


#         # val_loader = DatingDataLoader(dataset, X_val, y_val, model)

#         # model_file_name = os.path.basename(model_path).split(".")[0] + "_feats_val.pkl"
#         # dirs = os.path.dirname(model_path)
#         # save_path = os.path.join(dirs, model_file_name)
        
#         # predictor.predict(model, val_loader, save_path=save_path)