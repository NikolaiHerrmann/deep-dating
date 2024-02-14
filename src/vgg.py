import torch
from tqdm import tqdm
from deep_dating.util import get_torch_device, SEED
from torchvision.models import vgg19, VGG19_Weights
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from deep_dating.datasets import SetType, DatasetName
from deep_dating.preprocessing import PreprocessRunner
from torch import nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
torch.multiprocessing.set_sharing_strategy('file_system')


device = get_torch_device(verbose=True)

model_transforms = transforms.Compose([transforms.ToTensor(),
                                       transforms.Resize(256, antialias=True),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                            std=[0.229, 0.224, 0.225])])

class PatchDataset(Dataset):

    def __init__(self, set_type):
        self.X, self.y = PreprocessRunner(DatasetName.MPS, ext="_Set_Auto").read_preprocessing_header(set_type)

    def __getitem__(self, idx):
        img_path, img_date = self.X[idx], self.y[idx]
        img = Image.open(img_path).convert("RGB")
        img = model_transforms(img)
        return img, img_date, img_path

    def __len__(self):
        return len(self.y)


def extract_features(data_loader, set_type, model):
    all_outputs = []
    all_labels = []
    all_paths = []

    model.to(device)

    with torch.no_grad():
        for inputs, labels, paths in tqdm(data_loader):
            # plt.imshow(inputs[0, :].detach().numpy().transpose(1, 2, 0))
            # plt.show()
            
            inputs = inputs.to(device)
            outputs = model(inputs)
            # print(outputs)
            # print(np.sum(outputs[0, :].detach().numpy()))
            # print(outputs.shape)
            # exit()
            all_outputs.append(outputs.cpu().detach().numpy())
            all_labels.append(labels)
            all_paths += list(paths)

    with open(f"vgg19_feats_{str(set_type)}", "wb") as f:
        pickle.dump((all_labels, all_outputs, all_paths), f)


def read():
    with open("vgg19_feats_SetType.TRAIN", "rb") as f:
        all_labels, all_outputs, all_paths = pickle.load(f)

    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)

    print(all_outputs.shape)
    print(len(all_paths))

    preds = {}

    for i, img_name in enumerate(all_paths):
        img_name = PreprocessRunner.get_base_img_name(img_name)
        if not img_name in preds:
            preds[img_name] = [all_labels[i], [all_outputs[i]]]
        else:
            preds[img_name][1].append(all_outputs[i])

    features = []
    for key, val in preds.items():
        preds[key][1] = np.mean(preds[key][1], axis=0)
        features.append(preds[key][1])
        # print(preds[key][1].shape)
        # exit()
    features = np.array(features)

    print(features.shape)
    pca = PCA(n_components=np.min(features.shape), random_state=SEED)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_pca = pca.fit_transform(features)

    n = 1000
    x = np.arange(1, pca.n_components_ + 1)[:n]
    var_explained = (pca.explained_variance_ratio_ * 100)[:n]
    plt.plot(x, var_explained, "o-")
    print("First", n, "components:", var_explained)
    plt.xticks(x)
    plt.xlabel("Feature Dimension")
    plt.ylabel("Percentage of Explained Variance")
    plt.title(f"Scree Plot Showing First {n} Dimensions")
    plt.show()



if __name__ == "__main__":
    train_loader = DataLoader(PatchDataset(SetType.TRAIN), batch_size=32, num_workers=7)
    val_loader = DataLoader(PatchDataset(SetType.VAL), batch_size=32, num_workers=7)

    model = vgg19(weights=VGG19_Weights.DEFAULT)
    model.classifier = model.classifier[:-1]
    model.eval()

    extract_features(train_loader, SetType.TRAIN, model)
    extract_features(val_loader, SetType.VAL, model)
    #read()