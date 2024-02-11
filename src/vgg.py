import torch
from tqdm import tqdm
from deep_dating.util import get_torch_device
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
torch.multiprocessing.set_sharing_strategy('file_system')

device = get_torch_device(verbose=True)

model_transforms = transforms.Compose([transforms.ToTensor(),
                                       transforms.Resize(256, antialias=True),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                            std=[0.229, 0.224, 0.225])])

class PatchDataset(Dataset):

    def __init__(self, set_type):
        self.X, self.y = PreprocessRunner(DatasetName.MPS).read_preprocessing_header(set_type)

    def __getitem__(self, idx):
        img_path, img_date = self.X[idx], self.y[idx]
        img = Image.open(img_path)
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


if __name__ == "__main__":
    train_loader = DataLoader(PatchDataset(SetType.TRAIN), batch_size=32, num_workers=7)
    val_loader = DataLoader(PatchDataset(SetType.VAL), batch_size=32, num_workers=7)

    model = vgg19(weights=VGG19_Weights.DEFAULT)
    model.classifier = model.classifier[:-1]
    model.eval()

    extract_features(train_loader, SetType.TRAIN, model)
    extract_features(train_loader, SetType.VAL, model)