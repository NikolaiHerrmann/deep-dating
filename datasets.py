
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import os
import cv2


class MPSDataset(Dataset):

    def __init__(self, root_dir):
        self.img_list = glob.glob(os.path.join(root_dir, "*", "*"))
        assert len(self.img_list) != 0, "Empty directory!"

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        date = float(os.path.basename(img_path).split("_")[0][3:])

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        crop_size = min(550, min(img.shape))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.RandomCrop(crop_size),
                                        transforms.Resize(256, antialias=False)])
                                        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        img = transform(img)

        return img, torch.tensor(date, dtype=torch.float32) 