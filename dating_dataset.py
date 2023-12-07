
import glob 
import os
from torch.utils.data import Dataset

class Dataset(Dataset):

    def __init__(self, root_dir):
        self.img_list = glob.glob(os.path.join(root_dir, "*", "*"))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # img_path = self.img_list[idx]
        # date = float(img_path.split("/")[3])

        # img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(256)])
        # img = transform(img)

        # return img, torch.tensor(date, dtype=torch.float32) 
        return None
    

#if __name__ == "__main__":
