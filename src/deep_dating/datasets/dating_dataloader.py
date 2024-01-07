
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from deep_dating.preprocessing import Preprocessor
from deep_dating.util import save_figure


class DatingDataLoader(DataLoader):

    def __init__(self, dataset_name, set_type, model, batch_size=32, shuffle=True, num_workers=7):
        self.model_batch_size = batch_size
        super().__init__(self.PytorchDatingDataset(dataset_name, set_type, model.input_size), 
                         batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        
    def test_loading(self):
        images, labels = next(iter(self))
        num_cells = int(np.ceil(np.sqrt(self.model_batch_size)))
        fig, axs = plt.subplots(num_cells, num_cells)
        
        for i in range(self.model_batch_size):
            a = axs[i % num_cells, i // num_cells]
            a.imshow(images[i][0], cmap="gray")
            a.set_title(f"Date: {labels[i]}")

        fig.tight_layout()
        save_figure("example_batch", show=True)

    class PytorchDatingDataset(Dataset):

        def __init__(self, dataset_name, set_type, img_size):
            self.X, self.y = Preprocessor(dataset_name).read_preprocessing_header(set_type)
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Resize(img_size, antialias=True),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        def __getitem__(self, idx):
            img_path, img_date = self.X[idx], self.y[idx]
            
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = self.transform(img)

            return img, img_date, img_path
        
        def __len__(self):
            return self.X.shape[0]





# if __name__ == "__main__":
#     from dating_patch_extraction import PatchExtractor, PatchMethod

#     # for dataset in load_all_dating_datasets():
#     #     print(dataset)

#     mps = MPS()

#     dp = PatchExtractor(plot=False, method=PatchMethod.SLIDING_WINDOW_LINES)
#     mps.process_files(dp.extract_patches, SetType.VAL)

#     # pytorch_dataset = DatingDataLoader.PytorchDatingDataset(mps, SetType.VAL)
#     # print(pytorch_dataset.X.shape, pytorch_dataset.y.shape)
#     # print(pytorch_dataset.X)

