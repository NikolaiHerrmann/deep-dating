
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from deep_dating.preprocessing import PreprocessRunner
from deep_dating.networks import ModelType
from deep_dating.util import save_figure, to_index


class DatingDataLoader(DataLoader):

    def __init__(self, dataset_name, X, y, model, batch_size=32, shuffle=True, num_workers=7):
        self.model_batch_size = batch_size
        self.dataset_name = dataset_name
        self.torch_dataset = self.PytorchDatingDataset(X, y, model)
        super().__init__(self.torch_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        
    def test_loading(self):
        images, labels, _ = next(iter(self))
        num_cells = int(np.ceil(np.sqrt(self.model_batch_size)))
        fig, axs = plt.subplots(num_cells, num_cells)
        
        for i in range(self.model_batch_size):
            a = axs[i % num_cells, i // num_cells]
            a.imshow(images[i, :], cmap="gray")
            a.set_title(f"Date: {labels[i]}")

        fig.tight_layout()
        save_figure("example_batch", show=True)

    class PytorchDatingDataset(Dataset):

        def __init__(self, X, y, model):
            self.X = X
            self.y = y
            self.model = model
            if self.model.classification:
                self.y, self.y_unique = to_index(self.y)
        
        def decode_class(self, class_idxs):
            return self.y_unique[class_idxs]
        
        def get_class_dict(self):
            return {idx : int(date) for idx, date in enumerate(self.y_unique)}

        def __getitem__(self, idx):
            img_path, y = self.X[idx], self.y[idx]
            img = self.model.transform_img(img_path)

            if self.model.model_type == ModelType.AUTOENCODER:
                if self.model.training:
                    y = self.model.transform_img(y)

            return img, y, img_path
        
        def __len__(self):
            return self.X.shape[0]
