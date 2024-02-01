
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from deep_dating.preprocessing import PreprocessRunner
from deep_dating.util import save_figure, to_index


class DatingDataLoader(DataLoader):

    def __init__(self, dataset_name, set_type, model, batch_size=32, shuffle=True, num_workers=7, preprocess_ext="_Set"):
        self.model_batch_size = batch_size
        self.dataset_name = dataset_name
        self.torch_dataset = self.PytorchDatingDataset(dataset_name, set_type, model, preprocess_ext)
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

        def __init__(self, dataset_name, set_type, model, preprocess_ext):
            self.model = model
            self.X, self.y = PreprocessRunner(dataset_name, preprocess_ext).read_preprocessing_header(set_type)
            if self.model.classification:
                self.y, self.y_unique = to_index(self.y)
        
        def decode_class(self, class_idxs):
            return self.y_unique[class_idxs]

        def __getitem__(self, idx):
            img_path, img_date = self.X[idx], self.y[idx]
            img = self.model.transform_img(img_path)
            return img, img_date, img_path
        
        def __len__(self):
            return self.X.shape[0]
