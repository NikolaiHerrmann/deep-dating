
import cv2
import numpy as np
import matplotlib.pyplot as plt
from deep_dating.networks import Autoencoder
from deep_dating.preprocessing import PatchExtractor, PatchMethod
from deep_dating.util import plt_clear
from torch.utils.data import Dataset, DataLoader


class AutoencoderPredictor:

    class PatchDataset(Dataset):

        def __init__(self, patches, model):
            self.patches = patches
            self.model = model

        def __getitem__(self, idx):
            patch = self.patches[idx]
            return self.model.transform_input(patch)

        def __len__(self):
            return len(self.patches)

    def __init__(self, model_path="runs/unet1/v2_model_epoch_168.pt"):
        self.model = Autoencoder()
        self.model.load(model_path, continue_training=False)
        self.extractor = PatchExtractor(method=PatchMethod.SLIDING_WINDOW, plot=False)
        
    def run(self, img_path, plot=True):
        patches = self.extractor.extract_patches(img_path, plot=plot)

        data_loader = DataLoader(self.PatchDataset(patches, self.model), batch_size=32, shuffle=False, num_workers=7)

        outputs = []

        for input in data_loader:
            output = self.model(input)
            output = output.cpu().detach().numpy()
            outputs.append(output)

        outputs = np.concatenate(outputs)
          
        img = np.zeros((self.extractor.padded_height, self.extractor.padded_width))

        patch_drawing_info = self.extractor.get_extra_draw_info()

        for i, (x, y, w, h) in enumerate(patch_drawing_info):
            img[y:y+h, x:x+w] = outputs[i, 0, :, :]

        img = img[0:self.extractor.height, 0:self.extractor.width]
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
        img = img.astype(np.uint8)

        if plot:
            plt_clear()
            fig, ax = plt.subplots(2, 2)
            
            ax[0, 0].imshow(img, cmap="gray")
            ax[0, 0].set_title("BiNet")
            ax[0, 1].imshow(self.extractor.img, cmap="gray")
            ax[0, 1].set_title("Original Grayscale")
            ax[1, 0].imshow(self.extractor.img_bin, cmap="gray")
            ax[1, 0].set_title("Sauvola")
            ax[1, 1].imshow(self.extractor.img_bin_otsu, cmap="gray")
            ax[1, 1].set_title("Otsu")

            fig.tight_layout()
            plt.show()

        assert img.shape == self.extractor.img.shape, "Converted image not of original size!"
        return img