
import torch
import matplotlib.pyplot as plt
import numpy as np
from deep_dating.networks import Autoencoder
from deep_dating.util import save_figure, plt_clear
from deep_dating.datasets import MPS, ScribbleLens, CLaMM, CLaMM_Test_Task4
from deep_dating.preprocessing import PatchExtractor, PatchMethod
from deep_dating.prediction import AutoencoderPredictor
import random

# model = Autoencoder()
# model.load("runs/unet1/v2_model_epoch_168.pt", continue_training=False)

dataset = CLaMM_Test_Task4()
#extractor = PatchExtractor(method=PatchMethod.SLIDING_WINDOW, plot=True)
X = dataset.X
random.shuffle(X)
random.shuffle(X)
random.shuffle(X)
random.shuffle(X)
random.shuffle(X)

predictor = AutoencoderPredictor()

for x in X:
    #x = "/home/nikolai/Downloads/datasets/ICDAR2017_CLaMM_task2_task4/315556101_MS0118_0209.jpg"
    # patches = extractor.extract_patches(x)
    # print(len(patches))
    predictor.run(x, plot=True)
    continue
    exit()
    patches = [model.transform_input(x) for x in patches]
    patches = torch.from_numpy(np.array(patches))

    output = model(patches)
    output = output.cpu().detach().numpy()
    print(output.shape)
    
    img = np.zeros((extractor.padded_height, extractor.padded_width))
    
    #plt.show()

    patch_drawing_info = extractor.get_extra_draw_info()
    print(patch_drawing_info)
    
    # unique_colors = np.sort(np.unique(output))
    # cmap_full = plt.get_cmap("Spectral")
    # cmap = plt.get_cmap("Spectral", unique_colors.shape[0])
    # min_, max_ = np.min(output), np.max(output)
    # if true_label:
    #     min_, max_ = min(min_, true_label), max(max_, true_label)
    # norm = colors.Normalize(min_ - 1, max_ + 1)

    # # Draw each patch
    for i, (x, y, w, h) in enumerate(patch_drawing_info):

        img[y:y+h, x:x+w] = output[i, 0, :, :]

    img = img[0:extractor.height, 0:extractor.width]
    plt_clear()
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("BiNet")
    ax[2].imshow(extractor.img, cmap="gray")
    ax[2].set_title("Original")
    ax[1].imshow(extractor.img_bin, cmap="gray")
    ax[1].set_title("Sauvola")
    plt.show()
