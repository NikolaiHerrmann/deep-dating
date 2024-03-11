
import os
import numpy as np
import string
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from deep_dating.datasets import SetType
from deep_dating.util import save_figure, DATASETS_PATH, plt_clear
from deep_dating.prediction import AutoencoderPredictor
from deep_dating.preprocessing import PatchExtractor, PatchMethod


def pipeline_compare(p1_metrics, p2_metrics, p1p2_metrics):
    """
    Ordering is:
    mae         mse       cs_0      cs_25      cs_50      cs_75     cs_100
    """
    
    pipeline_names = ["Text Pipeline", "Layout Pipeline", "Pipeline Concat"]
    pipeline_metrics = [p1_metrics, p2_metrics, p1p2_metrics]

    for name, metrics in zip(pipeline_names, pipeline_metrics):
        means, stds = metrics
        print(means.to_numpy()[2:])
        #print(means[means.columns[-4:]])
        plt.plot([0, 25, 50, 75, 100], means.to_numpy()[2:], label=name)

    plt.show()


def binet_loss(file):
    df = pd.read_csv(file)

    _, ax = plt.subplots(figsize=(6.6, 5))
    
    for set_type, label_name in [("train", "Training Set:\n$\it{H-DIBCO \; 20[09, 10, 11, 12, 13, 14, 17, 18, 19] + Synthetic}$"), 
                                 ("eval", "Validation Set:\n$\it{H-DIBCO \; 2016}$")]:
        df_set = df[df["set_type"] == set_type]
        loss = df_set["mean_loss"]
        epochs = df_set["epoch"]
        ax.plot(epochs, loss, label=label_name)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("L1 (Mean Absolute Error) Loss")

    plt.legend()
    
    save_figure("binet_aug_norm_loss_curve", show=True)


def binet_compare():

    img_ls = [("MPS/Download/1500/MPS1500_0144.ppm", "MPS", True, (1294, 1835)), 
              ("MPS/Download/1400/MPS1400_0092.ppm", "MPS", True, None),
              ("CLaMM_Training_Clean/IRHT_P_005976.tif", "CLaMM", False, None),
              ("CLaMM_Training_Clean/IRHT_P_000455.tif", "CLaMM", False, None),
              ("scribblelens.supplement.original.pages/scribblelens.corpus.v1/nl/unsupervised/ridderschap/71/originalpage.NL-HaNA_1.04.02_5058_0071.jpg", "Scribble", False, None)
             ]
              
    def plot(gray_img, aug_img, non_aug_img, sauvola_img, otsu_img, i, crop):
        plt_clear()
        fig, ax = plt.subplots(1, 5, figsize=(10, 7))

        plot_ids = string.ascii_lowercase
        
        ax[0].imshow(gray_img, cmap="gray")
        ax[0].set_title(f"{plot_ids[i]}) {dataset_name} Grayscale")
        ax[1].imshow(aug_img, cmap="gray")
        ax[1].set_title("BiNet + Synthetic")
        ax[2].imshow(non_aug_img, cmap="gray")
        ax[2].set_title("BiNet")
        ax[3].imshow(sauvola_img, cmap="gray")
        ax[3].set_title("Sauvola")
        ax[4].imshow(otsu_img, cmap="gray")
        ax[4].set_title("Otsu")

        for x in ax:
            x.set_xticks([])
            x.set_yticks([])

        plt.tight_layout()
        crop = "_c" if crop else ""
        save_figure(f"binet_{dataset_name}_{i}{crop}", show=False)

    for i, (img_path, dataset_name, normalize, crop) in tqdm(enumerate(img_ls), total=len(img_ls)):
        img_1 = os.path.join(DATASETS_PATH, img_path)
        
        aug_model = AutoencoderPredictor(normalize_per_img=normalize, model_path="runs_v2/Binet_aug_norm/model_epoch_275_split_0.pt")
        aug_img, gray_img, sauvola_img, otsu_img = aug_model.run(img_1, plot=True, extra_output=True, show=False)
        
        non_aug_img = AutoencoderPredictor(normalize_per_img=normalize, model_path="runs_v2/Binet_norm/model_epoch_329_split_0.pt").run(img_1)

        if crop:
            start, end = crop
            aug_img_crop = aug_img[:, start:end]
            non_aug_img_crop = non_aug_img[:, start:end]
            gray_img_crop = gray_img[:, start:end]
            sauvola_img_crop = sauvola_img[:, start:end]
            otsu_img_crop = otsu_img[:, start:end]
            plot(gray_img_crop, aug_img_crop, non_aug_img_crop, sauvola_img_crop, otsu_img_crop, i, True)

        plot(gray_img, aug_img, non_aug_img, sauvola_img, otsu_img, i, False)


