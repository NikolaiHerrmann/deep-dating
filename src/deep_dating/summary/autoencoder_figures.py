
import os
import string
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from deep_dating.util import save_figure, DATASETS_PATH, plt_clear, remove_ticks
from deep_dating.prediction import BiNetPredictor
from deep_dating.augmentation import AugDoc


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

        remove_ticks(ax)

        plt.tight_layout()
        crop = "_c" if crop else ""
        save_figure(f"binet_{dataset_name}_{i}{crop}", show=False)

    for i, (img_path, dataset_name, normalize, crop) in tqdm(enumerate(img_ls), total=len(img_ls)):
        img_1 = os.path.join(DATASETS_PATH, img_path)
        
        aug_model = BiNetPredictor(normalize_per_img=normalize, model_path="runs_v2/Binet_aug_norm/model_epoch_275_split_0.pt")
        aug_img, org_img, gray_img, sauvola_img, otsu_img = aug_model.run(img_1, plot=True, extra_output=True, show=False)
        
        #non_aug_img = AutoencoderPredictor(normalize_per_img=normalize, model_path="runs_v2/Binet_norm/model_epoch_329_split_0.pt").run(img_1)

        if crop:
            start, end = crop
            aug_img_crop = aug_img[:, start:end]
            #non_aug_img_crop = non_aug_img[:, start:end]
            gray_img_crop = gray_img[:, start:end]
            sauvola_img_crop = sauvola_img[:, start:end]
            otsu_img_crop = otsu_img[:, start:end]

            plt_clear()
            plt.imshow(org_img)
            plt.title("Original Image", fontsize=7)
            plt.axis("off")
            rect = patches.Rectangle((start, 0), end - start, org_img.shape[0], linewidth=1, edgecolor='r', facecolor='none')
            ax = plt.gca()
            ax.add_patch(rect)
            save_figure(f"binet_{dataset_name}_{i}_c_extra", show=False, dpi=300)

            plt_clear()
            plt.title("BiNet + Synthetic", fontsize=7)
            plt.imshow(aug_img, cmap="gray")
            plt.axis("off")
            save_figure(f"binet_{dataset_name}_{i}_c_full", show=False, dpi=300)

            plot(gray_img_crop, aug_img_crop, non_aug_img_crop, sauvola_img_crop, otsu_img_crop, i, True)

        plot(gray_img, aug_img, non_aug_img, sauvola_img, otsu_img, i, False)


def binet_synthetic():
    examples = [3, 4, 2, 19]
    examples_copy = examples.copy()
    aug_doc = AugDoc(plot=True)
    count = 0
    print_count = -1

    while examples_copy:
        plt_clear()

        if count in examples_copy:
            print_count = examples.index(count) + 1

        aug_doc.make_img("", print_count=print_count)
 
        if count in examples_copy:
            examples_copy.remove(count)
            save_figure(f"aug_ex_{print_count}")

        count += 1