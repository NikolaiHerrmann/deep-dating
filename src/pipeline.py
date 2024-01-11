
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
import matplotlib.patheffects as pe
from matplotlib import cm, colors
from deep_dating.networks import DatingCNN
from deep_dating.preprocessing import PatchExtractor, PatchMethod
from deep_dating.datasets import MPS, ScribbleLens, CLaMM
from deep_dating.util import save_figure


def plot_patch_prediction(patch_extractor, output, final_prediction, true_label):

    # Clear plot from patch extractor
    plt.cla()
    plt.clf()
    plt.close()
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.imshow(patch_extractor.img, cmap="gray")

    patch_drawing_info = patch_extractor.get_extra_draw_info()
    
    unique_colors = np.sort(np.unique(output))
    cmap_full = plt.get_cmap("Spectral")
    cmap = plt.get_cmap("Spectral", unique_colors.shape[0])
    norm = colors.Normalize(np.min(output) - 1, np.max(output) + 1)

    # Draw each patch
    for i, (x, y, w, h) in enumerate(patch_drawing_info):
        label = output[i]
        color_idx = np.where(unique_colors == label)[0][0]
        color = cmap(color_idx)
        rect = plt_patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, 
                                     alpha=0.4, facecolor=color, linestyle="dotted")
        ax.add_patch(rect)
        ax.annotate(str(label), (x + (w / 2), y + (h / 2)), fontsize=14,
                    color="white", weight='bold', ha='center', va='center',
                    path_effects=[pe.withStroke(linewidth=3, foreground="black")])

    # Colorbar
    color_map = cm.ScalarMappable(norm=norm, cmap=cmap_full)
    cbar = fig.colorbar(color_map, ax=ax, orientation="horizontal", shrink=0.6, pad=0.05)
    box_plot = cbar.ax.boxplot(output, vert=False, positions=[0.5], widths=[0.5])
    for median in box_plot['medians']:
        median.set_color('none')
    cbar.ax.get_yaxis().set_visible(False)
    cbar.ax.axvline(final_prediction, c='black', linewidth=2)
    if true_label:
        cbar.ax.axvline(true_label, c='green', linewidth=2)
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label("Year Distribution of Patches", fontsize=12)

    label_text = f" - Label (Ground Truth): {int(true_label)}" if true_label else ""
    plt.title(f"Prediction (Median): {int(final_prediction)}" + label_text, fontsize=15)
    
    save_figure("patch_pred", fig=fig, show=True)


def run_patch_pipeline(img_path, agg_func=np.median, true_label=None, plot=True):

    model_path = "runs/Jan6-22-21-16/model_epoch_28.pt"
    model = DatingCNN("inception_resnet_v2", verbose=False)
    model.load(model_path, continue_training=False)
    
    patch_extractor = PatchExtractor(plot=plot, method=PatchMethod.SLIDING_WINDOW_LINES)
    patches = patch_extractor.extract_patches(img_path)

    patches = [model.apply_transforms(x) for x in patches]
    patches = torch.from_numpy(np.array(patches))

    output = model(patches)
    output = np.round(output.cpu().detach().numpy().flatten()).astype(int)
    final_prediction = int(np.round(agg_func(output)))

    if plot:
        plot_patch_prediction(patch_extractor, output, final_prediction, true_label)

    return final_prediction


def run_over_dataset(dataset):
    data = list(zip(dataset.X, dataset.y))
    random.shuffle(data)
    for img_file, label in data:
        run_patch_pipeline(img_file, true_label=label)


if __name__ =="__main__":
    run_over_dataset(MPS())