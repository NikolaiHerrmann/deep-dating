
import random
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
import matplotlib.patheffects as pe
from matplotlib import cm, colors
from deep_dating.networks import DatingCNN
from deep_dating.preprocessing import PatchExtractor, PatchMethod
from deep_dating.datasets import MPS, ScribbleLens, CLaMM, DatasetSplitter, SetType
from deep_dating.util import save_figure, plt_clear, to_index
from tqdm import tqdm


def plot_patch_prediction(patch_extractor, output, final_prediction, true_label):
    # Clear plot from patch extractor
    plt_clear()
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.imshow(patch_extractor.img, cmap="gray")

    patch_drawing_info = patch_extractor.get_extra_draw_info()
    
    unique_colors = np.sort(np.unique(output))
    cmap_full = plt.get_cmap("Spectral")
    cmap = plt.get_cmap("Spectral", unique_colors.shape[0])
    min_, max_ = np.min(output), np.max(output)
    if true_label:
        min_, max_ = min(min_, true_label), max(max_, true_label)
    norm = colors.Normalize(min_ - 1, max_ + 1)

    # Draw each patch
    for i, (x, y, w, h) in enumerate(patch_drawing_info):
        label = output[i]
        color_idx = np.where(unique_colors == label)[0][0]
        color = cmap(color_idx)
        rect = plt_patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, 
                                     alpha=0.4, facecolor=color, linestyle="dotted")
        ax.add_patch(rect)
        ax.annotate(str(label), (x + (w / 2), y + (h / 2)), fontsize=9,
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


def get_saliency_map(patch, model):
    """
    Adapted from https://github.com/sunnynevarekar/pytorch-saliency-maps/blob/master/Saliency_maps_in_pytorch.ipynb
    """
    model.eval()

    patches = [model.apply_transforms(x) for x in [patch]]
    patches = torch.from_numpy(np.array(patches))
    patches.requires_grad = True
    
    preds = model(patches)
    score, _ = torch.max(preds, 1)

    score.backward()
    
    slc, _ = torch.max(torch.abs(patches.grad[0]), dim=0) # get max along channel axis
    slc = (slc - slc.min()) / (slc.max()-slc.min()) # normalize to [0..1]

    return slc.numpy(), score.cpu().detach().numpy()[0]


def make_map(patch_extractor, patches, model):
    patch_drawing_info = patch_extractor.get_extra_draw_info()
    saliency_map = np.zeros(patch_extractor.img.shape)
    
    outputs = []

    for i, (x, y, w, h) in tqdm(enumerate(patch_drawing_info), total=len(patch_drawing_info)):
        patch_map, model_output = get_saliency_map(patches[i], model)
        patch_map = cv2.resize(patch_map, (h, w), interpolation=cv2.INTER_NEAREST)
        outputs.append(model_output)
        saliency_map[y:y+h, x:x+w] = np.maximum(patch_map, saliency_map[y:y+h, x:x+w])

    plt_clear()
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(patch_extractor.img, cmap="gray")
    
    color = "blue"
    x, y, _, _ = patch_drawing_info[0]
    x_end, y_end, w, h = patch_drawing_info[-1]
    w = w + x_end - x
    h = h + y_end - y
    # rect = plt_patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")
    # ax[0].add_patch(rect)
    ax[1].imshow(saliency_map, cmap=plt.cm.hot)
    
    save_figure("saliency_map", fig=fig, show=True)

    return outputs


# self.y, self.y_unique = to_index(self.y)
        
#         def decode_class(self, class_idxs):
#             return self.y_unique[class_idxs]


def run_patch_pipeline(img_path, agg_func=np.median, true_label=None, plot=True, make_saliency_map=True):

    model_path = "runs/Feb2-12-2-23/model_epoch_5.pt" #"runs/Jan8-19-25-16/model_epoch_3.pt" #"runs/Jan9-13-59-8/model_epoch_29.pt" #"runs/Jan6-22-21-16/model_epoch_28.pt" #"runs/Jan9-13-59-8/model_epoch_29.pt" # # #"runs/Jan8-19-25-16/model_epoch_3.pt" # # #"runs/Jan8-19-25-16/model_epoch_3.pt"# #
    model = DatingCNN("inception_resnet_v2", verbose=False, num_classes=6)
    model.load(model_path, continue_training=False)
    
    patch_extractor = PatchExtractor(plot=plot, method=PatchMethod.SLIDING_WINDOW_LINES)
    patches = patch_extractor.extract_patches(img_path)

    if plot and make_saliency_map:
        output = make_map(patch_extractor, patches, model)
    else:
        patches = [model.apply_transforms(x) for x in patches]
        patches = torch.from_numpy(np.array(patches))

        output = model(patches)
        output = output.cpu().detach().numpy()

        if model.classification:
            class_idxs = np.argmax(output, axis=1)
            _, labels_unique = to_index(y)
            output = labels_unique[class_idxs]
            print(output)
        else:
            output = output.flatten()
    
    output = np.round(output).astype(int)
    final_prediction = int(np.round(agg_func(output)))

    if plot:
        plot_patch_prediction(patch_extractor, output, final_prediction, true_label)

    return final_prediction


def run_over_dataset(dataset, set_type=SetType.VAL):
    global y
    x, y = DatasetSplitter(dataset).get_data(set_type)
    data = list(zip(x, y))
    random.shuffle(data)
    random.shuffle(data)
    for img_file, label in data:
        run_patch_pipeline(img_file, true_label=label, make_saliency_map=False)


if __name__ =="__main__":
    run_over_dataset(ScribbleLens())