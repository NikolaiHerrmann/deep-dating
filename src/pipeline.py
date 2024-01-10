
import random
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
from deep_dating.networks import DatingCNN
from deep_dating.preprocessing import PatchExtractor, PatchMethod
from deep_dating.datasets import MPS, ScribbleLens, CLaMM


def run_patch_pipeline(img_path, true_label=None, plot=True):

    model_path = "runs/Jan8-19-25-16/model_epoch_3.pt"
    model = DatingCNN("inception_resnet_v2", verbose=False)
    model.load(model_path, continue_training=False)
    
    patch_extractor = PatchExtractor(plot=plot, method=PatchMethod.SLIDING_WINDOW_LINES)
    patches = patch_extractor.extract_patches(img_path)

    patches = [model.apply_transforms(x) for x in patches]
    patches = torch.from_numpy(np.array(patches))

    output = model(patches)
    output = np.round(output.cpu().detach().numpy().flatten()).astype(int)
    final_prediction = int(np.round(np.median(output)))

    if plot:
        plt.clf()
        plt.cla()
        plt.imshow(patch_extractor.img_org)
        patch_drawing_info = patch_extractor.get_extra_draw_info()

        unique_colors = np.sort(np.unique(output))
        colors = sns.color_palette('Spectral', n_colors=unique_colors.shape[0])#, as_cmap=True)
        print(colors)

        ax = plt.gca()
        #fig, ax = plt.subplots()

        for i, (x, y, w, h) in enumerate(patch_drawing_info):
            label = output[i]
            color_idx = np.where(unique_colors == label)[0][0]
            color = colors[color_idx]
            rect = plt_patches.Rectangle((x, y), w, h, 
                                linewidth=2, edgecolor=color, alpha=0.2, 
                                facecolor=color, linestyle="dotted")
            ax.add_patch(rect)
            ax.annotate(str(label), (x + (w / 2), y + (h / 2)), 
                        color="white", weight='bold', ha='center', va='center')

        label_text = f" Label was {true_label}" if true_label else ""
        #heatmap = plt.pcolor(colors)
        # plt.colorbar(colors, ax=ax)
        plt.title(f"Final prediction (Median): {final_prediction}" + label_text)
        plt.show()

    return final_prediction

def run_over_dataset(dataset):
    data = list(zip(dataset.X, dataset.y))
    random.shuffle(data)
    for img_file, label in data:
        run_patch_pipeline(img_file, true_label=label)

if __name__ =="__main__":
    run_over_dataset(CLaMM())