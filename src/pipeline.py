
import numpy as np
import torch
from deep_dating.networks import DatingCNN
from deep_dating.preprocessing import PatchExtractor, PatchMethod
from deep_dating.datasets import MPS


def pipeline(img_path):

    model_path = "runs/Jan6-22-21-16/model_epoch_28.pt"
    model = DatingCNN("inception_resnet_v2")
    model.load(model_path, continue_training=False)
    
    patch_extractor = PatchExtractor(plot=True, method=PatchMethod.SLIDING_WINDOW_LINES)
    patches = patch_extractor.extract_patches(img_path)

    patches = [model.apply_transforms(x) for x in patches]
    patches = torch.from_numpy(np.array(patches))

    output = model(patches)
    output = np.round(output.cpu().detach().numpy().flatten()).astype(int)

    print(output)
    print(np.median(output))

    patch_extractor.save_plot(show=True)

def run_over_dataset(dataset):
    for img_file in dataset.X:
        pipeline(img_file)

if __name__ =="__main__":
    run_over_dataset(MPS())