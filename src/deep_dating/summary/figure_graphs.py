
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deep_dating.datasets import SetType
from deep_dating.util import save_figure


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
    
    for set_type, label_name in [("train", "Training Set:\n(H-)DIBCO 20[09, 10, 11, 12, 13, 14, 17, 18, 19] + Synthetic"), 
                                 ("eval", "Validation Set:\n(H-)DIBCO 2016")]:
        df_set = df[df["set_type"] == set_type]
        loss = df_set["mean_loss"]
        epochs = df_set["epoch"]
        ax.plot(epochs, loss, label=label_name)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("L1 (Mean Absolute Error) Loss")

    plt.legend()
    
    save_figure("binet_aug_norm_loss_curve", fig_dir=os.path.dirname(file), show=True)
