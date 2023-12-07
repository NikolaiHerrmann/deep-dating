
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot(path):
    df = pd.read_csv(path)

    fig, axs = plt.subplots(1, 3, figsize=(10, 4))

    for set_type in ["train", "eval"]:
        sub_pd = df[df["set"] == set_type]

        vals = sub_pd["mean_running_loss"].to_numpy()[5:]
        axs[0].plot(vals)

        axs[1].plot(sub_pd["mae"].to_numpy()[5:])

        axs[2].plot(sub_pd["cs_0"].to_numpy()[5:])
        axs[2].plot(sub_pd["cs_25"].to_numpy()[5:])

    # axs[0].xlabel("Epoch")
    # axplt.ylabel("log(MSE)")

    plt.show()

plot("runs/Dec06_11-04-06_p14s/epoch_log_12_6_11_4_6.csv")