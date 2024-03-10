
import matplotlib.pyplot as plt


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