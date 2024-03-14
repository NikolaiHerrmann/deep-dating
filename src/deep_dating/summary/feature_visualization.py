
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from deep_dating.util import save_figure, SEED
from deep_dating.prediction import DatingPredictor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


class FeatureVis:

    def __init__(self, feat_path):
        self._read(feat_path)

        self.unique_labels = list(np.unique(self.labels))
        self.num_unique_labels = len(self.unique_labels)
        print(f"{self.num_unique_labels} unique labels:", self.unique_labels)
        self.colors = sns.color_palette('Spectral', n_colors=self.num_unique_labels)

    def _read(self, feat_path):
        self.labels, self.features, self.all_paths = DatingPredictor().load(feat_path)
        self.labels = self.labels.flatten()

    def _get_date_dict(self, reduced_features):
        dict_ = {}

        for i, x in enumerate(self.labels):
            val = reduced_features[i, :]
            if x not in dict_:
                dict_[x] = [val]
            else:
                dict_[x].append(val)

        for key in dict_.keys():
            dict_[key] = np.vstack(dict_[key])

        return dict_

    def _plot_dims(self, reduced_features, axs_name, title_text, in_3d=False):
        dict_ = self._get_date_dict(reduced_features)
        keys = sorted(dict_.keys())
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111, projection="3d") if in_3d else plt.gca()

        for i, key in enumerate(keys):
            arr = dict_[key]
            z = arr[:, 2] if in_3d else None
            ax.scatter(arr[:, 0], arr[:, 1], z, color=self.colors[i], label=str(key))

        fig.legend(loc='center left', bbox_to_anchor=(0.9, 0.5), title="Date")
        plt.xlabel(axs_name + " 1")
        plt.ylabel(axs_name + " 2")
        if in_3d:
            ax.set_zlabel(axs_name + " 3")
        plt.title(title_text)
        title_text = title_text.lower().replace(" ", "_") + str("_3d" if in_3d else "")
        #save_figure(title_text, show=True)
        plt.show()

    def pca(self):

        pca = PCA(n_components=None, random_state=SEED)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.features)
        features_pca = pca.fit_transform(self.features)

        pca_title = "PCA of MPS Validation Set"
        self._plot_dims(features_pca, "PC", pca_title)
        self._plot_dims(features_pca, "PC", pca_title, in_3d=True)

    def tsne(self):
        perplexity=30
        tsne_2d = TSNE(n_components=2, perplexity=perplexity, random_state=SEED, n_jobs=-1)
        features_tsne_2d = tsne_2d.fit_transform(self.features)
        #tsne_3d = TSNE(n_components=3, perplexity=perplexity, random_state=SEED, n_jobs=-1)
        #features_tsne_3d = tsne_3d.fit_transform(features)

        tsne_title = "TSNE for MPS Validation Set"
        self._plot_dims(features_tsne_2d, "Dimension", tsne_title)
        #plot_dims(features_tsne_3d, "Dimension", tsne_title, in_3d=True)