import mandalka
import numpy as np

from .. import StorageWorld, Table
from sklearn.manifold import TSNE

def _kernel_to_distance(kernel):
    # returns SQUARED norm
    d = np.diagonal(kernel).reshape(-1, 1) + np.diagonal(kernel).reshape(1, -1) - 2 * kernel
    d[d<0.] = 0.
    return d

@mandalka.node
class KernelTSNE(StorageWorld):
    def build(self, source, kernel, n_components, perplexity, early_exaggeration, learning_rate):
        self.data = source.data.slice[:]
        distance = _kernel_to_distance(source.data[kernel])
        self.data["tsne"] = Table(TSNE(
            n_components=n_components,
            perplexity=perplexity,
            early_exaggeration=early_exaggeration,
            learning_rate=learning_rate,
            random_state=43,
            metric="precomputed").fit_transform(distance))
