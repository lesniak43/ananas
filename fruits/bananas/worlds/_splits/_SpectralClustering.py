from sklearn.cluster import SpectralClustering as sklearnSpectralClustering

import mandalka

from .. import Table, Variable
from . import GroupSplitter

@mandalka.node
class SpectralClustering(GroupSplitter):
    def build(self, *, source, kernel, n_groups):
        self.data = source.data.slice[:]
        del self.data["groups"]
        del self.data["n_groups"]
        del self.data[kernel]
        self.data["n_groups"] = Variable(n_groups)
        self.log("calculating clustering...")
        self.data["groups"] = Table(sklearnSpectralClustering(
            n_clusters=n_groups,
            eigen_solver=None,
            random_state=43,
            n_init=10,
            gamma=1.0,
            affinity="precomputed",
            n_neighbors=10,
            eigen_tol=0.0,
            assign_labels="kmeans",
            degree=3,
            coef0=1,
            kernel_params=None,
            n_jobs=1,
        ).fit_predict(source.data[kernel]))
