import numpy as np

import mandalka
from crisper.datadict import Table, Header, Variable
from bananas.worlds import StorageWorld, ProxyWorld, ProxyContainer

@mandalka.node
class KernelKNNClassifier(StorageWorld):
    def build(self, *, source, k):
        assert isinstance(k, int)
        assert k > 0
        self.data["k"] = Variable(k)
        self.data["value_original"] = Variable(source.data["value"])
    @mandalka.lazy
    def predict(self, *, source):
        return KernelKNNClassifierPrediction(
            source=source,
            trained=self,
        )

@mandalka.node
class KernelKNNClassifierPrediction(ProxyWorld):
    def build_proxy(self, *, source, trained):
        self.data[("value_predicted", "classes")] = ProxyContainer(
            lambda : np.unique(trained.data["value_original"]),
            Header,
        )
        def f():
            k = trained.data["k"]
            idx = np.argsort(-source.data["representation"], axis=1)[:,:k]
            classes = np.unique(trained.data["value_original"])
            result = np.zeros((len(idx), len(classes)), dtype=np.float)
            for i, _idx in enumerate(idx):
                for j, c in enumerate(classes):
                    result[i,j] = (trained.data["value_original"][_idx] == c).sum() / k
            result[:,-1] = 1-result[:,:-1].sum(axis=1)
            return result
        self.data[("value_predicted", "probability")] = ProxyContainer(
            f,
            Table,
        )
