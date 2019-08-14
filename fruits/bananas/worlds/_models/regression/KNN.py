import numpy as np

import mandalka
from crisper.datadict import Table, Variable
from bananas.worlds import StorageWorld, ProxyWorld, ProxyContainer

@mandalka.node
class KernelKNNRegressor(StorageWorld):
    def build(self, *, source, k):
        assert isinstance(k, int)
        assert k > 0
        self.data["k"] = Variable(k)
        self.data["value_original"] = Variable(source.data["value"])
    @mandalka.lazy
    def predict(self, *, source):
        return KernelKNNRegressorPrediction(
            source=source,
            trained=self,
        )

@mandalka.node
class KernelKNNRegressorPrediction(ProxyWorld):
    def build_proxy(self, *, source, trained):
        self.data["value_predicted"] = ProxyContainer(
            lambda : np.array(
                [np.mean(trained.data["value_original"][_idx]) \
                    for _idx in np.argsort(
                        -source.data["representation"],
                        axis=1,
                    )[:,:trained.data["k"]]]
            ),
            Table,
        )
