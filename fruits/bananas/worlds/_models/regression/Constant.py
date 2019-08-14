import numpy as np

import mandalka
from crisper.datadict import Table, Variable
from bananas.worlds import StorageWorld, ProxyWorld, ProxyContainer

@mandalka.node
class ConstantRegressor(StorageWorld):
    def build(self, *, source, variant):
        if variant == "mean":
            v = float(np.mean(source.data["value"]))
        elif variant == "median":
            v = float(np.median(source.data["value"]))
        else:
            raise ValueError("'variant' must be 'mean' or 'median'")
        self.data["value_constant"] = Variable(v)
    @mandalka.lazy
    def predict(self, *, source):
        return ConstantRegressorPrediction(
            source=source,
            trained=self,
        )

@mandalka.node
class ConstantRegressorPrediction(ProxyWorld):
    def build_proxy(self, *, source, trained):
        self.data["value_predicted"] = ProxyContainer(
            lambda : trained.data["value_constant"] * np.ones(source.data["representation"].shape[0], dtype=np.float32),
            Table,
        )
