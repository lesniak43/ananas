from collections import Counter

import numpy as np

import mandalka
from crisper.datadict import Table, Variable, Header
from bananas.worlds import StorageWorld, ProxyWorld, ProxyContainer

@mandalka.node
class ConstantClassifier(StorageWorld):
    def build(self, *, source):
        classes, probability = zip(*sorted(Counter(source.data["value"]).items()))
        classes = np.array(classes, dtype=source.data["value"].dtype)
        probability = np.array(probability, dtype=np.float)
        probability /= probability.sum()
        self.data[("value_statistics", "classes")] = Header(classes)
        self.data[("value_statistics", "probability")] = Variable(probability)
    @mandalka.lazy
    def predict(self, *, source):
        return ConstantClassifierPrediction(
            source=source,
            trained=self,
        )

@mandalka.node
class ConstantClassifierPrediction(ProxyWorld):
    def build_proxy(self, *, source, trained):
        self.data[("value_predicted", "classes")] = ProxyContainer(
            lambda : trained.data[("value_statistics", "classes")],
            Header,
        )
        self.data[("value_predicted", "probability")] = ProxyContainer(
            lambda: trained.data[("value_statistics", "probability")].reshape(1,-1) * np.ones(source.data["representation"].shape[0], dtype=np.float).reshape(-1,1),
            Table,
        )
