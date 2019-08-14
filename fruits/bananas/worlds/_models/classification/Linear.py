import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression as sklearnLogisticRegression

import mandalka
from crisper.datadict import Table, Variable, Header
from bananas.worlds import StorageWorld, ProxyWorld, ProxyContainer

@mandalka.node
class TwoClassLogisticRegression(StorageWorld):
    def build(self, *, source, C, class_weight):
        representation = source.data["representation"]
        value = source.data["value"]
        assert issubclass(value.dtype.type, np.integer)
        assert set(value) == set([0,1])
        _lr = sklearnLogisticRegression(
            penalty="l2", dual=False, tol=0.0001, C=C,
            fit_intercept=True, intercept_scaling=1,
            class_weight=class_weight, random_state=43,
            solver="liblinear", max_iter=100, multi_class="ovr",
            verbose=0, warm_start=False, n_jobs=1,
        )
        _lr.fit(representation, value)
        assert _lr.classes_[0] == 0
        assert _lr.classes_[1] == 1
        self.data["w"] = Variable(_lr.coef_.copy().reshape(1,-1))
        self.data["b"] = Variable(float(_lr.intercept_))
    @mandalka.lazy
    def predict(self, *, source):
        return TwoClassLogisticRegressionPrediction(
            source=source,
            trained=self,
        )

@mandalka.node
class TwoClassLogisticRegressionPrediction(ProxyWorld):
    def build_proxy(self, *, source, trained):
        self.data[("value_predicted", "classes")] = ProxyContainer(
            lambda : np.array([0,1], dtype=np.int),
            Header,
        )
        def f():
            x = source.data["representation"]
            w = trained.data["w"]
            b = trained.data["b"]
            # predict value == 1
            result = np.zeros((x.shape[0],2), dtype=np.float)
            result[:,1] = expit(np.sum(x * w, axis=1) + b)
            result[:,0] = 1 - result[:,1]
            return result
        self.data[("value_predicted", "probability")] = ProxyContainer(
            f,
            Table,
        )
