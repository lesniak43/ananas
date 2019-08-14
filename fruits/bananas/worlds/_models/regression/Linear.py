import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression as sklearnLinearRegression

import mandalka
from crisper.datadict import Table, Variable
from bananas.worlds import StorageWorld, ProxyWorld, ProxyContainer

@mandalka.node
class LinearRegression(StorageWorld):
    def build(self, *, source, alpha):
        representation = source.data["representation"]
        value = source.data["value"]
        if alpha is None:
            _linreg = sklearnLinearRegression()
        else:
            _linreg = Ridge(alpha=alpha)
        _linreg.fit(representation, value)
        self.data["w"] = Variable(_linreg.coef_.copy().reshape(1,-1))
        self.data["b"] = Variable(float(_linreg.intercept_))
    @mandalka.lazy
    def predict(self, *, source):
        return LinearRegressionPrediction(
            source=source,
            trained=self,
        )

@mandalka.node
class LinearRegressionPrediction(ProxyWorld):
    def build_proxy(self, *, source, trained):
        self.data["value_predicted"] = ProxyContainer(
            lambda : np.sum(
                source.data["representation"] * trained.data["w"],
                axis=1,
            ) + trained.data["b"],
            Table,
        )
