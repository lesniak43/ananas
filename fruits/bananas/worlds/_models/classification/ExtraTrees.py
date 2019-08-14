from sklearn.ensemble import ExtraTreesClassifier as sklearnExtraTreesClassifier

import mandalka
from crisper.datadict import Table, Header
from bananas.worlds import ProxyWorld, ProxyContainer

@mandalka.node
class ExtraTreesClassifier:
    def __init__(self, *, source, n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes, min_impurity_decrease, min_impurity_split, bootstrap, class_weight):
        self.source = source
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.bootstrap = bootstrap
        self.class_weight = class_weight
    @property
    def etc(self):
        return sklearnExtraTreesClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            min_impurity_split=self.min_impurity_split,
            bootstrap=self.bootstrap,
            class_weight=self.class_weight,
            #
            oob_score=False,
            n_jobs=1,
            random_state=43,
            verbose=0,
            warm_start=False,
        )
    @mandalka.lazy
    def predict(self, *, source):
        return ExtraTreesClassifierPrediction(
            source=source,
            trained=self,
        )

@mandalka.node
class ExtraTreesClassifierPrediction(ProxyWorld):
    def build_proxy(self, *, source, trained):
        etc = trained.etc.fit(
            trained.source.data["representation"],
            trained.source.data["value"],
        )
        self.data[("value_predicted", "classes")] = ProxyContainer(
            lambda : etc.classes_,
            Header,
        )
        self.data[("value_predicted", "probability")] = ProxyContainer(
            lambda: etc.predict_proba(source.data["representation"]),
            Table,
        )