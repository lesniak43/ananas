import numpy as np

from bananas.pipelines import mean_warszycki_logki
from bananas.worlds import (
    MolPrint2D,
    TanimotoMinMaxKernel,
    TanimotoMinMaxSimilarity,
    BalancedAgglomerativeClustering,
    CrossValidation,
    SubsetData,
    StorageWorld,
    Variable,
    DataDict,
)
import crisper
from herbivores import IndexedNDArray
import mandalka

from bananas.worlds import (
    AtomPairs,
    Morgan,
    TanimotoMinMaxKernel,
    MaxTanimotoMinMaxSimilarity,
    SMILESToMol,
    Table,
    Table2D,
)

@mandalka.node
class Benchmarks2018StructuralSimilarity:
    def __init__(self, *, source):
        self.data = source.data.slice[:]
        kernel_morgan = TanimotoMinMaxKernel(
            source=Morgan(
                radius=2,
                use_chirality=True,
                use_bond_types=True,
                use_features=False,
                converter=SMILESToMol(),
            )(source=source)
        ).data["tanimoto_kernel"]
        kernel_atompairs = TanimotoMinMaxKernel(
            source=AtomPairs(converter=SMILESToMol())(source=source)
        ).data["tanimoto_kernel"]
        self.data["kernel"] = Table2D(.5*(kernel_morgan+kernel_atompairs))
        self.data.lock()

@mandalka.node
class Benchmarks2018StructuralTestToTrainSimilarity:
    def __init__(self, *, tr, te):
        self.data = DataDict(source_dirname=None)
        morgan = Morgan(
            radius=2,
            use_chirality=True,
            use_bond_types=True,
            use_features=False,
            converter=SMILESToMol(),
        )
        atompairs = AtomPairs(converter=SMILESToMol())
        kernel_morgan = TanimotoMinMaxSimilarity(
            tr=morgan(source=tr),
            te=morgan(source=te),
        ).data["tanimoto_kernel"]
        kernel_atompairs = TanimotoMinMaxSimilarity(
            tr=atompairs(source=tr),
            te=atompairs(source=te),
        ).data["tanimoto_kernel"]
        self.data["kernel"] = Table(.5*(kernel_morgan+kernel_atompairs))
        self.data.lock()

@mandalka.node
class Benchmarks2018SplitTestToTrainCloserHalf(StorageWorld):
    def build(self, *, source):
        tr, te = source.get_train(), source.get_test()
        morgan_fpr = Morgan(
            radius=2,
            use_chirality=True,
            use_bond_types=True,
            use_features=False,
            converter=SMILESToMol(),
        )
        atompairs_fpr = AtomPairs(converter=SMILESToMol())
        kernel_max = .5 * (
            MaxTanimotoMinMaxSimilarity(
                tr=morgan_fpr(source=tr),
                te=morgan_fpr(source=te),
            ).data["kernel_max"] + \
            MaxTanimotoMinMaxSimilarity(
                tr=atompairs_fpr(source=tr),
                te=atompairs_fpr(source=te),
            ).data["kernel_max"]
        )
        self.data["mask"] = Table(kernel_max >= np.median(kernel_max))

class Benchmarks2018Problem:

    def __init__(self, *, threshold, ic50_conversion_strategy, fit_ic50):
        self._variant = {
            "threshold": threshold,
            "ic50_conversion_strategy": ic50_conversion_strategy,
            "fit_ic50": fit_ic50,
        }
        self.target_uids = self._get_target_uids()
        self.split_names = ["bac", "cv"]
        self.n_splits = 5

    def _get_target_uids(self):
        potential_target_uids = [
            "CHEMBL214", "CHEMBL224", "CHEMBL225", "CHEMBL3371",
            "CHEMBL3155", "CHEMBL226", "CHEMBL251", "CHEMBL217",
            "CHEMBL264", "CHEMBL216"
        ]
        crisper.evaluate(*[self.get_dataset(t) for t in potential_target_uids], label="Datasets")
        return [
            t for t in potential_target_uids \
                if len(self.get_dataset(t).data["smiles"]) >= 200
        ]

    def _get_summarizer(self):
        if self._variant["threshold"] is None:
            return Benchmarks2018ProblemRegressionSummary
        else:
            return Benchmarks2018ProblemClassificationSummary

    def get_dataset(self, target_uid):
        return mean_warszycki_logki(
            target_uid=target_uid,
            chembl_filename="chembl_24.db",
            threshold=self._variant["threshold"],
            ic50_conversion_strategy=self._variant["ic50_conversion_strategy"],
            fit_ic50=self._variant["fit_ic50"],
        )["final"]

    def get_split(self, split_name, dataset):
        if split_name == "cv":
            return CrossValidation(
                source=dataset,
                n_groups=self.n_splits,
                seed=43,
            )
        elif split_name == "bac":
            return BalancedAgglomerativeClustering(
                source=Benchmarks2018StructuralSimilarity(source=dataset),
                kernel="kernel",
                n_groups=self.n_splits,
            )
        else:
            raise ValueError("split_name: {}".format(split_name))

    def get_splits(self, split_name, dataset):
        return self.get_split(split_name, dataset).get_splits()

    def benchmark(self, solutions):

        solution_names = sorted(solutions.keys())
        target_uids = self.target_uids
        split_names = self.split_names
        n_split_idx = list(range(self.n_splits))

        predictions = None
        shape = [len(x) for x in [solution_names, target_uids, split_names, n_split_idx]]
        from copy import deepcopy
        for _s in reversed(shape):
            predictions = [deepcopy(predictions) for _ in range(_s)]

        for i, solution_name in enumerate(solution_names):
            solution = solutions[solution_name]
            for j, target_uid in enumerate(target_uids):
                dataset = self.get_dataset(target_uid)
                for k, split_name in enumerate(split_names):
                    for l, split in enumerate(self.get_splits(split_name, dataset)):
                        tr, te = split.get_train(), split.get_test()
                        tr = SubsetData(source=tr, data_names=["smiles", "value"], keep=True)
                        te_clean = SubsetData(source=te, data_names=["smiles"], keep=True)
                        trained = solution(source=tr)
                        prediction = trained(source=te_clean)
                        predictions[i][j][k][l] = (
                            Benchmarks2018SplitTestToTrainCloserHalf(source=split),
                            te,
                            prediction
                        )

        benchmark = {
            "predictions": predictions,
            "headers": [
                solution_names,
                target_uids,
                split_names,
                n_split_idx,
            ]
        }

        return self._get_summarizer()(benchmark=benchmark)

class _Benchmarks2018ProblemSummary(StorageWorld):
    # {metric_name: (function, reverse_ranking), ...}
    metrics = None

    def build(self, *, benchmark):
        solution_names, target_uids, split_names, n_split_idx = benchmark["headers"]
        metric_names = list(sorted(self.metrics.keys()))
        headers = [np.array(arr) for arr in [
            metric_names,
            solution_names,
            target_uids,
            split_names,
            n_split_idx,
        ]]
        header_names = np.array(["metric", "solution", "target", "split", "n_split"])
        scores_shape = tuple([len(h) for h in headers])
        predictions_shape = scores_shape[1:]

        values = np.zeros(scores_shape, dtype=np.float32)
        values.fill(np.nan)

        from itertools import product
        import tqdm
        for i, metric_name in enumerate(metric_names):
            print(metric_name)
            for idx in tqdm.tqdm(list(product(*[range(i) for i in predictions_shape]))):
                _pred = benchmark["predictions"]
                for _i in idx:
                    _pred = _pred[_i]
                values[i][idx] = self.metrics[metric_name][0](*_pred)

        self.data["values"] = Variable(values)
        self.data["header_names"] = Variable(header_names)
        for name, h in zip(header_names, headers):
            self.data[("headers", name)] = Variable(h)

    @property
    def scores(self):
        headers = [self.data[("headers", name)] for name in self.data["header_names"]]
        return IndexedNDArray(
            values=self.data["values"],
            indices=headers,
            index_names=self.data["header_names"],
        )

    def results(self, metric_name, split_name):
        reverse_ranking = self.metrics[metric_name][1]
        scores = self.scores.slice({"metric": metric_name, "split": split_name}).mean("n_split")
        mean_score = scores.mean("target")
        mean_ranking = scores.ranking("solution", reverse=reverse_ranking).mean("target")
        header = np.array(["Solution"] + list(scores.get_index("target")) + ["Mean Score", "Mean Ranking"])
        values = [scores.get_index("solution")] + list(scores.values.T) + [mean_score.values, mean_ranking.values]
        return header, values


@mandalka.node
class Benchmarks2018ProblemRegressionSummary(_Benchmarks2018ProblemSummary):
    # {metric_name: (function, reverse_ranking), ...}
    metrics = {
        "Mean_Absolute_Error": (
            lambda _, n_true, n_pred: np.mean(
                np.abs(n_true.data["value"] - n_pred.data["value_predicted"])),
            True,
        ),
        "Mean_Squared_Error": (
            lambda _, n_true, n_pred: np.mean(
                np.square(n_true.data["value"] - n_pred.data["value_predicted"])),
            True,
        ),
        "Mean_Squared_Error_Closer": (
            lambda n_mask, n_true, n_pred: np.mean(
                np.square(n_true.data["value"] - n_pred.data["value_predicted"])[
                    n_mask.data["mask"]
                ]
            ),
            True,
        ),
        "Mean_Squared_Error_Further": (
            lambda n_mask, n_true, n_pred: np.mean(
                np.square(n_true.data["value"] - n_pred.data["value_predicted"])[
                    np.logical_not(n_mask.data["mask"])
                ]
            ),
            True,
        ),
        "Spearman_Rho": (
            lambda _, n_true, n_pred: Benchmarks2018ProblemRegressionSummary._spearmanr2(n_true.data["value"], n_pred.data["value_predicted"]),
            False,
        ),
    }
    closer_further_metrics = ("Mean_Squared_Error",)
    @staticmethod
    def _spearmanr2(x,y):
        from scipy.stats import spearmanr
        if len(set(x)) == 1 or len(set(y)) == 1:
            return 0.
        else:
            return spearmanr(x,y).correlation

@mandalka.node
class Benchmarks2018ProblemClassificationSummary(_Benchmarks2018ProblemSummary):
    # {metric_name: (function, reverse_ranking), ...}
    metrics = {
        "Accuracy": (
            lambda _, n_true, n_pred: np.mean(
                n_pred.data[("value_predicted", "classes")][np.argmax(n_pred.data[("value_predicted", "probability")], axis=1)] == n_true.data["value"]
            ),
            False,
        ),
        "Weighted_Accuracy": (
            lambda _, n_true, n_pred: np.average(
                (n_pred.data[("value_predicted", "classes")][np.argmax(n_pred.data[("value_predicted", "probability")], axis=1)] == n_true.data["value"]).astype(np.float),
                weights=Benchmarks2018ProblemClassificationSummary._inverse_weights(n_true.data["value"]),
            ),
            False,
        ),
        "Weighted_Accuracy_Closer": (
            lambda n_mask, n_true, n_pred: np.average(
                (n_pred.data[("value_predicted", "classes")][np.argmax(n_pred.data[("value_predicted", "probability")], axis=1)] == n_true.data["value"]).astype(np.float)[
                    n_mask.data["mask"]
                ],
                weights=Benchmarks2018ProblemClassificationSummary._inverse_weights(n_true.data["value"])[
                    n_mask.data["mask"]
                ],
            ),
            False,
        ),
        "Weighted_Accuracy_Further": (
            lambda n_mask, n_true, n_pred: np.average(
                (n_pred.data[("value_predicted", "classes")][np.argmax(n_pred.data[("value_predicted", "probability")], axis=1)] == n_true.data["value"]).astype(np.float)[
                    np.logical_not(n_mask.data["mask"])
                ],
                weights=Benchmarks2018ProblemClassificationSummary._inverse_weights(n_true.data["value"])[np.logical_not(n_mask.data["mask"])],
            ),
            False,
        ),
    }
    closer_further_metrics = ("Weighted_Accuracy",)
    @staticmethod
    def _inverse_weights(arr):
        from collections import Counter
        d = Counter(arr)
        return np.vectorize(lambda x: 1./d[x])(arr)
