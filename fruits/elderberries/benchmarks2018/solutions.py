from itertools import product

import numpy as np

import mandalka
from crisper.datadict import Table
from bananas.worlds import StorageWorld

from bananas.worlds import (
    LinearRegression,
    KernelKNNRegressor,
    ConstantRegressor,
    ExtraTreesRegressor,
    RandomForestRegressor,
    ExtraTreesClassifier,
    RandomForestClassifier,
    KernelKNNClassifier,
    ConstantClassifier,
    TwoClassLogisticRegression,
)
from bananas.worlds import (
    ETKDG,
    SMILESToMol,
)
from bananas.worlds import (
    HashedFingerprinter,
    ConcatenatedFingerprinter,
    MACCS,
    RDKitDescriptors2DFingerprinter,
    RDKitDescriptors3DFingerprinter,
    GETAWAY,
    WHIM,
    RDF,
    MORSE,
    AUTOCORR3D,
    AtomPairs,
    TFD,
    EStateCount,
    KlekotaRoth,
    Morgan,
    RDKitFingerprinter,
    MolPrint2D,
    Spectrophores,
    Mordred2D,
    Mordred3D,
    NGramFingerprinter,
)
from bananas.worlds import (
    FingerprintRepresentationMaker,
    RepresentationMeanNormalizer,
    RepresentationMedianNormalizer,
    TanimotoMinMaxRepresentationMaker,
    SortedRowsRepresentation,
)

fingerprinter_by_name = {
    "maccs": MACCS(converter=SMILESToMol()),
    "rdkit2d": RDKitDescriptors2DFingerprinter(converter=SMILESToMol()),
    "rdkit3d": RDKitDescriptors3DFingerprinter(converter=ETKDG(seed=43)),
    "rdkitall": ConcatenatedFingerprinter(fingerprinters=(
        RDKitDescriptors2DFingerprinter(converter=SMILESToMol()),
        RDKitDescriptors3DFingerprinter(converter=ETKDG(seed=43)),
    )),
    "getaway": GETAWAY(converter=ETKDG(seed=43)),
    "whim": WHIM(converter=ETKDG(seed=43)),
    "rdf": RDF(converter=ETKDG(seed=43)),
    "morse": MORSE(converter=ETKDG(seed=43)),
    "autocorr3d": AUTOCORR3D(converter=ETKDG(seed=43)),
    "atompairs": AtomPairs(converter=SMILESToMol()),
    "torsion": TFD(converter=SMILESToMol()),
    "estatecount": EStateCount(converter=SMILESToMol()),
    "klekotaroth": KlekotaRoth(converter=SMILESToMol()),
    "morgan-2": Morgan(radius=2, use_chirality=True, use_bond_types=True, use_features=False, converter=SMILESToMol()),
    "morgan-3": Morgan(radius=3, use_chirality=True, use_bond_types=True, use_features=False, converter=SMILESToMol()),
    "morgan-4": Morgan(radius=4, use_chirality=True, use_bond_types=True, use_features=False, converter=SMILESToMol()),
    "morgan-2-F": Morgan(radius=2, use_chirality=True, use_bond_types=True, use_features=True, converter=SMILESToMol()),
    "morgan-3-F": Morgan(radius=3, use_chirality=True, use_bond_types=True, use_features=True, converter=SMILESToMol()),
    "morgan-4-F": Morgan(radius=4, use_chirality=True, use_bond_types=True, use_features=True, converter=SMILESToMol()),
    "rdkit1-7": RDKitFingerprinter(
        min_path=1, max_path=7, fp_size=2048, n_bits_per_hash=2,
        use_hs=True, tgt_density=0.0, min_size=128, branched_paths=True,
        use_bond_order=True, atom_invariants=0, from_atoms=0,
        atom_bits=None, bit_info=None, converter=SMILESToMol(),
    ),
    "rdkit3-5": RDKitFingerprinter(
        min_path=3, max_path=5, fp_size=2048, n_bits_per_hash=2,
        use_hs=True, tgt_density=0.0, min_size=128, branched_paths=True,
        use_bond_order=True, atom_invariants=0, from_atoms=0,
        atom_bits=None, bit_info=None, converter=SMILESToMol(),
    ),
    "molprint2d": MolPrint2D(),
    "spectrophores-20-3": Spectrophores(accuracy=20, resolution=3.0, converter=ETKDG(seed=43)),
    "spectrophores-20-2": Spectrophores(accuracy=20, resolution=2.0, converter=ETKDG(seed=43)),
    "spectrophores-20-4": Spectrophores(accuracy=20, resolution=4.0, converter=ETKDG(seed=43)),
    "spectrophores-15-3": Spectrophores(accuracy=15, resolution=3.0, converter=ETKDG(seed=43)),
    "spectrophores-30-3": Spectrophores(accuracy=30, resolution=3.0, converter=ETKDG(seed=43)),
    "mordred2d": Mordred2D(converter=SMILESToMol()),
    "mordred3d": Mordred3D(converter=ETKDG(seed=43)),
    "mordredall": ConcatenatedFingerprinter(fingerprinters=(
        Mordred2D(converter=SMILESToMol()),
        Mordred3D(converter=ETKDG(seed=43)),
    )),
    "ngramfingerprint1-7": NGramFingerprinter(ns=(1,2,3,4,5,6,7)),
    "ngramfingerprint3-5": NGramFingerprinter(ns=(3,4,5)),
}

COUNT_FINGERPRINTERS = [ # suitable for calculating tanimoto kernel
    "atompairs", "estatecount", "klekotaroth", "maccs", "molprint2d",
    "morgan-2", "morgan-2-F", "morgan-3", "morgan-3-F", "morgan-4",
    "morgan-4-F", "ngramfingerprint1-7", "ngramfingerprint3-5",
    "rdkit1-7", "rdkit3-5", "torsion", 
]
SPARSE_FINGERPRINTERS = [ # too big for being a representation
    "atompairs", "torsion", "morgan-2", "morgan-3", "morgan-4",
    "morgan-2-F", "morgan-3-F", "morgan-4-F", "ngramfingerprint1-7",
    "ngramfingerprint3-5", "molprint2d",
]
DENSE_FINGERPRINTERS = sorted(set(fingerprinter_by_name.keys())-set(SPARSE_FINGERPRINTERS))
DENSE_COUNT_FINGERPRINTERS = sorted(set(DENSE_FINGERPRINTERS)&set(COUNT_FINGERPRINTERS))
HASHED_SPARSE_FINGERPRINTERS = ["hashed1024-"+name for name in SPARSE_FINGERPRINTERS]
fingerprinter_by_name.update({
    hn: HashedFingerprinter(
        fingerprinter=fingerprinter_by_name[n],
        n_keys=1024,
        random_sign=True,
    ) for n, hn in zip(
        SPARSE_FINGERPRINTERS,
        HASHED_SPARSE_FINGERPRINTERS,
    )
})
ALL_FINGERPRINTERS = list(fingerprinter_by_name.keys())
ALL_DENSE_FINGERPRINTERS = DENSE_FINGERPRINTERS + HASHED_SPARSE_FINGERPRINTERS
FLAT_FINGERPRINTERS = [
    "maccs", "rdkit2d", "atompairs", "torsion", "estatecount", "klekotaroth",
    "morgan-2", "morgan-3", "morgan-4", "morgan-2-F", "morgan-3-F", "morgan-4-F",
    "rdkit1-7", "rdkit3-5", "molprint2d", "mordred2d", "ngramfingerprint1-7",
    "ngramfingerprint3-5", 
]
FLAT_FINGERPRINTERS += ["hashed1024-"+name for name in (set(SPARSE_FINGERPRINTERS)&set(FLAT_FINGERPRINTERS))]

def repr_factory(source_tr, fingerprint, kernel, sort_every_row, normalizer):
    def get_representation(source):
        fper = fingerprinter_by_name[fingerprint]
        fp_tr = fper(source=source_tr)
        fp = fper(source=source)
        if kernel is None:
            maker = FingerprintRepresentationMaker(fingerprint=fp_tr)
        elif kernel == "tanimoto":
            maker = TanimotoMinMaxRepresentationMaker(fingerprint=fp_tr)
        else:
            raise ValueError(kernel)
        repr_tr = maker(fingerprint=fp_tr)
        repr_ = maker(fingerprint=fp)
        if sort_every_row:
            repr_tr = SortedRowsRepresentation(source=repr_tr)
            repr_ = SortedRowsRepresentation(source=repr_)
        if normalizer is not None:
            if normalizer == "mean":
                normalizer_ = RepresentationMeanNormalizer(source=repr_tr)
            elif normalizer == "median":
                normalizer_ = RepresentationMedianNormalizer(source=repr_tr)
            else:
                raise ValueError(normalizer)
            repr_ = normalizer_(source=repr_)
        return repr_
    return get_representation


@mandalka.node
class Benchmarks2018SolutionTrained:
    def __init__(self, *, source, repr_kwargs, model, model_kwargs):
        self.get_representation = repr_factory(source, **repr_kwargs)
        self.model = globals()[model](
            source=self.get_representation(source),
            **model_kwargs,
        )
    @mandalka.lazy
    def __call__(self, source):
        return Benchmarks2018SolutionPrediction(source=source, trained=self)
@mandalka.node
class Benchmarks2018SolutionTrainedBuilder(mandalka.NodeBuilder):
    cls = Benchmarks2018SolutionTrained
@mandalka.node
class Benchmarks2018SolutionPrediction(StorageWorld):
    def build(self, *, source, trained):
        self.data = source.data.slice[:]
        pred = trained.model.predict(
            source=trained.get_representation(source)
        )
        for key in pred.data:
            if key[0] == "value_predicted":
                self.data[key] = pred.data.get_container(key)(pred.data[key])

### ### ### REGRESSION ### ### ###

SOLUTIONS_R = {}

### EXTRA TREES ###

for fingerprint in ALL_DENSE_FINGERPRINTERS:
    SOLUTIONS_R["ExtraTrees_fp:{}".format(fingerprint)] = Benchmarks2018SolutionTrainedBuilder(
        repr_kwargs={
            "fingerprint": fingerprint,
            "kernel": None,
            "sort_every_row": False,
            "normalizer": "median",
        },
        model="ExtraTreesRegressor",
        model_kwargs={
            "n_estimators": 50,
            "criterion": "mse",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.0,
            "max_features": "auto",
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.0,
            "min_impurity_split": None,
            "bootstrap": False,
        },
    )

### RANDOM FOREST ###

for fingerprint in ALL_DENSE_FINGERPRINTERS:
    SOLUTIONS_R["RandomForest_fp:{}_sort:False".format(fingerprint)] = Benchmarks2018SolutionTrainedBuilder(
        repr_kwargs={
            "fingerprint": fingerprint,
            "kernel": None,
            "sort_every_row": False,
            "normalizer": "median",
        },
        model="RandomForestRegressor",
        model_kwargs={
            "n_estimators": 60,
            "criterion": "mse",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.0,
            "max_features": "auto",
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.0,
            "min_impurity_split": None,
            "bootstrap": True,
        },
    )

### CONSTANT MODELS ###

SOLUTIONS_R.update({
    "PredictMedian": Benchmarks2018SolutionTrainedBuilder(
        repr_kwargs={
            "fingerprint": "molprint2d", # None
            "kernel": None,
            "sort_every_row": False,
            "normalizer": None,
        },
        model="ConstantRegressor",
        model_kwargs={
            "variant": "median",
        },
    ),
    "PredictMean": Benchmarks2018SolutionTrainedBuilder(
        repr_kwargs={
            "fingerprint": "molprint2d", # None
            "kernel": None,
            "sort_every_row": False,
            "normalizer": None,
        },
        model="ConstantRegressor",
        model_kwargs={
            "variant": "mean",
        },
    ),
})

### KNN ###

name_template = "KernelKNNRegressor_fp:{}_k:{}"
params = product(
    COUNT_FINGERPRINTERS,
    [1,2,3,4,5,6,7],
)
for fingerprint, k, in params:
    SOLUTIONS_R[name_template.format(fingerprint, k)] = Benchmarks2018SolutionTrainedBuilder(
        repr_kwargs={
            "fingerprint": fingerprint, # None
            "kernel": "tanimoto",
            "sort_every_row": False,
            "normalizer": None,
        },
        model="KernelKNNRegressor",
        model_kwargs={
            "k": k,
        },
    )

### LINEAR REGRESSION ###

name_template = "LinearRegression_fp:{}_ker:{}_sort:{}_norm:{}_alpha:{}"
all_params = [
    product(
        DENSE_COUNT_FINGERPRINTERS,
        [None],
        [False],
        ["mean"],
        [0.01, 0.1, 1., 10., None],
    ),
    product(
        ALL_DENSE_FINGERPRINTERS,
        [None],
        [False],
        ["median"],
        [0.01, 0.1, 1., 10., None],
    ),
    product(
        COUNT_FINGERPRINTERS,
        ["tanimoto"],
        [False],
        ["median", None],
        [0.01, 0.1, 1., 10., None],
    ),
]
for params in all_params:
    for fingerprint, kernel, sort_every_row, normalizer, alpha in params:
        SOLUTIONS_R[name_template.format(fingerprint, kernel, sort_every_row, normalizer, alpha)] = Benchmarks2018SolutionTrainedBuilder(
            repr_kwargs={
                "fingerprint": fingerprint,
                "kernel": kernel,
                "sort_every_row": sort_every_row,
                "normalizer": normalizer,
            },
            model="LinearRegression",
            model_kwargs={
                "alpha": alpha,
            },
        )


### ### ### CLASSIFICATION ### ### ###

SOLUTIONS_C = {}

### EXTRA TREES ###

for fingerprint in ALL_DENSE_FINGERPRINTERS:
    for class_weight in [None, "balanced"]:
        SOLUTIONS_C["ExtraTrees_fp:{}_class-weight:{}".format(fingerprint, class_weight)] = Benchmarks2018SolutionTrainedBuilder(
            repr_kwargs={
                "fingerprint": fingerprint,
                "kernel": None,
                "sort_every_row": False,
                "normalizer": "median",
            },
            model="ExtraTreesClassifier",
            model_kwargs={
                "n_estimators": 50,
                "criterion": "gini",
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "min_weight_fraction_leaf": 0.0,
                "max_features": "auto",
                "max_leaf_nodes": None,
                "min_impurity_decrease": 0.0,
                "min_impurity_split": None,
                "bootstrap": False,
                "class_weight": class_weight,
            },
        )

### RANDOM FOREST ###

for fingerprint in ALL_DENSE_FINGERPRINTERS:
    for class_weight in [None, "balanced"]:
        SOLUTIONS_C["RandomForest_fp:{}_class-weight:{}".format(fingerprint, class_weight)] = Benchmarks2018SolutionTrainedBuilder(
            repr_kwargs={
                "fingerprint": fingerprint,
                "kernel": None,
                "sort_every_row": False,
                "normalizer": "median",
            },
            model="RandomForestClassifier",
            model_kwargs={
                "n_estimators": 50,
                "criterion": "gini",
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "min_weight_fraction_leaf": 0.0,
                "max_features": "auto",
                "max_leaf_nodes": None,
                "min_impurity_decrease": 0.0,
                "min_impurity_split": None,
                "bootstrap": True,
                "class_weight": class_weight,
            },
        )

### CONSTANT MODELS ###

SOLUTIONS_C.update({
    "PredictConstant": Benchmarks2018SolutionTrainedBuilder(
        repr_kwargs={
            "fingerprint": "molprint2d", # None
            "kernel": None,
            "sort_every_row": False,
            "normalizer": None,
        },
        model="ConstantClassifier",
        model_kwargs={},
    ),
})

### KNN ###

name_template = "KernelKNNClassifier_fp:{}_k:{}"
params = product(
    COUNT_FINGERPRINTERS,
    [1,2,3,4,5,6,7],
)
for fingerprint, k, in params:
    SOLUTIONS_C[name_template.format(fingerprint, k)] = Benchmarks2018SolutionTrainedBuilder(
        repr_kwargs={
            "fingerprint": fingerprint, # None
            "kernel": "tanimoto",
            "sort_every_row": False,
            "normalizer": None,
        },
        model="KernelKNNClassifier",
        model_kwargs={
            "k": k,
        },
    )

### LOGISTIC REGRESSION ###

name_template = "TwoClassLogisticRegression_fp:{}_ker:{}_sort:{}_norm:{}_c:{}_class-weight:{}"
all_params = [
    product(
        DENSE_COUNT_FINGERPRINTERS,
        [None],
        [False],
        ["mean"],
        [0.01, 0.1, 1., 10.],
        [None, "balanced"],
    ),
    product(
        ALL_DENSE_FINGERPRINTERS,
        [None],
        [False],
        ["median"],
        [0.01, 0.1, 1., 10.],
        [None, "balanced"],
    ),
    product(
        COUNT_FINGERPRINTERS,
        ["tanimoto"],
        [False],
        ["median", None],
        [0.01, 0.1, 1., 10.],
        [None, "balanced"],
    ),
]
for params in all_params:
    for fingerprint, kernel, sort_every_row, normalizer, C, class_weight in params:
        SOLUTIONS_C[name_template.format(fingerprint, kernel, sort_every_row, normalizer, C, class_weight)] = Benchmarks2018SolutionTrainedBuilder(
            repr_kwargs={
                "fingerprint": fingerprint,
                "kernel": kernel,
                "sort_every_row": sort_every_row,
                "normalizer": normalizer,
            },
            model="TwoClassLogisticRegression",
            model_kwargs={
                "C": C,
                "class_weight": class_weight,
            },
        )
