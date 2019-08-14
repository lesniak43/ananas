import mandalka
from crisper.datadict import MandalkaDictStorage as StorageWorld
from crisper.datadict import MandalkaDictProxy as ProxyWorld
from crisper.datadict import (
    DataDict,
    Table,
    Table2D,
    Variable,
    Header,
    ProxyContainer,
)

from ._datasets._SQLiteDatabase import SQLiteDatabase
from ._datasets._WarszyckiBioactivity import (
    WarszyckiBioactivityRaw,
    WarszyckiBioactivity,
    WarszyckiLogKi)
from ._datasets._Subset import (
    Subset,
    SubsetData,
    Slice,
)
from ._datasets._ParentMolecules import ParentMolecules
from ._datasets._ConnectedSmiles import ConnectedSmiles
from ._datasets._CanonicalSmiles import CanonicalSmiles
from ._datasets._UniqueSmiles import UniqueSmiles
from ._datasets._EarliestYear import EarliestYear
from ._datasets._TSNE import KernelTSNE

from ._fingerprints import (
    HashedFingerprinter,
    ConcatenatedFingerprinter,
)
from ._fingerprints._rdkit import (
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
)
from ._fingerprints._openbabel import (
    MolPrint2D,
    Spectrophores,
)
from ._fingerprints._mordred import Mordred2D, Mordred3D
from ._fingerprints._ngram import NGramFingerprinter
from ._fingerprints._gmum import (
    DudaBDC,
    DudaUSR,
    DudaPCASH,
    DudaRIFSH,
)

from ._kernels._tanimoto import (
    TanimotoMinMaxKernel,
    TanimotoMinMaxSimilarity,
    MaxTanimotoMinMaxSimilarity,
    TanimotoMinMaxProjectionNorms,
)

from ._splits._CV import CrossValidation
from ._splits._years import YearSplit, FullYearSplit
from ._splits._balanced_agglomerative_clustering import (
    AgglomerativeClustering,
    BalancedAgglomerativeClustering,
)
from ._splits._SpectralClustering import SpectralClustering
from ._splits._scaffold import (
    MurckoScaffold,
    MurckoScaffoldSplit,
)
from ._splits._paper import PaperSplit

from ._models.regression.ExtraTrees import ExtraTreesRegressor
from ._models.regression.RandomForest import RandomForestRegressor
from ._models.regression.Linear import LinearRegression
from ._models.regression.KNN import KernelKNNRegressor
from ._models.regression.Constant import ConstantRegressor
from ._models.classification.ExtraTrees import ExtraTreesClassifier
from ._models.classification.RandomForest import RandomForestClassifier
from ._models.classification.KNN import KernelKNNClassifier
from ._models.classification.Constant import ConstantClassifier
from ._models.classification.Linear import TwoClassLogisticRegression

from ._representation import(
    FingerprintRepresentationMaker,
    RepresentationMeanNormalizer,
    RepresentationMedianNormalizer,
    TanimotoMinMaxRepresentationMaker,
    SortedRowsRepresentation,
)

from ._mols import(
    SMILESToMol,
    ETKDG,
)

@mandalka.node
class StoredCopy(StorageWorld):
    def build(self, *, source):
        self.data = source.data.slice[:]
