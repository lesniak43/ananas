try:
    from gmum_bananas.fingerprinters import duda
except ModuleNotFoundError:
    duda = None

import mandalka

from crisper.fingerprint import cached_fingerprinter

from . import CachedFingerprinter

@mandalka.node
class DudaBDC(CachedFingerprinter):
    def __init__(self, *, converter):
        self.converter = converter
    @cached_fingerprinter
    def calculate(self, smiles):
        return duda.BDC(self.converter(smiles))

@mandalka.node
class DudaUSR(CachedFingerprinter):
    def __init__(self, *, converter):
        self.converter = converter
    @cached_fingerprinter
    def calculate(self, smiles):
        return duda.USR(self.converter(smiles))

@mandalka.node
class DudaPCASH(CachedFingerprinter):
    def __init__(self, *, max_l, converter):
        self.converter = converter
        self.max_l = max_l
    @cached_fingerprinter
    def calculate(self, smiles):
        return duda.PCA_SH(self.converter(smiles), max_l=self.max_l)

@mandalka.node
class DudaRIFSH(CachedFingerprinter):
    def __init__(self, *, n_harmonics, converter):
        self.converter = converter
        self.n_harmonics = n_harmonics
    @cached_fingerprinter
    def calculate(self, smiles):
        return duda.RIF_SH(
            self.converter(smiles),
            n_harmonics=self.n_harmonics,
        )
