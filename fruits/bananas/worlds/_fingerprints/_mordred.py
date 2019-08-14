import mandalka
from crisper.fingerprint import cached_fingerprinter

from ...harvest.mordred import mordred_fingerprint2d, mordred_fingerprint3d
from . import CachedFingerprinter, ConcatenatedFingerprinter

@mandalka.node
class Mordred2D(CachedFingerprinter):
    def __init__(self, *, converter):
        self.converter = converter
    @cached_fingerprinter(max_request=50)
    def calculate(self, smiles):
        return mordred_fingerprint2d(self.converter(smiles))

@mandalka.node
class Mordred3D(CachedFingerprinter):
    def __init__(self, *, converter):
        self.converter = converter
    @cached_fingerprinter(max_request=50)
    def calculate(self, smiles):
        return mordred_fingerprint3d(self.converter(smiles))
