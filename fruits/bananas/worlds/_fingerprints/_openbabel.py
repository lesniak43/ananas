import mandalka

from crisper.fingerprint import cached_fingerprinter

from ...harvest.openbabel import (
    molprint2d_count_fingerprinter,
    spectrophores,
)
from . import CachedFingerprinter

@mandalka.node
class MolPrint2D(CachedFingerprinter):
    @cached_fingerprinter
    def calculate(self, smiles):
        return molprint2d_count_fingerprinter(smiles)

@mandalka.node
class Spectrophores(CachedFingerprinter):
    def __init__(self, *, accuracy, resolution, converter):
        assert accuracy in (1, 2, 5, 10, 15, 20, 30, 36, 45, 60)
        assert isinstance(resolution, float)
        assert resolution > 0.
        self.accuracy = accuracy
        self.resolution = resolution
        self.converter = converter
    @cached_fingerprinter
    def calculate(self, smiles):
        return spectrophores(
            self.converter(smiles),
            accuracy=self.accuracy,
            resolution=self.resolution,
        )
