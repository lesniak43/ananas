import mandalka

from crisper.fingerprint import cached_fingerprinter

from ...harvest.rdkit import (
    maccs,
    rdkit_descriptors_2d,
    rdkit_descriptors_3d,
    getaway,
    whim,
    rdf,
    morse,
    autocorr3d,
    pairs,
    torsions,
    morgan,
    rdk_fingerprinter,
    estate_count,
    klekota_roth,
)

from . import CachedFingerprinter

@mandalka.node
class MACCS(CachedFingerprinter):
    def __init__(self, *, converter):
        self.converter = converter
    @cached_fingerprinter
    def calculate(self, smiles):
        return maccs(self.converter(smiles))

@mandalka.node
class RDKitDescriptors2DFingerprinter(CachedFingerprinter):
    def __init__(self, *, converter):
        self.converter = converter
    @cached_fingerprinter
    def calculate(self, smiles):
        return rdkit_descriptors_2d(self.converter(smiles))

@mandalka.node
class RDKitDescriptors3DFingerprinter(CachedFingerprinter):
    def __init__(self, *, converter):
        self.converter = converter
    @cached_fingerprinter
    def calculate(self, smiles):
        return rdkit_descriptors_3d(self.converter(smiles))

@mandalka.node
class GETAWAY(CachedFingerprinter):
    def __init__(self, *, converter):
        self.converter = converter
    @cached_fingerprinter
    def calculate(self, smiles):
        return getaway(self.converter(smiles))

@mandalka.node
class WHIM(CachedFingerprinter):
    def __init__(self, *, converter):
        self.converter = converter
    @cached_fingerprinter
    def calculate(self, smiles):
        return whim(self.converter(smiles))

@mandalka.node
class RDF(CachedFingerprinter):
    def __init__(self, *, converter):
        self.converter = converter
    @cached_fingerprinter
    def calculate(self, smiles):
        return rdf(self.converter(smiles))

@mandalka.node
class MORSE(CachedFingerprinter):
    def __init__(self, *, converter):
        self.converter = converter
    @cached_fingerprinter
    def calculate(self, smiles):
        return morse(self.converter(smiles))

@mandalka.node
class AUTOCORR3D(CachedFingerprinter):
    def __init__(self, *, converter):
        self.converter = converter
    @cached_fingerprinter
    def calculate(self, smiles):
        return autocorr3d(self.converter(smiles))

@mandalka.node
class AtomPairs(CachedFingerprinter):
    def __init__(self, *, converter):
        self.converter = converter
    @cached_fingerprinter
    def calculate(self, smiles):
        return pairs(self.converter(smiles))

@mandalka.node
class TFD(CachedFingerprinter):
    def __init__(self, *, converter):
        self.converter = converter
    @cached_fingerprinter
    def calculate(self, smiles):
        return torsions(self.converter(smiles))

@mandalka.node
class EStateCount(CachedFingerprinter):
    def __init__(self, *, converter):
        self.converter = converter
    @cached_fingerprinter
    def calculate(self, smiles):
        return estate_count(self.converter(smiles))

@mandalka.node
class KlekotaRoth(CachedFingerprinter):
    def __init__(self, *, converter):
        self.converter = converter
    @cached_fingerprinter
    def calculate(self, smiles):
        return klekota_roth(self.converter(smiles))

@mandalka.node
class Morgan(CachedFingerprinter):
    def __init__(self, *, radius, use_chirality, use_bond_types, use_features, converter):
        self.radius = radius
        self.converter = converter
        self.use_chirality = use_chirality
        self.use_bond_types = use_bond_types
        self.use_features = use_features
    @cached_fingerprinter
    def calculate(self, smiles):
        return morgan(
            mols=self.converter(smiles),
            radius=self.radius,
            use_chirality=self.use_chirality,
            use_bond_types=self.use_bond_types,
            use_features=self.use_features,
        )

@mandalka.node
class RDKitFingerprinter(CachedFingerprinter):
    def __init__(
            self, *, min_path, max_path, fp_size, n_bits_per_hash,
            use_hs, tgt_density, min_size, branched_paths,
            use_bond_order, atom_invariants, from_atoms,
            atom_bits, bit_info, converter):
        self.min_path = min_path
        self.max_path = max_path
        self.fp_size = fp_size
        self.n_bits_per_hash = n_bits_per_hash
        self.use_hs = use_hs
        self.tgt_density = tgt_density
        self.min_size = min_size
        self.branched_paths = branched_paths
        self.use_bond_order = use_bond_order
        self.atom_invariants = atom_invariants
        self.from_atoms = from_atoms
        self.atom_bits = atom_bits
        self.bit_info = bit_info
        self.converter = converter
    @cached_fingerprinter
    def calculate(self, smiles):
        return rdk_fingerprinter(
            self.converter(smiles),
            min_path=self.min_path,
            max_path=self.max_path,
            fp_size=self.fp_size,
            n_bits_per_hash=self.n_bits_per_hash,
            use_hs=self.use_hs,
            tgt_density=self.tgt_density,
            min_size=self.min_size,
            branched_paths=self.branched_paths,
            use_bond_order=self.use_bond_order,
            atom_invariants=self.atom_invariants,
            from_atoms=self.from_atoms,
            atom_bits=self.atom_bits,
            bit_info=self.bit_info,
        )
