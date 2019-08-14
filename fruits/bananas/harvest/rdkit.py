import numpy as np
from rdkit.Chem import AllChem, MACCSkeys, MolToSmiles, MolFromSmiles, MolFromSmarts, Descriptors
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Chem.EState.Fingerprinter import FingerprintMol as EStateFingerprinter
from scipy.sparse.csr import csr_matrix
import tqdm

from herbivores import CSRBuilder

def canonical_smiles(smiles):
    def _single(s):
        return MolToSmiles(
            MolFromSmiles(s),
            canonical=True,
            isomericSmiles=True,
        )
    return np.vectorize(_single, otypes=[smiles.dtype])(smiles)

def smiles_to_mol(smiles):
    return MolFromSmiles(smiles)

def embed_etkdg(mol, seed):
    from rdkit.Chem import AllChem, AddHs
    mol = AddHs(mol)
    params = AllChem.ETKDG()
    params.randomSeed = seed
    AllChem.EmbedMolecule(mol, params)
    return mol

def is_smiles_correct(arr_smiles):
    if len(arr_smiles) == 0:
        return np.array([], dtype=bool)
    else:
        return np.vectorize(
            lambda s: MolFromSmiles(s) is not None
        )(arr_smiles).nonzero()[0]

def _rdkit_dense_fingerprinter(mols, which):
    assert len(mols.shape) == 1, "'mols' must be one-dimensional"
    test = np.array(which(MolFromSmiles("C"))) # check shape and dtype
    assert len(test.shape) == 1
    result = np.zeros((mols.shape[0], test.shape[0]), dtype=test.dtype)
    for i, m in tqdm.tqdm(enumerate(mols)):
        result[i,:] = np.array(which(m))
    return result

def _rdkit_sparse_fingerprinter(mols, which):
    b = CSRBuilder()
    for mol in tqdm.tqdm(mols):
        d = which(mol).GetNonzeroElements()
        if len(d) > 0 and isinstance(list(d.keys())[0], int):
            d = {str(k): v for k, v in d.items()}
        b.add_row(d)
    return b.build()

def _zfill_dense_header(header):
    l = max([len(h) for h in header])
    return np.array([h.zfill(l) for h in header], dtype=np.str)

def murcko_scaffold(smiles, generic, isomeric):
    from rdkit.Chem.Scaffolds.MurckoScaffold import (
        GetScaffoldForMol,
        MakeScaffoldGeneric,
    )
    assert isinstance(generic, bool)
    assert isinstance(isomeric, bool)
    mol = MolFromSmiles(smiles)
    mol = GetScaffoldForMol(mol)
    if generic:
        mol = MakeScaffoldGeneric(mol)
    return MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)

### FINGERPRINTERS ###

def pairs(mols):
    return _rdkit_sparse_fingerprinter(
        mols,
        Pairs.GetAtomPairFingerprint,
    )

def torsions(mols):
    return _rdkit_sparse_fingerprinter(
        mols,
        Torsions.GetTopologicalTorsionFingerprint,
    )

def morgan(mols, radius, use_chirality, use_bond_types, use_features):
    return _rdkit_sparse_fingerprinter(
        mols,
        (lambda mol: AllChem.GetMorganFingerprint(
            mol=mol,
            radius=radius,
            useChirality=use_chirality,
            useBondTypes=use_bond_types,
            useFeatures=use_features,
            useCounts=True
        )),
    )

def rdk_fingerprinter(
        mols, min_path=1, max_path=7, fp_size=2048, n_bits_per_hash=2,
        use_hs=True, tgt_density=0.0, min_size=128, branched_paths=True,
        use_bond_order=True, atom_invariants=0, from_atoms=0,
        atom_bits=None, bit_info=None):
    return (
        _rdkit_dense_fingerprinter(
            mols=mols,
            which=(lambda mol: RDKFingerprint(
                mol, minPath=min_path, maxPath=max_path, fpSize=fp_size,
                nBitsPerHash=n_bits_per_hash, useHs=use_hs,
                tgtDensity=tgt_density, minSize=min_size,
                branchedPaths=branched_paths, useBondOrder=use_bond_order, atomInvariants=atom_invariants, fromAtoms=from_atoms,
                atomBits=atom_bits, bitInfo=bit_info,
            )),
        ),
        _zfill_dense_header(np.array(range(fp_size), dtype=np.str)),
    )

def estate_count(mols):
    fp = _rdkit_dense_fingerprinter(
        mols=mols,
        which=(lambda mol: EStateFingerprinter(mol)[0]),
    )
    return fp, _zfill_dense_header(np.array(range(fp.shape[1]), dtype=np.str))

def klekota_roth(mols):
    from ._krfp_smarts import KRFP_SMARTS
    return (
        _rdkit_dense_fingerprinter(
            mols=mols,
            which=(lambda mol: [len(mol.GetSubstructMatches(MolFromSmarts(smarts))) for smarts in KRFP_SMARTS]),
        ),
        _zfill_dense_header(np.array(range(len(KRFP_SMARTS)), dtype=np.str)),
    )

def maccs(mols):
    fp = _rdkit_dense_fingerprinter(
        mols=mols,
        which=MACCSkeys.GenMACCSKeys)
    return fp, _zfill_dense_header(np.array(range(fp.shape[1]), dtype=np.str))

def rdkit_descriptors_2d(mols):
    knames, l_keys = map(list, zip(*Descriptors.descList))
    knames = np.array(knames)
    fp = _rdkit_dense_fingerprinter(
        mols=mols,
        which=(lambda mol: [key(mol) for key in l_keys]),
    )
    return fp, knames

def rdkit_descriptors_3d(mols):
    from rdkit.Chem.rdMolDescriptors import (
        CalcSpherocityIndex,
        CalcAsphericity,
        CalcEccentricity,
        CalcInertialShapeFactor,
        CalcRadiusOfGyration,
        CalcNPR1,
        CalcNPR2,
        CalcPMI1,
        CalcPMI2,
        CalcPMI3,
        CalcPBF,
    )
    fdesc = [CalcSpherocityIndex, CalcAsphericity, CalcEccentricity, CalcInertialShapeFactor, CalcRadiusOfGyration, CalcNPR1, CalcNPR2, CalcPMI1, CalcPMI2, CalcPMI3, CalcPBF]
    header = np.array(['SpherocityIndex', 'Asphericity', 'Eccentricity', 'InertialShapeFactor', 'RadiusOfGyration', 'NPR1', 'NPR2', 'PMI1', 'PMI2', 'PMI3', 'PBF'], dtype=np.str)
    result = np.empty((len(mols), len(fdesc)), dtype=np.float32)
    for i, m in enumerate(tqdm.tqdm(mols)):
        if m.GetNumConformers() == 1:
            for j in range(len(fdesc)):
                result[i,j] = fdesc[j](m)
        elif m.GetNumConformers() == 0:
            result[i] = np.nan
        else:
            raise ValueError("every molecule must have at most 1 conformer")
    return result, header

from rdkit.Chem.rdMolDescriptors import CalcGETAWAY, CalcWHIM, CalcRDF, CalcMORSE, CalcAUTOCORR3D

def _ldesc3d(mols, calculator, c_length):
    name = calculator.__name__[4:] # CalcNAME
    n_digits = int(np.ceil(np.log10(c_length)))
    header = np.array([name + str(i+1).zfill(n_digits) for i in range(c_length)], dtype=np.str)
    result = np.empty((len(mols), c_length), dtype=np.float32)
    for i, m in enumerate(tqdm.tqdm(mols)):
        if m.GetNumConformers() == 1:
            result[i,:] = np.array(calculator(m))
        elif m.GetNumConformers() == 0:
            result[i] = np.nan
        else:
            raise ValueError("every molecule must have at most 1 conformer")
    return result, header

def getaway(mols):
    return _ldesc3d(mols, CalcGETAWAY, 273)

def whim(mols):
    return _ldesc3d(mols, CalcWHIM, 114)

def rdf(mols):
    return _ldesc3d(mols, CalcRDF, 210)

def morse(mols):
    return _ldesc3d(mols, CalcMORSE, 224)

def autocorr3d(mols):
    return _ldesc3d(mols, CalcAUTOCORR3D, 80)
