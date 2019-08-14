import numpy as np
import tqdm

import mandalka

from ..harvest.rdkit import smiles_to_mol, embed_etkdg

### CONVERTERS ###

## all converters MUST work element-wise

@mandalka.node
class SMILESToMol:
    def __init__(self):
        pass
    def __call__(self, smiles):
        assert isinstance(smiles, np.ndarray)
        return np.array([smiles_to_mol(s) for s in smiles], dtype=np.object)

@mandalka.node
class ETKDG:
    def __init__(self, seed):
        self.seed = seed
    def __call__(self, smiles):
        assert isinstance(smiles, np.ndarray)
        mols = []
        for s in tqdm.tqdm(smiles):
            mols.append(embed_etkdg(smiles_to_mol(s), self.seed))
        return np.array(mols, dtype=np.object)
