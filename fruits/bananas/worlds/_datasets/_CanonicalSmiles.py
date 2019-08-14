import numpy as np

import mandalka

from .. import StorageWorld, Table
from ...harvest.rdkit import canonical_smiles
from ...harvest.rdkit import is_smiles_correct as rdkit_is_smiles_correct
from ...harvest.openbabel import is_smiles_correct as openbabel_is_smiles_correct

@mandalka.node
class CanonicalSmiles(StorageWorld):

    """
    Remove rows with SMILES that are invalid according
    to rdkit or openbabel.
    Change SMILES to canonical form.
    """

    def build(self, source):
        rdkit_idx = rdkit_is_smiles_correct(source.data["smiles"])
        canonical = canonical_smiles(source.data["smiles"][rdkit_idx])
        openbabel_idx = openbabel_is_smiles_correct(canonical)
        final_canonical = canonical[openbabel_idx]
        ok_idx = rdkit_idx[openbabel_idx]

        self.data = source.data.slice[ok_idx]
        self.data["noncanonical_smiles"] = Table(self.data["smiles"])
        self.data["smiles"] = self.data.get_container("smiles")(final_canonical)
