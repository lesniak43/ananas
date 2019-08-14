import numpy as np
import mandalka

from .. import StorageWorld

@mandalka.node
class ConnectedSmiles(StorageWorld):

    """
    Drop rows with SMILES containing '.'
    """

    def build(self, source):
        self.data = source.data.slice[
            np.vectorize(
                lambda s: '.' not in s, otypes=[np.bool]
            )(source.data["smiles"])
        ]
