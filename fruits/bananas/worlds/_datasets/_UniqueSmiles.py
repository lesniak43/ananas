from itertools import groupby

import mandalka
import numpy as np

from .. import StorageWorld

@mandalka.node
class UniqueSmiles(StorageWorld):

    """
    Chemical compounds are defined by their SMILES.
    We don't want to have two different uids with the same SMILES.
    Group uids by SMILES, replace all with (arbitrarily) the first uid.
    """

    def build(self, source):
        new_uids = {}
        key = lambda tup: tup[0]
        for k, g in groupby(sorted(zip(source.data["smiles"], source.data["uid"]), key=key), key):
            gs, gu = zip(*g)
            assert len(set(gs)) == 1
            new_uid = sorted(gu)[0]
            for uid in gu:
                new_uids[uid] = new_uid
        new_uids = np.vectorize(lambda u: new_uids[u], otypes=[source.data["uid"].dtype])(source.data["uid"])

        self.data = source.data.slice[:]
        self.data["uid"] = self.data.get_container("uid")(new_uids)
