from itertools import groupby

import mandalka
import numpy as np

from .. import StorageWorld

@mandalka.node
class EarliestYear(StorageWorld):

    """
    Set 'year' column to min(year) per uid.
    Assume Nan values were replaced with 0
    Assert min(year) >= 0
    Filter uids with min(year) == 0
    """

    def build(self, source):

        u = source.data["uid"]
        y = source.data["year"]
        du = source.data["doc_uid"]

        keep_uid = {}
        earliest_year = {}
        earliest_year_doc_uid = {}
        key = lambda x: x[0]
        for uid, g in groupby(sorted(zip(u, y, du), key=key), key):
            gu, gy, gdu = zip(*g)
            assert gu[0] == uid
            idx = np.lexsort((gdu, gy))[0]
            assert gy[idx] >= 0
            earliest_year[uid] = gy[idx]
            earliest_year_doc_uid[uid] = gdu[idx]
            keep_uid[uid] = (gy[idx] > 0)
        e_year = np.vectorize(lambda uid: earliest_year[uid])(u)
        e_doc_uid = np.vectorize(lambda uid: earliest_year_doc_uid[uid])(u)
        mask = np.vectorize(lambda uid: keep_uid[uid])(u)

        self.data = source.data.slice[mask]
        self.data["year"] = self.data.get_container("year")(e_year[mask])
        self.data["doc_uid"] = self.data.get_container("doc_uid")(e_doc_uid[mask])
