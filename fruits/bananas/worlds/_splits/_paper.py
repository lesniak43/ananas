import numpy as np

import mandalka

from ...harvest.rdkit import murcko_scaffold
from .. import Table, Variable
from . import DynamicGroupSplitter

@mandalka.node
class PaperSplit(DynamicGroupSplitter):
    def build(self, *, source):
        self.data = source.data.slice[:]
        del self.data["groups"]
        del self.data["n_groups"]
        def _enumerate(xs):
            _xs = dict([reversed(tup) for tup in enumerate(sorted(set(xs)))])
            return np.vectorize(lambda x: _xs[x])(xs)
        self.data["groups"] = Table(_enumerate(
            [int(uid[6:]) for uid in self.data["doc_uid"]]
        ))
        self.data["n_groups"] = Variable(len(set(self.data["groups"])))
