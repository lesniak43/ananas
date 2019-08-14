import numpy as np

import mandalka

from ...harvest.rdkit import murcko_scaffold
from .. import ProxyWorld, ProxyContainer, Table, Variable
from . import DynamicGroupSplitter

@mandalka.node
class MurckoScaffold(ProxyWorld):
    def build_proxy(self, *, source, generic, isomeric):
        self.data = source.data.proxy
        self.data["scaffolds"] = ProxyContainer(
            (lambda : np.vectorize(lambda s: murcko_scaffold(s, generic, isomeric), otypes=(np.object,))(self.data["smiles"])),
            Table,
        )

@mandalka.node
class MurckoScaffoldSplit(DynamicGroupSplitter):
    def build(self, *, source, generic, isomeric):
        self.data = source.data.slice[:]
        del self.data["groups"]
        del self.data["n_groups"]
        del self.data["scaffolds"]
        scaffolds = MurckoScaffold(
            source=source,
            generic=generic,
            isomeric=isomeric,
        ).data["scaffolds"]
        def _enumerate(xs):
            _xs = dict([reversed(tup) for tup in enumerate(sorted(set(xs)))])
            return np.vectorize(lambda x: _xs[x])(xs)
        self.data["groups"] = Table(_enumerate(scaffolds))
        self.data["n_groups"] = Variable(len(set(scaffolds)))
        self.data["scaffolds"] = Table(scaffolds)
