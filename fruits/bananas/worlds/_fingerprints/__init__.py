import numpy as np
from scipy.sparse import coo_matrix

import mandalka
from herbivores import hashed_columns

from .. import StorageWorld, ProxyWorld, ProxyContainer, Table, Header

class Fingerprinter:
    def __init__(self):
        pass
    def __call__(self, *, source):
        return Fingerprint(source=source, fingerprinter=self)
    def calculate(self, smiles):
        raise NotImplementedError()

@mandalka.node
class Fingerprint(StorageWorld):
    def build(self, *, source, fingerprinter):
        self.data = source.data.slice[:]
        fingerprint, keys = fingerprinter.calculate(source.data["smiles"])
        del self.data[("fingerprint", "data")]
        del self.data[("fingerprint", "keys")]
        self.data[("fingerprint", "data")] = Table(fingerprint)
        self.data[("fingerprint", "keys")] = Header(keys)

class CachedFingerprinter:
    def __init__(self):
        pass
    def __call__(self, *, source):
        return CachedFingerprint(source=source, fingerprinter=self)
    def calculate(self, smiles):
        raise NotImplementedError()

@mandalka.node
class CachedFingerprint(ProxyWorld):
    def build_proxy(self, *, source, fingerprinter):
        self.data = source.data.proxy
        del self.data[("fingerprint", "data")]
        del self.data[("fingerprint", "keys")]
        # fingerprint, keys = fingerprinter.calculate(source.data["smiles"])
        self.data[("fingerprint", "data")] = ProxyContainer(
            (lambda : fingerprinter.calculate(source.data["smiles"])[0]),
            Table,
        )
        self.data[("fingerprint", "keys")] = ProxyContainer(
            (lambda : fingerprinter.calculate(source.data["smiles"])[1]),
            Header,
        )

@mandalka.node
class HashedFingerprinter:
    def __init__(self, *, fingerprinter, n_keys, random_sign):
        assert isinstance(n_keys, int) and n_keys > 0
        assert isinstance(random_sign, bool)
        self.fingerprinter = fingerprinter
        self.n_keys = n_keys
        self.random_sign = random_sign
    def __call__(self, *, source):
        return HashedFingerprint(
            source=self.fingerprinter(source=source),
            n_keys=self.n_keys,
            random_sign=self.random_sign,
        )

@mandalka.node
class HashedFingerprint(ProxyWorld):
    def build_proxy(self, *, source, n_keys, random_sign):
        self.data = source.data.proxy
        del self.data[("fingerprint", "keys")]
        def new_keys():
            result = np.array(range(n_keys), dtype=np.str)
            l = max([len(k) for k in result])
            return np.array([k.zfill(l) for k in result], dtype=np.str)
        self.data[("fingerprint", "keys")] = ProxyContainer(
            new_keys,
            Header,
        )
        del self.data[("fingerprint", "data")]
        self.data[("fingerprint", "data")] = ProxyContainer(
            (lambda : hashed_columns(
                source.data[("fingerprint", "data")],
                source.data[("fingerprint", "keys")],
                n_keys,
                random_sign,
            )),
            Table,
        )

@mandalka.node
class ConcatenatedFingerprinter:
    def __init__(self, fingerprinters):
        assert isinstance(fingerprinters, tuple), "'fingerprinters' must be a tuple"
        self.fingerprinters = fingerprinters
    def __call__(self, *, source):
        return ConcatenatedFingerprint(source=source, fingerprinter=self)

@mandalka.node
class ConcatenatedFingerprint(ProxyWorld):
    def build_proxy(self, *, source, fingerprinter):
        self.data = source.data.proxy
        del self.data[("fingerprint", "data")]
        del self.data[("fingerprint", "keys")]
        self.data[("fingerprint", "data")] = ProxyContainer(
            (lambda : np.concatenate(
                [fpr(source=source).data[("fingerprint", "data")] \
                    for fpr in fingerprinter.fingerprinters],
                    axis=1
            )),
            Table,
        )
        self.data[("fingerprint", "keys")] = ProxyContainer(
            (lambda : np.concatenate(
                [fpr(source=source).data[("fingerprint", "keys")] \
                    for fpr in fingerprinter.fingerprinters],
                    axis=0
            )),
            Header,
        )
