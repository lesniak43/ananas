from collections import Counter

import tqdm

from crisper.fingerprint import cached_fingerprinter
from herbivores import CSRBuilder
import mandalka

from . import CachedFingerprinter

def _assert_sorted_nonempty_tuple_of_unique_positive_ints(ns):
    assert \
        isinstance(ns, tuple) and \
        len(ns) > 0 and \
        all([isinstance(n,int) for n in ns]) and \
        all([n>0 for n in ns]) and \
        ns == tuple(sorted(ns)) and \
        len(ns) == len(set(ns)), \
            "'ns' must be a sorted nonempty tuple of unique positive int"

@mandalka.node
class NGramFingerprinter(CachedFingerprinter):
    def __init__(self, ns):
        _assert_sorted_nonempty_tuple_of_unique_positive_ints(ns)
        self.ns = ns
    @cached_fingerprinter(max_request=5000)
    def calculate(self, smiles):
        return ngram_fingerprinter(smiles, self.ns)

def ngram_fingerprinter(smiles, ns):
    _assert_sorted_nonempty_tuple_of_unique_positive_ints(ns)
    builder = CSRBuilder()
    for _smiles in tqdm.tqdm(smiles):
        builder.add_row(Counter([s for _s in _smiles.split('.') for s in [s for n in ns for s in [_s[i:i+n] for i in range(len(_s)-n+1)]]]))
    return builder.build()
