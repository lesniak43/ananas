import mandalka
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import csr_matrix
from numba import njit

from herbivores import merge_columns

from .. import StorageWorld, Table, Table2D, DataDict

@mandalka.node
class TanimotoMinMaxKernel:
    def __init__(self, *, source):
        self.data = source.data.slice[:]
        del self.data["tanimoto_kernel"]
        self.data["tanimoto_kernel"] = Table2D(
            tanimoto_minmax_similarity(
                source.data[("fingerprint", "data")],
                source.data[("fingerprint", "data")],
            )
        )
        self.data.lock()

@mandalka.node
class TanimotoMinMaxSimilarity:
    def __init__(self, *, tr, te):
        self.data = DataDict(source_dirname=None)
        self.data["tanimoto_kernel"] = Table(
            tanimoto_minmax_similarity(*merge_columns(
                te.data[("fingerprint", "data")],
                te.data[("fingerprint", "keys")],
                tr.data[("fingerprint", "data")],
                tr.data[("fingerprint", "keys")],
                0.,
            )[:2])
        )
        self.data.lock()

@mandalka.node
class MaxTanimotoMinMaxSimilarity(StorageWorld):
    def build(self, *, tr, te):
        k_te = tanimoto_minmax_similarity(*merge_columns(
            te.data[("fingerprint", "data")],
            te.data[("fingerprint", "keys")],
            tr.data[("fingerprint", "data")],
            tr.data[("fingerprint", "keys")],
            0.,
        )[:2])
        self.data["kernel_max"] = Table(np.max(k_te, axis=1))

@mandalka.node
class TanimotoMinMaxProjectionNorms(StorageWorld):
    def build(self, *, tr, te, fingerprinter):
        def _norms(k_tr, k_te):
            assert np.max(np.abs(
                k_tr[range(len(k_tr)),range(len(k_tr))].ravel()-1.)) < 1e-6
            new_vectors = k_te
            l = np.linalg.lstsq(k_tr, new_vectors.T)[0].T
            norms = np.sum(new_vectors*l, axis=1)
            return norms
        tr = fingerprinter(source=tr)
        te = fingerprinter(source=te)
        k_tr = tanimoto_minmax_similarity(
            tr.data[("fingerprint", "data")],
            tr.data[("fingerprint", "data")],
        )
        k_te = tanimoto_minmax_similarity(*merge_columns(
            te.data[("fingerprint", "data")],
            te.data[("fingerprint", "keys")],
            tr.data[("fingerprint", "data")],
            tr.data[("fingerprint", "keys")],
            0.,
        )[:2])
        self.data["projection_norm"] = Table(_norms(k_tr, k_te))

def tanimoto_minmax_similarity(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return ndarray_tanimoto_minmax_similarity(a, b)
    elif isinstance(a, csr_matrix) and isinstance(b, csr_matrix):
        return csr_tanimoto_minmax_similarity(a, b)
    else:
        raise TypeError("'a' and 'b' must both be numpy.ndarray or scipy.sparse.csr_matrix, not {} and {}".format(type(a), type(b)))

def ndarray_tanimoto_minmax_similarity(a, b):
    assert isinstance(a, np.ndarray), "'a' must be numpy.ndarray"
    assert isinstance(b, np.ndarray), "'b' must be numpy.ndarray"
    assert a.shape[1] == b.shape[1], "number of columns must be equal"
    assert len(a.shape) == len(b.shape) == 2
    assert np.all(a >= 0)
    assert np.all(b >= 0)
    @njit
    def K(a, b):
        result = np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
        for i in range(a.shape[0]):
            for j in range(b.shape[0]):
                for k in range(a.shape[1]):
                    if a[i,k] > 0 and b[j,k] > 0:
                        result[i,j] += min(a[i,k], b[j,k])
        return result
    mins = K(a, b)
    sums = np.array(a.sum(axis=1)).reshape(-1,1) + np.array(b.sum(axis=1)).reshape(1,-1).astype(np.float32)
    kernel = mins / (sums - mins)
    kernel[np.isnan(kernel)] = 1. # here 0/0 = 1
    return kernel

def csr_tanimoto_minmax_similarity(a, b):
    assert isinstance(a, csr_matrix), "'a' must be scipy.sparse.csr_matrix"
    assert isinstance(b, csr_matrix), "'b' must be scipy.sparse.csr_matrix"
    assert a.shape[1] == b.shape[1], "number of columns must be equal"
    assert np.all(a.data >= 0)
    assert np.all(b.data >= 0)
    a = a.copy() # we modify index inplace
    b = b.copy()
    a.has_sorted_indices = False
    a.sort_indices()
    b.has_sorted_indices = False
    b.sort_indices()
    @njit
    def K(ad, ai, aptr, bd, bi, bptr):
        len_a = aptr.shape[0] - 1
        len_b = bptr.shape[0] - 1
        result = np.zeros((len_a, len_b), dtype=np.float32)
        for a_row in range(len_a):
            for b_row in range(len_b):
                _ai = aptr[a_row]
                _bi = bptr[b_row]
                # here we assume that ai and bi are (per row) sorted
                while _ai < aptr[a_row+1] and _bi < bptr[b_row+1]:
                    if ai[_ai] < bi[_bi]:
                        _ai += 1
                    elif ai[_ai] > bi[_bi]:
                        _bi += 1
                    else:
                        result[a_row, b_row] += min(ad[_ai], bd[_bi])
                        _ai += 1
                        _bi += 1
        return result
    mins = K(a.data, a.indices, a.indptr, b.data, b.indices, b.indptr)
    sums = np.array(a.sum(axis=1)).reshape(-1,1) + np.array(b.sum(axis=1)).reshape(1,-1).astype(np.float32)
    kernel = mins / (sums - mins)
    kernel[np.isnan(kernel)] = 1. # here 0/0 = 1
    return kernel
