import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix

_sparse_types = (csr_matrix, csc_matrix, coo_matrix)

class CSRBuilder:
    def __init__(self):
        self.data = []
        self.row_idx = []
        self.col_idx = []
        self.h_axis = []
        self._curr_h_axis_idx = 0
        self._curr_v_axis_idx = 0
        self._h_axis_idx = {}
    def add_row(self, d):
        # d - dict, {header_name: array_value, ...}
        assert all([isinstance(k, str) for k in d.keys()]), "'d' keys must be str, not {}".format(d.keys())
        for key, value in sorted(d.items()):
            self.row_idx.append(self._curr_v_axis_idx)
            if not key in self._h_axis_idx:
                self._h_axis_idx[key] = self._curr_h_axis_idx
                self._curr_h_axis_idx += 1
            self.col_idx.append(self._h_axis_idx[key])
            self.data.append(value)
        self._curr_v_axis_idx += 1
    def build(self):
        header = np.array([tup[0] for tup in sorted(self._h_axis_idx.items(), key=lambda x:x[1])])
        arr = csr_matrix(
            (self.data, (self.row_idx, self.col_idx)),
            shape=(self._curr_v_axis_idx, self._curr_h_axis_idx)
        )
        return arr, header

def align_columns(arr, columns, orig_columns, fill_value):
    assert isinstance(arr, (np.ndarray, csr_matrix))
    assert len(orig_columns) == len(set(orig_columns)), "columns must be unique"
    if isinstance(arr, csr_matrix):
        assert fill_value == 0, "'fill_value' must be 0 if 'arr' is sparse"

    if min(arr.shape) == 0:
        if not arr.shape[0] == 0:
            raise RuntimeError()
        result = np.zeros((0, len(orig_columns)), dtype=arr.dtype)
        if isinstance(arr, np.ndarray):
            return result
        elif isinstance(arr, csr_matrix):
            return csr_matrix(result)
        else:
            raise TypeError(arr)

    # https://stackoverflow.com/questions/8251541/
    # numpy-for-every-element-in-one-array-find-the-index-in-another-array
    y = orig_columns
    x = columns
    index = np.argsort(x)
    sorted_x = x[index]
    sorted_index = np.searchsorted(sorted_x, y)
    yindex = np.take(index, sorted_index, mode="clip")
    mask = x[yindex] == y

    in_idx = yindex[mask]
    out_idx = mask.nonzero()[0]
    del x, y, index, sorted_x, sorted_index, yindex, mask

    if isinstance(arr, csr_matrix):
        _arr = arr.tocoo()
        data, row, col = _arr.data, _arr.row, _arr.col
        del _arr
        _d = dict(zip(in_idx, out_idx))
        col = np.vectorize(lambda idx: _d[idx] if idx in _d else -1)(col)
        _mask = col != -1
        data, row, col = data[_mask], row[_mask], col[_mask]
        result = coo_matrix((data, (row, col)), shape=(arr.shape[0], len(orig_columns))).tocsr().astype(arr.dtype)
    else:
        result = np.empty((arr.shape[0], len(orig_columns)), dtype=arr.dtype)
        result.fill(fill_value)
        result[:,out_idx] = arr[:,in_idx]
    return result

def merge_columns(arr1, columns1, arr2, columns2, fill_value):
    merged_columns = np.array(sorted(set(columns1)|set(columns2)))
    return (
        align_columns(arr1, columns1, merged_columns, fill_value),
        align_columns(arr2, columns2, merged_columns, fill_value),
        merged_columns,
    )

def insert(arr, idx, values, axis=None):
    if isinstance(arr, np.ndarray) and isinstance(values, np.ndarray):
        return np.insert(arr, idx, values, axis)
    elif isinstance(arr, _sparse_types) and isinstance(values, _sparse_types):
        if not axis == 0:
            raise NotImplementedError("'axis' must be 0")
        assert arr.shape[1] == values.shape[1]
        assert len(idx.shape) == 1 and idx.shape[0] == values.shape[0]
        assert np.all(np.diff(idx) >= 0)
        _type = type(arr)
        arr, values = arr.tocoo(), values.tocoo()
        # .data .row .col
        new_data = np.concatenate((arr.data, values.data))
        new_col = np.concatenate((arr.col, values.col))
        _idx = idx + np.arange(len(idx))
        new_row_values = _idx[values.row]
        _deltas = np.zeros(len(arr.row), dtype=np.int)
        for i in idx:
            _deltas[arr.row >= i] += 1
        new_row_arr = arr.row + _deltas
        new_row = np.concatenate((new_row_arr, new_row_values))
        result = coo_matrix((new_data, (new_row, new_col)), shape=(arr.shape[0] + values.shape[0], arr.shape[1]))
        return _type(result)
    else:
        raise TypeError()

def array_equal(arr1, arr2):
    if all([isinstance(a, np.ndarray) for a in [arr1, arr2]]):
        return np.array_equal(arr1, arr2)
    elif all([isinstance(a, _sparse_types) for a in [arr1, arr2]]):
        return arr1.shape == arr2.shape and (arr1 != arr2).nnz == 0
    else:
        raise TypeError()

class IndexedNDArray:

    def __init__(self, *, values, indices, index_names):
        _unique = lambda l: len(l) == len(set(l))
        assert isinstance(values, np.ndarray)
        assert isinstance(index_names, np.ndarray)
        assert isinstance(indices, (list, tuple))
        assert len(indices) == len(index_names)
        assert _unique(index_names)
        assert values.shape == tuple((len(idx) for idx in indices))
        for idx in indices:
            assert isinstance(idx, np.ndarray)
            assert _unique(idx)
        self.values = values.copy()
        self.indices = tuple([idx.copy() for idx in indices])
        self.index_names = index_names.copy()

    def __str__(self):
        return "index names: " + ', '.join(self.index_names) + "\nshape: " + str(self.values.shape)

    def _numeric_idx(self, idx, arr):
        arr = list(arr)
        assert isinstance(idx, (int, str, list, tuple, np.ndarray))
        if isinstance(idx, int):
            return idx
        elif isinstance(idx, str):
            return arr.index(idx)
        elif isinstance(idx, (list, tuple, np.ndarray)):
            if isinstance(idx, np.ndarray):
                assert len(idx.shape) == 1
            new_idx = np.zeros(len(idx), dtype=np.int)
            for i, _idx in enumerate(idx):
                if isinstance(_idx, int):
                    new_idx[i] = _idx
                elif isinstance(_idx, str):
                    new_idx[i] = arr.index(_idx)
                else:
                    raise TypeError()
            return new_idx
        else:
            raise TypeError()

    @staticmethod
    def _reduce_index(n_axis, indices, index_names):
        mask = np.zeros(len(index_names), dtype=np.bool)
        mask[n_axis] = True
        mask = np.where(np.logical_not(mask))[0]
        return (
            tuple(indices[i] for i in mask),
            index_names[mask],
        )

    def get_index(self, idx):
        return self.indices[self._numeric_idx(idx, self.index_names)]

    def mean(self, axis):
        n_axis = self._numeric_idx(axis, self.index_names)
        new_values = self.values.mean(axis=(n_axis if isinstance(n_axis, int) else tuple(n_axis))) # checks if axis is unique
        new_indices, new_index_names = self._reduce_index(n_axis, self.indices, self.index_names)
        return IndexedNDArray(
            values=new_values,
            indices=new_indices,
            index_names=new_index_names,
        )

    def ranking(self, axis, reverse=False):
        axis = list(self.index_names).index(axis) if isinstance(axis, str) else axis
        if reverse:
            new_values = self.values.argsort(axis=axis).argsort(axis=axis) + 1.
        else:
            new_values = (-self.values).argsort(axis=axis).argsort(axis=axis) + 1.
        return IndexedNDArray(
            values=new_values,
            indices=self.indices,
            index_names=self.index_names,
        )

    def slice(self, d):
        assert isinstance(d, dict), "slice with dict {index_name: slice_obj, ...}"
        k, v = list(zip(*d.items()))
        for i, _k in enumerate(k):
            if isinstance(_k, int):
                k[i] = str(self.index_names[_k])
        assert all([isinstance(_k, str) for _k in k])
        assert len(k) == len(set(k)), "index names must be unique"

        new_indices = self.indices
        new_index_names = self.index_names
        new_values = self.values

        for _k, _v in zip(k, v):
            n_k = self._numeric_idx(_k, new_index_names)
            n_v = self._numeric_idx(_v, new_indices[n_k])
            if isinstance(n_v, int): # remove n_k-th dimension
                new_indices, new_index_names = self._reduce_index(n_k, new_indices, new_index_names)
            new_values = np.take(new_values, n_v, axis=n_k)

        return IndexedNDArray(
            values=new_values,
            indices=new_indices,
            index_names=new_index_names,
        )

import hashlib
def hashed_columns(arr, header, n_columns, random_sign):
    assert isinstance(n_columns, int) and n_columns > 0
    assert isinstance(random_sign, bool)
    int_hash = lambda x: int(hashlib.md5(str(x).encode("utf-8")).hexdigest()[:8], 16)
    hashed_column_idx = np.array([int_hash(x)%n_columns for x in header], dtype=np.int)
    sign = np.ones(len(header), dtype=np.float)
    if random_sign:
        sign[[(int_hash(x)%(2*n_columns))<n_columns for x in header]] = -1
    sign = sign.astype(arr.dtype)
    result = np.zeros((arr.shape[0], n_columns), dtype=arr.dtype)
    if isinstance(arr, _sparse_types):
        arr = arr.tocoo().multiply(coo_matrix(sign.reshape(1,-1))).tocoo()
        row, col, data = arr.row, arr.col, arr.data
        col = hashed_column_idx[col]
        np.add.at(result,(row,col),data)
    else:
        np.add.at(
            result.T,
            hashed_column_idx,
            (arr*sign.reshape(1,-1)).T,
        )
    return result
