import numpy as np
from copy import deepcopy

from ._Backend import Backend

class DataContainer:
    def __getitem__(self, key):
        raise NotImplementedError()
    @property
    def obj(self):
        if isinstance(self._obj, Backend):
            return self._obj.load()
        else:
            return self._obj
    @property
    def container(self):
        return self.__class__
    @property
    def proxy(self):
        return ProxyContainer((lambda: self.obj), self.container)

class ProxyContainer(DataContainer):
    def __init__(self, fn, container):
        self.fn = fn
        self._container = container
    @property
    def container(self):
        return self._container
    @property
    def obj(self):
        return self.fn()
    @property
    def proxy(self):
        return self
    def __getitem__(self, key):
        return self.container(self.fn())[key]
    def __str__(self):
        return "Proxy: " + str(self.container(self.fn()))

class Variable(DataContainer):
    def __init__(self, obj):
        self._obj = obj
    def __getitem__(self, key):
        return Variable(deepcopy(self.obj))

class Header(DataContainer):
    def __init__(self, obj):
        self._obj = obj
    def __getitem__(self, key):
        return Header(self.obj.copy())
    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return "{}: shape={}, type={}".format(self.__class__.__name__, self.obj.shape, self.obj.dtype)

class NDArray(DataContainer):
    n_dim_idx = None
    def __init__(self, obj):
        self._obj = obj
        self._check()
    def _check(self):
        n_dim_idx = self.__class__.n_dim_idx
        array = self.obj
        assert isinstance(n_dim_idx, int)
        assert n_dim_idx <= len(array.shape)
        for i in range(n_dim_idx):
            assert array.shape[i] == array.shape[0]
    def __getitem__(self, idx):
        new_array = self.obj
        for i in range(self.n_dim_idx):
            # works with slices
            new_array = np.rollaxis(np.rollaxis(new_array, i, 0)[idx], 0, i + 1)
        return self.__class__(new_array.copy())
    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return "{}: shape={}, type={}".format(self.__class__.__name__, self.obj.shape, self.obj.dtype)

class Table(NDArray):
    n_dim_idx = 1

def _table_nd(n_dim):
    return type(
        "Table{}D".format(n_dim),
        (NDArray,),
        {"n_dim_idx": n_dim})

Table2D = _table_nd(2)
Table3D = _table_nd(3)
# etc.
