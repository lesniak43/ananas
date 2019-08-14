import numpy as np
import mandalka

from .. import StorageWorld

@mandalka.node
class Subset(StorageWorld):

    """
    Keep only rows such that 'data_name' has
    values from 'allowed_values'.
    """

    def build(self, source, data_name, allowed_values, remove_data):
        ok_mask = np.vectorize(
            lambda x: x in allowed_values, otypes=[np.bool]
        )(source.data[data_name])
        self.data = source.data.slice[ok_mask]
        if remove_data:
            del self.data[data_name]

@mandalka.node
class SubsetData(StorageWorld):

    """
    Keep only or remove data_names (list of keys).
    """

    def build(self, *, source, data_names, keep):
        assert isinstance(keep, bool)
        _data_names = []
        for dn in data_names:
            if isinstance(dn, str):
                _data_names.append((dn,))
            elif isinstance(dn, tuple):
                for x in dn:
                    assert isinstance(x, str), "every data_name must be str or tuple of str"
                _data_names.append(dn)
            else:
                raise ValueError("every data_name must be str or tuple of str")
        self.data = source.data.slice[:]
        for key in tuple(self.data):
            if (key in _data_names) != keep:
                del self.data[key]

@mandalka.node
class Slice(StorageWorld):
    def build(self, *, source, index):
        self.data = source.data.slice[index]
