import os
import re

import mandalka

from .. import safe_path_join
from ..storage import MandalkaStorage

from . import _Backend, _DataContainer
from ._Backend import backend_by_object
from ._DataContainer import DataContainer

def _backend_by_name(name):
    return getattr(_Backend, name)

def _container_by_name(name):
    return getattr(_DataContainer, name)

def _names_to_paths(names):
    paths = []
    for name in names:
        for n in name:
            assert len(n) > 0
            assert n[0] != '.'
            assert '/' not in n
            assert '\n' not in n
        paths.append('/'.join(name))
    return paths

def _paths_to_names(paths):
    return [tuple(path.split('/')) for path in paths]

def _save(dirname, names, objects, description):
    paths = _names_to_paths(names)
    backend_dirnames = [safe_path_join(path, dirname=dirname) for path in paths]
    backends = [
        backend_by_object(obj)(_dirname) for (obj, _dirname) in zip(objects, backend_dirnames)]
    backend_names = [b.__class__.__name__ for b in backends]
    for _dirname in backend_dirnames:
        if not os.path.exists(_dirname):
            os.makedirs(_dirname)
    [backend.save(obj) for backend, obj in zip(backends, objects)]
    meta = re.sub('\n', '', description) + '\n'
    for bname, path in zip(backend_names, paths):
        meta += bname + '\n' + path + '\n'
    with open(safe_path_join("meta.txt", dirname=dirname), 'x') as f_out:
        f_out.write(meta)

def _load_backends(dirname):
    with open(safe_path_join("meta.txt", dirname=dirname), 'r') as f_in:
        content = f_in.read().split('\n')
        if len(content) % 2 == 1:
            raise RuntimeError("{} corrupted, aborting...".format(dirname))
        content = content[1:] # drop description
        if len(content) == 1:
            backend_names, paths = tuple(), tuple()
        else:
            # empty string after last \n is dropped
            backend_names, paths = zip(*zip(*[iter(content)]*2))
    backend_dirnames = [safe_path_join(path, dirname=dirname) for path in paths]
    backends = [
        _backend_by_name(bname)(dirname) \
            for (bname, dirname) in zip(backend_names, backend_dirnames)]
    names = _paths_to_names(paths)
    return names, backends


class DataDict:

    def __init__(self, source_dirname=None):
        self.__dict__["_storage"] = {}
        if source_dirname is not None:
            for name, backend in zip(*_load_backends(source_dirname)):
                _n = name[-1]
                assert '_' in _n
                container_name, _n = _n.split('_', 1)
                name = list(name)
                name[-1] = _n
                self._storage[tuple(name)] = _container_by_name(container_name)(backend)

    #########################
    def __setattr__(self, name, value):
        raise AttributeError("Don't.")

    #########################
    def __setitem__(self, name, value):
        assert not "_readonly" in self.__dict__, "Read-only mode."
        if isinstance(name, str):
            name = (name,)
        assert isinstance(name, tuple), "Key must be tuple of str."
        assert len(name) > 0, "Key cannot be an empty tuple."
        assert all([isinstance(n, str) for n in name]), "Key must be tuple of str."
        assert all([len(n) > 0 for n in name]), "Every str must be non-empty. Sorry."
        for key in self._storage.keys():
            if len(key) > len(name):
                assert name != key[:len(name)], "Key '{}' is a prefix of already existing key '{}'.".format(name, key)
        assert isinstance(value, DataContainer), "Don't forget to put data into a container."
        self._storage[name] = value
    def __getitem__(self, name):
        if not hasattr(self, "_storage"):
            raise RuntimeError("DataDict was closed.")
        if isinstance(name, str):
            name = (name,)
        return self._storage[name].obj
    def __delitem__(self, name):
        assert not "_readonly" in self.__dict__, "Read-only mode."
        if isinstance(name, str):
            name = (name,)
        if name in self._storage:
            del self._storage[name]
    def __iter__(self):
        return self.__dict__["_storage"].__iter__()
    def __contains__(self, item):
        if isinstance(item, str):
            item = (item,)
        return self.__dict__["_storage"].__contains__(item)

    #########################
    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        result = "DataDict object containing:\n"
        for key, value in self._storage.items():
            result += "  {}:\n    {}\n".format(key, str(value))
        return result[:-1]

    #########################
    def get_container(self, name):
        if isinstance(name, str):
            name = (name,)
        return self._storage[name].container

    #########################
    @property
    def slice(self):
        class DataDictSlicer:
            def __getitem__(_self, key):
                return self._slice(key)
            def __setattr__(_self, key, value):
                raise AttributeError("Don't.")
        return DataDictSlicer()
    def _slice(self, key):
        result = DataDict()
        for name, value in self._storage.items():
            result._storage[name] = value[key]
        return result

    #########################
    @property
    def proxy(self):
        dd = DataDict()
        for key in self:
            dd[key] = self._storage[key].proxy
        return dd

    #########################
    def _save(self, dirname, description):
        names, objects = [], []
        for name, value in self._storage.items():
            cls_name = value.__class__.__name__
            assert '_' not in cls_name
            _n = list(name)
            _n[-1] = '_'.join([cls_name, _n[-1]])
            names.append(tuple(_n))
            objects.append(value.obj)
        _save(dirname, names, objects, description)

    def _close(self):
        del self.__dict__["_storage"]
        self.lock()

    def lock(self):
        self.__dict__["_readonly"] = True

class MandalkaDictStorage(MandalkaStorage):
    def mandalka_build(self, *args, **kwargs):
        self.data = DataDict(source_dirname=None) # this may be replaced inside 'build'
        self.build(*args, **kwargs)
    def mandalka_save_cache(self):
        description = mandalka.describe(self, depth=1) + "___" + "STACK: " + str(mandalka.get_evaluation_stack())
        self.data._save(self.mandalka_cache_path(), description)
        self.data._close()
    def mandalka_clean_after_build(self):
        del self.data
    def mandalka_load(self, *args, **kwargs):
        self.data = DataDict(source_dirname=self.mandalka_results_path())
        self.build_proxy(*args, **kwargs)
        self.data.lock()
    def build(self, *args, **kwargs):
        raise NotImplementedError()
    def build_proxy(self, *args, **kwargs):
        # optional
        pass

class MandalkaDictProxy:
    def __init__(self, *args, **kwargs):
        mandalka.del_arguments(self)
        self.data = DataDict(source_dirname=None) # this may be replaced inside 'build_proxy'
        self.build_proxy(*args, **kwargs)
        self.data.lock()
    def build_proxy(self, *args, **kwargs):
        raise NotImplementedError()
