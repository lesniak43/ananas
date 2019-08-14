import hashlib
import os
import time

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

import herbivores
from herbivores import merge_columns
import mandalka

from crisper import (
    FINGERPRINTS_PATH,
    safe_path_join,
    moderately_safe_rmtree,
)
from crisper.datadict import DataDict, Variable

def sorted_a_not_in_b(a, b):
    # find a not in b
    mask = np.zeros(len(a), dtype=np.bool)
    idx = np.searchsorted(b, a, side="left", sorter=None)
    mask[idx == len(b)] = True
    _m = np.logical_not(mask)
    _m2 = a[_m] != b[idx[_m]]
    mask[np.arange(len(mask))[_m][_m2]] = True
    return a[mask]

class Response: pass
class ToDo(Response):
    def __init__(self, x, hx, dirname, lock_dirname):
        self.x = x
        self.hx = hx
        self.dirname = dirname
        self.lock_dirname = lock_dirname
class Result(Response):
    def __init__(self, y, header):
        self.y = y
        self.header = header
class Wait(Response):
    def __init__(self, t):
        self.t = t

class Client:
    def __init__(self, f, server):
        self.f = f
        self.server = server
    def __call__(self, x):
        response = self.server.request(x)
        while not isinstance(response, Result):
            if isinstance(response, ToDo):
                y, header = self.f(response.x)
                dd = DataDict()
                dd["hx"] = Variable(response.hx)
                dd["y"] = Variable(y)
                dd["header"] = Variable(header)
                dd._save(response.dirname, str(response))
                del dd, y, header
                self.server.done(response)
            elif isinstance(response, Wait):
                time.sleep(response.t)
            else:
                raise TypeError(response)
            response = self.server.request(x)
        return response.y, response.header
    def check(self, x):
        response = self.server.request(x, max_todo=0)
        assert isinstance(response, Result)
        fresh_y, fresh_header = self.f(x)
        assert herbivores.array_equal(response.y, fresh_y)
        assert np.all(response.header == fresh_header)

class Server:

    def __init__(self, uid, max_request=100, client_wait_time=10, server_wait_time=1):

        self.data_dirname = safe_path_join(
            "data", uid, dirname=FINGERPRINTS_PATH)
        self.backup_dirname = safe_path_join(
            "data", uid + ".backup", dirname=FINGERPRINTS_PATH)
        self.cache_dirname = safe_path_join(
            "cache", uid, dirname=FINGERPRINTS_PATH)
        self.pending_dirname = safe_path_join(
            "pending", uid, dirname=FINGERPRINTS_PATH)
        self.lock_dirname = safe_path_join(
            "cache", uid + ".lock", dirname=FINGERPRINTS_PATH)
        self.uid = uid
        self.max_request = max_request
        self.client_wait_time = client_wait_time
        self.server_wait_time = server_wait_time

    def __setattr__(self, name, value):
        if name not in ("data_dirname", "backup_dirname", "cache_dirname", "pending_dirname", "lock_dirname", "uid", "max_request", "client_wait_time", "server_wait_time") or hasattr(self, name):
            raise RuntimeError("Read-only.")
        else:
            super().__setattr__(name, value)

    def lock(self):
        while True:
            try:
                os.makedirs(self.lock_dirname)
                break
            except FileExistsError:
                time.sleep(self.server_wait_time)

    def unlock(self):
        os.rmdir(self.lock_dirname)

    def hash_x(self, x):
        assert len(x.shape) == 1
        hash_utf8 = lambda s: hashlib.sha256(s.encode("UTF-8")).digest()[:96] # should be enough...
        result = np.empty(len(x), dtype=np.dtype("S96"))
        for i, _x in enumerate(x):
            result[i] = hash_utf8(_x)
        return result

    def request(self, x, max_todo=None):
        if max_todo is None:
            max_todo = self.max_request
        else:
            assert isinstance(max_todo, int)
            max_todo = min(max_todo, self.max_request)

        hx = self.hash_x(x)
        uhx, x_idx, inv_idx = np.unique(hx, return_index=True, return_inverse=True)
        # hx[x_idx] == uhx and uhx[inv_idx] == hx

        self.lock()
        if os.path.exists(self.data_dirname):
            dd = DataDict(source_dirname=self.data_dirname)
            todo_uhx = sorted_a_not_in_b(uhx, dd["hx"])
        else:
            todo_uhx = uhx
        if todo_uhx.shape[0] == 0:
            idx = np.searchsorted(dd["hx"], uhx, side="left", sorter=None)
            y = dd["y"][idx][inv_idx].copy()
            header = dd["header"].copy()
            y = np.array(y) if isinstance(y, np.memmap) else y
            header = np.array(header) if isinstance(header, np.memmap) else header
            response = Result(y=y, header=header)
        else:
            really_todo_uhx = sorted_a_not_in_b(
                todo_uhx,
                self.get_all_pending_hx(),
            )
            if really_todo_uhx.shape[0] == 0:
                response = Wait(t=self.client_wait_time)
            else:
                really_really_todo_uhx = really_todo_uhx[:max_todo]
                dirname, lock_dirname = self.add_pending_hx(
                    really_really_todo_uhx)
                response = ToDo(
                    x=x[x_idx][np.searchsorted(uhx, really_really_todo_uhx, side="left", sorter=None)],
                    hx=really_really_todo_uhx,
                    dirname=dirname,
                    lock_dirname=lock_dirname,
                )

        self.unlock()
        return response

    def get_all_pending_hx(self):
        if not os.path.exists(self.pending_dirname) or len(os.listdir(self.pending_dirname)) == 0:
            return self.hash_x(np.array([], dtype=np.str))
        else:
            result = []
            for dirname in os.listdir(self.pending_dirname):
                result.append(DataDict(source_dirname=safe_path_join(dirname, dirname=self.pending_dirname))["hx"])
            return np.unique(np.concatenate(result))

    def add_pending_hx(self, hx):
        if not os.path.exists(self.pending_dirname):
            os.makedirs(self.pending_dirname)
        dirname = safe_path_join(str(os.getpid()), dirname=self.cache_dirname)
        lock_dirname = safe_path_join(str(os.getpid()), dirname=self.pending_dirname)
        assert not os.path.exists(dirname)
        assert not os.path.exists(lock_dirname)
        dd = DataDict()
        dd["hx"] = Variable(hx)
        dd._save(lock_dirname, self.uid)
        return dirname, lock_dirname

    def done(self, response):
        self.lock()
        assert not os.path.exists(self.backup_dirname)
        if not os.path.exists(self.data_dirname):
            # first batch, just save response
            dd = DataDict(source_dirname=response.dirname)
            dd._save(self.data_dirname, self.uid)
        else:
            # backup
            os.rename(self.data_dirname, self.backup_dirname)
            # merge
            dd1 = DataDict(source_dirname=self.backup_dirname)
            dd2 = DataDict(source_dirname=response.dirname)
            dd = self.merge_sorted_hx_data(dd1, dd2)
            # save
            dd._save(self.data_dirname, self.uid)
            # remove backup
            moderately_safe_rmtree(self.backup_dirname)
        # unset response.hx
        moderately_safe_rmtree(response.lock_dirname)
        if len(os.listdir(self.pending_dirname)) == 0:
            os.rmdir(self.pending_dirname)
        # remove cache
        moderately_safe_rmtree(response.dirname)
        self.unlock()

    def merge_sorted_hx_data(self, dd1, dd2):

        # insert dd1 into dd2, both already have sorted unique hx

        insert_idx = np.searchsorted(dd2["hx"], dd1["hx"], side="left", sorter=None)

        new_hx1_mask = np.zeros(dd1["hx"].shape[0], dtype=np.bool)
        new_hx1_mask[insert_idx == dd2["hx"].shape[0]] = True
        _mask = insert_idx < dd2["hx"].shape[0]
        _m2 = dd2["hx"][insert_idx[_mask]] != dd1["hx"][_mask]
        new_hx1_mask[np.arange(len(new_hx1_mask))[_mask][_m2]] = True
        del _mask, _m2

        if new_hx1_mask.shape[0] == 0:
            return dd2
        else:
            final_hx = herbivores.insert(
                dd2["hx"],
                insert_idx[new_hx1_mask],
                dd1["hx"][new_hx1_mask],
                axis=0,
            )
            y1, y2, final_header = merge_columns(
                dd1["y"], dd1["header"], dd2["y"], dd2["header"], 0.)
            final_y = herbivores.insert(
                y2, insert_idx[new_hx1_mask], y1, axis=0)
            dd = DataDict()
            dd["hx"] = Variable(final_hx)
            dd["y"] = Variable(final_y)
            dd["header"] = Variable(final_header)
            return dd

### node method decorators ###

def cached_fingerprinter(f=None, max_request=1000, client_wait_time=5, server_wait_time=1):
    if f is not None:
        def _f(node, smiles):
            uid = mandalka.unique_id(node)
            server = Server(
                uid,
                max_request=max_request,
                client_wait_time=client_wait_time,
                server_wait_time=server_wait_time,
            )
            client = Client((lambda smiles: f(node, smiles)), server)
            return client(smiles)
        return _f
    else:
        return lambda g: cached_fingerprinter(
            f=g,
            max_request=max_request,
            client_wait_time=client_wait_time,
            server_wait_time=server_wait_time,
        )
