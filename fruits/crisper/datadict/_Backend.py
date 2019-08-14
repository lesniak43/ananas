import os
import json

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None

class Backend:
    def __init__(self, dirname):
        self.dirname = dirname
    def save(self, obj):
        # store obj
        raise NotImplementedError()
    def load(self):
        # return saved obj
        raise NotImplementedError()

class TFGraph(Backend):
    """
    Tensorflow knows better how to name your files:
    - path.data-00000-of-00001 - weights
    - path.index - ??
    - path.meta - graph
    """
    def save(self, session_graph):
        path = os.path.join(self.dirname, "graph")
        assert not os.path.exists(path + ".data-00000-of-00001")
        assert not os.path.exists(path + ".index")
        assert not os.path.exists(path + ".meta")
        session, graph = session_graph
        with graph.as_default():
            saver = tf.train.Saver(
                var_list=tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES),
                max_to_keep=1,
                keep_checkpoint_every_n_hours=437437437, # wait for it
                sharded=False) # doesn't help, eh...
            saver.save(
                session,
                save_path=path,
                meta_graph_suffix="meta",
                write_meta_graph=True,
                write_state=False)
    def load(self):
        path = os.path.join(self.dirname, "graph")
        assert os.path.exists(path + ".data-00000-of-00001")
        assert os.path.exists(path + ".index")
        assert os.path.exists(path + ".meta")
        graph = tf.Graph()
        with graph.as_default():
            session = tf.Session(graph=graph)
            saver = tf.train.import_meta_graph(path + ".meta")
            saver.restore(session, path)
        return (session, graph)

class NumpyNDArray(Backend):
    def save(self, arr):
        np.save(os.path.join(self.dirname, "data.npy"), arr)
    def load(self):
        _path = os.path.join(self.dirname, "data.npy")
        return np.load(
            os.path.join(self.dirname, "data.npy"),
            mmap_mode='r',
            allow_pickle=False,
        )

class NumpyVarStr(Backend):
    @staticmethod
    def save_varstr_to_file(path, arr):
        assert len(arr.shape) == 1
        assert np.vectorize(lambda x: '\n' not in x, otypes=[np.bool])(arr).all()
        s = '\n'.join(list(arr)) + '\n'
        with open(path, 'w') as f_out:
            f_out.write(s)
    @staticmethod
    def load_varstr_from_file(path):
        with open(path, 'r') as f_in:
            content = f_in.read()
            assert content[-1] == '\n'
            if content == '\n':
                return np.array([], dtype=np.object)
            else:
                return np.array(content[:-1].split('\n'), dtype=np.object)
    def save(self, arr):
        NumpyVarStr.save_varstr_to_file(
            os.path.join(self.dirname, "data.txt"),
            arr
        )
    def load(self):
        return NumpyVarStr.load_varstr_from_file(
            os.path.join(self.dirname, "data.txt"))

class SciPyCSR(Backend):
    @staticmethod
    def save_csr_to_file(path, arr):
        assert path.endswith(".npz") # because scipy says so
        sparse.save_npz(path, arr, compressed=True)
    @staticmethod
    def load_csr_from_file(path):
        assert path.endswith(".npz") # because scipy says so
        return sparse.load_npz(path)
    def save(self, arr):
        SciPyCSR.save_csr_to_file(
            os.path.join(self.dirname, "data.npz"),
            arr
        )
    def load(self):
        return SciPyCSR.load_csr_from_file(
            os.path.join(self.dirname, "data.npz"))

class Basic(Backend):
    def save(self, obj):
        with open(os.path.join(self.dirname, "data.json"), 'x') as f_out:
            json.dump(obj, f_out)
    def load(self):
        with open(os.path.join(self.dirname, "data.json"), 'r') as f_in:
            return json.load(f_in)


def backend_by_object(obj):
    if isinstance(obj, (int, bool, float, str)) or obj is None:
        return Basic
    elif isinstance(obj, tuple):
        if tf is not None and isinstance(obj[1], tf.Graph):
            return TFGraph
        else:
            return None
    elif isinstance(obj, csr_matrix):
        return SciPyCSR
    elif isinstance(obj, np.ndarray):
        if obj.dtype == np.object:
            if np.vectorize(
                    lambda x: isinstance(x, str),
                    otypes=[np.bool])(obj).all():
                return NumpyVarStr
            else:
                return None
        else:
            return NumpyNDArray
    else:
        return None
