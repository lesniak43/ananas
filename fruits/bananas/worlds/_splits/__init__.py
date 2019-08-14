import numpy as np

import mandalka
from mandalka import MandalkaArgumentsError

from .. import StorageWorld

class GroupSplitter(StorageWorld):
    def build(self, *args, **kwargs):
        raise NotImplementedError()
    @mandalka.lazy
    def get_splits(self):
        try:
            n_groups = mandalka.arguments(self)["n_groups"]
        except MandalkaArgumentsError:
            n_groups = self.data["n_groups"]
        return [GroupSplit(source=self, n_split=i) for i in range(n_groups)]

class GroupSplitter(StorageWorld):
    def build(self, *args, **kwargs):
        raise NotImplementedError()
    @mandalka.lazy
    def get_splits(self):
        try:
            n_groups = mandalka.arguments(self)["n_groups"]
        except MandalkaArgumentsError:
            n_groups = self.data["n_groups"]
        return [GroupSplit(source=self, n_split=i) for i in range(n_groups)]

class DynamicGroupSplitter(StorageWorld):
    def build(self, *args, **kwargs):
        raise NotImplementedError()
    def get_splits(self):
        n_groups = self.data["n_groups"]
        return [GroupSplit(source=self, n_split=i) for i in range(n_groups)]

@mandalka.node
class GroupSplit:
    def __init__(self, *, source, n_split):
        self.source = source
        self.n_split = n_split
    @mandalka.lazy
    def get_train(self):
        return GroupSplitInner(split=self, which="train")
    @mandalka.lazy
    def get_test(self):
        return GroupSplitInner(split=self, which="test")

@mandalka.node
class GroupSplitInner(StorageWorld):
    def build(self, *, split, which):
        assert which in ["train", "test"]
        idx = split.source.data["groups"] == split.n_split
        if which == "train":
            idx = np.logical_not(idx)
        self.data = split.source.data.slice[idx]
        del self.data["n_groups"]
        del self.data["groups"]
