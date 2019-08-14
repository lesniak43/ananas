import numpy as np

import mandalka

from .. import Table, Variable
from . import GroupSplitter

@mandalka.node
class CrossValidation(GroupSplitter):
    """
    Add random splits indices.
    source.data["uid"] is considered to be the index.
    """
    def build(self, *, source, n_groups, seed):
        uid = source.data["uid"]
        """
        From numpy.unique doc:
        unique_indices (here unique_uids) : ndarray, optional
            The indices of the first occurrences of the unique values in the original array. Only provided if return_index is True.
        unique_inverse : ndarray, optional
            The indices to reconstruct the original array from the unique array. Only provided if return_inverse is True.
        """
        _, unique_uids, unique_inverse = np.unique(uid, return_index=True, return_inverse=True)
        """
        based on:
        https://github.com/scikit-learn/scikit-learn/
            blob/master/sklearn/model_selection/_split.py
        """
        n_samples = len(unique_uids)
        groups_per_uid = np.zeros(n_samples, dtype=np.int)
        rng = np.random.RandomState(seed=seed)
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        fold_sizes = (n_samples // n_groups) * np.ones(n_groups, dtype=np.int)
        fold_sizes[:n_samples % n_groups] += 1
        current = 0
        for group, fold_size in enumerate(fold_sizes):
            start, stop = current, current + fold_size
            groups_per_uid[indices[start:stop]] = group
            current = stop
        assert set(groups_per_uid) == set(range(n_groups))
        groups = groups_per_uid[unique_inverse]

        self.data = source.data.slice[:]
        del self.data["groups"]
        del self.data["n_groups"]
        self.data["groups"] = Table(groups)
        self.data["n_groups"] = Variable(n_groups)
