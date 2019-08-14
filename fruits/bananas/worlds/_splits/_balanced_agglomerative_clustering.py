from itertools import combinations

import numpy as np
from numba import njit
from scipy.cluster.hierarchy import linkage as calculate_linkage

import mandalka

from .. import DataDict, Table, Variable
from . import GroupSplitter

@mandalka.node
class AgglomerativeClustering:
    def __init__(self, *, source, kernel):
        # for 'linkage' definition, see:
        #   https://docs.scipy.org/doc/scipy/reference/generated/
        #       scipy.cluster.hierarchy.linkage.html
        self.data = DataDict()
        kernel = source.data[kernel]
        pdist = np.zeros(kernel.shape[0] * (kernel.shape[0] - 1) // 2, dtype=kernel.dtype)
        @njit
        def to_pdist(kernel, pdist):
            i = 0
            for j in range(kernel.shape[0]):
                for k in range(j+1, kernel.shape[0]):
                    pdist[i] = np.sqrt(kernel[j,j]**2 + kernel[k,k]**2 - 2 * kernel[j,k])
                    i += 1
        to_pdist(kernel, pdist)
        linkage = calculate_linkage(
            pdist,
            method="average",
            metric=None,
            optimal_ordering=False,
        )
        self.data[("linkage", "children")] = Variable(
            linkage[:,[0,1]].astype(np.int))
        self.data[("linkage", "distances")] = Variable(
            linkage[:,2].astype(np.int))
        self.data[("linkage", "counts")] = Variable(
            linkage[:,3].astype(np.int))
        self.data.lock()
        mandalka.del_arguments(self)

@mandalka.node
class BalancedAgglomerativeClustering(GroupSplitter):

    def build(self, *, source, kernel, n_groups):
        self.data = source.data.slice[:]
        del self.data["groups"]
        del self.data["n_groups"]
        del self.data[kernel]

        self.log("calculating clustering...")
        # 'average' linkage seems to be more stable than 'complete'
        ac = AgglomerativeClustering(source=source, kernel=kernel)
        clusters = _ac_find_balanced_clusters(
            ac.data[("linkage", "children")],
            ac.data[("linkage", "counts")],
            n_groups,
        )
        self.data["groups"] = Table(clusters)
        self.data["n_groups"] = Variable(n_groups)


def _ac_find_balanced_clusters(linkage_children, linkage_counts, n_groups):

    children = linkage_children
    n_leaves = int(children.shape[0] + 1)
    n_nodes = int(n_leaves + children.shape[0])

    result = np.zeros(n_leaves, dtype=np.int)
    result.fill(n_groups-1)
    parents = np.zeros(n_nodes, dtype=np.int)
    n_leaves_below = np.ones(n_nodes, dtype=np.int)
    n_leaves_below[n_leaves:] = linkage_counts
    stack = np.zeros(n_nodes, dtype=np.int)

    @njit
    def _find_balanced_clusters(children, n_groups, result, parents, n_leaves_below, stack):

        n_leaves = children.shape[0] + 1
        n_nodes = n_leaves + children.shape[0]
        root_idx = n_nodes - 1
        stack_ptr = -1

        for i in range(children.shape[0]):
            parents[children[i,0]] = parents[children[i,1]] = i + n_leaves

        for n_cluster in range(n_groups - 1): # everything left is the last cluster
            # search for the most "balanced" cluster
            idx = np.argmin(np.abs(n_leaves_below - n_leaves // n_groups))
            _s = n_leaves_below[idx]
            # remove leaves (reduce n_leaves_below)
            # go towards root
            p_idx = idx
            while p_idx != root_idx:
                p_idx = parents[p_idx]
                n_leaves_below[p_idx] -= _s
            # go towards leaves
            stack_ptr = 0
            stack[stack_ptr] = idx
            while stack_ptr >= 0:
                _idx = stack[stack_ptr]
                stack_ptr -= 1
                if n_leaves_below[_idx] > 0: # not visited in previous iterations of the for loop
                    n_leaves_below[_idx] = 0
                    if _idx < n_leaves: # leaf
                        result[_idx] = n_cluster # mark the leaf
                    else:
                        stack_ptr += 1
                        stack[stack_ptr] = children[_idx-n_leaves,0]
                        stack_ptr += 1
                        stack[stack_ptr] = children[_idx-n_leaves,1]

    _find_balanced_clusters(children, n_groups, result, parents, n_leaves_below, stack)
    return result
