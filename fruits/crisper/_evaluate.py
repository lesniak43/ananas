import os
import time

import mandalka

from . import CACHE_PATH, MODE, LOGGER, safe_path_join
from .storage import MandalkaStorage

class LockedError(Exception):
    pass
def lock2(node):
    try:
        os.makedirs(safe_path_join(
            mandalka.unique_id(node) + ".lock2",
            dirname=CACHE_PATH))
    except FileExistsError:
        raise LockedError()
def unlock2(node):
    try:
        os.rmdir(safe_path_join(
            mandalka.unique_id(node) + ".lock2",
            dirname=CACHE_PATH))
    except FileNotFoundError as e:
        pass
def is_locked2(node):
    return os.path.exists(safe_path_join(
        mandalka.unique_id(node) + ".lock2",
        dirname=CACHE_PATH,
    ))

class _EvaluationGraph:
    def __init__(self, *nodes, label):
        def _inputs(node):
            try:
                return mandalka.inputs(node)
            except mandalka.MandalkaArgumentsError:
                return []
        self.label = label
        self.by_uid = {}
        visited = set()
        edges = []
        q = list(nodes)
        while len(q) > 0:
            n = q.pop()
            uid = mandalka.unique_id(n)
            if not uid in visited:
                visited.add(uid)
                if isinstance(n, MandalkaStorage):
                    if not n.mandalka_exists():
                        self.by_uid[uid] = n
                        inp = _inputs(n)
                        q += inp
                        edges += [(mandalka.unique_id(_n), uid) for _n in inp]
                else:
                    inp = _inputs(n)
                    q += inp
                    edges += [(mandalka.unique_id(_n), uid) for _n in inp]
        self.children = {uid: set() for uid in visited}
        self.parents = {uid: set() for uid in visited}
        while len(edges) > 0:
            e = edges.pop()
            self.parents[e[0]].add(e[1])
            self.children[e[1]].add(e[0])
        self.leaves = set([k for k, v in self.children.items() if len(v) == 0])
        self.total = len(self.by_uid)
    def _remove(self, uid):
        assert len(self.children[uid]) == 0
        assert uid in self.leaves
        self.leaves.remove(uid)
        del self.children[uid]
        if uid in self.by_uid:
            del self.by_uid[uid]
        for p in self.parents[uid]:
            self.children[p].remove(uid)
            if len(self.children[p]) == 0:
                self.leaves.add(p)
        del self.parents[uid]
    def _try_evaluate(self, uid):
        assert uid in self.leaves
        if not uid in self.by_uid:
            self._remove(uid)
            return True
        else:
            n = self.by_uid[uid]
            if n.mandalka_exists():
                self._remove(uid)
                return True
            else:
                try:
                    lock2(n)
                    if n.mandalka_exists():
                        unlock2(n)
                        self._remove(uid)
                        return True
                    else:
                        mandalka.evaluate(n)
                        unlock2(n)
                        self._remove(uid)
                        return True
                except LockedError:
                    return False
    def _scan_leaves(self):
        for uid in self.leaves:
            if self._try_evaluate(uid):
                return True
        return False
    def run(self):
        while True:
            while self._scan_leaves():
                pass
            if len(self.leaves) == 0:
                break
            LOGGER.info("All nodes locked, waiting 10 seconds...")
            time.sleep(10)

class _MonitorGraph:
    def __init__(self, *nodes, label):
        def _inputs(node):
            try:
                return mandalka.inputs(node)
            except mandalka.MandalkaArgumentsError:
                return []
        self.label = label
        self.by_uid = {}
        visited = set()
        edges = []
        q = list(nodes)
        while len(q) > 0:
            n = q.pop()
            uid = mandalka.unique_id(n)
            if not uid in visited:
                visited.add(uid)
                if isinstance(n, MandalkaStorage):
                    if not n.mandalka_exists():
                        self.by_uid[uid] = n
                        inp = _inputs(n)
                        q += inp
                        edges += [(mandalka.unique_id(_n), uid) for _n in inp]
                else:
                    inp = _inputs(n)
                    q += inp
                    edges += [(mandalka.unique_id(_n), uid) for _n in inp]
        self.children = {uid: set() for uid in visited}
        self.parents = {uid: set() for uid in visited}
        while len(edges) > 0:
            e = edges.pop()
            self.parents[e[0]].add(e[1])
            self.children[e[1]].add(e[0])
        self.leaves = set([k for k, v in self.children.items() if len(v) == 0])
        self.total = len(self.by_uid)
        self.locked = {}
    def _remove(self, uid):
        assert len(self.children[uid]) == 0
        assert uid in self.leaves
        self.leaves.remove(uid)
        del self.children[uid]
        if uid in self.by_uid:
            del self.by_uid[uid]
        for p in self.parents[uid]:
            self.children[p].remove(uid)
            if len(self.children[p]) == 0:
                self.leaves.add(p)
        del self.parents[uid]
        if uid in self.locked:
            del self.locked[uid]
    def _try_evaluate(self, uid):
        assert uid in self.leaves
        if not uid in self.by_uid:
            self._remove(uid)
        else:
            n = self.by_uid[uid]
            if n.mandalka_exists():
                self._remove(uid)
            else:
                if is_locked2(n) and uid not in self.locked:
                    self.locked[uid] = time.time()
    def run(self):
        while True:
            for uid in tuple(self.leaves):
                self._try_evaluate(uid)
            t = time.time()
            print(chr(27) + "[2J" + chr(27) + "[H")
            print("Monitoring: '{}'".format(self.label))
            print("Currently being evaluated:")
            for tup in sorted([(int(t-v), k[:16]+"...") for k, v in self.locked.items()]):
                print("{}s : {}".format(*tup))
            print("Evaluated {}/{} storage nodes so far...".format(
                self.total - len(self.by_uid),
                self.total,
            ))
            if len(self.leaves) == 0:
                break
            print("Waiting 10 seconds")
            for _ in range(10):
                print('.', end='', flush=True)
                time.sleep(1)

def evaluate(*nodes, label=None):
    if MODE == "MONITOR":
        _MonitorGraph(*nodes, label=label).run()
    else:
        _EvaluationGraph(*nodes, label=label).run()
