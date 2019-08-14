from itertools import groupby

import numpy as np

import mandalka

from ..worlds import (
    SQLiteDatabase,
    WarszyckiBioactivityRaw,
    WarszyckiBioactivity,
    WarszyckiLogKi,
    Subset,
    ParentMolecules,
    ConnectedSmiles,
    CanonicalSmiles,
    UniqueSmiles,
    Table,
    Variable,
    StorageWorld,
    EarliestYear,
)

@mandalka.node
class MeanLogKiWithUidSmiles(StorageWorld):
    def build(self, *, source):
        u = source.data["uid"]
        s = source.data["smiles"]
        v = source.data["value"]
        assert not np.isnan(v).any(), "Input source contains nan values"
        result = []
        key = lambda x: x[0]
        for k, g in groupby(sorted(zip(u, s, v), key=key), key):
            gu, gs, gv = zip(*g)
            assert len(set(gs)) == 1
            result.append((
                gu[0],
                gs[0],
                np.mean(gv)))
        if len(result) > 0:
            arrs = zip(*result)
        else:
            arrs = [(), (), ()]
        for arr, name, dtype in zip(
                    arrs,
                    ["uid", "smiles", "value"],
                    [np.str, np.object, np.float32]
                ):
            self.data[name] = Table(np.array(arr, dtype=dtype))

@mandalka.node
class FitOriginalIC50ToKi(StorageWorld):
    def build(self, *, source):
        u = source.data["uid"]
        s = source.data["smiles"]
        v = source.data["value"]

        assert not np.isnan(v).any(), "Input source contains nan values"

        def _mean(_u, _s, _v):
            result = []
            key = lambda x: x[0]
            for k, g in groupby(sorted(zip(_u, _s, _v), key=key), key):
                gu, gs, gv = zip(*g)
                assert len(set(gs)) == 1
                result.append((
                    gu[0],
                    gs[0],
                    np.mean(gv)))
            if len(result) > 0:
                arrs = zip(*result)
            else:
                arrs = [(), (), ()]
            return arrs

        mask_Ki = source.data["original_type"] == "Ki"
        mask_IC50 = source.data["original_type"] == "IC50"
        ki_uid, _, ki_value = (np.array(arr) for arr in _mean(u[mask_Ki], s[mask_Ki], v[mask_Ki]))
        ki_idx = np.argsort(ki_uid)
        ki_uid, ki_value = ki_uid[ki_idx], ki_value[ki_idx]
        ic50_uid, _, ic50_value = (np.array(arr) for arr in _mean(u[mask_IC50], s[mask_IC50], v[mask_IC50]))
        ic50_idx = np.argsort(ic50_uid)
        ic50_uid, ic50_value = ic50_uid[ic50_idx], ic50_value[ic50_idx]
        uids = np.sort(list(set(ki_uid)&set(ic50_uid)))

        delta = np.mean(np.concatenate((
            [0.],
            ki_value[np.searchsorted(ki_uid, uids)] - ic50_value[np.searchsorted(ic50_uid, uids)],
        )))

        self.data = source.data.slice[:]
        del self.data["value"]
        v = v.copy()
        v[mask_IC50] += delta
        self.data["value"] = Table(v)
        self.data["IC50_correction"] = Variable(delta)
        self.data["how_many_uids_to_estimate_correction"] = Variable(len(uids))

@mandalka.node
class MeanLogKiWithUidSmilesYearDocUid(StorageWorld):
    def build(self, *, source):
        u = source.data["uid"]
        s = source.data["smiles"]
        v = source.data["value"]
        y = source.data["year"]
        du = source.data["doc_uid"]
        assert not np.isnan(v).any(), "Input source contains nan values"
        result = []
        key = lambda x: x[0]
        for k, g in groupby(sorted(zip(u, s, v, y, du), key=key), key):
            gu, gs, gv, gy, gdu = zip(*g)
            assert len(set(gs)) == 1
            assert len(set(gy)) == 1
            assert len(set(gdu)) == 1
            assert gy[0] > 0
            result.append((
                gu[0],
                gs[0],
                np.mean(gv),
                gy[0],
                gdu[0],
            ))
        if len(result) > 0:
            arrs = zip(*result)
        else:
            arrs = [(), (), (), (), ()]
        for arr, name, dtype in zip(
                    arrs,
                    ["uid", "smiles", "value", "year", "doc_uid"],
                    [np.str, np.object, np.float32, np.int, np.str]
                ):
            self.data[name] = Table(np.array(arr, dtype=dtype))

@mandalka.node
class ThresholdedMeanLogKiWithUidSmiles(StorageWorld):
    def build(self, *, source, threshold):
        if isinstance(threshold, str):
            assert threshold in ("mean", "median")
        self.data = source.data.slice[:]
        if isinstance(threshold, str):
            if threshold == "mean":
                threshold = float(np.mean(self.data["value"]))
            elif threshold == "median":
                threshold = float(np.median(self.data["value"]))
            else:
                raise ValueError(threshold)
        thr_value = (self.data["value"] <= threshold).astype(np.int)
        del self.data["value"]
        self.data["value"] = Table(thr_value)
        self.data["value_threshold"] = Variable(threshold)

@mandalka.node
class JJThresholdedMeanLogKiWithUidSmiles(StorageWorld):
    def build(self, *, source, thresholds):
        assert isinstance(thresholds, tuple)
        assert len(thresholds) == 2
        assert thresholds[0] <= thresholds[1]
        self.data = source.data.slice[:]
        thr_value = (self.data["value"] <= thresholds[0]).astype(np.int)
        mask = np.logical_or(
            self.data["value"] <= thresholds[0],
            self.data["value"] >= thresholds[1],
        )
        del self.data["value"]
        self.data["value"] = Table(thr_value)
        self.data = self.data.slice[mask]

class _Pipeline:
    def __init__(self):
        self._nodes = []
        self._data_nodes = []
    def add_node(self, node):
        self._nodes.append(node)
    def add_data_node(self, node):
        self._nodes.append(node)
        self._data_nodes.append(node)
    @property
    def final(self):
        return self._nodes[-1]
    @property
    def nodes(self):
        return tuple(self._nodes)
    @property
    def data_nodes(self):
        return tuple(self._data_nodes)

def mean_warszycki_logki(
        target_uid,
        chembl_filename,
        threshold=None,
        include_earliest_year=None,
        ic50_conversion_strategy="all_relations_half_ic50",
        fit_ic50=False,):

    assert include_earliest_year in (
        None,
        "all_bioactivity_records",
        "Ki_IC50_records",
        "Ki_records",
    )
    assert ic50_conversion_strategy in (
        "all_relations_half_ic50",
        "only_equal_half_ic50",
        "all_relations_only_Ki",
        "only_equal_only_Ki",
    )

    p = _Pipeline()

    p.add_node(SQLiteDatabase(filename=chembl_filename))
    p.add_data_node(WarszyckiBioactivityRaw(db=p.final, target_uid=target_uid))
    if include_earliest_year == "all_bioactivity_records":
        p.add_data_node(EarliestYear(source=p.final))
    p.add_data_node(WarszyckiBioactivity(source=p.final))
    if include_earliest_year == "Ki_IC50_records":
        p.add_data_node(EarliestYear(source=p.final))
    elif include_earliest_year == "Ki_records":
        p.add_data_node(Subset(
            source=p.final,
            data_name="type",
            allowed_values=["Ki"],
            remove_data=False,
        ))
        p.add_data_node(EarliestYear(source=p.final))
    p.add_data_node(WarszyckiLogKi(
        source=p.final,
        conversion_strategy=ic50_conversion_strategy,
    ))
    p.add_data_node(Subset(
        source=p.final,
        data_name="validity_comment",
        allowed_values=["", "Manually validated"],
        remove_data=True,
    ))
    p.add_data_node(Subset(
        source=p.final,
        data_name="potential_duplicate",
        allowed_values=["", "0"],
        remove_data=True,
    ))

    p.add_data_node(ParentMolecules(source=p.final))
    p.add_data_node(ConnectedSmiles(source=p.final))
    p.add_data_node(CanonicalSmiles(source=p.final))
    p.add_data_node(UniqueSmiles(source=p.final))

    if include_earliest_year is not None:
        p.add_data_node(EarliestYear(source=p.final)) # again

    if fit_ic50 is True:
        p.add_data_node(FitOriginalIC50ToKi(source=p.final))

    if include_earliest_year is None:
        p.add_data_node(MeanLogKiWithUidSmiles(source=p.final))
    else:
        p.add_data_node(MeanLogKiWithUidSmilesYearDocUid(source=p.final))

    if threshold is None:
        pass
    elif isinstance(threshold, tuple):
        p.add_data_node(JJThresholdedMeanLogKiWithUidSmiles(
            source=p.final,
            thresholds=threshold,
        ))
    elif isinstance(threshold, (float, str)):
        p.add_data_node(ThresholdedMeanLogKiWithUidSmiles(
            source=p.final,
            threshold=threshold,
        ))
    else:
        raise ValueError()

    return {
        "final": p.final,
        "nodes": p.nodes,
        "data_nodes": p.data_nodes,
    }
