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
from ._mean_warszycki_logki import FitOriginalIC50ToKi, _Pipeline

def raw_logki(
        target_uid,
        chembl_filename,
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

    return {
        "final": p.final,
        "nodes": p.nodes,
        "data_nodes": p.data_nodes,
    }
