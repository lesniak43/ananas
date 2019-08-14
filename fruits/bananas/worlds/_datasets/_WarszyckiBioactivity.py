"""
References
----------
    Warszycki D, Mordalski S, Kristiansen K, Kafel R, Sylte I,
    Chilmonczyk Z, et al. (2013) A Linear Combination of
    Pharmacophore Hypotheses as a New Tool in Search of New
    Active Compounds – An Application for 5-HT1A Receptor
    Ligands. PLoS ONE 8(12): e84510.
    doi:10.1371/journal.pone.0084510
"""

import numpy as np

import mandalka

from .. import StorageWorld, Table

builtin_type = type
type = None

@mandalka.node
class WarszyckiBioactivityRaw(StorageWorld):

    """
    Data stored in this world:
    "uid", "smiles" # index and structure
    "parent_uid", "parent_smiles" # parent index and structure
    "type", "relation", "value", "unit",
    "validity_comment", "potential_duplicate",
    "assay_type",
    "pubmed_id", "year" # bioactivity
    """

    def build(self, db, target_uid):
        for n, v in get_bioactivity_data(db, target_uid):
            self.data[n] = Table(v)

        allowed_validity_comments = set([
            "",
            "Potential missing data",
            "Potential author error",
            "Manually validated",
            "Potential transcription error",
            "Outside typical range",
            "Non standard unit for type",
            #"Non standard unit type",
            "Author confirmed error",
        ])
        allowed_assay_types = set("ABFPTU")
        assert set(self.data["validity_comment"]).issubset(allowed_validity_comments), "Invalid validity_comment: " + ' '.join(sorted(set(self.data["validity_comment"]) - allowed_validity_comments))
        assert set(self.data["assay_type"]).issubset(allowed_assay_types), "Invalid assay_type: " + ' '.join(sorted(set(self.data["assay_type"]) - allowed_assay_types))

@mandalka.node
class WarszyckiBioactivity(StorageWorld):

    """
    Convert types to major types:
        "Ki", "Log Ki", "pKi" -> "Ki",
        "IC50", "Log IC50", "pIC50" -> "IC50".

    Known relations and inversed relations:
        "": "",
        "<": ">",
        "<=": ">=",
        "=": "=",
        ">": "<",
        ">=": "<=",
        "~": "~".

    Known units and their ratio to "nM":
        "M": 1e9,
        "mM": 1e6,
        "uM": 1e3,
        "nM": 1e0,
        "pM": 1e-3,
        "fM": 1e-6.

    Convert units to:
        "Ki", "IC50" -> "nM".

    Drop everything else.
    """

    def build(self, source):

        def _standardize(type, value, unit, relation):
            try:
                return standardize(type, value, unit, relation)
            except ValueError:
                return "", np.nan, "", ""

        type, value, unit, relation = zip(*[
            _standardize(*r) for r in zip(
                source.data["type"],
                source.data["value"],
                source.data["unit"],
                source.data["relation"]
            )
        ])

        type = np.array(type, dtype=source.data["type"].dtype)
        value = np.array(value, dtype=source.data["value"].dtype)
        unit = np.array(unit, dtype=source.data["unit"].dtype)
        relation = np.array(relation, dtype=source.data["relation"].dtype)
        ok_mask = np.logical_not(np.isnan(value))

        self.data = source.data.slice[ok_mask]
        self.data["type"] = self.data.get_container("type")(type[ok_mask])
        self.data["value"] = self.data.get_container("value")(value[ok_mask])
        self.data["unit"] = self.data.get_container("unit")(unit[ok_mask])
        self.data["relation"] = self.data.get_container("relation")(relation[ok_mask])

@mandalka.node
class WarszyckiLogKi(StorageWorld):

    """
    Standarize units and relations to log10(Ki[nM]).
    Drop IC50 with "assay_type" other than "B"
    Drop columns "type", "unit", "assay_type", add "original_type".
    Convert if:
    - input type/unit in [Ki[nM], IC50[nM]] AND
    - relation in ["=", ">", ">=", "<", "<=", "~"] AND
    - log10(value_to_Ki) makes sense,
    drop otherwise.
    """

    def build(self, source, conversion_strategy):

        assert conversion_strategy in (
            "all_relations_half_ic50",
            "only_equal_half_ic50",
            "all_relations_only_Ki",
            "only_equal_only_Ki",
        )

        def standardize_logki(type, value, unit, relation, assay_type):
            relations = {
                "all_relations_half_ic50": ["=", ">", ">=", "<", "<=", "~"],
                "only_equal_half_ic50": ["=", "~"],
                "all_relations_only_Ki": ["=", ">", ">=", "<", "<=", "~"],
                "only_equal_only_Ki": ["=", "~"],
            }[conversion_strategy]
            types = {
                "all_relations_half_ic50": ["Ki", "IC50"],
                "only_equal_half_ic50": ["Ki", "IC50"],
                "all_relations_only_Ki": ["Ki"],
                "only_equal_only_Ki": ["Ki"],
            }[conversion_strategy]
            if type == "IC50" and assay_type != "B":
                return "", np.nan, ""
            if unit == "nM" and relation in relations and type in types:
                if type == "Ki":
                    value = np.log10(value) # might be nan
                elif type == "IC50":
                    value = np.log10(value/2.) # might be nan
                else:
                    raise ValueError()
            else:
                return "", np.nan, ""
            return type, value, relation

        def _standardize(type, value, unit, relation, assay_type):
            return standardize_logki(type, value, unit, relation, assay_type)

        l = [
            _standardize(*r) for r in zip(
                source.data["type"],
                source.data["value"],
                source.data["unit"],
                source.data["relation"],
                source.data["assay_type"]
            )
        ]
        if len(l) > 0:
            original_type, value, relation = zip(*l)
        else:
            original_type, value, relation = [], [], []

        original_type = np.array(original_type, dtype=source.data["type"].dtype)
        value = np.array(value, dtype=source.data["value"].dtype)
        relation = np.array(relation, dtype=source.data["relation"].dtype)
        ok_mask = np.logical_not(np.isnan(value))

        self.data = source.data.slice[ok_mask]
        del self.data["type"]
        del self.data["unit"]
        del self.data["assay_type"]
        self.data["value"] = self.data.get_container("value")(value[ok_mask])
        self.data["relation"] = self.data.get_container("relation")(relation[ok_mask])
        self.data["original_type"] = Table(original_type[ok_mask])

def get_bioactivity_data(db, target_uid):
    q = """
        SELECT
            md.chembl_id,
            cs.canonical_smiles,
            parent_md.chembl_id,
            parent_cs.canonical_smiles,
            act.standard_type,
            act.standard_relation,
            act.standard_value,
            act.standard_units,
            act.data_validity_comment,
            act.potential_duplicate,
            a.assay_type,
            d.chembl_id,
            d.pubmed_id,
            d.year
        FROM
            activities act
        JOIN
            molecule_dictionary md
            ON md.molregno = act.molregno
        JOIN
            compound_structures cs
            ON cs.molregno = act.molregno
        LEFT JOIN
            molecule_hierarchy mh
            ON act.molregno = mh.molregno
        LEFT JOIN
            molecule_dictionary parent_md
            ON parent_md.molregno = mh.parent_molregno
        LEFT JOIN
            compound_structures parent_cs
            ON parent_cs.molregno = parent_md.molregno
        LEFT JOIN
            docs d
            ON act.doc_id = d.doc_id
        JOIN
            assays a
            ON act.assay_id = a.assay_id
        JOIN
            target_dictionary td
            ON a.tid = td.tid
        WHERE
            td.chembl_id = ?
    """
    names = [
        "uid", "smiles",
        "parent_uid", "parent_smiles",
        "type", "relation", "value", "unit",
        "validity_comment", "potential_duplicate",
        "assay_type",
        "doc_uid", "pubmed_id", "year"
    ]
    arrays = db.query2(
        q,
        target_uid,
        dtypes=[
            np.str,     # uid
            np.object,  # smiles
            np.str,      # parent_uid
            np.object,  # parent_smiles
            np.str,     # type
            np.str,     # relation
            np.float32, # value
            np.str,     # unit
            np.str,     # validity_comment
            np.str,     # potential_duplicate
            np.str,     # assay_type
            np.str,     # doc_uid
            np.str,     # pubmed_id
            np.int,     # year
        ],
        defaults=[
            "",         # uid
            "",         # smiles
            "",         # parent_uid
            "",         # parent_smiles
            "",         # type
            "",         # relation
            np.nan,     # value
            "",         # unit
            "",         # validity_comment
            "",         # potential_duplicate
            "",         # assay_type
            "",         # doc_uid
            "",         # pubmed_id
            0,          # year
        ],
    )
    return zip(names, arrays)

def standardize(type, value, unit, relation):
    # Ki[nM]
    if type in ["Ki", "Log Ki", "pKi"]:
        if type in ["Log Ki", "pKi"]:
            value, unit, relation = undo_chem_log_mols(value, unit, relation)
        elif type in ["Ki"]:
            pass
        else:
            raise ValueError()
        value, unit = scale_to_nm(value, unit)
        assert unit == "nM"
        return "Ki", value, unit, relation
    # IC50[nM]
    elif type in ["IC50", "Log IC50", "pIC50"]:
        if type in ["Log IC50", "pIC50"]:
            value, unit, relation = undo_chem_log_mols(value, unit, relation)
        elif type in ["IC50"]:
            pass
        else:
            raise ValueError()
        value, unit = scale_to_nm(value, unit)
        assert unit == "nM"
        return "IC50", value, unit, relation
    else:
        raise ValueError()

def undo_chem_log_mols(value, unit, relation):
    """
    Sometimes chemists like to report -log10(value[M])
    as Log sth or psth.
    This is usually a bad idea, let's undo.
    'unit' must be equal to "". Result unit will always be "M"
    10 ** (-value) is a decreasing function,
    we need to invert the relation.
    """
    inverted_relation = {
        "": "",
        "<": ">",
        "<=": ">=",
        "=": "=",
        ">": "<",
        ">=": "<=",
        "~": "~",
    }
    if not relation in inverted_relation:
        raise ValueError()
    elif not unit in [""]:
        raise ValueError()
    else:
        return 10 ** (-value), "M", inverted_relation[relation]

def scale_to_nm(value, unit):
    """
    Everyone loves nanomols.
    """
    ratio = {
        "M": 1e9,
        "mM": 1e6,
        "uM": 1e3,
        "nM": 1e0,
        "pM": 1e-3,
        "fM": 1e-6,
    }
    if unit in ratio:
        return value * ratio[unit], "nM"
    else:
        raise ValueError()

