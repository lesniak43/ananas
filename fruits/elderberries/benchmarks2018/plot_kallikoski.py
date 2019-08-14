#!/usr/bin/env python3

from itertools import product
import os
import sys

import numpy as np
import tqdm

from bananas.pipelines import mean_warszycki_logki
from crisper import safe_path_join
from herbivores._html import (
    to_arr_header,
    columns_width,
    to_html,
    sanitize_html,
    doc_template,
    style_template,
    div_template,
    table_style_1,
    table_style_2,
    div_style_1,
    href,
    tablesorter,
)
import mandalka



from bananas.worlds import (
    Subset,
    Table,
    Variable,
    StorageWorld,
)




def dump_node_to_html(node, fname):
    href_chembl_compound = lambda uid: href(
        "https://www.ebi.ac.uk/chembl/compound/inspect/{}".format(uid),
        uid,
    )
    href_chembl_document = lambda uid: href(
        "https://www.ebi.ac.uk/chembl/doc/inspect/{}".format(uid),
        uid,
    )
    arr, header = to_arr_header(node.data)
    width = columns_width(arr, header, 30)
    arr, header = sanitize_html(arr), sanitize_html(header)
    for i, key in enumerate(header):
        if "uid" in key and not "doc" in key:
            arr[:,i] = np.vectorize(href_chembl_compound, otypes=(np.str,))(arr[:,i])
        if "uid" in key and "doc" in key:
            arr[:,i] = np.vectorize(href_chembl_document, otypes=(np.str,))(arr[:,i])
    with open(fname, 'x') as f_out:
        f_out.write(doc_template(
            style_template(
                table_style_1("data_table"),
                div_style_1(None),
            ) + '\n' + tablesorter(),
            to_html(arr, header, width, "data_table"),
        ))



@mandalka.node
class Kallikoski(StorageWorld):
    def build(self, *, source, type_data_name):
        self.data = source.data.slice[:]
        mask_Ki = np.vectorize(lambda s: "Ki" in s)(source.data[type_data_name])
        mask_IC50 = np.vectorize(lambda s: "IC50" in s)(source.data[type_data_name])
        assert np.all(np.logical_xor(mask_Ki, mask_IC50))
        set_Ki_uids = set(self.data["uid"][mask_Ki])
        set_IC50_uids = set(self.data["uid"][mask_IC50])
        ki = np.isin(self.data["uid"], list(set_Ki_uids - set_IC50_uids)).astype(np.int)
        ic = np.isin(self.data["uid"], list(set_IC50_uids - set_Ki_uids)).astype(np.int)
        both = np.isin(self.data["uid"], list(set_Ki_uids & set_IC50_uids)).astype(np.int)
        assert np.all((ki + ic + both) == 1), sum(ki + ic + both)
        self.data["Ki_IC50_mask"] = Table(ic + 2*both)
    def get_only_Ki(self):
        return Subset(
            source=self,
            data_name="Ki_IC50_mask",
            allowed_values=[0],
            remove_data=True
        )
    def get_only_IC50(self):
        return Subset(
            source=self,
            data_name="Ki_IC50_mask",
            allowed_values=[1],
            remove_data=True
        )
    def get_both(self):
        return Subset(
            source=self,
            data_name="Ki_IC50_mask",
            allowed_values=[2],
            remove_data=True
        )


if __name__ == "__main__":
    TARGET_UIDS = [
        "CHEMBL214", "CHEMBL224", "CHEMBL225", "CHEMBL3371", "CHEMBL3155",
        "CHEMBL226", "CHEMBL251", "CHEMBL217", "CHEMBL264", "CHEMBL216"
    ]
    CONV = ["all_relations_half_ic50", "only_equal_half_ic50"]
    output = os.path.join(
        os.getenv("ANANAS_RESULTS_PATH"),
        "elderberries-Benchmarks2018",
        "Kallikoski"
    )
    if not os.path.exists(output):
        os.makedirs(output)
    else:
        if not os.path.isdir(output):
            raise OSError("remove {}".format(output))
    for target_uid, ic50_conversion_strategy in product(TARGET_UIDS, CONV):
        n = None
        for _n in mean_warszycki_logki(
            target_uid=target_uid,
            chembl_filename="chembl_24.db",
            threshold=None,
            include_earliest_year=None,
            ic50_conversion_strategy=ic50_conversion_strategy,
            fit_ic50=False,
        )["data_nodes"][::-1]:
            if "original_type" in _n.data:
                n = _n
                break
        n = Kallikoski(source=n, type_data_name="original_type")
        fname = target_uid + '_' + ic50_conversion_strategy + "_"
        dump_node_to_html(n.get_only_Ki(), os.path.join(output, fname + "only_Ki.html"))
        dump_node_to_html(n.get_only_IC50(), os.path.join(output, fname + "only_IC50.html"))
        dump_node_to_html(n.get_both(), os.path.join(output, fname + "both_Ki_IC50.html"))
