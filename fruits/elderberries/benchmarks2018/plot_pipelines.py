#!/usr/bin/env python3

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

def dump_nodes_to_html(nodes, dirname):
    describe = lambda node, depth: mandalka.unique_id(node) + ": " + mandalka.describe(node, depth)
    href_chembl_compound = lambda uid: href(
        "https://www.ebi.ac.uk/chembl/compound/inspect/{}".format(uid),
        uid,
    )
    href_chembl_document = lambda uid: href(
        "https://www.ebi.ac.uk/chembl/doc/inspect/{}".format(uid),
        uid,
    )
    def dump_node_to_html(node, fname):
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
    os.makedirs(dirname)
    os.makedirs(safe_path_join("per_uid", dirname=dirname))
    os.makedirs(safe_path_join("per_node", dirname=dirname))
    body = []
    for node in nodes:
        nname = sanitize_html(describe(node, depth=1))
        fname = "{}.html".format(node)
        dump_node_to_html(node, safe_path_join("per_node", fname, dirname=dirname))
        body.append(div_template(href(fname, nname)))
        body.append('<pre white-space="pre-wrap">' + sanitize_html(str(node.data)) + '</pre>')
    with open(safe_path_join("per_node", "index.html", dirname=dirname), 'x') as f_out:
        f_out.write(doc_template(
            style_template(
                div_style_1(None),
            ),
            '\n'.join(body),
        ))

    href_uid = lambda uid: href("{}.html".format(uid), uid)
    all_uids = sorted(set([uid for n in nodes for uid in n.data["uid"]]))
    with open(safe_path_join("per_uid", "index.html", dirname=dirname), 'x') as f_out:
        f_out.write(doc_template(
            "",
            '\t'.join([href_uid(uid) for uid in all_uids]),
        ))

    body = []
    for i in range(len(nodes)):
        u1 = set(nodes[i-1].data["uid"]) if i > 0 else set()
        u2 = set(nodes[i].data["uid"])
        new_uids = '\t'.join([href_uid(uid) for uid in sorted(u2-u1)])
        deleted_uids = '\t'.join([href_uid(uid) for uid in sorted(u1-u2)])
        body += [
            div_template("&#8680;\t" + sanitize_html(describe(nodes[i], depth=1))),
            div_template("NEW UIDS"),
            div_template(new_uids),
            div_template("DELETED UIDS"),
            div_template(deleted_uids),
        ]
    with open(safe_path_join("per_uid", "deltas.html", dirname=dirname), 'x') as f_out:
        f_out.write(doc_template(
            style_template(div_style_1()),
            '\n'.join(body),
        ))

    for iuid, uid in enumerate(tqdm.tqdm(all_uids)):
        body = []
        body.append(div_template(href_chembl_compound(uid)))
        for n in nodes:
            body.append(div_template(sanitize_html(describe(n, depth=1))))
            data = n.data.slice[n.data["uid"] == uid]
            arr, header = to_arr_header(data)
            width = columns_width(arr, header, 30)
            arr, header = sanitize_html(arr), sanitize_html(header)
            for i, key in enumerate(header):
                if "uid" in key:
                    arr[:,i] = np.vectorize(href_uid, otypes=(np.str,))(arr[:,i])
            body.append(to_html(arr, header, width, "data_table"))
        body.append(div_template(
            ' '.join([
                href_uid(all_uids[iuid-1]),
                href("index.html", "INDEX"),
                href_uid(all_uids[(iuid+1)%len(all_uids)]),
            ])
        ))
        with open(safe_path_join("per_uid", uid+".html", dirname=dirname), 'x') as f_out:
            f_out.write(doc_template(
                style_template(
                    table_style_1("data_table"),
                    div_style_1() + '\n' + tablesorter(),
                ),
                '\n'.join(body),
            ))

def export(
        target_uid, threshold, include_earliest_year,
        ic50_conversion_strategy, fit_ic50):
    mandalka.config(fake_del_arguments=True) # necessary to describe nodes
    dirname = (
        "target:{}--threshold:{}--year:{}--which:{}"
        "--fitIC50:{}".format(
            target_uid, threshold, include_earliest_year,
            ic50_conversion_strategy, fit_ic50,
        )
    )
    output = os.path.join(
        os.getenv("ANANAS_RESULTS_PATH"),
        "elderberries-Benchmarks2018",
        "data_pipelines",
        dirname,
    )
    if not os.path.exists(output):
        os.makedirs(output)
    else:
        if not os.path.isdir(output):
            raise OSError("remove {}".format(output))
    if len(target_uid)>6 and target_uid[:6] == "CHEMBL" and target_uid[6:].isnumeric():
        print("exporting target {}...".format(target_uid))
        pipeline = mean_warszycki_logki(
            target_uid=target_uid,
            chembl_filename="chembl_24.db",
            threshold=threshold,
            include_earliest_year=include_earliest_year,
            ic50_conversion_strategy=ic50_conversion_strategy,
            fit_ic50=fit_ic50,
        )
        dirname = safe_path_join(target_uid, dirname=output)
        if os.path.exists(dirname):
            print("{} already exists, skipping...".format(dirname))
        else:
            dump_nodes_to_html(
                pipeline["data_nodes"],
                dirname,
            )
    else:
        print("{} is not a valid target, skipping...".format(target_uid))

if __name__ == "__main__":
    TARGET_UIDS = [
        "CHEMBL214", "CHEMBL224", "CHEMBL225", "CHEMBL3371",
        "CHEMBL3155", "CHEMBL226", "CHEMBL251", "CHEMBL217",
        "CHEMBL264", "CHEMBL216"
    ]
    for target_uid in TARGET_UIDS:
        for threshold in (None, 2., "median"):
            for ic50_conversion_strategy in (
                    "all_relations_half_ic50",
                    "only_equal_half_ic50",
                    "all_relations_only_Ki",
                    "only_equal_only_Ki"):
                export(
                    target_uid=target_uid,
                    threshold=threshold,
                    include_earliest_year=None,
                    ic50_conversion_strategy=ic50_conversion_strategy,
                    fit_ic50=False,
                )
