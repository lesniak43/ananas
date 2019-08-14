from collections import Counter
from itertools import groupby

FONTSIZE = 15

import matplotlib
matplotlib.use('Agg')
matplotlib.rc('font', size=FONTSIZE)
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from numba import njit
import numpy as np
from scipy.stats import spearmanr, pearsonr, norm, uniform
import tqdm

import crisper

from bananas.pipelines import mean_warszycki_logki
from bananas.worlds import (
    BalancedAgglomerativeClustering,
    CrossValidation,
    KernelTSNE,
    Morgan,
    MurckoScaffoldSplit,
    PaperSplit,
    SMILESToMol,
    SpectralClustering,
    StoredCopy,
    TanimotoMinMaxRepresentationMaker,
    TwoClassLogisticRegression,
)
from elderberries.benchmarks2018.problem import (
    Benchmarks2018StructuralSimilarity,
    Benchmarks2018ProblemClassificationSummary,
)
from elderberries.benchmarks2018.solutions import (
    fingerprinter_by_name,
)

def target_name(target_uid):
    return {
        "CHEMBL214": "5-HT1A",
        "CHEMBL224": "5-HT2A",
        "CHEMBL225": "5-HT2C",
        "CHEMBL3371": "5-HT6",
        "CHEMBL3155": "5-HT7",
        "CHEMBL226": "A1",
        "CHEMBL251": "A2A",
        "CHEMBL217": "D2",
        "CHEMBL264": "H3",
        "CHEMBL216": "M1",
    }[target_uid]

weighted_accuracy = Benchmarks2018ProblemClassificationSummary.metrics["Weighted_Accuracy"][0]
accuracy = Benchmarks2018ProblemClassificationSummary.metrics["Accuracy"][0]

spearman = lambda x, y: spearmanr(x,y)[0]

to_pki = lambda logki: 9. - logki

def _table(rows, cols, content, delimiter='\t'):
    result = [delimiter.join([''] + list(cols)) + '\n']
    for row_name, row in zip(rows, content):
        result.append(delimiter.join([row_name] + list(row)) + '\n')
    return ''.join(result)

def _arr_header_to_html(arr, header):
    from herbivores._html import (
        to_arr_header,
        columns_width,
        to_html,
        sanitize_html,
        doc_template,
        style_template,
        table_style_1,
        div_style_1,
        href,
        tablesorter,
    )
    href_chembl_compound = lambda uid: href(
        "https://www.ebi.ac.uk/chembl/compound/inspect/{}".format(uid),
        uid,
    )
    href_chembl_document = lambda uid: href(
        "https://www.ebi.ac.uk/chembl/doc/inspect/{}".format(uid),
        uid,
    )
    width = columns_width(arr, header, 30)
    arr, header = sanitize_html(arr), sanitize_html(header)
    for i, key in enumerate(header):
        if "uid" in key and not "doc" in key:
            arr[:,i] = np.vectorize(href_chembl_compound, otypes=(np.str,))(arr[:,i])
        if "uid" in key and "doc" in key:
            arr[:,i] = np.vectorize(href_chembl_document, otypes=(np.str,))(arr[:,i])
    return doc_template(
        style_template(
            table_style_1("data_table"),
            div_style_1(None),
        ) + '\n' + tablesorter(),
        to_html(arr, header, width, "data_table"),
    )

def jj_thresholded_ki(N=10, N_SPLITS=5, target_uid="CHEMBL214", split_name="cv", C=10., class_weight="balanced", weighted_score=True):

    from elderberries.benchmarks2018.problem import Benchmarks2018StructuralSimilarity

    preds = {}
    scores = np.zeros((N_SPLITS,N,N), dtype=np.float)
    lspace = np.linspace(0.,3.,10)
    for i in range(N):
        for j in range(N):
            if i <= j:
                thresholds = tuple((lspace[x] for x in (i,j)))
                dataset = mean_warszycki_logki(
                    target_uid=target_uid,
                    chembl_filename="chembl_24.db",
                    threshold=thresholds,
                )["final"]

                if split_name == "cv":
                    split_ = CrossValidation(
                        source=dataset,
                        n_groups=N_SPLITS,
                        seed=43,
                    )
                elif split_name == "bac":
                    split_ =  BalancedAgglomerativeClustering(
                        source=Benchmarks2018StructuralSimilarity(source=dataset),
                        kernel="kernel",
                        n_groups=N_SPLITS,
                    )
                else:
                    raise ValueError("split_name: {}".format(split_name))

                for n_split, split in enumerate(split_.get_splits()):
                    tr, te = split.get_train(), split.get_test()
                    fpr = Morgan(
                        radius=4,
                        use_chirality=True,
                        use_bond_types=True,
                        use_features=False,
                        converter=SMILESToMol(),
                    )
                    fp_tr = fpr(source=tr)
                    fp_te = fpr(source=te)
                    repr_maker = TanimotoMinMaxRepresentationMaker(
                        fingerprint=fp_tr)
                    repr_tr = repr_maker(fingerprint=fp_tr)
                    repr_te = repr_maker(fingerprint=fp_te)
                    model = TwoClassLogisticRegression(
                        source=repr_tr,
                        C=C,
                        class_weight=class_weight,
                    )
                    pred = StoredCopy(source=model.predict(source=repr_te))
                    preds[(n_split, i, j)] = (te, pred)

    crisper.evaluate(
        *[k for tup in preds.values() for k in tup],
        label="J&J"
    )

    for (n_split, i, j), (te, pred) in tqdm.tqdm(preds.items()):
        if weighted_score:
            scores[n_split, i, j] = weighted_accuracy(None, te, pred)
        else:
            scores[n_split, i, j] = accuracy(None, te, pred)
    scores_ = scores.mean(axis=0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(scores_, origin="lower", vmin=sorted(set(scores_.ravel()))[1], vmax=sorted(scores_.ravel())[-1])
    lspace_ = np.array(["{:.2f}".format(to_pki(x)) for x in lspace])
    idx = np.arange(0,N,2)
    ax.set_xticks(idx)
    ax.set_xticklabels(lspace_[idx])
    ax.set_yticks(idx)
    ax.set_yticklabels(lspace_[idx])
    ax.set_xlabel("Inactivity threshold (pKi)")
    ax.set_ylabel("Activity threshold (pKi)")
    if weighted_score:
        ax.set_title("Weighted Accuracy")
    else:
        ax.set_title("Accuracy")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig

def fingercheats(
        target_uids, fpr_names, include_earliest_year=None,
        ic50_conversion_strategy="all_relations_half_ic50",
        fit_ic50=False):
    cor = np.zeros((len(target_uids), len(fpr_names)), dtype=np.float)
    cor2 = np.zeros((len(target_uids), len(fpr_names)), dtype=np.float)
    for i, target_uid in enumerate(target_uids):
        ds = mean_warszycki_logki(
            target_uid=target_uid,
            chembl_filename="chembl_24.db",
            threshold=None,
            include_earliest_year=include_earliest_year,
            ic50_conversion_strategy=ic50_conversion_strategy,
            fit_ic50=fit_ic50,
        )["final"]
        for j, fpr_name in enumerate(fpr_names):
            fpr = fingerprinter_by_name[fpr_name]
            a = np.array(fpr(source=ds).data[("fingerprint", "data")].sum(axis=1)).ravel()
            b = ds.data["value"]
            cor[i,j] = spearmanr(a,b)[0]
            cor2[i,j] = pearsonr(a,b)[0]

    fig = plt.figure(figsize=(16,6))

    ax = fig.add_subplot(121)
    fig.colorbar(ax.imshow(cor), ax=ax, orientation="horizontal")
    ax.set_title("Spearman rank-order correlation coefficient")
    ax.set_yticks(np.arange(len(target_uids)))
    ax.set_yticklabels([target_name(u) for u in target_uids])
    ax.set_xticks(range(0, len(fpr_names), 2))
    ax.set_xticklabels(["FP{}".format(i+1) for i in range(0, len(fpr_names), 2)])

    ax = fig.add_subplot(122)
    fig.colorbar(ax.imshow(cor2), ax=ax, orientation="horizontal")
    ax.set_title("Pearson correlation coefficient")
    ax.set_yticks(np.arange(len(target_uids)))
    ax.set_yticklabels([target_name(u) for u in target_uids])
    ax.set_xticks(range(0, len(fpr_names), 2))
    ax.set_xticklabels(["FP{}".format(i+1) for i in range(0, len(fpr_names), 2)])

    return (
        fig,
        ''.join(["FP{}: {}\n".format(i+1, fpr_name) \
            for i, fpr_name in enumerate(fpr_names)]),
    )

def fingercheats_thr(
        target_uids, fpr_names, threshold=2., include_earliest_year=None,
        ic50_conversion_strategy="all_relations_half_ic50",
        fit_ic50=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    result = '\t'.join([''] + ["FP{}".format(i+1) for i in range(len(fpr_names))]) + '\n'
    acc = np.zeros((len(target_uids), len(fpr_names)), dtype=np.float)
    for i, target_uid in enumerate(target_uids):
        row = "{}\t".format(target_name(target_uid))
        ds = mean_warszycki_logki(
            target_uid=target_uid,
            chembl_filename="chembl_24.db",
            threshold=threshold,
            include_earliest_year=include_earliest_year,
            ic50_conversion_strategy=ic50_conversion_strategy,
            fit_ic50=fit_ic50,
        )["final"]
        for j, fpr_name in enumerate(fpr_names):
            fpr = fingerprinter_by_name[fpr_name]
            X = np.array(fpr(source=ds).data[("fingerprint", "data")].sum(axis=1)).reshape(-1,1)
            y = ds.data["value"].ravel()
            assert set(y) == set([0., 1.])
            lr = LogisticRegression(class_weight="balanced")
            lr.fit(X, y)
            acc[i,j] = balanced_accuracy_score(y, lr.predict(X))
            row += "{:.3f}\t".format(acc[i,j])
        result += row + '\n'
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    fig.colorbar(ax.imshow(acc), ax=ax, orientation="horizontal")
    ax.set_title("Weighted accuracy")
    ax.set_yticks(np.arange(len(target_uids)))
    ax.set_yticklabels([target_name(u) for u in target_uids])
    ax.set_xticks(range(0, len(fpr_names), 2))
    ax.set_xticklabels(["FP{}".format(i+1) for i in range(0, len(fpr_names), 2)])
    return (
        fig,
        result,
        ''.join(["FP{}: {}\n".format(i+1, fpr_name) \
            for i, fpr_name in enumerate(fpr_names)]),
    )

def min_max_mean_per_paper(
        target_uids,
        include_earliest_year,
        ic50_conversion_strategy,
        fit_ic50,
        min_paper_size):
    fig = plt.figure(figsize=(12.3, len(target_uids)*4))
    counter = 0
    axes = []
    results = []
    for target_uid in target_uids:
        d = mean_warszycki_logki(
            target_uid=target_uid,
            chembl_filename="chembl_24.db",
            threshold=None,
            include_earliest_year=include_earliest_year,
            ic50_conversion_strategy=ic50_conversion_strategy,
            fit_ic50=fit_ic50,
        )["final"]
        result = []
        doc_uid = d.data["doc_uid"]
        value = to_pki(d.data["value"])
        key = lambda x: x[0]
        for k, g in groupby(sorted(zip(doc_uid, value), key=key), key):
            gu, gv = zip(*g)
            if len(gv) >= min_paper_size:
                tup = (np.min(gv), np.max(gv), np.mean(gv))
                result.append(tup)
                results.append(tup)
        for h in zip(*result):
            counter += 1
            ax = fig.add_subplot(len(target_uids), 3, counter)
            axes.append(ax)
            ax.hist(h, bins=43, range=(value.min(), value.max()))
            if counter % 3 == 1:
                ax.set_ylabel(target_name(target_uid) + '\n')
            ax.set_xlabel({
                1: "Min pKi per paper (earliest)",
                2: "Max pKi per paper (earliest)",
                0: "Mean pKi per paper (earliest)",
            }[counter % 3])
    xlim = (np.array(results).min()-.1, np.array(results).max()+.1)
    [ax.set_xlim(xlim) for ax in axes] 
    fig.tight_layout()
    return fig

def how_many_records_per_paper(
        target_uids,
        include_earliest_year="all_bioactivity_records",
        ic50_conversion_strategy="all_relations_half_ic50",
        fit_ic50=True):
    fig = plt.figure(figsize=(4.3, len(target_uids)*4))
    for i, target_uid in enumerate(target_uids):
        d = mean_warszycki_logki(
            target_uid=target_uid,
            chembl_filename="chembl_24.db",
            threshold=None,
            include_earliest_year=include_earliest_year,
            ic50_conversion_strategy=ic50_conversion_strategy,
            fit_ic50=fit_ic50,
        )["final"]
        ax = fig.add_subplot(len(target_uids), 1, i+1)
        v = list(Counter(d.data["doc_uid"]).values())
        ax.hist(v, bins=int(np.max(v)))
        ax.set_xlabel("Earliest records per paper")
        ax.set_ylabel(target_name(target_uid) + "\n")
        ax.set_yscale("log", nonposy='clip')
    fig.tight_layout()
    return fig

def earliest_year_variants(target_uids):
    def compare_year(*ds):
            uids = np.array(sorted(set.union(*[set(d.data["uid"]) for d in ds])))
            years = np.empty((len(uids),len(ds)),dtype=np.float)
            years.fill(np.nan)
            for i, d in enumerate(ds):
                idx = np.searchsorted(uids, d.data["uid"])
                years[idx,i] = d.data["year"]
            return years
    result = [
        "Reference method: 'all_bioactivity_records'\n",
        "Other:\n",
        "    'Ki_IC50_records'\n",
        "    'Ki_records'\n",
        "target: differing/total\n",
    ]
    for target_uid in target_uids:
        ds = []
        for include_earliest_year in ["all_bioactivity_records", "Ki_IC50_records", "Ki_records"]:
            ds.append(mean_warszycki_logki(
                target_uid=target_uid,
                chembl_filename="chembl_24.db",
                threshold=None,
                include_earliest_year=include_earliest_year,
                ic50_conversion_strategy="all_relations_half_ic50",
            )["final"])
        y = compare_year(*ds)
        a = np.all(
            np.logical_or(
                y == np.nanmax(y, axis=1).reshape(-1,1),
                np.isnan(y)
            ),
            axis=1,
        )
        result.append("{}: {}/{}\n".format(target_name(target_uid), len(a)-sum(a), len(a)))
    return ''.join(result)

def activity_variants(target_uids, conversion_strategies, reference_idx):
    def compare_Ki(*ds):
        uids = np.array(sorted(set.union(*[set(d.data["uid"]) for d in ds])))
        value = np.empty((len(uids),len(ds)),dtype=np.float)
        value.fill(np.nan)
        for i, d in enumerate(ds):
            idx = np.searchsorted(uids, d.data["uid"])
            assert np.all(uids[idx] == d.data["uid"])
            value[idx,i] = d.data["value"]
        return value

    fig = plt.figure(figsize=(4*len(conversion_strategies),4*len(target_uids)))
    fig2 = plt.figure(figsize=(4*len(conversion_strategies),4*len(target_uids)))
    ax_counter = 0
    for target_uid in target_uids:
        ds = []
        corrections = []
        for ic50_conversion_strategy, fit_ic50, _ in conversion_strategies:
            dct = mean_warszycki_logki(
                target_uid=target_uid,
                chembl_filename="chembl_24.db",
                threshold=None,
                include_earliest_year=None,
                ic50_conversion_strategy=ic50_conversion_strategy,
                fit_ic50=fit_ic50,
            )
            ds.append(dct["final"])
            correction = None
            if fit_ic50:
                for n in reversed(dct["data_nodes"]):
                    try:
                        correction = n.data["IC50_correction"]
                        break
                    except KeyError:
                        pass
                assert correction is not None
            else:
                correction = 0.
            corrections.append(correction)
        value = compare_Ki(*ds)
        ref_label = conversion_strategies[reference_idx][2]
        for i, (_, fit_ic50, label) in enumerate(conversion_strategies):
            ax_counter += 1
            ax = fig.add_subplot(len(target_uids),len(conversion_strategies),ax_counter)
            ax.scatter(to_pki(value[:,reference_idx]), to_pki(value[:,i]), s=8)
            if fit_ic50:
                ax.set_title("(coefficient: {:.3f})".format(2*10**(-corrections[i])))
            ax.set_xlabel("{} (reference)".format(ref_label))
            ax.set_ylabel(
                target_name(target_uid) + '\n\n' + label if i == 0 else label
            )
            ax = fig2.add_subplot(len(target_uids),len(conversion_strategies),ax_counter)
            ax.hist(to_pki(ds[i].data["value"]), bins=43)
            ax.set_xlabel(label)
            if i == 0:
                ax.set_ylabel(target_name(target_uid) + '\n')
    fig.tight_layout()
    fig2.tight_layout()
    return fig, fig2

def median_thresholded_activity_variants(
        target_uids, conversion_strategies):
    medians = np.zeros(
        (len(target_uids), len(conversion_strategies)),
        dtype=np.float,
    )
    for i, target_uid in enumerate(target_uids):
        for j, (ic50_conversion_strategy, fit_ic50, _) in enumerate(conversion_strategies):
            medians[i,j] = mean_warszycki_logki(
                target_uid=target_uid,
                chembl_filename="chembl_24.db",
                threshold="median",
                include_earliest_year=None,
                ic50_conversion_strategy=ic50_conversion_strategy,
                fit_ic50=fit_ic50,
            )["final"].data["value_threshold"]
    medians = to_pki(medians)
    labels = [l for _, _, l in conversion_strategies]
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111)
    im = ax.imshow(medians.T)

    ax.set_xticks(range(len(target_uids)))
    ax.set_xticklabels([target_name(u) for u in target_uids])
#    ax.set_xlabel("Target")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)

    ax.set_yticks(range(len(conversion_strategies)))
    ax.set_yticklabels(labels)
#    ax.set_ylabel("log Ki variant")

    fig.colorbar(im, ax=ax, orientation='horizontal')
    ax.set_title("Median pKi", fontsize=int(FONTSIZE*1.5))

    fig.tight_layout()
    txt = _table(
        rows=np.array([target_name(u) for u in target_uids]),
        cols=np.array(labels),
        content=np.vectorize(lambda f: "{:.3f}".format(f))(medians),
        delimiter='\t'
    )
    return fig, txt

def density_bias(target_uids):
    def _distance_to_nth_neighbour(kernel, value):
        result = []
        for row in reversed(np.sort(kernel, axis=0)):
            result.append(spearman(row, value))
        return np.array(result, dtype=np.float)
    def _n_neighbours_in_radius(kernel, value):
        result = []
        lsp = np.linspace(0,1,201)
        for thr in lsp:
            x = np.sum(kernel>=thr, axis=1)
            result.append(spearman(x, value))
        return lsp, np.array(result, dtype=np.float)
    def _stationary(kernel, value, n=None):
        if n is not None:
            mask = np.zeros(kernel.shape, dtype=np.bool)
            for i, row in enumerate(kernel):
                mask[i,np.argsort(row)[-n:]] = True
            kernel = 0.001 * np.ones(kernel.shape, dtype=np.float)
            kernel[mask] = 1.
        _a = kernel/kernel.sum(axis=0).reshape(1,-1)
        a = _a - np.eye(len(value))
        b = np.zeros(len(value)+1)
        a = np.concatenate((a, np.ones(len(value)).reshape(1,-1)), axis=0)
        b[-1] = 1.
        x = np.linalg.lstsq(a,b)[0]
        return spearman(x, value)
    fig = plt.figure(figsize=(8, 4*len(target_uids)))
    for i, target_uid in enumerate(target_uids):
        ds = mean_warszycki_logki(
            target_uid=target_uid,
            chembl_filename="chembl_24.db",
            threshold=None,
            include_earliest_year="all_bioactivity_records",
            ic50_conversion_strategy="all_relations_half_ic50",
            fit_ic50=False,
        )["final"]
        c_doc_uid = Counter(ds.data["doc_uid"])
        x = np.vectorize(lambda uid: c_doc_uid[uid])(ds.data["doc_uid"])
        y = value = to_pki(ds.data["value"])
        kernel = Benchmarks2018StructuralSimilarity(source=ds).data["kernel"]
        result1 = _distance_to_nth_neighbour(kernel, value)
        lsp2, result2 = _n_neighbours_in_radius(kernel, value)
        _min, _max = min(np.nanmin(result1), np.nanmin(result2)), max(np.nanmax(result1), np.nanmax(result2))

        ax = fig.add_subplot(len(target_uids),2,2*i+1)
        x = np.arange(len(result1))
        mask = np.logical_not(np.isnan(result1))
        ax.plot(x[mask], result1[mask])
        ax.set_xlabel("Distance-sorted neighbours")
        ax.set_ylabel(target_name(target_uid) + "\n\nSpearman's Rho")
        ax.set_ylim((_min-.05, _max+.05))

        ax = fig.add_subplot(len(target_uids),2,2*i+2)
        mask = np.logical_not(np.isnan(result2))
        ax.plot(lsp2[mask], result2[mask])
        ax.set_xlabel("Similarity threshold")
        ax.set_ylabel("Spearman's Rho")
        ax.set_ylim((_min-.05, _max+.05))

    fig.tight_layout()
    return fig

def similar_compounds(target_uid, n_top, n_bottom, n_random, seed=43):
    ds = mean_warszycki_logki(
        target_uid=target_uid,
        chembl_filename="chembl_24.db",
        threshold=None,
        include_earliest_year="all_bioactivity_records",
        ic50_conversion_strategy="all_relations_half_ic50",
        fit_ic50=False,
    )["final"]
    uid = ds.data["uid"]
    kernel = Benchmarks2018StructuralSimilarity(source=ds).data["kernel"]
    ix, iy = np.tril_indices(kernel.shape[0], -1)
    idx = np.argsort(kernel[ix, iy])
    l = len(idx)
    idx = idx[np.sort(np.concatenate((
        np.arange(n_bottom),
        np.arange(l-n_top, l),
        n_bottom + np.random.RandomState(seed=seed).choice(
            l - n_top - n_bottom,
            size=n_random,
            replace=False,
        )
    )))]
    ix, iy = ix[idx], iy[idx]
    uid1, uid2 = uid[ix], uid[iy]
    sim = np.vectorize(lambda f: "~{:.4f}".format(f))(kernel[ix, iy])
    arr = np.stack((uid1, uid2, sim), axis=1)
    header = np.array(["uid", "uid", "similarity"])
    return _arr_header_to_html(arr, header)

def same_paper_cross_paper(target_uids):
    fig = plt.figure(figsize=(len(target_uids)*4, 4))
    for i, target_uid in enumerate(target_uids):
        d = Benchmarks2018StructuralSimilarity(source=mean_warszycki_logki(
            target_uid=target_uid,
            chembl_filename="chembl_24.db",
            threshold=None,
            include_earliest_year="all_bioactivity_records",
            ic50_conversion_strategy="all_relations_half_ic50",
            fit_ic50=True,
        )["final"])
        kernel = d.data["kernel"]
        same_paper = d.data["doc_uid"].reshape(1,-1) == d.data["doc_uid"].reshape(-1,1)
        cross_paper = np.logical_not(same_paper)
        same_paper[range(len(same_paper)),range(len(same_paper))] = False
        ax = fig.add_subplot(1,len(target_uids),i+1)
        ax.hist(kernel.ravel()[same_paper.ravel()], bins=43, label="same paper", alpha=.5, density=True)
        ax.hist(kernel.ravel()[cross_paper.ravel()], bins=43, label="cross paper", alpha=.5, density=True)
        ax.legend()
        ax.set_xlabel("Structural similarity")
        ax.set_title(target_name(target_uid))
    fig.tight_layout()
    return fig

def year_structural_pareto(target_uids):
    @njit
    def _first(arr, x):
        for i in range(len(arr)):
            if arr[i] == x:
                return i
        raise ValueError()
    result = []
    for i, target_uid in enumerate(target_uids):
        result.append("TARGET: {}".format(target_name(target_uid)))
        result.append("")
        d = Benchmarks2018StructuralSimilarity(source=mean_warszycki_logki(
            target_uid=target_uid,
            chembl_filename="chembl_24.db",
            threshold=None,
            include_earliest_year="all_bioactivity_records",
            ic50_conversion_strategy="all_relations_half_ic50",
            fit_ic50=True,
        )["final"])
        kernel = d.data["kernel"]
        year = d.data["year"]
        idx = np.flip(np.argsort(kernel.ravel()))
        delta_year = np.abs(year.reshape(-1,1) - year.reshape(1,-1)).ravel()[idx]
        for dy in sorted(set(delta_year.ravel())-set([0,0.])):
            _idx = idx[_first(delta_year, dy)]
            i, j = _idx // kernel.shape[0], _idx % kernel.shape[0]
            result.append("SIMILARITY: {:.3f}, DELTA YEAR: {}".format(
                kernel[i,j],
                int(dy)
            ))
            for m in (i,j):
                result.append("UID: {}, SMILES: {}, VALUE: {}, YEAR: {}, DOC_UID: {}".format(
                    d.data["uid"][m],
                    d.data["smiles"][m],
                    d.data["value"][m],
                    int(d.data["year"][m]),
                    d.data["doc_uid"][m],
                ))
            result.append("")
    return '\n'.join(result) + '\n'

def aaaiiaii(value, groups, kernel, time_split):
    from numba import jit, njit
    result_all = np.zeros((kernel.size, 4), dtype=np.float)
    result_all_groups = np.zeros((kernel.size, 4), dtype=np.float)
    result_all_counter = np.zeros(4, dtype=np.int)
    result_nearest = np.empty((kernel.shape[0],2), dtype=np.float)
    result_nearest.fill(np.nan)
    @njit
    def f(value, groups, kernel, result_all, result_all_groups, result_all_counter, result_nearest):
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                if groups[i] > groups[j] or (groups[i] < groups[j] and not time_split): # test to train
                    idx = 3-(2*int(value[i])+int(value[j])) # aa ai ia ii
                    result_all[result_all_counter[idx], idx] = kernel[i,j]
                    result_all_groups[result_all_counter[idx], idx] = groups[i]
                    result_all_counter[idx] += 1
                    if np.isnan(result_nearest[i, value[j]]) or kernel[i,j] > result_nearest[i, value[j]]:
                        result_nearest[i, value[j]] = kernel[i,j]
    f(value, groups, kernel, result_all, result_all_groups, result_all_counter, result_nearest)
    return {
        "aa": (result_all[:result_all_counter[0],0], result_all_groups[:result_all_counter[0],0]),
        "ai": (result_all[:result_all_counter[1],1], result_all_groups[:result_all_counter[1],1]),
        "ia": (result_all[:result_all_counter[2],2], result_all_groups[:result_all_counter[2],2]),
        "ii": (result_all[:result_all_counter[3],3], result_all_groups[:result_all_counter[3],3]),
        "nearest_i": result_nearest[:,0],
        "nearest_a": result_nearest[:,1],
    }

def splits_analysis(target_uids):
    def plot(value, groups, kernel, axes, split_label, time_split=False):
        dct = aaaiiaii(value, groups, kernel, time_split)

        not_nan_mask = np.logical_not(np.logical_or(
            np.isnan(dct["nearest_a"]),
            np.isnan(dct["nearest_i"]),
        ))
        aa, ai, ia, ii = (
            dct["nearest_a"][not_nan_mask][value[not_nan_mask]==1],
            dct["nearest_i"][not_nan_mask][value[not_nan_mask]==1],
            dct["nearest_a"][not_nan_mask][value[not_nan_mask]==0],
            dct["nearest_i"][not_nan_mask][value[not_nan_mask]==0],
        )

        histtype, linewidth = "step", 3
        axes[0].hist(
            aa, bins=43, label="AA",
            density=True, histtype=histtype, linewidth=linewidth,
        )
        axes[0].hist(
            ai, bins=43, label="AI",
            density=True, histtype=histtype, linewidth=linewidth,
        )
        axes[0].hist(
            ia, bins=43, label="IA",
            density=True, histtype=histtype, linewidth=linewidth,
        )
        axes[0].hist(
            ii, bins=43, label="II",
            density=True, histtype=histtype, linewidth=linewidth,
        )
        axes[0].set_xlim((0.,1.))
        axes[0].set_xlabel("Nearest neighbour similarity")
        axes[0].set_ylabel(split_label + '\n')
        axes[0].legend()

        S = 8
        axes[1].scatter(ia, ii, label="inactive", c="green", s=S, alpha=.3)
        axes[1].scatter(aa, ai, label="active", c="xkcd:sky blue", s=S, alpha=.3)
        axes[1].scatter(ia.mean(), ii.mean(), facecolors="none", edgecolors='red', s=150)
        axes[1].scatter(ia.mean(), ii.mean(), c="green", marker="x", s=43)
        axes[1].scatter(aa.mean(), ai.mean(), facecolors="none", edgecolors="red", s=150)
        axes[1].scatter(aa.mean(), ai.mean(), c="blue", marker="x", s=43)
        axes[1].plot([0.2, 0.9], [0.2, 0.9])
        axes[1].set_aspect("equal")
        axes[1].legend()
        axes[1].set_xlabel("Nearest active similarity")
        axes[1].set_ylabel("Nearest inactive similarity")

        return [np.mean(x) for x in (aa, ai, ia, ii)]

    figs = []
    muv_result = []
    for target_uid in target_uids:
        muv_result.append(target_name(target_uid))
        d = mean_warszycki_logki(
            target_uid=target_uid,
            chembl_filename="chembl_24.db",
            threshold=2.,
            include_earliest_year="all_bioactivity_records",
            ic50_conversion_strategy="all_relations_half_ic50",
            fit_ic50=True,
        )["final"]
        value = d.data["value"]
        kd = Benchmarks2018StructuralSimilarity(source=d)
        kernel = kd.data["kernel"]
        bac_groups = BalancedAgglomerativeClustering(
            source=kd,
            kernel="kernel",
            n_groups=5,
        ).data["groups"]
        cv_groups = CrossValidation(
            source=d,
            n_groups=5,
            seed=43,
        ).data["groups"]
        spectral_groups = SpectralClustering(
            source=kd,  
            kernel="kernel",
            n_groups=5,
        ).data["groups"]
        scaffold_groups = MurckoScaffoldSplit(
            source=d,
            generic=True,
            isomeric=False,
        ).data["groups"]
        paper_groups = PaperSplit(source=d).data["groups"]
        year_groups = d.data["year"]
        fig = plt.figure(figsize=(8,24))
        fig.axes_counter = 0
        def _axes():
            axes = []
            for _ in range(2):
                fig.axes_counter += 1
                axes.append(fig.add_subplot(6,2,fig.axes_counter))
            return axes
        for groups, split_label in (
                (paper_groups, "paper split"),
                (bac_groups, "balanced agglomerative clustering"),
                (spectral_groups, "spectral clustering"),
                (cv_groups, "cross validation"),
                (scaffold_groups, "scaffold split"),
                ):
            aa, ai, ia, ii = plot(value, groups, kernel, _axes(), split_label)
            muv = aa - ai + ii - ia
            muv_result.append("{:.3f} - {:.3f} + {:.3f} - {:.3f} = {:.3f}".format(aa, ai, ii, ia, muv))
        aa, ai, ia, ii = plot(value, year_groups, kernel, _axes(), split_label="time split", time_split=True)
        muv = aa - ai + ii - ia
        muv_result.append("{:.3f} - {:.3f} + {:.3f} - {:.3f} = {:.3f}".format(aa, ai, ii, ia, muv))
        fig.tight_layout()
        figs.append(fig)

    return tuple(['\n'.join(muv_result)+'\n'] + figs)

def splits_analysis_3_columns(target_uids):
    def plot(value, groups, kernel, axes, split_label, time_split=False):
        dct = aaaiiaii(value, groups, kernel, time_split)
        for k in ["aa", "ai", "ia", "ii"]:
            axes[0].hist(
                dct[k][0], bins=43, label=k.upper(),
                density=True, histtype="step", linewidth=3,
            )
        axes[0].set_xlim((0.,1.))
        axes[0].set_xlabel("All pairs similarity")
        axes[0].set_ylabel(split_label + '\n')
        axes[0].legend()

        not_nan_mask = np.logical_not(np.logical_or(
            np.isnan(dct["nearest_a"]),
            np.isnan(dct["nearest_i"]),
        ))
        aa, ai, ia, ii = (
            dct["nearest_a"][not_nan_mask][value[not_nan_mask]==1],
            dct["nearest_i"][not_nan_mask][value[not_nan_mask]==1],
            dct["nearest_a"][not_nan_mask][value[not_nan_mask]==0],
            dct["nearest_i"][not_nan_mask][value[not_nan_mask]==0],
        )

        histtype, linewidth = "step", 3
        axes[1].hist(
            aa, bins=43, label="AA",
            density=True, histtype=histtype, linewidth=linewidth,
        )
        axes[1].hist(
            ai, bins=43, label="AI",
            density=True, histtype=histtype, linewidth=linewidth,
        )
        axes[1].hist(
            ia, bins=43, label="IA",
            density=True, histtype=histtype, linewidth=linewidth,
        )
        axes[1].hist(
            ii, bins=43, label="II",
            density=True, histtype=histtype, linewidth=linewidth,
        )
        axes[1].set_xlim((0.,1.))
        axes[1].set_xlabel("Nearest neighbour similarity")
        axes[1].legend()

        S = 8
        axes[2].scatter(ia, ii, label="inactive", c="green", s=S, alpha=.3)
        axes[2].scatter(aa, ai, label="active", c="xkcd:sky blue", s=S, alpha=.3)
        axes[2].scatter(ia.mean(), ii.mean(), facecolors="none", edgecolors='red', s=150)
        axes[2].scatter(ia.mean(), ii.mean(), c="green", marker="x", s=43)
        axes[2].scatter(aa.mean(), ai.mean(), facecolors="none", edgecolors="red", s=150)
        axes[2].scatter(aa.mean(), ai.mean(), c="blue", marker="x", s=43)
        axes[2].plot([0.2, 0.9], [0.2, 0.9])
        axes[2].set_aspect("equal")
        axes[2].legend()
        axes[2].set_xlabel("Nearest active similarity")
        axes[2].set_ylabel("Nearest inactive similarity")

        return [np.mean(x) for x in (aa, ai, ia, ii)]

    figs = []
    muv_result = []
    for target_uid in target_uids:
        muv_result.append(target_name(target_uid))
        d = mean_warszycki_logki(
            target_uid=target_uid,
            chembl_filename="chembl_24.db",
            threshold=2.,
            include_earliest_year="all_bioactivity_records",
            ic50_conversion_strategy="all_relations_half_ic50",
            fit_ic50=True,
        )["final"]
        value = d.data["value"]
        kd = Benchmarks2018StructuralSimilarity(source=d)
        kernel = kd.data["kernel"]
        bac_groups = BalancedAgglomerativeClustering(
            source=kd,
            kernel="kernel",
            n_groups=5,
        ).data["groups"]
        cv_groups = CrossValidation(
            source=d,
            n_groups=5,
            seed=43,
        ).data["groups"]
        spectral_groups = SpectralClustering(
            source=kd,  
            kernel="kernel",
            n_groups=5,
        ).data["groups"]
        scaffold_groups = MurckoScaffoldSplit(
            source=d,
            generic=True,
            isomeric=False,
        ).data["groups"]
        paper_groups = PaperSplit(source=d).data["groups"]
        year_groups = d.data["year"]
        fig = plt.figure(figsize=(12,24))
        fig.axes_counter = 0
        def _axes():
            axes = []
            for _ in range(3):
                fig.axes_counter += 1
                axes.append(fig.add_subplot(6,3,fig.axes_counter))
            return axes
        for groups, split_label in (
                (paper_groups, "paper split"),
                (bac_groups, "balanced agglomerative clustering"),
                (spectral_groups, "spectral clustering"),
                (cv_groups, "cross validation"),
                (scaffold_groups, "scaffold split"),
                ):
            aa, ai, ia, ii = plot(value, groups, kernel, _axes(), split_label)
            muv = aa - ai + ii - ia
            muv_result.append("{:.3f} - {:.3f} + {:.3f} - {:.3f} = {:.3f}".format(aa, ai, ii, ia, muv))
        aa, ai, ia, ii = plot(value, year_groups, kernel, _axes(), split_label="time split", time_split=True)
        muv = aa - ai + ii - ia
        muv_result.append("{:.3f} - {:.3f} + {:.3f} - {:.3f} = {:.3f}".format(aa, ai, ii, ia, muv))
        fig.tight_layout()
        figs.append(fig)

    return tuple(['\n'.join(muv_result)+'\n'] + figs)

def splits_analysis_2(target_uids):
    fig = plt.figure(figsize=(4*(len(target_uids)+1),4))
    for i, target_uid in enumerate(target_uids):
        d = mean_warszycki_logki(
            target_uid=target_uid,
            chembl_filename="chembl_24.db",
            threshold=2.,
            include_earliest_year="all_bioactivity_records",
            ic50_conversion_strategy="all_relations_half_ic50",
            fit_ic50=True,
        )["final"]
        value = d.data["value"]
        kd = Benchmarks2018StructuralSimilarity(source=d)
        kernel = kd.data["kernel"]
        bac_groups = BalancedAgglomerativeClustering(
            source=kd,
            kernel="kernel",
            n_groups=5,
        ).data["groups"]
        cv_groups = CrossValidation(
            source=d,
            n_groups=5,
            seed=43,
        ).data["groups"]
        spectral_groups = SpectralClustering(
            source=kd,  
            kernel="kernel",
            n_groups=5,
        ).data["groups"]
        scaffold_groups = MurckoScaffoldSplit(
            source=d,
            generic=True,
            isomeric=False,
        ).data["groups"]
        paper_groups = PaperSplit(source=d).data["groups"]
        year_groups = d.data["year"]
        ax = fig.add_subplot(1,len(target_uids),i+1)
        for groups, label in (
                (paper_groups, "paper split"),
                (bac_groups, "balanced agglomerative clustering"),
                (spectral_groups, "spectral clustering"),
                (cv_groups, "cross validation"),
                (scaffold_groups, "scaffold split"),
                ):
            dct = aaaiiaii(value, groups, kernel, time_split=False)
            x = np.maximum(dct["nearest_a"], dct["nearest_i"])
            ax.hist(
                x, bins=43, label=label,
                density=True, histtype="step", linewidth=1,
            )
        dct = aaaiiaii(value, year_groups, kernel, time_split=True)
        x = np.maximum(dct["nearest_a"], dct["nearest_i"])
        x = x[np.logical_not(np.isnan(x))]
        ax.hist(
            x, bins=43, label="time split",
            density=True, histtype="step", linewidth=3,
        )
        ax.set_xlabel("Nearest neighbour similarity")
        ax.set_title(target_name(target_uid))
        if i == len(target_uids) - 1:
            ax.legend(fontsize="small", bbox_to_anchor=(1.04,1))
    fig.tight_layout()
    return fig

def simplest_dataset_hist(mus):
    fig = plt.figure(figsize=(4*len(mus), 8))
    alpha = .6
    for i, mu in enumerate(mus):

        ax = fig.add_subplot(2,len(mus),i+1)
        xs = np.linspace(-4.3,4.3,437)
        ax.fill_between(
            xs, norm(loc=mu).pdf(xs),
            label='"inactive"', alpha=alpha,
        )
        ax.fill_between(
            xs, norm(loc=-mu).pdf(xs),
            label='"active"', alpha=alpha,
        )
        ax.set_xlabel("mean: {:.1f}".format(mu))
        ax.set_ylim((0.,0.6))
        ax.legend()
        if i == 0:
            ax.set_ylabel("Normal\n")

        ax = fig.add_subplot(2,len(mus),len(mus)+i+1)
        xs = np.linspace(-2.1,2.1,437)
        ax.fill_between(
            xs, uniform(loc=mu-1, scale=2.).pdf(xs),
            label='"inactive"', alpha=alpha,
        )
        ax.fill_between(
            xs, uniform(loc=-mu-1, scale=2.).pdf(xs),
            label='"active"', alpha=alpha,
        )
        ax.set_xlabel("mean: {:.1f}".format(mu))
        ax.set_ylim((0.,0.7))
        ax.legend()
        if i == 0:
            ax.set_ylabel("Uniform\n")

    fig.tight_layout()
    return fig

def muv_on_simplest_dataset(mus, ns):
    def dataset(mu, n_train, n_test, distr, seed=43):
        if isinstance(n_train, int):
            rng = np.random.RandomState(seed=43)
            if distr == "normal":
                distr = rng.normal
                acc = norm.cdf(mu)
            elif distr == "uniform":
                distr = lambda size: rng.uniform(size=size) * 2. - 1.
                acc = min(1., .5 + .5*abs(mu))
            else:
                raise ValueError(distr)
            tr0 = distr(size=n_train) + mu
            tr1 = distr(size=n_train) - mu
            te0 = distr(size=n_test) + mu
            te1 = distr(size=n_test) - mu
            def _dist(x,y):
                return np.abs(x.reshape(-1,1)-y.reshape(1,-1))
            aa = np.min(_dist(te1, tr1), axis=1).mean()
            ai = np.min(_dist(te1, tr0), axis=1).mean()
            ia = np.min(_dist(te0, tr1), axis=1).mean()
            ii = np.min(_dist(te0, tr0), axis=1).mean()
            return {
                "acc": acc,
                "muv": aa - ai,
                "atomwise": aa - ai + ii - ia,
            }
        elif n_train == "infty" and n_test == "infty":
            if distr == "normal":
                aa, ai, ia, ii = 0., 0., 0., 0.
                return {
                    "acc": norm.cdf(mu),
                    "muv": aa - ai,
                    "atomwise": aa - ai + ii - ia,
                }
            elif distr == "uniform":
                aa, ai, ia, ii = 0., min(mu**2,1.), min(mu**2,1.), 0.
                return {
                    "acc": min(1., .5 + .5*abs(mu)),
                    "muv": aa - ai,
                    "atomwise": aa - ai + ii - ia,
                }
            else:
                raise ValueError()
        else:
            raise ValueError()
    result_normal = np.zeros((3, len(ns), len(mus)), dtype=np.float)
    result_uniform = np.zeros((3, len(ns), len(mus)), dtype=np.float)
    for i, n in enumerate(ns):
        for j, mu in enumerate(mus):
            d = dataset(mu, n, n, "normal")
            result_normal[0,i,j] = d["acc"]
            result_normal[1,i,j] = d["muv"]
            result_normal[2,i,j] = d["atomwise"]

            d = dataset(mu, n, n, "uniform")
            result_uniform[0,i,j] = d["acc"]
            result_uniform[1,i,j] = d["muv"]
            result_uniform[2,i,j] = d["atomwise"]
    fig = plt.figure(figsize=(24,6))
    axes = [fig.add_subplot(1,4,i+1) for i in range(4)]
    alpha = .6
    s = 43
    for i, n in enumerate(ns):
        label = "4 x {}".format(n) if n != "infty" else '∞'

        ax = axes[0]
        ax.scatter(
            result_normal[1,i,:], result_normal[0,i,:], label=label, alpha=alpha, s=s)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Bias measure (MUV part)")
        ax.set_title("Normal")

        ax = axes[1]
        ax.scatter(
            result_normal[2,i,:], result_normal[0,i,:], label=label, alpha=alpha, s=s)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Bias measure")
        ax.set_title("Normal")

        ax = axes[2]
        ax.scatter(
            result_uniform[1,i,:], result_uniform[0,i,:], label=label, alpha=alpha, s=s)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Bias measure (MUV part)")
        ax.set_title("Uniform")

        ax = axes[3]
        ax.scatter(
            result_uniform[2,i,:], result_uniform[0,i,:], label=label, alpha=alpha, s=s)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Bias measure")
        ax.set_title("Uniform")

    [ax.legend(loc="lower left", title="Benchmark size", fontsize="small") for ax in axes]
    fig.tight_layout()
    return fig

def splits_tsne(target_uids):
    S = 8
    fig = plt.figure(figsize=(30,4*len(target_uids)))
    counter = 0
    for target_uid in target_uids:
        d = mean_warszycki_logki(
            target_uid=target_uid,
            chembl_filename="chembl_24.db",
            threshold=None,
            include_earliest_year="all_bioactivity_records",
            ic50_conversion_strategy="all_relations_half_ic50",
            fit_ic50=True,
        )["final"]
        kd = Benchmarks2018StructuralSimilarity(source=d)
        tsne = KernelTSNE(
            source=kd,
            kernel="kernel",
            n_components=2,
            perplexity=43.,
            early_exaggeration=43.,
            learning_rate=4343.,
        ).data["tsne"]
        bac_groups = BalancedAgglomerativeClustering(
            source=kd,
            kernel="kernel",
            n_groups=5,
        ).data["groups"]
        cv_groups = CrossValidation(
            source=d,
            n_groups=5,
            seed=43,
        ).data["groups"]
        spectral_groups = SpectralClustering(
            source=kd,
            kernel="kernel",
            n_groups=5,
        ).data["groups"]
        scaffold_groups = MurckoScaffoldSplit(
            source=d,
            generic=True,
            isomeric=False,
        ).data["groups"]
        paper_groups = PaperSplit(source=d).data["groups"]

        for c, split_label in (
                (paper_groups, "paper split"),
                (bac_groups, "balanced agglomerative clustering"),
                (spectral_groups, "spectral clustering"),
                (cv_groups, "cross validation"),
                (scaffold_groups, "scaffold split"),
                (d.data["year"], "time split")):
            counter += 1
            ax = fig.add_subplot(len(target_uids),6,counter)
            a = ax.scatter(tsne.T[0], tsne.T[1], s=S, c=c)
            ax.set_xlim((-105,105))
            ax.set_ylim((-105,105))
            ax.set_aspect("equal")
            ax.set_xlabel(split_label)
            if split_label == "paper split":
                ax.set_ylabel(target_name(target_uid) + '\n')
            bar = fig.colorbar(a)
            bar.locator = MaxNLocator(integer=True)
            bar.update_ticks()
    fig.tight_layout()
    return fig

def noise_analysis(
        target_uids,
        delta_measurement_threshold,
        delta_measurement_upper_threshold,
        ic50_conversion_strategy,
        fit_ic50):
    fig = plt.figure(figsize=(16,len(target_uids)*4))
    N_PLOTS = 4
    counter = 0
    t1 = []
    t2 = []
    for target_uid in target_uids:
        _d = mean_warszycki_logki(
            target_uid=target_uid,
            chembl_filename="chembl_24.db",
            threshold=None,
            include_earliest_year=None,
            ic50_conversion_strategy=ic50_conversion_strategy,
            fit_ic50=fit_ic50,
        )
        all_values = _d["data_nodes"][-2] # threshold: None -> -2, not None -> -3
        mean_values = _d["final"]

        count_uid = Counter(all_values.data["uid"])
        how_many_samples = np.vectorize(lambda uid: count_uid[uid])(all_values.data["uid"])

        uid_to_mean_value = dict(zip(mean_values.data["uid"], mean_values.data["value"]))
        a = np.vectorize(lambda uid: uid_to_mean_value[uid])(all_values.data["uid"])
        b = all_values.data["value"]

        two_measurements_same_paper = []
        two_measurements_different_paper = []
        key = lambda x: x[0]
        for k, g in groupby(sorted(zip(all_values.data["uid"], all_values.data["smiles"], all_values.data["value"], all_values.data["doc_uid"]), key=key), key):
            gu, gs, gv, gdu = zip(*g)
            if len(gu) == 2:
                if np.abs(gv[0]-gv[1]) > delta_measurement_threshold:
                    if np.abs(gv[0]-gv[1]) <= delta_measurement_upper_threshold:
                        if gdu[0] == gdu[1]:
                            two_measurements_same_paper.append(gv)
                        else:
                            two_measurements_different_paper.append(gv)
                    else:
                        t1.append("TARGET: {}, UID: {}, SMILES: {}, DOC1: {}, VALUE1: {}, DOC2: {}, VALUE2: {}".format(
                            target_name(target_uid),
                            gu[0],
                            gs[0],
                            gdu[0],
                            gv[0],
                            gdu[1],
                            gv[1],
                        ))

        _a = np.array(two_measurements_same_paper)
        _b = np.array(two_measurements_different_paper)
        _a = np.abs(_a[:,0]-_a[:,1])
        _b = np.abs(_b[:,0]-_b[:,1])

        counter += 1
        ax = fig.add_subplot(len(target_uids),N_PLOTS,counter)
        ax.hist(_a, bins=43)
        ax.set_xlabel("pKi abs. difference, same paper")
        ax.set_ylabel(target_name(target_uid) + "\n")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        counter += 1
        ax = fig.add_subplot(len(target_uids),N_PLOTS,counter)
        ax.hist(_b, bins=43)
        ax.set_xlabel("pKi abs. difference, two papers")
        t2.append("TARGET UID: {}, SAME: {:.3f} [{} SAMPLES], DIFFERENT: {:.3f} [{} SAMPLES]".format(
            target_name(target_uid),
            np.mean(np.square(_a))/2,
            len(_a),
            np.mean(np.square(_b))/2,
            len(_b),
        ))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        result = []
        for j in range(1,np.max(list(count_uid.values()))):
            mask = how_many_samples > j
            result.append(np.square(b[mask]-a[mask]).mean())

        counter += 1
        ax = fig.add_subplot(len(target_uids),N_PLOTS,counter)
        ax.scatter(to_pki(a), to_pki(b), s=8)
        ax.set_xlabel("Mean pKi")
        ax.set_ylabel("Reported pKi")

        counter += 1
        count_count_uid = Counter(count_uid.values())
        x = np.array([
            count_count_uid[1],
            count_count_uid[2],
            len(mean_values.data["uid"])-count_count_uid[1]-count_count_uid[2],
        ])
        assert sum(x) == len(mean_values.data["uid"])
        ax = fig.add_subplot(len(target_uids),N_PLOTS,counter)
        ax.bar(x=[0,1,2], height=x)
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(["1", "2", ">2"])
        ax.set_xlabel("Records per SMILES")
        ax.set_yscale("log", nonposy='clip')

        for rect, label in zip(ax.patches, x):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 5,
                label,
                ha='center',
                va='bottom',
                bbox=dict(
                    boxstyle="square",
                    ec=(1., 0.5, 0.5),
                    fc=(1., 0.8, 0.8),
                ),
            )

    fig.tight_layout()
    return fig, '\n'.join(t1)+'\n', '\n'.join(t2)+'\n'

def how_many_active_inactive(target_uids, conversion_strategies, threshold):
    result = np.empty((len(target_uids), len(conversion_strategies)), dtype=np.object)
    result.fill("")
    for i, target_uid in enumerate(target_uids):
        for j, (ic50_conversion_strategy, fit_ic50, _) in enumerate(conversion_strategies):
            dct = mean_warszycki_logki(
                target_uid=target_uid,
                chembl_filename="chembl_24.db",
                threshold=threshold,
                include_earliest_year=None,
                ic50_conversion_strategy=ic50_conversion_strategy,
                fit_ic50=fit_ic50,
            )
            result[i,j] = "a:{} ia:{} p:{}".format(
                (dct["final"].data["value"] == 1.).sum(),
                (dct["final"].data["value"] == 0.).sum(),
                len(set(dct["data_nodes"][-3].data["doc_uid"])),
            )
    rows = [target_name(u) for u in target_uids]
    cols = list(list(zip(*conversion_strategies))[2])
    return _table(rows, cols, result, '\t')

def ic50_delta(target_uids, conversion_strategies):
    result = np.empty((len(target_uids), len(conversion_strategies)), dtype=np.object)
    result.fill("")
    for i, target_uid in enumerate(target_uids):
        for j, (ic50_conversion_strategy, name) in enumerate(conversion_strategies):
            n = mean_warszycki_logki(
                target_uid=target_uid,
                chembl_filename="chembl_24.db",
                threshold=None,
                include_earliest_year=None,
                ic50_conversion_strategy=ic50_conversion_strategy,
                fit_ic50=True,
            )["data_nodes"][-2]
            assert n.__class__.__name__ == "FitOriginalIC50ToKi"
            result[i,j] = "{:.3f} / {}".format(
                2*10**(-n.data["IC50_correction"]),
                n.data["how_many_uids_to_estimate_correction"],
            )
    rows = [target_name(u) for u in target_uids]
    cols = list(list(zip(*conversion_strategies))[1])
    return _table(rows, cols, result, '\t')
