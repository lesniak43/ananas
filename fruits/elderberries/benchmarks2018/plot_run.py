#!/usr/bin/env python3

import itertools
import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from crisper import safe_path_join
from herbivores._html import (
    columns_width,
    to_html,
    sanitize_html,
    doc_template,
    style_template,
    table_style_1,
    tablesorter,
)
from elderberries.benchmarks2018.run import SUMMARIES
from elderberries.benchmarks2018.solutions import (
    FLAT_FINGERPRINTERS,
    fingerprinter_by_name,
)


def by_fingerprint(solutions, reference_solutions):
    _done = np.zeros(len(solutions), dtype=np.bool)
    labels = np.empty(len(solutions), dtype=np.object)
    markers = np.empty(len(solutions), dtype=np.object)
    colors = np.empty(len(solutions), dtype=np.object)
    labels.fill("constant")
    markers.fill("x")
    colors.fill("black")
    filled_markers = (u'o', u'v', u'^', u'<', u'>', u'8', u's', u'p', u'*', u'h', u'H', u'D', u'd', ',', '+', '.')
    marker = itertools.cycle(filled_markers)
    color = itertools.cycle([u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'])
    for n in sorted([k for k in fingerprinter_by_name.keys() if "hashed" not in k]):
        mask = np.array([n+"_" in _n or _n.endswith(n) for _n in solutions])
        labels[mask] = n
        markers[mask] = next(marker)
        colors[mask] = next(color)
        assert np.all(np.logical_not(_done[mask]))
        _done[mask] = True
    for i in np.where(np.logical_not(_done))[0]:
        assert solutions[i] in reference_solutions
    return labels, markers, colors

def _iter(labels, markers, colors):
    for l in sorted(set(labels)):
        mask = labels == l
        yield mask, l, markers[mask][0], colors[mask][0]

def plot(fname,s,x,y,xlim,ylim,title,xlabel,ylabel,reference_solutions,best_only=False,reversed_metric=True,color_flat=False,legend=True):
    assert not os.path.exists(fname)
    fig = plt.figure(figsize=(8,8))
    for mask, l, m, c in _iter(*by_fingerprint(s, reference_solutions)):
        if best_only:
            if reversed_metric:
                idx = [np.argmin(x[mask]), np.argmin(y[mask])]
            else:
                idx = [np.argmax(x[mask]), np.argmax(y[mask])]
            _x, _y = x[mask][idx], y[mask][idx]
        else:
            _x, _y = x[mask], y[mask]
        if color_flat:
            c = '#1f77b4' if l in FLAT_FINGERPRINTERS else '#ff7f0e'
            if l == "constant":
                c = "black"
        plt.scatter(_x, _y, label=l, marker=m, color=c, alpha=.8)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def plot_all(summary_name, summary, reference_solutions):

    output_dir = safe_path_join(
        "elderberries-Benchmarks2018",
        "scores",
        summary_name,
        "figures",
        dirname=os.getenv("ANANAS_RESULTS_PATH"),
    )
    os.makedirs(output_dir)
    
    metric_pairs = [(metric, metric, reversed_metric) for metric, (_, reversed_metric) in summary.metrics.items()]
    for metric in summary.closer_further_metrics:
        metric_pairs.append((metric, metric+"_Closer", summary.metrics[metric][1]))
        metric_pairs.append((metric, metric+"_Further", summary.metrics[metric][1]))
    
    for cv_metric, bac_metric, reversed_metric in metric_pairs:

        metric_plot_name = "CV_{}_BAC_{}".format(cv_metric, bac_metric)
        header, cv = summary.results(cv_metric, "cv")
        header, bac = summary.results(bac_metric, "bac")
        assert np.all(cv[0] == bac[0])
        solutions = cv[0]

        _ref_mask = np.vectorize(lambda x: x in reference_solutions)(solutions)
        _cv = np.concatenate([cv[i].reshape(-1,1) for i in range(1, len(header)-1)], axis=1)
        _bac = np.concatenate([bac[i].reshape(-1,1) for i in range(1, len(header)-1)], axis=1)

        if not reversed_metric:
            _cv, _bac = -_cv, -_bac
        xlim = np.array((_cv.min(), min(_cv[_ref_mask].max()*2-_cv.min(), _cv.max())))
        ylim = np.array((_bac.min(), min(_bac[_ref_mask].max()*2-_bac.min(), _bac.max())))
        if not reversed_metric:
            xlim, ylim = -xlim[::-1], -ylim[::-1]
        xlim += .05*np.array((xlim[0]-xlim[1], xlim[1]-xlim[0]))
        ylim += .05*np.array((ylim[0]-ylim[1], ylim[1]-ylim[0]))
        xlabel = "cross validation score"
        ylabel = "balanced agglomerative clustering score"
        for i in range(1,13):
            if i == len(header)-1: # ranking
                xlim = ylim = (-3,len(solutions)+3)
                reversed_metric=True
            plot(safe_path_join(metric_plot_name+'_'+header[i]+".svg", dirname=output_dir),
                solutions, cv[i], bac[i], title=header[i], reference_solutions=reference_solutions, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, best_only=False, color_flat=False, legend=True, reversed_metric=reversed_metric)
            plot(safe_path_join(metric_plot_name+'_'+header[i]+"_.svg", dirname=output_dir),
                solutions, cv[i], bac[i], title=header[i], reference_solutions=reference_solutions, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, best_only=False, color_flat=True, legend=True, reversed_metric=reversed_metric)
            plot(safe_path_join(metric_plot_name+'_'+header[i]+"__.svg", dirname=output_dir),
                solutions, cv[i], bac[i], title=header[i], reference_solutions=reference_solutions, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, best_only=True, color_flat=False, legend=False, reversed_metric=reversed_metric)
            plot(safe_path_join(metric_plot_name+'_'+header[i]+"___.svg", dirname=output_dir),
                solutions, cv[i], bac[i], title=header[i], reference_solutions=reference_solutions, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, best_only=True, color_flat=True, legend=False, reversed_metric=reversed_metric)


if __name__ == "__main__":

    for summary_name, summary, reference_solutions in SUMMARIES:

        output = safe_path_join(
            "elderberries-Benchmarks2018",
            "scores",
            summary_name,
            dirname=os.getenv("ANANAS_RESULTS_PATH"),
        )
        try:
            os.makedirs(output)
            plot_all(summary_name, summary, reference_solutions)
            output2 = safe_path_join("tables", dirname=output)
            os.makedirs(output2)
            for score in summary.metrics:
                for split in ["bac", "cv"]:
                    fname = score + '-' + split + ".html"
                    header, columns = summary.results(score, split)
                    arr = np.concatenate([c.astype(np.str).reshape(-1,1) for c in columns], axis=1)
                    width = columns_width(arr, header, 30)
                    arr, header = sanitize_html(arr), sanitize_html(header)
                    with open(safe_path_join(fname, dirname=output2), 'x') as f_out:
                        f_out.write(doc_template(
                            style_template(
                                table_style_1("data_table"),
                            ) + '\n' + tablesorter(),
                            to_html(arr, header, width, "data_table"),
                        ))
        except FileExistsError:
            pass
