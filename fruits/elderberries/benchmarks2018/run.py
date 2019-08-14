#!/usr/bin/env python3

import os

import numpy as np

#import mandalka; mandalka.config(lazy=False)
import crisper
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
from elderberries.benchmarks2018.problem import Benchmarks2018Problem
from elderberries.benchmarks2018.solutions import (
    SOLUTIONS_R,
    SOLUTIONS_C,
)

SUMMARIES = [
    (
        "full_regression",
        Benchmarks2018Problem(
            threshold=None,
            ic50_conversion_strategy="all_relations_half_ic50",
            fit_ic50=False,
        ).benchmark(SOLUTIONS_R),
        ["PredictMean", "PredictMedian"],
    ),
    (
        "full_classification_logKi2",
        Benchmarks2018Problem(
            threshold=2.,
            ic50_conversion_strategy="all_relations_half_ic50",
            fit_ic50=False,
        ).benchmark(SOLUTIONS_C),
        ["PredictConstant"],
    ),
    (
        "full_classification_median",
        Benchmarks2018Problem(
            threshold="median",
            ic50_conversion_strategy="all_relations_half_ic50",
            fit_ic50=False,
        ).benchmark(SOLUTIONS_C),
        ["PredictConstant"],
    ),
    (
        "minimal_regression",
        Benchmarks2018Problem(
            threshold=None,
            ic50_conversion_strategy="only_equal_only_Ki",
            fit_ic50=False,
        ).benchmark(SOLUTIONS_R),
        ["PredictMean", "PredictMedian"],
    ),
    (
        "minimal_classification_logKi2",
        Benchmarks2018Problem(
            threshold=2.,
            ic50_conversion_strategy="only_equal_only_Ki",
            fit_ic50=False,
        ).benchmark(SOLUTIONS_C),
        ["PredictConstant"],
    ),
    (
        "minimal_classification_median",
        Benchmarks2018Problem(
            threshold="median",
            ic50_conversion_strategy="only_equal_only_Ki",
            fit_ic50=False,
        ).benchmark(SOLUTIONS_C),
        ["PredictConstant"],
    ),
]

if __name__ == "__main__":
    [crisper.evaluate(s[1], label=s[0]) for s in SUMMARIES]
