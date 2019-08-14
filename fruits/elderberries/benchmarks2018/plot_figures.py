#!/usr/bin/env python3

import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from elderberries.benchmarks2018.figures import (
    jj_thresholded_ki,
    earliest_year_variants,
    activity_variants,
    median_thresholded_activity_variants,
    how_many_active_inactive,
    ic50_delta,
    fingercheats,
    fingercheats_thr,
    min_max_mean_per_paper,
    how_many_records_per_paper,
    density_bias,
    same_paper_cross_paper,
    year_structural_pareto,
    splits_analysis,
    splits_analysis_2,
    splits_tsne,
    simplest_dataset_hist,
    muv_on_simplest_dataset,
    noise_analysis,
    similar_compounds,
)

TARGET_UIDS = [
    "CHEMBL214", "CHEMBL224", "CHEMBL225", "CHEMBL3371", "CHEMBL3155",
    "CHEMBL226", "CHEMBL251", "CHEMBL217", "CHEMBL264", "CHEMBL216"
]

def draw(func, fname):
    print(fname)
    output_dir = os.path.join(
        os.getenv("ANANAS_RESULTS_PATH"),
        "elderberries-Benchmarks2018",
        "figures",
    )
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if isinstance(fname, str):
        fname = (fname,)
    fname = tuple([os.path.join(output_dir, _fname) if _fname is not None else None for _fname in fname])
    if not all([(fn is None or os.path.exists(fn)) for fn in fname]):
        result = func()
        if not isinstance(result, tuple):
            result = (result,)
        for _result, _fname in zip(result, fname):
            if _fname is not None:
                if isinstance(_result, matplotlib.figure.Figure):
                    _result.savefig(_fname)
                    plt.close(_result)
                elif isinstance(_result, str):
                    with open(_fname, 'w') as f_out:
                        f_out.write(_result)
                else:
                    raise TypeError(_result)

if __name__ == "__main__":

    # CHAPTER 2

    draw(
        lambda : jj_thresholded_ki(
            N=10,
            N_SPLITS=5,
            target_uid="CHEMBL214",
            split_name="cv",
            C=10.,
            class_weight="balanced",
            weighted_score=True,
        ),
        "CHAPTER_2_ki_hole_n=10_splits=5_CHEMBL214_cv_C=10_balanced_wacc.svg",
    )

    draw(
        lambda : jj_thresholded_ki(
            N=10,
            N_SPLITS=5,
            target_uid="CHEMBL214",
            split_name="bac",
            C=10.,
            class_weight="balanced",
            weighted_score=True,
        ),
        "CHAPTER_2_ki_hole_n=10_splits=5_CHEMBL214_bac_C=10_balanced_wacc.svg",
    )

    draw(
        lambda : jj_thresholded_ki(
            N=10,
            N_SPLITS=5,
            target_uid="CHEMBL251",
            split_name="bac",
            C=10.,
            class_weight="balanced",
            weighted_score=True,
        ),
        "CHAPTER_2_ki_hole_n=10_splits=5_CHEMBL251_bac_C=10_balanced_wacc.svg",
    )

    # CHAPTER 3

    draw(
        lambda : ic50_delta(
            target_uids=TARGET_UIDS,
            conversion_strategies=[
                ("only_equal_half_ic50", "only '='"),
                ("all_relations_half_ic50", "all relations"),
            ],
        ),
        (
            "CHAPTER_3_ic50_delta.txt",
        ),
    )

    draw(
        lambda : earliest_year_variants(target_uids=TARGET_UIDS),
        "CHAPTER_3_earliest_year_variants.txt",
    )

    draw(
        lambda : activity_variants(
            target_uids=TARGET_UIDS,
            conversion_strategies=[
                ("only_equal_only_Ki", False, "only '=',\nonly 'Ki'"),
                ("only_equal_half_ic50", False, "only '=',\n'Ki' and 'IC50/2'"),
                ("only_equal_half_ic50", True, "only '=',\n'Ki' and fit 'IC50'"),
                ("all_relations_only_Ki", False, "all relations,\nonly 'Ki'"),
                ("all_relations_half_ic50", False, "all relations,\n'Ki' and 'IC50/2'"),
                ("all_relations_half_ic50", True, "all relations,\n'Ki' and fit 'IC50'"),
            ],
            reference_idx=0,
        ),
        (
            "CHAPTER_3_activity_variants_compare_1.svg",
            None,
        ),
    )

    draw(
        lambda : activity_variants(
            target_uids=TARGET_UIDS,
            conversion_strategies=[
                ("only_equal_only_Ki", False, "only '=',\nonly 'Ki'"),
                ("only_equal_half_ic50", False, "only '=',\n'Ki' and 'IC50/2'"),
                ("only_equal_half_ic50", True, "only '=',\n'Ki' and fit 'IC50'"),
                ("all_relations_only_Ki", False, "all relations,\nonly 'Ki'"),
                ("all_relations_half_ic50", False, "all relations,\n'Ki' and 'IC50/2'"),
                ("all_relations_half_ic50", True, "all relations,\n'Ki' and fit 'IC50'"),
            ],
            reference_idx=4,
        ),
        (
            "CHAPTER_3_activity_variants_compare_2.svg",
            None,
        ),
    )

    draw(
        lambda : median_thresholded_activity_variants(
            target_uids=TARGET_UIDS,
            conversion_strategies=[
                ("only_equal_only_Ki", False, "only '=',\nonly 'Ki'"),
                ("only_equal_half_ic50", False, "only '=',\n'Ki' and 'IC50/2'"),
                ("only_equal_half_ic50", True, "only '=',\n'Ki' and fit 'IC50'"),
                ("all_relations_only_Ki", False, "all relations,\nonly 'Ki'"),
                ("all_relations_half_ic50", False, "all relations,\n'Ki' and 'IC50/2'"),
                ("all_relations_half_ic50", True, "all relations,\n'Ki' and fit 'IC50'"),
            ],
        ),
        (
            "CHAPTER_3_median_thresholded_activity_variants.svg",
            "CHAPTER_3_median_thresholded_activity_variants.txt",
        ),
    )

    draw(
        lambda : how_many_active_inactive(
            target_uids=TARGET_UIDS,
            conversion_strategies=[
                ("only_equal_only_Ki", False, "only '=',\nonly 'Ki'"),
                ("only_equal_half_ic50", False, "only '=',\n'Ki' and 'IC50/2'"),
                ("only_equal_half_ic50", True, "only '=',\n'Ki' and fit 'IC50'"),
                ("all_relations_only_Ki", False, "all relations,\nonly 'Ki'"),
                ("all_relations_half_ic50", False, "all relations,\n'Ki' and 'IC50/2'"),
                ("all_relations_half_ic50", True, "all relations,\n'Ki' and fit 'IC50'"),
            ],
            threshold=2.,
        ),
        (
            "CHAPTER_3_how_many_active_inactive_thr_2.txt",
        ),
    )

    # CHAPTER 4

    draw(
        lambda : fingercheats(
            target_uids=TARGET_UIDS,
            fpr_names=[
                "maccs", "klekotaroth", "molprint2d",
                "rdkit1-7", "rdkit3-5",
                "atompairs", "torsion", "estatecount",
                "morgan-2", "morgan-3", "morgan-4",
                "morgan-2-F", "morgan-3-F", "morgan-4-F",
                "ngramfingerprint1-7", "ngramfingerprint3-5",
            ],
            include_earliest_year=None,
            ic50_conversion_strategy="all_relations_half_ic50",
            fit_ic50=False,
        ),
        (
            "CHAPTER_4_fingerprints_cheat.svg",
            "CHAPTER_4_fingerprints_cheat.txt",
        ),
    )

    draw(
        lambda : fingercheats_thr(
            target_uids=TARGET_UIDS,
            fpr_names=[
                "maccs", "klekotaroth", "molprint2d",
                "rdkit1-7", "rdkit3-5",
                "atompairs", "torsion", "estatecount",
                "morgan-2", "morgan-3", "morgan-4",
                "morgan-2-F", "morgan-3-F", "morgan-4-F",
                "ngramfingerprint1-7", "ngramfingerprint3-5",
            ],
            threshold=2.,
            include_earliest_year=None,
            ic50_conversion_strategy="all_relations_half_ic50",
            fit_ic50=False,
        ),
        (
            "CHAPTER_4_fingerprints_cheat_thr_2.svg",
            "CHAPTER_4_fingerprints_cheat_thr_2_table.txt",
            "CHAPTER_4_fingerprints_cheat_thr_2_names.txt",
        ),
    )

    # CHAPTER 5

    draw(
        lambda : activity_variants(
            target_uids=TARGET_UIDS,
            conversion_strategies=[
                ("only_equal_only_Ki", False, "only '=',\nonly 'Ki'"),
                ("only_equal_half_ic50", False, "only '=',\n'Ki' and 'IC50/2'"),
                ("only_equal_half_ic50", True, "only '=',\n'Ki' and fit 'IC50'"),
                ("all_relations_only_Ki", False, "all relations,\nonly 'Ki'"),
                ("all_relations_half_ic50", False, "all relations,\n'Ki' and 'IC50/2'"),
                ("all_relations_half_ic50", True, "all relations,\n'Ki' and fit 'IC50'"),
            ],
            reference_idx=0,
        ),
        (
            None,
            "CHAPTER_5_activity_variants_hist.svg",
        ),
    )

    draw(
        lambda : min_max_mean_per_paper(
            target_uids=TARGET_UIDS,
            include_earliest_year="all_bioactivity_records",
            ic50_conversion_strategy="all_relations_half_ic50",
            fit_ic50=True,
            min_paper_size=1,
        ),
        "CHAPTER_5_min_max_mean_per_paper.svg",
    )

    draw(
        lambda : min_max_mean_per_paper(
            target_uids=TARGET_UIDS,
            include_earliest_year="all_bioactivity_records",
            ic50_conversion_strategy="all_relations_half_ic50",
            fit_ic50=True,
            min_paper_size=5,
        ),
        "CHAPTER_5_min_max_mean_per_paper_5_or_more.svg",
    )

    draw(
        lambda : how_many_records_per_paper(
            target_uids=TARGET_UIDS,
            include_earliest_year="all_bioactivity_records",
            ic50_conversion_strategy="all_relations_half_ic50",
            fit_ic50=False,
        ),
        "CHAPTER_5_how_many_records_per_earliest_paper.svg",
    )

    draw(
        lambda : same_paper_cross_paper(target_uids=TARGET_UIDS),
        "CHAPTER_5_same_paper_cross_paper.svg",
    )

    draw(
        lambda : year_structural_pareto(target_uids=TARGET_UIDS),
        "CHAPTER_5_year_structural_pareto.txt",
    )

    draw(
        lambda : noise_analysis(
            target_uids=TARGET_UIDS,
            delta_measurement_threshold=.01,
            delta_measurement_upper_threshold=2.,
            ic50_conversion_strategy="only_equal_only_Ki",
            fit_ic50=False,
        ),
        (
            "CHAPTER_5_noise_analysis.svg",
            "CHAPTER_5_noise_analysis_1.txt",
            "CHAPTER_5_noise_analysis_2.txt",
        ),
    )

    for target_uid in TARGET_UIDS:
        draw(
            lambda : similar_compounds(
                target_uid=target_uid,
                n_top=437,
                n_bottom=437,
                n_random=437,
                seed=43,
            ),
            (
                "CHAPTER_5_similar_compounds_{}.html".format(target_uid),
            ),
        )
    del target_uid

    # CHAPTER 6

    draw(
        lambda : splits_analysis(target_uids=TARGET_UIDS),
        tuple(
            ["CHAPTER_6_splits_analysis.txt"] + \
            ["CHAPTER_6_splits_analysis_{}.svg".format(target_uid) for target_uid in TARGET_UIDS]
        ),
    )

    draw(
        lambda : splits_analysis_2(target_uids=TARGET_UIDS),
        "CHAPTER_6_splits_analysis.svg",
    )

    draw(
        lambda : splits_tsne(target_uids=TARGET_UIDS),
        "CHAPTER_6_splits_tsne.svg",
    )

    # CHAPTER 7

    draw(
        lambda : density_bias(target_uids=TARGET_UIDS),
        "CHAPTER_7_density_bias.svg",
    )

    draw(
        lambda : simplest_dataset_hist(mus=np.linspace(0,1,11)),
        "CHAPTER_7_simplest_dataset_hist.svg",
    )

    draw(
        lambda : muv_on_simplest_dataset(
            mus=np.linspace(0,1,11),
            ns=(10, 30, 100, 300, 1000, 3000, "infty"),
        ),
        "CHAPTER_7_simplest_dataset_muv.svg",
    )
