"""Experiment C10: how much do encoding and scorer change the existence baseline?

Runs :class:`existence_importance.ExistenceImportance` on BPIC11 f1, f2, f3 with
the original top-10 Apriori activity sets (min_support 0.5, size > 1) for every
combination of encoding {count, binary} x scoring {accuracy, f1_weighted} x
scored part {train, test}, plus

* a "faithful" configuration that mimics ``Classical_Permutation.py``
  (unfiltered cases, substring count, accuracy, training fold),
* a fold-seed stability check of the four combinations, and
* a count of how often substring matching differs from exact matching.

All outputs go to ``results/experiments/C10/`` (CSV files, plots, ``tables.md``).

Usage (from anywhere)::

    <project>/.venv/Scripts/python.exe <project>/experiments/run_c10.py
"""
from __future__ import annotations

import itertools
import sys
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from existence_importance import (  # noqa: E402
    ENCODINGS, PROJECT_ROOT, SCORE_ON, SCORINGS, ExistenceImportance, apriori_itemsets,
    encode_existence, itemset_label, load_log)

OUT_DIR = PROJECT_ROOT / "results" / "experiments" / "C10"
DATASET_NAMES = ("f1", "f2", "f3")
MIN_SUPPORT, TOP_K = 0.5, 10
DEFAULT = {"encoding": "count", "scoring": "f1_weighted", "score_on": "test"}
STABILITY_SEEDS = (2023, 0, 1, 2, 3)
COMBINATIONS = tuple(itertools.product(ENCODINGS, SCORINGS))


def log_line(message: str) -> None:
    """Print with a wall-clock prefix (stdout is redirected to run.log)."""
    print("[%s] %s" % (time.strftime("%H:%M:%S"), message), flush=True)


def to_markdown(frame: pd.DataFrame, float_format: str = "%.4f", index: bool = True) -> str:
    """Minimal DataFrame -> GitHub markdown table (no extra dependency)."""
    if index:
        frame = frame.reset_index()

    def cell(value):
        if isinstance(value, (float, np.floating)):
            return "nan" if np.isnan(value) else float_format % value
        return str(value)

    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(cell(value) for value in row) + " |"
            for row in frame.itertuples(index=False)]
    return "\n".join([header, rule] + body)


def combo_name(encoding: str, scoring: str) -> str:
    """Short column name for one encoding/scoring combination."""
    return "%s/%s" % (encoding, "acc" if scoring == "accuracy" else "f1w")


def spearman(first, second) -> float:
    """Spearman rho (average ranks for ties; nan if one input is constant)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(spearmanr(first, second).statistic)


def spearman_table(means: pd.DataFrame) -> pd.DataFrame:
    """Spearman rho between all pairs of columns of ``means``."""
    columns = list(means.columns)
    rho = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns)
    for first, second in itertools.combinations(columns, 2):
        rho.loc[first, second] = rho.loc[second, first] = spearman(means[first], means[second])
    return rho.rename_axis("rho")


# --------------------------------------------------------------------------- #
# Diagnostics of the encoding
# --------------------------------------------------------------------------- #
def substring_report(logs: dict, itemsets) -> pd.DataFrame:
    """Original substring count vs exact token count on the itemset features.

    One row per log version; a "cell" is one (case, activity set) feature value.
    """
    rows = {}
    for name, log in logs.items():
        exact = encode_existence(log.traces, itemsets, "count", "exact")
        substring = encode_existence(log.traces, itemsets, "count", "substring")
        differs = exact != substring
        rows[name] = {
            "cases": len(exact),
            "cells": exact.size,
            "cells differ": int(differs.values.sum()),
            "cases differ": int(differs.any(axis=1).sum()),
            "0/1 presence differs (cells)": int(((exact > 0) != (substring > 0)).values.sum()),
            "max count gap": int((substring - exact).values.max()),
            "sets affected": "; ".join("%s: %d cases" % (itemset_label(itemsets[i]), differs[i].sum())
                                       for i in differs.columns if differs[i].any()) or "-",
        }
    return pd.DataFrame(rows).T.rename_axis("log version")


def alphabet_substring_pairs(log) -> list:
    """All pairs (a, b) of distinct activity names where a is a substring of b."""
    alphabet = sorted({activity for trace in log.traces for activity in trace})
    return [(short, other) for short in alphabet for other in alphabet
            if short != other and short in other]


def feature_redundancy(log, itemsets) -> pd.DataFrame:
    """How similar the existence columns are to each other (PFI dilution)."""
    rows = {}
    for encoding in ENCODINGS:
        encoded = encode_existence(log.traces, itemsets, encoding, "exact")
        correlation = encoded.corr().values
        off_diagonal = correlation[np.triu_indices_from(correlation, k=1)]
        twins = [(itemset_label(itemsets[i]), itemset_label(itemsets[j]))
                 for i, j in itertools.combinations(range(len(itemsets)), 2)
                 if (encoded[i] == encoded[j]).all()]
        rows[encoding] = {
            "columns": encoded.shape[1],
            "distinct columns": int((~encoded.T.duplicated()).sum()),
            "min pairwise r": off_diagonal.min(),
            "median pairwise r": float(np.median(off_diagonal)),
            "identical column pairs": "; ".join("%s = %s" % pair for pair in twins) or "-",
        }
    return pd.DataFrame(rows).T.rename_axis("encoding")


# --------------------------------------------------------------------------- #
# Importance runs
# --------------------------------------------------------------------------- #
def run_grid(log, itemsets) -> dict:
    """All encoding x scoring x score_on combinations (exact matching, same folds)."""
    results = {}
    for (encoding, scoring), score_on in itertools.product(COMBINATIONS, SCORE_ON):
        start = time.perf_counter()
        estimator = ExistenceImportance(encoding=encoding, scoring=scoring, score_on=score_on)
        results[(encoding, scoring, score_on)] = estimator.run(log, itemsets)
        log_line("  %-6s %-11s %-5s done in %.1f s"
                 % (encoding, scoring, score_on, time.perf_counter() - start))
    return results


def mean_table(results: dict, score_on: str, itemsets, supports) -> pd.DataFrame:
    """Mean importance and rank per activity set for the four combinations."""
    table = pd.DataFrame({"support": supports}, index=[itemset_label(i) for i in itemsets])
    for encoding, scoring in COMBINATIONS:
        table[combo_name(encoding, scoring)] = results[(encoding, scoring, score_on)].mean_importance()
    for encoding, scoring in COMBINATIONS:
        table["rank " + combo_name(encoding, scoring)] = results[(encoding, scoring, score_on)].ranking()
    return table.rename_axis("activity set")


def model_quality(results: dict, labels: pd.Series) -> pd.DataFrame:
    """Mean fold scores of the existence-feature model, next to the majority-class baseline."""
    rows = {encoding + " encoding": results[(encoding, "accuracy", "test")].fold_scores.mean()
            for encoding in ENCODINGS}
    majority = np.full(len(labels), int(labels.mean() > 0.5))
    rows["always predict label %d" % majority[0]] = pd.Series({
        "test_accuracy": accuracy_score(labels, majority),
        "test_f1_weighted": f1_score(labels, majority, average="weighted"),
        "test_share_predicted_1": float(majority[0])})
    return pd.DataFrame(rows).T.rename_axis("model")


def seed_stability(log, itemsets) -> dict:
    """Mean importances (held-out fold) of every combination under several fold seeds.

    Returns ``{combination name: DataFrame(activity sets x seeds)}``.
    """
    means = {}
    for encoding, scoring in COMBINATIONS:
        means[combo_name(encoding, scoring)] = pd.DataFrame({
            "seed %d" % seed: ExistenceImportance(encoding, scoring, score_on="test", fold_seed=seed)
            .run(log, itemsets).mean_importance() for seed in STABILITY_SEEDS})
        log_line("  seed stability %s done" % combo_name(encoding, scoring))
    return means


def seed_summary(seed_means: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Between-seed rho per combination, seed-averaged means, rho between combinations."""
    between_seeds = {}
    for name, means in seed_means.items():
        rho = spearman_table(means).values[np.triu_indices(means.shape[1], k=1)]
        between_seeds[name] = {"min rho": np.nanmin(rho), "median rho": np.nanmedian(rho),
                               "max rho": np.nanmax(rho)}
    averaged = pd.DataFrame({name: means.mean(axis=1) for name, means in seed_means.items()})
    for name in seed_means:
        averaged["rank " + name] = averaged[name].rank(ascending=False, method="min").astype(int)
    return (pd.DataFrame(between_seeds).T.rename_axis("combination"),
            averaged.rename_axis("activity set"), spearman_table(averaged[list(seed_means)]))


def long_frame(dataset: str, results: dict) -> pd.DataFrame:
    """Raw importances of one dataset in long format (for the CSV)."""
    parts = []
    for (encoding, scoring, score_on), result in results.items():
        part = result.importances.stack().rename("importance").reset_index()
        part = part.rename(columns={part.columns[2]: "activity_set"})
        for position, (column, value) in enumerate([("dataset", dataset), ("encoding", encoding),
                                                     ("scoring", scoring), ("score_on", score_on)]):
            part.insert(position, column, value)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def plot_grid(dataset: str, results: dict, score_on: str, path: Path) -> None:
    """2 x 2 box plots (encoding x scoring) in the style of the original script."""
    figure, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
    for axis, (encoding, scoring) in zip(axes.ravel(), COMBINATIONS):
        frame = results[(encoding, scoring, score_on)].importances
        frame = frame[frame.mean().sort_values().index]
        axis.boxplot(frame.values, orientation="horizontal", whis=10, tick_labels=list(frame.columns))
        axis.axvline(x=0, color="k", linestyle="--")
        axis.set_title("%s encoding, scoring = %s" % (encoding, scoring))
        axis.set_xlabel("decrease in %s (%s fold)" % (scoring, score_on))
    figure.suptitle("Existence importance, BPIC11 %s, 5 folds x 20 repeats" % dataset)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_dataset(dataset: str, sections: list, raw_frames: list) -> None:
    """Run everything for one log and append markdown sections / raw frames."""
    log_line("=== %s ===" % dataset)
    log, data_manager = load_log(dataset, frq_threshold=2)
    unfiltered_log, _ = load_log(dataset, frq_threshold=1)
    start = time.perf_counter()
    itemsets, supports = apriori_itemsets(data_manager, MIN_SUPPORT, TOP_K)
    intro = ("%d cases after the rare-activity filter (%d unfiltered), %d activities, share of label 1 = %.3f, "
             "%d activity sets (Apriori took %.1f s)."
             % (len(log.traces), len(unfiltered_log.traces), data_manager.data[data_manager.activity].nunique(),
                log.labels.mean(), len(itemsets), time.perf_counter() - start))
    log_line(intro)
    sections.append("## %s\n\n%s" % (dataset, intro))

    logs = {"filtered (frq_threshold=2)": log, "unfiltered (frq_threshold=1)": unfiltered_log}
    pairs = alphabet_substring_pairs(unfiltered_log)
    sections.append("**Substring vs exact matching on the activity-set features:**\n\n%s\n\n"
                    "Activity names that are substrings of another activity name (whole unfiltered alphabet): %s"
                    % (to_markdown(substring_report(logs, itemsets)), pairs or "none"))
    sections.append("**Feature redundancy (exact matching, filtered log):**\n\n"
                    + to_markdown(feature_redundancy(log, itemsets), "%.3f"))

    results = run_grid(log, itemsets)
    raw_frames.append(long_frame(dataset, results))
    sections.append("**Model quality (mean over 5 folds; the activity-set columns are the only features):**\n\n"
                    + to_markdown(model_quality(results, log.labels)))
    for score_on in SCORE_ON:
        table = mean_table(results, score_on, itemsets, supports)
        table.to_csv(OUT_DIR / ("mean_importance_%s_%s.csv" % (dataset, score_on)))
        rho = spearman_table(table[[combo_name(e, s) for e, s in COMBINATIONS]])
        rho.to_csv(OUT_DIR / ("spearman_%s_%s.csv" % (dataset, score_on)))
        sections.append("**Mean importance and rank, scored on the %s fold (fold seed 2023):**\n\n%s\n\n"
                        "**Spearman rho between the four combinations (%s fold):**\n\n%s"
                        % (score_on, to_markdown(table), score_on, to_markdown(rho, "%.3f")))
        plot_grid(dataset, results, score_on, OUT_DIR / ("existence_%s_%s.png" % (dataset, score_on)))

    train_test = pd.DataFrame({
        combo_name(e, s): [spearman(results[(e, s, "train")].mean_importance(),
                                    results[(e, s, "test")].mean_importance())]
        for e, s in COMBINATIONS}, index=["rho(train, test)"])
    sections.append("**Spearman rho between train-fold and test-fold importances:**\n\n"
                    + to_markdown(train_test, "%.3f"))

    faithful = ExistenceImportance("count", "accuracy", "substring", "train").run(unfiltered_log, itemsets)
    fixed_default = results[(DEFAULT["encoding"], DEFAULT["scoring"], DEFAULT["score_on"])]
    comparison = pd.DataFrame({
        "faithful (substring count, acc, train, unfiltered)": faithful.mean_importance(),
        "rank faithful": faithful.ranking(),
        "default (exact count, f1w, test, filtered)": fixed_default.mean_importance(),
        "rank default": fixed_default.ranking()}).rename_axis("activity set")
    comparison.to_csv(OUT_DIR / ("faithful_vs_default_%s.csv" % dataset))
    sections.append("**Faithful configuration vs project default (Spearman rho = %.3f):**\n\n%s"
                    % (spearman(comparison.iloc[:, 0], comparison.iloc[:, 2]), to_markdown(comparison)))

    seed_means = seed_stability(log, itemsets)
    between_seeds, averaged, between_combos = seed_summary(seed_means)
    pd.concat(seed_means, axis=1).to_csv(OUT_DIR / ("seed_means_%s.csv" % dataset))
    sections.append(
        "**Fold-seed stability (test fold, seeds %s): pairwise Spearman rho between seeds, per combination:**\n\n%s\n\n"
        "**Mean importance averaged over the %d seeds, and rank:**\n\n%s\n\n"
        "**Spearman rho between the four combinations on the seed-averaged importances:**\n\n%s\n\n"
        "**Default combination (%s) per seed:**\n\n%s"
        % (list(STABILITY_SEEDS), to_markdown(between_seeds, "%.3f"), len(STABILITY_SEEDS),
           to_markdown(averaged), to_markdown(between_combos, "%.3f"),
           combo_name(DEFAULT["encoding"], DEFAULT["scoring"]),
           to_markdown(seed_means[combo_name(DEFAULT["encoding"], DEFAULT["scoring"])].rename_axis("activity set"))))


def main() -> None:
    """Run C10 on the three logs and write CSVs, plots and ``tables.md``."""
    warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    sections, raw_frames = ["# C10 generated tables (written by experiments/run_c10.py)"], []
    for dataset in DATASET_NAMES:
        run_dataset(dataset, sections, raw_frames)
    pd.concat(raw_frames, ignore_index=True).to_csv(OUT_DIR / "importances_long.csv", index=False)
    (OUT_DIR / "tables.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    log_line("TOTAL %.1f s" % (time.perf_counter() - start))


if __name__ == "__main__":
    main()
