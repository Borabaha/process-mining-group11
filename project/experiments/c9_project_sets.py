"""Experiment C9, add-on: seed stability for the project's own activity sets.

The main C9 analysis uses the 10 itemsets of the original code. The project
compares 10 sets per size (1, 2, 3) from Apriori with 10 per size from
IMPresseD. This script reads the per-seed files of a ``--itemsets project``
run of ``c9_seed_stability.py`` and reports, for groups of sets (a strategy,
a strategy and a size, all 60):

* how well two fold seeds agree on the ranking inside the group;
* how well two studies agree that each pool 5 (or 10) fold seeds;
* how many (set, fold) cells are not measurable on the scored fold;
* the mean importance of the group per fold seed, and whether the difference
  Apriori - IMPresseD keeps its sign across fold seeds.

Run:  python c9_project_sets.py --dataset f1 [--repeats 10]

Writes results/experiments/C9/tables_<dataset>_project_k5_groups.md and
summary_<dataset>_project_k5_groups.json.
"""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np

from c9_analysis import (
    TOP_K,
    Cube,
    between_seeds,
    build_cube,
    fmt,
    load_rows,
    md_table,
    noise_and_signal,
    pooled_seeds,
    seed_means,
)
from c9_seed_stability import RESULT_DIR

STRATEGIES = ("apriori", "impressed")
SIZES = (1, 2, 3)
SETTINGS = ("fixed_test", "fixed_train")
GROUP_SIZES = (5, 10)  # fold seeds pooled per study, where enough seeds exist


def sub_cube(cube: Cube, strategy: str | None, size: int | None) -> Cube:
    """The part of ``cube`` whose set ids match ``strategy`` and ``size``.

    Set ids look like ``'apriori|2|03'`` (strategy | size | rank).
    """
    keep = [index for index, itemset_id in enumerate(cube.itemset_ids)
            if (strategy is None or itemset_id.split("|")[0] == strategy)
            and (size is None or int(itemset_id.split("|")[1]) == size)]
    return Cube(cube.values[:, :, keep, :], cube.baseline, cube.seeds,
                [cube.itemset_ids[i] for i in keep], [cube.labels[i] for i in keep])


def groups() -> dict:
    """Group name -> (strategy or None, size or None)."""
    named = {"all 60 sets": (None, None)}
    for strategy in STRATEGIES:
        named[f"{strategy}, all sizes (30)"] = (strategy, None)
        for size in SIZES:
            named[f"{strategy}, size {size} (10)"] = (strategy, size)
    return named


def pooled_sizes(cube: Cube) -> list[int]:
    """The ``GROUP_SIZES`` for which two disjoint groups of seeds exist."""
    return [size for size in GROUP_SIZES if 2 * size <= len(cube.seeds)]


def stability_header(cube: Cube) -> list[str]:
    """Column names of the table built by ``group_stability``."""
    header = ["group", "two seeds: Spearman mean [min, max]", "two seeds: Kendall",
              f"two seeds: top-{TOP_K} overlap"]
    for size in pooled_sizes(cube):
        header += [f"two groups of {size} seeds: Spearman",
                   f"two groups of {size} seeds: Kendall",
                   f"two groups of {size} seeds: top-{TOP_K} overlap"]
    return [*header, "sets above zero (mean - 2 s.e. > 0)",
            "(seed, fold, set) not measurable"]


def count_sets_above_zero(cube: Cube, repeats: int) -> int:
    """Number of sets whose mean over all fold seeds is clearly above zero.

    "Clearly" = mean minus two standard errors > 0, the standard error taken
    over the seed means (seeds in which the set is not measurable are left
    out). A rough screen, not a formal test: the fold seeds re-split the
    same log, so they are not independent.
    """
    means = seed_means(cube, range(repeats))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        n_seeds = np.isfinite(means).sum(axis=0)
        error = np.nanstd(means, axis=0, ddof=1) / np.sqrt(n_seeds)
        return int((np.nanmean(means, axis=0) - 2 * error > 0).sum())


def group_stability(cube: Cube, repeats: int, rng) -> tuple[list, dict]:
    """Table rows and summary of the ranking stability of every group."""
    rows, summary = [], {}
    for name, (strategy, size) in groups().items():
        part = sub_cube(cube, strategy, size)
        one = between_seeds(part, repeats, blocks=[range(repeats)])
        not_measurable = int(np.isnan(part.values[:, :, :, 0]).sum())
        cells = part.values[:, :, :, 0].size
        above_zero = count_sets_above_zero(part, repeats)
        summary[name] = {"two_seeds": one, "sets_above_zero": above_zero,
                         "noise_and_signal": noise_and_signal(part, repeats),
                         "cells_not_measurable": not_measurable, "cells": cells}
        row = [name,
               f"{fmt(one['spearman_mean'], 2)} [{fmt(one['spearman_min'], 2)}, "
               f"{fmt(one['spearman_max'], 2)}]",
               fmt(one["kendall_mean"], 2), fmt(one["top_overlap_mean"], 2)]
        for group_size in pooled_sizes(cube):
            pooled = pooled_seeds(part, repeats, group_size, rng)
            summary[name][f"two_groups_of_{group_size}_seeds"] = pooled
            row += [fmt(pooled["spearman_mean"], 2), fmt(pooled["kendall_mean"], 2),
                    fmt(pooled["top_overlap_mean"], 2)]
        rows.append([*row, f"{above_zero} of {len(part.itemset_ids)}",
                     f"{not_measurable} of {cells}"])
    return rows, summary


def strategy_levels(cube: Cube, repeats: int) -> tuple[list, dict]:
    """Mean importance per strategy and size, and Apriori minus IMPresseD.

    Every number is first computed per fold seed (mean over the group's sets,
    the folds and the repeats); the table shows mean and std over the seeds
    and in how many seeds the difference is positive.
    """
    rows, summary = [], {}
    for size in (*SIZES, None):
        level = {}
        for strategy in STRATEGIES:
            means = seed_means(sub_cube(cube, strategy, size), range(repeats))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                level[strategy] = np.nanmean(means, axis=1)  # one value per seed
        difference = level["apriori"] - level["impressed"]
        name = "all sizes" if size is None else f"size {size}"
        summary[name] = {
            "apriori_mean": float(level["apriori"].mean()),
            "impressed_mean": float(level["impressed"].mean()),
            "difference_mean": float(difference.mean()),
            "difference_std_between_seeds": float(difference.std(ddof=1)),
            "seeds_with_positive_difference": int((difference > 0).sum()),
            "n_seeds": len(difference),
        }
        rows.append([
            name,
            f"{fmt(level['apriori'].mean())} ± {fmt(level['apriori'].std(ddof=1))}",
            f"{fmt(level['impressed'].mean())} ± "
            f"{fmt(level['impressed'].std(ddof=1))}",
            f"{fmt(difference.mean())} ± {fmt(difference.std(ddof=1))}",
            f"{(difference > 0).sum()} of {len(difference)}",
        ])
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="f1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-seeds", type=int, default=None,
                        help="use only this many fold seeds (the lowest ones); "
                             "the output files get the suffix _first<N>")
    args = parser.parse_args()

    rows = load_rows(args.dataset, "project", args.folds, args.max_seeds)
    rng = np.random.default_rng(0)
    text = [f"# C9 add-on: project sets on {args.dataset}\n"]
    summary = {"dataset": args.dataset, "repeats": args.repeats, "settings": {}}
    for label in SETTINGS:
        cube = build_cube(rows[rows["setting"] == label])
        text.append(f"## Setting `{label}` ({len(cube.seeds)} fold seeds, "
                    f"{cube.values.shape[1]} folds, {args.repeats} repeats)\n")
        text.append(f"Weighted F1 of the scored fold: {cube.baseline.mean():.4f} "
                    f"(std between seed means "
                    f"{cube.baseline.mean(axis=1).std(ddof=1):.4f}).\n")
        stability_rows, stability = group_stability(cube, args.repeats, rng)
        text.append("### Ranking stability inside a group of sets\n")
        text.append(md_table(stability_header(cube), stability_rows))
        level_rows, levels = strategy_levels(cube, args.repeats)
        text.append("### Level of the importance per strategy "
                    "(mean ± std over fold seeds)\n")
        text.append(md_table(
            ["sets", "Apriori", "IMPresseD", "Apriori - IMPresseD",
             "seeds with a positive difference"], level_rows))
        summary["settings"][label] = {"n_seeds": len(cube.seeds),
                                      "groups": stability, "levels": levels}

    stem = f"{args.dataset}_project_k{args.folds}_groups"
    if args.max_seeds is not None:
        stem += f"_first{args.max_seeds}"
    (RESULT_DIR / f"tables_{stem}.md").write_text("\n".join(text), encoding="utf-8")
    (RESULT_DIR / f"summary_{stem}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("\n".join(text))



if __name__ == "__main__":
    main()
