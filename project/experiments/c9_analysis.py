"""Experiment C9, part 2: how stable is the ranking of activity sets? (analysis)

Reads the per-seed files written by ``c9_seed_stability.py`` and reports,
per engine setting:

* the model quality per fold seed (weighted F1 of the scored fold);
* the mean importance and the rank of every itemset per fold seed;
* the agreement of the rankings of two fold seeds (Spearman, Kendall tau-b,
  overlap of the three top-ranked sets), for 3, 5, 10 and 30 repeats;
* the same agreement when only the permutations differ (same folds, same
  models), which is the best any number of fold seeds could reach;
* the agreement of two independent studies that each pool several seeds;
* noise (std of one seed's itemset mean) against signal (spread between the
  itemset means).

Vocabulary: a "seed mean" is the mean importance of one itemset over all
folds and the chosen repeats of one fold seed. A ranking orders the itemsets
by their seed mean, rank 1 = most important.

Run:  python c9_analysis.py --dataset f1 [--itemsets original] [--folds 5]

Writes, into results/experiments/C9/: ``tables_<study>.md`` (all tables),
``summary_<study>.json``, ``per_seed_<study>.csv`` and ``pairwise_<study>.csv``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from c9_seed_stability import RESULT_DIR, SETTINGS, study_dir

REPEAT_SIZES = (3, 5, 10, 30)
GROUP_SIZES = (1, 2, 3, 5, 10)
N_GROUP_DRAWS = 300
N_BOOTSTRAP = 2000
TOP_K = 3
KEYS = ["seed", "fold", "itemset_id", "repeat"]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
@dataclass
class Cube:
    """The importance values of one setting as a 4-dimensional array.

    values   : seed x fold x itemset x repeat; NaN where the itemset occurs
               in no scored trace of that fold (not measurable).
    baseline : seed x fold, weighted F1 of the scored fold before permuting.
    """

    values: np.ndarray
    baseline: np.ndarray
    seeds: list
    itemset_ids: list
    labels: list

    @property
    def n_repeats(self) -> int:
        return self.values.shape[3]


def load_rows(dataset: str, itemsets: str, n_folds: int,
              max_seeds: int | None = None) -> pd.DataFrame:
    """All per-seed value files of one study in one table.

    ``max_seeds`` keeps only the ``max_seeds`` lowest fold seeds.
    """
    files = sorted(study_dir(dataset, itemsets, n_folds).glob("seed_*.csv"))
    if not files:
        raise FileNotFoundError("no seed_*.csv; run c9_seed_stability.py first")
    rows = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    if max_seeds is not None:
        rows = rows[rows["seed"].isin(sorted(rows["seed"].unique())[:max_seeds])]
    return rows


def build_cube(rows: pd.DataFrame) -> Cube:
    """Arrange the rows of ONE setting as a ``Cube`` (checks completeness)."""
    rows = rows.sort_values(KEYS)
    levels = [sorted(rows[key].unique()) for key in KEYS]
    shape = tuple(len(level) for level in levels)
    if len(rows) != np.prod(shape) or rows.duplicated(KEYS).any():
        raise ValueError(f"incomplete grid: {len(rows)} rows for shape {shape}")
    values = rows["importance"].to_numpy(dtype=float).reshape(shape)
    measurable = (rows["n_traces_with_itemset"] > 0).to_numpy().reshape(shape)
    labels = rows.groupby("itemset_id")["itemset"].first().loc[levels[2]].tolist()
    baseline = rows["baseline"].to_numpy().reshape(shape)[:, :, 0, 0]
    return Cube(np.where(measurable, values, np.nan), baseline,
                levels[0], levels[2], labels)


def seed_means(cube: Cube, repeats) -> np.ndarray:
    """Seed x itemset: mean importance over the folds and ``repeats``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN itemsets
        return np.nanmean(cube.values[:, :, :, list(repeats)], axis=(1, 3))


def repeat_sizes(label: str, cube: Cube) -> list[int]:
    """The repeat counts that can be studied for the setting ``label``.

    Fixed mode: every size of ``REPEAT_SIZES`` that was run, because a subset
    of the repeats is itself a valid run. Faithful mode: only the full run,
    because every repeat depends on all repeats before it.
    """
    if SETTINGS[label][0] == "fixed":
        return [size for size in REPEAT_SIZES if size <= cube.n_repeats]
    return [cube.n_repeats]


def repeat_blocks(n_repeats: int, size: int) -> list[range]:
    """Disjoint blocks of ``size`` repeats: [0..size-1], [size..2*size-1], ..."""
    return [range(start, start + size)
            for start in range(0, n_repeats - size + 1, size)]


def ranks(means: np.ndarray) -> np.ndarray:
    """Rank per row, 1 = largest mean; ties get the average rank."""
    return stats.rankdata(-means, axis=1, nan_policy="omit")


# --------------------------------------------------------------------------
# Agreement of two rankings
# --------------------------------------------------------------------------
def agreement(first: np.ndarray, second: np.ndarray) -> tuple[float, float, float]:
    """Spearman, Kendall tau-b and top-``TOP_K`` overlap of two mean vectors.

    Itemsets without a value in one of the vectors are left out. The overlap
    is the share of the ``TOP_K`` highest itemsets of ``first`` that are also
    among the ``TOP_K`` highest of ``second``.
    """
    both = np.isfinite(first) & np.isfinite(second)
    first, second = first[both], second[both]
    if len(first) < 3 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return np.nan, np.nan, np.nan
    top_first = set(np.argsort(-first, kind="stable")[:TOP_K])
    top_second = set(np.argsort(-second, kind="stable")[:TOP_K])
    return (float(stats.spearmanr(first, second).statistic),
            float(stats.kendalltau(first, second).statistic),
            len(top_first & top_second) / TOP_K)


def summarise_agreements(triples) -> dict:
    """Mean, minimum and maximum of a list of ``agreement`` results."""
    table = np.array(list(triples), dtype=float).reshape(-1, 3)
    if not len(table):
        return {"n": 0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return {
            "n": len(table),
            "spearman_mean": float(np.nanmean(table[:, 0])),
            "spearman_min": float(np.nanmin(table[:, 0])),
            "spearman_max": float(np.nanmax(table[:, 0])),
            "kendall_mean": float(np.nanmean(table[:, 1])),
            "kendall_min": float(np.nanmin(table[:, 1])),
            "kendall_max": float(np.nanmax(table[:, 1])),
            "top_overlap_mean": float(np.nanmean(table[:, 2])),
        }


def between_seeds(cube: Cube, size: int, seed_rows=None, blocks=None) -> dict:
    """Agreement of two fold seeds that each use ``size`` repeats.

    Averaged over all pairs of seeds (``seed_rows`` = positions of the seeds
    to use, default all) and over ``blocks`` of repeats (default: every
    disjoint block of ``size`` repeats, to use all the data).
    """
    seed_rows = range(len(cube.seeds)) if seed_rows is None else seed_rows
    blocks = repeat_blocks(cube.n_repeats, size) if blocks is None else blocks
    triples = []
    for block in blocks:
        means = seed_means(cube, block)
        triples += [agreement(means[a], means[b])
                    for a, b in itertools.combinations(seed_rows, 2)]
    return summarise_agreements(triples)


def seed_bootstrap_interval(cube: Cube, size: int, rng) -> tuple[float, float]:
    """95 % bootstrap interval of the mean Spearman between two fold seeds.

    The fold seeds are resampled with replacement (a seed is never compared
    with itself). The interval shows how much the stability figure itself
    depends on which fold seeds happened to be run.
    """
    n_seeds = len(cube.seeds)
    blocks = repeat_blocks(cube.n_repeats, size)
    matrix = np.zeros((n_seeds, n_seeds))
    for block in blocks:
        means = seed_means(cube, block)
        for a, b in itertools.combinations(range(n_seeds), 2):
            matrix[a, b] += agreement(means[a], means[b])[0] / len(blocks)
    matrix += matrix.T
    draws = []
    for _ in range(N_BOOTSTRAP):
        pick = rng.integers(n_seeds, size=n_seeds)
        different = pick[:, None] != pick[None, :]
        draws.append(np.nanmean(matrix[np.ix_(pick, pick)][different]))
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def same_folds(cube: Cube, size: int) -> dict:
    """Agreement of two runs that differ ONLY in the permutations.

    Both runs use the same fold seed (same folds, same models) and disjoint
    blocks of ``size`` repeats. Needs at least two blocks.
    """
    block_means = [seed_means(cube, block)
                   for block in repeat_blocks(cube.n_repeats, size)]
    triples = [agreement(first[row], second[row])
               for first, second in itertools.combinations(block_means, 2)
               for row in range(len(cube.seeds))]
    return summarise_agreements(triples)


def pooled_seeds(cube: Cube, size: int, group_size: int, rng) -> dict:
    """Agreement of two studies that each pool ``group_size`` fold seeds.

    The two groups are disjoint random draws from the available seeds. Every
    draw also picks one of the disjoint blocks of ``size`` repeats at random
    (the same block for all seeds). Mean over ``N_GROUP_DRAWS`` draws.
    """
    block_means = [seed_means(cube, block)
                   for block in repeat_blocks(cube.n_repeats, size)]
    triples = []
    for _ in range(N_GROUP_DRAWS):
        means = block_means[rng.integers(len(block_means))]
        order = rng.permutation(len(cube.seeds))
        first, second = order[:group_size], order[group_size:2 * group_size]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            triples.append(agreement(np.nanmean(means[first], axis=0),
                                     np.nanmean(means[second], axis=0)))
    return summarise_agreements(triples)


# --------------------------------------------------------------------------
# Noise against signal
# --------------------------------------------------------------------------
def noise_and_signal(cube: Cube, size: int) -> dict:
    """Std of a seed mean (noise) and spread between the itemsets (signal).

    noise  : std over the fold seeds of an itemset's seed mean with ``size``
             repeats, averaged over itemsets and repeat blocks.
    ranking noise : the same after subtracting, per fold seed, the mean over
             all itemsets. A fold seed that moves all itemsets up or down
             together does not change their order, so this is the noise
             that matters for a ranking.
    signal : std over the itemsets of their mean over ALL seeds and repeats;
             ``median_gap`` is the median distance between neighbours in
             that order (what must be resolved to rank them).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        noise, ranking_noise = [], []
        for block in repeat_blocks(cube.n_repeats, size):
            means = seed_means(cube, block)
            centred = means - np.nanmean(means, axis=1, keepdims=True)
            noise.append(np.nanmean(np.nanstd(means, axis=0, ddof=1)))
            ranking_noise.append(np.nanmean(np.nanstd(centred, axis=0, ddof=1)))
        noise, ranking_noise = float(np.mean(noise)), float(np.mean(ranking_noise))
        grand = np.nanmean(seed_means(cube, range(cube.n_repeats)), axis=0)
    grand = grand[np.isfinite(grand)]
    signal = float(np.std(grand, ddof=1))
    return {
        "noise_std_of_seed_mean": noise,
        "ranking_noise_std_of_centred_seed_mean": ranking_noise,
        "signal_std_between_itemsets": signal,
        "signal_to_noise": signal / noise,
        "signal_to_ranking_noise": signal / ranking_noise,
        "median_gap_between_neighbours": float(np.median(np.diff(np.sort(grand)))),
    }


def variance_sources(cube: Cube) -> dict:
    """Std between repeats (same fold), fold means (same seed) and seed means."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fold_means = np.nanmean(cube.values, axis=3)
        return {
            "std_between_repeats_same_fold": float(
                np.nanmean(np.nanstd(cube.values, axis=3, ddof=1))),
            "std_between_fold_means_same_seed": float(
                np.nanmean(np.nanstd(fold_means, axis=1, ddof=1))),
            "std_between_seed_means": float(
                np.nanmean(np.nanstd(np.nanmean(fold_means, axis=1), axis=0,
                                     ddof=1))),
            "share_values_above_zero": float(np.nanmean(cube.values > 0)),
            "share_values_equal_zero": float(np.nanmean(cube.values == 0)),
            "share_values_below_zero": float(np.nanmean(cube.values < 0)),
            "share_cells_not_measurable": float(np.mean(np.isnan(cube.values))),
        }


# --------------------------------------------------------------------------
# Markdown output
# --------------------------------------------------------------------------
def md_table(header, rows) -> str:
    """A GitHub-style markdown table."""
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def fmt(value, digits: int = 4) -> str:
    """Fixed-point text of a number ('-' for a missing one)."""
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def model_quality_table(cube: Cube, seed_rows) -> str:
    """Weighted F1 of the scored fold per fold seed (mean, std, range)."""
    rows = [[cube.seeds[row], fmt(cube.baseline[row].mean()),
             fmt(cube.baseline[row].std(ddof=1)),
             fmt(cube.baseline[row].min()), fmt(cube.baseline[row].max())]
            for row in seed_rows]
    part = cube.baseline[list(seed_rows)]
    rows.append(["all listed", fmt(part.mean()),
                 fmt(part.mean(axis=1).std(ddof=1)) + " (between seeds)",
                 fmt(part.min()), fmt(part.max())])
    return md_table(["fold seed", "mean over folds", "std over folds",
                     "lowest fold", "highest fold"], rows)


def per_seed_tables(cube: Cube, seed_rows, size: int) -> tuple[str, str]:
    """Mean importance and rank per itemset (rows) and fold seed (columns)."""
    means = seed_means(cube, range(size))[list(seed_rows)]
    rank = ranks(means)
    pooled_rank = stats.rankdata(-means.mean(axis=0))
    seeds = [f"seed {cube.seeds[row]}" for row in seed_rows]
    mean_rows, rank_rows = [], []
    for col, (itemset_id, label) in enumerate(zip(cube.itemset_ids, cube.labels)):
        mean_rows.append([itemset_id, label, *[fmt(v) for v in means[:, col]],
                          fmt(means[:, col].mean()),
                          fmt(means[:, col].std(ddof=1))])
        rank_rows.append([itemset_id, label, *[f"{v:g}" for v in rank[:, col]],
                          f"{rank[:, col].min():g}-{rank[:, col].max():g}",
                          f"{pooled_rank[col]:g}"])
    return (md_table(["id", "itemset", *seeds, "mean", "std between seeds"],
                     mean_rows),
            md_table(["id", "itemset", *seeds, "range",
                      "rank of the mean over these seeds"], rank_rows))


def pair_matrix(cube: Cube, seed_rows, size: int) -> str:
    """Spearman above the diagonal, Kendall tau-b below it."""
    means = seed_means(cube, range(size))
    names = [f"seed {cube.seeds[row]}" for row in seed_rows]
    body = []
    for a in seed_rows:
        cells = []
        for b in seed_rows:
            if a == b:
                cells.append("")
            else:
                spearman, kendall, _ = agreement(means[a], means[b])
                cells.append(fmt(spearman if b > a else kendall, 2))
        body.append([f"**seed {cube.seeds[a]}**", *cells])
    return md_table(["", *names], body)


def agreement_row(label, result: dict) -> list:
    """One table row for a ``summarise_agreements`` result."""
    if not result.get("n"):
        return [label, "-", "-", "-", "-"]
    return [label,
            f"{fmt(result['spearman_mean'], 2)} "
            f"[{fmt(result['spearman_min'], 2)}, {fmt(result['spearman_max'], 2)}]",
            f"{fmt(result['kendall_mean'], 2)} "
            f"[{fmt(result['kendall_min'], 2)}, {fmt(result['kendall_max'], 2)}]",
            fmt(result["top_overlap_mean"], 2), result["n"]]


AGREEMENT_HEADER = ["", "Spearman mean [min, max]", "Kendall tau-b mean [min, max]",
                    f"top-{TOP_K} overlap", "comparisons"]


# --------------------------------------------------------------------------
# One setting
# --------------------------------------------------------------------------
def analyse_setting(label: str, cube: Cube, n_primary: int, primary_repeats: int):
    """Markdown text and summary dict of one engine setting."""
    primary = range(min(n_primary, len(cube.seeds)))
    # Only fixed mode allows "fewer repeats" to be studied on subsets of the
    # repeats; in faithful mode a repeat depends on all repeats before it.
    subsets_valid = SETTINGS[label][0] == "fixed"
    sizes = repeat_sizes(label, cube)
    primary_repeats = (min(primary_repeats, cube.n_repeats) if subsets_valid
                       else cube.n_repeats)
    rng = np.random.default_rng(0)
    summary = {"n_seeds": len(cube.seeds), "n_repeats_run": cube.n_repeats,
               "n_folds": cube.values.shape[1], "n_itemsets": len(cube.itemset_ids),
               "variance_sources": variance_sources(cube)}
    text = [f"## Setting `{label}`\n"]

    scored = SETTINGS[label][1]
    text.append(f"### Weighted F1 of the scored ({scored}) fold, per fold seed\n")
    text.append(model_quality_table(cube, primary))
    summary["baseline_mean_all_seeds"] = float(cube.baseline.mean())
    summary["baseline_std_between_seed_means"] = float(
        cube.baseline.mean(axis=1).std(ddof=1))

    means_text, ranks_text = per_seed_tables(cube, primary, primary_repeats)
    design = (f"first {len(primary)} fold seeds, {cube.values.shape[1]} folds x "
              f"{primary_repeats} repeats")
    text.append(f"### Mean importance per itemset and fold seed ({design})\n")
    text.append(means_text)
    text.append(f"### Rank per itemset and fold seed ({design}; 1 = most important)\n")
    text.append(ranks_text)
    text.append(f"### Agreement of the rankings of two fold seeds ({design})\n")
    text.append("Spearman above the diagonal, Kendall tau-b below it.\n")
    text.append(pair_matrix(cube, primary, primary_repeats))

    text.append("### Effect of the number of repeats on the agreement of two "
                "fold seeds\n")
    rows, summary["between_seeds"] = [], {}
    for size in sizes:
        first = between_seeds(cube, size, primary, [range(size)])
        every = between_seeds(cube, size)
        low, high = seed_bootstrap_interval(cube, size, rng)
        every["spearman_mean_bootstrap_95"] = [low, high]
        summary["between_seeds"][size] = {"primary_first_block": first,
                                          "all_seeds_all_blocks": every}
        rows.append(agreement_row(f"{size} repeats, first {len(primary)} seeds, "
                                  f"repeats 0..{size - 1}", first))
        rows.append(agreement_row(
            f"{size} repeats, all {len(cube.seeds)} seeds, all disjoint repeat "
            f"blocks (95 % bootstrap interval of the Spearman mean: "
            f"{fmt(low, 2)} to {fmt(high, 2)})", every))
    text.append(md_table(AGREEMENT_HEADER, rows))

    text.append(f"### The same design with other groups of {len(primary)} fold "
                "seeds\n")
    rows, summary["seed_groups"] = [], {}
    for start in range(0, len(cube.seeds) - len(primary) + 1, len(primary)):
        group = range(start, start + len(primary))
        name = f"seeds {cube.seeds[group[0]]}-{cube.seeds[group[-1]]}"
        for size in sorted({primary_repeats, min(10, cube.n_repeats)}):
            result = between_seeds(cube, size, group, [range(size)])
            summary["seed_groups"][f"{name}, {size} repeats"] = result
            rows.append(agreement_row(f"{name}, repeats 0..{size - 1}", result))
    text.append(md_table(AGREEMENT_HEADER, rows))

    rows, summary["same_folds"] = [], {}
    for size in sizes if subsets_valid else []:
        if len(repeat_blocks(cube.n_repeats, size)) >= 2:
            summary["same_folds"][size] = same_folds(cube, size)
            rows.append(agreement_row(f"{size} repeats", summary["same_folds"][size]))
    if rows:
        text.append("### Same folds and models, only the permutations differ\n")
        text.append(md_table(AGREEMENT_HEADER, rows))

    text.append("### Two independent studies that each pool several fold seeds\n")
    rows, summary["pooled_seeds"] = [], {}
    for size in [s for s in sizes if s in (5, 10, 30) or not subsets_valid]:
        for group_size in GROUP_SIZES:
            if 2 * group_size > len(cube.seeds):
                continue
            result = pooled_seeds(cube, size, group_size, rng)
            summary["pooled_seeds"][f"{group_size}x{size}"] = result
            rows.append(agreement_row(
                f"{group_size} seed(s) x {cube.values.shape[1]} folds x {size} "
                f"repeats = {group_size * cube.values.shape[1] * size} values "
                "per itemset", result))
    text.append(md_table(AGREEMENT_HEADER, rows))

    text.append("### Noise of one seed mean against the spread between itemsets\n")
    rows, summary["noise_and_signal"] = [], {}
    for size in sizes:
        result = noise_and_signal(cube, size)
        summary["noise_and_signal"][size] = result
        rows.append([size, fmt(result["noise_std_of_seed_mean"], 5),
                     fmt(result["ranking_noise_std_of_centred_seed_mean"], 5),
                     fmt(result["signal_std_between_itemsets"], 5),
                     fmt(result["signal_to_ranking_noise"], 2),
                     fmt(result["median_gap_between_neighbours"], 5)])
    text.append(md_table(["repeats", "std of a seed mean (noise)",
                          "the same without the seed's common shift "
                          "(ranking noise)",
                          "std between itemset means (signal)",
                          "signal / ranking noise",
                          "median gap between neighbouring itemsets"], rows))
    sources = summary["variance_sources"]
    text.append(md_table(
        ["std between repeats (same fold)", "std between fold means (same seed)",
         "std between seed means", "values > 0", "values = 0", "values < 0"],
        [[fmt(sources["std_between_repeats_same_fold"], 5),
          fmt(sources["std_between_fold_means_same_seed"], 5),
          fmt(sources["std_between_seed_means"], 5),
          f"{100 * sources['share_values_above_zero']:.1f} %",
          f"{100 * sources['share_values_equal_zero']:.1f} %",
          f"{100 * sources['share_values_below_zero']:.1f} %"]]))
    return "\n".join(text), summary


def reference_table(cubes: dict) -> str:
    """Mean over all seeds and repeats per itemset and setting, with ranks."""
    header, columns = ["id", "itemset"], []
    for label, cube in cubes.items():
        means = seed_means(cube, range(cube.n_repeats))
        grand = np.nanmean(means, axis=0)
        error = np.nanstd(means, axis=0, ddof=1) / np.sqrt(len(cube.seeds))
        rank = stats.rankdata(-grand, nan_policy="omit")
        header += [f"{label}: mean ± s.e.", "rank"]
        columns.append([f"{fmt(m, 5)} ± {fmt(e, 5)}" for m, e in zip(grand, error)])
        columns.append([f"{r:g}" for r in rank])
    first = next(iter(cubes.values()))
    rows = [[itemset_id, label, *[column[i] for column in columns]]
            for i, (itemset_id, label) in enumerate(zip(first.itemset_ids,
                                                        first.labels))]
    return md_table(header, rows)


def between_settings(cubes: dict) -> str:
    """Spearman / Kendall between the all-seed rankings of two settings."""
    grand = {label: np.nanmean(seed_means(cube, range(cube.n_repeats)), axis=0)
             for label, cube in cubes.items()}
    rows = []
    for a, b in itertools.combinations(grand, 2):
        spearman, kendall, overlap = agreement(grand[a], grand[b])
        rows.append([f"{a} vs {b}", fmt(spearman, 2), fmt(kendall, 2),
                     fmt(overlap, 2)])
    return md_table(["settings", "Spearman", "Kendall tau-b",
                     f"top-{TOP_K} overlap"], rows)


def export_csv(cubes: dict, stem: str) -> None:
    """Per-seed means/ranks and the pairwise agreements as tidy CSV files."""
    per_seed, pairwise = [], []
    for label, cube in cubes.items():
        for size in repeat_sizes(label, cube):
            means = seed_means(cube, range(size))
            rank = ranks(means)
            for row, seed in enumerate(cube.seeds):
                for col, itemset_id in enumerate(cube.itemset_ids):
                    per_seed.append((label, size, seed, itemset_id,
                                     cube.labels[col], means[row, col],
                                     rank[row, col]))
            for a, b in itertools.combinations(range(len(cube.seeds)), 2):
                pairwise.append((label, size, cube.seeds[a], cube.seeds[b],
                                 *agreement(means[a], means[b])))
    pd.DataFrame(per_seed, columns=[
        "setting", "repeats", "seed", "itemset_id", "itemset", "mean_importance",
        "rank"]).to_csv(RESULT_DIR / f"per_seed_{stem}.csv", index=False)
    pd.DataFrame(pairwise, columns=[
        "setting", "repeats", "seed_a", "seed_b", "spearman", "kendall",
        f"top{TOP_K}_overlap"]).to_csv(RESULT_DIR / f"pairwise_{stem}.csv",
                                       index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="f1")
    parser.add_argument("--itemsets", choices=["original", "project"],
                        default="original")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--primary-seeds", type=int, default=5,
                        help="number of fold seeds shown in the per-seed tables")
    parser.add_argument("--primary-repeats", type=int, default=5,
                        help="repeats used in the per-seed tables")
    parser.add_argument("--max-seeds", type=int, default=None,
                        help="use only this many fold seeds (the lowest ones); "
                             "the output files get the suffix _first<N>")
    args = parser.parse_args()

    rows = load_rows(args.dataset, args.itemsets, args.folds, args.max_seeds)
    cubes = {label: build_cube(rows[rows["setting"] == label])
             for label in SETTINGS if (rows["setting"] == label).any()}
    stem = f"{args.dataset}_{args.itemsets}_k{args.folds}"
    if args.max_seeds is not None:
        stem += f"_first{args.max_seeds}"
    text = [f"# C9 tables: {args.dataset}, {args.itemsets} itemsets, "
            f"{args.folds} folds\n"]
    summary = {"dataset": args.dataset, "itemsets": args.itemsets,
               "n_folds": args.folds, "settings": {}}
    for label, cube in cubes.items():
        part, summary["settings"][label] = analyse_setting(
            label, cube, args.primary_seeds, args.primary_repeats)
        text.append(part)
    text.append("## All settings: mean over all fold seeds and repeats\n")
    text.append(reference_table(cubes))
    text.append("### Do two settings rank the itemsets alike?\n")
    text.append(between_settings(cubes))

    (RESULT_DIR / f"tables_{stem}.md").write_text("\n".join(text), encoding="utf-8")
    (RESULT_DIR / f"summary_{stem}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    export_csv(cubes, stem)
    print("\n".join(text))


if __name__ == "__main__":
    main()
