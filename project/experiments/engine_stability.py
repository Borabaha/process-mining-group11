"""How stable are importance rankings? (experiment C2, noise of held-out scores)

Held-out importances in fixed mode are small compared with their spread, so
a ranking from one 5-fold split and 10 repeats may be noise. This script
measures it: for several fold seeds it runs 5 folds x the original's Apriori
itemsets with three engine settings

* faithful, training fold, 10 repeats   (what the original code computes)
* fixed, training fold, ``--repeats`` repeats
* fixed, held-out fold, ``--repeats`` repeats

and reports, per setting: mean importance per itemset with a 95 % interval
over fold seeds, where the variance comes from (repeats, folds, seeds), how
well two fold seeds agree on the ranking (Spearman), and how many itemset
pairs can be ordered at all.

Run:  python engine_stability.py [--dataset f1] [--seeds 10] [--repeats 30]
      python engine_stability.py --dataset f1 --reuse   (summary only)

Writes results/experiments/C2/stability_<dataset>.csv (one row per variant x
seed x fold x itemset x repeat), stability_<dataset>_itemsets.csv (the
per-itemset table) and stability_<dataset>.json (summary).
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from xgboost import XGBClassifier

from engine import (
    DATASETS,
    EventLog,
    IndexEncoder,
    LocationPermutationImportance,
    make_folds,
)
from engine_full_grid import original_apriori_itemsets

RESULT_DIR = Path(__file__).resolve().parents[1] / "results" / "experiments" / "C2"
ORIGINAL_REPEATS = 10
VARIANTS = {
    # label: (mode, score_on, use the --repeats option instead of 10 repeats)
    "faithful_train": ("faithful", "train", False),
    "fixed_train": ("fixed", "train", True),
    "fixed_test": ("fixed", "test", True),
}


def run_seed(log, encoder, itemsets, seed: int, n_repeats: int) -> pd.DataFrame:
    """All variants for the 5 folds of one fold seed (one model per fold)."""
    tables = []
    for fold, (train, test) in enumerate(make_folds(log.labels, k=5, seed=seed)):
        model = XGBClassifier()
        model.fit(encoder.transform(log.traces[c] for c in train),
                  [log.labels[c] for c in train])
        for label, (mode, score_on, many) in VARIANTS.items():
            table = LocationPermutationImportance(
                mode=mode, score_on=score_on, random_state=2023,
                n_repeats=n_repeats if many else ORIGINAL_REPEATS,
            ).compute(model, log, encoder, train, test, itemsets, fold=fold)
            table.insert(0, "variant", label)
            table.insert(1, "seed", seed)
            tables.append(table[["variant", "seed", "fold", "itemset_id", "itemset",
                                 "repeat", "importance"]])
    return pd.concat(tables, ignore_index=True)


def seed_means(rows: pd.DataFrame, n_repeats: int | None = None) -> pd.DataFrame:
    """Mean importance per (seed, itemset): rows = seeds, columns = itemsets."""
    if n_repeats is not None:
        rows = rows[rows["repeat"] < n_repeats]
    return rows.pivot_table(index="seed", columns="itemset_id", values="importance",
                            aggfunc="mean")


def mean_pairwise_spearman(means: pd.DataFrame) -> float:
    """Average Spearman correlation between the itemset rankings of two seeds."""
    correlations = [stats.spearmanr(means.loc[a], means.loc[b])[0]
                    for a, b in itertools.combinations(means.index, 2)]
    return float(np.mean(correlations))


def share_of_ordered_pairs(means: pd.DataFrame) -> float:
    """Share of itemset pairs whose order is significant (paired t, 5 %).

    The fold seeds are the replicates; no correction for multiple testing.
    """
    pairs = list(itertools.combinations(means.columns, 2))
    significant = sum(
        stats.ttest_rel(means[a], means[b]).pvalue < 0.05 for a, b in pairs)
    return significant / len(pairs)


def summarise_variant(rows: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Summary numbers and the per-itemset table of one variant."""
    means = seed_means(rows)
    n_seeds = len(means)
    half_width = stats.t.ppf(0.975, n_seeds - 1) * means.std(ddof=1) / n_seeds ** 0.5
    per_itemset = pd.DataFrame({
        "itemset": rows.groupby("itemset_id")["itemset"].first(),
        "mean": means.mean(),
        "std_between_seeds": means.std(ddof=1),
        "ci95_low": means.mean() - half_width,
        "ci95_high": means.mean() + half_width,
        "share_of_values_above_zero": (rows["importance"] > 0)
        .groupby(rows["itemset_id"]).mean(),
    })
    per_itemset["rank"] = per_itemset["mean"].rank(ascending=False).astype(int)

    cells = rows.groupby(["itemset_id", "seed", "fold"])["importance"]
    fold_means = cells.mean()
    halves = [means.iloc[: n_seeds // 2].mean(), means.iloc[n_seeds // 2:].mean()]
    summary = {
        "n_seeds": n_seeds,
        "n_repeats": int(rows["repeat"].max()) + 1,
        "mean_importance": float(rows["importance"].mean()),
        "spread_between_itemset_means": float(per_itemset["mean"].std(ddof=1)),
        "std_between_repeats_same_fold": float(cells.std(ddof=1).mean()),
        "std_between_folds_same_seed": float(
            fold_means.groupby(level=["itemset_id", "seed"]).std(ddof=1).mean()),
        "std_between_seed_means": float(per_itemset["std_between_seeds"].mean()),
        "mean_ci95_half_width": float(half_width.mean()),
        "itemsets_with_interval_above_zero": int((per_itemset["ci95_low"] > 0).sum()),
        "n_itemsets": len(per_itemset),
        "spearman_between_two_seeds_10_repeats": mean_pairwise_spearman(
            seed_means(rows, ORIGINAL_REPEATS)),
        "spearman_between_two_seeds_all_repeats": mean_pairwise_spearman(means),
        "spearman_between_two_halves_of_the_seeds": float(
            stats.spearmanr(halves[0], halves[1])[0]),
        "share_of_itemset_pairs_ordered": share_of_ordered_pairs(means),
    }
    return summary, per_itemset


def write_summary(rows: pd.DataFrame, dataset: str, seconds: float) -> None:
    """Write and print the per-itemset table and the summary of all variants."""
    summary = {"dataset": dataset, "fold_seeds": sorted(rows["seed"].unique().tolist()),
               "seconds_total": seconds, "variants": {}}
    per_itemset_tables = []
    for label, part in rows.groupby("variant", sort=False):
        summary["variants"][label], per_itemset = summarise_variant(part)
        per_itemset.insert(0, "variant", label)
        per_itemset_tables.append(per_itemset.reset_index())
    per_itemset = pd.concat(per_itemset_tables, ignore_index=True)
    # Do two settings rank the itemsets alike? (means over all seeds)
    pooled = per_itemset.pivot(index="itemset_id", columns="variant", values="mean")
    summary["spearman_between_settings"] = {
        f"{a} vs {b}": float(stats.spearmanr(pooled[a], pooled[b])[0])
        for a, b in itertools.combinations(VARIANTS, 2)}

    per_itemset.to_csv(RESULT_DIR / f"stability_{dataset}_itemsets.csv",
                       index=False, float_format="%.6f")
    (RESULT_DIR / f"stability_{dataset}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(per_itemset.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="f1")
    parser.add_argument("--seeds", type=int, default=10,
                        help="number of fold seeds (0, 1, ...)")
    parser.add_argument("--repeats", type=int, default=30,
                        help="permutation repeats in fixed mode")
    parser.add_argument("--reuse", action="store_true",
                        help="summarise the existing stability_<dataset>.csv again "
                             "instead of recomputing it")
    args = parser.parse_args()
    rows_path = RESULT_DIR / f"stability_{args.dataset}.csv"
    summary_path = RESULT_DIR / f"stability_{args.dataset}.json"

    if args.reuse:
        rows = pd.read_csv(rows_path)
        seconds = json.loads(summary_path.read_text(encoding="utf-8"))["seconds_total"]
    else:
        log = EventLog(DATASETS[args.dataset])
        encoder = IndexEncoder(log.traces.values(), log.activities)
        itemsets = original_apriori_itemsets(DATASETS[args.dataset])
        start = time.perf_counter()
        tables = []
        for seed in range(args.seeds):
            tables.append(run_seed(log, encoder, itemsets, seed, args.repeats))
            print(f"seed {seed} done ({time.perf_counter() - start:.0f} s)", flush=True)
        rows = pd.concat(tables, ignore_index=True)
        seconds = time.perf_counter() - start
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        rows.to_csv(rows_path, index=False, float_format="%.17g")
    write_summary(rows, args.dataset, seconds)


if __name__ == "__main__":
    main()
