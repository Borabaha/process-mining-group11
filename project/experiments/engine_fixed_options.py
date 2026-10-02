"""Do the open fixed-mode options change the result? (experiment C2)

Two choices of fixed mode are not pinned down by the paper:

* ``allowed_from``: position pools from the whole log or from the training
  fold only;
* ``draw``: positions of a set drawn sequentially (as the original) or
  uniformly over all order-preserving tuples.

For several fold seeds this script runs 5 folds x the original's Apriori
itemsets x ``--repeats`` repeats with the default (sequential draw, pools
from the log) and with each alternative, and reports the mean importance with
a 95 % interval over fold seeds, the paired difference to the default and
whether the itemsets are ranked alike.

Run:  python engine_fixed_options.py [--dataset f1] [--seeds 10] [--repeats 10]

Writes results/experiments/C2/fixed_options_<dataset>.csv (one row per
setting x seed x fold x itemset x repeat) and fixed_options_<dataset>.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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
DEFAULT = "sequential_draw_log_pools"
SETTINGS = {
    # label: (engine settings, folds it is scored on)
    DEFAULT: ({}, ("train", "test")),
    "uniform_draw_log_pools": ({"draw": "uniform"}, ("train", "test")),
    "sequential_draw_train_pools": ({"allowed_from": "train"}, ("test",)),
}
COUNT_COLUMNS = ["n_traces_changed", "n_traces_unchanged_infeasible",
                 "n_other_events_shifted", "n_other_events_unobserved"]


def run_seed(log, encoder, itemsets, seed: int, n_repeats: int) -> pd.DataFrame:
    """All settings for the 5 folds of one fold seed (one model per fold)."""
    tables = []
    for fold, (train, test) in enumerate(make_folds(log.labels, k=5, seed=seed)):
        model = XGBClassifier()
        model.fit(encoder.transform(log.traces[c] for c in train),
                  [log.labels[c] for c in train])
        for label, (settings, folds_scored) in SETTINGS.items():
            for score_on in folds_scored:
                table = LocationPermutationImportance(
                    mode="fixed", score_on=score_on, n_repeats=n_repeats,
                    random_state=2023, **settings,
                ).compute(model, log, encoder, train, test, itemsets, fold=fold)
                table.insert(0, "setting", label)
                table.insert(1, "score_on", score_on)
                table.insert(2, "seed", seed)
                tables.append(table)
    columns = ["setting", "score_on", "seed", "fold", "itemset_id", "itemset",
               "repeat", "importance", *COUNT_COLUMNS]
    return pd.concat(tables, ignore_index=True)[columns]


def interval(values: pd.Series) -> dict:
    """Mean and 95 % t-interval of per-seed values."""
    half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / len(values) ** 0.5
    mean = float(values.mean())
    return {"mean": mean, "ci95_low": mean - half, "ci95_high": mean + half}


def summarise(rows: pd.DataFrame) -> dict:
    """Per (score_on, setting): level, difference to the default, ranking."""
    summary = {}
    for score_on, part in rows.groupby("score_on", sort=False):
        by_seed = part.pivot_table(index="seed", columns="setting",
                                   values="importance", aggfunc="mean")
        by_itemset = part.pivot_table(index="itemset_id", columns="setting",
                                      values="importance", aggfunc="mean")
        totals = part.groupby("setting", sort=False)[COUNT_COLUMNS].sum()
        summary[score_on] = {}
        for label in by_seed.columns:
            entry = {
                "importance": interval(by_seed[label]),
                "share_of_values_above_zero": float(
                    (part.loc[part["setting"] == label, "importance"] > 0).mean()),
                "traces_changed": int(totals.loc[label, "n_traces_changed"]),
                "traces_unchanged_infeasible": int(
                    totals.loc[label, "n_traces_unchanged_infeasible"]),
                "share_of_shifted_events_at_unobserved_position": float(
                    totals.loc[label, "n_other_events_unobserved"]
                    / totals.loc[label, "n_other_events_shifted"]),
            }
            if label != DEFAULT:
                entry["difference_to_default"] = interval(
                    by_seed[label] - by_seed[DEFAULT])
                entry["spearman_with_default_ranking"] = float(
                    stats.spearmanr(by_itemset[label], by_itemset[DEFAULT])[0])
            summary[score_on][label] = entry
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="f1")
    parser.add_argument("--seeds", type=int, default=10,
                        help="number of fold seeds (0, 1, ...)")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    log = EventLog(DATASETS[args.dataset])
    encoder = IndexEncoder(log.traces.values(), log.activities)
    itemsets = original_apriori_itemsets(DATASETS[args.dataset])
    start = time.perf_counter()
    tables = []
    for seed in range(args.seeds):
        tables.append(run_seed(log, encoder, itemsets, seed, args.repeats))
        print(f"seed {seed} done ({time.perf_counter() - start:.0f} s)", flush=True)
    rows = pd.concat(tables, ignore_index=True)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(RESULT_DIR / f"fixed_options_{args.dataset}.csv", index=False,
                float_format="%.17g")
    summary = {"dataset": args.dataset, "fold_seeds": list(range(args.seeds)),
               "n_repeats": args.repeats, "n_itemsets": len(itemsets),
               "seconds_total": time.perf_counter() - start,
               "scored_on": summarise(rows)}
    (RESULT_DIR / f"fixed_options_{args.dataset}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
