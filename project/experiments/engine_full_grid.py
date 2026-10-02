"""Full grid with engine.py only (experiment C2, end-to-end timing).

Runs the grid of the original script (5 seeded folds x the original's Apriori
itemsets x 10 repeats) with five engine settings and reports how long it
takes and what the importance values look like:

* faithful, scored on the training fold  (= what the original code computes)
* fixed, scored on the training fold     (only the permutation is repaired)
* fixed, scored on the held-out fold     (the working default of the project)
* as the previous one, position pools from the training fold only
* as the working default, positions drawn uniformly over all feasible tuples

Run:  python engine_full_grid.py [--dataset f1] [--repeats 10]

Writes results/experiments/C2/full_grid_<dataset>.csv (tidy, one row per
variant x fold x itemset x repeat) and full_grid_<dataset>.json (summary).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

import engine_reference as reference
from engine import (
    DATASETS,
    EventLog,
    IndexEncoder,
    LocationPermutationImportance,
    make_folds,
    observed_locations,
    weighted_f1,
)

RESULT_DIR = Path(__file__).resolve().parents[1] / "results" / "experiments" / "C2"
FOLD_SEED = 0
VARIANTS = {
    # label: (mode, score_on, further engine settings)
    "faithful_train": ("faithful", "train", {}),
    "fixed_train": ("fixed", "train", {}),
    "fixed_test": ("fixed", "test", {}),
    "fixed_test_train_pools": ("fixed", "test", {"allowed_from": "train"}),
    "fixed_test_uniform_draw": ("fixed", "test", {"draw": "uniform"}),
}


def original_apriori_itemsets(path) -> dict:
    """The itemsets the original selects (min_support 0.5, top_k 10)."""
    manager = reference.make_original_manager(path)
    with reference.quiet():
        candidates, _ = manager.frequent_activity_sets(0.5, 10)
    return dict(enumerate(candidates))


def count_position_pairs(log: EventLog, train_cases) -> tuple[int, int]:
    """(activity, position) pairs in the log / of those, unseen in training."""
    in_training = observed_locations(log.traces[case] for case in train_cases)
    n_log = sum(len(positions) for positions in log.allowed_locations.values())
    n_training = sum(len(positions) for positions in in_training.values())
    return n_log, n_log - n_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="f1")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    log = EventLog(DATASETS[args.dataset])
    encoder = IndexEncoder(log.traces.values(), log.activities)
    itemsets = original_apriori_itemsets(DATASETS[args.dataset])
    folds = make_folds(log.labels, k=5, seed=FOLD_SEED)

    seconds = {label: 0.0 for label in VARIANTS}
    seconds_fit = 0.0
    test_f1, tables, held_out_only_pairs = [], [], []
    for fold, (train, test) in enumerate(folds):
        n_position_pairs, n_held_out_only = count_position_pairs(log, train)
        held_out_only_pairs.append(n_held_out_only)
        start = time.perf_counter()
        model = XGBClassifier()
        model.fit(encoder.transform(log.traces[c] for c in train),
                  [log.labels[c] for c in train])
        seconds_fit += time.perf_counter() - start
        test_f1.append(weighted_f1(
            [log.labels[c] for c in test],
            model.predict(encoder.transform(log.traces[c] for c in test))))
        for label, (mode, score_on, settings) in VARIANTS.items():
            start = time.perf_counter()
            table = LocationPermutationImportance(
                mode=mode, score_on=score_on, n_repeats=args.repeats,
                random_state=2023, **settings,
            ).compute(model, log, encoder, train, test, itemsets, fold=fold)
            seconds[label] += time.perf_counter() - start
            table.insert(0, "variant", label)
            tables.append(table)
        print(f"fold {fold} done (test F1 {test_f1[-1]:.4f})", flush=True)

    grid = pd.concat(tables, ignore_index=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(RESULT_DIR / f"full_grid_{args.dataset}.csv", index=False,
                float_format="%.17g")

    summary_table = grid.pivot_table(
        index=["itemset_id", "itemset"], columns="variant", values="importance",
        aggfunc=["mean", "std"], sort=False)
    n_iterations = len(folds) * len(itemsets) * args.repeats
    by_variant = grid.groupby("variant", sort=False)
    fixed = grid[grid["variant"] != "faithful_train"].groupby("variant", sort=False)
    shifted = fixed["n_other_events_shifted"].sum()
    summary = {
        "dataset": args.dataset,
        "folds": len(folds),
        "n_itemsets": len(itemsets),
        "n_repeats": args.repeats,
        "iterations_per_variant": n_iterations,
        "test_f1_per_fold": test_f1,
        "seconds_model_fits": seconds_fit,
        "seconds_total_per_variant": seconds,
        "seconds_per_iteration": {k: v / n_iterations for k, v in seconds.items()},
        "infeasible_traces_total": {
            label: int(part["n_traces_unchanged_infeasible"].sum())
            for label, part in by_variant
        },
        "baseline_mean": by_variant["baseline"].mean().to_dict(),
        "importance_mean": by_variant["importance"].mean().to_dict(),
        "share_of_values_above_zero": (grid["importance"] > 0)
        .groupby(grid["variant"], sort=False).mean().to_dict(),
        "traces_changed_total": by_variant["n_traces_changed"].sum().to_dict(),
        # Events that were not moved but were shifted by the move, and how
        # many of them ended on a position outside their activity's pool.
        "other_events_shifted_total": shifted.astype(int).to_dict(),
        "other_events_at_unobserved_position_total": fixed[
            "n_other_events_unobserved"].sum().astype(int).to_dict(),
        "share_of_shifted_events_at_unobserved_position": (
            fixed["n_other_events_unobserved"].sum() / shifted).to_dict(),
        "position_pairs_whole_log": n_position_pairs,
        "position_pairs_only_in_held_out_fold_per_fold": held_out_only_pairs,
    }
    (RESULT_DIR / f"full_grid_{args.dataset}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_table.to_string(float_format=lambda v: f"{v:.4f}"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
