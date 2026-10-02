"""Speed of engine.py versus the original 2024 code (experiment C2, part f).

Setting: BPIC11 f1, fold 0 of a seeded 5-fold split, the original's ten
Apriori itemsets (min_support 0.5, top_k 10), 2 repeats each, one XGBoost
model fitted once on the original's training frame and shared by all runs.

Run:  python engine_benchmark.py [--dataset f1] [--repeats 2] [--timing-runs 3]

Writes results/experiments/C2/speed_<dataset>.json and
speed_importances_<dataset>.csv, and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import statistics
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
)

RESULT_DIR = Path(__file__).resolve().parents[1] / "results" / "experiments" / "C2"
FOLD_SEED = 0
ENGINE_VARIANTS = {
    # label: (mode, score_on, allowed_from)
    "engine faithful (train)": ("faithful", "train", "log"),
    "engine fixed (train, pools from log)": ("fixed", "train", "log"),
    "engine fixed (test, pools from log)": ("fixed", "test", "log"),
    "engine fixed (test, pools from train)": ("fixed", "test", "train"),
}


def timed(function, *args, **kwargs):
    """Return ``(result, seconds)`` of one call."""
    start = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="f1")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timing-runs", type=int, default=3,
                        help="how often each engine variant is timed")
    args = parser.parse_args()
    path = DATASETS[args.dataset]

    # One-time costs: loading and encoding the whole log.
    manager, seconds_manager = timed(reference.make_original_manager, path)
    encoded, seconds_encoding = timed(reference.original_encoding, manager)
    log, seconds_log = timed(EventLog, path)
    encoder, seconds_encoder_fit = timed(
        IndexEncoder, log.traces.values(), log.activities, "hash")
    _, seconds_transform = timed(encoder.transform, log.traces.values())

    train, test = make_folds(log.labels, k=5, seed=FOLD_SEED)[0]
    features, labels = reference.original_fold_data(manager, encoded, train)
    model, seconds_fit = timed(XGBClassifier().fit, features, labels)
    with reference.quiet():
        candidates, _ = manager.frequent_activity_sets(0.5, 10)
    itemsets = dict(enumerate(candidates))
    n_iterations = len(itemsets) * args.repeats
    print(f"{args.dataset}: {len(train)} train / {len(test)} test cases, "
          f"{len(itemsets)} itemsets x {args.repeats} repeats", flush=True)

    original, seconds_original = timed(
        reference.original_importance, manager, model, features, labels, train,
        itemsets, args.repeats)
    per_iteration = {"original (int cast, train)": seconds_original / n_iterations}
    all_runs = {"original (int cast, train)": [seconds_original / n_iterations]}
    print(f"original: {seconds_original:.1f} s total", flush=True)

    tables = []
    for label, (mode, score_on, allowed_from) in ENGINE_VARIANTS.items():
        importance = LocationPermutationImportance(
            mode=mode, score_on=score_on, n_repeats=args.repeats, random_state=2023,
            allowed_from=allowed_from)
        runs = []
        for _ in range(args.timing_runs):
            table, seconds = timed(
                importance.compute, model, log, encoder, train, test, itemsets,
                fold=0)
            runs.append(seconds / n_iterations)
        per_iteration[label] = statistics.median(runs)
        all_runs[label] = runs
        table.insert(0, "variant", label)
        tables.append(table)
        print(f"{label}: median {per_iteration[label]:.4f} s per iteration", flush=True)

    # The faithful run must reproduce the original numbers of this benchmark too.
    faithful = tables[0]
    faithful["original_importance"] = [
        original[row.itemset_id].iloc[row.repeat] for row in faithful.itertuples()]
    difference = faithful["importance"] - faithful["original_importance"]
    max_diff = float(difference.abs().max())

    reference_seconds = per_iteration["original (int cast, train)"]
    summary = {
        "dataset": args.dataset,
        "train_cases": len(train),
        "test_cases": len(test),
        "n_itemsets": len(itemsets),
        "n_repeats": args.repeats,
        "itemsets": {str(key): value for key, value in itemsets.items()},
        "seconds_per_iteration": per_iteration,
        "seconds_per_iteration_all_runs": all_runs,
        "speedup_vs_original": {
            label: reference_seconds / seconds
            for label, seconds in per_iteration.items()
        },
        "faithful_vs_original_max_abs_diff": max_diff,
        "one_time_seconds": {
            "original DataManager()": seconds_manager,
            "original index_encoding(whole log) + int cast": seconds_encoding,
            "engine EventLog()": seconds_log,
            "engine IndexEncoder() fit": seconds_encoder_fit,
            "engine IndexEncoder.transform(whole log)": seconds_transform,
            "XGBClassifier().fit (original frame, fold 0)": seconds_fit,
        },
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / f"speed_{args.dataset}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    pd.concat(tables).to_csv(
        RESULT_DIR / f"speed_importances_{args.dataset}.csv", index=False,
        float_format="%.17g")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
