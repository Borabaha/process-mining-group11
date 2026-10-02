"""Experiment C1, part 5: does the engine need XGBoost's threads?

``XGBClassifier()`` uses every logical core by default (``n_jobs=None``).
The engine calls ``model.predict`` once per (itemset, repeat) on a few
hundred rows, so most of that thread pool is waiting. This script times the
same workload (fold 0, the ten Apriori sets of one size, 10 repeats) with
models that differ only in ``n_jobs`` and records how many cores the process
really used.

Run:  python c1_threads.py --dataset f1 [--size 2] [--timing-runs 3]

Writes results/experiments/C1/threads_<dataset>.json.
"""

from __future__ import annotations

import argparse
import statistics
import time
import warnings

from xgboost import XGBClassifier

from c1_common import (
    FOLD_SEED,
    N_FOLDS,
    RESULT_DIR,
    LoadMeter,
    load_itemsets,
    sample_system_load,
    write_json,
)
from engine import (
    DATASETS,
    EventLog,
    IndexEncoder,
    LocationPermutationImportance,
    make_folds,
)

VARIANTS = {
    # label: (mode, score_on)
    "faithful_train": ("faithful", "train"),
    "fixed_train": ("fixed", "train"),
    "fixed_test": ("fixed", "test"),
}
N_JOBS = (None, 1, 2, 4)   # None = XGBoost default = all logical cores
N_REPEATS = 10


def say(text: str) -> None:
    """Progress line with a wall-clock stamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="f1")
    parser.add_argument("--size", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--timing-runs", type=int, default=3)
    args = parser.parse_args()

    load_before = sample_system_load()
    log = EventLog(DATASETS[args.dataset])
    encoder = IndexEncoder(log.traces.values(), log.activities)
    train, test = make_folds(log.labels, k=N_FOLDS, seed=FOLD_SEED)[0]
    train_matrix = encoder.transform(log.traces[case] for case in train)
    train_labels = [log.labels[case] for case in train]
    itemsets = load_itemsets(log, args.dataset, "apriori")[args.size]
    n_iterations = len(itemsets) * N_REPEATS

    records, reference_tables = [], {}
    for n_jobs in N_JOBS:
        with LoadMeter() as meter:
            model = XGBClassifier(n_jobs=n_jobs).fit(train_matrix, train_labels)
        fit = meter.as_dict()
        for label, (mode, score_on) in VARIANTS.items():
            engine = LocationPermutationImportance(
                mode=mode, score_on=score_on, n_repeats=N_REPEATS, random_state=2023)
            runs = []
            for _ in range(args.timing_runs):
                with LoadMeter() as meter, warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    table = engine.compute(model, log, encoder, train, test,
                                           itemsets, fold=0)
                runs.append(meter.as_dict())
            # The number of threads must not change any importance value.
            importance = table["importance"].tolist()
            same = importance == reference_tables.setdefault(label, importance)
            records.append({
                "n_jobs": "default (all cores)" if n_jobs is None else n_jobs,
                "variant": label,
                "fit_seconds": fit["seconds"],
                "seconds_per_iteration_median": statistics.median(
                    run["seconds"] for run in runs) / n_iterations,
                "seconds_per_iteration_runs": [
                    run["seconds"] / n_iterations for run in runs],
                "own_cores_median": statistics.median(
                    run["own_cores"] for run in runs),
                "others_percent_median": statistics.median(
                    run["others_percent"] for run in runs),
                "same_importance_as_default": same,
            })
            say(f"n_jobs={n_jobs} {label}: "
                f"{records[-1]['seconds_per_iteration_median']:.4f} s/iteration, "
                f"{records[-1]['own_cores_median']:.2f} cores, "
                f"fit {fit['seconds']:.2f} s, "
                f"same values: {same}")

    write_json(RESULT_DIR / f"threads_{args.dataset}.json", {
        "dataset": args.dataset,
        "itemset_size": args.size,
        "n_itemsets": len(itemsets),
        "n_repeats": N_REPEATS,
        "timing_runs": args.timing_runs,
        "system_load_before_percent": load_before,
        "records": records,
    })


if __name__ == "__main__":
    main()
