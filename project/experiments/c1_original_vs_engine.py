"""Experiment C1, part 1: seconds per (itemset, repeat), original vs engine.

One fold (fold 0 of the seeded 5-fold split), for every set size 1, 2, 3 and
both selection strategies: the first ``--n-itemsets`` sets x ``--repeats``
repeats are run with

* the ORIGINAL ``itemset_permutation_importance`` (int-cast subclass, scored
  on the training fold as the original script does), timed per iteration via
  the progress line it prints;
* the engine in faithful mode on the training fold;
* the engine in fixed mode on the training fold and on the held-out fold.

All four use the same fitted model (XGBoost fitted on the original's training
frame). The script also times the one-off steps (loading, encoding, Apriori)
and one model fit per fold, and checks that faithful mode still returns the
original's importance values.

Run:  python c1_original_vs_engine.py --dataset f1 [--tag _run2]

Writes results/experiments/C1/one_fold_<dataset><tag>.json and
one_fold_<dataset><tag>_iterations.csv; progress goes to stdout.
"""

from __future__ import annotations

import argparse
import contextlib
import time
import warnings

import pandas as pd
from xgboost import XGBClassifier

import engine_reference as reference
from apriori_selector import AprioriSelector
from c1_common import (
    APRIORI_MIN_SUPPORT,
    FOLD_SEED,
    N_FOLDS,
    RESULT_DIR,
    SIZES,
    TOP_K,
    IterationClock,
    LoadMeter,
    count_cases_with,
    load_itemsets,
    sample_system_load,
    summarise,
    write_json,
)
from engine import (
    DATASETS,
    EventLog,
    IndexEncoder,
    LocationPermutationImportance,
    make_folds,
)

ENGINE_VARIANTS = {
    # label: (mode, score_on)
    "engine_faithful_train": ("faithful", "train"),
    "engine_fixed_train": ("fixed", "train"),
    "engine_fixed_test": ("fixed", "test"),
}


@contextlib.contextmanager
def original_output_to(clock: IterationClock):
    """Send the original's prints to ``clock`` and hide its pandas warnings."""
    with warnings.catch_warnings(), contextlib.redirect_stdout(clock):
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)
        warnings.simplefilter("ignore", FutureWarning)
        yield


def say(text: str) -> None:
    """Progress line with a wall-clock stamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def time_one_off_steps(path) -> tuple[dict, dict]:
    """Time loading and encoding in both implementations.

    Returns ``(objects, seconds)``; ``objects`` holds the original manager,
    its encoded frame, the engine's log and encoder.
    """
    seconds = {}
    with LoadMeter() as meter:
        manager = reference.make_original_manager(path)
    seconds["original DataManager()"] = meter.as_dict()
    with LoadMeter() as meter:
        encoded = reference.original_encoding(manager)
    seconds["original index_encoding(whole log) + int cast"] = meter.as_dict()
    with LoadMeter() as meter:
        log = EventLog(path)
    seconds["engine EventLog()"] = meter.as_dict()
    with LoadMeter() as meter:
        # 'hash' order = the original's column order inside this process, so
        # the model fitted on the original frame can be used by the engine.
        encoder = IndexEncoder(log.traces.values(), log.activities, "hash")
    seconds["engine IndexEncoder()"] = meter.as_dict()
    with LoadMeter() as meter:
        matrix = encoder.transform(log.traces.values())
    seconds["engine IndexEncoder.transform(whole log)"] = meter.as_dict()
    with LoadMeter() as meter:
        AprioriSelector(APRIORI_MIN_SUPPORT, max_len=max(SIZES), top_k=TOP_K).select(
            log.traces)
    seconds["Apriori per size (max_len 3, min_support 0.45)"] = meter.as_dict()
    objects = {"manager": manager, "encoded": encoded, "log": log,
               "encoder": encoder, "matrix": matrix}
    return objects, seconds


def time_model_fits(objects, folds) -> tuple[object, dict]:
    """Fit one XGBoost model per fold on both kinds of input and time it.

    Returns the fold-0 model fitted on the original frame (used by every
    importance run of this script) and the timing records.
    """
    manager, encoded = objects["manager"], objects["encoded"]
    log, matrix = objects["log"], objects["matrix"]
    row_of = {case: row for row, case in enumerate(log.case_ids)}
    records = {"fit on original DataFrame (int64)": [],
               "fit on engine matrix (int8)": [],
               "baseline predict, original DataFrame, training fold": [],
               "baseline predict, engine matrix, training fold": []}
    first_model = None
    for fold, (train, _) in enumerate(folds):
        features, labels = reference.original_fold_data(manager, encoded, train)
        with LoadMeter() as meter:
            model = XGBClassifier().fit(features, labels)
        records["fit on original DataFrame (int64)"].append(meter.as_dict())
        with LoadMeter() as meter:
            model.predict(features)
        records["baseline predict, original DataFrame, training fold"].append(
            meter.as_dict())
        first_model = first_model or model

        rows = [row_of[case] for case in train]
        train_matrix, train_labels = matrix[rows], [log.labels[c] for c in train]
        with LoadMeter() as meter:
            engine_model = XGBClassifier().fit(train_matrix, train_labels)
        records["fit on engine matrix (int8)"].append(meter.as_dict())
        with LoadMeter() as meter:
            engine_model.predict(train_matrix)
        records["baseline predict, engine matrix, training fold"].append(meter.as_dict())
        say(f"fold {fold}: fits timed")
    return first_model, records


def run_original(objects, model, train, itemsets, repeats) -> dict:
    """Run the original itemset routine and time every (itemset, repeat)."""
    manager = objects["manager"]
    features, labels = reference.original_fold_data(manager, objects["encoded"], train)
    clock = IterationClock()
    error, result = None, None
    with LoadMeter() as meter:
        try:
            with original_output_to(clock):
                clock.restart()
                result = manager.itemset_permutation_importance(
                    model, features, labels, train, dict(enumerate(itemsets)),
                    constrain=True, n_repeats=repeats)
        except Exception as problem:  # the original can crash on rare sets
            error = f"{type(problem).__name__}: {problem}"
    return {"durations": clock.durations(), "load": meter.as_dict(),
            "error": error, "result": result}


def run_engine(objects, model, train, test, itemsets, repeats, timing_runs) -> dict:
    """Time ``compute`` of every engine variant ``timing_runs`` times."""
    log, encoder = objects["log"], objects["encoder"]
    n_iterations = len(itemsets) * repeats
    outcome = {}
    for label, (mode, score_on) in ENGINE_VARIANTS.items():
        engine = LocationPermutationImportance(
            mode=mode, score_on=score_on, n_repeats=repeats, random_state=2023)
        runs, loads, table, error = [], [], None, None
        for _ in range(timing_runs):
            try:
                with LoadMeter() as meter, warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    table = engine.compute(model, log, encoder, train, test,
                                           itemsets, fold=0)
            except ValueError as problem:  # faithful mode refuses absent sets
                error = f"{type(problem).__name__}: {problem}"
                break
            runs.append(meter.seconds / n_iterations)
            loads.append(meter.as_dict())
        outcome[label] = {"seconds_per_iteration_runs": runs, "loads": loads,
                          "table": table, "error": error}
    return outcome


def max_abs_difference(original_result, faithful_table) -> float | None:
    """Largest |original importance - faithful importance| (None if no data)."""
    if original_result is None or faithful_table is None:
        return None
    differences = [
        abs(original_result[row.itemset_id].iloc[row.repeat] - row.importance)
        for row in faithful_table.itertuples()
    ]
    return float(max(differences))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="f1")
    parser.add_argument("--strategies", nargs="+", default=["apriori", "impressed"],
                        choices=["apriori", "impressed"])
    parser.add_argument("--n-itemsets", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timing-runs", type=int, default=5,
                        help="how often each engine variant is timed")
    parser.add_argument("--tag", default="",
                        help="suffix of the output file names, e.g. _run2")
    args = parser.parse_args()

    say(f"{args.dataset}: system load before the start "
        f"{sample_system_load():.1f} % of 12 logical cores")
    objects, one_off = time_one_off_steps(DATASETS[args.dataset])
    log = objects["log"]
    folds = make_folds(log.labels, k=N_FOLDS, seed=FOLD_SEED)
    train, test = folds[0]
    model, fit_records = time_model_fits(objects, folds)
    train_traces = [log.traces[case] for case in train]

    groups, iteration_rows = [], []
    for strategy in args.strategies:
        selected = load_itemsets(log, args.dataset, strategy)
        for size in SIZES:
            itemsets = selected[size][:args.n_itemsets]
            say(f"{strategy} size {size}: original, {len(itemsets)} sets x "
                f"{args.repeats} repeats ...")
            original = run_original(objects, model, train, itemsets, args.repeats)
            engine = run_engine(objects, model, train, test, itemsets,
                                args.repeats, args.timing_runs)
            group = {
                "strategy": strategy,
                "size": size,
                "itemsets": itemsets,
                "train_cases_with_itemset": [
                    count_cases_with(train_traces, items) for items in itemsets],
                "original": {
                    "seconds_per_iteration": summarise(original["durations"]),
                    "durations": original["durations"],
                    "load": original["load"],
                    "error": original["error"],
                },
                "faithful_vs_original_max_abs_diff": max_abs_difference(
                    original["result"], engine["engine_faithful_train"]["table"]),
            }
            for label, outcome in engine.items():
                group[label] = {
                    "seconds_per_iteration": summarise(
                        outcome["seconds_per_iteration_runs"]),
                    "runs": outcome["seconds_per_iteration_runs"],
                    "loads": outcome["loads"],
                    "error": outcome["error"],
                }
            groups.append(group)
            for number, seconds in enumerate(original["durations"]):
                iteration_rows.append({
                    "dataset": args.dataset, "strategy": strategy, "size": size,
                    "itemset": ", ".join(sorted(itemsets[number // args.repeats])),
                    "repeat": number % args.repeats, "seconds_original": seconds,
                })
            medians = {label: group[label]["seconds_per_iteration"]["median"]
                       for label in ENGINE_VARIANTS}
            say(f"  original median {group['original']['seconds_per_iteration']['median']}"
                f" s (others' load {original['load']['others_percent']:.1f} %), "
                f"engine medians {medians}, "
                f"diff {group['faithful_vs_original_max_abs_diff']}, "
                f"errors {original['error']} / "
                f"{[engine[label]['error'] for label in ENGINE_VARIANTS]}")

    summary = {
        "dataset": args.dataset,
        "fold_seed": FOLD_SEED,
        "train_cases": len(train),
        "test_cases": len(test),
        "n_activities": len(log.activities),
        "n_features": objects["encoder"].n_features,
        "n_itemsets_per_group": args.n_itemsets,
        "n_repeats": args.repeats,
        "engine_timing_runs": args.timing_runs,
        "one_off_seconds": one_off,
        "per_fold_seconds": fit_records,
        "groups": groups,
        "system_load_at_end_percent": sample_system_load(),
    }
    stem = f"one_fold_{args.dataset}{args.tag}"
    write_json(RESULT_DIR / f"{stem}.json", summary)
    pd.DataFrame(iteration_rows).to_csv(
        RESULT_DIR / f"{stem}_iterations.csv", index=False)
    say(f"{args.dataset}: done")


if __name__ == "__main__":
    main()
