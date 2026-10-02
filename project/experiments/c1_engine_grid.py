"""Experiment C1, part 2: the full project grid, run and timed with the engine.

For one log this runs, with seeded folds and one XGBoost model per fold:

* the project grid: 2 strategies (Apriori, IMPresseD) x 3 set sizes x 10 sets
  x 5 folds x ``--repeats`` repeats;
* the single-activity analysis: all activities x 5 folds x ``--repeats``;

each in three engine settings: faithful mode on the training fold (what the
original code computes), fixed mode on the training fold and fixed mode on
the held-out fold. Nothing is extrapolated here: the wall time of every
``compute`` call is measured.

Run:  python c1_engine_grid.py --dataset f1 [--repeats 10] [--fold-seed 0]
                                  [--n-jobs 1] [--tag _demo]

Writes results/experiments/C1/engine_grid_<dataset><tag>.json (timings) and
engine_grid_<dataset><tag>.csv (the importance values, a by-product).
"""

from __future__ import annotations

import argparse
import time
import warnings

import pandas as pd
from xgboost import XGBClassifier

from c1_common import (
    FOLD_SEED,
    N_FOLDS,
    RESULT_DIR,
    SIZES,
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
STRATEGIES = ("apriori", "impressed")


def say(text: str) -> None:
    """Progress line with a wall-clock stamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def timed_call(function, *args, **kwargs):
    """Call ``function``; return ``(table or None, LoadMeter dict, error)``.

    A ``ValueError`` is reported instead of raised: faithful mode refuses
    sets that no scored trace contains, exactly where the original crashes.
    """
    table, error = None, None
    with LoadMeter() as meter, warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            table = function(*args, **kwargs)
        except ValueError as problem:
            error = f"{type(problem).__name__}: {problem}"
    return table, meter.as_dict(), error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="f1")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--fold-seed", type=int, default=FOLD_SEED,
                        help="seed of the stratified 5-fold split")
    parser.add_argument("--tag", default="",
                        help="suffix of the output file names, e.g. _demo")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="threads of XGBoost (default: all logical cores)")
    args = parser.parse_args()

    load_before = sample_system_load()
    say(f"{args.dataset}: system load before the start {load_before:.1f} %")
    with LoadMeter() as whole_run:
        with LoadMeter() as meter:
            log = EventLog(DATASETS[args.dataset])
            encoder = IndexEncoder(log.traces.values(), log.activities)
        seconds_load = meter.as_dict()
        itemsets = {strategy: load_itemsets(log, args.dataset, strategy)
                    for strategy in STRATEGIES}
        folds = make_folds(log.labels, k=N_FOLDS, seed=args.fold_seed)
        # Faithful mode keeps the original's seed 2023. Fixed mode adds the
        # fold seed, so that two fold seeds do not reuse the same random
        # streams (the engine seeds by fold NUMBER, not by fold seed).
        engines = {
            label: LocationPermutationImportance(
                mode=mode, score_on=score_on, n_repeats=args.repeats,
                random_state=2023 + (args.fold_seed if mode == "fixed" else 0))
            for label, (mode, score_on) in VARIANTS.items()
        }

        fit_records, call_records, tables = [], [], []
        for fold, (train, test) in enumerate(folds):
            with LoadMeter() as meter:
                model = XGBClassifier(n_jobs=args.n_jobs).fit(
                    encoder.transform(log.traces[case] for case in train),
                    [log.labels[case] for case in train])
            fit_records.append(meter.as_dict())

            for label, engine in engines.items():
                for strategy in STRATEGIES:
                    for size in SIZES:
                        sets = itemsets[strategy][size]
                        table, load, error = timed_call(
                            engine.compute, model, log, encoder, train, test,
                            sets, fold=fold)
                        call_records.append({
                            "part": "grid", "variant": label, "strategy": strategy,
                            "size": size, "fold": fold,
                            "n_iterations": len(sets) * args.repeats,
                            "error": error, **load})
                        if table is not None:
                            table.insert(0, "size", size)
                            table.insert(0, "strategy", strategy)
                            table.insert(0, "variant", label)
                            tables.append(table)

                table, load, error = timed_call(
                    engine.compute_single_activities, model, log, encoder,
                    train, test, fold=fold)
                call_records.append({
                    "part": "single", "variant": label, "strategy": "all activities",
                    "size": 1, "fold": fold,
                    "n_iterations": len(log.activities) * args.repeats,
                    "error": error, **load})
                if table is not None:
                    table.insert(0, "size", 1)
                    table.insert(0, "strategy", "all activities")
                    table.insert(0, "variant", label)
                    tables.append(table)
            say(f"fold {fold} done")

    calls = pd.DataFrame(call_records)
    ok = calls[calls["error"].isna()]
    totals = (ok.groupby(["part", "variant"])["seconds"].sum().unstack("part")
              .to_dict("index"))
    summary = {
        "dataset": args.dataset,
        "fold_seed": args.fold_seed,
        "n_folds": N_FOLDS,
        "n_repeats": args.repeats,
        "xgboost_n_jobs": args.n_jobs,
        "n_activities": len(log.activities),
        "system_load_before_percent": load_before,
        "whole_run": whole_run.as_dict(),
        "seconds_load_log_and_build_encoder": seconds_load,
        "fit_per_fold": fit_records,
        "total_seconds_by_variant": totals,
        "calls": call_records,
    }
    stem = f"engine_grid_{args.dataset}{args.tag}"
    write_json(RESULT_DIR / f"{stem}.json", summary)
    pd.concat(tables).to_csv(RESULT_DIR / f"{stem}.csv", index=False,
                             float_format="%.10g")
    say(f"{args.dataset}: whole run {whole_run.seconds:.1f} s, totals {totals}, "
        f"errors: {calls['error'].dropna().str.slice(0, 120).tolist()}")


if __name__ == "__main__":
    main()
