"""Experiment C9, part 1: importance values for many fold seeds (runner).

For every fold seed this script splits the log into stratified folds, fits
one XGBoost model per fold and computes the location permutation importance
of every activity set with ``engine.py`` in up to three settings:

* ``fixed_test``      fixed mode, scored on the held-out fold
* ``fixed_train``     fixed mode, scored on the training fold
* ``faithful_train``  faithful mode on the training fold, i.e. what the
                      original code computes (optional, ``--faithful-repeats``)

Seeds
-----
``fold seed``         seed of ``StratifiedKFold`` (which cases form a fold).
``permutation seed``  ``PERMUTATION_SEED + fold seed`` in fixed mode, so two
                      fold seeds never share a random stream. Faithful mode
                      keeps the original's constant 2023.

Fixed mode gives every (fold, itemset, repeat) its own random generator.
Repeat ``r`` of a run with 30 repeats is therefore the same value as repeat
``r`` of a run with 5 repeats, and the analysis can study "3, 5 or 10
repeats" by taking subsets of one long run. This does not hold in faithful
mode (permutations accumulate), which is why it has its own repeat count.

Run (one process handles a range of fold seeds; start several in parallel):

    python c9_seed_stability.py --dataset f1 --first-seed 0 --n-seeds 5

Writes, per fold seed,
``results/experiments/C9/raw/<dataset>_<itemsets>_k<folds>/seed_<seed>.csv``
(one row per setting x fold x itemset x repeat) and ``seed_<seed>.json``
(timings). ``c9_analysis.py`` turns these files into the tables.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from apriori_selector import original_top10
from c1_common import load_itemsets as load_project_itemsets
from engine import (
    DATASETS,
    EventLog,
    IndexEncoder,
    LocationPermutationImportance,
    make_folds,
)

RESULT_DIR = Path(__file__).resolve().parents[1] / "results" / "experiments" / "C9"
PERMUTATION_SEED = 2023
# Support at which the ORIGINAL selection returns 10 itemsets (experiment C7:
# 0.5 for f1 and f2 as in the paper; f3 has only 5 itemsets at 0.5).
ORIGINAL_MIN_SUPPORT = {"f1": 0.5, "f2": 0.5, "f3": 0.49}
SETTINGS = {
    # label: (engine mode, scored fold)
    "fixed_test": ("fixed", "test"),
    "fixed_train": ("fixed", "train"),
    "faithful_train": ("faithful", "train"),
}
KEPT_COLUMNS = ["setting", "seed", "fold", "itemset_id", "itemset", "repeat",
                "baseline", "importance", "n_traces_with_itemset",
                "n_traces_changed"]


def study_dir(dataset: str, itemsets: str, n_folds: int) -> Path:
    """Folder that holds the per-seed files of one study."""
    return RESULT_DIR / "raw" / f"{dataset}_{itemsets}_k{n_folds}"


def select_itemsets(log: EventLog, dataset: str, which: str) -> dict:
    """The activity sets of the study as ``{itemset id: [activities]}``.

    ``which='original'``: the 10 itemsets the original code selects (Apriori,
    size > 1, most frequent first), ids 0..9 in the original's order.
    ``which='project'``: the working selection of the project, 10 sets per
    size 1-3 for Apriori (experiment C7) and for IMPresseD (experiment C4),
    ids like ``'apriori|2|03'`` (strategy | size | rank).
    """
    if which == "original":
        sets, _ = original_top10(log.traces, ORIGINAL_MIN_SUPPORT[dataset], top_k=10)
        return {number: sorted(items) for number, items in enumerate(sets)}
    if which == "project":
        return {
            f"{strategy}|{size}|{rank:02d}": items
            for strategy in ("apriori", "impressed")
            for size, sets in load_project_itemsets(log, dataset, strategy).items()
            for rank, items in enumerate(sets, start=1)
        }
    raise ValueError("which must be 'original' or 'project'")


def make_engines(seed: int, repeats: int, faithful_repeats: int) -> dict:
    """One engine per setting for fold seed ``seed`` (see module docstring)."""
    engines = {}
    for label, (mode, score_on) in SETTINGS.items():
        n_repeats = faithful_repeats if mode == "faithful" else repeats
        if n_repeats == 0:
            continue
        random_state = PERMUTATION_SEED + (seed if mode == "fixed" else 0)
        engines[label] = LocationPermutationImportance(
            mode=mode, score_on=score_on, n_repeats=n_repeats,
            random_state=random_state)
    return engines


def run_seed(log, encoder, itemsets, seed, n_folds, engines, n_jobs):
    """All settings for all folds of one fold seed.

    Returns ``(rows, seconds)``: the tidy value table and a dict with the
    seconds spent on model fitting and on every setting.
    """
    tables = []
    seconds = {"fit": 0.0, **{label: 0.0 for label in engines}}
    folds = make_folds(log.labels, k=n_folds, seed=seed)
    for fold, (train, test) in enumerate(folds):
        start = time.perf_counter()
        model = XGBClassifier(n_jobs=n_jobs).fit(
            encoder.transform(log.traces[case] for case in train),
            [log.labels[case] for case in train])
        seconds["fit"] += time.perf_counter() - start

        for label, engine in engines.items():
            start = time.perf_counter()
            with warnings.catch_warnings():
                # Fixed mode warns when a set occurs in no scored trace; the
                # analysis sees this in the column n_traces_with_itemset.
                warnings.simplefilter("ignore", UserWarning)
                table = engine.compute(model, log, encoder, train, test,
                                       itemsets, fold=fold)
            seconds[label] += time.perf_counter() - start
            table.insert(0, "setting", label)
            table.insert(1, "seed", seed)
            tables.append(table[KEPT_COLUMNS])
    return pd.concat(tables, ignore_index=True), seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="f1")
    parser.add_argument("--itemsets", choices=["original", "project"],
                        default="original")
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30,
                        help="permutation repeats in fixed mode")
    parser.add_argument("--faithful-repeats", type=int, default=10,
                        help="repeats in faithful mode (0 = skip faithful mode)")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="threads of XGBoost (1 = same values, see C1)")
    args = parser.parse_args()

    log = EventLog(DATASETS[args.dataset])
    encoder = IndexEncoder(log.traces.values(), log.activities)
    itemsets = select_itemsets(log, args.dataset, args.itemsets)
    out_dir = study_dir(args.dataset, args.itemsets, args.folds)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.dataset}: {len(log.traces)} cases, {len(itemsets)} itemsets, "
          f"{args.folds} folds, fold seeds {args.first_seed}.."
          f"{args.first_seed + args.n_seeds - 1}", flush=True)

    for seed in range(args.first_seed, args.first_seed + args.n_seeds):
        engines = make_engines(seed, args.repeats, args.faithful_repeats)
        start = time.perf_counter()
        rows, seconds = run_seed(log, encoder, itemsets, seed, args.folds,
                                 engines, args.n_jobs)
        rows.to_csv(out_dir / f"seed_{seed}.csv", index=False,
                    float_format="%.17g")
        info = {
            "dataset": args.dataset, "itemsets": args.itemsets, "seed": seed,
            "n_folds": args.folds, "n_itemsets": len(itemsets),
            "repeats": {label: engine.n_repeats
                        for label, engine in engines.items()},
            "permutation_seed": {label: engine.random_state
                                 for label, engine in engines.items()},
            "xgboost_n_jobs": args.n_jobs,
            "seconds": seconds,
            "seconds_total": time.perf_counter() - start,
        }
        (out_dir / f"seed_{seed}.json").write_text(
            json.dumps(info, indent=2), encoding="utf-8")
        print(f"[{time.strftime('%H:%M:%S')}] seed {seed}: {len(rows)} rows, "
              f"{info['seconds_total']:.0f} s "
              f"({', '.join(f'{k} {v:.0f}' for k, v in seconds.items())})",
              flush=True)


if __name__ == "__main__":
    main()
