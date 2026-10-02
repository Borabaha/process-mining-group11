"""Experiment C8, supplement: separate the shuffle defects from the accumulation.

Setting A (faithful) and setting B (fixed, training fold) of ``c8_run.py``
differ in two things at once: A uses the original shuffle with its defects
(index bookkeeping bug, positions not clipped to the trace length, sequential
moves) AND lets the permutations accumulate. Two more settings complete the
2 x 2 table, all scored on the training fold:

                      permutations accumulate    every repeat starts from
                      (one working copy)         the unpermuted traces
    original shuffle  A  (c8_run.py)             E  (this script)
    fixed shuffle     F  (this script)           B  (c8_run.py)

    A against E, F against B  = what accumulation does
    A against F, E against B  = what the defects of the shuffle do

E keeps the rest of the original: one legacy random stream seeded with 2023
per fold and the original's re-encoding of the shuffled subset. The first
iteration of a fold (first itemset, repeat 0) is therefore the same
computation as in setting A and must give the same value (``c8_report.py``
checks this). F keeps the rest of fixed mode (correct encoding, NumPy
Generator) and, like the original, scores an (itemset, repeat) on the baseline
predictions with only the traces of the current itemset replaced.

Run:  python c8_supplement.py --dataset f1 [--settings E F]

Uses the itemsets, fold seeds, repeats and model settings of ``c8_run.py``
(run that first: its ``run_<dataset>.json`` is read) and writes
``results/experiments/C8/values_E_<dataset>.csv`` and ``values_F_<dataset>.csv``
with the columns of ``values_<dataset>.csv``.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from c8_run import N_FOLDS, RESULT_DIR, SEED, VALUE_COLUMNS, fit_model
from engine import (
    DATASETS,
    EventLog,
    IndexEncoder,
    find_occurrences,
    itemset_label,
    make_folds,
    permute_trace_fixed,
    shuffle_sequence_faithful,
    weighted_f1,
)

SETTINGS = {
    # key: (label, shuffle, permutations accumulate)
    "E": ("E_original_shuffle_no_accumulation", "original", False),
    "F": ("F_fixed_shuffle_with_accumulation", "fixed", True),
}


def shuffle_trace(trace, items, allowed, rng, shuffle: str) -> list[str]:
    """One permutation of ``items`` in ``trace`` with the chosen shuffle.

    ``shuffle='original'`` needs a ``numpy.random.RandomState``,
    ``shuffle='fixed'`` a ``numpy.random.Generator``. The fixed shuffle
    returns the trace unchanged when it has no feasible placement.
    """
    if shuffle == "original":
        return shuffle_sequence_faithful(trace, items, allowed, rng)
    new_trace, _ = permute_trace_fixed(
        trace, find_occurrences(trace, items), allowed, rng)
    return list(trace) if new_trace is None else new_trace


def supplement_importance(model, log, encoder, train_cases, itemsets,
                          n_repeats: int, shuffle: str, accumulate: bool,
                          rng) -> list[dict]:
    """Training-fold importance with a chosen shuffle, with or without accumulation.

    Returns one record per (itemset, repeat) with the columns of
    ``LocationPermutationImportance.compute`` that apply here.
    """
    original = [log.traces[case] for case in train_cases]
    y_true = [log.labels[case] for case in train_cases]
    base_pred = model.predict(encoder.transform(original))
    baseline = weighted_f1(y_true, base_pred)
    working = [list(trace) for trace in original]
    records = []
    for position, (itemset_id, items) in enumerate(itemsets.items(), start=1):
        items = set(items)
        # The original's rule for "this trace is permuted" (tools.py:521-523).
        rows = [row for row, trace in enumerate(original)
                if items.issubset(trace) and len(items) != len(trace)]
        for repeat in range(n_repeats):
            source = working if accumulate else original
            shuffled = [shuffle_trace(source[row], items, log.allowed_locations,
                                      rng, shuffle) for row in rows]
            if accumulate:
                for row, trace in zip(rows, shuffled):
                    working[row] = trace
            # The original re-encodes only the shuffled cases, so its padding
            # stops at the longest shuffled trace (tools.py:529-531).
            pad_until = (max(len(trace) for trace in shuffled)
                         if shuffle == "original" else None)
            pred = base_pred.copy()
            pred[rows] = model.predict(encoder.transform(shuffled, pad_until=pad_until))
            permuted = weighted_f1(y_true, pred)
            records.append({
                "itemset_id": itemset_id,
                "itemset": itemset_label(items),
                "position": position,
                "repeat": repeat,
                "baseline": baseline,
                "permuted": permuted,
                "importance": baseline - permuted,
                "n_traces_with_itemset": len(rows),
                "n_traces_changed": sum(
                    new != original[row] for new, row in zip(shuffled, rows)),
                "n_traces_unchanged_infeasible": 0,
            })
    return records


def make_rng(shuffle: str, fold_seed: int, fold: int):
    """The random stream of one fold (legacy stream for the original shuffle)."""
    if shuffle == "original":
        return np.random.RandomState(SEED)  # = np.random.seed(2023), tools.py:499
    return np.random.default_rng([SEED + fold_seed, fold])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="f1")
    parser.add_argument("--settings", nargs="+", choices=sorted(SETTINGS),
                        default=sorted(SETTINGS))
    args = parser.parse_args()

    run_info = json.loads(
        (RESULT_DIR / f"run_{args.dataset}.json").read_text(encoding="utf-8"))
    itemsets = {int(rank): items for rank, items in run_info["itemsets"].items()}
    log = EventLog(DATASETS[args.dataset])
    encoder = IndexEncoder(log.traces.values(), log.activities)

    start = time.perf_counter()
    tables = {key: [] for key in args.settings}
    for fold_seed in run_info["fold_seeds"]:
        folds = make_folds(log.labels, k=N_FOLDS, seed=fold_seed)
        for fold, (train, _test) in enumerate(folds):
            model = fit_model(log, encoder, train, run_info["xgboost_n_jobs"])
            for key in args.settings:
                label, shuffle, accumulate = SETTINGS[key]
                table = pd.DataFrame.from_records(supplement_importance(
                    model, log, encoder, train, itemsets, run_info["n_repeats"],
                    shuffle, accumulate, make_rng(shuffle, fold_seed, fold)))
                table["setting"] = label
                table["fold_seed"] = fold_seed
                table["fold"] = fold
                tables[key].append(table[VALUE_COLUMNS])
        print(f"fold seed {fold_seed} done ({time.perf_counter() - start:.0f} s)",
              flush=True)
    for key, parts in tables.items():
        pd.concat(parts, ignore_index=True).to_csv(
            RESULT_DIR / f"values_{key}_{args.dataset}.csv", index=False,
            float_format="%.17g")
    print(f"total {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
