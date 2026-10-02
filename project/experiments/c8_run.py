"""Experiment C8, part 1: compute the importances in four settings.

Question: how much do the original code's behaviours (accumulating
permutations, the bookkeeping bug of ``shuffle_sequence``, scoring on the
training fold) change the result of the 2024 pipeline?

For the ORIGINAL Apriori itemsets of one log (``original_top10``: sets of
size > 1, the 10 most frequent) this script computes the location permutation
importance in four settings. Inside one fold seed all four settings use the
same 5 folds and the same fitted model per fold:

A  faithful, training fold            what the original code computes
B  fixed, training fold               only the permutation is repaired
C  fixed, held-out fold               permutation repaired + held-out scoring
D  faithful, training fold, itemsets processed in REVERSE order
                                      (shows the order dependence that
                                      accumulation causes)

Run (from the ``experiments`` folder, with the project's interpreter):

    python c8_run.py --dataset f1
    python c8_run.py --dataset f3 --min-support 0.49 --fold-seeds 0 1 2

Writes to ``results/experiments/C8/``:

values_<dataset>.csv   one row per setting x fold seed x fold x itemset x repeat
models_<dataset>.csv   training and held-out weighted F1 of every fitted model
run_<dataset>.json     settings, itemsets, timings

``c8_report.py`` turns these files into tables and figures.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from apriori_selector import original_top10
from engine import (
    DATASETS,
    EventLog,
    IndexEncoder,
    LocationPermutationImportance,
    make_folds,
    weighted_f1,
)

RESULT_DIR = Path(__file__).resolve().parents[1] / "results" / "experiments" / "C8"
SEED = 2023          # the original's permutation seed (tools.py:499)
N_FOLDS = 5
TOP_K = 10
# The original default is 0.5. On f3 that gives only 5 itemsets; 0.49 is the
# largest support of the C7 grid that gives the 10 itemsets of the paper's
# figure (every value <= 0.494 gives the same 10, see C7/RESULT.md 3.3).
DEFAULT_MIN_SUPPORT = {"f1": 0.5, "f2": 0.5, "f3": 0.49}
SETTINGS = {
    # label: (engine mode, scored fold, process the itemsets in reverse order)
    "A_faithful_train": ("faithful", "train", False),
    "B_fixed_train": ("fixed", "train", False),
    "C_fixed_test": ("fixed", "test", False),
    "D_faithful_train_reversed": ("faithful", "train", True),
}
VALUE_COLUMNS = [
    "setting", "fold_seed", "fold", "itemset_id", "itemset", "position", "repeat",
    "baseline", "permuted", "importance", "n_traces_with_itemset",
    "n_traces_changed", "n_traces_unchanged_infeasible",
]


def select_itemsets(log: EventLog, min_support: float) -> dict[int, list[str]]:
    """The original's itemsets as ``{rank (0 = most frequent): activities}``.

    The dict order is the order in which the original code processes the
    itemsets; it matters in faithful mode.
    """
    candidates, _ = original_top10(log.traces, min_support, TOP_K)
    return {rank: sorted(items) for rank, items in enumerate(candidates)}


def fit_model(log, encoder, train_cases, n_jobs: int) -> XGBClassifier:
    """The original's classifier (default XGBoost) fitted on the training fold."""
    model = XGBClassifier(n_jobs=n_jobs)
    model.fit(encoder.transform(log.traces[case] for case in train_cases),
              [log.labels[case] for case in train_cases])
    return model


def model_quality(model, log, encoder, cases) -> float:
    """Weighted F1 of ``model`` on ``cases``."""
    predicted = model.predict(encoder.transform(log.traces[case] for case in cases))
    return weighted_f1([log.labels[case] for case in cases], predicted)


def compute_setting(label, model, log, encoder, train, test, itemsets,
                    fold_seed: int, fold: int, n_repeats: int) -> pd.DataFrame:
    """Importance table of one setting for one fold."""
    mode, score_on, reverse = SETTINGS[label]
    ordered = dict(reversed(itemsets.items())) if reverse else itemsets
    # Faithful mode keeps the original's seed. Fixed mode derives its random
    # streams from (seed, fold number, itemset, repeat), so the fold seed is
    # added to keep the streams of two fold seeds apart.
    random_state = SEED if mode == "faithful" else SEED + fold_seed
    table = LocationPermutationImportance(
        mode=mode, score_on=score_on, n_repeats=n_repeats, random_state=random_state,
    ).compute(model, log, encoder, train, test, ordered, fold=fold)
    position = {itemset_id: index + 1 for index, itemset_id in enumerate(ordered)}
    table["position"] = table["itemset_id"].map(position)
    table["setting"] = label
    table["fold_seed"] = fold_seed
    return table[VALUE_COLUMNS]


def run_fold_seed(log, encoder, itemsets, fold_seed: int, n_repeats: int,
                  n_jobs: int, seconds: dict) -> tuple[pd.DataFrame, list[dict]]:
    """All settings for the 5 folds of one fold seed.

    Adds the time spent per setting (and on model fitting) to ``seconds``.
    Returns the importance rows and one quality record per fitted model.
    """
    tables, models = [], []
    folds = make_folds(log.labels, k=N_FOLDS, seed=fold_seed)
    for fold, (train, test) in enumerate(folds):
        start = time.perf_counter()
        model = fit_model(log, encoder, train, n_jobs)
        seconds["model_fits"] += time.perf_counter() - start
        models.append({
            "fold_seed": fold_seed, "fold": fold,
            "n_train": len(train), "n_test": len(test),
            "train_f1": model_quality(model, log, encoder, train),
            "test_f1": model_quality(model, log, encoder, test),
        })
        for label in SETTINGS:
            start = time.perf_counter()
            tables.append(compute_setting(label, model, log, encoder, train, test,
                                          itemsets, fold_seed, fold, n_repeats))
            seconds[label] += time.perf_counter() - start
    return pd.concat(tables, ignore_index=True), models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="f1")
    parser.add_argument("--min-support", type=float, default=None,
                        help="Apriori support (default: 0.5; 0.49 on f3)")
    parser.add_argument("--fold-seeds", type=int, nargs="+", default=list(range(10)),
                        help="seeds of the stratified 5-fold split (default 0..9)")
    parser.add_argument("--repeats", type=int, default=10,
                        help="permutation repeats per itemset (original: 10)")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="XGBoost threads (1 gives the same model, see C1)")
    args = parser.parse_args()
    min_support = args.min_support or DEFAULT_MIN_SUPPORT[args.dataset]

    log = EventLog(DATASETS[args.dataset])
    encoder = IndexEncoder(log.traces.values(), log.activities)
    itemsets = select_itemsets(log, min_support)
    print(f"{args.dataset}: {len(log.traces)} cases, {len(itemsets)} itemsets at "
          f"min_support {min_support}", flush=True)

    seconds = {"model_fits": 0.0, **{label: 0.0 for label in SETTINGS}}
    tables, models = [], []
    start = time.perf_counter()
    for fold_seed in args.fold_seeds:
        table, quality = run_fold_seed(log, encoder, itemsets, fold_seed,
                                       args.repeats, args.n_jobs, seconds)
        tables.append(table)
        models.extend(quality)
        print(f"fold seed {fold_seed} done ({time.perf_counter() - start:.0f} s)",
              flush=True)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(
        RESULT_DIR / f"values_{args.dataset}.csv", index=False, float_format="%.17g")
    pd.DataFrame(models).to_csv(
        RESULT_DIR / f"models_{args.dataset}.csv", index=False, float_format="%.17g")
    run_info = {
        "dataset": args.dataset,
        "min_support": min_support,
        "itemsets": {str(rank): items for rank, items in itemsets.items()},
        "fold_seeds": args.fold_seeds,
        "n_folds": N_FOLDS,
        "n_repeats": args.repeats,
        "xgboost_n_jobs": args.n_jobs,
        "permutation_seed": SEED,
        "seconds_total": time.perf_counter() - start,
        "seconds_per_part": seconds,
    }
    (RESULT_DIR / f"run_{args.dataset}.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8")
    print(json.dumps(run_info["seconds_per_part"], indent=2))


if __name__ == "__main__":
    main()
