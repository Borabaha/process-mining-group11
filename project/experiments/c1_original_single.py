"""Experiment C1, part 3: runtime of the ORIGINAL single-activity routine.

Runs the original ``trace_permutation_importance`` (constrain=True, int-cast
subclass) on the training fold of fold 0 with ONE repeat and records the time
of every activity, without touching the original code:

* a thin wrapper around the model timestamps every ``predict`` call (one per
  scored activity);
* the routine's own line "No shuffled cases. skipped!" is timestamped for the
  activities that occur in fewer than 3 training cases;
* activities with a single allowed position are skipped by the original
  before any work is done (time 0).

Run:  python c1_original_single.py --dataset f1 [--max-iterations 0]

``--max-iterations N`` stops after N timed activities (0 = all activities).
Writes results/experiments/C1/original_single_<dataset>.csv (one row per
activity) and original_single_<dataset>.json (summary).
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
import warnings

import pandas as pd
from xgboost import XGBClassifier

import engine_reference as reference
from c1_common import (
    FOLD_SEED,
    N_FOLDS,
    RESULT_DIR,
    IterationClock,
    LoadMeter,
    StopTiming,
    sample_system_load,
    summarise,
    write_json,
)
from engine import DATASETS, EventLog, make_folds

GUARD_LINE = "No shuffled cases"


class TimedModel:
    """Model wrapper that tells the clock when a prediction is finished.

    The first ``predict`` call of the original routine is the baseline; it
    restarts the clock. Every later call ends one scored activity.
    """

    def __init__(self, model, clock: IterationClock):
        self._model = model
        self._clock = clock
        self._baseline_done = False

    def predict(self, features):
        """Predict with the wrapped model and timestamp the call."""
        predicted = self._model.predict(features)
        if self._baseline_done:
            self._clock.mark("scored")
        else:
            self._baseline_done = True
            self._clock.restart()
        return predicted


class ProgressClock(IterationClock):
    """An ``IterationClock`` that reports every 20th event on stderr.

    Stdout is redirected to the clock while the original runs, so stderr is
    the only channel left for progress.
    """

    def mark(self, kind: str) -> None:
        if (len(self.events) + 1) % 20 == 0:
            elapsed = time.perf_counter() - self.start
            print(f"[{time.strftime('%H:%M:%S')}] {len(self.events) + 1} "
                  f"activities timed, {elapsed:.0f} s", file=sys.stderr, flush=True)
        super().mark(kind)


def say(text: str) -> None:
    """Progress line with a wall-clock stamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def activity_counts(log: EventLog, cases) -> pd.DataFrame:
    """Per activity: training cases containing it and its number of events."""
    n_cases = {act: 0 for act in log.activities}
    n_events = {act: 0 for act in log.activities}
    for case in cases:
        trace = log.traces[case]
        for act in trace:
            n_events[act] += 1
        for act in set(trace):
            n_cases[act] += 1
    return pd.DataFrame({
        "activity": log.activities,
        "n_train_cases": [n_cases[act] for act in log.activities],
        "n_train_events": [n_events[act] for act in log.activities],
        "n_allowed_positions": [len(log.allowed_locations[act])
                                for act in log.activities],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="f1")
    parser.add_argument("--max-iterations", type=int, default=0,
                        help="stop after this many timed activities (0 = all)")
    args = parser.parse_args()
    path = DATASETS[args.dataset]

    load_before = sample_system_load()
    say(f"{args.dataset}: system load before the start {load_before:.1f} %")
    manager = reference.make_original_manager(path)
    encoded = reference.original_encoding(manager)
    log = EventLog(path)
    train, _ = make_folds(log.labels, k=N_FOLDS, seed=FOLD_SEED)[0]
    features, labels = reference.original_fold_data(manager, encoded, train)
    model = XGBClassifier().fit(features, labels)

    # The original walks the activities in order of first appearance, which
    # is EventLog.activities; activities with one allowed position cost nothing.
    table = activity_counts(log, train)
    if table["activity"].tolist() != manager.data[manager.activity].unique().tolist():
        raise RuntimeError("activity order differs from the original")
    table["kind"] = "instant (one allowed position)"
    table["seconds"] = 0.0
    timed_rows = table.index[table["n_allowed_positions"] >= 2].tolist()

    clock = ProgressClock(markers=(GUARD_LINE,),
                          max_events=args.max_iterations or None)
    say(f"running the original on {len(train)} training cases, "
        f"{len(timed_rows)} activities to time ...")
    with LoadMeter() as meter:
        with warnings.catch_warnings(), contextlib.redirect_stdout(clock):
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)
            warnings.simplefilter("ignore", FutureWarning)
            with contextlib.suppress(StopTiming):
                manager.trace_permutation_importance(
                    TimedModel(model, clock), features, labels, train,
                    constrain=True, n_repeats=1)

    durations = clock.durations()
    done = timed_rows[:len(durations)]
    table.loc[done, "seconds"] = durations
    table.loc[done, "kind"] = [
        "skipped (fewer than 3 cases)" if kind == GUARD_LINE else "scored"
        for kind, _ in clock.events
    ]
    if len(done) < len(timed_rows):  # stopped early: keep what was reached
        table = table.loc[:done[-1]] if done else table.iloc[:0]
    # Consistency: the "fewer than 3 cases" rule must match the case counts.
    guard = table["kind"] == "skipped (fewer than 3 cases)"
    timed = table["n_allowed_positions"] >= 2
    if not (guard == (timed & (table["n_train_cases"] < 3))).all():
        raise RuntimeError("timestamps do not line up with the activity list")

    by_kind = {kind: summarise(group["seconds"])
               for kind, group in table.groupby("kind")}
    summary = {
        "dataset": args.dataset,
        "fold_seed": FOLD_SEED,
        "train_cases": len(train),
        "n_activities": len(log.activities),
        "n_repeats": 1,
        "complete": len(durations) == len(timed_rows),
        "activities_covered": len(table),
        "seconds_sum_measured": float(table["seconds"].sum()),
        "seconds_per_activity_all": summarise(table["seconds"]),
        "seconds_per_activity_by_kind": by_kind,
        "system_load_before_percent": load_before,
        "load_during_run": meter.as_dict(),
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    table.insert(0, "dataset", args.dataset)
    table.to_csv(RESULT_DIR / f"original_single_{args.dataset}.csv", index=False)
    write_json(RESULT_DIR / f"original_single_{args.dataset}.json", summary)
    say(f"{args.dataset}: {len(table)} activities, "
        f"{summary['seconds_sum_measured']:.1f} s in total, "
        f"others' load {meter.others_percent:.1f} %")


if __name__ == "__main__":
    main()
