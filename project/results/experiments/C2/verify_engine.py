"""Independent verification of experiments/engine.py (experiment C2).

Written by the verifier, not by the builder of engine.py. Nothing here reuses
test_engine.py or engine_reference.py: the original 2024 code is loaded by a
loader defined in this file, the log is re-read with a second, independent
loader, the encoding is re-done with a naive encoder and every constraint is
re-checked with code written here.

Parts (``--part``):

encoder   IndexEncoder vs the original ``DataManager.index_encoding``
          (int cast) on f1, f2, f3: same rows, columns, order and values.
faithful  faithful mode vs the original ``itemset_permutation_importance``
          on a fold and itemsets the builder did not use (``--dataset``).
fixed     fixed mode: the two constraints of the paper (Section 3.2) and the
          other guarantees, on f1 and f3, sizes 1, 2 and 3.
edge      edge cases a student will hit (no trace changed, absent itemset).
grid      re-run the builder's full grid and compare with his CSV files.
digest    print a hash of small result tables (used by ``hashseed``).
hashseed  run ``digest`` in two processes with different PYTHONHASHSEED.

Run, with the project's Python 3.12 venv:

    python verify_engine.py --part encoder
    python verify_engine.py --part faithful --dataset f1
    python verify_engine.py --part fixed

Each part writes ``verify_<part>[_<dataset>].json`` next to this file.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import hashlib
import importlib.util
import io
import itertools
import json
import os
import random
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
ORIGINAL_REPO = PROJECT_ROOT / "external" / "PermutationLocationImportance"
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

import engine  # noqa: E402  (the module under test)

CASE = "case:concept:name"
ACT = "concept:name"
LABEL = "label"
DATASETS = {
    "f1": ORIGINAL_REPO / "datasets" / "BPIC11_f1_trunc36.csv",
    "f2": ORIGINAL_REPO / "datasets" / "BPIC11_f2_trunc40.csv",
    "f3": ORIGINAL_REPO / "datasets" / "BPIC11_f3_trunc31.csv",
}
VERIFY_FOLD_SEED = 7   # the builder used seed 0, fold 0
VERIFY_FOLD = 2
VERIFY_PICKS = (3, 4, 5, 6, 7)  # 4th-8th Apriori itemsets; builder: 0, 3, 4


# --------------------------------------------------------------------------
# Independent building blocks
# --------------------------------------------------------------------------
class IndependentLog:
    """Second loader of a BPIC11 CSV, written without looking at EventLog.

    ``allowed`` comes from the ``event_nr`` column itself (not list indexes).
    """

    def __init__(self, name: str):
        frame = pd.read_csv(DATASETS[name])
        frame[CASE] = frame[CASE].astype(str)
        frame[ACT] = frame[ACT].str.lower().str.replace(r"[ \-_]", "", regex=True)
        counts = frame[ACT].value_counts()
        rare = set(counts[counts < 2].index)
        bad_cases = set(frame.loc[frame[ACT].isin(rare), CASE])
        frame = frame[~frame[CASE].isin(bad_cases)]
        frame = frame.sort_values([CASE, "event_nr"], kind="mergesort")
        self.traces = {case: part[ACT].tolist()
                       for case, part in frame.groupby(CASE, sort=True)}
        mapping = {"deviant": 1, "regular": 0}
        self.labels = {
            case: int(mapping.get(part[LABEL].iloc[0], part[LABEL].iloc[0]))
            for case, part in frame.groupby(CASE, sort=True)}
        self.allowed = {
            act: set(int(v) for v in part["event_nr"])
            for act, part in frame.groupby(ACT)}
        self.max_len = max(len(t) for t in self.traces.values())


def positions_in(traces) -> dict:
    """Activity -> set of 1-based positions observed in ``traces``."""
    seen = collections.defaultdict(set)
    for trace in traces:
        for index, act in enumerate(trace):
            seen[act].add(index + 1)
    return dict(seen)


def naive_encode(traces, feature_names, max_len) -> np.ndarray:
    """Index encoding straight from its definition (slow, obviously right)."""
    column = {name: i for i, name in enumerate(feature_names)}
    matrix = np.zeros((len(traces), len(feature_names)), dtype=np.int8)
    for row, trace in enumerate(traces):
        for index in range(max_len):
            if index < len(trace):
                matrix[row, column[f"e{index + 1}_{trace[index]}"]] = 1
            elif f"e{index + 1}_0" in column:
                matrix[row, column[f"e{index + 1}_0"]] = 1
    return matrix


def brute_occurrences(trace, itemset) -> list:
    """Greedy non-overlapping occurrences by enumerating all combinations."""
    wanted, kept, used = set(itemset), [], set()
    for combo in itertools.combinations(range(len(trace)), len(wanted)):
        if {trace[i] for i in combo} == wanted and not used.intersection(combo):
            kept.append(combo)
            used.update(combo)
    return kept


def my_folds(labels: dict, seed: int) -> list:
    """Stratified 5-fold split written directly with scikit-learn."""
    cases = list(labels)
    outcome = [labels[c] for c in cases]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return [([cases[i] for i in train], [cases[i] for i in test])
            for train, test in splitter.split(cases, outcome)]


def my_weighted_f1(y_true, y_pred) -> float:
    return float(f1_score(list(y_true), list(y_pred), average="weighted"))


def load_original_manager(name: str):
    """Original DataManager(path, 2, None, L_max_perc=0.8) with the int cast."""
    module_name = "verifier_tools_2024"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            module_name, ORIGINAL_REPO / "tools.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    tools = sys.modules[module_name]

    class IntCastManager(tools.DataManager):
        def index_encoding(self, data):
            encoded = super().index_encoding(data)
            features = [c for c in encoded.columns
                        if c not in (self.case_id, self.outcome)]
            encoded[features] = encoded[features].astype(int)
            return encoded

    return IntCastManager(str(DATASETS[name]), 2, None, L_max_perc=0.8)


@contextlib.contextmanager
def silenced():
    """Hide the original's prints and pandas warnings."""
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        yield


def check_permutation(trace, new_trace, itemset, placements, allowed) -> list:
    """Names of all violated properties of one permuted trace (empty = ok).

    ``allowed`` maps activity -> set of 1-based positions. Checks marked
    'free' do not use the engine's reported ``placements`` at all.
    """
    bad = []
    items = set(itemset)
    if len(new_trace) != len(trace):
        bad.append("length")
    if collections.Counter(new_trace) != collections.Counter(trace):
        bad.append("multiset")
    if ([a for a in trace if a not in items]
            != [a for a in new_trace if a not in items]):
        bad.append("non_itemset_order(free)")

    occurrences = brute_occurrences(trace, items)
    if [tuple(old for old, _ in placed) for placed in placements] != occurrences:
        bad.append("placements_do_not_cover_the_occurrences")
    moved_old = [old for placed in placements for old, _ in placed]
    moved_new = [new for placed in placements for _, new in placed]
    if len(set(moved_new)) != len(moved_new):
        bad.append("two_moved_events_on_one_position")
    for placed in placements:
        new_positions = [new for _, new in placed]  # in old-index order
        if any(b <= a for a, b in zip(new_positions, new_positions[1:])):
            bad.append("order")
        for old, new in placed:
            if not 0 <= new < len(trace):
                bad.append("beyond_trace")
            elif new_trace[new] != trace[old]:
                bad.append("placement_not_in_new_trace")
            if (new + 1) not in allowed.get(trace[old], ()):
                bad.append("feasibility")
    if ([a for i, a in enumerate(trace) if i not in set(moved_old)]
            != [a for i, a in enumerate(new_trace) if i not in set(moved_new)]):
        bad.append("unmoved_order")

    # Placement-free versions, possible when every event of an itemset
    # activity belongs to an occurrence (no left-over duplicates).
    if all(trace.count(a) == len(occurrences) for a in items):
        for index, act in enumerate(new_trace):
            if act in items and (index + 1) not in allowed.get(act, ()):
                bad.append("feasibility(free)")
    if all(trace.count(a) == 1 for a in items):
        if ([a for a in trace if a in items]
                != [a for a in new_trace if a in items]):
            bad.append("order(free)")
    return bad


class PermuteRecorder:
    """Records every call of engine.permute_trace_fixed while active."""

    def __init__(self):
        self.calls = []

    def __enter__(self):
        self._real = engine.permute_trace_fixed
        engine.permute_trace_fixed = self._wrapped
        return self

    def __exit__(self, *exc):
        engine.permute_trace_fixed = self._real

    def _wrapped(self, trace, occurrences, allowed, rng):
        before = list(trace)
        new_trace, placements = self._real(trace, occurrences, allowed, rng)
        self.calls.append({"object": trace, "before": before,
                           "unmutated": list(trace) == before,
                           "new": new_trace, "placements": placements})
        return new_trace, placements


class RecordingModel:
    """Exposes only ``predict`` (so the engine cannot refit) and logs inputs."""

    def __init__(self, model):
        self._model = model
        self.inputs = []

    def predict(self, features):
        self.inputs.append(np.array(features, copy=True))
        return self._model.predict(features)


def write_json(name: str, payload) -> None:
    (HERE / name).write_text(json.dumps(payload, indent=2, default=str),
                             encoding="utf-8")
    print(f"wrote {HERE / name}")


# --------------------------------------------------------------------------
# Part 1: encoder
# --------------------------------------------------------------------------
def part_encoder(names=("f1", "f2", "f3")) -> dict:
    """IndexEncoder vs the original index_encoding, plus the two loaders."""
    out = {}
    for name in names:
        start = time.perf_counter()
        manager = load_original_manager(name)
        with silenced():
            encoded = manager.index_encoding(manager.data)
        seconds_original = time.perf_counter() - start
        features = encoded.drop(columns=[manager.case_id, manager.outcome])
        original_values = features.to_numpy()

        log = engine.EventLog(DATASETS[name])
        mine = IndependentLog(name)
        result = {
            "original_shape": list(features.shape),
            "seconds_original_encoding": round(seconds_original, 2),
            "eventlog_traces_equal_independent_loader": log.traces == mine.traces,
            "eventlog_case_order_sorted": log.case_ids == sorted(mine.traces),
            "eventlog_labels_equal": log.labels == mine.labels,
            "eventlog_allowed_equal_event_nr_column": {
                a: set(v) for a, v in log.allowed_locations.items()} == mine.allowed,
            "eventlog_allowed_equal_original": (
                log.allowed_locations == manager.Allowed_locations),
        }

        start = time.perf_counter()
        hashed = engine.IndexEncoder(log.traces.values(), log.activities, "hash")
        matrix_hash = hashed.transform(log.traces.values())
        result["seconds_engine_encoding"] = round(time.perf_counter() - start, 3)
        result["hash_rows_identical"] = list(features.index) == log.case_ids
        result["hash_columns_identical_incl_order"] = (
            list(features.columns) == hashed.feature_names)
        result["hash_values_identical"] = bool(
            original_values.shape == matrix_hash.shape
            and np.array_equal(original_values, matrix_hash))

        ordered = engine.IndexEncoder(log.traces.values(), log.activities)
        matrix_sorted = ordered.transform(log.traces.values())
        n_observed = int((original_values.sum(axis=0) > 0).sum())
        result["sorted_same_column_set"] = (
            set(ordered.feature_names) == set(features.columns)
            and len(ordered.feature_names) == features.shape[1])
        result["sorted_columns_identical_incl_order"] = (
            ordered.feature_names == list(features.columns))
        result["sorted_first_block_identical_order"] = (
            ordered.feature_names[:n_observed] == list(features.columns)[:n_observed])
        result["sorted_values_identical_after_reindex"] = bool(np.array_equal(
            features[ordered.feature_names].to_numpy(), matrix_sorted))
        result["observed_columns"] = n_observed
        result["never_observed_block_all_zero"] = not bool(
            original_values[:, n_observed:].any())
        result["labels_identical"] = (
            [int(v) for v in encoded[manager.outcome]]
            == [log.labels[c] for c in log.case_ids])
        naive = naive_encode(list(mine.traces.values()), ordered.feature_names,
                             mine.max_len)
        result["engine_equals_naive_encoder"] = bool(
            np.array_equal(naive, matrix_sorted))
        result["every_row_sums_to_max_len"] = bool(
            (matrix_sorted.sum(axis=1, dtype=np.int64) == mine.max_len).all())
        result["all_ok"] = all(
            v for k, v in result.items()
            if isinstance(v, bool) and k != "sorted_columns_identical_incl_order")
        out[name] = result
        print(name, json.dumps(result, indent=2), flush=True)
    write_json("verify_encoder.json", out)
    return out


# --------------------------------------------------------------------------
# Part 2: faithful mode
# --------------------------------------------------------------------------
def part_faithful(name: str, n_repeats: int = 3) -> dict:
    """Faithful mode vs the original function (fold 2 of seed 7, sets 4-8)."""
    manager = load_original_manager(name)
    with silenced():
        encoded = manager.index_encoding(manager.data)
        candidates, _ = manager.frequent_activity_sets(0.5, 10)
    log = engine.EventLog(DATASETS[name])
    train, test = my_folds(log.labels, VERIFY_FOLD_SEED)[VERIFY_FOLD]
    engine_folds = engine.make_folds(log.labels, k=5, seed=VERIFY_FOLD_SEED)
    folds_equal = engine_folds[VERIFY_FOLD] == (train, test)

    part = encoded[encoded[manager.case_id].isin(train)]
    train_y = part[manager.outcome].tolist()
    train_x = part.drop(columns=[manager.case_id, manager.outcome])
    model = XGBClassifier()
    model.fit(train_x, train_y)

    itemsets = {i: list(candidates[i]) for i in VERIFY_PICKS}
    single = {100: [engine.most_frequent_activities(log, 1)[0]]}
    hashed = engine.IndexEncoder(log.traces.values(), log.activities, "hash")
    assert hashed.feature_names == list(train_x.columns), "column order differs"

    rows = []
    timing = {}
    for label, sets, repeats in (("apriori_4th_to_8th", itemsets, n_repeats),
                                 ("single_activity", single, 2)):
        start = time.perf_counter()
        with silenced():
            original = manager.itemset_permutation_importance(
                model, train_x, train_y, train, sets, constrain=True,
                n_repeats=repeats)
        seconds_original = time.perf_counter() - start
        start = time.perf_counter()
        table = engine.LocationPermutationImportance(
            mode="faithful", score_on="train", n_repeats=repeats,
            random_state=2023,
        ).compute(model, log, hashed, train, test, sets, fold=VERIFY_FOLD)
        seconds_engine = time.perf_counter() - start
        n_iterations = len(sets) * repeats
        timing[label] = {
            "iterations": n_iterations,
            "original_s_per_iteration": seconds_original / n_iterations,
            "engine_s_per_iteration": seconds_engine / n_iterations}
        for row in table.itertuples():
            value = float(original[row.itemset_id].iloc[row.repeat])
            rows.append({"group": label, "itemset": row.itemset,
                         "size": len(sets[row.itemset_id]), "repeat": row.repeat,
                         "original": value, "engine": float(row.importance),
                         "abs_diff": abs(value - float(row.importance)),
                         "n_traces": int(row.n_traces_with_itemset)})
        baseline_ok = float(table["baseline"].iloc[0]) == my_weighted_f1(
            train_y, model.predict(train_x))
        timing[label]["baseline_equal"] = bool(baseline_ok)

    # End to end the way a student would use the engine: default encoder and
    # a model fitted on the engine's own matrix, against the original values.
    ordered = engine.IndexEncoder(log.traces.values(), log.activities)
    own_model = XGBClassifier()
    own_model.fit(ordered.transform(log.traces[c] for c in train),
                  [log.labels[c] for c in train])
    own = engine.LocationPermutationImportance(
        mode="faithful", score_on="train", n_repeats=n_repeats, random_state=2023,
    ).compute(own_model, log, ordered, train, test, itemsets, fold=VERIFY_FOLD)
    reference = [r["original"] for r in rows if r["group"] == "apriori_4th_to_8th"]
    end_to_end_diff = float(np.max(np.abs(own["importance"].to_numpy()
                                          - np.array(reference))))

    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / f"verify_faithful_{name}.csv", index=False,
                 float_format="%.17g")
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.17g}"))
    result = {
        "dataset": name, "fold_seed": VERIFY_FOLD_SEED, "fold": VERIFY_FOLD,
        "train_cases": len(train), "test_cases": len(test),
        "make_folds_equals_sklearn_directly": bool(folds_equal),
        "itemsets": itemsets, "single": single, "n_repeats": n_repeats,
        "values_compared": len(rows),
        "max_abs_diff": float(frame["abs_diff"].max()),
        "max_abs_diff_below_1e-12": bool(frame["abs_diff"].max() < 1e-12),
        "values_nonzero": int((frame["original"] != 0).sum()),
        "end_to_end_engine_only_max_abs_diff": end_to_end_diff,
        "timing": timing,
    }
    print(json.dumps(result, indent=2), flush=True)
    write_json(f"verify_faithful_{name}.json", result)
    return result


# --------------------------------------------------------------------------
# Part 3: fixed mode
# --------------------------------------------------------------------------
def direct_trials(name, mine, pools, cases, size, n_feasible, seed) -> dict:
    """Random (trace, itemset of ``size``) pairs through permute_trace_fixed."""
    picker = random.Random(seed)
    generator = np.random.default_rng(seed)
    allowed_lists = {a: sorted(v) for a, v in pools.items()}
    stats = collections.Counter()
    violations = collections.Counter()
    while stats["feasible"] < n_feasible:
        trace = mine.traces[picker.choice(cases)]
        distinct = sorted(set(trace))
        if len(distinct) < size or size == len(trace):
            continue
        items = set(picker.sample(distinct, size))
        occurrences = engine.find_occurrences(trace, items)
        if occurrences != brute_occurrences(trace, items):
            violations["find_occurrences_differs_from_brute_force"] += 1
        before = list(trace)
        new_trace, placements = engine.permute_trace_fixed(
            trace, occurrences, allowed_lists, generator)
        if trace != before:
            violations["input_mutated"] += 1
        if new_trace is None:
            stats["infeasible"] += 1
            first = [[p for p in allowed_lists.get(trace[i], ()) if p <= len(trace)]
                     for i in occurrences[0]]
            exists = any(all(a < b for a, b in zip(c, c[1:]))
                         for c in itertools.product(*first))
            if len(occurrences) == 1 and exists:
                violations["declared_infeasible_but_feasible"] += 1
            continue
        stats["feasible"] += 1
        stats["changed"] += new_trace != trace
        stats["multi_occurrence"] += len(occurrences) > 1
        stats["moved_events"] += sum(len(p) for p in placements)
        for problem in check_permutation(trace, new_trace, items, placements, pools):
            violations[problem] += 1
    return {"dataset": name, "size": size, **dict(stats),
            "violations": dict(violations)}


def pick_itemsets(mine, cases) -> dict:
    """Deterministic 1-, 2- and 3-sets of varying frequency in ``cases``."""
    support = collections.Counter(a for c in cases for a in set(mine.traces[c]))
    ranked = [a for a, _ in sorted(support.items(), key=lambda kv: (-kv[1], kv[0]))]
    ranks = [(0,), (1,), (5,), (12,), (0, 1), (2, 7), (4, 9), (3, 11),
             (0, 1, 2), (3, 8, 14), (1, 6, 10), (2, 5, 13)]
    return {i: [ranked[r] for r in rank] for i, rank in enumerate(ranks)}


def recorded_run(log, encoder, model, train, test, itemsets, **settings):
    """engine.compute with the permutation calls and model inputs recorded."""
    recording_model = RecordingModel(model)
    with PermuteRecorder() as recorder:
        table = engine.LocationPermutationImportance(
            mode="fixed", **settings,
        ).compute(recording_model, log, encoder, train, test, itemsets,
                  fold=VERIFY_FOLD)
    return table, recorder.calls, recording_model.inputs


def verify_recorded_run(table, calls, inputs, log, mine, encoder, model, scored,
                        pools, itemsets) -> dict:
    """Re-derive everything in ``table`` from the recorded permutations."""
    violations = collections.Counter()
    stats = collections.Counter()
    row_of = {id(log.traces[case]): row for row, case in enumerate(scored)}
    originals = [mine.traces[case] for case in scored]
    y_true = [mine.labels[case] for case in scored]
    names, max_len = encoder.feature_names, mine.max_len

    base_matrix = naive_encode(originals, names, max_len)
    if not np.array_equal(inputs[0], base_matrix):
        violations["baseline_matrix_is_not_the_scored_cases"] += 1
    base_pred = model.predict(base_matrix)
    my_baseline = my_weighted_f1(y_true, base_pred)
    if not (table["baseline"] == my_baseline).all():
        violations["baseline_differs_from_my_weighted_f1"] += 1

    cursor, next_input = 0, 1
    for record in table.itertuples():
        items = set(itemsets[record.itemset_id])
        eligible = sum(items.issubset(t) and len(items) != len(t) for t in originals)
        if eligible != record.n_traces_with_itemset:
            violations["n_traces_with_itemset_wrong"] += 1
        segment = calls[cursor:cursor + record.n_traces_with_itemset]
        cursor += record.n_traces_with_itemset
        permuted, changed, n_infeasible = list(originals), [], 0
        for call in segment:
            row = row_of.get(id(call["object"]))
            if row is None or call["before"] != originals[row]:
                violations["permuted_something_else_than_an_original_trace"] += 1
                continue
            if not call["unmutated"]:
                violations["input_mutated"] += 1
            if call["new"] is None:
                n_infeasible += 1
                continue
            stats["permuted_traces_checked"] += 1
            stats[f"size_{len(items)}"] += 1
            stats["multi_occurrence"] += len(call["placements"]) > 1
            for problem in check_permutation(originals[row], call["new"], items,
                                             call["placements"], pools):
                violations[problem] += 1
            if call["new"] != originals[row]:
                permuted[row] = call["new"]
                changed.append((row, call["new"]))
        if len(changed) != record.n_traces_changed:
            violations["n_traces_changed_wrong"] += 1
        if n_infeasible != record.n_traces_unchanged_infeasible:
            violations["n_infeasible_wrong"] += 1
        if changed:
            changed.sort()
            expected = naive_encode([t for _, t in changed], names, max_len)
            if not np.array_equal(inputs[next_input], expected):
                violations["re_encoded_rows_differ_from_naive_encoding"] += 1
            next_input += 1
        my_permuted = my_weighted_f1(
            y_true, model.predict(naive_encode(permuted, names, max_len)))
        if my_permuted != record.permuted:
            violations["permuted_f1_differs_from_full_recomputation"] += 1
        if record.importance != my_baseline - my_permuted:
            violations["importance_differs_from_full_recomputation"] += 1
        stats["table_rows_recomputed"] += 1
    if cursor != len(calls):
        violations["unexplained_permutation_calls"] += 1
    if next_input != len(inputs):
        violations["unexplained_model_calls"] += 1
    return {**dict(stats), "baseline": my_baseline, "violations": dict(violations)}


def fixed_for_dataset(name: str) -> dict:
    """All fixed-mode checks for one log."""
    out = {}
    log = engine.EventLog(DATASETS[name])
    mine = IndependentLog(name)
    snapshot = copy.deepcopy(log.traces)
    folds = my_folds(mine.labels, VERIFY_FOLD_SEED)
    train, test = folds[VERIFY_FOLD]
    encoder = engine.IndexEncoder(log.traces.values(), log.activities)
    model = XGBClassifier()
    model.fit(naive_encode([mine.traces[c] for c in train], encoder.feature_names,
                           mine.max_len), [mine.labels[c] for c in train])

    # (a) folds: a partition, stratified, equal to engine.make_folds
    all_test = [c for _, fold_test in folds for c in fold_test]
    out["folds"] = {
        "engine_make_folds_equal_mine": engine.make_folds(
            log.labels, k=5, seed=VERIFY_FOLD_SEED) == folds,
        "train_test_disjoint_every_fold": all(
            not set(a) & set(b) for a, b in folds),
        "train_plus_test_is_all_cases_every_fold": all(
            set(a) | set(b) == set(mine.traces) for a, b in folds),
        "every_case_in_exactly_one_test_fold": (
            sorted(all_test) == sorted(mine.traces)),
        "label_1_share_per_test_fold": [
            round(float(np.mean([mine.labels[c] for c in b])), 4) for _, b in folds],
        "model_trained_on": len(train), "held_out": len(test),
    }

    # (b) direct random trials: sizes 1-3, whole-log pools and training pools
    pools_log = mine.allowed
    pools_train = positions_in(mine.traces[c] for c in train)
    out["direct_trials"] = []
    for size in (1, 2, 3):
        for label, pools, cases in (("log", pools_log, list(mine.traces)),
                                    ("train", pools_train, test)):
            trial = direct_trials(name, mine, pools, cases, size, 350,
                                  seed=1000 + 10 * size + len(label))
            trial["pools"] = label
            out["direct_trials"].append(trial)
            print(name, "direct", trial, flush=True)

    # (c) through compute(): every scored permutation re-checked and the
    #     importance recomputed from scratch
    out["compute_runs"] = {}
    tables = {}
    for score_on, allowed_from in (("train", "log"), ("test", "log"),
                                   ("test", "train")):
        scored = train if score_on == "train" else test
        pools = pools_log if allowed_from == "log" else pools_train
        itemsets = pick_itemsets(mine, scored)
        settings = dict(score_on=score_on, allowed_from=allowed_from,
                        n_repeats=3, random_state=2023)
        table, calls, inputs = recorded_run(log, encoder, model, train, test,
                                            itemsets, **settings)
        summary = verify_recorded_run(table, calls, inputs, log, mine, encoder,
                                      model, scored, pools, itemsets)
        summary["scored_cases"] = len(scored)
        summary["first_model_call_rows"] = int(inputs[0].shape[0])
        summary["infeasible_total"] = int(
            table["n_traces_unchanged_infeasible"].sum())
        key = f"score_on={score_on},allowed_from={allowed_from}"
        out["compute_runs"][key] = summary
        tables[key] = (table, calls, itemsets, settings)
        print(name, key, summary, flush=True)

    # (d) determinism, seeds, no accumulation, independence of context
    key = "score_on=test,allowed_from=log"
    table, calls, itemsets, settings = tables[key]
    again, calls_again, _ = recorded_run(log, encoder, model, train, test,
                                         itemsets, **settings)
    other_settings = dict(settings, random_state=99)
    other, calls_other, _ = recorded_run(log, encoder, model, train, test,
                                         itemsets, **other_settings)
    new_traces = [c["new"] for c in calls]
    more_repeats = engine.LocationPermutationImportance(
        mode="fixed", **dict(settings, n_repeats=6),
    ).compute(model, log, encoder, train, test, itemsets, fold=VERIFY_FOLD)
    reversed_sets = dict(reversed(list(itemsets.items())))
    backwards = engine.LocationPermutationImportance(
        mode="fixed", **settings,
    ).compute(model, log, encoder, train, test, reversed_sets, fold=VERIFY_FOLD)
    last_id = list(itemsets)[-1]
    alone = engine.LocationPermutationImportance(
        mode="fixed", **settings,
    ).compute(model, log, encoder, train, test, {last_id: itemsets[last_id]},
              fold=VERIFY_FOLD)
    sort_cols = ["itemset_id", "repeat"]
    first_three = more_repeats[more_repeats["repeat"] < 3].reset_index(drop=True)
    out["determinism"] = {
        "same_seed_identical_table": table.equals(again),
        "same_seed_identical_permuted_traces": (
            new_traces == [c["new"] for c in calls_again]),
        "other_seed_different_permuted_traces": (
            new_traces != [c["new"] for c in calls_other]),
        "share_of_traces_that_differ_with_other_seed": round(float(np.mean(
            [a != b for a, b in zip(new_traces, [c["new"] for c in calls_other])])),
            4),
        "other_seed_same_baseline": bool(
            (other["baseline"] == table["baseline"]).all()),
        "repeat_0_1_2_same_when_6_repeats_are_run": table.equals(first_three),
        "result_independent_of_itemset_order": table.sort_values(sort_cols)
        .reset_index(drop=True).equals(
            backwards.sort_values(sort_cols).reset_index(drop=True)),
        "last_itemset_alone_equals_last_itemset_in_list": alone.equals(
            table[table["itemset_id"] == last_id].reset_index(drop=True)),
        "log_traces_untouched_after_all_runs": log.traces == snapshot,
        "repeats_of_one_itemset_differ_from_each_other": bool(
            len({tuple(map(tuple, (c["new"] or [] for c in calls[i:i + n])))
                 for i, n in [(0, int(table["n_traces_with_itemset"].iloc[0])),
                              (int(table["n_traces_with_itemset"].iloc[0]),
                               int(table["n_traces_with_itemset"].iloc[0]))]}) == 2),
    }
    print(name, "determinism", out["determinism"], flush=True)

    # (e) information: how much the whole-log pools owe to held-out cases,
    #     and how often an unmoved event lands where it was never observed
    pairs_log = {(a, p) for a, ps in pools_log.items() for p in ps}
    pairs_train = {(a, p) for a, ps in pools_train.items() for p in ps}
    table, calls, itemsets, _ = tables["score_on=test,allowed_from=log"]
    unmoved_total = unmoved_unobserved = 0
    for call in calls:
        if call["new"] is None or call["new"] == call["before"]:
            continue
        moved_new = {new for placed in call["placements"] for _, new in placed}
        for index, act in enumerate(call["new"]):
            if index not in moved_new and call["before"][index] != act:
                unmoved_total += 1
                unmoved_unobserved += (index + 1) not in pools_log[act]
    out["information"] = {
        "activity_position_pairs_whole_log": len(pairs_log),
        "pairs_only_known_from_held_out_fold": len(pairs_log - pairs_train),
        "shifted_unmoved_events": unmoved_total,
        "shifted_unmoved_events_at_never_observed_position": unmoved_unobserved,
    }

    # (f) size-1 draw: uniform over the allowed positions of the trace?
    trace = max(mine.traces.values(), key=len)
    act = max(set(trace), key=lambda a: (trace.count(a) == 1,
                                         len([p for p in pools_log[a]
                                              if p <= len(trace)])))
    pool = sorted(p for p in pools_log[act] if p <= len(trace))
    generator = np.random.default_rng(5)
    allowed_lists = {a: sorted(v) for a, v in pools_log.items()}
    occurrences = engine.find_occurrences(trace, {act})
    draws = collections.Counter()
    n_draws = 20000
    for _ in range(n_draws):
        new_trace, placements = engine.permute_trace_fixed(
            trace, occurrences, allowed_lists, generator)
        draws[placements[0][0][1] + 1] += 1
    expected = n_draws / len(pool)
    chi2 = sum((draws[p] - expected) ** 2 / expected for p in pool)
    out["size_1_uniformity"] = {
        "single_occurrence": trace.count(act) == 1, "pool_size": len(pool),
        "draws": n_draws, "positions_drawn_outside_pool": sorted(
            set(draws) - set(pool)), "chi2": round(chi2, 2),
        "degrees_of_freedom": len(pool) - 1,
    }
    return out


def part_fixed(names=("f1", "f3")) -> dict:
    out = {name: fixed_for_dataset(name) for name in names}
    total = n_violations = 0
    for name in names:
        for trial in out[name]["direct_trials"]:
            total += trial["feasible"]
            n_violations += sum(trial["violations"].values())
        for run in out[name]["compute_runs"].values():
            total += run.get("permuted_traces_checked", 0)
            n_violations += sum(run["violations"].values())
    out["permuted_traces_checked_total"] = total
    out["violations_total"] = n_violations
    print(json.dumps(out, indent=2, default=str), flush=True)
    write_json("verify_fixed.json", out)
    return out


# --------------------------------------------------------------------------
# Part 4: edge cases
# --------------------------------------------------------------------------
def part_edge(name: str = "f1") -> dict:
    """Situations the project grid will meet; each reports what happens."""
    log = engine.EventLog(DATASETS[name])
    encoder = engine.IndexEncoder(log.traces.values(), log.activities)
    train, test = my_folds(log.labels, VERIFY_FOLD_SEED)[VERIFY_FOLD]
    model = XGBClassifier()
    model.fit(encoder.transform(log.traces[c] for c in train),
              [log.labels[c] for c in train])
    in_test = {a for c in test for a in log.traces[c]}
    absent = sorted(a for a in log.activities if a not in in_test)
    one_place = sorted(a for a in log.activities
                       if len(log.allowed_locations[a]) == 1)

    def attempt(description, itemsets, **settings):
        try:
            table = engine.LocationPermutationImportance(
                mode="fixed", n_repeats=2, **settings,
            ).compute(model, log, encoder, train, test, itemsets, fold=VERIFY_FOLD)
            outcome = {"status": "ok",
                       "importance": table["importance"].tolist(),
                       "n_traces_changed": table["n_traces_changed"].tolist()}
        except Exception as error:  # noqa: BLE001 - the outcome is the result
            outcome = {"status": "CRASH", "error": f"{type(error).__name__}: {error}",
                       "where": traceback.format_exc().strip().splitlines()[-3]}
        print(description, outcome, flush=True)
        return outcome

    out = {
        "activities_absent_from_held_out_fold": len(absent),
        "activities_with_one_allowed_position": len(one_place),
        "encoder_transform_of_empty_list": None,
        "itemset_absent_from_scored_fold": attempt(
            "absent itemset", [[absent[0]]], score_on="test"),
        "activity_with_a_single_allowed_position": attempt(
            "one allowed position", [[one_place[0]]], score_on="train")
        if one_place else "no such activity",
        "unknown_activity_name": attempt(
            "unknown activity", [["no_such_activity"]], score_on="test"),
        "string_instead_of_list": attempt(
            "string as itemset", [log.activities[0]], score_on="test"),
    }
    try:
        out["encoder_transform_of_empty_list"] = list(encoder.transform([]).shape)
    except Exception as error:  # noqa: BLE001
        out["encoder_transform_of_empty_list"] = (
            f"CRASH {type(error).__name__}: {error}")

    # How many single activities would crash a held-out single-activity grid?
    crashes = 0
    for act in log.activities:
        try:
            engine.LocationPermutationImportance(
                mode="fixed", score_on="test", n_repeats=1,
            ).compute(model, log, encoder, train, test, [[act]], fold=VERIFY_FOLD)
        except ValueError:
            crashes += 1
    out["single_activity_grid_held_out"] = {
        "activities": len(log.activities), "crashing": crashes}
    print(json.dumps(out, indent=2), flush=True)
    write_json("verify_edge.json", out)
    return out


# --------------------------------------------------------------------------
# Part 5: the builder's grid files
# --------------------------------------------------------------------------
def part_grid(names=("f1", "f2", "f3")) -> dict:
    """Re-run the builder's full grid; compare with full_grid_<name>.csv."""
    out = {}
    for name in names:
        published = pd.read_csv(HERE / f"full_grid_{name}.csv")
        manager = load_original_manager(name)
        with silenced():
            candidates, _ = manager.frequent_activity_sets(0.5, 10)
        itemsets = dict(enumerate(candidates))
        log = engine.EventLog(DATASETS[name])
        encoder = engine.IndexEncoder(log.traces.values(), log.activities)
        result = {"itemsets": len(itemsets),
                  "itemset_sizes": sorted(len(s) for s in candidates)}
        tables = []
        held_out_f1 = []
        for fold, (train, test) in enumerate(my_folds(log.labels, 0)):
            model = XGBClassifier()
            model.fit(encoder.transform(log.traces[c] for c in train),
                      [log.labels[c] for c in train])
            held_out_f1.append(my_weighted_f1(
                [log.labels[c] for c in test],
                model.predict(encoder.transform(log.traces[c] for c in test))))
            for variant, mode, score_on in (("faithful_train", "faithful", "train"),
                                            ("fixed_train", "fixed", "train"),
                                            ("fixed_test", "fixed", "test")):
                table = engine.LocationPermutationImportance(
                    mode=mode, score_on=score_on, n_repeats=10, random_state=2023,
                ).compute(model, log, encoder, train, test, itemsets, fold=fold)
                table.insert(0, "variant", variant)
                tables.append(table)
        grid = pd.concat(tables, ignore_index=True)
        keys = ["variant", "fold", "itemset_id", "repeat"]
        merged = published.merge(grid, on=keys, suffixes=("_builder", "_mine"))
        result["rows_builder"] = len(published)
        result["rows_matched"] = len(merged)
        for column in ("baseline", "importance"):
            result[f"max_abs_diff_{column}"] = float(
                (merged[f"{column}_builder"] - merged[f"{column}_mine"]).abs().max())
        result["held_out_f1_per_fold"] = [round(v, 4) for v in held_out_f1]
        fixed_test = grid[grid["variant"] == "fixed_test"]
        per_fold = fixed_test.groupby("fold")["baseline"].first().tolist()
        result["fixed_test_baseline_equals_my_held_out_f1"] = per_fold == held_out_f1
        result["mean_importance"] = {
            k: round(float(v), 4)
            for k, v in grid.groupby("variant")["importance"].mean().items()}
        result["share_above_zero"] = {
            k: round(float(v), 3) for k, v in
            (grid["importance"] > 0).groupby(grid["variant"]).mean().items()}
        result["std_of_itemset_means_fixed_test"] = round(float(
            fixed_test.groupby("itemset_id")["importance"].mean().std()), 5)
        result["mean_within_itemset_std_fixed_test"] = round(float(
            fixed_test.groupby("itemset_id")["importance"].std().mean()), 5)
        out[name] = result
        print(name, json.dumps(result, indent=2), flush=True)
    write_json("verify_grid.json", out)
    return out


# --------------------------------------------------------------------------
# Part 6: independence of the interpreter's string-hash seed
# --------------------------------------------------------------------------
def part_digest(name: str = "f3") -> str:
    """Hash of a fixed and a faithful result table (default encoder)."""
    log = engine.EventLog(DATASETS[name])
    encoder = engine.IndexEncoder(log.traces.values(), log.activities)
    train, test = my_folds(log.labels, VERIFY_FOLD_SEED)[VERIFY_FOLD]
    model = XGBClassifier()
    model.fit(encoder.transform(log.traces[c] for c in train),
              [log.labels[c] for c in train])
    itemsets = pick_itemsets(IndependentLog(name), train)
    text = hashlib.sha256("|".join(encoder.feature_names).encode()).hexdigest()
    for mode, score_on in (("fixed", "test"), ("fixed", "train"),
                           ("faithful", "train")):
        table = engine.LocationPermutationImportance(
            mode=mode, score_on=score_on, n_repeats=2, random_state=2023,
        ).compute(model, log, encoder, train, test, itemsets, fold=VERIFY_FOLD)
        text += table.to_csv(index=False, float_format="%.17g")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"DIGEST {digest}")
    return digest


def part_hashseed() -> dict:
    """Two fresh interpreters with different PYTHONHASHSEED give one digest?"""
    digests = {}
    for seed in ("1", "2"):
        environment = dict(os.environ, PYTHONHASHSEED=seed)
        done = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--part", "digest"],
            capture_output=True, text=True, env=environment, check=False)
        lines = [ln for ln in done.stdout.splitlines() if ln.startswith("DIGEST")]
        digests[seed] = lines[-1].split()[1] if lines else done.stderr[-500:]
    out = {"digests": digests,
           "identical_across_hash_seeds": len(set(digests.values())) == 1}
    print(json.dumps(out, indent=2))
    write_json("verify_hashseed.json", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", required=True, choices=[
        "encoder", "faithful", "fixed", "edge", "grid", "digest", "hashseed"])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="f1")
    args = parser.parse_args()
    start = time.perf_counter()
    if args.part == "encoder":
        part_encoder()
    elif args.part == "faithful":
        part_faithful(args.dataset)
    elif args.part == "fixed":
        part_fixed()
    elif args.part == "edge":
        part_edge()
    elif args.part == "grid":
        part_grid()
    elif args.part == "digest":
        part_digest()
    else:
        part_hashseed()
    print(f"part {args.part} finished in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
