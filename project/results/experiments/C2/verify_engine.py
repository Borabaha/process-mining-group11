"""Independent re-verification of experiments/engine.py (experiment C2).

Written by the second verifier from scratch. It does not import
test_engine.py, engine_reference.py or the first verifier's script (kept as
verify_engine_round1.py). The original 2024 code is loaded read-only by a
loader defined here, the logs are re-read by a separate loader, matrices are
re-encoded / decoded by naive code and every property is checked by code in
this file.

Parts (``--part``):

encoder   IndexEncoder vs the original ``DataManager.index_encoding`` (int
          cast) on f1, f2, f3: rows, columns, column order, values; plus the
          original's "re-encode a subset" behaviour vs ``pad_until``.
faithful  faithful mode vs the original ``itemset_permutation_importance``
          (``--dataset``, ``--score-on``), fold 2 of a fold seed the builder
          did not use, the 4th-8th original Apriori itemsets, 3 repeats.
single    faithful ``compute_single_activities`` vs the original
          ``trace_permutation_importance`` on a held-out fold.
fixed     fixed mode on one log (``--dataset``): the two constraints of the
          paper (Section 3.2) and the other guarantees, sizes 1, 2 and 3.
toy       hand-made traces: every feasible outcome reachable, no other.
noise     side information: level and fold-seed stability of fixed-mode
          importances on the held-out and the training fold (f1, f2, f3).
deadend   how often fixed mode gives up on a trace with several occurrences
          although a valid assignment exists (f1, f3, whole-log pools).
digest    print a hash of small result tables (used by ``hashseed``).
hashseed  run ``digest`` in two processes with different PYTHONHASHSEED.

Run with the project's Python 3.12 venv, e.g.

    python verify_engine.py --part encoder
    python verify_engine.py --part faithful --dataset f2
    python verify_engine.py --part fixed --dataset f3

Each part writes ``verify_r2_<part>[_<dataset>...].json`` next to this file.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import io
import itertools
import json
import os
import subprocess
import sys
import time
import warnings
from collections import Counter
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
NR = "event_nr"
DATASETS = {
    "f1": ORIGINAL_REPO / "datasets" / "BPIC11_f1_trunc36.csv",
    "f2": ORIGINAL_REPO / "datasets" / "BPIC11_f2_trunc40.csv",
    "f3": ORIGINAL_REPO / "datasets" / "BPIC11_f3_trunc31.csv",
}
FOLD_SEED = 11          # builder: 0, first verifier: 7
FOLD = 2                # builder: 0
PICKS = (3, 4, 5, 6, 7)  # 4th-8th original Apriori itemsets
TOLERANCE = 1e-12


# --------------------------------------------------------------------------
# Independent building blocks
# --------------------------------------------------------------------------
class MyLog:
    """The log as this verifier reads it (no engine code involved)."""

    def __init__(self, path: Path):
        frame = pd.read_csv(path)
        frame[CASE] = frame[CASE].astype(str)
        frame[ACT] = [
            name.lower().replace(" ", "").replace("-", "").replace("_", "")
            for name in frame[ACT]
        ]
        n_events = Counter(frame[ACT])
        rare = {act for act, count in n_events.items() if count < 2}
        bad_cases = set(frame.loc[frame[ACT].isin(rare), CASE])
        frame = frame[~frame[CASE].isin(bad_cases)]
        frame = frame.sort_values([CASE, NR], kind="stable")
        self.traces: dict[str, list[str]] = {}
        self.labels: dict[str, int] = {}
        self.allowed: dict[str, set[int]] = {}
        for case, act, label, number in zip(
                frame[CASE], frame[ACT], frame[LABEL], frame[NR]):
            self.traces.setdefault(case, []).append(act)
            self.labels.setdefault(case, int(label))
            self.allowed.setdefault(act, set()).add(int(number))


def pools_of(traces) -> dict[str, set[int]]:
    """Activity -> set of 1-based positions at which it occurs in ``traces``."""
    pools: dict[str, set[int]] = {}
    for trace in traces:
        for index, act in enumerate(trace):
            pools.setdefault(act, set()).add(index + 1)
    return pools


def naive_encode(traces, feature_names) -> np.ndarray:
    """Index encoding written the slow, obvious way."""
    column = {name: number for number, name in enumerate(feature_names)}
    max_len = max(int(name.split("_", 1)[0][1:]) for name in feature_names)
    matrix = np.zeros((len(traces), len(feature_names)), dtype=np.int8)
    for row, trace in enumerate(traces):
        for index, act in enumerate(trace):
            matrix[row, column[f"e{index + 1}_{act}"]] = 1
        for index in range(len(trace), max_len):
            if f"e{index + 1}_0" in column:
                matrix[row, column[f"e{index + 1}_0"]] = 1
    return matrix


def decode_rows(matrix, feature_names) -> list:
    """Turn an encoded matrix back into traces (None for a malformed row)."""
    parsed = [name.split("_", 1) for name in feature_names]
    parsed = [(int(position[1:]), act) for position, act in parsed]
    traces = []
    for row in np.asarray(matrix):
        hot = sorted(parsed[column] for column in np.nonzero(row)[0])
        if not hot:
            traces.append(None)
            continue
        positions = [position for position, _ in hot]
        acts = [act for _, act in hot]
        real = [act for act in acts if act != "0"]
        well_formed = (
            positions == list(range(positions[0], positions[0] + len(hot)))
            and positions[0] == 1
            and acts[:len(real)] == real          # padding only at the end
        )
        traces.append(real if well_formed else None)
    return traces


def brute_occurrences(trace, itemset) -> list:
    """The original's ``find_itemset_indexes`` rule, re-typed from the text.

    All index combinations in lexicographic order that hold every itemset
    activity exactly once; keep each one that does not overlap a kept one.
    """
    wanted, kept, used = set(itemset), [], set()
    for combination in itertools.combinations(range(len(trace)), len(wanted)):
        if used.intersection(combination):
            continue
        if {trace[index] for index in combination} == wanted:
            kept.append(combination)
            used.update(combination)
    return kept


def weighted_f1_by_hand(y_true, y_pred) -> float:
    """Weighted F1 from the confusion counts (no sklearn)."""
    y_true, y_pred = list(y_true), list(y_pred)
    total = 0.0
    for cls in set(y_true) | set(y_pred):
        tp = sum(t == cls and p == cls for t, p in zip(y_true, y_pred))
        fp = sum(t != cls and p == cls for t, p in zip(y_true, y_pred))
        fn = sum(t == cls and p != cls for t, p in zip(y_true, y_pred))
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        total += f1 * sum(t == cls for t in y_true)
    return total / len(y_true)


def load_original_manager(path: Path):
    """Original ``DataManager(path, 2, None, L_max_perc=0.8)`` + int cast."""
    module_name = "verifier2_original_tools_2024"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            module_name, ORIGINAL_REPO / "tools.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    original = sys.modules[module_name]

    class IntCastManager(original.DataManager):
        """pandas >= 2: ``get_dummies`` gives bool; cast features to int."""

        def index_encoding(self, data):
            encoded = super().index_encoding(data)
            features = [c for c in encoded.columns if c not in (CASE, LABEL)]
            encoded[features] = encoded[features].astype(int)
            return encoded

    return IntCastManager(str(path), 2, None, L_max_perc=0.8)


@contextlib.contextmanager
def silenced():
    """Hide the original's prints and pandas warnings."""
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        yield


def frame_digest(frame: pd.DataFrame) -> str:
    text = frame.to_csv(index=False, float_format="%.17g")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(stem: str, payload) -> None:
    target = HERE / f"verify_r2_{stem}.json"
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"wrote {target.name}")


# --------------------------------------------------------------------------
# Part 1: encoder
# --------------------------------------------------------------------------
def encoder_for_dataset(name: str) -> dict:
    out: dict = {}
    manager = load_original_manager(DATASETS[name])
    start = time.perf_counter()
    with silenced():
        encoded = manager.index_encoding(manager.data)
    out["seconds_original_encoding"] = round(time.perf_counter() - start, 2)
    features = encoded.drop(columns=[CASE, LABEL])
    out["original_shape"] = list(features.shape)

    log = engine.EventLog(DATASETS[name])
    mine = MyLog(DATASETS[name])
    out["n_cases"], out["n_activities"] = len(log.traces), len(log.activities)
    out["eventlog_traces_equal_my_loader"] = (
        log.traces == mine.traces and list(log.traces) == list(mine.traces))
    out["eventlog_labels_equal_my_loader"] = log.labels == mine.labels
    out["eventlog_allowed_equal_my_loader"] = (
        {a: set(p) for a, p in log.allowed_locations.items()} == mine.allowed)
    out["eventlog_allowed_equal_original"] = (
        log.allowed_locations == manager.Allowed_locations)
    out["allowed_lists_sorted"] = all(
        p == sorted(p) for p in log.allowed_locations.values())
    out["activity_order_equal_original"] = (
        log.activities == manager.data[ACT].unique().tolist())
    out["row_order_equal"] = list(features.index) == log.case_ids
    out["labels_equal_original"] = (
        [int(v) for v in encoded[LABEL]] == [log.labels[c] for c in log.case_ids])

    traces = list(log.traces.values())
    start = time.perf_counter()
    enc_hash = engine.IndexEncoder(traces, log.activities, missing_order="hash")
    matrix_hash = enc_hash.transform(traces)
    out["seconds_engine_encoding"] = round(time.perf_counter() - start, 3)
    out["hash_columns_identical_incl_order"] = (
        list(features.columns) == enc_hash.feature_names)
    out["hash_values_identical"] = bool(
        features.shape == matrix_hash.shape
        and np.array_equal(features.to_numpy(), matrix_hash))

    enc_sorted = engine.IndexEncoder(traces, log.activities)  # default order
    matrix_sorted = enc_sorted.transform(traces)
    out["sorted_same_column_set"] = (
        set(features.columns) == set(enc_sorted.feature_names)
        and len(set(enc_sorted.feature_names)) == len(enc_sorted.feature_names))
    out["sorted_columns_identical_incl_order"] = (
        list(features.columns) == enc_sorted.feature_names)
    out["sorted_values_identical_by_name"] = bool(np.array_equal(
        features[enc_sorted.feature_names].to_numpy(), matrix_sorted))
    n_observed = int((features.to_numpy().sum(axis=0) > 0).sum())
    out["n_observed_columns"] = n_observed
    out["sorted_first_block_identical_order"] = (
        list(features.columns[:n_observed]) == enc_sorted.feature_names[:n_observed])
    out["tail_block_all_zero_in_original"] = bool(
        features.iloc[:, n_observed:].to_numpy().sum() == 0)
    out["engine_equals_naive_encoder"] = bool(np.array_equal(
        naive_encode(traces, enc_sorted.feature_names), matrix_sorted))
    out["decode_roundtrip"] = (
        decode_rows(matrix_sorted, enc_sorted.feature_names) == traces)

    # The original re-encodes only the shuffled cases; padding then stops at
    # the longest trace of that subset. Engine: transform(..., pad_until=...).
    rng = np.random.default_rng(99)
    subset_ok = []
    for size in (5, 40, 200):
        cases = sorted(rng.choice(log.case_ids, size=size, replace=False).tolist())
        with silenced():
            part = manager.index_encoding(manager.data[manager.data[CASE].isin(cases)])
        missing = [c for c in features.columns if c not in part.columns]
        with silenced():
            part[missing] = 0
        part = part.drop(columns=[CASE, LABEL])[features.columns]
        sub_traces = [log.traces[c] for c in cases]
        longest = max(len(t) for t in sub_traces)
        mine_part = enc_hash.transform(sub_traces, pad_until=longest)
        subset_ok.append(bool(
            list(part.index) == cases and np.array_equal(part.to_numpy(), mine_part)))
    out["subset_reencoding_equals_pad_until"] = subset_ok
    out["all_ok"] = all(
        value if isinstance(value, bool) else all(value)
        for key, value in out.items()
        if isinstance(value, (bool, list))
        and key not in ("sorted_columns_identical_incl_order", "original_shape"))
    return out


def part_encoder() -> dict:
    result = {name: encoder_for_dataset(name) for name in DATASETS}
    for name, out in result.items():
        print(name, json.dumps(out, indent=2))
    write_json("encoder", result)
    return result


# --------------------------------------------------------------------------
# Part 2: faithful mode against the original function
# --------------------------------------------------------------------------
def original_fold(manager, encoded):
    """Fold ``FOLD`` of a seeded split, built like the original script."""
    cases = manager.data[CASE].unique()
    first = manager.data.drop_duplicates(CASE).set_index(CASE)[LABEL]
    outcome = [int(first[case]) for case in cases]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=FOLD_SEED)
    train_index, test_index = list(splitter.split(cases, outcome))[FOLD]
    parts = {}
    for key, index in (("train", train_index), ("test", test_index)):
        case_list = pd.Series(cases[index])
        frame = encoded[encoded[CASE].isin(case_list)]
        labels = frame[LABEL].tolist()
        parts[key] = (frame.drop(columns=[CASE, LABEL]), labels, case_list)
    return parts


def odd_itemsets(traces: list) -> list:
    """Itemsets the Apriori selection never produces (stress for faithful mode).

    A single frequent activity, the single activity that repeats most inside
    traces, a pair contained in only 2-4 of ``traces`` (re-encoding of a
    small subset) and a pair / triple with many repeated occurrences.
    """
    support = Counter(act for trace in traces for act in set(trace))
    events = Counter(act for trace in traces for act in trace)
    frequent = [a for a, _ in sorted(support.items(), key=lambda kv: (-kv[1], kv[0]))]
    repeating = max((a for a in support if support[a] >= 20),
                    key=lambda a: (events[a] / support[a], a))
    counts = [Counter(trace) for trace in traces]

    def n_double(combo):
        return sum(min(count[a] for a in combo) >= 2 for count in counts)

    pairs = list(itertools.combinations(frequent[:25], 2))
    rare = next(
        list(pair) for pair in itertools.combinations(sorted(support), 2)
        if 2 <= sum(set(pair) <= set(trace) and len(trace) > 2 for trace in traces) <= 4)
    return [
        [frequent[0]],
        [repeating],
        rare,
        list(max(pairs, key=n_double)),
        list(max(itertools.combinations(frequent[:12], 3), key=n_double)),
    ]


def part_faithful(name: str, score_on: str, n_repeats: int = 3,
                  odd: bool = False) -> dict:
    manager = load_original_manager(DATASETS[name])
    with silenced():
        candidates, _ = manager.frequent_activity_sets(0.5, 10)
        encoded = manager.index_encoding(manager.data)
    parts = original_fold(manager, encoded)
    if odd:
        ordered = manager.data.sort_values([CASE, NR])
        scored_cases = set(parts[score_on][2])
        scored_traces = [group[ACT].tolist() for case, group in ordered.groupby(CASE)
                         if case in scored_cases]
        candidates = odd_itemsets(scored_traces)
        picks = tuple(range(len(candidates)))
    else:
        # f3 has only 5 Apriori itemsets: take all but the first there.
        picks = (PICKS if len(candidates) > max(PICKS)
                 else tuple(range(1, len(candidates))))
    itemsets = {pick: list(candidates[pick]) for pick in picks}
    train_x, train_y, train_list = parts["train"]
    model = XGBClassifier()
    model.fit(train_x, train_y)
    data_before = frame_digest(manager.data)

    scored_x, scored_y, scored_list = parts[score_on]
    start = time.perf_counter()
    with silenced():
        original = manager.itemset_permutation_importance(
            model, scored_x, scored_y, scored_list, itemsets,
            constrain=True, n_repeats=n_repeats)
    seconds_original = time.perf_counter() - start

    log = engine.EventLog(DATASETS[name])
    encoder = engine.IndexEncoder(
        log.traces.values(), log.activities, missing_order="hash")
    start = time.perf_counter()
    table = engine.LocationPermutationImportance(
        mode="faithful", score_on=score_on, n_repeats=n_repeats, random_state=2023,
    ).compute(model, log, encoder, train_list.tolist(),
              parts["test"][2].tolist(), itemsets, fold=FOLD)
    seconds_engine = time.perf_counter() - start

    rows = []
    for pick in picks:
        mine = table[table["itemset_id"] == pick].sort_values("repeat")
        for repeat in range(n_repeats):
            reference = float(original[pick].iloc[repeat])
            value = float(mine["importance"].iloc[repeat])
            rows.append({
                "itemset_id": pick, "itemset": engine.itemset_label(itemsets[pick]),
                "repeat": repeat, "original": reference, "engine": value,
                "abs_diff": abs(reference - value),
            })
    comparison = pd.DataFrame(rows)
    stem = f"faithful_{name}_{score_on}" + ("_odd" if odd else "")
    comparison.to_csv(HERE / f"verify_r2_{stem}.csv", index=False,
                      float_format="%.17g")
    baseline_mine = f1_score(scored_y, model.predict(scored_x), average="weighted")
    result = {
        "dataset": name, "score_on": score_on, "fold_seed": FOLD_SEED, "fold": FOLD,
        "n_scored_cases": len(scored_y), "n_train_cases": len(train_y),
        "itemsets": {str(k): sorted(v) for k, v in itemsets.items()},
        "n_repeats": n_repeats, "n_values": len(comparison),
        "max_abs_diff": float(comparison["abs_diff"].max()),
        "n_values_original_nonzero": int((comparison["original"] != 0).sum()),
        "n_distinct_original_values": int(comparison["original"].nunique()),
        "original_min": float(comparison["original"].min()),
        "original_max": float(comparison["original"].max()),
        "baseline_engine": float(table["baseline"].iloc[0]),
        "baseline_recomputed": float(baseline_mine),
        "baseline_abs_diff": abs(float(table["baseline"].iloc[0]) - baseline_mine),
        "manager_data_untouched_by_original": frame_digest(manager.data) == data_before,
        "seconds_original": round(seconds_original, 1),
        "seconds_original_per_iteration": round(seconds_original / len(comparison), 2),
        "seconds_engine": round(seconds_engine, 2),
        "seconds_engine_per_iteration": round(seconds_engine / len(comparison), 4),
    }
    result["pass"] = (result["max_abs_diff"] < TOLERANCE
                      and result["baseline_abs_diff"] < TOLERANCE)
    print(comparison.to_string(index=False))
    print(json.dumps(result, indent=2))
    write_json(stem, result)
    return result


def part_single(name: str = "f3", n_repeats: int = 2) -> dict:
    """Faithful single-activity routine vs the original, held-out fold."""
    manager = load_original_manager(DATASETS[name])
    with silenced():
        encoded = manager.index_encoding(manager.data)
    parts = original_fold(manager, encoded)
    train_x, train_y, train_list = parts["train"]
    test_x, test_y, test_list = parts["test"]
    model = XGBClassifier()
    model.fit(train_x, train_y)

    start = time.perf_counter()
    with silenced():
        original = manager.trace_permutation_importance(
            model, test_x, test_y, test_list, constrain=True, n_repeats=n_repeats)
    seconds_original = time.perf_counter() - start

    log = engine.EventLog(DATASETS[name])
    encoder = engine.IndexEncoder(
        log.traces.values(), log.activities, missing_order="hash")
    start = time.perf_counter()
    table = engine.LocationPermutationImportance(
        mode="faithful", score_on="test", n_repeats=n_repeats, random_state=2023,
    ).compute_single_activities(
        model, log, encoder, train_list.tolist(), test_list.tolist(), fold=FOLD)
    seconds_engine = time.perf_counter() - start

    diffs, n_nonzero = [], 0
    for act in log.activities:
        reference = [float(v) for v in original[act].tolist()]
        mine = table[table["itemset_id"] == act].sort_values("repeat")
        values = [float(v) for v in mine["importance"]]
        n_nonzero += sum(v != 0 for v in reference)
        diffs.extend(abs(a - b) for a, b in zip(reference, values))
        if len(reference) != len(values):
            diffs.append(float("inf"))
    result = {
        "dataset": name, "score_on": "test", "n_scored_cases": len(test_y),
        "n_activities": len(log.activities), "n_repeats": n_repeats,
        "n_values": len(diffs), "max_abs_diff": max(diffs),
        "n_values_original_nonzero": n_nonzero,
        "guards": table["guard"].value_counts().to_dict(),
        "seconds_original": round(seconds_original, 1),
        "seconds_engine": round(seconds_engine, 2),
    }
    result["pass"] = result["max_abs_diff"] < TOLERANCE
    print(json.dumps(result, indent=2))
    write_json(f"single_{name}", result)
    return result


# --------------------------------------------------------------------------
# Part 3: fixed mode
# --------------------------------------------------------------------------
_OCCURRENCE_CACHE: dict = {}


def cached_occurrences(trace, itemset) -> list:
    key = (tuple(trace), frozenset(itemset))
    if key not in _OCCURRENCE_CACHE:
        _OCCURRENCE_CACHE[key] = brute_occurrences(trace, itemset)
    return _OCCURRENCE_CACHE[key]


def check_permutation(trace, new_trace, itemset, placements, pools) -> list:
    """Names of the properties one permuted trace violates (empty = fine).

    ``pools``: activity -> set of 1-based positions where it was observed.
    ``placements``: per occurrence, (old index, new index) pairs (0-based).
    """
    problems = []
    length = len(trace)
    if len(new_trace) != length:
        problems.append("length")
    if Counter(new_trace) != Counter(trace):
        problems.append("multiset")

    # The moved events must be the original's complete occurrences.
    moved_old = [tuple(old for old, _ in placed) for placed in placements]
    if moved_old != cached_occurrences(trace, itemset):
        problems.append("moved_events_are_not_the_occurrences")
    all_old = [old for placed in placements for old, _ in placed]
    all_new = [new for placed in placements for _, new in placed]
    if len(set(all_old)) != len(all_old) or len(set(all_new)) != len(all_new):
        problems.append("two_events_on_one_position")

    for placed in placements:
        in_old_order = [trace[old] for old, _ in sorted(placed)]
        in_new_order = [trace[old] for old, _ in sorted(placed, key=lambda p: p[1])]
        if in_old_order != in_new_order:
            problems.append("ORDER_within_occurrence")
        for old, new in placed:
            if not 0 <= new < length:
                problems.append("position_outside_trace")
            elif new_trace[new] != trace[old]:
                problems.append("activity_not_at_reported_position")
            if (new + 1) not in pools.get(trace[old], ()):
                problems.append("FEASIBILITY_unobserved_position")

    old_set, new_set = set(all_old), set(all_new)
    if ([a for i, a in enumerate(trace) if i not in old_set]
            != [a for i, a in enumerate(new_trace) if i not in new_set]):
        problems.append("unmoved_events_changed_order")
    wanted = set(itemset)
    if ([a for a in trace if a not in wanted]
            != [a for a in new_trace if a not in wanted]):
        problems.append("non_itemset_events_changed_order")

    # Checks that do not trust ``placements`` at all (possible when every
    # itemset activity occurs exactly once in the trace).
    if all(trace.count(act) == 1 for act in wanted):
        if ([a for a in trace if a in wanted]
                != [a for a in new_trace if a in wanted]):
            problems.append("ORDER_placement_free")
        for act in wanted:
            if act in new_trace and (new_trace.index(act) + 1) not in pools.get(act, ()):
                problems.append("FEASIBILITY_placement_free")
    return problems


def assignment_exists(trace, occurrences, pools, budget: int = 300000):
    """Brute force: can all occurrences be placed at once? (None = gave up)."""
    slots = [(number, index) for number, occ in enumerate(occurrences) for index in occ]
    length = len(trace)
    counter = [0]

    def place(slot, taken, previous):
        if slot == len(slots):
            return True
        counter[0] += 1
        if counter[0] > budget:
            raise TimeoutError
        number, index = slots[slot]
        low = previous if slot and slots[slot - 1][0] == number else 0
        for position in sorted(pools.get(trace[index], ())):
            if low < position <= length and position not in taken:
                if place(slot + 1, taken | {position}, position):
                    return True
        return False

    try:
        return place(0, frozenset(), 0)
    except TimeoutError:
        return None


def choose_itemsets(traces: dict, seed: int) -> dict:
    """Per size 1, 2, 3: the 5 most frequent sets and 5 random ones.

    Candidates are combinations of the 25 activities contained in the most
    cases; random ones need at least 10 % case support.
    """
    rng = np.random.default_rng(seed)
    as_sets = [set(trace) for trace in traces.values()]
    support = Counter(act for acts in as_sets for act in acts)
    top = [a for a, _ in sorted(support.items(), key=lambda kv: (-kv[1], kv[0]))[:25]]
    chosen = {}
    for size in (1, 2, 3):
        scored = []
        for combo in itertools.combinations(top, size):
            wanted = set(combo)
            scored.append((sum(wanted <= acts for acts in as_sets), combo))
        scored.sort(key=lambda item: (-item[0], item[1]))
        frequent = [combo for _, combo in scored[:5]]
        rest = [combo for count, combo in scored[5:] if count >= 0.1 * len(as_sets)]
        picks = rng.choice(len(rest), size=min(5, len(rest)), replace=False)
        for number, combo in enumerate(frequent + [rest[i] for i in sorted(picks)]):
            chosen[f"size{size}_{number}"] = list(combo)
    return chosen


class RecordingModel:
    """Passes ``predict`` through and keeps a copy of every matrix."""

    def __init__(self, model, events: list):
        self._model = model
        self._events = events
        self.n_features_in_ = model.n_features_in_

    def predict(self, matrix):
        self._events.append(("predict", np.array(matrix, copy=True)))
        return self._model.predict(matrix)


def recorded_run(model, log, encoder, train, test, itemsets, fold, **settings):
    """``compute`` in fixed mode; returns (table, events, warnings).

    ``events`` lists, in call order, every ``permute_trace_fixed`` call
    (inputs before and after the call, outputs) and every matrix the model
    was asked to predict.
    """
    events: list = []
    real = engine.permute_trace_fixed

    def spy(trace, occurrences, allowed, rng, **options):
        before = list(trace)
        new_trace, placements = real(trace, occurrences, allowed, rng, **options)
        events.append(("permute", {
            "before": before, "after_call": list(trace),
            "occurrences": copy.deepcopy(occurrences), "allowed": allowed,
            "new": None if new_trace is None else list(new_trace),
            "placements": copy.deepcopy(placements),
        }))
        return new_trace, placements

    engine.permute_trace_fixed = spy
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            table = engine.LocationPermutationImportance(
                mode="fixed", **settings,
            ).compute(RecordingModel(model, events), log, encoder, train, test,
                      itemsets, fold=fold)
    finally:
        engine.permute_trace_fixed = real
    return table, events, [str(w.message) for w in caught]


def verify_run(table, events, mine: MyLog, scored, pools, itemsets, model,
               feature_names, n_repeats) -> dict:
    """Re-derive everything one recorded fixed-mode run claims."""
    stats: Counter = Counter()
    violations: Counter = Counter()
    examples = []
    traces = [mine.traces[case] for case in scored]
    y_true = [mine.labels[case] for case in scored]

    kind, baseline_matrix = events[0]
    stats["baseline_matrix_is_scored_fold"] = int(
        kind == "predict"
        and decode_rows(baseline_matrix, feature_names) == traces)
    base_pred = model.predict(naive_encode(traces, feature_names))
    baseline = f1_score(y_true, base_pred, average="weighted")
    stats["baseline_equals_sklearn"] = int(
        (table["baseline"] - baseline).abs().max() < TOLERANCE)
    stats["baseline_equals_by_hand"] = int(
        abs(weighted_f1_by_hand(y_true, base_pred) - baseline) < TOLERANCE)

    cursor, row_number = 1, 0
    max_score_diff = 0.0
    u_sum = u_var = 0.0
    for itemset_id, items in itemsets.items():
        wanted = set(items)
        eligible = [row for row, trace in enumerate(traces)
                    if wanted <= set(trace) and len(wanted) != len(trace)]
        for repeat in range(n_repeats):
            record = table.iloc[row_number]
            row_number += 1
            if record["itemset_id"] != itemset_id or record["repeat"] != repeat:
                violations["table_row_order"] += 1
            permuted, changed = list(traces), []
            n_infeasible = 0
            for row in eligible:
                kind, call = events[cursor]
                cursor += 1
                if kind != "permute":
                    violations["event_sequence"] += 1
                    continue
                stats["permute_calls"] += 1
                trace = traces[row]
                if call["before"] != trace:
                    violations["input_is_not_the_original_trace"] += 1
                if call["after_call"] != call["before"]:
                    violations["input_trace_mutated"] += 1
                if {a: set(p) for a, p in call["allowed"].items()} != pools:
                    violations["wrong_position_pools"] += 1
                if len(call["occurrences"]) > 1:
                    stats["calls_with_2plus_occurrences"] += 1
                if call["new"] is None:
                    n_infeasible += 1
                    stats["infeasible_calls"] += 1
                    exists = assignment_exists(trace, call["occurrences"], pools)
                    stats[f"infeasible_but_assignment_exists_{exists}"] += 1
                    continue
                problems = check_permutation(
                    trace, call["new"], wanted, call["placements"], pools)
                for problem in problems:
                    violations[problem] += 1
                if problems and len(examples) < 5:
                    examples.append({"itemset": sorted(wanted), "trace": trace,
                                     "new": call["new"], "problems": problems})
                if call["new"] != trace:
                    permuted[row] = call["new"]
                    changed.append(call["new"])
                    stats[f"changed_traces_size{len(wanted)}"] += 1
                    stats["changed_traces"] += 1
                    stats["moved_events"] += sum(len(p) for p in call["placements"])
                else:
                    stats["calls_trace_unchanged"] += 1
                if len(wanted) == 1 and len(call["occurrences"]) == 1:
                    act = trace[call["occurrences"][0][0]]
                    pool = sorted(p for p in pools[act] if p <= len(trace))
                    drawn = call["placements"][0][0][1] + 1
                    if drawn not in pool:
                        continue
                    u_sum += (pool.index(drawn) + 0.5) / len(pool) - 0.5
                    u_var += (1 - 1 / len(pool) ** 2) / 12
                    stats["single_draws"] += 1

            if changed:
                kind, matrix = events[cursor]
                cursor += 1
                if kind != "predict" or decode_rows(matrix, feature_names) != changed:
                    violations["predicted_rows_are_not_the_permuted_traces"] += 1
            pred = model.predict(naive_encode(permuted, feature_names))
            permuted_score = f1_score(y_true, pred, average="weighted")
            max_score_diff = max(
                max_score_diff,
                abs(permuted_score - record["permuted"]),
                abs((baseline - permuted_score) - record["importance"]))
            if (record["n_traces_with_itemset"] != len(eligible)
                    or record["n_traces_changed"] != len(changed)
                    or record["n_traces_unchanged_infeasible"] != n_infeasible):
                violations["count_columns"] += 1
    if cursor != len(events) or row_number != len(table):
        violations["unconsumed_events_or_rows"] += 1
    stats["blocks"] = row_number
    return {
        "stats": dict(stats), "violations": dict(violations),
        "max_abs_diff_recomputed_score": max_score_diff,
        "uniformity_z_single_draws": (u_sum / u_var ** 0.5) if u_var else None,
        "examples": examples,
    }


def new_traces_of(events) -> list:
    return [call["new"] for kind, call in events if kind == "permute"]


def part_fixed(name: str, n_repeats: int = 3) -> dict:
    result: dict = {"dataset": name, "fold_seed": FOLD_SEED, "fold": FOLD}
    mine = MyLog(DATASETS[name])
    log = engine.EventLog(DATASETS[name])
    snapshot = copy.deepcopy(log.traces)
    encoder = engine.IndexEncoder(log.traces.values(), log.activities)
    names = encoder.feature_names

    # Folds: a partition, stratified, reproducible, equal to sklearn's.
    folds = engine.make_folds(log.labels, k=5, seed=FOLD_SEED)
    cases = sorted(mine.traces)
    outcome = [mine.labels[case] for case in cases]
    reference = [
        ([cases[i] for i in tr], [cases[i] for i in te])
        for tr, te in StratifiedKFold(
            n_splits=5, shuffle=True, random_state=FOLD_SEED).split(cases, outcome)]
    test_union = [case for _, test in folds for case in test]
    share = sum(outcome) / len(outcome)
    result["folds"] = {
        "equal_to_sklearn_on_sorted_cases": folds == reference,
        "reproducible": folds == engine.make_folds(log.labels, k=5, seed=FOLD_SEED),
        "other_seed_differs": folds != engine.make_folds(log.labels, k=5, seed=FOLD_SEED + 1),
        "train_test_disjoint": all(not set(tr) & set(te) for tr, te in folds),
        "train_plus_test_is_all": all(sorted(tr + te) == cases for tr, te in folds),
        "test_folds_partition_the_log": sorted(test_union) == cases,
        "max_abs_dev_positive_share_in_test": max(
            abs(sum(mine.labels[c] for c in te) / len(te) - share) for _, te in folds),
    }
    train, test = folds[FOLD]

    # Model fitted by this script on the training cases only.
    model = XGBClassifier()
    model.fit(naive_encode([mine.traces[c] for c in train], names),
              [mine.labels[c] for c in train])
    itemsets = choose_itemsets(mine.traces, seed=FOLD_SEED)
    result["itemsets"] = itemsets
    result["n_train"], result["n_test"] = len(train), len(test)

    pools_log = mine.allowed
    pools_train = pools_of(mine.traces[c] for c in train)
    configurations = {
        "test_logpools": dict(score_on="test", allowed_from="log"),
        "train_logpools": dict(score_on="train", allowed_from="log"),
        "test_trainpools": dict(score_on="test", allowed_from="train"),
        "test_logpools_uniform": dict(score_on="test", allowed_from="log",
                                      draw="uniform"),
    }
    runs = {}
    for label, settings in configurations.items():
        start = time.perf_counter()
        table, events, caught = recorded_run(
            model, log, encoder, train, test, itemsets, FOLD,
            n_repeats=n_repeats, random_state=2023, **settings)
        seconds = time.perf_counter() - start
        scored = train if settings["score_on"] == "train" else test
        pools = pools_log if settings["allowed_from"] == "log" else pools_train
        checked = verify_run(table, events, mine, scored, pools, itemsets, model,
                             names, n_repeats)
        checked["seconds_engine_incl_recording"] = round(seconds, 2)
        checked["warnings"] = caught
        checked["importance_mean"] = float(table["importance"].mean())
        checked["importance_min"] = float(table["importance"].min())
        checked["importance_max"] = float(table["importance"].max())
        result[label] = checked
        runs[label] = (table, events)
        print(label, json.dumps(checked, indent=2), flush=True)

    # No leakage: with score_on='test' the model only ever sees held-out
    # cases (or permutations of them), and it was fitted without them.
    test_table, test_events = runs["test_logpools"]
    seen_rows = sum(len(m) for kind, m in test_events if kind == "predict")
    test_multisets = Counter(
        tuple(sorted(mine.traces[c])) for c in test)
    foreign = 0
    for kind, matrix in test_events:
        if kind == "predict":
            for trace in decode_rows(matrix, names):
                foreign += tuple(sorted(trace)) not in test_multisets
    result["leakage"] = {
        "train_test_overlap": len(set(train) & set(test)),
        "model_fitted_on_n_cases": len(train),
        "rows_shown_to_model": seen_rows,
        "rows_that_are_not_a_held_out_trace_or_its_permutation": foreign,
        "baseline_rows": len(test_events[0][1]),
    }

    # Determinism and seeds.
    settings = configurations["test_logpools"]
    again_table, again_events, _ = recorded_run(
        model, log, encoder, train, test, itemsets, FOLD,
        n_repeats=n_repeats, random_state=2023, **settings)
    other_table, other_events, _ = recorded_run(
        model, log, encoder, train, test, itemsets, FOLD,
        n_repeats=n_repeats, random_state=2024, **settings)
    fold_table, fold_events, _ = recorded_run(
        model, log, encoder, train, test, itemsets, FOLD + 1,
        n_repeats=n_repeats, random_state=2023, **settings)
    base = new_traces_of(test_events)

    def share_different(other_events_list):
        other = new_traces_of(other_events_list)
        return sum(a != b for a, b in zip(base, other)) / len(base)

    # One itemset alone, in reverse order, with more repeats: same numbers.
    key = "size2_3"
    alone, _, _ = recorded_run(
        model, log, encoder, train, test, {key: itemsets[key]}, FOLD,
        n_repeats=n_repeats + 2, random_state=2023, **settings)
    reverse, _, _ = recorded_run(
        model, log, encoder, train, test, dict(reversed(list(itemsets.items()))),
        FOLD, n_repeats=n_repeats, random_state=2023, **settings)
    wanted = test_table[test_table["itemset_id"] == key].reset_index(drop=True)
    columns = ["repeat", "permuted", "importance", "n_traces_changed"]
    reverse_sorted = reverse.sort_values(["itemset_id", "repeat"]).reset_index(drop=True)
    test_sorted = test_table.sort_values(["itemset_id", "repeat"]).reset_index(drop=True)
    # Repeats within one (itemset) must not be copies of each other.
    per_block: Counter = Counter()
    index = 0
    scored_traces = [mine.traces[c] for c in test]
    for itemset_id, items in itemsets.items():
        n_eligible = sum(set(items) <= set(t) and len(set(items)) != len(t)
                         for t in scored_traces)
        blocks = [base[index + r * n_eligible: index + (r + 1) * n_eligible]
                  for r in range(n_repeats)]
        index += n_repeats * n_eligible
        if n_eligible:
            per_block["itemsets"] += 1
            per_block["repeat0_equals_repeat1"] += blocks[0] == blocks[1]
    result["determinism"] = {
        "same_seed_table_identical": bool(test_table.equals(again_table)),
        "same_seed_traces_identical": base == new_traces_of(again_events),
        "other_random_state_share_of_calls_different": share_different(other_events),
        "other_random_state_table_identical": bool(test_table.equals(other_table)),
        "other_fold_number_share_of_calls_different": share_different(fold_events),
        "itemset_alone_more_repeats_same_values": bool(
            alone.iloc[:n_repeats][columns].equals(wanted[columns])),
        "reversed_itemset_order_same_values": bool(
            reverse_sorted.equals(test_sorted)),
        "repeats_of_one_itemset": dict(per_block),
    }
    result["log_traces_untouched"] = (
        log.traces == snapshot and log.traces == mine.traces)

    total_changed = sum(
        result[label]["stats"].get("changed_traces", 0) for label in configurations)
    total_violations = sum(
        sum(result[label]["violations"].values()) for label in configurations)
    result["total_changed_traces_checked"] = total_changed
    result["total_changed_by_size"] = {
        size: sum(result[label]["stats"].get(f"changed_traces_size{size}", 0)
                  for label in configurations) for size in (1, 2, 3)}
    result["total_violations"] = total_violations
    result["max_abs_diff_recomputed_score"] = max(
        result[label]["max_abs_diff_recomputed_score"] for label in configurations)
    result["pass"] = bool(
        total_violations == 0
        and total_changed >= 3000
        and result["max_abs_diff_recomputed_score"] < TOLERANCE
        and all(result[label]["stats"]["baseline_equals_sklearn"] == 1
                and result[label]["stats"]["baseline_equals_by_hand"] == 1
                and result[label]["stats"]["baseline_matrix_is_scored_fold"] == 1
                for label in configurations)
        and result["leakage"]["train_test_overlap"] == 0
        and result["leakage"]["rows_that_are_not_a_held_out_trace_or_its_permutation"] == 0
        and result["determinism"]["same_seed_table_identical"]
        and result["determinism"]["same_seed_traces_identical"]
        and result["determinism"]["other_random_state_share_of_calls_different"] > 0.5
        and result["determinism"]["itemset_alone_more_repeats_same_values"]
        and result["determinism"]["reversed_itemset_order_same_values"]
        and result["determinism"]["repeats_of_one_itemset"]["repeat0_equals_repeat1"] == 0
        and result["log_traces_untouched"]
        and all(v is True or not isinstance(v, bool) for v in result["folds"].values())
    )
    summary = {k: v for k, v in result.items() if k not in configurations}
    print(json.dumps(summary, indent=2, default=str))
    write_json(f"fixed_{name}", result)
    return result


# --------------------------------------------------------------------------
# Part: toy traces
# --------------------------------------------------------------------------
def all_outcomes(trace, occurrence, pools) -> set:
    """Every trace reachable by moving ONE occurrence within the constraints."""
    length = len(trace)
    rest = [act for index, act in enumerate(trace) if index not in occurrence]
    outcomes = set()
    candidate_pools = [
        [p for p in sorted(pools[trace[index]]) if p <= length] for index in occurrence]
    for positions in itertools.product(*candidate_pools):
        if list(positions) != sorted(set(positions)):
            continue
        others = iter(rest)
        placed = dict(zip(positions, occurrence))
        outcomes.add(tuple(
            trace[placed[p]] if p in placed else next(others)
            for p in range(1, length + 1)))
    return outcomes


def part_toy(n_draws: int = 20000) -> dict:
    trace = ["a", "x", "b", "y", "c", "z", "w"]
    pools = {"a": [1, 2, 4, 6], "b": [2, 3, 5], "c": [3, 5, 6, 7],
             "x": [2], "y": [4], "z": [6], "w": [7]}
    result = {}
    for itemset in (["a"], ["a", "b"], ["a", "b", "c"], ["c", "a"]):
        occurrences = engine.find_occurrences(trace, itemset)
        expected = all_outcomes(trace, occurrences[0], pools)
        for draw in ("sequential", "uniform"):
            rng = np.random.default_rng(5)
            seen: Counter = Counter()
            for _ in range(n_draws):
                new_trace, _ = engine.permute_trace_fixed(
                    trace, occurrences, pools, rng, draw=draw)
                seen[tuple(new_trace)] += 1
            counts = np.array([seen[outcome] for outcome in sorted(expected)])
            expected_count = n_draws / len(expected)
            chi2 = float(((counts - expected_count) ** 2 / expected_count).sum())
            result[f"{'+'.join(itemset)}|{draw}"] = {
                "n_feasible_outcomes": len(expected),
                "n_outcomes_seen": len(seen),
                "seen_equals_feasible_set": set(seen) == expected,
                "chi2_vs_uniform": round(chi2, 1),
                "chi2_degrees_of_freedom": len(expected) - 1,
                "min_share": round(float(counts.min()) / n_draws, 4),
                "max_share": round(float(counts.max()) / n_draws, 4),
            }
    # Two occurrences that compete for the same positions: the engine places
    # them one after the other and gives up on the whole trace on a dead end.
    trace2 = ["a", "b", "a", "b"]
    pools2 = {"a": [1, 3], "b": [2, 4]}
    rng = np.random.default_rng(5)
    outcomes: Counter = Counter()
    for _ in range(4000):
        new_trace, _ = engine.permute_trace_fixed(
            trace2, engine.find_occurrences(trace2, ["a", "b"]), pools2, rng)
        outcomes["None" if new_trace is None else "".join(new_trace)] += 1
    result["two_occurrences_abab"] = dict(outcomes)
    result["pass"] = all(
        v["seen_equals_feasible_set"] for k, v in result.items() if "|" in k)
    print(json.dumps(result, indent=2))
    write_json("toy", result)
    return result


# --------------------------------------------------------------------------
# Part: dead ends with several occurrences
# --------------------------------------------------------------------------
def part_deadend(n_repeats: int = 5) -> dict:
    """How often does fixed mode give up on a trace that could be permuted?

    With whole-log pools every trace has at least one valid assignment (its
    own positions), so every ``None`` from ``permute_trace_fixed`` is a dead
    end of the one-occurrence-after-the-other placement. Itemsets: per size
    2 and 3, the 10 sets (of the 25 most frequent activities) with the most
    traces holding two or more complete occurrences.
    """
    result = {}
    for name in ("f1", "f3"):
        mine = MyLog(DATASETS[name])
        traces = list(mine.traces.values())
        pools = {act: sorted(positions) for act, positions in mine.allowed.items()}
        support = Counter(act for trace in traces for act in set(trace))
        top = [a for a, _ in sorted(support.items(), key=lambda kv: (-kv[1], kv[0]))[:25]]
        counts = [Counter(trace) for trace in traces]
        per_size = {}
        for size in (2, 3):
            ranked = sorted(
                itertools.combinations(top, size),
                key=lambda combo: -sum(
                    min(count[a] for a in combo) >= 2 for count in counts))[:10]
            stats: Counter = Counter()
            rng = np.random.default_rng(1)
            for combo in ranked:
                wanted = set(combo)
                for trace in traces:
                    if not wanted <= set(trace) or len(wanted) == len(trace):
                        continue
                    occurrences = engine.find_occurrences(trace, wanted)
                    if len(occurrences) < 2:
                        continue
                    for draw in ("sequential", "uniform"):
                        for _ in range(n_repeats):
                            new_trace, placements = engine.permute_trace_fixed(
                                trace, occurrences, pools, rng, draw=draw)
                            stats[f"{draw}_calls"] += 1
                            if new_trace is None:
                                stats[f"{draw}_gave_up"] += 1
                            elif check_permutation(
                                    trace, new_trace, wanted, placements, mine.allowed):
                                stats[f"{draw}_violations"] += 1
            for draw in ("sequential", "uniform"):
                calls = stats[f"{draw}_calls"]
                stats[f"{draw}_gave_up_share"] = (
                    round(stats[f"{draw}_gave_up"] / calls, 5) if calls else None)
            per_size[f"size{size}"] = dict(stats)
        result[name] = per_size
    print(json.dumps(result, indent=2))
    write_json("deadend", result)
    return result


# --------------------------------------------------------------------------
# Part: how stable are fixed-mode rankings? (side information, not a check)
# --------------------------------------------------------------------------
def part_noise(n_repeats: int = 10) -> dict:
    """Fixed mode, 30 itemsets, 5 folds, two fold seeds, per log.

    Reports the level of the importance values on the held-out and on the
    training fold and the Spearman correlation of the per-itemset mean
    importance between two independent fold seeds.
    """
    from scipy.stats import spearmanr

    result = {}
    for name in DATASETS:
        log = engine.EventLog(DATASETS[name])
        encoder = engine.IndexEncoder(log.traces.values(), log.activities)
        itemsets = choose_itemsets(log.traces, seed=FOLD_SEED)
        means, out = {}, {}
        for fold_seed in (FOLD_SEED, FOLD_SEED + 1):
            tables = {"test": [], "train": []}
            for fold, (train, test) in enumerate(
                    engine.make_folds(log.labels, k=5, seed=fold_seed)):
                model = XGBClassifier()
                model.fit(encoder.transform(log.traces[c] for c in train),
                          [log.labels[c] for c in train])
                for score_on in tables:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        tables[score_on].append(engine.LocationPermutationImportance(
                            mode="fixed", score_on=score_on, n_repeats=n_repeats,
                        ).compute(model, log, encoder, train, test, itemsets, fold=fold))
            for score_on, parts in tables.items():
                table = pd.concat(parts, ignore_index=True)
                per_itemset = table.groupby("itemset_id", sort=True)["importance"].mean()
                means[(score_on, fold_seed)] = per_itemset
                out[f"{score_on}_seed{fold_seed}"] = {
                    "mean_importance": float(table["importance"].mean()),
                    "share_of_values_above_zero": float((table["importance"] > 0).mean()),
                    "share_of_itemset_means_above_zero": float((per_itemset > 0).mean()),
                    "mean_baseline": float(table["baseline"].mean()),
                }
        for score_on in ("test", "train"):
            rho = spearmanr(means[(score_on, FOLD_SEED)],
                            means[(score_on, FOLD_SEED + 1)]).statistic
            out[f"spearman_between_fold_seeds_{score_on}"] = float(rho)
        out["spearman_train_vs_test_same_seed"] = float(spearmanr(
            means[("train", FOLD_SEED)], means[("test", FOLD_SEED)]).statistic)
        result[name] = out
        print(name, json.dumps(out, indent=2), flush=True)
    write_json("noise", result)
    return result


# --------------------------------------------------------------------------
# Parts: digest / hashseed
# --------------------------------------------------------------------------
def part_digest(name: str = "f3") -> str:
    log = engine.EventLog(DATASETS[name])
    encoder = engine.IndexEncoder(log.traces.values(), log.activities)
    train, test = engine.make_folds(log.labels, k=5, seed=FOLD_SEED)[FOLD]
    model = XGBClassifier()
    model.fit(encoder.transform(log.traces[c] for c in train),
              [log.labels[c] for c in train])
    itemsets = choose_itemsets(log.traces, seed=FOLD_SEED)
    few = dict(list(itemsets.items())[::5])
    tables = [
        engine.LocationPermutationImportance(
            mode="fixed", score_on="test", n_repeats=2,
        ).compute(model, log, encoder, train, test, itemsets, fold=FOLD),
        engine.LocationPermutationImportance(
            mode="faithful", score_on="test", n_repeats=2,
        ).compute(model, log, encoder, train, test, few, fold=FOLD),
    ]
    digest = hashlib.sha256(
        "".join(frame_digest(t) for t in tables).encode("ascii")).hexdigest()
    print("DIGEST", digest)
    return digest


def part_hashseed() -> dict:
    digests = {}
    for seed in ("1", "2"):
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--part", "digest"],
            capture_output=True, text=True, check=True,
            env={**os.environ, "PYTHONHASHSEED": seed})
        digests[seed] = [line.split()[1] for line in completed.stdout.splitlines()
                         if line.startswith("DIGEST")][0]
    result = {"digests": digests,
              "identical_across_hash_seeds": len(set(digests.values())) == 1}
    result["pass"] = result["identical_across_hash_seeds"]
    print(json.dumps(result, indent=2))
    write_json("hashseed", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", required=True, choices=[
        "encoder", "faithful", "single", "fixed", "toy", "deadend", "noise",
        "digest",
        "hashseed"])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="f1")
    parser.add_argument("--score-on", choices=["train", "test"], default="train")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--odd", action="store_true",
                        help="faithful part: unusual itemsets instead of Apriori's")
    args = parser.parse_args()

    start = time.perf_counter()
    if args.part == "encoder":
        part_encoder()
    elif args.part == "faithful":
        part_faithful(args.dataset, args.score_on, args.repeats or 3, args.odd)
    elif args.part == "single":
        part_single(args.dataset, args.repeats or 2)
    elif args.part == "fixed":
        part_fixed(args.dataset, args.repeats or 3)
    elif args.part == "toy":
        part_toy()
    elif args.part == "deadend":
        part_deadend()
    elif args.part == "noise":
        part_noise(args.repeats or 10)
    elif args.part == "digest":
        part_digest()
    else:
        part_hashseed()
    print(f"part {args.part} finished in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
