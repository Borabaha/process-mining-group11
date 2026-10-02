"""Verification checks for engine.py (experiment C2).

Run (from anywhere, with the project's Python 3.12 venv):

    python test_engine.py           # all checks (20-40 minutes)
    python test_engine.py --quick   # skip the checks that run the slow original
    python test_engine.py --only edge            # checks whose name contains "edge"
    python test_engine.py --full-single --only faithful_single_full_f1
                                    # whole-fold single-activity proof (slow)

Every check compares the engine either with the ORIGINAL 2024 code (loaded
read-only through engine_reference.py) or with a property that must hold.
Results are printed and written to results/experiments/C2/:
test_engine_results.json, faithful_equality_<dataset>.csv and
faithful_single_equality_<dataset>_<cases>.csv.
"""

from __future__ import annotations

import argparse
import collections
import copy
import functools
import itertools
import json
import random
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import engine
import engine_reference as reference
from engine import (
    DATASETS,
    EventLog,
    IndexEncoder,
    LocationPermutationImportance,
    count_shifted_events,
    find_occurrences,
    make_folds,
    normalise_activity,
    observed_locations,
    permute_trace_fixed,
    shuffle_sequence_faithful,
    weighted_f1,
)

RESULT_DIR = Path(__file__).resolve().parents[1] / "results" / "experiments" / "C2"
FOLD_SEED = 0
# Positions (in the original Apriori output, min_support 0.5) of the three
# itemsets used for the faithful-equality proof: sizes 2, 3 and 2 / 2, 3, 2.
EQUALITY_PICKS = {"f1": [0, 3, 4], "f3": [0, 2, 3]}
# (cases, activities) left after the rare-activity filter with frq_threshold=2
# (verified earlier against the original DataManager).
EXPECTED_SHAPES = {"f1": (1130, 164), "f2": (1130, 207), "f3": (1111, 156)}


# --------------------------------------------------------------------------
# Cached fixtures
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def get_log(name: str) -> EventLog:
    return EventLog(DATASETS[name])


@functools.lru_cache(maxsize=None)
def get_manager(name: str):
    return reference.make_original_manager(DATASETS[name])


@functools.lru_cache(maxsize=None)
def get_original_encoding(name: str) -> pd.DataFrame:
    return reference.original_encoding(get_manager(name))


@functools.lru_cache(maxsize=None)
def get_encoder(name: str, missing_order: str = "hash") -> IndexEncoder:
    log = get_log(name)
    return IndexEncoder(log.traces.values(), log.activities, missing_order)


@functools.lru_cache(maxsize=None)
def get_apriori_itemsets(name: str) -> tuple:
    """The original's Apriori itemsets (min_support 0.5, top_k 10)."""
    with reference.quiet():
        candidates, _ = get_manager(name).frequent_activity_sets(0.5, 10)
    return tuple(tuple(itemset) for itemset in candidates)


@functools.lru_cache(maxsize=None)
def get_engine_model(name: str):
    """XGBoost fitted on fold 0 of the engine's own encoding ('sorted')."""
    log = get_log(name)
    encoder = get_encoder(name, "sorted")
    train, test = make_folds(log.labels, k=5, seed=FOLD_SEED)[0]
    model = XGBClassifier()
    model.fit(encoder.transform(log.traces[c] for c in train),
              [log.labels[c] for c in train])
    return model, encoder, train, test


# --------------------------------------------------------------------------
# Checks against the original
# --------------------------------------------------------------------------
def check_event_log(name: str) -> dict:
    """EventLog equals the original DataManager preprocessing."""
    log, manager = get_log(name), get_manager(name)
    data = manager.data.sort_values([manager.case_id, "event_nr"])
    grouped = data.groupby(manager.case_id, sort=False)
    original_traces = grouped[manager.activity].apply(list).to_dict()
    original_labels = {c: int(v) for c, v in grouped[manager.outcome].first().items()}

    found = (len(log.traces), len(log.activities))
    assert found == EXPECTED_SHAPES[name], f"{name}: {found} cases/activities"
    assert list(log.traces) == list(original_traces), "case ids / order differ"
    assert log.traces == original_traces, "traces differ"
    assert log.labels == original_labels, "labels differ"
    assert log.activities == manager.data[manager.activity].unique().tolist()
    assert log.allowed_locations == manager.Allowed_locations
    assert log.l_max == float(manager.L_max)
    return {"cases": len(log.traces), "activities": len(log.activities),
            "events": int(sum(map(len, log.traces.values()))),
            "max_len": log.max_len, "l_max": log.l_max,
            "label_1_cases": int(sum(log.labels.values()))}


def check_encoder(name: str) -> dict:
    """IndexEncoder equals the original index_encoding (after int cast)."""
    log, manager = get_log(name), get_manager(name)
    original = get_original_encoding(name)
    features = original.drop([manager.case_id, manager.outcome], axis=1)

    hashed = get_encoder(name, "hash")
    matrix = hashed.transform(log.traces.values())
    assert list(features.index) == log.case_ids, "row order differs"
    assert list(features.columns) == hashed.feature_names, "column order differs"
    assert np.array_equal(features.to_numpy(), matrix), "values differ"
    assert original[manager.outcome].tolist() == [log.labels[c] for c in log.case_ids]

    ordered = get_encoder(name, "sorted")
    assert set(ordered.feature_names) == set(features.columns)
    assert np.array_equal(features[ordered.feature_names].to_numpy(),
                          ordered.transform(log.traces.values()))
    n_observed = int((matrix.sum(axis=0) > 0).sum())
    assert ordered.feature_names[:n_observed] == hashed.feature_names[:n_observed]
    assert not matrix[:, n_observed:].any(), "never-observed block is not all zero"
    return {"shape": list(matrix.shape), "observed_columns": n_observed,
            "never_observed_columns": matrix.shape[1] - n_observed,
            "ones": int(matrix.sum()), "dtype": str(matrix.dtype)}


def check_same_model(name: str = "f1") -> dict:
    """A model fitted on the engine's matrix equals one fitted the original way.

    Compares predicted probabilities for all cases of three XGBoost models
    fitted on fold 0: original frame (int64, named columns), engine matrix
    with the original column order ('hash') and with the default 'sorted'
    order of the never-observed columns.
    """
    log, manager = get_log(name), get_manager(name)
    train, _ = make_folds(log.labels, k=5, seed=FOLD_SEED)[0]
    encoded = get_original_encoding(name)
    features, labels = reference.original_fold_data(manager, encoded, train)
    everything = encoded.drop([manager.case_id, manager.outcome], axis=1)
    probabilities = {
        "original": XGBClassifier().fit(features, labels).predict_proba(everything)}
    for order in ("hash", "sorted"):
        encoder = get_encoder(name, order)
        model = XGBClassifier().fit(
            encoder.transform(log.traces[c] for c in train),
            [log.labels[c] for c in train])
        probabilities[order] = model.predict_proba(
            encoder.transform(log.traces.values()))
    for order in ("hash", "sorted"):
        assert np.array_equal(probabilities["original"], probabilities[order]), order
    return {"cases_compared": len(everything), "max_abs_diff_hash": 0.0,
            "max_abs_diff_sorted": 0.0}


def check_reencoding(name: str = "f1", n_changed: int = 300) -> dict:
    """write_rows on changed traces equals the original encoding of them."""
    log, manager = get_log(name), get_manager(name)
    encoder = get_encoder(name, "hash")
    rng = random.Random(7)
    changed = rng.sample(range(len(log.case_ids)), n_changed)
    traces = [list(log.traces[c]) for c in log.case_ids]
    for row in changed:
        rng.shuffle(traces[row])

    # (1) whole log with changed traces vs the original encoding of that log
    matrix = encoder.transform(log.traces.values())
    encoder.write_rows(matrix, changed, [traces[row] for row in changed])
    assert np.array_equal(matrix, encoder.transform(traces))
    frame = manager.data.sort_values([manager.case_id, "event_nr"]).copy()
    frame[manager.activity] = [act for trace in traces for act in trace]
    with reference.quiet():
        original = manager.index_encoding(frame)
    assert set(original.columns) - {manager.case_id, manager.outcome} == set(
        encoder.feature_names)
    assert np.array_equal(original[encoder.feature_names].to_numpy(), matrix)

    # (2) the original's subset re-encoding (tools.py:529-534): padding stops
    #     at the longest trace of the subset.
    short_rows = sorted(row for row in changed if len(traces[row]) <= 20)
    subset_cases = [log.case_ids[row] for row in short_rows]
    with reference.quiet():
        subset = frame[frame[manager.case_id].isin(subset_cases)]
        corrupt = manager.index_encoding(subset)
        zero_features = [c for c in encoder.feature_names if c not in corrupt.columns]
        corrupt[zero_features] = 0
    longest = max(len(traces[row]) for row in short_rows)
    mine = encoder.transform([traces[row] for row in short_rows], pad_until=longest)
    assert list(corrupt.index) == subset_cases
    assert np.array_equal(corrupt[encoder.feature_names].to_numpy(), mine)
    return {"changed_traces": n_changed, "subset_traces": len(short_rows),
            "subset_longest_trace": longest, "log_longest_trace": log.max_len}


def check_find_occurrences(name: str = "f1") -> dict:
    """find_occurrences equals the original brute-force find_itemset_indexes."""
    manager, log = get_manager(name), get_log(name)
    rng = random.Random(11)
    n_synthetic = 0
    for _ in range(3000):
        alphabet = "abcde"[: rng.randint(2, 5)]
        trace = [rng.choice(alphabet) for _ in range(rng.randint(1, 12))]
        size = rng.randint(1, min(3, len(alphabet)))
        itemset = set(rng.sample(alphabet, size))
        assert find_occurrences(trace, itemset) == manager.find_itemset_indexes(
            trace, itemset), (trace, itemset)
        n_synthetic += 1

    n_real = 0
    cases = rng.sample(log.case_ids, 300)
    for itemset in get_apriori_itemsets(name):
        for case in cases:
            trace = log.traces[case]
            assert find_occurrences(trace, itemset) == manager.find_itemset_indexes(
                trace, set(itemset))
            n_real += 1
    return {"synthetic_pairs": n_synthetic, "real_pairs": n_real}


def check_shuffle_faithful(name: str) -> dict:
    """shuffle_sequence_faithful equals the original shuffle_sequence."""
    manager, log = get_manager(name), get_log(name)
    n_compared = n_moved = 0
    for itemset in get_apriori_itemsets(name):
        items = set(itemset)
        for seed, trace in enumerate(log.traces.values()):
            if not items.issubset(trace) or len(items) == len(trace):
                continue
            np.random.seed(seed)
            original = manager.shuffle_sequence(list(trace), items)
            mine = shuffle_sequence_faithful(
                trace, items, log.allowed_locations, np.random.RandomState(seed))
            assert mine == original, (itemset, trace)
            n_compared += 1
            n_moved += mine != trace
    return {"itemsets": len(get_apriori_itemsets(name)),
            "trace_itemset_pairs": n_compared, "pairs_changed": int(n_moved)}


def check_faithful_equality(name: str, n_repeats: int = 3) -> dict:
    """Faithful mode returns the original importance values (same model)."""
    log, manager = get_log(name), get_manager(name)
    encoder = get_encoder(name, "hash")
    encoded = get_original_encoding(name)
    train, test = make_folds(log.labels, k=5, seed=FOLD_SEED)[0]
    features, labels = reference.original_fold_data(manager, encoded, train)
    assert list(features.columns) == encoder.feature_names
    model = XGBClassifier()
    model.fit(features, labels)

    apriori = get_apriori_itemsets(name)
    itemsets = {i: list(apriori[i]) for i in EQUALITY_PICKS[name]}

    start = time.perf_counter()
    original = reference.original_importance(
        manager, model, features, labels, train, itemsets, n_repeats)
    seconds_original = time.perf_counter() - start

    start = time.perf_counter()
    table = LocationPermutationImportance(
        mode="faithful", score_on="train", n_repeats=n_repeats, random_state=2023,
    ).compute(model, log, encoder, train, test, itemsets, fold=0)
    seconds_engine = time.perf_counter() - start

    table["original_importance"] = [
        original[row.itemset_id].iloc[row.repeat] for row in table.itertuples()]
    table["abs_diff"] = (table["importance"] - table["original_importance"]).abs()
    table = table.rename(columns={"importance": "engine_importance"})
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULT_DIR / f"faithful_equality_{name}.csv", index=False,
                 float_format="%.17g")
    print(table[["itemset", "repeat", "original_importance", "engine_importance",
                 "abs_diff", "n_traces_with_itemset"]].to_string(
                     index=False, float_format=lambda v: f"{v:.17g}"))

    baseline_original = weighted_f1(labels, model.predict(features))
    max_diff = float(table["abs_diff"].max())
    assert table["baseline"].iloc[0] == baseline_original, "baseline differs"
    assert max_diff < 1e-12, f"max abs diff {max_diff}"
    n_iterations = len(itemsets) * n_repeats
    return {"train_cases": len(train), "itemsets": {k: v for k, v in itemsets.items()},
            "n_repeats": n_repeats, "baseline_train_f1": float(baseline_original),
            "max_abs_diff": max_diff,
            "seconds_per_iteration_original": seconds_original / n_iterations,
            "seconds_per_iteration_engine": seconds_engine / n_iterations}


# --------------------------------------------------------------------------
# Fixed-mode properties
# --------------------------------------------------------------------------
def _is_feasible_bruteforce(trace, occurrence, allowed) -> bool:
    """Is there any increasing tuple of allowed positions for one occurrence?"""
    pools = [[p for p in allowed.get(trace[i], ()) if p <= len(trace)]
             for i in occurrence]
    return any(all(a < b for a, b in zip(combo, combo[1:]))
               for combo in itertools.product(*pools))


def _assert_fixed_properties(trace, itemset, occurrences, new_trace, placements,
                             allowed) -> None:
    """All fixed-mode guarantees for one permuted trace."""
    assert len(new_trace) == len(trace), "length changed"
    assert collections.Counter(new_trace) == collections.Counter(trace), "multiset"
    assert [tuple(old for old, _ in placed) for placed in placements] == occurrences
    old_moved, new_moved = set(), set()
    for placed in placements:
        new_positions = [new for _, new in placed]
        assert new_positions == sorted(set(new_positions)), "order not preserved"
        for old, new in placed:
            act = trace[old]
            assert act in itemset
            assert new_trace[new] == act, "activity is not at its drawn position"
            assert (new + 1) in allowed[act], "position not allowed"
            assert new + 1 <= len(trace), "position beyond the trace"
            old_moved.add(old)
            new_moved.add(new)
    assert len(new_moved) == sum(len(placed) for placed in placements), "collision"
    others_before = [a for i, a in enumerate(trace) if i not in old_moved]
    others_after = [a for i, a in enumerate(new_trace) if i not in new_moved]
    assert others_before == others_after, "another event changed its relative order"
    # Check that does not use the reported placements: when every itemset
    # activity occurs exactly once, their order of appearance is unchanged.
    if all(trace.count(act) == 1 for act in itemset):
        assert ([a for a in trace if a in itemset]
                == [a for a in new_trace if a in itemset]), "order not preserved"


def _draw_itemset(trace, apriori, rng):
    """A random itemset contained in ``trace`` (Apriori set or random 1-3 set)."""
    if rng.random() < 0.5:
        contained = [s for s in apriori
                     if set(s).issubset(trace) and len(s) != len(trace)]
        if contained:
            return set(rng.choice(contained))
    distinct = sorted(set(trace))
    size = rng.randint(1, min(3, len(distinct)))
    if size == len(trace):
        return None
    return set(rng.sample(distinct, size))


def run_fixed_property_trials(name, cases, allowed, n_feasible, seed,
                              draw="sequential") -> dict:
    """Permute random (trace, itemset) pairs until ``n_feasible`` were checked."""
    log = get_log(name)
    allowed_sets = {act: set(positions) for act, positions in allowed.items()}
    apriori = get_apriori_itemsets(name)
    rng, generator = random.Random(seed), np.random.default_rng(seed)
    stats = collections.Counter()
    while stats["feasible_checked"] < n_feasible:
        trace = log.traces[rng.choice(cases)]
        itemset = _draw_itemset(trace, apriori, rng)
        if itemset is None:
            continue
        snapshot = list(trace)
        occurrences = find_occurrences(trace, itemset)
        new_trace, placements = permute_trace_fixed(trace, occurrences, allowed,
                                                    generator, draw)
        assert trace == snapshot, "input trace was mutated"
        if new_trace is None:
            stats["infeasible"] += 1
            if len(occurrences) == 1:
                assert not _is_feasible_bruteforce(trace, occurrences[0], allowed)
                stats["infeasible_confirmed_by_bruteforce"] += 1
            continue
        _assert_fixed_properties(trace, itemset, occurrences, new_trace, placements,
                                 allowed)
        stats["feasible_checked"] += 1
        stats[f"size_{len(itemset)}"] += 1
        stats["multi_occurrence"] += len(occurrences) > 1
        stats["identical_to_original"] += new_trace == trace
        n_moved = sum(len(placed) for placed in placements)
        stats["moved_events"] += n_moved
        n_shifted, n_unobserved = count_shifted_events(trace, placements,
                                                       allowed_sets)
        assert 0 <= n_unobserved <= n_shifted <= len(trace) - n_moved
        stats["other_events"] += len(trace) - n_moved
        stats["other_events_shifted"] += n_shifted
        stats["other_events_unobserved"] += n_unobserved
    return dict(stats)


def check_fixed_properties(name: str = "f1") -> dict:
    """2000 random fixed-mode permutations satisfy every guarantee."""
    log = get_log(name)
    whole_log = run_fixed_property_trials(
        name, log.case_ids, log.allowed_locations, n_feasible=2000, seed=101)
    train, test = make_folds(log.labels, k=5, seed=FOLD_SEED)[0]
    train_pools = observed_locations(log.traces[c] for c in train)
    held_out = run_fixed_property_trials(
        name, test, train_pools, n_feasible=1000, seed=202)
    return {"allowed_from_log_all_cases": whole_log,
            "allowed_from_train_on_test_cases": held_out}


def _three_itemsets(name: str) -> dict:
    apriori = get_apriori_itemsets(name)
    return {i: list(apriori[i]) for i in EQUALITY_PICKS[name]}


def check_determinism(name: str = "f1") -> dict:
    """Same seed -> identical table; other seed -> different draws."""
    model, encoder, train, test = get_engine_model(name)
    log, itemsets = get_log(name), _three_itemsets(name)

    def run(mode, score_on, seed):
        return LocationPermutationImportance(
            mode=mode, score_on=score_on, n_repeats=3, random_state=seed,
        ).compute(model, log, encoder, train, test, itemsets, fold=0)

    fixed_a, fixed_b = run("fixed", "test", 2023), run("fixed", "test", 2023)
    pd.testing.assert_frame_equal(fixed_a, fixed_b, check_exact=True)
    other_seed = run("fixed", "test", 7)
    assert not fixed_a["permuted"].equals(other_seed["permuted"])
    faithful_a = run("faithful", "train", 2023)
    faithful_b = run("faithful", "train", 2023)
    pd.testing.assert_frame_equal(faithful_a, faithful_b, check_exact=True)
    return {"rows_compared_fixed": len(fixed_a),
            "rows_compared_faithful": len(faithful_a),
            "fixed_importance_seed_2023": fixed_a["importance"].round(6).tolist(),
            "fixed_importance_seed_7": other_seed["importance"].round(6).tolist()}


def check_no_accumulation(name: str = "f1") -> dict:
    """Fixed mode always permutes the ORIGINAL traces."""
    model, encoder, train, test = get_engine_model(name)
    log, itemsets = get_log(name), _three_itemsets(name)
    snapshot = copy.deepcopy(log.traces)
    original_objects = {id(trace) for trace in log.traces.values()}
    seen = collections.Counter()
    real_permute = engine.permute_trace_fixed

    def recording_permute(trace, *args, **kwargs):
        assert id(trace) in original_objects, "a non-original trace was permuted"
        seen["calls"] += 1
        return real_permute(trace, *args, **kwargs)

    def run(sets, n_repeats, mode="fixed", score_on="test"):
        return LocationPermutationImportance(
            mode=mode, score_on=score_on, n_repeats=n_repeats, random_state=2023,
        ).compute(model, log, encoder, train, test, sets, fold=0)

    engine.permute_trace_fixed = recording_permute
    try:
        full = run(itemsets, 3)
    finally:
        engine.permute_trace_fixed = real_permute
    assert log.traces == snapshot, "the log's traces were modified"

    # A result must not depend on what was computed before it.
    ids = list(itemsets)
    last_alone = run({ids[-1]: itemsets[ids[-1]]}, 3)
    expected = full[full["itemset_id"] == ids[-1]].reset_index(drop=True)
    pd.testing.assert_frame_equal(last_alone, expected, check_exact=True)
    two_repeats = run(itemsets, 2)
    expected = full[full["repeat"] < 2].reset_index(drop=True)
    pd.testing.assert_frame_equal(two_repeats, expected, check_exact=True)

    # Contrast (not an assertion of correctness): faithful mode does depend on it.
    faithful_full = run(itemsets, 3, "faithful", "train")
    faithful_alone = run({ids[-1]: itemsets[ids[-1]]}, 3, "faithful", "train")
    assert log.traces == snapshot, "faithful mode modified the log's traces"
    after_others = faithful_full[faithful_full["itemset_id"] == ids[-1]]
    return {"permute_calls_checked": seen["calls"],
            "fixed_last_itemset_alone_equals_after_others": True,
            "faithful_last_itemset_alone": faithful_alone["importance"].tolist(),
            "faithful_last_itemset_after_others": after_others["importance"].tolist()}


# --------------------------------------------------------------------------
# Edge cases, input validation, single activities, uniform draw
# --------------------------------------------------------------------------
def _expect(error_type, fragment, function) -> str:
    """Assert that ``function()`` raises ``error_type`` mentioning ``fragment``."""
    try:
        function()
    except error_type as error:
        assert fragment in str(error), f"{fragment!r} not in {str(error)!r}"
        return f"{error_type.__name__}: {error}"
    raise AssertionError(f"no {error_type.__name__} raised")


def _fixed(score_on="test", n_repeats=2, **settings):
    return LocationPermutationImportance(
        mode="fixed", score_on=score_on, n_repeats=n_repeats, random_state=2023,
        **settings)


def check_edge_cases(name: str = "f1") -> dict:
    """Nothing-changed situations give importance 0 instead of a crash."""
    model, encoder, train, test = get_engine_model(name)
    log = get_log(name)

    # Encoder: empty input, generator input, wrong input.
    assert encoder.transform([]).shape == (0, encoder.n_features)
    some = [log.traces[case] for case in log.case_ids[:5]]
    expected = encoder.transform(some)
    assert np.array_equal(encoder.transform(trace for trace in some), expected)
    matrix = np.zeros_like(expected)
    encoder.write_rows(matrix, range(5), (trace for trace in some))
    assert np.array_equal(matrix, expected), "generator input encoded differently"
    encoder.write_rows(matrix, [], [])
    assert np.array_equal(matrix, expected), "empty write changed the matrix"
    messages = {
        "trace_too_long": _expect(ValueError, "at most", lambda: encoder.transform(
            [[log.activities[0]] * (encoder.max_len + 1)])),
        "unknown_activity": _expect(ValueError, "unknown to the encoder",
                                    lambda: encoder.transform([["no_such_activity"]])),
        "rows_traces_mismatch": _expect(ValueError, "rows for",
                                        lambda: encoder.write_rows(matrix, [0], some)),
    }

    def compute(itemsets, **settings):
        return _fixed(**settings).compute(model, log, encoder, train, test,
                                          itemsets, fold=0)

    # (1) An itemset contained in no trace of the scored (held-out) fold.
    in_test = {act for case in test for act in log.traces[case]}
    absent = [act for act in log.activities if act not in in_test]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        table = compute([[absent[0]]])
    assert len(table) == 2 and (table["importance"] == 0).all()
    assert (table["n_traces_with_itemset"] == 0).all()
    assert len(caught) == 1 and "occur in no scored trace" in str(caught[0].message)

    # (2) An activity with one allowed position: present, but it cannot move.
    one_place = [act for act in log.activities
                 if len(log.allowed_locations[act]) == 1]
    single = compute([[one_place[0]]], score_on="train")
    assert (single["importance"] == 0).all()
    assert (single["n_traces_with_itemset"] > 0).all()
    assert (single["n_traces_changed"] == 0).all()

    # (3) Such an itemset does not cost the other itemsets their results.
    normal = list(get_apriori_itemsets(name)[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mixed = compute({"gone": [absent[0]], "normal": normal})
    alone = compute({"normal": normal})
    pd.testing.assert_frame_equal(
        mixed[mixed["itemset_id"] == "normal"].reset_index(drop=True), alone,
        check_exact=True)
    return {"activities_absent_from_held_out_fold": len(absent),
            "activities_with_one_allowed_position": len(one_place),
            "error_messages": messages}


def check_single_activity_grid(name: str = "f1") -> dict:
    """Every single activity runs on the held-out fold; no change -> exactly 0."""
    model, encoder, train, test = get_engine_model(name)
    log = get_log(name)
    out = {}
    for allowed_from in ("log", "train"):
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            table = _fixed(allowed_from=allowed_from).compute_single_activities(
                model, log, encoder, train, test, fold=0)
        seconds = time.perf_counter() - start
        assert len(table) == 2 * len(log.activities)
        assert list(table["itemset_id"].unique()) == log.activities
        unchanged = table[table["n_traces_changed"] == 0]
        assert (unchanged["importance"] == 0).all()
        out[f"allowed_from_{allowed_from}"] = {
            "rows": len(table),
            "rows_without_changed_trace": len(unchanged),
            "activities_in_no_scored_trace": int(
                table.loc[table["n_traces_with_itemset"] == 0, "itemset_id"].nunique()),
            "infeasible_traces": int(table["n_traces_unchanged_infeasible"].sum()),
            "rows_with_nonzero_importance": int((table["importance"] != 0).sum()),
            "seconds": round(seconds, 2),
        }
    # compute_single_activities is compute() with one set per activity.
    picked = log.activities[:5]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        by_method = _fixed().compute_single_activities(
            model, log, encoder, train, test, fold=0, activities=picked)
        by_compute = _fixed().compute(
            model, log, encoder, train, test, {a: [a] for a in picked}, fold=0)
    pd.testing.assert_frame_equal(by_method, by_compute, check_exact=True)
    return out


def check_input_validation(name: str = "f1") -> dict:
    """Wrong itemsets, a forgotten fold and a foreign model raise clear errors."""
    model, encoder, train, test = get_engine_model(name)
    log = get_log(name)
    importance = _fixed()
    first = log.activities[0]

    def compute(itemsets, engine_object=importance, used_model=model,
                used_encoder=encoder, **kwargs):
        kwargs.setdefault("fold", 0)
        return lambda: engine_object.compute(
            used_model, log, used_encoder, train, test, itemsets, **kwargs)

    assert normalise_activity("AC-123 x_Y") == "ac123xy"
    messages = {
        "itemsets_is_a_string": _expect(TypeError, "collection", compute(first)),
        "itemset_is_a_string": _expect(TypeError, "is the string", compute([first])),
        "unknown_activity": _expect(ValueError, "not activities of the log",
                                    compute([["no_such_activity"]])),
        "name_not_normalised": _expect(ValueError, f"did you mean {first!r}",
                                       compute([[first.upper()]])),
        "empty_itemset": _expect(ValueError, "is empty", compute([[]])),
    }
    # fold is a required keyword argument.
    arguments = (model, log, encoder, train, test, [[first]])
    for call in (lambda: importance.compute(*arguments),
                 lambda: importance.compute(*arguments, 0)):
        try:
            call()
        except TypeError as error:
            messages.setdefault("fold_missing_or_positional", []).append(str(error))
        else:
            raise AssertionError("compute() accepted a call without fold=")

    # Faithful mode refuses an itemset that no scored trace contains (the
    # original crashes), before any other itemset is computed.
    in_train = {act for case in train for act in log.traces[case]}
    absent = [act for act in log.activities if act not in in_train]
    faithful = LocationPermutationImportance(mode="faithful", score_on="train",
                                             n_repeats=1)
    if absent:
        messages["faithful_absent_itemset"] = _expect(
            ValueError, "no scored trace", compute(
                [list(get_apriori_itemsets(name)[0]), [absent[0]]], faithful))

    # A model fitted on the original frame (hash order of the never-observed
    # columns) must not be used with the default ('sorted') encoder.
    manager, encoded = get_manager(name), get_original_encoding(name)
    features, labels = reference.original_fold_data(manager, encoded, train)
    foreign = XGBClassifier(n_estimators=5).fit(features, labels)
    hashed = get_encoder(name, "hash")
    assert list(features.columns) == hashed.feature_names
    if hashed.feature_names != encoder.feature_names:
        messages["model_fitted_on_other_column_order"] = _expect(
            ValueError, "different order", compute([[first]], used_model=foreign))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compute([[first]], used_model=foreign, used_encoder=hashed)()  # accepted
    narrow = XGBClassifier(n_estimators=2).fit(
        encoder.transform(log.traces[c] for c in train)[:, :50],
        [log.labels[c] for c in train])
    messages["model_with_other_feature_count"] = _expect(
        ValueError, "fitted on 50 features", compute([[first]], used_model=narrow))
    return messages


def check_shifted_event_count() -> dict:
    """count_shifted_events on two hand-made permutations."""
    trace = ["a", "b", "c", "d"]
    # a moves from index 0 to index 2: new trace b, c, a, d.
    allowed = {"b": {1}, "c": {3}, "d": {4}}
    first = count_shifted_events(trace, [[(0, 2)]], allowed)
    assert first == (2, 1), first  # b and c shift; c lands on unobserved position 2
    # d moves from index 3 to index 0: new trace d, a, b, c (c has no pool).
    second = count_shifted_events(trace, [[(3, 0)]], {"a": {2}, "b": {2, 3}})
    assert second == (3, 1), second
    assert count_shifted_events(trace, [[(1, 1)]], {}) == (0, 0)
    return {"example_1": list(first), "example_2": list(second)}


def _feasible_tuples(trace, occurrence, allowed) -> list:
    """All strictly increasing tuples of allowed positions for one occurrence."""
    pools = [[p for p in allowed[trace[i]] if p <= len(trace)] for i in occurrence]
    return [combo for combo in itertools.product(*pools)
            if all(a < b for a, b in zip(combo, combo[1:]))]


def check_uniform_draw(name: str = "f1") -> dict:
    """draw='uniform' is uniform over the feasible tuples; properties hold."""
    log = get_log(name)
    properties = run_fixed_property_trials(
        name, log.case_ids, log.allowed_locations, n_feasible=2000, seed=303,
        draw="uniform")

    out = {"property_trials": properties, "uniformity": {}}
    apriori = get_apriori_itemsets(name)
    for size in (1, 2, 3):
        itemset = set(sorted(next(s for s in apriori if len(s) == 3))[:size])
        trace = next(t for t in log.traces.values()
                     if len(t) >= 15 and len(find_occurrences(t, itemset)) == 1)
        (occurrence,) = find_occurrences(trace, itemset)
        tuples = _feasible_tuples(trace, occurrence, log.allowed_locations)
        n_draws = max(20000, 20 * len(tuples))
        expected = n_draws / len(tuples)
        result = {"trace_length": len(trace), "feasible_tuples": len(tuples),
                  "draws": n_draws, "degrees_of_freedom": len(tuples) - 1}
        for draw in ("uniform", "sequential"):
            generator = np.random.default_rng(404)
            seen = collections.Counter()
            for _ in range(n_draws):
                _, placements = permute_trace_fixed(
                    trace, [occurrence], log.allowed_locations, generator, draw)
                seen[tuple(new + 1 for _, new in placements[0])] += 1
            assert set(seen) <= set(tuples), "a drawn tuple is not feasible"
            chi2 = sum((seen[t] - expected) ** 2 / expected for t in tuples)
            result[f"chi2_{draw}"] = round(chi2, 1)
            result[f"tuples_never_drawn_{draw}"] = len(tuples) - len(seen)
        # Under uniformity chi2 has mean dof and standard deviation
        # sqrt(2 dof); 5 standard deviations is a generous limit.
        dof = len(tuples) - 1
        assert result["chi2_uniform"] < dof + 5 * (2 * dof) ** 0.5, result
        out["uniformity"][f"size_{size}"] = result
    return out


# --------------------------------------------------------------------------
# Faithful single-activity routine against the original
# --------------------------------------------------------------------------
def check_faithful_single_equality(name: str, n_cases: int | None = 200,
                                   n_repeats: int = 2) -> dict:
    """Faithful ``compute_single_activities`` = ``trace_permutation_importance``.

    Same fitted model for both (fitted on the whole training fold). The
    original loops over every activity of the log and filters the frame once
    per case, so ``n_cases`` limits the scored cases to the first ``n_cases``
    of the training fold to keep the check short (None = whole fold).
    """
    log, manager = get_log(name), get_manager(name)
    encoder = get_encoder(name, "hash")
    encoded = get_original_encoding(name)
    train, test = make_folds(log.labels, k=5, seed=FOLD_SEED)[0]
    features, labels = reference.original_fold_data(manager, encoded, train)
    model = XGBClassifier()
    model.fit(features, labels)
    cases = train if n_cases is None else train[:n_cases]
    features, labels = reference.original_fold_data(manager, encoded, cases)

    start = time.perf_counter()
    original = reference.original_single_importance(
        manager, model, features, labels, cases, n_repeats)
    seconds_original = time.perf_counter() - start

    start = time.perf_counter()
    table = LocationPermutationImportance(
        mode="faithful", score_on="train", n_repeats=n_repeats, random_state=2023,
    ).compute_single_activities(model, log, encoder, cases, test, fold=0)
    seconds_engine = time.perf_counter() - start

    assert set(original.columns) == set(log.activities)
    table["original_importance"] = [
        float(original[row.itemset_id].iloc[row.repeat]) for row in table.itertuples()]
    table["abs_diff"] = (table["importance"] - table["original_importance"]).abs()
    table = table.rename(columns={"importance": "engine_importance"})
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        RESULT_DIR / f"faithful_single_equality_{name}_{len(cases)}cases.csv",
        index=False, float_format="%.17g")

    max_diff = float(table["abs_diff"].max())
    assert table["baseline"].iloc[0] == weighted_f1(labels, model.predict(features))
    assert max_diff < 1e-12, f"max abs diff {max_diff}"
    guards = table["guard"].value_counts().to_dict()
    n_iterations = len(table)
    return {"scored_cases": len(cases), "activities": len(log.activities),
            "n_repeats": n_repeats, "values_compared": n_iterations,
            "values_nonzero_in_original": int(
                (table["original_importance"] != 0).sum()),
            "guard_rows": {key or "none": value for key, value in guards.items()},
            "baseline_f1": float(table["baseline"].iloc[0]),
            "max_abs_diff": max_diff,
            "largest_original_importance": float(table["original_importance"].max()),
            "seconds_per_iteration_original": seconds_original / n_iterations,
            "seconds_per_iteration_engine": seconds_engine / n_iterations}


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def build_checks(quick: bool, full_single: bool = False):
    checks = []
    for name in ("f1", "f2", "f3"):
        checks.append((f"event_log_{name}", functools.partial(check_event_log, name)))
        checks.append((f"encoder_{name}", functools.partial(check_encoder, name)))
    checks.append(("same_model_f1", check_same_model))
    checks.append(("reencoding_f1", check_reencoding))
    checks.append(("find_occurrences_f1", check_find_occurrences))
    checks.append(("fixed_properties_f1", check_fixed_properties))
    checks.append(("uniform_draw_f1", check_uniform_draw))
    checks.append(("shifted_event_count", check_shifted_event_count))
    checks.append(("determinism_f1", check_determinism))
    checks.append(("no_accumulation_f1", check_no_accumulation))
    checks.append(("edge_cases_f1", check_edge_cases))
    checks.append(("single_activity_grid_f1", check_single_activity_grid))
    checks.append(("single_activity_grid_f3",
                   functools.partial(check_single_activity_grid, "f3")))
    checks.append(("input_validation_f1", check_input_validation))
    if not quick:
        for name in ("f1", "f3"):
            checks.append((f"shuffle_faithful_{name}",
                           functools.partial(check_shuffle_faithful, name)))
            checks.append((f"faithful_equality_{name}",
                           functools.partial(check_faithful_equality, name)))
        # 200 scored cases; the original needs about 13 minutes for them.
        checks.append(("faithful_single_equality_f3",
                       functools.partial(check_faithful_single_equality, "f3")))
    if full_single:
        for name in ("f1", "f3"):
            checks.append((f"faithful_single_full_{name}", functools.partial(
                check_faithful_single_equality, name, n_cases=None)))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="skip the checks that run the slow original importance")
    parser.add_argument("--full-single", action="store_true",
                        help="add the whole-training-fold single-activity proof "
                             "against the original (slow)")
    parser.add_argument("--only", default="",
                        help="run only the checks whose name contains this text")
    args = parser.parse_args()

    results = {}
    for check_name, check in build_checks(args.quick, args.full_single):
        if args.only not in check_name:
            continue
        start = time.perf_counter()
        try:
            details, status = check(), "PASS"
        except Exception:  # noqa: BLE001 - report every failing check
            details, status = traceback.format_exc(), "FAIL"
        seconds = time.perf_counter() - start
        results[check_name] = {"status": status, "seconds": round(seconds, 2),
                               "details": details}
        print(f"[{status}] {check_name} ({seconds:.1f} s)", flush=True)
        print(f"       {details}", flush=True)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ("_quick" if args.quick else "") + (f"_{args.only}" if args.only else "")
    out_name = f"test_engine_results{suffix}.json"
    (RESULT_DIR / out_name).write_text(json.dumps(results, indent=2), encoding="utf-8")
    failed = [name for name, result in results.items() if result["status"] == "FAIL"]
    print(f"{len(results) - len(failed)} passed, {len(failed)} failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
