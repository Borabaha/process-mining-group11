"""Adversarial verification of ``experiments/impressed_chain.py`` (experiment C4).

Run from anywhere (about 3-6 minutes, most of it inside the ORIGINAL 2023 code):

    python results/experiments/C4/verify_impressed.py

Nothing here re-uses the builder's cross-check script.  The checks are:

1. paper / code rules on hand-made toy traces (expected children written by hand);
2. the original ``IMIPD`` functions on a DIFFERENT random subset (200 f1 cases,
   seed 4711) with DIFFERENT seed activities, plus the stored full-log run of
   the original automatic mode (experiment C3);
3. projection and selection against a tiny independent implementation;
4. which cases are mined and where the labels enter;
5. a few mechanical code-quality checks.

Outputs next to this file: ``verify_impressed.json`` (all numbers) and the
console log.  Every check is recorded as PASS / FAIL / INFO; nothing is fixed
or hidden here.
"""
from __future__ import annotations

import argparse
import contextlib
import inspect
import io
import itertools
import json
import math
import os
import pickle
import subprocess
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from paretoset import paretoset  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
EXPERIMENTS = PROJECT_ROOT / "experiments"
ORIGINAL_2023 = PROJECT_ROOT / "external" / "InteractivePatternDetection"
DATASET_DIR = PROJECT_ROOT / "external" / "PermutationLocationImportance" / "datasets"
C3_DIR = PROJECT_ROOT / "results" / "experiments" / "C3"
DATASET_FILES = {"f1": "BPIC11_f1_trunc36.csv", "f2": "BPIC11_f2_trunc40.csv", "f3": "BPIC11_f3_trunc31.csv"}

# The 2023 repo's ``tools.py`` must be the only module called ``tools`` on the path.
sys.path.insert(0, str(ORIGINAL_2023))
sys.path.insert(0, str(EXPERIMENTS))

import impressed_chain as ic  # noqa: E402
from case_distance import pairwise_case_distance, pattern_case_distance  # noqa: E402
from impressed_chain import DIRECT, EVENTUAL, ImpressedChainSelector, Pattern  # noqa: E402

CASE, ACT, TIME, LABEL, POS = "case:concept:name", "concept:name", "time:timestamp", "label", "event_nr"
NUMERIC = ["Age"]
CATEGORICAL = ["Diagnosis", "Treatment code", "Diagnosis code", "Specialism code"]
MAX_GAP = 3
SUBSET_SIZE, SUBSET_SEED = 200, 4711
BUILDER_SEEDS = {"ac370000", "ac419100", "370488j", "ac370419", "370407"}  # must NOT be re-used here
OBJ = ["IG", "coverage", "CD"]

CHECKS: list[dict] = []
RESULTS: dict = {}


def say(message: str) -> None:
    """Time-stamped progress line."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def check(name: str, ok, detail="") -> bool:
    """Record one check (``ok`` may be True, False or None = information only)."""
    status = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    detail = " ".join(str(detail).split())
    CHECKS.append({"check": name, "status": status, "detail": detail})
    say(f"{status:4s} {name}" + (f" :: {detail}" if detail != "" else ""))
    return bool(ok)


@contextlib.contextmanager
def quiet():
    """Silence prints and warnings of the original code."""
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        yield


# --------------------------------------------------------------------------
# Independent helpers (written for this script, not imported from the builder)
# --------------------------------------------------------------------------
def brute_instances(labels, edges, trace, max_gap) -> set[tuple[int, ...]]:
    """All position tuples of a chain pattern in one trace, by plain enumeration."""
    found = set()

    def grow(prefix):
        node = len(prefix)
        if node == len(labels):
            found.add(tuple(prefix))
            return
        last = prefix[-1]
        if edges[node - 1] == DIRECT:
            options = [last + 1]
        else:
            options = range(last + 2, last + max_gap + 2)
        for position in options:
            if position < len(trace) and trace[position] == labels[node]:
                grow(prefix + [position])

    for start, activity in enumerate(trace):
        if activity == labels[0]:
            grow([start])
    return found


def brute_counts(pattern: Pattern, traces, max_gap) -> np.ndarray:
    """Per-case instance counts of a pattern by plain enumeration."""
    return np.array([len(brute_instances(pattern.labels, pattern.edges, trace, max_gap)) for trace in traces])


def mutual_information_nats(x, y) -> float:
    """Information gain of a discrete feature about a label, from the contingency table."""
    x, y = np.asarray(x), np.asarray(y)
    total = 0.0
    for value_x in np.unique(x):
        p_x = np.mean(x == value_x)
        for value_y in np.unique(y):
            p_xy = np.mean((x == value_x) & (y == value_y))
            if p_xy > 0:
                total += p_xy * math.log(p_xy / (p_x * np.mean(y == value_y)))
    return total


def naive_case_distance(dist, present) -> float:
    """Mean distance over (case with, case without) pairs, by an explicit double loop."""
    inside = [i for i, flag in enumerate(present) if flag]
    outside = [i for i, flag in enumerate(present) if not flag]
    if not inside or not outside:
        return float("nan")
    return sum(dist[i][j] for i in inside for j in outside) / (len(inside) * len(outside))


def dominates(a, b) -> bool:
    """a dominates b for (IG max, coverage max, CD min)."""
    no_worse = a[0] >= b[0] and a[1] >= b[1] and a[2] <= b[2]
    better = a[0] > b[0] or a[1] > b[1] or a[2] < b[2]
    return no_worse and better


def my_layers(points) -> list[int]:
    """Non-dominated sorting by peeling (1 = front); ties stay in the same layer."""
    points = [tuple(p) for p in points]
    layer = [0] * len(points)
    remaining = set(range(len(points)))
    current = 1
    while remaining:
        front = [i for i in remaining if not any(dominates(points[j], points[i]) for j in remaining if j != i)]
        for i in front:
            layer[i] = current
        remaining -= set(front)
        current += 1
    return layer


def my_selection(patterns: pd.DataFrame, k: int, lengths=(1, 2, 3)) -> dict:
    """Independent projection + selection from the table of evaluated patterns.

    Returns ``{size: [(itemset, layer, source pattern name), ...]}`` (top k).
    """
    by_set: dict[tuple, list] = {}
    for row in patterns.itertuples(index=False):
        activities = tuple(sorted(set(row.pattern.labels)))
        if len(activities) in lengths:
            by_set.setdefault(activities, []).append((row.IG, row.coverage, row.CD, row.name))
    result = {}
    for size in lengths:
        representatives = []
        for itemset, rows in by_set.items():
            if len(itemset) != size:
                continue
            layers = my_layers([r[:3] for r in rows])
            best = min((r for r, lay in zip(rows, layers) if lay == 1), key=lambda r: (-r[0], -r[1], r[2], r[3]))
            representatives.append((itemset, best))
        layers = my_layers([best[:3] for _, best in representatives])
        ranked = sorted(zip(representatives, layers),
                        key=lambda item: (item[1], -item[0][1][0], -item[0][1][1], item[0][1][2], item[0][0]))
        result[size] = [(itemset, layer, best[3]) for (itemset, best), layer in ranked[:k]]
    return result


def load_filtered_log(log_name: str) -> pd.DataFrame:
    """Own re-implementation of the 2024 ``DataManager`` cleaning (tools.py:36-64)."""
    frame = pd.read_csv(DATASET_DIR / DATASET_FILES[log_name])
    frame[CASE] = frame[CASE].astype(str)
    frame[ACT] = [name.lower().replace(" ", "").replace("-", "").replace("_", "") for name in frame[ACT]]
    frame = frame.sort_values([CASE, POS])
    n_events = frame.groupby(ACT)[CASE].count()
    rare = set(n_events[n_events < 2].index)
    bad = set(frame.loc[frame[ACT].isin(rare), CASE])
    return frame[~frame[CASE].isin(bad)].reset_index(drop=True)


def my_distance_matrix(case_table: pd.DataFrame) -> np.ndarray:
    """Explicit case distance with plain loops (mismatch share + min-max scaled |age difference|)."""
    n = len(case_table)
    ages = case_table[NUMERIC[0]].to_numpy(dtype=float)
    cats = case_table[CATEGORICAL].astype(str).to_numpy()
    raw = np.abs(ages[:, None] - ages[None, :])
    upper = raw[np.triu_indices(n, 1)]
    numeric = (raw - upper.min()) / (upper.max() - upper.min())
    share = np.zeros((n, n))
    for column in range(cats.shape[1]):
        share += cats[:, column][:, None] != cats[:, column][None, :]
    share /= cats.shape[1]
    combined = (len(CATEGORICAL) * share + numeric) / (1 + len(CATEGORICAL))
    np.fill_diagonal(combined, 0.0)
    return combined


def graph_key(graph: nx.DiGraph) -> Pattern:
    """Original pattern graph -> (labels, edge types); raises if it is not a directed path."""
    order = list(nx.topological_sort(graph))
    if graph.number_of_edges() != len(order) - 1:
        raise ValueError("not a path")
    edges = []
    for source, target in zip(order, order[1:]):
        if not graph.has_edge(source, target):
            raise ValueError("not a path")
        edges.append(EVENTUAL if graph.edges[source, target]["eventually"] else DIRECT)
    return Pattern(tuple(graph.nodes[node]["value"] for node in order), tuple(edges))


# --------------------------------------------------------------------------
# 1. Toy traces: extension rules and interest functions, expected values by hand
# --------------------------------------------------------------------------
def children_as_text(children: dict) -> dict[str, set]:
    """Child patterns as 'a -> b ~> c' -> set of position tuples (case index dropped for 1 trace)."""
    return {str(pattern): {positions for _case, positions in found} for pattern, found in children.items()}


def part1_toy() -> None:
    say("=== 1. toy traces (hand-computed expectations) ===")
    # --- single activity, repeated, gap limit ---------------------------------
    trace = list("xaabacdef")  # positions 0..8, 'a' at 1, 2, 4
    index = ic.build_position_index([trace])
    got = children_as_text(ic.extend_pattern(Pattern(("a",)), [trace], index, max_gap=2))
    expected = {
        "x -> a": {(0, 1)}, "a -> a": {(1, 2)}, "a ~> b": {(1, 3)}, "a ~> a": {(1, 4), (2, 4)},
        "x -> a -> a": {(0, 1, 2)}, "a -> b": {(2, 3)}, "a ~> c": {(2, 5)}, "x ~> a": {(0, 2)},
        "a -> a -> b": {(1, 2, 3)}, "b -> a": {(3, 4)}, "a -> c": {(4, 5)}, "a ~> d": {(4, 6)},
        "a ~> e": {(4, 7)}, "b -> a -> c": {(3, 4, 5)},
    }
    check("1.1 single activity with repeats, gap 2: 14 hand-computed children with instances",
          got == expected, f"missing={sorted(set(expected) - set(got))} extra={sorted(set(got) - set(expected))} "
          f"wrong_instances={[k for k in expected if k in got and got[k] != expected[k]]}")

    # --- start / end of trace -------------------------------------------------
    two = list("ab")
    idx2 = ic.build_position_index([two, ["a"]])
    first = children_as_text(ic.extend_pattern(Pattern(("a",)), [two, ["a"]], idx2, 3))
    last = children_as_text(ic.extend_pattern(Pattern(("b",)), [two, ["a"]], idx2, 3))
    check("1.2 first/last event: only 'a -> b', no context, nothing from a 1-event trace",
          first == {"a -> b": {(0, 1)}} and last == {"a -> b": {(0, 1)}}, f"{first} {last}")

    # --- gap boundary both directions ----------------------------------------
    far = list("apqrsa")  # 'a' at 0 and 5: 4 events in between
    idx3 = ic.build_position_index([far])
    gap3 = children_as_text(ic.extend_pattern(Pattern(("a",)), [far], idx3, 3))
    gap4 = children_as_text(ic.extend_pattern(Pattern(("a",)), [far], idx3, 4))
    exp3 = {"a -> p": {(0, 1)}, "a ~> q": {(0, 2)}, "a ~> r": {(0, 3)}, "a ~> s": {(0, 4)},
            "s -> a": {(4, 5)}, "p ~> a": {(1, 5)}, "q ~> a": {(2, 5)}, "r ~> a": {(3, 5)}}
    exp4 = dict(exp3, **{"a ~> a": {(0, 5)}})
    check("1.3 gap limit: 4 events in between is outside gap 3 and inside gap 4 (found from both ends, once)",
          gap3 == exp3 and gap4 == exp4, f"gap3 ok={gap3 == exp3} gap4 ok={gap4 == exp4}")

    # --- step-2 extension of a 2-node pattern ---------------------------------
    long = list("xyabcdeab")  # 'a b' at (2,3) and (7,8)
    idx4 = ic.build_position_index([long])
    got = children_as_text(ic.extend_pattern(Pattern(("a", "b"), (DIRECT,)), [long], idx4, 2))
    expected = {
        "y -> a -> b": {(1, 2, 3)}, "a -> b -> c": {(2, 3, 4)}, "a -> b ~> d": {(2, 3, 5)},
        "a -> b ~> e": {(2, 3, 6)}, "x ~> a -> b": {(0, 2, 3)}, "e -> a -> b": {(6, 7, 8)},
        "c ~> a -> b": {(4, 7, 8)}, "d ~> a -> b": {(5, 7, 8)},
    }
    check("1.4 2-node parent, gap 2: 8 hand-computed children, no context pattern, pattern at trace end",
          got == expected, f"missing={sorted(set(expected) - set(got))} extra={sorted(set(got) - set(expected))}")

    # --- repeated activity in step 2 -----------------------------------------
    aaaa = list("aaaa")
    idx5 = ic.build_position_index([aaaa])
    got = children_as_text(ic.extend_pattern(Pattern(("a", "a"), (DIRECT,)), [aaaa], idx5, 1))
    expected = {"a -> a -> a": {(0, 1, 2), (1, 2, 3)}, "a -> a ~> a": {(0, 1, 3)}, "a ~> a -> a": {(0, 2, 3)}}
    check("1.5 'a -> a' in <a,a,a,a>, gap 1: children counted once each", got == expected, got)

    # --- eventual parents are not extended -----------------------------------
    eventual = Pattern(("a", "b"), (EVENTUAL,))
    try:
        ic.extend_pattern(eventual, [long], idx4, 2)
        raised = False
    except ValueError:
        raised = True
    pool, _parents = ic.extend_patterns([eventual], [long], idx4, 2)
    check("1.6 a parent with an eventual edge is never extended", raised and pool == {}, f"raised={raised}")

    # --- count_instances against plain enumeration ---------------------------
    check("1.7 count_instances: 'a ~> a' in <a,a,a,a> is 2 (gap 1) and 3 (gap 2)",
          ic.count_instances(Pattern(("a", "a"), (EVENTUAL,)), aaaa, 1) == 2
          and ic.count_instances(Pattern(("a", "a"), (EVENTUAL,)), aaaa, 2) == 3)

    rng = np.random.default_rng(99)
    n_patterns, n_bad_scan, n_bad_ext = 0, 0, 0
    for _round in range(30):
        traces = [list(rng.choice(list("abc"), size=rng.integers(1, 12))) for _ in range(8)]
        position_index = ic.build_position_index(traces)
        gap = int(rng.integers(1, 4))
        parents = [Pattern((a,)) for a in "abc"]
        for _step in range(2):
            pool, _ = ic.extend_patterns(parents, traces, position_index, gap)
            for pattern, found in pool.items():
                truth = {(case, pos) for case, trace in enumerate(traces)
                         for pos in brute_instances(pattern.labels, pattern.edges, trace, gap)}
                scan = [ic.count_instances(pattern, trace, gap) for trace in traces]
                n_patterns += 1
                n_bad_ext += found != truth
                n_bad_scan += scan != [len(brute_instances(pattern.labels, pattern.edges, t, gap)) for t in traces]
            parents = [p for p in pool if p.extendable]
    check("1.8 random toy logs: instances from extension == plain enumeration of ALL instances (Def. 5 with gap)",
          n_bad_ext == 0 and n_bad_scan == 0, f"{n_patterns} patterns, {n_bad_ext} wrong via extension, "
          f"{n_bad_scan} wrong via count_instances")
    RESULTS["toy_random_patterns"] = n_patterns

    # --- interest functions ----------------------------------------------------
    counts = np.array([[2], [1], [0], [0], [1], [0]])
    labels = np.array([1, 1, 0, 0, 0, 1])
    dist = np.abs(np.arange(6)[:, None] - np.arange(6)[None, :]) / 10.0
    scores = ic.score_patterns(counts, labels, dist)
    hand_ig, hand_cov, hand_cd = 0.143841, 0.5, 2.1 / 9
    check("1.9 interest values by hand: IG 0.143841 nats (count as category), coverage 0.5, CD 0.2333 (mean)",
          abs(scores["IG"][0] - hand_ig) < 1e-6 and scores["coverage"][0] == hand_cov
          and abs(scores["CD"][0] - hand_cd) < 1e-12
          and abs(mutual_information_nats(counts[:, 0], labels) - hand_ig) < 1e-6,
          f"module IG={scores['IG'][0]:.6f} cov={scores['coverage'][0]} CD={scores['CD'][0]:.6f}; "
          f"presence-only IG would be 0.056633; the paper's printed CD formula (sum/|L|) would give {2.1 / 6:.4f}")
    all_in = ic.score_patterns(np.ones((6, 1), dtype=int), labels, dist)
    check("1.10 pattern in ALL cases: CD = undefined_cd (1.0), IG = 0", all_in["CD"][0] == 1.0
          and abs(all_in["IG"][0]) < 1e-12, f"CD={all_in['CD'][0]} IG={all_in['IG'][0]}")

    # --- case distance by hand -----------------------------------------------
    table = pd.DataFrame({"Age": [10, 20, 40], "c1": ["A", "A", "B"], "c2": ["X", "Y", "Y"]})
    got = pairwise_case_distance(table, ["Age"], ["c1", "c2"])
    hand = np.array([[0, 1 / 3, 1.0], [1 / 3, 0, 0.5], [1.0, 0.5, 0]])
    check("1.11 case distance by hand: (2*mismatch share + min-max |age diff|)/3", np.allclose(got, hand), got.round(4))


# --------------------------------------------------------------------------
# 2. Original code on a different subset
# --------------------------------------------------------------------------
class OriginalRunner:
    """Feeds a (sub)log to the unchanged 2023 functions, mirroring Auto_IMPID.py:45-73."""

    def __init__(self, data: pd.DataFrame):
        import IMIPD  # the 2023 module; imports its own tools.py
        self.imipd = IMIPD
        self.data = data
        colours = dict.fromkeys(data[ACT].unique(), "#000000") | {"start": "k", "end": "k"}
        self.case_ids = list(data[CASE].drop_duplicates())
        with quiet():
            self.graphs = {case: IMIPD.Trace_graph_generator(data, -1, case, colours, CASE, ACT, TIME)
                           for case in self.case_ids}

    def step1(self, cores, dictionary=None) -> dict:
        """Extend every core activity into ``dictionary`` (shared when several cores are given)."""
        dictionary = {} if dictionary is None else dictionary
        for core in cores:
            cases = self.data.loc[self.data[ACT] == core, CASE]
            filtered = self.data[self.data[CASE].isin(cases)]
            new_ids: list = []
            for case in filtered[CASE].unique():
                case_data = filtered[filtered[CASE] == case]
                with quiet():
                    dictionary, new_ids = self.imipd.Pattern_extension(
                        case_data, self.graphs[case].copy(), core, CASE, dictionary, MAX_GAP, new_ids)
        return dictionary

    def step2(self, dictionary: dict, parent_id: str, patient: pd.DataFrame) -> dict:
        """Children of one parent pattern through ``Single_Pattern_Extender``."""
        with quiet():
            _all, stage, _patient = self.imipd.Single_Pattern_Extender(
                dict(dictionary), parent_id, patient.copy(), self.graphs, self.data, MAX_GAP, ACT, CASE)
        return stage


def original_counts(entry: dict, case_ids: list[str]) -> np.ndarray:
    """Per-case length of the original instance list."""
    cases = entry["Instances"]["case"]
    return np.array([cases.count(case) for case in case_ids])


def compare_children(original: dict, ours: dict, case_ids: list[str]) -> dict:
    """Pattern sets and per-case counts: original dictionary vs our child -> instances mapping."""
    keys = {}
    n_duplicate_ids = 0
    for pid, entry in original.items():
        key = graph_key(entry["pattern"])
        n_duplicate_ids += key in keys
        keys[key] = pid
    ours_counts = {pattern: np.bincount([case for case, _ in found], minlength=len(case_ids))
                   for pattern, found in ours.items()}
    factors: dict[str, int] = {}
    n_presence_mismatch = n_non_uniform = 0
    for key, pid in keys.items():
        if key not in ours_counts:
            continue
        theirs, mine = original_counts(original[pid], case_ids), ours_counts[key]
        if not np.array_equal(theirs > 0, mine > 0):
            n_presence_mismatch += 1
            continue
        ratio = np.unique(theirs[mine > 0] / mine[mine > 0])
        if len(ratio) != 1:
            n_non_uniform += 1
        else:
            factors[str(float(ratio[0]))] = factors.get(str(float(ratio[0])), 0) + 1
    return {
        "n_original": len(keys), "n_ours": len(ours), "n_duplicate_original_ids": n_duplicate_ids,
        "only_original": sorted(str(k) for k in set(keys) - set(ours)),
        "only_ours": sorted(str(k) for k in set(ours) - set(keys)),
        "count_factor_original_over_ours": factors, "n_factor_differs_between_cases": n_non_uniform,
        "n_presence_mismatch": n_presence_mismatch,
    }


def part2_original() -> None:
    say("=== 2. original IMIPD code on 200 f1 cases (seed 4711), other seed activities ===")
    frame = load_filtered_log("f1")
    all_cases = list(frame[CASE].drop_duplicates())
    rng = np.random.default_rng(SUBSET_SEED)
    chosen = sorted(rng.choice(all_cases, size=SUBSET_SIZE, replace=False).tolist())
    data = frame[frame[CASE].isin(chosen)].sort_values([CASE, POS]).reset_index(drop=True)
    data[TIME] = pd.to_datetime(data[TIME])
    traces = [group[ACT].tolist() for _case, group in data.groupby(CASE, sort=True)]
    case_ids = sorted(data[CASE].unique())
    assert case_ids == chosen
    labels = data.groupby(CASE, sort=True)[LABEL].first().to_numpy()
    case_table = data.drop_duplicates(CASE).set_index(CASE).loc[case_ids, NUMERIC + CATEGORICAL]
    dist = my_distance_matrix(case_table)
    module_dist = pairwise_case_distance(case_table, NUMERIC, CATEGORICAL)
    check("2.0 case_distance.pairwise_case_distance == own loop implementation (200 cases)",
          np.allclose(dist, module_dist, atol=1e-12), f"max abs diff {np.abs(dist - module_dist).max():.2e}")

    # seed activities: next five by number of cases after removing the builder's seeds, plus the
    # remaining activity with the most adjacent self-repeats (exercises the double counting)
    in_cases = pd.Series({a: sum(a in t for t in traces) for a in sorted({x for t in traces for x in t})})
    candidates = in_cases.drop(labels=[a for a in BUILDER_SEEDS if a in in_cases.index])
    cores = list(candidates.sort_values(ascending=False, kind="stable").index[:5])
    repeats = pd.Series({a: sum(u == v == a for t in traces for u, v in zip(t, t[1:])) for a in candidates.index})
    extra = repeats.drop(labels=cores).sort_values(ascending=False, kind="stable").index[0]
    cores.append(extra)
    say(f"subset: {len(chosen)} cases, {len(data)} events, {data[ACT].nunique()} activities; cores {cores}")
    RESULTS["part2"] = {"cores": cores, "n_events": int(len(data)), "n_activities": int(data[ACT].nunique())}

    start = time.perf_counter()
    runner = OriginalRunner(data)
    n_chain = sum(
        [runner.graphs[c].nodes[i]["value"] for i in range(len(t))] == t
        and set(runner.graphs[c].edges) == {(i, i + 1) for i in range(len(t) - 1)}
        and not any(nx.get_node_attributes(runner.graphs[c], "parallel").values())
        for c, t in zip(case_ids, traces))
    check("2.1 original Trace_graph_generator(delta_time=-1) gives chains equal to our traces",
          n_chain == len(case_ids), f"{n_chain}/{len(case_ids)} ({time.perf_counter() - start:.1f} s)")

    position_index = ic.build_position_index(traces)

    # A: every core with a fresh dictionary
    report_a, seconds_a = [], 0.0
    for core in cores:
        start = time.perf_counter()
        original = runner.step1([core])
        seconds_a += time.perf_counter() - start
        ours = ic.extend_pattern(Pattern((core,)), traces, position_index, MAX_GAP)
        report_a.append({"core": core, **compare_children(original, ours, case_ids)})
    ok = all(not r["only_original"] and not r["only_ours"] and r["n_presence_mismatch"] == 0
             and r["n_factor_differs_between_cases"] == 0 and r["n_duplicate_original_ids"] == 0 for r in report_a)
    factors_a: dict[str, int] = {}
    for r in report_a:
        for factor, n in r["count_factor_original_over_ours"].items():
            factors_a[factor] = factors_a.get(factor, 0) + n
    check("2.2 Pattern_extension, fresh dictionary per core: same child patterns, same cases, uniform count factor",
          ok, f"patterns original/ours {[(r['n_original'], r['n_ours']) for r in report_a]}; factors {factors_a}; "
          f"original {seconds_a:.1f} s")
    RESULTS["part2"]["A"] = report_a

    # B: all cores in one shared dictionary (the automatic mode's step 1)
    start = time.perf_counter()
    shared = runner.step1(cores)
    seconds_b = time.perf_counter() - start
    start = time.perf_counter()
    ours_shared, _parents = ic.extend_patterns([Pattern((c,)) for c in cores], traces, position_index, MAX_GAP)
    ours_seconds = time.perf_counter() - start
    report_b = compare_children(shared, ours_shared, case_ids)
    check("2.3 shared dictionary (step 1 of the automatic mode): same patterns, same cases, uniform factor",
          not report_b["only_original"] and not report_b["only_ours"] and report_b["n_presence_mismatch"] == 0
          and report_b["n_factor_differs_between_cases"] == 0,
          f"{report_b['n_original']}/{report_b['n_ours']} patterns; factors "
          f"{report_b['count_factor_original_over_ours']}; original {seconds_b:.1f} s, ours {ours_seconds:.3f} s")
    RESULTS["part2"]["B"] = report_b

    # C: step 2 for a deterministic choice of parents
    key_of = {pid: graph_key(entry["pattern"]) for pid, entry in shared.items()}
    n_unique = {pid: len(ours_shared[key_of[pid]]) for pid in shared}
    eligible = sorted(pid for pid in shared if key_of[pid].extendable and 3 <= n_unique[pid] <= 40)
    pick_rng = np.random.default_rng(SUBSET_SEED)
    parents = list(pick_rng.choice(eligible, size=min(8, len(eligible)), replace=False))
    for wanted in (lambda k: len(k.labels) == 2 and k.labels[0] == k.labels[1],   # repeated activity
                   lambda k: len(k.labels) == 3,                                   # context pattern
                   lambda k: len(k.labels) == 2 and set(k.labels) <= set(cores) and k.labels[0] != k.labels[1]):
        more = [pid for pid in eligible if wanted(key_of[pid]) and pid not in parents]
        parents.extend(more[:1])
    patient = pd.DataFrame({CASE: case_ids, LABEL: labels})
    report_c, seconds_c, step2_original = [], 0.0, {}
    for parent in parents:
        start = time.perf_counter()
        stage = runner.step2(shared, parent, patient)
        seconds_c += time.perf_counter() - start
        ours = ic.extend_pattern(key_of[parent], traces, position_index, MAX_GAP)
        report_c.append({"parent": str(key_of[parent]), "parent_id": parent,
                         "parent_instances_original": len(shared[parent]["Instances"]["case"]),
                         "parent_instances_ours": n_unique[parent], **compare_children(stage, ours, case_ids)})
        step2_original.update({(parent, pid): entry for pid, entry in stage.items()})
    ok = all(not r["only_original"] and not r["only_ours"] and r["n_presence_mismatch"] == 0
             and r["n_factor_differs_between_cases"] == 0 for r in report_c)
    factors_c: dict[str, int] = {}
    for r in report_c:
        for factor, n in r["count_factor_original_over_ours"].items():
            factors_c[factor] = factors_c.get(factor, 0) + n
    check("2.4 Single_Pattern_Extender (step 2): same children, same cases, uniform count factor",
          ok, f"{len(parents)} parents, children original/ours "
          f"{sum(r['n_original'] for r in report_c)}/{sum(r['n_ours'] for r in report_c)}; factors {factors_c}; "
          f"original {seconds_c:.1f} s; parents {[r['parent'] for r in report_c]}")
    RESULTS["part2"]["C"] = report_c

    # D: interest values of 30 patterns through the original scoring functions
    pool = [("s1", pid, entry) for pid, entry in shared.items()] + \
           [("s2", f"{parent}|{pid}", entry) for (parent, pid), entry in step2_original.items()]
    order = np.random.default_rng(SUBSET_SEED + 1).permutation(len(pool))
    picked = [pool[i] for i in order[:20]] + [p for p in (pool[i] for i in order[20:]) if p[0] == "s2"][:10]
    columns = [f"p{i}" for i in range(len(picked))]
    table = patient.copy()
    for column, (_kind, _pid, entry) in zip(columns, picked):
        table[column] = original_counts(entry, case_ids)
    n = len(case_ids)
    condensed = squareform(dist, checks=False)
    pair_cases = [(a, b) for a in range(n) for b in range(a + 1, n)]
    start_points, i = [], 0
    for k in range(n):
        start_points.append(k * n - (i + k))
        i += k
    start = time.perf_counter()
    with quiet():
        original_scores = runner.imipd.create_pattern_attributes(table, LABEL, columns, condensed, pair_cases,
                                                                 start_points, "binary")
    seconds_d = time.perf_counter() - start
    our_counts = np.column_stack([brute_counts(graph_key(entry["pattern"]), traces, MAX_GAP)
                                  for _kind, _pid, entry in picked])
    ours = ic.score_patterns(our_counts, labels, dist)
    diff = {
        "IG": float(np.abs(original_scores["Outcome_Interest"].to_numpy(float) - ours["IG"]).max()),
        "coverage": float(np.abs(original_scores["Frequency_Interest"].to_numpy(float) - ours["coverage"]).max()),
        "CD": float(np.abs(original_scores["Case_Distance_Interest"].to_numpy(float) - ours["CD"]).max()),
    }
    naive = {
        "IG": max(abs(mutual_information_nats(our_counts[:, c], labels) - ours["IG"][c]) for c in range(len(picked))),
        "CD": max(abs(naive_case_distance(dist, our_counts[:, c] > 0) - ours["CD"][c]) for c in range(len(picked))),
    }
    check("2.5 interest values of 30 patterns: original create_pattern_attributes vs score_patterns",
          max(diff.values()) < 1e-9, f"max abs diff {diff}; original {seconds_d:.1f} s; ranges IG "
          f"{ours['IG'].min():.4f}-{ours['IG'].max():.4f}, coverage {ours['coverage'].min():.3f}-"
          f"{ours['coverage'].max():.3f}, CD {ours['CD'].min():.4f}-{ours['CD'].max():.4f}; "
          f"{sum(len(graph_key(e['pattern']).labels) == 3 for _, _, e in picked)} three-node, "
          f"{sum(not graph_key(e['pattern']).extendable for _, _, e in picked)} with an eventual edge")
    check("2.6 the same 30 patterns: own contingency-table IG and double-loop CD vs score_patterns",
          max(naive.values()) < 1e-9, f"max abs diff {naive}")
    RESULTS["part2"]["D"] = {"max_abs_diff_vs_original": diff, "max_abs_diff_vs_naive": naive}


# --------------------------------------------------------------------------
# 2b. Stored full-log runs of the original automatic mode (experiment C3)
# --------------------------------------------------------------------------
# (tag, pattern export, folder with the distance pickle, log name, log preparation)
C3_RUNS = [
    ("C3 ORIGINAL code, raw f1", "original_patterns_f1.json", "f1_step1", "f1", "gui"),
    ("C3 patched copy, f1 2024-filtered", "patched_patterns_f1_dm2024.json", "f1_dm2024_step2_patched", "f1", "dm2024"),
    ("C3 patched copy, f2 2024-filtered", "patched_patterns_f2_dm2024.json", "f2_dm2024_step2_patched", "f2", "dm2024"),
    ("C3 patched copy, f3 2024-filtered", "patched_patterns_f3_dm2024.json", "f3_dm2024_step2_patched", "f3", "dm2024"),
]


def own_first_front(values: np.ndarray) -> np.ndarray:
    """Boolean mask of the rows no other row dominates (IG max, coverage max, CD min)."""
    mask = np.ones(len(values), dtype=bool)
    for i, row in enumerate(values):
        no_worse = (values[:, 0] >= row[0]) & (values[:, 1] >= row[1]) & (values[:, 2] <= row[2])
        better = (values[:, 0] > row[0]) | (values[:, 1] > row[1]) | (values[:, 2] < row[2])
        mask[i] = not bool((no_worse & better).any())
    return mask


def record_key(record: dict) -> Pattern:
    """Pattern export record of C3 -> (labels, edge types); asserts that it is a path in label order."""
    edges = sorted(record["edges"], key=lambda e: e["source"])
    assert [(e["source"], e["target"]) for e in edges] == [(i, i + 1) for i in range(len(record["labels"]) - 1)]
    return Pattern(tuple(record["labels"]), tuple(DIRECT if e["type"] == "directly" else EVENTUAL for e in edges))


def compare_stored_step(records, instances, n_cases, train_rows, labels, dist_train) -> dict:
    """One step of a stored run against our instances: pattern set, interest values, front."""
    keys = [record_key(record) for record in records]
    missing = sorted({str(k) for k in keys if k not in instances})
    extra = sorted({str(k) for k in instances if k not in set(keys)})
    if missing or extra:
        return {"keys": keys, "ok": False, "missing": missing[:10], "extra": extra[:10],
                "n_missing": len(missing), "n_extra": len(extra)}
    column_of = {k: np.bincount([case for case, _ in instances[k]], minlength=n_cases) for k in set(keys)}
    counts = np.column_stack([column_of[k] for k in keys])
    scores = ic.score_patterns(counts[train_rows], labels[train_rows], dist_train, undefined_cd=np.nan)
    theirs = pd.DataFrame(records)
    cd_theirs = theirs["Case_Distance_Interest"].to_numpy(dtype=float)
    nan_equal = bool(np.array_equal(np.isnan(cd_theirs), np.isnan(scores["CD"].to_numpy())))
    diffs = {
        "IG": float(np.abs(theirs["Outcome_Interest"].to_numpy(float) - scores["IG"]).max()),
        "coverage": float(np.abs(theirs["Frequency_Interest"].to_numpy(float) - scores["coverage"]).max()),
        "CD": float(np.nanmax(np.abs(cd_theirs - scores["CD"].to_numpy()))),
    }
    support_equal = bool(np.array_equal(theirs["Case_Support"].to_numpy(float), (counts[train_rows] > 0).sum(axis=0)))
    freq_ratio = theirs["Pattern_Frequency"].to_numpy(float) / np.maximum(counts[train_rows].sum(axis=0), 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        like_original = np.asarray(paretoset(scores[OBJ].to_numpy(), sense=["max", "max", "min"], distinct=True))
    on_front = theirs["on_front"].to_numpy(bool)
    filled = scores.copy()
    filled["CD"] = filled["CD"].fillna(1.0)
    ours_front = ic.pareto_front(filled, OBJ, distinct=False)
    extra_rows = np.flatnonzero(ours_front & ~on_front)
    front_values = {tuple(v) for v in filled.loc[on_front, OBJ].to_numpy()}
    n_ties = sum(tuple(filled.loc[r, OBJ]) in front_values for r in extra_rows)
    ok = max(diffs.values()) < 1e-9 and nan_equal and support_equal
    return {
        "keys": keys, "ok": bool(ok), "n_rows": len(keys), "n_distinct_patterns": len(set(keys)),
        "max_abs_diff": diffs, "nan_pattern_equal": nan_equal, "n_nan_cd": int(np.isnan(cd_theirs).sum()),
        "case_support_equal": support_equal,
        "pattern_frequency_ratio_values": sorted({float(x) for x in np.unique(freq_ratio)}),
        "stored_front_size": int(on_front.sum()),
        "stored_front_reproduced_from_our_values": bool(np.array_equal(like_original, on_front)),
        "front_size_our_rules": int(ours_front.sum()),
        "stored_front_is_subset_of_ours": bool((ours_front | ~on_front).all()),
        "extra_front_rows": int(len(extra_rows)), "extra_rows_that_tie_with_a_stored_front_row": int(n_ties),
        "pareto_front_distinct_false_equals_own_dominance": bool(
            np.array_equal(own_first_front(filled[OBJ].to_numpy()), ours_front)),
    }


def part2b_c3() -> None:
    say("=== 2b. stored full-log runs of the original automatic mode (C3): train scoring, original distance ===")
    RESULTS["part2b_c3"] = {}
    for tag, json_name, run_folder, log_name, prep in C3_RUNS:
        json_path = C3_DIR / json_name
        pickle_path = C3_DIR / run_folder / "dist" / "pairwise_case_distances.pkl"
        if not json_path.exists() or not pickle_path.exists():
            check(f"2b [{tag}] files exist", None, "skipped: pattern export or distance pickle is missing")
            continue
        stored = json.loads(json_path.read_text(encoding="utf-8"))
        settings = stored["meta"]["settings"]
        raw = pd.read_csv(DATASET_DIR / DATASET_FILES[log_name])
        if prep == "dm2024":  # as run_original_impressed.apply_2024_log_filter
            raw[ACT] = raw[ACT].str.lower()
            n_events = raw.groupby(ACT)[CASE].count()
            bad = raw.loc[raw[ACT].isin(n_events[n_events < 2].index), CASE].unique()
            raw = raw[~raw[CASE].isin(bad)].reset_index(drop=True)
        raw[ACT] = raw[ACT].str.replace("_", "-")
        raw[CASE] = raw[CASE].astype(str)
        in_order = bool((raw.groupby(CASE)[POS].diff().dropna() > 0).all())
        patient = raw.drop_duplicates(CASE).sort_values(CASE).reset_index(drop=True)[[CASE, LABEL]]
        by_case = raw.groupby(CASE, sort=False)[ACT].apply(list)
        traces = [by_case[case] for case in patient[CASE]]
        labels = patient[LABEL].to_numpy()
        train, _test = train_test_split(patient, test_size=settings["test_data_percentage"], random_state=42,
                                        stratify=patient[LABEL])
        train_rows = np.sort(train.index.to_numpy())
        with open(pickle_path, "rb") as handle:
            dist = squareform(pickle.load(handle))
        dist_train = dist[np.ix_(train_rows, train_rows)]
        setup_ok = (len(patient) == stored["meta"]["n_cases"] and in_order and dist.shape[0] == len(patient)
                    and len(train_rows) == stored["meta"]["train_X_shape"][0])
        position_index = ic.build_position_index(traces)
        instances = {Pattern((a,)): {(c, (p,)) for c, p in events} for a, events in position_index.items()}
        run_report, all_ok, front_ok, details = {}, setup_ok, True, []
        for step in sorted(stored["steps"], key=int):
            records = stored["steps"][step]["patterns"]
            if int(step) > 0:
                previous = stored["steps"][str(int(step) - 1)]["patterns"]
                parents = [record_key(r) for r in previous if r["on_front"]]
                start = time.perf_counter()
                instances, _parents = ic.extend_patterns(parents, traces, position_index,
                                                         settings["max_gap_between_events"])
                seconds = time.perf_counter() - start
            else:
                seconds = 0.0
            report = compare_stored_step(records, instances, len(traces), train_rows, labels, dist_train)
            keys = report.pop("keys")
            if report["ok"] and int(step) > 0:
                report["instance_list_ratio_values"] = sorted(
                    {r["n_instances_whole_log"] / len(instances[k]) for r, k in zip(records, keys)})
            report["our_extension_seconds"] = round(seconds, 3)
            run_report[step] = report
            all_ok &= report["ok"]
            front_ok &= report.get("stored_front_reproduced_from_our_values", False)
            details.append(
                f"step {step}: {report.get('n_rows')} rows ({report.get('n_distinct_patterns')} distinct), "
                f"max diff {report.get('max_abs_diff')}, NaN CD {report.get('n_nan_cd')}, front "
                f"{report.get('stored_front_size')} reproduced {report.get('stored_front_reproduced_from_our_values')}"
                f", ours(distinct=False, NaN->1) {report.get('front_size_our_rules')} with "
                f"{report.get('extra_rows_that_tie_with_a_stored_front_row')}/{report.get('extra_front_rows')} extra "
                f"rows tying, count ratios {report.get('pattern_frequency_ratio_values')}"
                f", instance-list ratios {report.get('instance_list_ratio_values')}"
                + (f", MISSING {report.get('n_missing')} EXTRA {report.get('n_extra')}" if not report["ok"] else ""))
        check(f"2b [{tag}]: {len(patient)} cases, every step: same patterns as the stored run, same IG/coverage/CD "
              f"on the {len(train_rows)} training cases", all_ok, " || ".join(details))
        check(f"2b [{tag}]: stored Pareto fronts reproduced from our values with the original's paretoset call",
              front_ok, [run_report[s].get("stored_front_size") for s in run_report])
        RESULTS["part2b_c3"][tag] = run_report


# --------------------------------------------------------------------------
# 3. Projection and selection
# --------------------------------------------------------------------------
def toy_selector(rows) -> ImpressedChainSelector:
    """A selector whose ``patterns_`` table is written by hand (no mining)."""
    selector = ImpressedChainSelector(k=3)
    frame = pd.DataFrame([{"pattern": p, "name": str(p), "step": 0, "n_nodes": len(p.labels), "IG": ig,
                           "coverage": cov, "CD": cd, "on_front": True} for p, ig, cov, cd in rows])
    selector.patterns_ = frame
    return selector


def load_inputs(log_name: str):
    """Traces, labels and distance matrix from own loader; cross-checked with the builder's loader."""
    from c4_run_selector import load_selector_inputs
    log, dist_builder, _table = load_selector_inputs(log_name)
    frame = load_filtered_log(log_name)
    traces = {case: group[ACT].tolist() for case, group in frame.groupby(CASE, sort=True)}
    labels = frame.groupby(CASE, sort=True)[LABEL].first().astype(int).to_dict()
    raw = pd.read_csv(DATASET_DIR / DATASET_FILES[log_name])
    raw[CASE] = raw[CASE].astype(str)
    case_table = raw.sort_values([CASE, POS]).drop_duplicates(CASE).set_index(CASE).loc[list(traces)]
    dist = my_distance_matrix(case_table[NUMERIC + CATEGORICAL])
    same = (list(traces) == list(log.case_ids) and all(traces[c] == list(log.traces[c]) for c in traces)
            and labels == dict(log.labels) and np.allclose(dist, dist_builder, atol=1e-12))
    constant = bool((raw.groupby(CASE)[NUMERIC + CATEGORICAL].nunique() == 1).all().all())
    return traces, labels, dist, same, constant


def selection_as_lists(selector: ImpressedChainSelector) -> dict:
    table = selector.select_table()
    return {int(size): [(tuple(row.itemset), int(row.layer), row.source_pattern)
                        for row in table[table["size"] == size].itertuples()] for size in selector.lengths}


def part3_selection() -> dict:
    say("=== 3. projection and selection ===")
    a, b, c, d = "A", "B", "C", "D"
    rows = [
        (Pattern((a,)), 0.5, 0.5, 0.5), (Pattern((a, a), (DIRECT,)), 0.6, 0.2, 0.5),
        (Pattern((b,)), 0.5, 0.5, 0.5), (Pattern((c,)), 0.1, 0.9, 0.9), (Pattern((d,)), 0.05, 0.05, 0.95),
        (Pattern((a, b), (DIRECT,)), 0.3, 0.3, 0.4), (Pattern((b, a), (EVENTUAL,)), 0.3, 0.3, 0.4),
        (Pattern((a, b, c, d), (DIRECT,) * 3), 0.9, 0.9, 0.1),
        (Pattern((a, b, a, c), (DIRECT,) * 3), 0.2, 0.2, 0.2),
        (Pattern((c, d), (DIRECT,)), 0.3, 0.3, 0.4),
    ]
    table = toy_selector(rows).full_table()
    got = [(tuple(r.itemset), r.size, r.rank, r.layer, r.source_pattern, r.n_patterns) for r in table.itertuples()]
    expected = [
        ((a,), 1, 1, 1, "A -> A", 2), ((b,), 1, 2, 1, "B", 1), ((c,), 1, 3, 1, "C", 1), ((d,), 1, 4, 2, "D", 1),
        ((a, b), 2, 1, 1, "A -> B", 2), ((c, d), 2, 2, 1, "C -> D", 1),
        ((a, b, c), 3, 1, 1, "A -> B -> A -> C", 1),
    ]
    check("3.1 toy projection by hand: A->A is length 1, dedupe by set, 4-activity set dropped, "
          "4-node/3-activity set kept, ties ordered by name", got == expected, got)
    selected = toy_selector(rows).select()
    check("3.2 toy select(): k=3 per length, sorted activity lists, fewer than k is allowed",
          selected == {1: [[a], [b], [c]], 2: [[a, b], [c, d]], 3: [[a, b, c]]}, selected)

    rng = np.random.default_rng(5)
    n_bad = 0
    for _ in range(200):
        points = pd.DataFrame(rng.integers(0, 4, size=(rng.integers(1, 40), 3)) / 4.0, columns=OBJ)
        n_bad += list(ic.pareto_layers(points, OBJ, distinct=False)) != my_layers(points.to_numpy())
        n_bad += list(ic.pareto_front(points, OBJ, distinct=False)) != [x == 1 for x in my_layers(points.to_numpy())]
    check("3.3 pareto_layers / pareto_front (distinct=False) == own peeling on 200 random tables with many ties",
          n_bad == 0, f"{n_bad} disagreements")

    report, selectors = {}, {}
    for log_name in ("f1", "f2", "f3"):
        traces, labels, dist, same_inputs, constant = load_inputs(log_name)
        start = time.perf_counter()
        selector = ImpressedChainSelector(max_gap=MAX_GAP, steps=2, k=10).fit(traces, labels, dist)
        fit_seconds = time.perf_counter() - start
        selectors[log_name] = (selector, traces, labels, dist)
        patterns = selector.patterns_
        trace_list = [traces[case] for case in selector.case_ids_]
        y = np.array([labels[case] for case in selector.case_ids_])

        # counts and scores of evaluated patterns, recomputed independently on a sample
        sample = np.random.default_rng(11).choice(len(patterns), size=min(250, len(patterns)), replace=False)
        n_count_bad, max_diff = 0, {"IG": 0.0, "coverage": 0.0, "CD": 0.0}
        for row in sample:
            pattern = patterns.at[row, "pattern"]
            counts = brute_counts(pattern, trace_list, MAX_GAP)
            n_count_bad += not np.array_equal(counts, selector.counts_[pattern])
            present = counts > 0
            cd = 1.0 if present.all() else float(present @ dist @ ~present) / (present.sum() * (~present).sum())
            max_diff["IG"] = max(max_diff["IG"], abs(mutual_information_nats(counts, y) - patterns.at[row, "IG"]))
            max_diff["coverage"] = max(max_diff["coverage"], abs(present.mean() - patterns.at[row, "coverage"]))
            max_diff["CD"] = max(max_diff["CD"], abs(cd - patterns.at[row, "CD"]))

        # step fronts and step candidates
        fronts_ok, chain_ok = True, True
        for step, step_table in enumerate(selector.step_tables_):
            fronts_ok &= list(step_table["on_front"]) == [x == 1 for x in my_layers(step_table[OBJ].to_numpy())] \
                if len(step_table) <= 2000 else bool(_front_check_large(step_table))
            if step > 0:
                previous = selector.step_tables_[step - 1]
                parents = {p for p in previous.loc[previous["on_front"], "pattern"] if p.extendable}
                expected_children = set()
                for parent in parents:
                    for case, trace in enumerate(trace_list):
                        expected_children |= naive_children(parent, trace, MAX_GAP)
                chain_ok &= expected_children == set(step_table["pattern"])

        mine = my_selection(patterns, k=10)
        theirs = selection_as_lists(selector)
        stored = json.loads((HERE / f"impressed_sets_{log_name}.json").read_text(encoding="utf-8"))["selection"]
        stored_equal = {int(s): [tuple(x) for x in stored[str(s)]] == [t[0] for t in theirs[int(s)]] for s in (1, 2, 3)}
        lengths_ok = all(len(itemset) == size and len(set(itemset)) == size
                         for size, items in theirs.items() for itemset, _l, _s in items)
        full = selector.full_table()
        dedupe_ok = not full["itemset"].duplicated().any() and set(full["size"]) <= {1, 2, 3}
        n_sets_expected = len({tuple(sorted(set(p.labels))) for p in patterns["pattern"] if len(set(p.labels)) <= 3})
        refit = ImpressedChainSelector(max_gap=MAX_GAP, steps=2, k=10).fit(traces, labels, dist)
        deterministic = selection_as_lists(refit) == theirs and refit.patterns_.drop(columns="pattern").equals(
            patterns.drop(columns="pattern"))

        # alternative reading of "select patterns, then project": layer the PATTERNS of a size, dedupe by set
        alternative, pushed_down = {}, {}
        for size in (1, 2, 3):
            subset = patterns[[len(set(p.labels)) == size for p in patterns["pattern"]]].reset_index(drop=True)
            layers = np.asarray(ic.pareto_layers(subset, OBJ, distinct=False))
            order = sorted(range(len(subset)), key=lambda i: (layers[i], -subset.at[i, "IG"],
                                                              -subset.at[i, "coverage"], subset.at[i, "CD"],
                                                              subset.at[i, "name"]))
            seen: list = []
            for i in order:
                itemset = tuple(sorted(set(subset.at[i, "pattern"].labels)))
                if itemset not in seen:
                    seen.append(itemset)
            alternative[size] = seen[:10]
            front_sets = {tuple(sorted(set(subset.at[i, "pattern"].labels))) for i in range(len(subset))
                          if layers[i] == 1}
            builder_layer1 = set(full.loc[(full["size"] == size) & (full["layer"] == 1), "itemset"])
            pushed_down[size] = len(front_sets - builder_layer1)
        overlap_alt = {size: len(set(alternative[size]) & {t[0] for t in theirs[size]}) for size in (1, 2, 3)}

        # float-noise ties: nearly equal but not identical objective values among candidate sets
        near = 0
        values = full[OBJ].to_numpy()
        for column in range(3):
            ordered = np.sort(values[:, column])
            gaps = np.diff(ordered)
            near += int(((gaps > 0) & (gaps < 1e-12)).sum())
        rounded = patterns.copy()
        rounded[OBJ] = rounded[OBJ].round(10)
        mine_rounded = my_selection(rounded, k=10)
        rounding_same = {s: [m[0] for m in mine_rounded[s]] == [t[0] for t in theirs[s]] for s in (1, 2, 3)}

        # where the selected sets come from, and how many cases the 2024 permutation step could shuffle
        # (a case is eligible when it contains the whole set and is longer than the set, tools.py:521-523)
        name_on_front = dict(zip(patterns["name"], patterns["on_front"]))
        sets_with_front_pattern = {tuple(sorted(set(p.labels))) for p, flag in zip(patterns["pattern"],
                                                                                   patterns["on_front"]) if flag}
        source_on_step_front = {s: sum(bool(name_on_front[t[2]]) for t in theirs[s]) for s in (1, 2, 3)}
        set_on_step_front = {s: sum(t[0] in sets_with_front_pattern for t in theirs[s]) for s in (1, 2, 3)}
        eligible = {s: [sum(set(t[0]) <= set(trace) and len(trace) > len(t[0]) for trace in trace_list)
                        for t in theirs[s]] for s in (1, 2, 3)}
        source_cases = {s: [int((selector.counts_[patterns.loc[patterns["name"] == t[2], "pattern"].iloc[0]] > 0).sum())
                            for t in theirs[s]] for s in (1, 2, 3)}

        report[log_name] = {
            "fit_seconds": round(fit_seconds, 2), "n_patterns": int(len(patterns)),
            "inputs_equal_builder_loader": bool(same_inputs), "case_attributes_constant_per_case": constant,
            "sample_patterns_recounted": int(len(sample)), "patterns_with_wrong_counts": int(n_count_bad),
            "max_abs_diff_scores_vs_own": max_diff, "step_fronts_equal_own_dominance": bool(fronts_ok),
            "step_candidates_equal_children_of_previous_front": bool(chain_ok),
            "selection_equal_own_implementation": {s: [m[0] for m in mine[s]] == [t[0] for t in theirs[s]]
                                                   and [m[1:] for m in mine[s]] == [t[1:] for t in theirs[s]]
                                                   for s in (1, 2, 3)},
            "selection_equal_stored_json": stored_equal, "lengths_ok": bool(lengths_ok),
            "one_row_per_set_and_sizes_1_to_3": bool(dedupe_ok),
            "n_candidate_sets": int(len(full)), "n_candidate_sets_expected": int(n_sets_expected),
            "refit_identical": bool(deterministic),
            "overlap_with_pattern_level_layering_top10": overlap_alt,
            "sets_with_a_first_front_pattern_but_not_in_layer_1": pushed_down,
            "near_equal_objective_values_below_1e-12": near,
            "selection_unchanged_after_rounding_objectives_to_10_decimals": rounding_same,
            "selected_sets_whose_source_pattern_is_on_a_step_front": source_on_step_front,
            "selected_sets_with_any_pattern_on_a_step_front": set_on_step_front,
            "eligible_cases_per_selected_set": eligible,
            "cases_with_source_pattern_per_selected_set": source_cases,
            "selected": {s: [{"set": list(t[0]), "layer": t[1], "source": t[2]} for t in theirs[s]] for s in (1, 2, 3)},
        }
        r = report[log_name]
        check(f"3.4 {log_name}: inputs from own loader == builder's loader (traces, labels, distance matrix)",
              same_inputs and constant, f"{len(traces)} cases, attributes constant per case: {constant}")
        check(f"3.5 {log_name}: counts and interest values of {len(sample)} sampled evaluated patterns == own",
              n_count_bad == 0 and max(max_diff.values()) < 1e-9, f"wrong counts {n_count_bad}, max diff {max_diff}")
        check(f"3.6 {log_name}: step fronts == own dominance; step s candidates == children of the step s-1 front",
              fronts_ok and chain_ok, f"fronts {fronts_ok}, candidates {chain_ok}; "
              f"{selector.step_summary_.to_dict('records')}")
        check(f"3.7 {log_name}: selected k=10 per length == own projection + non-dominated sorting + tie-break",
              all(r["selection_equal_own_implementation"].values()), r["selection_equal_own_implementation"])
        check(f"3.8 {log_name}: selection == stored impressed_sets_{log_name}.json; lengths; one row per set",
              all(stored_equal.values()) and lengths_ok and dedupe_ok and len(full) == n_sets_expected,
              f"stored {stored_equal}, candidate sets {len(full)} (expected {n_sets_expected})")
        check(f"3.9 {log_name}: second fit gives the identical table and selection", deterministic)
        check(f"3.10 {log_name}: overlap of the top 10 with 'layer the patterns, then dedupe by set'", None,
              f"{overlap_alt}; sets owning a first-front pattern that the selector puts below layer 1: {pushed_down}; "
              f"near-ties below 1e-12: {near}; selection unchanged after rounding objectives to 10 decimals: "
              f"{rounding_same}")
        check(f"3.11 {log_name}: of the 10 selected sets per length, how many have their source pattern / any of "
              f"their patterns on an IMPresseD step front", None,
              f"source pattern on a step front {source_on_step_front}; any pattern of the set {set_on_step_front}")
        check(f"3.12 {log_name}: cases the 2024 permutation could shuffle per selected set (whole set in the trace)",
              None, f"{eligible}; sets with fewer than 15 such cases: "
              f"{ {s: sum(e < 15 for e in eligible[s]) for s in eligible} }; cases containing the SOURCE PATTERN: "
              f"{source_cases}")
    RESULTS["part3"] = report
    return selectors


def _front_check_large(step_table: pd.DataFrame) -> bool:
    """Front check for big tables: nobody on the front is dominated, everybody else is dominated by a front row."""
    values = step_table[OBJ].to_numpy()
    flags = step_table["on_front"].to_numpy(bool)
    front = values[flags]
    for row, flag in zip(values, flags):
        dominated = bool((((front[:, 0] >= row[0]) & (front[:, 1] >= row[1]) & (front[:, 2] <= row[2]))
                          & ((front[:, 0] > row[0]) | (front[:, 1] > row[1]) | (front[:, 2] < row[2]))).any())
        if flag == dominated:
            return False
    return True


def naive_children(parent: Pattern, trace, max_gap) -> set[Pattern]:
    """Children of a direct-only parent in one trace, written straight from IMIPD.py:168-328 / :612-749."""
    size = len(parent.labels)
    children = set()
    for start in range(len(trace) - size + 1):
        if tuple(trace[start:start + size]) != parent.labels:
            continue
        end = start + size - 1
        before, after = start - 1, end + 1
        if before >= 0:
            children.add(Pattern((trace[before],) + parent.labels, (DIRECT,) + parent.edges))
            for far in range(before - max_gap, before):      # min(in) - gap <= node < min(in)
                if far >= 0:
                    children.add(Pattern((trace[far],) + parent.labels, (EVENTUAL,) + parent.edges))
        if after < len(trace):
            children.add(Pattern(parent.labels + (trace[after],), parent.edges + (DIRECT,)))
            for far in range(after + 1, after + max_gap + 1):  # max(out) < node <= max(out) + gap
                if far < len(trace):
                    children.add(Pattern(parent.labels + (trace[far],), parent.edges + (EVENTUAL,)))
        if size == 1 and before >= 0 and after < len(trace):
            children.add(Pattern((trace[before], parent.labels[0], trace[after]), (DIRECT, DIRECT)))
    return children


# --------------------------------------------------------------------------
# 4. Leakage / semantics, determinism across processes and case order
# --------------------------------------------------------------------------
def part4_semantics(selectors: dict) -> None:
    say("=== 4. which cases are mined, where labels enter, determinism ===")
    selector, traces, labels, dist = selectors["f1"]
    case_ids = list(traces)
    base = selection_as_lists(selector)

    # (a) labels only enter through IG
    takes_labels = [name for name in ("build_position_index", "find_instances", "extend_pattern", "extend_patterns",
                                      "count_matrix", "count_instances", "pareto_front", "pareto_layers")
                    if any("label" in p for p in inspect.signature(getattr(ic, name)).parameters)]
    rng = np.random.default_rng(3)
    shuffled = dict(zip(case_ids, rng.permutation([labels[c] for c in case_ids]).tolist()))
    other = ImpressedChainSelector(max_gap=MAX_GAP, steps=2, k=10).fit(traces, shuffled, dist)
    s0, o0 = selector.step_tables_[0], other.step_tables_[0]
    same_non_label = (list(s0["name"]) == list(o0["name"]) and s0["coverage"].equals(o0["coverage"])
                      and s0["CD"].equals(o0["CD"]) and s0["n_instances"].equals(o0["n_instances"]))
    ig_changed = not np.allclose(s0["IG"], o0["IG"])
    check("4.1 labels enter only through IG: extension/count functions take no labels; with permuted labels "
          "step-0 candidates, counts, coverage and CD are unchanged and IG changes",
          not takes_labels and same_non_label and ig_changed,
          f"functions with a label parameter: {takes_labels}; max IG at step 0 real {s0['IG'].max():.4f} vs "
          f"permuted {o0['IG'].max():.4f}; step-1 candidates real {len(selector.step_tables_[1])} vs permuted "
          f"{len(other.step_tables_[1])} (labels steer WHICH patterns are extended, by design)")

    # (b) the mined cases are whatever is passed to fit
    train_ids = sorted(rng.choice(case_ids, size=int(0.8 * len(case_ids)), replace=False).tolist())
    rows = [case_ids.index(c) for c in train_ids]
    sub = ImpressedChainSelector(max_gap=MAX_GAP, steps=2, k=10).fit(
        {c: traces[c] for c in train_ids}, {c: labels[c] for c in train_ids}, dist[np.ix_(rows, rows)])
    sub_sel = selection_as_lists(sub)
    overlap = {s: len({t[0] for t in sub_sel[s]} & {t[0] for t in base[s]}) for s in (1, 2, 3)}
    held_out = [c for c in case_ids if c not in set(train_ids)]
    check("4.2 mining scope is the caller's choice: fit() on an 80 % subset runs and uses only those cases",
          sub.case_ids_ == train_ids and len(next(iter(sub.counts_.values()))) == len(train_ids),
          f"default runner (c4_run_selector.run_log) passes ALL cases = whole-log mining incl. labels of every "
          f"case; top-10 overlap whole log vs one random 80 % subset: {overlap} of 10; held-out cases {len(held_out)}")
    RESULTS["part4"] = {"overlap_whole_vs_80pct": overlap, "functions_with_label_parameter": takes_labels}

    # (c) case order must not matter
    order = list(rng.permutation(len(case_ids)))
    permuted_ids = [case_ids[i] for i in order]
    permuted = ImpressedChainSelector(max_gap=MAX_GAP, steps=2, k=10).fit(
        {c: traces[c] for c in permuted_ids}, labels, dist[np.ix_(order, order)])
    perm_sel = selection_as_lists(permuted)
    same_sets = {s: [t[0] for t in perm_sel[s]] == [t[0] for t in base[s]] for s in (1, 2, 3)}
    same_all = {s: perm_sel[s] == base[s] for s in (1, 2, 3)}
    merged = selector.patterns_.merge(permuted.patterns_, on="name", how="outer", suffixes=("", "_p"), indicator=True)
    both = merged[merged["_merge"] == "both"]
    value_diff = {c: float(np.abs(both[c] - both[c + "_p"]).max()) for c in OBJ}
    n_flag = int((both["on_front"] != both["on_front_p"]).sum())
    check("4.3 shuffled case order (f1): same selected sets in the same order",
          all(same_sets.values()), f"sets {same_sets}, incl. layer/source {same_all}; evaluated patterns "
          f"{len(selector.patterns_)} vs {len(permuted.patterns_)} "
          f"(only in one: {int((merged['_merge'] != 'both').sum())}); "
          f"max value diff {value_diff}; on_front flags that differ: {n_flag}")
    RESULTS["part4"]["case_order"] = {"same_sets": same_sets, "same_incl_layer_source": same_all,
                                      "patterns_only_in_one_run": int((merged["_merge"] != "both").sum()),
                                      "max_value_diff": value_diff, "on_front_flags_differ": n_flag}

    # (d) hash seed must not matter
    outputs = []
    for seed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        done = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--print-selection", "f1"],
                              capture_output=True, text=True, env=env, check=False)
        outputs.append(done.stdout.strip().splitlines()[-1] if done.returncode == 0 and done.stdout.strip() else
                       f"ERROR {done.stderr[-300:]}")
    this = json.dumps({str(s): [[list(t[0]), t[1], t[2]] for t in base[s]] for s in (1, 2, 3)}, sort_keys=True)
    check("4.4 two fresh processes with PYTHONHASHSEED 1 and 2 give this process's selection",
          outputs[0] == outputs[1] == this, "identical" if outputs[0] == outputs[1] == this else outputs[0][:300])


def print_selection(log_name: str) -> None:
    """Helper for the hash-seed check: print the selection of one log as one JSON line."""
    traces, labels, dist, _same, _constant = load_inputs(log_name)
    selector = ImpressedChainSelector(max_gap=MAX_GAP, steps=2, k=10).fit(traces, labels, dist)
    selection = selection_as_lists(selector)
    print(json.dumps({str(s): [[list(t[0]), t[1], t[2]] for t in selection[s]] for s in (1, 2, 3)}, sort_keys=True))


# --------------------------------------------------------------------------
# 5. Mechanical code-quality checks
# --------------------------------------------------------------------------
def part5_quality() -> None:
    say("=== 5. code quality (mechanical part) ===")
    done = subprocess.run([sys.executable, "-m", "unittest", "test_impressed_chain", "-q"], cwd=EXPERIMENTS,
                          capture_output=True, text=True, check=False)
    tail = (done.stderr.strip().splitlines() or [""])[-3:]
    check("5.1 the builder's own unit tests pass", done.returncode == 0, " | ".join(tail))
    source = (EXPERIMENTS / "impressed_chain.py").read_text(encoding="utf-8").splitlines()
    long_lines = [i + 1 for i, line in enumerate(source) if len(line) > 120]
    absolute = [i + 1 for i, line in enumerate(source) if "C:/" in line or "C:\\" in line]
    check("5.2 impressed_chain.py: no line over 120 characters, no absolute path", not long_lines and not absolute,
          f"{len(source)} lines; long {long_lines}; absolute {absolute}; lines over 99: "
          f"{sum(len(line) > 99 for line in source)}")
    selector = ImpressedChainSelector()
    unfitted = None
    try:
        selector.select()
    except Exception as error:  # noqa: BLE001 - we want to see which error a student would get
        unfitted = f"{type(error).__name__}: {error}"
    check("5.3 calling select() before fit()", None, unfitted)
    bad_shape = None
    try:
        ImpressedChainSelector().fit({"a": ["x", "y"], "b": ["y"]}, {"a": 0, "b": 1}, np.zeros((3, 3)))
        bad_shape = "no error raised for a 3x3 distance matrix with 2 cases"
    except Exception as error:  # noqa: BLE001
        bad_shape = f"{type(error).__name__}: {error}"
    check("5.4 fit() with a distance matrix of the wrong size", None, bad_shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--print-selection", metavar="LOG", help="internal: print one log's selection as JSON")
    parser.add_argument("--skip-original", action="store_true", help="skip part 2 (the slow original code)")
    args = parser.parse_args()
    if args.print_selection:
        print_selection(args.print_selection)
        return
    start = time.perf_counter()
    part1_toy()
    if not args.skip_original:
        part2_original()
    part2b_c3()
    selectors = part3_selection()
    part4_semantics(selectors)
    part5_quality()
    counts = {status: sum(c["status"] == status for c in CHECKS) for status in ("PASS", "FAIL", "INFO")}
    say(f"done in {time.perf_counter() - start:.0f} s: {counts}")
    for item in CHECKS:
        if item["status"] == "FAIL":
            say(f"FAILED: {item['check']}")
    output = {"checks": CHECKS, "summary": counts, "results": RESULTS}
    (HERE / "verify_impressed.json").write_text(json.dumps(output, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
