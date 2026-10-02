"""Experiment C4, part (f): cross-check ``impressed_chain.py`` against the ORIGINAL 2023 code.

Run from anywhere (start it in the background, it takes several minutes):
    python -u experiments/c4_crosscheck_original.py

The original functions are imported unchanged from
``external/InteractivePatternDetection`` (never the GUI module) and are fed a
random subset of 150 f1 cases with ``delta_time = -1`` (chain traces).

Parts
-----
A  ``Pattern_extension`` for 5 frequent seed activities, a fresh dictionary per
   seed: same child patterns?  same per-case instance counts?  timing.
B  The same 5 seeds in ONE shared dictionary, the way ``Auto_IMPID.py:45-73``
   does it: shows the cross-core merge and the count columns the tool writes.
C  ``Single_Pattern_Extender`` (step 2) for several step-1 parents.
D  The original interest functions on the patterns of B and C versus
   ``score_patterns``.
E  The whole original automatic mode (``AutoStepWise_PPD``, 2 extension
   steps) with instrumentation; at every step our extension and our interest
   values are compared GIVEN the original's front of the previous step.
F  (``--full-core``) one ``Pattern_extension`` pass on the full f1 log for one
   core activity: the timing asked for in Guide Part 6, Table C, row C4.
G  (``--full-front``) the original step-1 extension on the full log for all
   activities of our step-0 Pareto front, in one shared dictionary.

Outputs: ``results/experiments/C4/crosscheck.json`` and ``crosscheck_tables.md``.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # IMIPD imports pyplot; never open a window

import networkx as nx  # noqa: E402
import networkx.algorithms.isomorphism as iso  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from paretoset import paretoset  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
ORIGINAL_DIR = PROJECT_ROOT / "external" / "InteractivePatternDetection"
RESULT_DIR = PROJECT_ROOT / "results" / "experiments" / "C4"

# The 2023 and the 2024 repository both ship a module named tools.py: only the 2023 folder is added.
sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(0, str(ORIGINAL_DIR))

import Auto_IMPID  # noqa: E402
import IMIPD  # noqa: E402

from c4_run_selector import load_selector_inputs, markdown_table  # noqa: E402
from engine import ACTIVITY_COL, CASE_COL, DATASETS, LABEL_COL, POSITION_COL  # noqa: E402
from impressed_chain import (  # noqa: E402
    DIRECT,
    EVENTUAL,
    SENSES,
    ImpressedChainSelector,
    Pattern,
    build_position_index,
    count_instances,
    count_matrix,
    extend_pattern,
    extend_patterns,
    pareto_front,
    score_patterns,
)

TIME_COL = "time:timestamp"
SEED = 2023
N_CASES = 150
N_SEEDS = 5
MAX_GAP = 3
DELTA_TIME = -1
N_ISOMORPHISM_PAIRS = 3000
OBJECTIVES = ["IG", "coverage", "CD"]
ORIGINAL_COLUMNS = {"IG": "Outcome_Interest", "coverage": "Frequency_Interest", "CD": "Case_Distance_Interest"}


def say(message: str) -> None:
    """Time-stamped progress line."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


@contextlib.contextmanager
def quiet():
    """Silence the prints and pandas warnings of the original code."""
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        yield


# --------------------------------------------------------------------------
# Inputs in the shape the original code expects
# --------------------------------------------------------------------------
class OriginalInputs:
    """A (sub)log prepared for both implementations.

    Attributes
    ----------
    case_ids : list of str, sorted
    traces : list of activity lists (order of ``case_ids``)
    labels : numpy array of 0/1
    dist : square distance matrix of these cases (sliced from the whole-log matrix)
    data : event table for the original code (case id, activity, timestamp)
    graphs : case id -> trace graph from the original ``Trace_graph_generator``
    """

    def __init__(self, log_name: str, n_cases: int | None, seed: int):
        log, dist_all, _ = load_selector_inputs(log_name)
        if n_cases is None:
            chosen = list(log.case_ids)
        else:
            rng = np.random.default_rng(seed)
            chosen = sorted(rng.choice(log.case_ids, size=n_cases, replace=False).tolist())
        rows = [log.case_ids.index(case) for case in chosen]
        self.case_ids = chosen
        self.traces = [log.traces[case] for case in chosen]
        self.labels = np.array([log.labels[case] for case in chosen])
        self.dist = dist_all[np.ix_(rows, rows)]
        self.position_index = build_position_index(self.traces)
        self.data = self._event_table(log_name)
        self.colours = dict.fromkeys(self.data[ACTIVITY_COL].unique(), "#000000") | {"start": "k", "end": "k"}

        start = time.perf_counter()
        self.graphs = {
            case: IMIPD.Trace_graph_generator(self.data, DELTA_TIME, case, self.colours,
                                              CASE_COL, ACTIVITY_COL, TIME_COL)
            for case in chosen
        }
        self.graph_seconds = time.perf_counter() - start

    def _event_table(self, log_name: str) -> pd.DataFrame:
        """Events of the chosen cases with real timestamps and the engine's activity names."""
        raw = pd.read_csv(DATASETS[log_name])
        raw[CASE_COL] = raw[CASE_COL].astype(str)
        raw = raw[raw[CASE_COL].isin(self.case_ids)].sort_values([CASE_COL, POSITION_COL])
        data = raw[[CASE_COL, TIME_COL, LABEL_COL]].reset_index(drop=True)
        data[TIME_COL] = pd.to_datetime(data[TIME_COL])
        data[ACTIVITY_COL] = [activity for trace in self.traces for activity in trace]
        assert list(data[CASE_COL].drop_duplicates()) == self.case_ids
        return data

    def chain_report(self) -> dict:
        """Check that every original trace graph is the chain 0 -> 1 -> ... -> L-1."""
        n_chain = 0
        for case, trace in zip(self.case_ids, self.traces):
            graph = self.graphs[case]
            values = [graph.nodes[node]["value"] for node in range(len(trace))]
            edges = {(u, v) for u, v, flag in graph.edges(data="eventually") if flag is False}
            no_parallel = not any(nx.get_node_attributes(graph, "parallel").values())
            n_chain += (values == list(trace) and no_parallel and graph.number_of_edges() == len(trace) - 1
                        and edges == {(i, i + 1) for i in range(len(trace) - 1)})
        return {"n_graphs": len(self.graphs), "n_chain_graphs": int(n_chain),
                "seconds_Trace_graph_generator": self.graph_seconds}

    def patient_data(self) -> pd.DataFrame:
        """Case table as the GUI builds it: id, outcome, one count column per activity."""
        patient = pd.DataFrame({CASE_COL: self.case_ids, LABEL_COL: self.labels})
        activities = list(self.data[ACTIVITY_COL].unique())
        counts = np.array([[trace.count(activity) for activity in activities] for trace in self.traces])
        return pd.concat([patient, pd.DataFrame(counts, columns=activities)], axis=1)

    def distance_lookup(self):
        """Condensed distances and the pair look-up lists of GUI_IMPresseD_tool.py:404-411."""
        n = len(self.case_ids)
        condensed = squareform(self.dist, checks=False)
        pair_cases = [(a, b) for a in range(n) for b in range(a + 1, n)]
        start_search_points, i = [], 0
        for k in range(n):
            start_search_points.append(k * n - (i + k))
            i += k
        return condensed, pair_cases, start_search_points


# --------------------------------------------------------------------------
# Original graph pattern -> Pattern tuple
# --------------------------------------------------------------------------
def graph_to_pattern(graph: nx.DiGraph) -> Pattern:
    """Read an original pattern graph as (labels, edge types); fails if it is not a path."""
    nodes = sorted(graph.nodes)  # node ids are trace positions of the first instance
    if graph.number_of_edges() != len(nodes) - 1 or not all(graph.has_edge(u, v) for u, v in zip(nodes, nodes[1:])):
        raise ValueError(f"original pattern is not a path: {list(graph.edges)}")
    edges = tuple(EVENTUAL if graph.edges[u, v]["eventually"] else DIRECT for u, v in zip(nodes, nodes[1:]))
    return Pattern(tuple(graph.nodes[node]["value"] for node in nodes), edges)


def isomorphism_report(dictionary: dict, rng) -> dict:
    """Is ``nx.is_isomorphic`` (as the original calls it) the same test as tuple equality?"""
    ids = list(dictionary)
    keys = {pid: graph_to_pattern(dictionary[pid]["pattern"]) for pid in ids}
    node_match = iso.categorical_node_match("value", None)
    edge_match = iso.categorical_node_match("eventually", [True, False])  # as IMIPD.py:175

    by_labels: dict[tuple, list] = {}
    for pid in ids:
        by_labels.setdefault(tuple(sorted(keys[pid].labels)), []).append(pid)
    hard_pairs = [pair for group in by_labels.values() for pair in itertools.combinations(group, 2)]
    random_pairs = [tuple(rng.choice(len(ids), size=2, replace=False)) for _ in range(N_ISOMORPHISM_PAIRS)]
    pairs = hard_pairs + [(ids[a], ids[b]) for a, b in random_pairs]
    disagreements = 0
    for first, second in pairs:
        isomorphic = nx.is_isomorphic(dictionary[first]["pattern"], dictionary[second]["pattern"],
                                      node_match=node_match, edge_match=edge_match)
        disagreements += int(isomorphic != (keys[first] == keys[second]))
    return {"n_patterns": len(ids), "n_distinct_tuples": len(set(keys.values())),
            "pairs_tested": len(pairs), "pairs_with_same_activity_multiset": len(hard_pairs),
            "disagreements_isomorphic_vs_tuple_equal": disagreements}


# --------------------------------------------------------------------------
# Running the original extension
# --------------------------------------------------------------------------
def original_step1(inputs: OriginalInputs, cores: list[str], dictionary: dict, patient: pd.DataFrame | None):
    """Auto_IMPID.py:48-73 for the given cores; returns per-core ids and timings.

    ``dictionary`` is filled in place.  When ``patient`` is given the count
    columns are written exactly as Auto_IMPID.py:68-73 does.
    """
    data = inputs.data
    ids_per_core, seconds_per_core, extension_seconds_per_core = {}, {}, {}
    for core in cores:
        start = time.perf_counter()
        extension_seconds = 0.0
        filtered_cases = data.loc[data[ACTIVITY_COL] == core, CASE_COL]
        filtered_main_data = data[data[CASE_COL].isin(filtered_cases)]
        new_patterns_for_core = []
        for case in filtered_main_data[CASE_COL].unique():
            case_data = filtered_main_data[filtered_main_data[CASE_COL] == case]
            trace_graph = inputs.graphs[case].copy()
            tick = time.perf_counter()
            dictionary, new_patterns_for_core = IMIPD.Pattern_extension(
                case_data, trace_graph, core, CASE_COL, dictionary, MAX_GAP, new_patterns_for_core)
            extension_seconds += time.perf_counter() - tick
        if patient is not None:
            with quiet():
                patient[new_patterns_for_core] = 0
                for pid in new_patterns_for_core:
                    cases = dictionary[pid]["Instances"]["case"]
                    for case in np.unique(cases):
                        patient.loc[patient[CASE_COL] == case, pid] = cases.count(case)
        ids_per_core[core] = new_patterns_for_core
        seconds_per_core[core] = time.perf_counter() - start
        extension_seconds_per_core[core] = extension_seconds
    return ids_per_core, seconds_per_core, extension_seconds_per_core


def original_counts(entry: dict, case_ids: list[str]) -> np.ndarray:
    """Per-case number of instances stored in one dictionary entry."""
    per_case = Counter(entry["Instances"]["case"])
    return np.array([per_case.get(case, 0) for case in case_ids])


def compare_children(original: dict, ours: dict, case_ids: list[str], predicted_factor) -> dict:
    """Compare original dictionary entries with our children.

    ``original`` maps a pattern id to its dictionary entry, ``ours`` a Pattern
    to its set of instances.  ``predicted_factor(pattern)`` is the number of
    times we expect the original to count each instance.
    """
    original_keys = {pid: graph_to_pattern(entry["pattern"]) for pid, entry in original.items()}
    ours_counts = dict(zip(ours, count_matrix(ours, len(case_ids)).T))
    shared = [pid for pid, key in original_keys.items() if key in ours]
    factors, identical, non_uniform, as_predicted = Counter(), 0, 0, 0
    for pid in shared:
        key = original_keys[pid]
        theirs, mine = original_counts(original[pid], case_ids), ours_counts[key]
        identical += int(np.array_equal(theirs, mine))
        factor = theirs.sum() / mine.sum()
        if np.array_equal(theirs, factor * mine):
            factors[int(factor) if factor == int(factor) else float(factor)] += 1
            as_predicted += int(factor == predicted_factor(key))
        else:
            non_uniform += 1
    return {
        "n_original_patterns": len(original), "n_original_distinct_tuples": len(set(original_keys.values())),
        "n_our_patterns": len(ours),
        "only_in_original": sorted(str(key) for key in set(original_keys.values()) - set(ours)),
        "only_in_ours": sorted(str(key) for key in set(ours) - set(original_keys.values())),
        "patterns_with_identical_counts": identical,
        "patterns_by_count_factor_original_over_ours": {str(k): v for k, v in sorted(factors.items())},
        "patterns_where_factor_differs_between_cases": non_uniform,
        "patterns_with_predicted_factor": as_predicted,
    }


# --------------------------------------------------------------------------
# Parts A - D
# --------------------------------------------------------------------------
def part_a(inputs: OriginalInputs, seeds: list[str], rng) -> tuple[list[dict], dict]:
    """Step-1 extension per seed with a fresh dictionary; returns rows and the dictionaries."""
    rows, dictionaries = [], {}
    for seed_activity in seeds:
        dictionary: dict = {}
        _, seconds, extension_seconds = original_step1(inputs, [seed_activity], dictionary, None)
        start = time.perf_counter()
        ours = extend_pattern(Pattern((seed_activity,)), inputs.traces, inputs.position_index, MAX_GAP)
        our_seconds = time.perf_counter() - start

        def predicted(pattern):  # a -> a and a ~> a are found from both of their nodes
            return 2 if len(pattern.labels) == 2 and pattern.labels[0] == pattern.labels[1] else 1

        comparison = compare_children(dictionary, ours, inputs.case_ids, predicted)
        comparison.update({
            "seed": seed_activity,
            "n_cases_with_seed": int(sum(seed_activity in trace for trace in inputs.traces)),
            "n_seed_events": int(sum(trace.count(seed_activity) for trace in inputs.traces)),
            "seconds_original_loop": seconds[seed_activity],
            "seconds_original_Pattern_extension_calls": extension_seconds[seed_activity],
            "seconds_ours": our_seconds,
            "isomorphism": isomorphism_report(dictionary, rng),
        })
        rows.append(comparison)
        dictionaries[seed_activity] = dictionary
        say(f"A  seed {seed_activity}: original {len(dictionary)} patterns in {seconds[seed_activity]:.1f} s, "
            f"ours {len(ours)} in {our_seconds:.4f} s")
    return rows, dictionaries


def part_b(inputs: OriginalInputs, seeds: list[str]) -> tuple[dict, dict, pd.DataFrame, dict]:
    """Step-1 extension of all seeds into one shared dictionary (as the automatic mode)."""
    dictionary: dict = {}
    patient = inputs.patient_data()
    ids_per_core, seconds, _ = original_step1(inputs, seeds, dictionary, patient)
    start = time.perf_counter()
    ours, _ = extend_patterns([Pattern((seed,)) for seed in seeds], inputs.traces, inputs.position_index, MAX_GAP)
    our_seconds = time.perf_counter() - start

    def predicted(pattern):  # found once from every node whose activity is one of the cores
        return sum(label in seeds for label in pattern.labels) if len(pattern.labels) == 2 else 1

    comparison = compare_children(dictionary, ours, inputs.case_ids, predicted)

    # The count COLUMNS the tool writes (frozen after the creating core's loop).
    keys = {pid: graph_to_pattern(entry["pattern"]) for pid, entry in dictionary.items()}
    ours_counts = dict(zip(ours, count_matrix(ours, len(inputs.case_ids)).T))
    column_factors = Counter()
    for pid, key in keys.items():
        theirs, mine = patient[pid].to_numpy(), ours_counts[key]
        factor = theirs.sum() / mine.sum()
        column_factors[str(int(factor)) if np.array_equal(theirs, factor * mine) else "not uniform"] += 1
    comparison.update({
        "seeds": seeds, "seconds_original": sum(seconds.values()), "seconds_ours": our_seconds,
        "count_columns_by_factor_original_over_ours": dict(column_factors),
    })
    say(f"B  shared dictionary: original {len(dictionary)} patterns in {sum(seconds.values()):.1f} s, "
        f"ours {len(ours)} in {our_seconds:.4f} s")
    return comparison, dictionary, patient, ours


def choose_parents(dictionary: dict, seed: str) -> list[str]:
    """Step-2 parents for one seed: 2 most frequent direct pairs, the self-loop, the top context."""
    keys = {pid: graph_to_pattern(entry["pattern"]) for pid, entry in dictionary.items()}
    size = {pid: len(entry["Instances"]["case"]) for pid, entry in dictionary.items()}
    direct = sorted((pid for pid, key in keys.items() if key.extendable), key=lambda pid: -size[pid])
    pairs = [pid for pid in direct if len(keys[pid].labels) == 2 and len(set(keys[pid].labels)) == 2][:2]
    loops = [pid for pid in direct if keys[pid] == Pattern((seed, seed), (DIRECT,))]
    contexts = [pid for pid in direct if len(keys[pid].labels) == 3][:1]
    return pairs + loops + contexts


def part_c(inputs: OriginalInputs, dictionaries: dict) -> tuple[list[dict], dict, pd.DataFrame]:
    """Step-2 extension of chosen parents with the original ``Single_Pattern_Extender``."""
    rows, children_all = [], {}
    patient = inputs.patient_data()
    for seed, dictionary in dictionaries.items():
        for parent_id in choose_parents(dictionary, seed):
            parent = graph_to_pattern(dictionary[parent_id]["pattern"])
            # fresh dictionary of part A: only a -> a holds every instance twice
            parent_factor = 2 if len(parent.labels) == 2 and len(set(parent.labels)) == 1 else 1
            start = time.perf_counter()
            with quiet():
                _, children, patient = IMIPD.Single_Pattern_Extender(
                    dict(dictionary), parent_id, patient, inputs.graphs, inputs.data, MAX_GAP,
                    ACTIVITY_COL, CASE_COL)
            seconds = time.perf_counter() - start
            start = time.perf_counter()
            ours = extend_pattern(parent, inputs.traces, inputs.position_index, MAX_GAP)
            our_seconds = time.perf_counter() - start

            def predicted(pattern, parent_factor=parent_factor):
                # an all-direct child of one repeated activity is reached as "preceding" and as "following"
                same = len(set(pattern.labels)) == 1 and pattern.extendable
                return parent_factor * (2 if same else 1)

            comparison = compare_children(children, ours, inputs.case_ids, predicted)
            comparison.update({"parent": str(parent), "parent_id": parent_id,
                               "parent_instances_in_original": len(dictionary[parent_id]["Instances"]["case"]),
                               "seconds_original": seconds, "seconds_ours": our_seconds})
            rows.append(comparison)
            children_all.update(children)
            say(f"C  parent {parent}: original {len(children)} children in {seconds:.1f} s, "
                f"ours {len(ours)} in {our_seconds:.4f} s")
    return rows, children_all, patient


def compare_scores(original_frame: pd.DataFrame, ours: pd.DataFrame) -> dict:
    """Largest absolute difference per interest function (NaN must coincide with NaN)."""
    report = {"n_patterns": int(len(ours))}
    for name, column in ORIGINAL_COLUMNS.items():
        theirs = original_frame[column].to_numpy(dtype=float)
        mine = ours[name].to_numpy(dtype=float)
        both = ~np.isnan(theirs) & ~np.isnan(mine)
        report[f"max_abs_diff_{name}"] = float(np.abs(theirs[both] - mine[both]).max()) if both.any() else 0.0
        report[f"nan_mismatch_{name}"] = int((np.isnan(theirs) != np.isnan(mine)).sum())
        report[f"n_nan_{name}"] = int(np.isnan(theirs).sum())
        report[f"original_range_{name}"] = (f"{np.nanmin(theirs):.4f} .. {np.nanmax(theirs):.4f}"
                                            if both.any() else "all NaN")
    return report


def part_d(inputs: OriginalInputs, entries: dict, patient: pd.DataFrame, label: str) -> dict:
    """Original interest functions on the tool's own count columns versus ours on our counts."""
    condensed, pair_cases, start_search_points = inputs.distance_lookup()
    ids = list(entries)
    start = time.perf_counter()
    with quiet():
        original_frame = IMIPD.create_pattern_attributes(patient, LABEL_COL, ids, condensed, pair_cases,
                                                         start_search_points, "binary")
    original_seconds = time.perf_counter() - start

    # Our counts come from the independent plain scan, not from the extension code.
    keys = [graph_to_pattern(entries[pid]["pattern"]) for pid in ids]
    counts = np.array([[count_instances(key, trace, MAX_GAP) for key in keys] for trace in inputs.traces])
    start = time.perf_counter()
    ours = score_patterns(counts, inputs.labels, inputs.dist, undefined_cd=np.nan)
    our_seconds = time.perf_counter() - start
    report = compare_scores(original_frame, ours)
    report.update({"patterns": label, "seconds_original": original_seconds, "seconds_ours": our_seconds})
    say(f"D  {label}: {report}")
    return report


# --------------------------------------------------------------------------
# Part E: the original automatic mode, step by step
# --------------------------------------------------------------------------
class AutoModeRecorder:
    """Wraps the functions ``AutoStepWise_PPD`` calls, to keep what it does not return."""

    def __init__(self):
        self.frames, self.masks, self.scored_cases, self.dictionary = [], [], [], {}

    def install(self, module):
        original = {name: getattr(module, name) for name in
                    ("create_pattern_attributes", "paretoset", "Pattern_extension", "Single_Pattern_Extender")}

        def create_pattern_attributes(patient_data, *args, **kwargs):
            frame = original["create_pattern_attributes"](patient_data, *args, **kwargs)
            self.frames.append(frame.copy())
            self.scored_cases.append(list(patient_data[CASE_COL]))
            return frame

        def paretoset(costs, *args, **kwargs):
            mask = original["paretoset"](costs, *args, **kwargs)
            self.masks.append(np.asarray(mask))
            return mask

        def pattern_extension(*args, **kwargs):
            result = original["Pattern_extension"](*args, **kwargs)
            self.dictionary = result[0]
            return result

        def single_pattern_extender(*args, **kwargs):
            result = original["Single_Pattern_Extender"](*args, **kwargs)
            self.dictionary = result[0]
            return result

        module.create_pattern_attributes = create_pattern_attributes
        module.paretoset = paretoset
        module.Pattern_extension = pattern_extension
        module.Single_Pattern_Extender = single_pattern_extender
        return original

    @staticmethod
    def uninstall(module, original):
        for name, function in original.items():
            setattr(module, name, function)


def part_e(inputs: OriginalInputs, max_extension_step: int = 2) -> dict:
    """Run ``AutoStepWise_PPD`` and compare every step with our functions."""
    condensed, pair_cases, start_search_points = inputs.distance_lookup()
    save_path = RESULT_DIR / "original_auto_output"
    save_path.mkdir(parents=True, exist_ok=True)
    recorder = AutoModeRecorder()
    originals = recorder.install(Auto_IMPID)
    start = time.perf_counter()
    try:
        with quiet():
            Auto_IMPID.AutoStepWise_PPD(
                max_extension_step, MAX_GAP, 0.2, inputs.data, inputs.patient_data(), condensed, pair_cases,
                start_search_points, CASE_COL, ACTIVITY_COL, LABEL_COL, "binary", TIME_COL,
                [ORIGINAL_COLUMNS[name] for name in OBJECTIVES], ["Max", "Max", "Min"], DELTA_TIME,
                inputs.colours, str(save_path))
    finally:
        recorder.uninstall(Auto_IMPID, originals)
    original_seconds = time.perf_counter() - start
    say(f"E  AutoStepWise_PPD finished in {original_seconds:.1f} s")

    train_rows = [inputs.case_ids.index(case) for case in recorder.scored_cases[0]]
    train_labels = inputs.labels[train_rows]
    train_dist = inputs.dist[np.ix_(train_rows, train_rows)]

    def key_of(pattern_id: str) -> Pattern:
        if pattern_id in recorder.dictionary:
            return graph_to_pattern(recorder.dictionary[pattern_id]["pattern"])
        return Pattern((pattern_id,))  # step 0: the id is the activity

    steps, our_seconds = [], 0.0
    previous_front: list[Pattern] = []
    for step, (frame, mask) in enumerate(zip(recorder.frames, recorder.masks)):
        original_keys = [key_of(pid) for pid in frame["patterns"]]
        start = time.perf_counter()
        if step == 0:
            ours = {Pattern((activity,)): {(case, (pos,)) for case, pos in events}
                    for activity, events in inputs.position_index.items()}
        else:
            ours, _ = extend_patterns(previous_front, inputs.traces, inputs.position_index, MAX_GAP)
        counts = count_matrix(ours, len(inputs.case_ids))[train_rows]
        scores = score_patterns(counts, train_labels, train_dist, undefined_cd=np.nan)
        our_seconds += time.perf_counter() - start
        row_of = {key: row for row, key in enumerate(ours)}

        in_ours = np.array([key in row_of for key in original_keys])
        our_rows = [row_of[key] for key in original_keys if key in row_of]
        values = compare_scores(frame[in_ours], scores.iloc[our_rows])

        # Their front from OUR values: same row order, NaN kept, distinct=True (paretoset's default).
        senses = [SENSES[name] for name in OBJECTIVES]
        in_their_order = scores.iloc[our_rows][OBJECTIVES]
        emulated = paretoset(in_their_order.to_numpy(), sense=senses, distinct=True)
        without_nan = paretoset(in_their_order.fillna({"CD": 1.0}).to_numpy(), sense=senses, distinct=True)

        # Front by OUR rules on the same candidates: distinct=False, undefined distance = 1.0.
        our_rule_scores = scores.fillna({"CD": 1.0})
        on_our_front = pareto_front(our_rule_scores, OBJECTIVES, distinct=False)
        our_front = {key for key, on_front in zip(ours, on_our_front) if on_front}
        original_front = [key for key, on_front in zip(original_keys, mask) if on_front]
        front_vectors = {tuple(our_rule_scores.iloc[row_of[key]][OBJECTIVES]) for key in original_front}
        extra = our_front - set(original_front)
        extra_tied = sum(tuple(our_rule_scores.iloc[row_of[key]][OBJECTIVES]) in front_vectors for key in extra)
        steps.append({
            "step": step,
            "original_candidates": len(original_keys),
            "original_distinct_tuples": len(set(original_keys)),
            "our_candidates": len(ours),
            "only_in_original": len(set(original_keys) - set(ours)),
            "only_in_ours": len(set(ours) - set(original_keys)),
            **{name: value for name, value in values.items()
               if name != "n_patterns" and not name.startswith("original_range")},
            "original_front": int(mask.sum()),
            "original_front_distinct_tuples": len(set(original_front)),
            "original_front_reproduced_from_our_values": bool(np.array_equal(emulated, mask[in_ours])),
            "front_distinct_true_undefined_cd_1": int(without_nan.sum()),
            "front_by_our_rules": len(our_front),
            "in_both_fronts": len(set(original_front) & our_front),
            "our_rules_front_only_tied_with_original_front_member": int(extra_tied),
            "original_front_only": sorted(str(key) for key in set(original_front) - our_front),
            "our_rules_front_only": sorted(str(key) for key in our_front - set(original_front)),
        })
        previous_front = list(dict.fromkeys(original_front))  # teacher forcing: extend THEIR front
        say(f"E  step {step}: {steps[-1]}")
    return {"seconds_original_AutoStepWise_PPD": original_seconds,
            "seconds_ours_extension_and_scoring": our_seconds,
            "n_train_cases": len(train_rows), "steps": steps}


# --------------------------------------------------------------------------
# Part F: one core activity on the full log
# --------------------------------------------------------------------------
def part_f(inputs: OriginalInputs, cores: list[str], tag: str) -> dict:
    """Original step-1 extension of ``cores`` on the whole log (one shared dictionary) versus ours."""
    dictionary: dict = {}
    _, seconds, extension_seconds = original_step1(inputs, cores, dictionary, None)
    start = time.perf_counter()
    ours, _ = extend_patterns([Pattern((core,)) for core in cores], inputs.traces, inputs.position_index, MAX_GAP)
    our_seconds = time.perf_counter() - start

    def predicted(pattern):  # found once from every node whose activity is one of the cores
        return sum(label in cores for label in pattern.labels) if len(pattern.labels) == 2 else 1

    report = compare_children(dictionary, ours, inputs.case_ids, predicted)
    report.update({
        "cores": cores, "n_cases": len(inputs.case_ids),
        "n_core_events": int(sum(trace.count(core) for trace in inputs.traces for core in cores)),
        "seconds_original_loop": sum(seconds.values()),
        "seconds_original_per_core": {core: round(value, 1) for core, value in seconds.items()},
        "seconds_original_Pattern_extension_calls": sum(extension_seconds.values()),
        "seconds_ours": our_seconds, "seconds_Trace_graph_generator": inputs.graph_seconds,
        "isomorphism": isomorphism_report(dictionary, np.random.default_rng(SEED)),
    })
    say(f"{tag}  cores {cores}: original {len(dictionary)} patterns in {sum(seconds.values()):.1f} s, "
        f"ours {len(ours)} in {our_seconds:.4f} s; per core {report['seconds_original_per_core']}")
    return report


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def flat(row: dict, columns: list[str]) -> dict:
    """Pick columns of a result row; lists are shown by their length."""
    return {column: (len(row[column]) if isinstance(row[column], list) else
                     json.dumps(row[column]) if isinstance(row[column], dict) else row[column])
            for column in columns}


def write_tables(results: dict) -> None:
    """Write ``crosscheck_tables.md`` from the result dictionary."""
    child_columns = ["n_original_patterns", "n_our_patterns", "only_in_original", "only_in_ours",
                     "patterns_with_identical_counts", "patterns_by_count_factor_original_over_ours",
                     "patterns_where_factor_differs_between_cases", "patterns_with_predicted_factor",
                     "seconds_original", "seconds_ours"]
    parts = ["# C4 cross-check tables (generated by c4_crosscheck_original.py)\n",
             f"Subset: {results['subset']}\n", f"Chain check: {results['chain_check']}\n"]
    rows_a = [{"seed": row["seed"], "cases": row["n_cases_with_seed"], "events": row["n_seed_events"],
               **flat(dict(row, seconds_original=row["seconds_original_loop"]), child_columns),
               "iso pairs tested": row["isomorphism"]["pairs_tested"],
               "iso disagreements": row["isomorphism"]["disagreements_isomorphic_vs_tuple_equal"]}
              for row in results["A_step1_fresh_dictionary"]]
    parts += ["## A. Step-1 extension, fresh dictionary per seed\n", markdown_table(pd.DataFrame(rows_a))]
    row_b = results["B_step1_shared_dictionary"]
    parts += ["\n## B. Step-1 extension, one shared dictionary\n",
              markdown_table(pd.DataFrame([flat(row_b, child_columns)])),
              f"\nCount columns written by the tool, factor original/ours: "
              f"{row_b['count_columns_by_factor_original_over_ours']}"]
    rows_c = [{"parent": row["parent"], "parent instances (original)": row["parent_instances_in_original"],
               **flat(row, child_columns)} for row in results["C_step2"]]
    parts += ["\n## C. Step-2 extension (Single_Pattern_Extender)\n", markdown_table(pd.DataFrame(rows_c))]
    parts += ["\n## D. Interest values, original functions vs ours\n",
              markdown_table(pd.DataFrame(results["D_interest_values"]))]
    if "E_automatic_mode" in results:
        part = results["E_automatic_mode"]
        parts += ["\n## E. Original automatic mode, step by step\n",
                  f"AutoStepWise_PPD: {part['seconds_original_AutoStepWise_PPD']:.1f} s; our extension and "
                  f"scoring of the same steps: {part['seconds_ours_extension_and_scoring']:.2f} s; "
                  f"{part['n_train_cases']} training cases.\n",
                  markdown_table(pd.DataFrame([flat(row, list(row)) for row in part["steps"]]))]
    full_rows = [results[name] for name in ("F_full_log_core", "G_full_log_front") if name in results]
    if full_rows:
        columns = ["cores", "n_cases", "n_core_events"] + child_columns[:-2] + [
            "seconds_original_loop", "seconds_original_Pattern_extension_calls", "seconds_ours"]
        parts += ["\n## F/G. Step-1 extension on the full log (one shared dictionary per row)\n",
                  markdown_table(pd.DataFrame([flat(dict(row, cores=", ".join(row["cores"])), columns)
                                               for row in full_rows]))]
        for row in full_rows:
            parts.append(f"\nOriginal seconds per core: {row['seconds_original_per_core']}; "
                         f"isomorphism check: {row['isomorphism']}")
    (RESULT_DIR / "crosscheck_tables.md").write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--log", default="f1", choices=sorted(DATASETS))
    parser.add_argument("--skip-auto", action="store_true", help="skip part E (the original automatic mode)")
    parser.add_argument("--full-core", default=None, help="part F: core activity for a full-log pass, e.g. ac370000")
    parser.add_argument("--full-front", action="store_true",
                        help="part G: full-log pass for all activities on our step-0 Pareto front")
    args = parser.parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    inputs = OriginalInputs(args.log, N_CASES, SEED)
    coverage = Counter(activity for trace in inputs.traces for activity in set(trace))
    seeds = [activity for activity, _ in sorted(coverage.items(), key=lambda item: (-item[1], item[0]))[:N_SEEDS]]
    results = {
        "subset": {"log": args.log, "n_cases": len(inputs.case_ids), "seed": SEED,
                   "n_events": int(len(inputs.data)), "n_activities": int(inputs.data[ACTIVITY_COL].nunique()),
                   "max_gap": MAX_GAP, "delta_time": DELTA_TIME, "seed_activities": seeds},
        "chain_check": inputs.chain_report(),
    }
    say(f"subset {results['subset']}; chain check {results['chain_check']}")

    def save():
        with open(RESULT_DIR / "crosscheck.json", "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=1, default=str)
        write_tables(results)

    rows_a, dictionaries = part_a(inputs, seeds, rng)
    results["A_step1_fresh_dictionary"] = rows_a
    row_b, shared_dictionary, patient_b, _ = part_b(inputs, seeds)
    results["B_step1_shared_dictionary"] = row_b
    rows_c, children_c, patient_c = part_c(inputs, dictionaries)
    results["C_step2"] = rows_c
    results["D_interest_values"] = [
        part_d(inputs, shared_dictionary, patient_b, "step 1 (part B)"),
        part_d(inputs, children_c, patient_c, "step 2 (part C)"),
    ]
    save()
    if not args.skip_auto:
        results["E_automatic_mode"] = part_e(inputs)
        save()
    if args.full_core or args.full_front:
        full = OriginalInputs(args.log, None, SEED)
        say(f"full {args.log}: {len(full.case_ids)} cases, graphs in {full.graph_seconds:.1f} s")
        if args.full_core:
            results["F_full_log_core"] = part_f(full, [args.full_core], "F")
            save()
        if args.full_front:
            step0 = ImpressedChainSelector(max_gap=MAX_GAP, steps=0).fit(
                dict(zip(full.case_ids, full.traces)), dict(zip(full.case_ids, full.labels)), full.dist)
            front = [pattern.labels[0] for pattern in step0.patterns_.loc[step0.patterns_["on_front"], "pattern"]]
            results["G_full_log_front"] = part_f(full, front, "G")
            save()
    say("done")


if __name__ == "__main__":
    main()
