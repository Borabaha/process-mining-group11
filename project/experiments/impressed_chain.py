"""Chain-only re-implementation of IMPresseD's automatic mode (experiment C4, prototype).

IMPresseD = Vazifehdoostirani et al. (2023), "Interactive Multi-Interest Process
Pattern Discovery"; original code in ``external/InteractivePatternDetection``
(``IMIPD.py``, ``Auto_IMPID.py``, ``tools.py``).  This module re-implements the
part of that code the project needs, for traces WITHOUT concurrency, and
projects the discovered patterns to activity SETS for the 2024 pipeline.

How a chain pattern maps to an original graph pattern
-----------------------------------------------------
With ``delta_time < 0`` the original ``Trace_graph_generator`` (IMIPD.py:331-400)
turns a trace of ``L`` events into the path graph ``0 -> 1 -> ... -> L-1``: one
node per event position, node attribute ``value`` = activity, ``parallel`` =
False everywhere, every edge has ``eventually=False``.  Every pattern the
original code then builds is itself a directed path:

=========================  =================================================
original ``nx.DiGraph``    :class:`Pattern`
=========================  =================================================
nodes in path order        ``labels`` = their ``value`` attributes, in order
edge ``eventually=False``  ``'direct'``   (the two events are adjacent)
edge ``eventually=True``   ``'eventual'`` (1 to ``max_gap`` events in between)
``parallel``, ``color``    dropped (never used to compare patterns)
=========================  =================================================

The original recognises "the same pattern" with ``nx.is_isomorphic`` matching
node ``value`` and edge ``eventually`` (tools.py:85-87).  A directed path has
exactly one order-preserving mapping onto another directed path, so two such
graphs are isomorphic if and only if their label tuples and edge-type tuples
are equal.  Comparing :class:`Pattern` tuples is therefore the same test
(checked against the original in ``c4_crosscheck_original.py``).

Instances
---------
An instance is a tuple of event positions in one trace.  Consecutive nodes
joined by a direct edge sit at adjacent positions; nodes joined by an eventual
edge are 2 to ``max_gap + 1`` positions apart (the original window
``max(out) < node <= max(out) + Max_gap_between_events``, IMIPD.py:249-251,
where ``out`` is the direct successor).  A pattern with only direct edges is an
n-gram, so its instances are found by scanning the trace lists.

Deliberate differences from the original code (all measured in experiment C4)
-----------------------------------------------------------------------------
1. Every instance is counted ONCE.  The original counts some instances twice
   or four times (e.g. ``a -> a`` is found as "a followed by a" and as
   "a preceded by a").  The factor is the same for every case of a pattern, so
   coverage, information gain and case distance are unchanged.
2. A pattern reached from two parents is ONE pattern here.  The original keeps
   one dictionary per parent from step 2 on, so it scores the same graph
   several times under different IDs.
3. Interest values are computed on all cases given to :meth:`fit`.  The
   original scores on an 80 % training split but extends on the whole log
   (Auto_IMPID.py:17, :49-50).
4. ``distinct=False`` by default (original: paretoset's default ``True``) and
   an undefined case distance becomes ``undefined_cd`` instead of NaN.
5. The case distance matrix is an input (see ``case_distance.py``).
6. The projection to activity sets and the selection of ``k`` sets per length
   are new: they belong to the project, not to IMPresseD.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd
from paretoset import paretorank, paretoset
from sklearn.feature_selection import mutual_info_classif

from case_distance import pattern_case_distance

DIRECT = "direct"
EVENTUAL = "eventual"
# Interest functions and the direction in which each one is optimised.
SENSES = {"IG": "max", "coverage": "max", "CD": "min"}
SCORE_COLUMNS = ["IG", "coverage", "CD"]

# One instance = (index of the case, positions of the pattern's nodes in that trace).
Instance = tuple[int, tuple[int, ...]]


# --------------------------------------------------------------------------
# (a) Pattern representation
# --------------------------------------------------------------------------
class Pattern(NamedTuple):
    """A process pattern on chain traces.

    ``labels`` are the activities of the nodes in order; ``edges[i]`` is the
    relation between node ``i`` and node ``i + 1`` (``'direct'`` or
    ``'eventual'``), so ``len(edges) == len(labels) - 1``.
    """

    labels: tuple[str, ...]
    edges: tuple[str, ...] = ()

    @property
    def extendable(self) -> bool:
        """False for patterns with an eventual edge: the original never extends them."""
        return EVENTUAL not in self.edges

    @property
    def activity_set(self) -> frozenset[str]:
        """The distinct activities of the pattern (order and repetitions dropped)."""
        return frozenset(self.labels)

    def __str__(self) -> str:
        parts = [self.labels[0]]
        for edge, label in zip(self.edges, self.labels[1:]):
            parts.append("->" if edge == DIRECT else "~>")
            parts.append(label)
        return " ".join(parts)


def count_instances(pattern: Pattern, trace: Sequence[str], max_gap: int) -> int:
    """Number of instances of ``pattern`` in one trace, by a plain scan.

    Independent of the extension code below (used to verify it): ``ways[p]`` is
    the number of ways to place the nodes handled so far with the last one at
    position ``p``.
    """
    ways = [int(activity == pattern.labels[0]) for activity in trace]
    for edge, label in zip(pattern.edges, pattern.labels[1:]):
        steps = (1,) if edge == DIRECT else range(2, max_gap + 2)
        ways = [
            sum(ways[pos - step] for step in steps if pos - step >= 0) if activity == label else 0
            for pos, activity in enumerate(trace)
        ]
    return sum(ways)


# --------------------------------------------------------------------------
# (b) Extension
# --------------------------------------------------------------------------
def build_position_index(traces: Sequence[Sequence[str]]) -> dict[str, list[tuple[int, int]]]:
    """Activity -> list of (case index, position) of all its events."""
    index: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for case, trace in enumerate(traces):
        for position, activity in enumerate(trace):
            index[activity].append((case, position))
    return dict(index)


def find_instances(pattern: Pattern, traces: Sequence[Sequence[str]],
                   position_index: Mapping[str, list[tuple[int, int]]]) -> list[tuple[int, int]]:
    """All (case index, start position) at which a direct-only pattern occurs."""
    if not pattern.extendable:
        raise ValueError(f"only patterns without an eventual edge are located: {pattern}")
    labels = list(pattern.labels)
    return [
        (case, start)
        for case, start in position_index.get(labels[0], [])
        if list(traces[case][start:start + len(labels)]) == labels
    ]


def extend_pattern(pattern: Pattern, traces: Sequence[Sequence[str]],
                   position_index: Mapping[str, list[tuple[int, int]]],
                   max_gap: int) -> dict[Pattern, set[Instance]]:
    """Extend one pattern at every one of its instances.

    Replicates, for chains, ``Pattern_extension`` (IMIPD.py:168-328) when the
    pattern is a single activity and ``Single_Pattern_Extender``
    (IMIPD.py:612-749) otherwise.  With the pattern at positions
    ``start..end`` of a trace the children are, in the original's order:

    * direct preceding:      ``trace[start-1] -> pattern``
    * direct following:      ``pattern -> trace[end+1]``
    * eventually following:  ``pattern ~> trace[j]``, ``end+2 <= j <= end+1+max_gap``
    * eventually preceding:  ``trace[j] ~> pattern``, ``start-1-max_gap <= j <= start-2``
    * direct context (single activities only, IMIPD.py:292-316):
      ``trace[start-1] -> activity -> trace[start+1]``

    The concurrent rule yields nothing on chains.

    Returns
    -------
    dict
        Child pattern -> set of its instances found here.  Because instances
        are kept in a set, an instance reached twice is counted once.
    """
    labels, edges = pattern.labels, pattern.edges
    children: dict[Pattern, set[Instance]] = defaultdict(set)
    for case, start in find_instances(pattern, traces, position_index):
        trace = traces[case]
        end = start + len(labels) - 1
        inside = tuple(range(start, end + 1))
        has_predecessor, has_successor = start > 0, end < len(trace) - 1

        if has_predecessor:
            child = Pattern((trace[start - 1],) + labels, (DIRECT,) + edges)
            children[child].add((case, (start - 1,) + inside))
        if has_successor:
            child = Pattern(labels + (trace[end + 1],), edges + (DIRECT,))
            children[child].add((case, inside + (end + 1,)))
        for far in range(end + 2, min(len(trace) - 1, end + 1 + max_gap) + 1):
            child = Pattern(labels + (trace[far],), edges + (EVENTUAL,))
            children[child].add((case, inside + (far,)))
        for far in range(max(0, start - 1 - max_gap), start - 1):
            child = Pattern((trace[far],) + labels, (EVENTUAL,) + edges)
            children[child].add((case, (far,) + inside))
        if len(labels) == 1 and has_predecessor and has_successor:
            child = Pattern((trace[start - 1], labels[0], trace[start + 1]), (DIRECT, DIRECT))
            children[child].add((case, (start - 1, start, start + 1)))
    return dict(children)


def extend_patterns(parents: Iterable[Pattern], traces: Sequence[Sequence[str]],
                    position_index: Mapping[str, list[tuple[int, int]]],
                    max_gap: int) -> tuple[dict[Pattern, set[Instance]], dict[Pattern, list[Pattern]]]:
    """Extend several parents into ONE pool of children (one extension step).

    Parents with an eventual edge are skipped, as in Auto_IMPID.py:108-110.

    Returns ``(instances, parents_of)``: child -> set of instances (merged over
    all parents that reach it) and child -> list of those parents.
    """
    instances: dict[Pattern, set[Instance]] = {}
    parents_of: dict[Pattern, list[Pattern]] = defaultdict(list)
    for parent in parents:
        if not parent.extendable:
            continue
        for child, found in extend_pattern(parent, traces, position_index, max_gap).items():
            instances.setdefault(child, set()).update(found)
            parents_of[child].append(parent)
    return instances, dict(parents_of)


def count_matrix(instances: Mapping[Pattern, set[Instance]], n_cases: int) -> np.ndarray:
    """Per-case instance counts: one row per case, one column per pattern (dict order)."""
    counts = np.zeros((n_cases, len(instances)), dtype=np.int64)
    for column, found in enumerate(instances.values()):
        for case, _positions in found:
            counts[case, column] += 1
    return counts


# --------------------------------------------------------------------------
# (c) Interest functions
# --------------------------------------------------------------------------
def score_patterns(counts: np.ndarray, labels: Sequence[int], dist_matrix: np.ndarray | None = None,
                   undefined_cd: float = 1.0) -> pd.DataFrame:
    """The three interest values of every pattern (column of ``counts``).

    * ``IG`` - outcome interest for a binary outcome, exactly as IMIPD.py:68:
      ``mutual_info_classif(counts, labels, discrete_features=True)``, i.e. the
      information gain (in nats) of the per-case instance COUNT about the label.
      ``random_state`` is fixed although it is not used for discrete features.
    * ``coverage`` - share of cases with at least one instance (IMIPD.py:98-99).
    * ``CD`` - mean distance between the cases with and the cases without the
      pattern (IMIPD.py:37-60) via ``case_distance.pattern_case_distance``;
      ``undefined_cd`` when the pattern is in all or in no case.  NaN when no
      ``dist_matrix`` is given.

    ``dist_matrix`` rows/columns must be in the row order of ``counts``.
    """
    n_cases, n_patterns = counts.shape
    if n_patterns == 0:
        return pd.DataFrame(columns=SCORE_COLUMNS, dtype=float)
    present = counts > 0
    info_gain = mutual_info_classif(counts, np.asarray(labels), discrete_features=True, random_state=0)
    if dist_matrix is None:
        case_distance = np.full(n_patterns, np.nan)
    else:
        case_distance = np.array([
            pattern_case_distance(dist_matrix, present[:, column], undefined_value=undefined_cd)
            for column in range(n_patterns)
        ])
    return pd.DataFrame({"IG": info_gain, "coverage": present.sum(axis=0) / n_cases, "CD": case_distance})


def _objective_table(scores: pd.DataFrame, objectives: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """Objective values as an array plus their senses; refuses NaN (see experiment C11)."""
    values = scores[list(objectives)].to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError("an objective is NaN; paretoset must never see NaN (experiment C11)")
    return values, [SENSES[name] for name in objectives]


def pareto_front(scores: pd.DataFrame, objectives: Sequence[str], distinct: bool = False) -> np.ndarray:
    """Boolean mask of the non-dominated rows of ``scores``."""
    if len(scores) == 0:
        return np.zeros(0, dtype=bool)
    values, senses = _objective_table(scores, objectives)
    return np.asarray(paretoset(values, sense=senses, distinct=distinct), dtype=bool)


def pareto_layers(scores: pd.DataFrame, objectives: Sequence[str], distinct: bool = False) -> np.ndarray:
    """Non-dominated sorting: 1 for the Pareto front, 2 for the front of the rest, ..."""
    if len(scores) == 0:
        return np.zeros(0, dtype=int)
    values, senses = _objective_table(scores, objectives)
    return np.asarray(paretorank(values, sense=senses, distinct=distinct), dtype=int)


def tie_break_key(row: Mapping, name) -> tuple:
    """Sort key of the documented tie-break (smaller = better).

    Higher information gain first, then higher coverage, then lower case
    distance, then ``name`` (so the order never depends on dict or hash order).
    """
    case_distance = 0.0 if pd.isna(row["CD"]) else row["CD"]
    return (-row["IG"], -row["coverage"], case_distance, name)


# --------------------------------------------------------------------------
# (d) + (e) Automatic mode and projection to activity sets
# --------------------------------------------------------------------------
class ImpressedChainSelector:
    """IMPresseD automatic mode on chain traces, projected to activity sets.

    Follows ``Auto_IMPID.AutoStepWise_PPD``:

    * step 0 - every activity is a pattern; score them; take the Pareto front
      (Auto_IMPID.py:28-36);
    * step 1 - extend every front activity into one shared pool of children,
      score the pool, take ONE joint front (Auto_IMPID.py:45-85);
    * step s >= 2 - extend every pattern of the previous front that has no
      eventual edge, score all children of the step, take one front
      (Auto_IMPID.py:103-134).

    The deliberate differences are listed in the module docstring.

    Parameters
    ----------
    max_gap : int
        ``Max_gap_between_events`` of the original: an eventual edge skips 1 to
        ``max_gap`` events.
    steps : int
        Number of extension steps after step 0.  NOTE: the original runs
        ``for ext in range(1, Max_extension_step)`` after a first extension that
        always happens, so ``steps`` equals the original ``Max_extension_step``
        for values >= 1 (``steps=2`` -> steps 0, 1, 2).
    objectives : sequence of str
        Interest functions used for the Pareto fronts, a subset of
        ``('IG', 'coverage', 'CD')``; IG and coverage are maximised, CD minimised.
    distinct : bool
        Passed to ``paretoset``/``paretorank``.  False keeps all patterns with
        identical interest values (experiment C11).
    k : int
        Number of activity sets selected per length.
    lengths : sequence of int
        Set sizes (number of distinct activities) that are kept.
    undefined_cd : float
        Case distance given to a pattern that occurs in ALL cases, where the
        distance is undefined (1.0 = the worst possible value).
    candidate_pool : {'evaluated', 'front'}
        Which patterns may supply activity sets: every pattern that was scored
        in some step, or only the patterns on a step's Pareto front.

    Attributes (after :meth:`fit`)
    ------------------------------
    patterns_ : pandas.DataFrame
        One row per evaluated pattern: ``pattern`` (:class:`Pattern`), ``name``,
        ``step`` (first step that scored it), ``n_nodes``, ``IG``, ``coverage``,
        ``CD``, ``n_instances``, ``n_cases``, ``on_front`` (on the front of some
        step), ``parents`` (names of the parents that reach it).
    step_tables_ : list of pandas.DataFrame
        The same columns per step, before merging: the candidates of step ``s``
        and whether each is on that step's front.
    step_summary_ : pandas.DataFrame
        Per step: number of scored candidates, front size, extendable front patterns.
    counts_ : dict
        Pattern -> per-case instance counts (numpy array, order of ``case_ids_``).
    """

    def __init__(self, max_gap: int = 3, steps: int = 2,
                 objectives: Sequence[str] = ("IG", "coverage", "CD"), distinct: bool = False,
                 k: int = 10, lengths: Sequence[int] = (1, 2, 3), undefined_cd: float = 1.0,
                 candidate_pool: str = "evaluated"):
        unknown = set(objectives) - set(SENSES)
        if unknown or not objectives:
            raise ValueError(f"objectives must be a non-empty subset of {sorted(SENSES)}")
        if candidate_pool not in ("evaluated", "front"):
            raise ValueError("candidate_pool must be 'evaluated' or 'front'")
        self.max_gap = max_gap
        self.steps = steps
        self.objectives = tuple(objectives)
        self.distinct = distinct
        self.k = k
        self.lengths = tuple(lengths)
        self.undefined_cd = undefined_cd
        self.candidate_pool = candidate_pool

    # ---------------------------------------------------------------- mining
    def fit(self, traces: Mapping[str, Sequence[str]], labels: Mapping[str, int],
            dist_matrix: np.ndarray | None = None) -> "ImpressedChainSelector":
        """Mine and score the patterns.

        Parameters
        ----------
        traces : mapping case id -> list of activity names (in trace order)
        labels : mapping case id -> 0/1 outcome
        dist_matrix : square array or None
            Pairwise case distances in the order of ``traces`` (from
            ``case_distance.pairwise_case_distance``).  Required when 'CD' is
            an objective.
        """
        if "CD" in self.objectives and dist_matrix is None:
            raise ValueError("objective 'CD' needs dist_matrix")
        self.case_ids_ = list(traces)
        trace_list = [list(traces[case]) for case in self.case_ids_]
        outcome = np.array([labels[case] for case in self.case_ids_])
        position_index = build_position_index(trace_list)

        # Step 0: single activities; an instance is one event.
        instances: dict[Pattern, set[Instance]] = {
            Pattern((activity,)): {(case, (position,)) for case, position in events}
            for activity, events in sorted(position_index.items())
        }
        parents_of: dict[Pattern, list[Pattern]] = {}
        self.counts_: dict[Pattern, np.ndarray] = {}
        step_tables, summary = [], []
        for step in range(self.steps + 1):
            counts = count_matrix(instances, len(trace_list))
            table = score_patterns(counts, outcome, dist_matrix, self.undefined_cd)
            table.insert(0, "pattern", list(instances))
            table.insert(1, "name", [str(pattern) for pattern in instances])
            table.insert(2, "step", step)
            table.insert(3, "n_nodes", [len(pattern.labels) for pattern in instances])
            table["n_instances"] = counts.sum(axis=0)
            table["n_cases"] = (counts > 0).sum(axis=0)
            table["on_front"] = pareto_front(table, self.objectives, self.distinct)
            table["parents"] = [
                "; ".join(str(parent) for parent in parents_of.get(pattern, [])) for pattern in instances
            ]
            for column, pattern in enumerate(instances):
                self.counts_.setdefault(pattern, counts[:, column])
            step_tables.append(table)

            front = list(table.loc[table["on_front"], "pattern"])
            summary.append({
                "step": step, "n_candidates": len(table), "front_size": len(front),
                "front_extendable": sum(pattern.extendable for pattern in front),
            })
            if step == self.steps:
                break
            instances, parents_of = extend_patterns(front, trace_list, position_index, self.max_gap)
            if not instances:
                break

        self.step_tables_ = step_tables
        self.step_summary_ = pd.DataFrame(summary)
        self.patterns_ = self._merge_steps(step_tables)
        return self

    @staticmethod
    def _merge_steps(step_tables: list[pd.DataFrame]) -> pd.DataFrame:
        """One row per pattern: the first step that scored it; ``on_front`` in any step.

        A pattern can be scored in two steps (``x -> a -> b`` is the context of
        ``a`` in step 1 and a child of ``a -> b`` in step 2); its counts and
        interest values are the same both times.
        """
        stacked = pd.concat(step_tables, ignore_index=True)
        on_any_front = stacked.groupby("name", sort=False)["on_front"].any()  # name <-> pattern is 1:1
        merged = stacked.drop_duplicates("name", keep="first").reset_index(drop=True)
        merged["on_front"] = merged["name"].map(on_any_front).to_numpy()
        return merged

    # ------------------------------------------------------------ projection
    def full_table(self) -> pd.DataFrame:
        """Every candidate activity set of an allowed length, ranked per length.

        A pattern is projected to the set of its distinct activities; the
        length is the size of that set (``a -> a`` has length 1).  Each set is
        represented by its best pattern: non-dominated among the patterns of
        that set, tie-break of :func:`tie_break_key`.  Per length the sets are
        ranked by non-dominated sorting layer, then by the same tie-break.

        Columns: ``itemset`` (tuple of sorted activity names), ``size``,
        ``rank`` (1-based per size), ``layer`` (1 = Pareto front of the sets of
        that size), ``IG``, ``coverage``, ``CD``, ``source_pattern`` (name),
        ``pattern`` (:class:`Pattern`), ``step``, ``n_patterns`` (candidate
        patterns that project to this set).
        """
        pool = self.patterns_
        if self.candidate_pool == "front":
            pool = pool[pool["on_front"]]
        rows_of_set: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for row, pattern in zip(pool.index, pool["pattern"]):
            if len(pattern.activity_set) in self.lengths:
                rows_of_set[tuple(sorted(pattern.activity_set))].append(row)

        sets_of_size: dict[int, list[dict]] = defaultdict(list)
        for itemset, rows in rows_of_set.items():
            group = pool.loc[rows]
            non_dominated = group[pareto_front(group, self.objectives, distinct=False)]
            best = min(non_dominated.to_dict("records"), key=lambda row: tie_break_key(row, row["name"]))
            sets_of_size[len(itemset)].append({
                "itemset": itemset, "size": len(itemset), "IG": best["IG"], "coverage": best["coverage"],
                "CD": best["CD"], "source_pattern": best["name"], "pattern": best["pattern"],
                "step": int(best["step"]), "n_patterns": len(group),
            })

        ranked = []
        for size in sorted(sets_of_size):
            sets = sets_of_size[size]
            layers = pareto_layers(pd.DataFrame(sets), self.objectives, self.distinct)
            for row, layer in zip(sets, layers):
                row["layer"] = int(layer)
            sets.sort(key=lambda row: (row["layer"],) + tie_break_key(row, row["itemset"]))
            for rank, row in enumerate(sets, start=1):
                row["rank"] = rank
            ranked.extend(sets)
        columns = ["itemset", "size", "rank", "layer", "IG", "coverage", "CD",
                   "source_pattern", "pattern", "step", "n_patterns"]
        return pd.DataFrame(ranked, columns=columns)

    def select_table(self) -> pd.DataFrame:
        """The ``k`` best rows of :meth:`full_table` for every size."""
        table = self.full_table()
        return table[table["rank"] <= self.k].reset_index(drop=True)

    def select(self) -> dict[int, list[list[str]]]:
        """Return ``{size: [sorted activity names, ...]}`` - the format of ``AprioriSelector.select``.

        A size with fewer than ``k`` candidate sets returns the ones that exist
        (possibly an empty list).
        """
        selected = self.select_table()
        return {
            int(size): [list(itemset) for itemset in selected.loc[selected["size"] == size, "itemset"]]
            for size in self.lengths
        }

    def front_sizes(self) -> dict[int, int]:
        """Size of the first Pareto front among the candidate sets of every length."""
        table = self.full_table()
        return {int(size): int(((table["size"] == size) & (table["layer"] == 1)).sum()) for size in self.lengths}
