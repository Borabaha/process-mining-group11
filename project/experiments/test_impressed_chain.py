"""Unit tests for impressed_chain.py on toy traces with hand-computed expectations.

Run from the project root:
    python -m unittest experiments/test_impressed_chain.py -v
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from impressed_chain import (  # noqa: E402
    DIRECT,
    EVENTUAL,
    ImpressedChainSelector,
    Pattern,
    build_position_index,
    count_instances,
    count_matrix,
    extend_pattern,
    extend_patterns,
    find_instances,
    pareto_front,
    pareto_layers,
    score_patterns,
)


def pat(text: str) -> Pattern:
    """Build a pattern from its printed form, e.g. ``pat('a -> b ~> c')``."""
    tokens = text.split()
    edges = tuple(DIRECT if arrow == "->" else EVENTUAL for arrow in tokens[1::2])
    return Pattern(tuple(tokens[0::2]), edges)


def extend(pattern: Pattern, traces, max_gap):
    """``extend_pattern`` as {printed child: sorted list of instances}."""
    children = extend_pattern(pattern, traces, build_position_index(traces), max_gap)
    return {str(child): sorted(found) for child, found in children.items()}


class PatternTest(unittest.TestCase):
    def test_printing_and_parsing(self):
        pattern = Pattern(("a", "b", "c"), (DIRECT, EVENTUAL))
        self.assertEqual(str(pattern), "a -> b ~> c")
        self.assertEqual(pat("a -> b ~> c"), pattern)
        self.assertEqual(str(Pattern(("a",))), "a")

    def test_only_patterns_without_eventual_edge_are_extendable(self):
        self.assertTrue(pat("a").extendable)
        self.assertTrue(pat("a -> b -> c").extendable)
        self.assertFalse(pat("a ~> b").extendable)

    def test_activity_set_drops_order_and_repetitions(self):
        self.assertEqual(pat("a -> a").activity_set, frozenset({"a"}))
        self.assertEqual(pat("b -> a -> b").activity_set, frozenset({"a", "b"}))
        self.assertEqual(pat("a -> b").activity_set, pat("b ~> a").activity_set)


class CountInstancesTest(unittest.TestCase):
    """The plain scan: direct = adjacent, eventual = 2 .. max_gap + 1 positions apart."""

    def test_direct_patterns_are_ngrams_with_overlap(self):
        self.assertEqual(count_instances(pat("a -> a"), list("aaa"), 3), 2)      # (0,1) and (1,2)
        self.assertEqual(count_instances(pat("a -> a -> a"), list("aaa"), 3), 1)
        self.assertEqual(count_instances(pat("a -> b"), list("abab"), 3), 2)
        self.assertEqual(count_instances(pat("a -> b"), list("acb"), 3), 0)      # b only eventually follows a

    def test_eventual_edge_excludes_the_direct_neighbour(self):
        self.assertEqual(count_instances(pat("a ~> b"), list("ab"), 3), 0)
        self.assertEqual(count_instances(pat("a ~> b"), list("acb"), 3), 1)
        self.assertEqual(count_instances(pat("a ~> a"), list("aaa"), 3), 1)      # only (0,2)
        self.assertEqual(count_instances(pat("a ~> b"), list("abab"), 3), 1)     # only (0,3)

    def test_eventual_edge_is_bounded_by_max_gap(self):
        trace = list("axxxxb")  # b is 5 positions after a: 4 events in between
        self.assertEqual(count_instances(pat("a ~> b"), trace, 3), 0)
        self.assertEqual(count_instances(pat("a ~> b"), trace, 4), 1)

    def test_mixed_pattern(self):
        # a at 0, b at 1, then c at 3 (1 event in between) and c at 4 (2 in between)
        self.assertEqual(count_instances(pat("a -> b ~> c"), list("abxcc"), 3), 2)
        self.assertEqual(count_instances(pat("x ~> a -> b"), list("xyab"), 3), 1)
        self.assertEqual(count_instances(pat("x ~> a -> b"), list("xab"), 3), 0)  # x directly precedes a


class ExtensionTest(unittest.TestCase):
    def test_guide_example_single_activity(self):
        """Guide 4.1: extending b in <a,b,c,d,e> with gap 2 gives a->b, b->c, b~>d, b~>e, a->b->c."""
        children = extend(pat("b"), [list("abcde")], max_gap=2)
        self.assertEqual(children, {
            "a -> b": [(0, (0, 1))],
            "b -> c": [(0, (1, 2))],
            "b ~> d": [(0, (1, 3))],
            "b ~> e": [(0, (1, 4))],
            "a -> b -> c": [(0, (0, 1, 2))],
        })

    def test_eventually_preceding_window(self):
        # c at position 3: direct predecessor b (2); eventual predecessors a (1) and x (0) with gap 3,
        # only a with gap 1.
        self.assertEqual(set(extend(pat("c"), [list("xabc")], max_gap=3)), {"b -> c", "a ~> c", "x ~> c"})
        self.assertEqual(set(extend(pat("c"), [list("xabc")], max_gap=1)), {"b -> c", "a ~> c"})

    def test_repeated_activity_instance_is_counted_once(self):
        """In <x,a,a,y> the instance a->a is reached from both a's; it must appear once."""
        children = extend(pat("a"), [list("xaay")], max_gap=3)
        self.assertEqual(children, {
            "x -> a": [(0, (0, 1))],
            "a -> a": [(0, (1, 2))],
            "a ~> y": [(0, (1, 3))],
            "x -> a -> a": [(0, (0, 1, 2))],
            "a -> y": [(0, (2, 3))],
            "x ~> a": [(0, (0, 2))],
            "a -> a -> y": [(0, (1, 2, 3))],
        })

    def test_first_and_last_event_have_no_context(self):
        self.assertEqual(set(extend(pat("a"), [list("ab")], max_gap=3)), {"a -> b"})
        self.assertEqual(set(extend(pat("b"), [list("ab")], max_gap=3)), {"a -> b"})
        self.assertEqual(extend(pat("a"), [list("a")], max_gap=3), {})

    def test_step2_extension_has_no_context(self):
        """Parent a->b at positions 1-2 of <x,a,b,y,z,w,v>, gap 2."""
        children = extend(pat("a -> b"), [list("xabyzwv")], max_gap=2)
        self.assertEqual(children, {
            "x -> a -> b": [(0, (0, 1, 2))],
            "a -> b -> y": [(0, (1, 2, 3))],
            "a -> b ~> z": [(0, (1, 2, 4))],
            "a -> b ~> w": [(0, (1, 2, 5))],
        })

    def test_find_instances_overlapping(self):
        traces = [list("aaab"), list("ba")]
        index = build_position_index(traces)
        self.assertEqual(find_instances(pat("a -> a"), traces, index), [(0, 0), (0, 1)])
        self.assertEqual(find_instances(pat("a"), traces, index), [(0, 0), (0, 1), (0, 2), (1, 1)])
        with self.assertRaises(ValueError):
            find_instances(pat("a ~> b"), traces, index)

    def test_shared_pool_merges_parents_and_skips_eventual_parents(self):
        traces = [list("abc"), list("ab")]
        index = build_position_index(traces)
        instances, parents_of = extend_patterns([pat("a"), pat("b"), pat("a ~> c")], traces, index, max_gap=3)
        # a->b is found from core a and from core b: one pattern, each instance once.
        self.assertEqual(instances[pat("a -> b")], {(0, (0, 1)), (1, (0, 1))})
        self.assertEqual(parents_of[pat("a -> b")], [pat("a"), pat("b")])
        self.assertEqual(set(map(str, instances)), {"a -> b", "a ~> c", "b -> c", "a -> b -> c"})
        np.testing.assert_array_equal(count_matrix({pat("a -> b"): instances[pat("a -> b")]}, 2), [[1], [1]])


class InterestTest(unittest.TestCase):
    DIST = np.array([[0.0, 0.2, 0.4, 0.6],
                     [0.2, 0.0, 0.8, 1.0],
                     [0.4, 0.8, 0.0, 0.1],
                     [0.6, 1.0, 0.1, 0.0]])

    def test_guide_tiny_example(self):
        """Counts (2,1,0,0), labels (1,1,0,0): coverage 0.5, information gain ln 2 = 0.693 nats."""
        scores = score_patterns(np.array([[2], [1], [0], [0]]), [1, 1, 0, 0], self.DIST)
        self.assertAlmostEqual(scores.loc[0, "coverage"], 0.5)
        self.assertAlmostEqual(scores.loc[0, "IG"], math.log(2))
        # in = cases 0,1; out = cases 2,3: mean of 0.4, 0.6, 0.8, 1.0
        self.assertAlmostEqual(scores.loc[0, "CD"], 0.7)

    def test_information_gain_uses_the_count_not_the_presence(self):
        counts = np.array([[2, 1], [1, 1], [2, 1], [1, 1]])  # both patterns are in every case
        scores = score_patterns(counts, [1, 0, 1, 0], self.DIST)
        self.assertAlmostEqual(scores.loc[0, "IG"], math.log(2))  # count 2 <=> label 1
        self.assertAlmostEqual(scores.loc[1, "IG"], 0.0)
        self.assertEqual(list(scores["coverage"]), [1.0, 1.0])
        self.assertEqual(list(scores["CD"]), [1.0, 1.0])  # in all cases: undefined -> undefined_cd

    def test_counting_every_instance_twice_changes_nothing(self):
        counts = np.array([[3], [1], [0], [1], [2]])
        labels = [1, 0, 0, 1, 1]
        dist = np.arange(25, dtype=float).reshape(5, 5)
        dist = (dist + dist.T) / 100
        pd.testing.assert_frame_equal(score_patterns(counts, labels, dist), score_patterns(2 * counts, labels, dist))

    def test_no_distance_matrix_gives_nan(self):
        scores = score_patterns(np.array([[1], [0]]), [1, 0])
        self.assertTrue(math.isnan(scores.loc[0, "CD"]))

    def test_no_patterns(self):
        self.assertEqual(len(score_patterns(np.zeros((3, 0), dtype=int), [1, 0, 1])), 0)


class ParetoTest(unittest.TestCase):
    SCORES = pd.DataFrame({
        "IG": [0.5, 0.5, 0.6, 0.4],
        "coverage": [0.5, 0.5, 0.4, 0.4],
        "CD": [0.5, 0.5, 0.5, 0.6],
    })  # rows 0 and 1 are identical; row 3 is dominated by row 0

    def test_front_keeps_ties_by_default(self):
        objectives = ["IG", "coverage", "CD"]
        self.assertEqual(list(pareto_front(self.SCORES, objectives)), [True, True, True, False])
        self.assertEqual(list(pareto_front(self.SCORES, objectives, distinct=True)), [True, False, True, False])
        self.assertEqual(list(pareto_layers(self.SCORES, objectives)), [1, 1, 1, 2])

    def test_objective_subset(self):
        # coverage only: rows 0 and 1 are best
        self.assertEqual(list(pareto_front(self.SCORES, ["coverage"])), [True, True, False, False])
        # case distance is minimised: rows 0, 1, 2 tie at 0.5
        self.assertEqual(list(pareto_front(self.SCORES, ["CD"])), [True, True, True, False])

    def test_nan_is_refused(self):
        scores = self.SCORES.copy()
        scores.loc[0, "CD"] = np.nan
        with self.assertRaises(ValueError):
            pareto_front(scores, ["IG", "coverage", "CD"])
        self.assertEqual(list(pareto_front(scores, ["IG", "coverage"])), [True, True, True, False])


class SelectorToyLogTest(unittest.TestCase):
    """Four cases, coverage as the only objective, everything computed by hand.

    c1 = <a,b,c> (1), c2 = <a,b,d> (1), c3 = <b,a> (0), c4 = <c> (0).

    Step 0: coverage a 3/4, b 3/4, c 2/4, d 1/4          -> front {a, b}.
    Step 1: children of a and b: a->b (2/4); a~>c, a~>d, b->a, b->c, b->d,
            a->b->c, a->b->d (1/4 each)                   -> front {a->b}.
    Step 2: children of a->b: a->b->c, a->b->d (1/4 each) -> front = both.
    """

    TRACES = {"c1": list("abc"), "c2": list("abd"), "c3": list("ba"), "c4": list("c")}
    LABELS = {"c1": 1, "c2": 1, "c3": 0, "c4": 0}

    def setUp(self):
        self.selector = ImpressedChainSelector(objectives=("coverage",), k=2).fit(self.TRACES, self.LABELS)

    def test_steps(self):
        summary = self.selector.step_summary_
        self.assertEqual(list(summary["n_candidates"]), [4, 8, 2])
        self.assertEqual(list(summary["front_size"]), [2, 1, 2])
        step1 = self.selector.step_tables_[1]
        self.assertEqual(set(step1["name"]), {"a -> b", "a ~> c", "a ~> d", "b -> a", "b -> c", "b -> d",
                                              "a -> b -> c", "a -> b -> d"})
        self.assertEqual(list(step1.loc[step1["on_front"], "name"]), ["a -> b"])

    def test_every_pattern_is_kept_once(self):
        patterns = self.selector.patterns_.set_index("name")
        self.assertEqual(len(patterns), 12)  # 4 + 8 + 2, minus the two contexts scored again in step 2
        self.assertEqual(patterns.loc["a -> b -> c", "step"], 1)
        self.assertTrue(patterns.loc["a -> b -> c", "on_front"])  # on the front of step 2
        self.assertFalse(patterns.loc["b -> a", "on_front"])
        self.assertEqual(patterns.loc["a -> b", "parents"], "a; b")
        self.assertEqual(patterns.loc["a -> b", "n_instances"], 2)

    def test_interest_values(self):
        patterns = self.selector.patterns_.set_index("name")
        # a: counts (1,1,1,0) vs labels (1,1,0,0):
        # 0.5*ln(4/3) + 0.25*ln(2/3) + 0.25*ln(2) = 0.215762
        self.assertAlmostEqual(patterns.loc["a", "IG"], 0.215762, places=5)
        self.assertAlmostEqual(patterns.loc["a -> b", "IG"], math.log(2))
        self.assertEqual(patterns.loc["a", "coverage"], 0.75)
        self.assertEqual(patterns.loc["a -> b", "coverage"], 0.5)

    def test_selection(self):
        # length 2: {a,b} is alone on the first front (a->b beats b->a inside the set); the four
        # sets with coverage 1/4 have the same information gain, so the name decides: (a,c).
        self.assertEqual(self.selector.select(), {1: [["a"], ["b"]], 2: [["a", "b"], ["a", "c"]],
                                                  3: [["a", "b", "c"], ["a", "b", "d"]]})
        self.assertEqual(self.selector.front_sizes(), {1: 2, 2: 1, 3: 2})
        table = self.selector.full_table()
        row = table[table["itemset"] == ("a", "b")].iloc[0]
        self.assertEqual((row["source_pattern"], row["n_patterns"], row["layer"], row["rank"]), ("a -> b", 2, 1, 1))
        self.assertEqual(list(table.loc[table["size"] == 2, "layer"]), [1, 2, 2, 2, 2])

    def test_front_patterns_only(self):
        selector = ImpressedChainSelector(objectives=("coverage",), k=2, candidate_pool="front")
        selector.fit(self.TRACES, self.LABELS)
        self.assertEqual(selector.select(), {1: [["a"], ["b"]], 2: [["a", "b"]],
                                             3: [["a", "b", "c"], ["a", "b", "d"]]})

    def test_one_step_only(self):
        selector = ImpressedChainSelector(objectives=("coverage",), steps=1).fit(self.TRACES, self.LABELS)
        self.assertEqual(list(selector.step_summary_["step"]), [0, 1])

    def test_case_distance_objective(self):
        dist = np.array([[0.0, 0.1, 0.5, 0.9],
                         [0.1, 0.0, 0.6, 0.7],
                         [0.5, 0.6, 0.0, 0.2],
                         [0.9, 0.7, 0.2, 0.0]])
        selector = ImpressedChainSelector().fit(self.TRACES, self.LABELS, dist)
        patterns = selector.patterns_.set_index("name")
        self.assertAlmostEqual(patterns.loc["a", "CD"], (0.9 + 0.7 + 0.2) / 3)       # c1,c2,c3 vs c4
        self.assertAlmostEqual(patterns.loc["a -> b", "CD"], (0.5 + 0.9 + 0.6 + 0.7) / 4)  # c1,c2 vs c3,c4
        with self.assertRaises(ValueError):
            ImpressedChainSelector().fit(self.TRACES, self.LABELS)  # CD is an objective: needs the matrix


class ProjectionTest(unittest.TestCase):
    """Projection and selection on a hand-made pattern table (no mining)."""

    ROWS = [  # printed pattern, IG, coverage, CD
        ("a -> b", 0.30, 0.5, 0.4),            # {a,b}: representative (highest IG of the non-dominated)
        ("b ~> a", 0.20, 0.6, 0.4),            # {a,b}: non-dominated too (better coverage)
        ("a -> b -> a", 0.10, 0.4, 0.5),       # {a,b}: dominated by a -> b
        ("a -> a", 0.05, 0.2, 0.3),            # {a}: length 1 although it has 2 nodes
        ("a", 0.04, 0.7, 0.6),                 # {a}: non-dominated, lower IG
        ("a -> b -> c -> d", 0.90, 0.9, 0.1),  # 4 distinct activities: dropped
        ("a -> b -> c -> a", 0.02, 0.1, 0.5),  # 4 nodes but 3 distinct activities: kept
        ("c -> d", 0.30, 0.5, 0.4),            # {c,d}: exactly the values of {a,b}
        ("b -> c", 0.10, 0.1, 0.9),            # {b,c}: dominated -> second layer
    ]

    def selector(self, **kwargs):
        selector = ImpressedChainSelector(k=2, **kwargs)
        selector.patterns_ = pd.DataFrame({
            "pattern": pd.Series([pat(row[0]) for row in self.ROWS], dtype=object),
            "name": [row[0] for row in self.ROWS],
            "step": 1,
            "IG": [row[1] for row in self.ROWS],
            "coverage": [row[2] for row in self.ROWS],
            "CD": [row[3] for row in self.ROWS],
            "on_front": True,
        })
        return selector

    def test_full_table(self):
        table = self.selector().full_table()
        self.assertEqual(list(table["itemset"]), [("a",), ("a", "b"), ("c", "d"), ("b", "c"), ("a", "b", "c")])
        self.assertEqual(list(table["rank"]), [1, 1, 2, 3, 1])
        self.assertEqual(list(table["layer"]), [1, 1, 1, 2, 1])
        self.assertEqual(list(table["source_pattern"]),
                         ["a -> a", "a -> b", "c -> d", "b -> c", "a -> b -> c -> a"])
        self.assertEqual(list(table["n_patterns"]), [2, 3, 1, 1, 1])
        self.assertEqual(list(table.loc[table["itemset"] == ("a", "b"), "IG"]), [0.30])

    def test_select_and_front_sizes(self):
        selector = self.selector()
        self.assertEqual(selector.select(), {1: [["a"]], 2: [["a", "b"], ["c", "d"]], 3: [["a", "b", "c"]]})
        self.assertEqual(selector.front_sizes(), {1: 1, 2: 2, 3: 1})

    def test_distinct_true_pushes_a_tied_set_out_of_the_first_front(self):
        self.assertEqual(self.selector(distinct=True).front_sizes()[2], 1)

    def test_lengths_parameter(self):
        self.assertEqual(self.selector(lengths=(2,)).select(), {2: [["a", "b"], ["c", "d"]]})


class ConsistencyTest(unittest.TestCase):
    def test_extension_counts_equal_the_plain_scan_on_random_traces(self):
        rng = np.random.default_rng(0)
        traces = {f"case{i}": list(rng.choice(list("abcd"), size=rng.integers(1, 9))) for i in range(40)}
        labels = {case: int(rng.integers(0, 2)) for case in traces}
        selector = ImpressedChainSelector(max_gap=2, objectives=("IG", "coverage")).fit(traces, labels)
        self.assertGreater(len(selector.counts_), 50)
        for pattern, counts in selector.counts_.items():
            scanned = [count_instances(pattern, trace, 2) for trace in traces.values()]
            self.assertEqual(list(counts), scanned, msg=str(pattern))


if __name__ == "__main__":
    unittest.main()
