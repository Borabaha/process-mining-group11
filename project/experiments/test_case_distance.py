"""Unit tests for case_distance.py on tiny hand-computed examples.

Run from the project root:
    python -m unittest experiments/test_case_distance.py -v
"""
from __future__ import annotations

import math
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, str(Path(__file__).resolve().parent))

from case_distance import (  # noqa: E402
    build_case_attribute_table,
    categorical_mismatch_share,
    numeric_minmax_euclidean,
    pairwise_case_distance,
    pattern_case_distance,
)


class CategoricalPartTest(unittest.TestCase):
    """Share of categorical attributes on which two cases differ."""

    def test_three_cases_by_hand(self):
        table = pd.DataFrame({"x": ["a", "a", "z"], "y": ["b", "b", "b"], "z": ["c", "d", "d"]})
        # case0-case1 differ on z only -> 1/3; case0-case2 on x and z -> 2/3; case1-case2 on x -> 1/3
        expected = np.array([[0, 1 / 3, 2 / 3], [1 / 3, 0, 1 / 3], [2 / 3, 1 / 3, 0]])
        np.testing.assert_allclose(pairwise_case_distance(table, [], ["x", "y", "z"]), expected)

    def test_the_two_tiny_pairs_of_the_guide(self):
        """[[0,1,2],[0,1,3]] -> 1/3 and [[0,1,2],[1,1,3]] -> 2/3.

        scipy 1.11.4 pdist('jaccard') gives 0.5 and 0.667 for these pairs,
        scipy 1.18.1 gives 0.0 and 0.333 (experiment C12). Only the second pair
        agrees with scipy 1.11.4; see RESULT.md for why 1/3 is the value we want.
        """
        cols = ["a", "b", "c"]
        first = pd.DataFrame([[0, 1, 2], [0, 1, 3]], columns=cols)
        second = pd.DataFrame([[0, 1, 2], [1, 1, 3]], columns=cols)
        self.assertAlmostEqual(pairwise_case_distance(first, [], cols)[0, 1], 1 / 3)
        self.assertAlmostEqual(pairwise_case_distance(second, [], cols)[0, 1], 2 / 3)

    def test_equals_scipy_hamming_on_label_codes(self):
        rng = np.random.default_rng(0)
        codes = rng.integers(0, 4, size=(30, 5))
        table = pd.DataFrame(codes, columns=list("abcde"))
        expected = squareform(pdist(codes, "hamming"))
        np.testing.assert_allclose(categorical_mismatch_share(table, list("abcde")), expected)

    def test_does_not_depend_on_the_names_of_the_categories(self):
        """Renaming categories changes the label codes, but must not change the distance."""
        table = pd.DataFrame({"x": ["a", "b", "c", "a"], "y": ["p", "p", "q", "r"]})
        renamed = table.replace({"x": {"a": "zz"}, "y": {"p": "zz"}})  # 'a'/'p' no longer sort first
        np.testing.assert_allclose(
            pairwise_case_distance(table, [], ["x", "y"]), pairwise_case_distance(renamed, [], ["x", "y"])
        )


class NumericPartTest(unittest.TestCase):
    """Euclidean distance, min-max scaled over all pairs."""

    def test_three_ages_by_hand(self):
        table = pd.DataFrame({"Age": [10, 20, 40]})
        # raw distances: (0,1)=10, (0,2)=30, (1,2)=20 -> min 10, max 30
        # scaled: (10-10)/20=0, (30-10)/20=1, (20-10)/20=0.5
        expected = np.array([[0, 0, 1], [0, 0, 0.5], [1, 0.5, 0]])
        np.testing.assert_allclose(pairwise_case_distance(table, ["Age"], []), expected)

    def test_two_numeric_columns_use_raw_values(self):
        table = pd.DataFrame({"u": [0, 3, 0], "v": [0, 4, 10]})
        # raw distances: (0,1)=5, (0,2)=10, (1,2)=sqrt(9+36)=6.708...
        d12 = (math.sqrt(45) - 5) / 5
        expected = np.array([[0, 0, 1], [0, 0, d12], [1, d12, 0]])
        np.testing.assert_allclose(numeric_minmax_euclidean(table, ["u", "v"]), expected)

    def test_constant_column_gives_zero(self):
        table = pd.DataFrame({"Age": [50, 50, 50]})
        np.testing.assert_array_equal(pairwise_case_distance(table, ["Age"], []), np.zeros((3, 3)))


class CombinedDistanceTest(unittest.TestCase):
    """(m * categorical + numeric) / (1 + m), IMIPD.py:158."""

    def setUp(self):
        self.table = pd.DataFrame({"Age": [10, 20, 40], "x": ["a", "a", "b"], "y": ["p", "q", "q"]})

    def test_combination_by_hand(self):
        # numeric (see NumericPartTest):   (0,1)=0    (0,2)=1    (1,2)=0.5
        # categorical, m=2:                (0,1)=1/2  (0,2)=2/2  (1,2)=1/2
        # combined (2*cat + num) / 3:      (0,1)=1/3  (0,2)=1    (1,2)=0.5
        expected = np.array([[0, 1 / 3, 1], [1 / 3, 0, 0.5], [1, 0.5, 0]])
        np.testing.assert_allclose(pairwise_case_distance(self.table, ["Age"], ["x", "y"]), expected)

    def test_symmetric_zero_diagonal_and_within_unit_interval(self):
        dist = pairwise_case_distance(self.table, ["Age"], ["x", "y"])
        np.testing.assert_allclose(dist, dist.T)
        np.testing.assert_array_equal(np.diag(dist), np.zeros(3))
        self.assertTrue(((dist >= 0) & (dist <= 1)).all())

    def test_input_table_is_not_modified(self):
        before = self.table.copy()
        pairwise_case_distance(self.table, ["Age"], ["x", "y"])
        pd.testing.assert_frame_equal(self.table, before)

    def test_missing_value_raises(self):
        table = self.table.astype({"Age": float})
        table.loc[1, "Age"] = np.nan
        with self.assertRaises(ValueError):
            pairwise_case_distance(table, ["Age"], ["x", "y"])

    def test_no_columns_raises(self):
        with self.assertRaises(ValueError):
            pairwise_case_distance(self.table, [], [])


class PatternCaseDistanceTest(unittest.TestCase):
    """Mean distance over (case with pattern, case without pattern) pairs."""

    def setUp(self):
        self.dist = np.array(
            [
                [0.0, 0.1, 0.2, 0.3],
                [0.1, 0.0, 0.4, 0.5],
                [0.2, 0.4, 0.0, 0.6],
                [0.3, 0.5, 0.6, 0.0],
            ]
        )

    def test_two_in_two_out(self):
        # pairs (0,2)=0.2, (0,3)=0.3, (1,2)=0.4, (1,3)=0.5 -> mean 0.35
        self.assertAlmostEqual(pattern_case_distance(self.dist, [True, True, False, False]), 0.35)

    def test_one_in_three_out(self):
        # pairs (3,0)=0.3, (3,1)=0.5, (3,2)=0.6 -> mean 1.4/3
        self.assertAlmostEqual(pattern_case_distance(self.dist, [False, False, False, True]), 1.4 / 3)

    def test_mask_from_pattern_counts(self):
        counts = np.array([2, 0, 1, 0])  # pattern occurs twice in case 0, once in case 2
        # pairs (0,1)=0.1, (0,3)=0.3, (2,1)=0.4, (2,3)=0.6 -> mean 0.35
        self.assertAlmostEqual(pattern_case_distance(self.dist, counts > 0), 0.35)

    def test_pattern_in_all_or_no_cases_is_nan_without_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # the original code emits a RuntimeWarning here
            self.assertTrue(math.isnan(pattern_case_distance(self.dist, [True] * 4)))
            self.assertTrue(math.isnan(pattern_case_distance(self.dist, [False] * 4)))

    def test_undefined_value_can_be_chosen(self):
        self.assertEqual(pattern_case_distance(self.dist, [True] * 4, undefined_value=1.0), 1.0)


class CaseAttributeTableTest(unittest.TestCase):
    """First value per case, ordered by event number."""

    def test_first_event_per_case(self):
        events = pd.DataFrame(
            {
                "case:concept:name": ["c2", "c1", "c1", "c2"],
                "event_nr": [2, 2, 1, 1],
                "Age": [70, 31, 30, np.nan],
                "Diagnosis": ["d2-late", "d1-late", "d1", "d2"],
            }
        )
        table = build_case_attribute_table(events, ["Age", "Diagnosis"])
        self.assertEqual(list(table.index), ["c1", "c2"])
        self.assertEqual(list(table["Diagnosis"]), ["d1", "d2"])
        self.assertEqual(table.loc["c1", "Age"], 30)
        self.assertTrue(math.isnan(table.loc["c2", "Age"]))  # the missing first value is not skipped


if __name__ == "__main__":
    unittest.main()
