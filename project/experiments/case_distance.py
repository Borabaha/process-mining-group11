"""Explicit case distance for the IMPresseD "case distance" interest function.

Background (2023 code, ``InteractivePatternDetection/IMIPD.py``)
---------------------------------------------------------------
``calculate_pairwise_case_distance`` (IMIPD.py:135-165) computes one distance
per pair of cases from the case attributes:

* categorical attributes are label-encoded and passed to
  ``scipy.spatial.distance.pdist(codes, 'jaccard')`` (IMIPD.py:140-146);
* numeric attributes go through ``pdist(values, 'euclid')`` and the resulting
  distances (not the attributes) are min-max scaled to [0, 1] (IMIPD.py:148-155);
* with ``m`` categorical attributes the two parts are combined as
  ``(m * categorical + numeric) / (1 + m)`` (IMIPD.py:158).

``similarity_measuring_patterns`` (IMIPD.py:37-60) then gives every pattern the
MEAN distance over all (case with the pattern, case without the pattern) pairs.

Why this module exists
----------------------
``pdist(..., 'jaccard')`` on integer label codes changes its meaning with the
scipy version (experiment C12): scipy 1.11.4 ignores attributes on which both
cases carry label code 0, scipy >= 1.15 turns every code into "is it non-zero".
Both depend on which category happens to be encoded as 0. This module replaces
that call by the explicit share of categorical attributes on which two cases
differ (identical to ``pdist(codes, 'hamming')``), which does not depend on the
label encoding or on the scipy version. The numeric part and the combination
formula are kept exactly as in the original code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

# Default case attributes for BPIC11 (Guide Part 6, Table B, row B12).
BPIC11_NUMERIC_COLS = ["Age"]
BPIC11_CATEGORICAL_COLS = ["Diagnosis", "Treatment code", "Diagnosis code", "Specialism code"]


def build_case_attribute_table(events, attribute_cols, case_col="case:concept:name", order_col="event_nr"):
    """Return one row per case holding the attribute values of its first event.

    Parameters
    ----------
    events : pandas.DataFrame
        Event log, one row per event.
    attribute_cols : list of str
        Case attributes to keep.
    case_col, order_col : str
        Case identifier and the column that orders the events inside a case.

    Returns
    -------
    pandas.DataFrame
        Indexed by case id (sorted), columns ``attribute_cols``. Missing values
        are kept as they are (``groupby().first()`` would skip them).
    """
    ordered = events.sort_values([case_col, order_col], kind="stable")
    first_events = ordered.drop_duplicates(subset=case_col, keep="first")
    return first_events.set_index(case_col)[list(attribute_cols)]


def categorical_mismatch_share(case_table, categorical_cols):
    """Share of categorical attributes on which two cases differ (0 = same on all).

    Returns an ``n x n`` matrix. With ``m`` attributes the possible values are
    0, 1/m, ..., 1. Categories are compared by value, so the result does not
    depend on any label encoding.
    """
    n_cases = len(case_table)
    n_mismatches = np.zeros((n_cases, n_cases))
    for col in categorical_cols:
        codes = pd.factorize(case_table[col])[0]
        n_mismatches += codes[:, None] != codes[None, :]
    return n_mismatches / len(categorical_cols)


def numeric_minmax_euclidean(case_table, numeric_cols):
    """Euclidean distance on the raw numeric attributes, min-max scaled over all pairs.

    Reproduces IMIPD.py:148-155: the DISTANCES of all case pairs are scaled with
    ``(d - min d) / (max d - min d)``; the attributes themselves are not scaled.
    Consequences worth knowing: the result depends on which cases are in the
    table, and the smallest observed distance becomes 0 even if it was not 0.
    If all pairs have the same distance the result is 0 (as sklearn's
    ``MinMaxScaler`` does).
    """
    values = case_table[list(numeric_cols)].to_numpy(dtype=float)
    distances = pdist(values, "euclidean")
    if distances.size == 0:  # a single case has no pairs
        return np.zeros((len(case_table), len(case_table)))
    spread = distances.max() - distances.min()
    if spread > 0:
        distances = (distances - distances.min()) / spread
    else:
        distances = np.zeros_like(distances)
    return squareform(distances)


def pairwise_case_distance(case_table, numeric_cols, categorical_cols):
    """Distance between every pair of cases, based on their case attributes.

    Parameters
    ----------
    case_table : pandas.DataFrame
        One row per case (see ``build_case_attribute_table``).
    numeric_cols, categorical_cols : list of str
        Attribute columns of each kind; either list may be empty, not both.

    Returns
    -------
    numpy.ndarray
        Symmetric ``n x n`` matrix with a zero diagonal and values in [0, 1];
        row/column ``i`` is row ``i`` of ``case_table``.

        * both kinds: ``(m * categorical + numeric) / (1 + m)`` with
          ``m = len(categorical_cols)`` (IMIPD.py:158) - every categorical
          attribute weighs as much as ALL numeric attributes together;
        * only categorical: the mismatch share; only numeric: the scaled distance.

    Raises
    ------
    ValueError
        If a used column contains missing values (decide how to impute first).
    """
    numeric_cols, categorical_cols = list(numeric_cols), list(categorical_cols)
    if not numeric_cols and not categorical_cols:
        raise ValueError("give at least one numeric or categorical column")
    n_missing = case_table[numeric_cols + categorical_cols].isna().sum()
    if n_missing.any():
        raise ValueError(f"missing values in case attributes: {n_missing[n_missing > 0].to_dict()}")

    if not numeric_cols:
        return categorical_mismatch_share(case_table, categorical_cols)
    numeric = numeric_minmax_euclidean(case_table, numeric_cols)
    if not categorical_cols:
        return numeric
    n_categorical = len(categorical_cols)
    categorical = categorical_mismatch_share(case_table, categorical_cols)
    return (n_categorical * categorical + numeric) / (1 + n_categorical)


def pattern_case_distance(dist_matrix, in_mask, undefined_value=np.nan):
    """Mean distance over all (case with the pattern, case without the pattern) pairs.

    Same quantity as ``Case_Distance_Interest`` in IMIPD.py:37-60, computed on
    the square matrix instead of a list of pairs.

    Parameters
    ----------
    dist_matrix : numpy.ndarray
        Square matrix from ``pairwise_case_distance``.
    in_mask : array-like of bool
        ``True`` for the cases that contain the pattern (e.g. ``counts > 0``),
        in the row order of ``dist_matrix``. To score on a subset of the cases
        (e.g. a training fold), pass ``dist_matrix[np.ix_(rows, rows)]`` and
        ``in_mask[rows]``.
    undefined_value : float
        Returned when the pattern is in ALL cases or in NO case. There is then
        no pair to average and the distance is undefined; the original code
        returns NaN (``np.mean`` of an empty list) with a RuntimeWarning. The
        default keeps NaN, without the warning. NaN must not reach
        ``paretoset`` (experiment C11): replace it first, e.g. call this
        function with ``undefined_value=1.0``, the worst possible distance.

    Returns
    -------
    float
    """
    in_mask = np.asarray(in_mask, dtype=bool)
    if in_mask.all() or not in_mask.any():
        return undefined_value
    return float(dist_matrix[np.ix_(in_mask, ~in_mask)].mean())
