"""Apriori-based selection of activity sets (the baseline strategy of the 2024 pipeline).

Two selectors live here:

* :class:`AprioriSelector` - the *variant-friendly* selector: the ``top_k`` most
  frequent activity sets **per size** (1, 2, 3), with a deterministic tie-break.
* :func:`original_top10` - a line-by-line replica of the ORIGINAL selection
  (``DataManager.frequent_activity_sets``, ``tools.py:160-177`` of the 2024 repo):
  the ``top_k`` most frequent sets of size > 1, no size cap, pandas tie order.

Both work on plain traces (``{case_id: [activity, activity, ...]}``), so this module
does not import anything from the original repositories.

A *transaction* is the set of distinct activities of one case (order and repetitions
are discarded), exactly as in the original code. The *support* of an activity set is
the fraction of cases that contain all of its activities.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

TABLE_COLUMNS = ["itemset", "size", "support", "count"]


def encode_transactions(traces: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """Turn traces into the one-hot transaction table used by Apriori.

    Replicates ``tools.py:161-166``: one row per case, one boolean column per
    activity (columns sorted by name), True when the activity occurs in the case.
    """
    transactions = [list(trace) for trace in traces.values()]
    encoder = TransactionEncoder()
    onehot = encoder.fit(transactions).transform(transactions)
    return pd.DataFrame(onehot, columns=encoder.columns_)


class AprioriSelector:
    """Select the ``top_k`` most frequent activity sets for every size 1..``max_len``.

    Ranking inside one size: support descending, ties broken lexicographically on
    the sorted activity names, so the result never depends on pandas/numpy sort
    internals or on Python's string-hash randomisation.

    Parameters
    ----------
    min_support : float
        Minimum fraction of cases that must contain the set (``support >= min_support``).
    max_len : int or None
        Largest set size that is mined; ``None`` = no cap, as in the original code.
    top_k : int
        Number of sets kept per size.
    """

    def __init__(self, min_support: float, max_len: int | None = 3, top_k: int = 10):
        self.min_support = min_support
        self.max_len = max_len
        self.top_k = top_k

    def full_table(self, traces: Mapping[str, Sequence[str]]) -> pd.DataFrame:
        """Return every frequent set: columns ``itemset, size, support, count``.

        ``itemset`` is a tuple of sorted activity names, ``count`` the number of
        cases containing it. Rows are ordered by size, then by the ranking rule.
        """
        onehot = encode_transactions(traces)
        mined = apriori(onehot, min_support=self.min_support, use_colnames=True,
                        max_len=self.max_len)
        n_cases = len(onehot)
        rows = [
            (tuple(sorted(items)), len(items), float(support), int(round(support * n_cases)))
            for items, support in zip(mined["itemsets"], mined["support"])
        ]
        # Sorting on the integer case count makes ties exact (no float noise).
        rows.sort(key=lambda row: (row[1], -row[3], row[0]))
        return pd.DataFrame(rows, columns=TABLE_COLUMNS)

    def select_table(self, traces: Mapping[str, Sequence[str]]) -> pd.DataFrame:
        """Return the selected rows of :meth:`full_table` plus a 1-based ``rank`` per size."""
        table = self.full_table(traces)
        selected = table.groupby("size", sort=True).head(self.top_k).reset_index(drop=True)
        selected["rank"] = selected.groupby("size").cumcount() + 1
        return selected

    def select(self, traces: Mapping[str, Sequence[str]]) -> dict[int, list[list[str]]]:
        """Return ``{size: [sorted activity names, ...]}`` for size 1..``max_len``.

        A size with fewer than ``top_k`` frequent sets returns the ones that exist
        (possibly an empty list) - check the lengths if you need exactly ``top_k``.
        With ``max_len=None`` only the sizes that actually occur are returned.
        """
        selected = self.select_table(traces)
        sizes = range(1, self.max_len + 1) if self.max_len else sorted(selected["size"].unique())
        return {
            int(size): [list(itemset)
                        for itemset in selected.loc[selected["size"] == size, "itemset"]]
            for size in sizes
        }


def original_top10(traces: Mapping[str, Sequence[str]], min_support: float = 0.5,
                   top_k: int = 10) -> tuple[list[list[str]], pd.DataFrame]:
    """Reproduce the ORIGINAL selection of ``DataManager.frequent_activity_sets``.

    Same statements as ``tools.py:160-177``: Apriori without a size cap, keep the
    ``top_k + number_of_activities`` most frequent sets, drop the sets of size 1,
    keep the ``top_k`` most frequent of the rest. Ties are left to pandas'
    ``sort_values`` exactly as in the original, and fewer than ``top_k`` sets are
    returned silently when the support threshold is too high.

    Returns ``(list of activity lists, selected rows)`` like the original. The order
    of the activities inside one list is the iteration order of a ``frozenset`` -
    as in the original it is not sorted (sort it yourself if you need a fixed order).
    """
    onehot = encode_transactions(traces)
    frequent = apriori(onehot, min_support=min_support, use_colnames=True)
    frequent = frequent.sort_values(["support"], ascending=False).head(top_k + onehot.shape[1])
    frequent["item_size"] = frequent.itemsets.apply(lambda x: len(list(x)))
    frequent = frequent[frequent["item_size"] > 1]
    selected = frequent.sort_values(["support"], ascending=False).head(top_k)
    return selected.itemsets.apply(lambda x: list(x)).to_list(), selected
