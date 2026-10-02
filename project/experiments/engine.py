"""Fast location-permutation-importance engine (experiment C2, prototype).

Re-implements the pipeline core of Vazifehdoostirani et al. (2024),
"Uncovering the Hidden Significance of Activities Location in Predictive
Process Monitoring", without the per-case pandas filtering that makes the
original ``tools.py`` slow.

Building blocks
---------------
EventLog
    Loads a BPIC11 CSV exactly like ``DataManager._load_df`` (tools.py:36-78).
IndexEncoder
    Index-based one-hot encoding, same matrix as ``DataManager.index_encoding``
    (tools.py:338-364) after an int cast, written with direct array writes.
make_folds
    Seeded stratified k-fold split over cases.
LocationPermutationImportance
    ``mode='faithful'`` reproduces the original code bit for bit, including
    its known defects: ``compute`` = ``itemset_permutation_importance``
    (tools.py:498-552), ``compute_single_activities`` =
    ``trace_permutation_importance`` (tools.py:366-448).
    ``mode='fixed'`` implements what the paper describes (Section 3.2).

Positions are 1-based wherever they mean "location in the trace" (as
``event_nr`` in the CSV) and 0-based wherever they are Python list indexes.

Activity names are the original's normalised names (lower case, without
spaces, '-' and '_'); use ``normalise_activity`` on names that come from
another tool (e.g. IMPresseD output).

Run ``python engine.py --help`` for a small demo; ``test_engine.py`` holds the
verification checks.
"""

from __future__ import annotations

import argparse
import time
import warnings
import zlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

CASE_COL = "case:concept:name"
ACTIVITY_COL = "concept:name"
LABEL_COL = "label"
POSITION_COL = "event_nr"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "external" / "PermutationLocationImportance" / "datasets"
DATASETS = {
    "f1": DATASET_DIR / "BPIC11_f1_trunc36.csv",
    "f2": DATASET_DIR / "BPIC11_f2_trunc40.csv",
    "f3": DATASET_DIR / "BPIC11_f3_trunc31.csv",
}


def normalise_activity(name: str) -> str:
    """The engine's spelling of an activity name (as tools.py:39-50).

    Lower case, with every space, '-' and '_' removed.
    """
    return name.lower().replace(" ", "").replace("-", "").replace("_", "")


# --------------------------------------------------------------------------
# (a) Event log
# --------------------------------------------------------------------------
class EventLog:
    """An event log preprocessed exactly like the original ``DataManager``.

    Attributes
    ----------
    traces : dict[str, list[str]]
        Case id -> activity names in ``event_nr`` order. Never mutated by the
        engine; treat it as read-only.
    labels : dict[str, int]
        Case id -> outcome label (0/1). Insertion order = sorted case ids.
    case_ids : list[str]
        Case ids in the order of ``traces`` (sorted as strings).
    activities : list[str]
        Distinct activity names in order of first appearance.
    allowed_locations : dict[str, list[int]]
        Activity -> sorted 1-based positions at which it occurs in the log.
    l_max : float
        ``L_max_perc`` quantile of ``event_nr`` (computed, unused downstream).
    max_len : int
        Length of the longest trace.

    The numbers of cases and activities that must come out for the three
    BPIC11 logs are checked in ``test_engine.py`` (``check_event_log``), not
    here.
    """

    def __init__(self, path, frq_threshold: int = 2, l_max_perc: float = 0.8):
        self.path = Path(path)
        frame = self._load_frame(self.path, frq_threshold)
        self.traces = (
            frame.groupby(CASE_COL, sort=False)[ACTIVITY_COL].apply(list).to_dict()
        )
        first_label = frame.groupby(CASE_COL, sort=False)[LABEL_COL].first()
        self.labels = {case: int(label) for case, label in first_label.items()}
        self.case_ids = list(self.traces)
        self.activities = frame[ACTIVITY_COL].unique().tolist()
        observed = frame.groupby(ACTIVITY_COL)[POSITION_COL].unique()
        self.allowed_locations = {
            act: sorted(int(pos) for pos in observed[act]) for act in self.activities
        }
        self.l_max = float(np.quantile(frame[POSITION_COL], l_max_perc))
        self.max_len = max(len(trace) for trace in self.traces.values())

    @staticmethod
    def _load_frame(path: Path, frq_threshold: int) -> pd.DataFrame:
        """Replicate tools.py:36-64 (cleaning, sorting, rare-activity filter)."""
        frame = pd.read_csv(path, sep=",")
        if "lifecycle:transition" in frame.columns:
            raise NotImplementedError(
                "lifecycle logs (bpic2012) need tools.py:41-46; BPIC11 has none"
            )
        frame[CASE_COL] = frame[CASE_COL].astype(str)
        names = frame[ACTIVITY_COL].str.lower()
        for char in (" ", "-", "_"):
            names = names.str.replace(char, "", regex=False)
        frame[ACTIVITY_COL] = names
        frame = frame.sort_values([CASE_COL, POSITION_COL])
        frame[LABEL_COL] = frame[LABEL_COL].replace({"deviant": 1, "regular": 0})

        # Drop every case that contains an activity with < frq_threshold events.
        event_counts = frame.groupby(ACTIVITY_COL)[CASE_COL].count()
        rare = event_counts[event_counts < frq_threshold].index
        bad_cases = frame.loc[frame[ACTIVITY_COL].isin(rare), CASE_COL].unique()
        frame = frame[~frame[CASE_COL].isin(bad_cases)]

        # The engine identifies event_nr with "list index + 1"; make sure of it.
        expected_nr = frame.groupby(CASE_COL).cumcount() + 1
        if not (frame[POSITION_COL] == expected_nr).all():
            raise ValueError("event_nr is not 1..n within every case")
        return frame[[CASE_COL, ACTIVITY_COL, LABEL_COL, POSITION_COL]]


def observed_locations(traces: Iterable[Sequence[str]]) -> dict[str, list[int]]:
    """Activity -> sorted 1-based positions at which it occurs in ``traces``."""
    seen: dict[str, set[int]] = {}
    for trace in traces:
        for index, act in enumerate(trace):
            seen.setdefault(act, set()).add(index + 1)
    return {act: sorted(positions) for act, positions in seen.items()}


# --------------------------------------------------------------------------
# (b) Index-based encoding
# --------------------------------------------------------------------------
class IndexEncoder:
    """Index-based one-hot encoder equal to ``DataManager.index_encoding``.

    Feature ``e{i}_{act}`` is 1 when activity ``act`` sits at position ``i``;
    feature ``e{i}_0`` is 1 when the trace is shorter than ``i`` (padding).

    Column order (same as the original):
      1. per position, the columns ``pd.get_dummies`` creates: the padding
         column (if any fitted trace is shorter than that position) followed
         by the activities observed at that position, sorted by name;
      2. all (position, activity) pairs never observed in the fitted traces.
         The original iterates a Python ``set`` here, so its order changes
         with the interpreter's string-hash seed. ``missing_order='sorted'``
         (default) makes it deterministic; ``missing_order='hash'`` iterates
         ``set(activities)`` like the original, which matches the original
         column for column inside the same Python process.
    Block 2 columns are all zero in the fitted data, so their order cannot
    change a tree model trained on the matrix. It does matter for
    *predicting*: a model must be given the columns in the order it was
    fitted on (``LocationPermutationImportance`` checks this when the model
    knows its feature names).

    Parameters
    ----------
    traces : iterable of activity lists
        The traces that define the column layout (normally the whole log).
    activities : sequence of str
        All activity names (``EventLog.activities``).
    missing_order : {'sorted', 'hash'}
    dtype : numpy integer dtype of the produced matrices.
    """

    def __init__(self, traces, activities, missing_order="sorted", dtype=np.int8):
        if missing_order not in ("sorted", "hash"):
            raise ValueError("missing_order must be 'sorted' or 'hash'")
        traces = list(traces)
        self.dtype = dtype
        self.max_len = max(len(trace) for trace in traces)
        min_len = min(len(trace) for trace in traces)
        present = [set() for _ in range(self.max_len)]
        for trace in traces:
            for index, act in enumerate(trace):
                present[index].add(act)

        names: list[str] = []
        for index in range(self.max_len):
            if index >= min_len:  # some fitted trace is padded at this position
                names.append(f"e{index + 1}_0")
            names.extend(f"e{index + 1}_{act}" for act in sorted(present[index]))
        ordered = set(activities) if missing_order == "hash" else sorted(activities)
        for index in range(self.max_len):
            names.extend(
                f"e{index + 1}_{act}" for act in ordered if act not in present[index]
            )
        self.feature_names = names
        self.n_features = len(names)

        column_of = {name: column for column, name in enumerate(names)}
        self._code = {act: code for code, act in enumerate(activities)}
        # _column[index, activity code] and _pad_column[index]; -1 = no column.
        self._column = np.array(
            [
                [column_of[f"e{index + 1}_{act}"] for act in activities]
                for index in range(self.max_len)
            ],
            dtype=np.intp,
        )
        self._pad_column = np.array(
            [column_of.get(f"e{index + 1}_0", -1) for index in range(self.max_len)],
            dtype=np.intp,
        )
        self._index = np.arange(self.max_len)

    def transform(self, traces, pad_until: int | None = None) -> np.ndarray:
        """Encode ``traces`` into a new (n_traces, n_features) matrix.

        An empty ``traces`` gives a (0, n_features) matrix.
        """
        traces = list(traces)
        matrix = np.zeros((len(traces), self.n_features), dtype=self.dtype)
        self.write_rows(matrix, np.arange(len(traces)), traces, pad_until)
        return matrix

    def write_rows(self, matrix, rows, traces, pad_until: int | None = None):
        """Overwrite ``matrix[rows]`` with the encoding of ``traces`` in place.

        This is the fast path for re-encoding the traces changed by a
        permutation: three vectorised array writes, no pandas. ``traces`` may
        be any iterable (also a generator); nothing happens when it is empty.

        ``pad_until`` is the last position that receives a padding 1. The
        default (``max_len``) is the correct encoding. A smaller value
        reproduces the original's behaviour when it re-encodes only a subset
        of cases whose longest trace is shorter than the log's longest trace.

        Raises ``ValueError`` for a trace longer than ``max_len`` or an
        activity the encoder was not built with.
        """
        traces = list(traces)
        rows = np.asarray(rows, dtype=np.intp)
        if len(rows) != len(traces):
            raise ValueError(f"{len(rows)} rows for {len(traces)} traces")
        if not traces:
            return
        pad_until = self.max_len if pad_until is None else pad_until
        lengths = np.fromiter((len(t) for t in traces), dtype=np.intp, count=len(rows))
        if lengths.max() > self.max_len:
            raise ValueError(
                f"a trace has {lengths.max()} events; the encoder was built for "
                f"traces of at most {self.max_len} events"
            )
        try:
            codes = np.fromiter(
                (self._code[act] for trace in traces for act in trace),
                dtype=np.intp,
                count=int(lengths.sum()),
            )
        except KeyError as error:
            raise ValueError(
                f"activity {error.args[0]!r} is unknown to the encoder"
            ) from None
        positions = np.concatenate([self._index[:n] for n in lengths])
        matrix[rows] = 0
        matrix[np.repeat(rows, lengths), self._column[positions, codes]] = 1
        padded = (self._index >= lengths[:, None]) & (self._index < pad_until)
        padded &= self._pad_column >= 0
        row_pos, index_pos = np.nonzero(padded)
        matrix[rows[row_pos], self._pad_column[index_pos]] = 1

    def to_frame(self, matrix: np.ndarray, case_ids) -> pd.DataFrame:
        """Wrap a matrix in a DataFrame with feature names (for inspection)."""
        return pd.DataFrame(matrix, index=list(case_ids), columns=self.feature_names)


# --------------------------------------------------------------------------
# (c) Seeded folds
# --------------------------------------------------------------------------
def make_folds(labels: Mapping[str, int], k: int = 5, seed: int = 0):
    """Seeded stratified k-fold split over cases.

    Returns a list of ``(train_cases, test_cases)`` tuples (lists of case
    ids, in the order of ``labels``). The original uses an unseeded
    ``StratifiedKFold(shuffle=True)`` (tools.py:231).
    """
    cases = np.array(list(labels))
    outcome = np.array([labels[case] for case in cases])
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return [
        (cases[train].tolist(), cases[test].tolist())
        for train, test in splitter.split(cases, outcome)
    ]


# --------------------------------------------------------------------------
# Itemset occurrences and the shuffles
# --------------------------------------------------------------------------
def find_occurrences(trace: Sequence[str], itemset: Iterable[str]):
    """Greedy non-overlapping complete occurrences of ``itemset`` in ``trace``.

    Same result as the original ``find_itemset_indexes`` (tools.py:478-496),
    which enumerates all C(len, k) index combinations in lexicographic order
    and keeps each match that does not overlap an earlier kept one. That
    greedy rule always keeps "the j-th occurrence of every activity" as the
    j-th match, so it can be computed directly (checked against a brute-force
    copy of the original in test_engine.py).

    Returns a list of tuples of 0-based indexes, each tuple sorted ascending.
    """
    positions = [
        [index for index, act in enumerate(trace) if act == wanted]
        for wanted in set(itemset)
    ]
    return [tuple(sorted(match)) for match in zip(*positions)]


def shuffle_sequence_faithful(sequence, itemset, allowed_locations, rng):
    """Line-by-line port of the original ``shuffle_sequence`` (tools.py:450-476).

    Kept bug-compatible on purpose:
    * ``locations`` is keyed by the current index and updated with a strict
      window, so a later move can pop the wrong element (order not always
      preserved, other activities can move);
    * the allowed positions are not clipped to the trace length (a draw
      beyond the end appends);
    * the moves are sequential, so a later move shifts earlier ones.

    ``rng`` must be a ``numpy.random.RandomState``; it is called exactly like
    the original calls ``np.random.choice``.
    """
    shuffled = list(sequence)
    locations = {index: index for index in range(len(sequence))}
    for occurrence in find_occurrences(sequence, itemset):
        subset = [sequence[index] for index in occurrence]
        max_possible = {}
        max_location = 10000
        for act in reversed(subset):
            # Raises ValueError (max of empty list) exactly like the original
            # when the allowed positions cannot be chained.
            max_possible[act] = max(
                loc for loc in allowed_locations[act] if loc < max_location
            )
            max_location = max_possible[act]

        previous = -1
        for position_in_subset, act in enumerate(subset):
            window = [
                loc
                for loc in allowed_locations[act]
                if previous < loc <= max_possible[act]
            ]
            drawn = rng.choice(window, 1)[0]
            previous = drawn
            current = locations[occurrence[position_in_subset]]
            shuffled.insert(drawn - 1, shuffled.pop(current))
            locations[current] = min(drawn - 1, len(sequence) - 1)
            for loc in locations:
                if current < loc < drawn - 1:
                    locations[loc] -= 1
    return shuffled


def shuffle_activity_faithful(sequence, activity, allowed_locations, rng):
    """Port of the single-activity move of the original (tools.py:392-404).

    This is the inner loop of ``trace_permutation_importance`` with
    ``constrain=True``. Every occurrence of ``activity`` is moved, one after
    the other, to a position drawn from the activity's allowed positions
    that fit in the trace (``i < len(trace) + 1``).

    Kept bug-compatible on purpose: the occurrence indexes are collected
    before the first move and not updated, so from the second occurrence on
    the popped event can be a different activity.

    ``rng`` must be a ``numpy.random.RandomState``.
    """
    shuffled = list(sequence)
    indexes = [index for index, act in enumerate(shuffled) if act == activity]
    for index in indexes:
        pool = [loc for loc in allowed_locations[activity] if loc < len(shuffled) + 1]
        drawn = rng.choice(pool, 1)[0] - 1
        shuffled.insert(drawn, shuffled.pop(index))
    return shuffled


def _draw_sequential(pools, length, rng):
    """Positions for one occurrence, drawn one activity after the other.

    Each activity is drawn uniformly from the positions after the previous
    one that still leave room for the activities behind it (the original's
    window logic, tools.py:459-467). Not uniform over all increasing tuples:
    the first activity is drawn from its whole window, the later ones are
    squeezed behind it. Returns None when no increasing tuple exists.
    """
    # Backward pass: the latest position each activity may take so that the
    # activities after it still fit.
    upper_bounds = []
    bound = length + 1
    for pool in reversed(pools):
        below = [loc for loc in pool if loc < bound]
        if not below:
            return None
        bound = below[-1]
        upper_bounds.append(bound)
    upper_bounds.reverse()

    drawn, previous = [], 0
    for pool, upper in zip(pools, upper_bounds):
        window = [loc for loc in pool if previous < loc <= upper]
        previous = window[rng.integers(len(window))]
        drawn.append(previous)
    return drawn


def _draw_uniform(pools, rng):
    """Positions for one occurrence, uniform over all increasing tuples.

    ``ways[i][j]`` counts the strictly increasing completions when activity
    ``i`` takes ``pools[i][j]``; drawing each activity with probability
    proportional to its count gives every feasible tuple the same
    probability. Returns None when no increasing tuple exists.
    """
    ways = [[1] * len(pools[-1])]
    for pool, later_pool in zip(reversed(pools[:-1]), reversed(pools[1:])):
        later_ways = ways[0]
        ways.insert(0, [
            sum(w for later, w in zip(later_pool, later_ways) if later > loc)
            for loc in pool
        ])

    drawn, previous = [], 0
    for pool, counts in zip(pools, ways):
        candidates = [(loc, w) for loc, w in zip(pool, counts) if loc > previous and w]
        total = sum(w for _, w in candidates)
        if total == 0:
            return None
        # Pick candidate number ``pick`` when each is listed ``weight`` times.
        pick = int(rng.integers(total))
        for loc, weight in candidates:
            if pick < weight:
                break
            pick -= weight
        previous = loc
        drawn.append(loc)
    return drawn


def permute_trace_fixed(trace, occurrences, allowed_locations, rng, draw="sequential"):
    """Move every itemset occurrence to drawn allowed positions (paper 3.2).

    All occurrences are placed simultaneously in the *final* trace:

    * Feasibility constraint: an activity lands only on a position (1-based)
      that is in ``allowed_locations[activity]``, not beyond the trace length
      and not already taken by another moved event of this trace.
    * Ordering constraint: inside one occurrence the activities keep their
      relative order (their new positions are strictly increasing).
    * Every event that is not moved keeps its relative order and fills the
      remaining positions. Such an event can be shifted by one or more places
      and can then sit on a position where its activity was never observed:
      the constraints apply to the moved events only (a limitation the paper
      acknowledges; ``count_shifted_events`` measures it).

    Drawing (per occurrence, occurrences one after the other):

    * ``draw='sequential'`` (default, the paper/original procedure): the
      activities are drawn one after the other, each uniformly from the
      positions that still leave room for the activities after it. No redraw
      loop is needed and no draw can dead-end, but for sets of 2 or 3 the
      result is not uniform over all order-preserving position tuples.
    * ``draw='uniform'``: every order-preserving tuple of allowed positions
      of the occurrence is equally likely.

    For a single activity the two are the same distribution (uniform over
    the pool).

    Parameters
    ----------
    trace : sequence of str (not modified)
    occurrences : list of index tuples from ``find_occurrences``
    allowed_locations : dict activity -> sorted list of 1-based positions
    rng : numpy.random.Generator
    draw : {'sequential', 'uniform'}

    Returns
    -------
    (new_trace, placements) where ``placements`` holds, per occurrence, a
    list of ``(old_index, new_index)`` pairs (0-based); or ``(None, None)``
    when some occurrence has no feasible assignment (the caller then leaves
    the trace unchanged).
    """
    length = len(trace)
    taken: dict[int, int] = {}  # new 1-based position -> old 0-based index
    placements = []
    for occurrence in occurrences:
        pools = [
            [
                loc
                for loc in allowed_locations.get(trace[index], ())
                if loc <= length and loc not in taken
            ]
            for index in occurrence
        ]
        if draw == "uniform":
            drawn = _draw_uniform(pools, rng)
        else:
            drawn = _draw_sequential(pools, length, rng)
        if drawn is None:
            return None, None
        for index, position in zip(occurrence, drawn):
            taken[position] = index
        placements.append([(index, pos - 1) for index, pos in zip(occurrence, drawn)])

    moved = set(taken.values())
    others = (act for index, act in enumerate(trace) if index not in moved)
    new_trace = [
        trace[taken[loc]] if loc in taken else next(others)
        for loc in range(1, length + 1)
    ]
    return new_trace, placements


def count_shifted_events(trace, placements, allowed_sets) -> tuple[int, int]:
    """Side effect of one fixed-mode permutation on the events NOT moved.

    Returns ``(n_shifted, n_unobserved)``: how many of the other events
    changed position, and how many of those now sit on a position that is
    not in ``allowed_sets[activity]`` (activity -> set of 1-based positions).
    """
    moved_from = {old for placed in placements for old, _ in placed}
    moved_to = {new for placed in placements for _, new in placed}
    old_indexes = (index for index in range(len(trace)) if index not in moved_from)
    new_indexes = (index for index in range(len(trace)) if index not in moved_to)
    n_shifted = n_unobserved = 0
    for old, new in zip(old_indexes, new_indexes):
        if old != new:
            n_shifted += 1
            n_unobserved += (new + 1) not in allowed_sets.get(trace[old], ())
    return n_shifted, n_unobserved


# --------------------------------------------------------------------------
# (d) Importance
# --------------------------------------------------------------------------
def weighted_f1(y_true, y_pred) -> float:
    """Weighted F1, the score used by the original (tools.py:504, :539)."""
    return f1_score(y_true, y_pred, average="weighted")


def itemset_label(itemset: Iterable[str]) -> str:
    """Canonical text form of an activity set, e.g. ``'a, b, c'``."""
    return ", ".join(sorted(set(itemset)))


def validate_itemsets(itemsets, activities) -> dict:
    """Return ``itemsets`` as a dict id -> set of activities, or raise.

    ``itemsets`` is a sequence of activity collections or a dict id ->
    collection. Raises ``TypeError`` when an activity collection (or the
    whole argument) is a bare string, because ``set('abc')`` would silently
    become the characters, and ``ValueError`` for an empty set or for names
    that are not activities of the log (for example names that were not
    normalised with ``normalise_activity``).
    """
    if isinstance(itemsets, str):
        raise TypeError("itemsets must be a collection of activity collections")
    if not isinstance(itemsets, Mapping):
        itemsets = dict(enumerate(itemsets))
    known = set(activities)
    checked, unknown = {}, {}
    for itemset_id, items in itemsets.items():
        if isinstance(items, str):
            raise TypeError(
                f"itemset {itemset_id!r} is the string {items!r}; pass a "
                f"collection of activity names, e.g. [{items!r}]"
            )
        checked[itemset_id] = set(items)
        if not checked[itemset_id]:
            raise ValueError(f"itemset {itemset_id!r} is empty")
        for name in sorted(checked[itemset_id] - known):
            hint = normalise_activity(name)
            unknown[name] = hint if hint in known else None
    if unknown:
        details = ", ".join(
            f"{name!r}" + (f" (did you mean {hint!r}?)" if hint else "")
            for name, hint in unknown.items()
        )
        raise ValueError(
            f"not activities of the log: {details}. Activity names are lower "
            "case without spaces, '-' and '_' (see normalise_activity)."
        )
    return checked


def check_model_matches_encoder(model, encoder: IndexEncoder) -> None:
    """Raise ``ValueError`` if the model was fitted on other columns.

    The engine predicts on a bare NumPy matrix, so a model fitted on a
    DataFrame with another column order (for example the original frame with
    an encoder built with the default ``missing_order='sorted'``) would
    return wrong predictions without any error. The check uses whatever the
    model knows: its number of features and, when it was fitted on named
    columns, the names. A model fitted on a bare matrix carries no names, so
    there the caller must make sure it was fitted on this encoder's matrix.
    """
    n_features = getattr(model, "n_features_in_", None)
    if n_features is not None and n_features != encoder.n_features:
        raise ValueError(
            f"the model was fitted on {n_features} features, the encoder "
            f"produces {encoder.n_features}"
        )
    names = getattr(model, "feature_names_in_", None)
    if names is None and hasattr(model, "get_booster"):
        names = model.get_booster().feature_names
    if names is not None and list(names) != encoder.feature_names:
        same_set = set(names) == set(encoder.feature_names)
        raise ValueError(
            "the model's feature names differ from encoder.feature_names ("
            + ("same columns, different order" if same_set else "different columns")
            + "); fit the model on encoder.transform(...) of this encoder"
        )


class LocationPermutationImportance:
    """Location permutation importance of activity sets.

    Parameters
    ----------
    mode : {'fixed', 'faithful'}
        'faithful' behaves exactly like the original code: one working copy
        of the traces in which permutations accumulate over repeats and
        itemsets, the original (buggy) shuffles, NumPy's legacy random stream
        seeded with ``random_state``, and the original's re-encoding of the
        shuffled subset. 'fixed' implements the paper: every repeat starts
        from the original traces and both constraints hold
        (``permute_trace_fixed``).
    score_on : {'train', 'test'}
        Which fold is permuted and scored with the trained model. The
        original cross-validation script always passes the training fold.
    n_repeats : int
        Permutation repeats per itemset (original: 10).
    random_state : int
        Seed (original: 2023). In fixed mode every (fold, itemset, repeat)
        gets its own generator derived from this seed, so a result does not
        depend on which other itemsets are computed or in which order.
    allowed_from : {'log', 'train'}
        Fixed mode only: where the position pools come from. 'log': the whole
        log, as the original does. The model never sees these pools, but they
        contain positions that occur only in held-out cases, so held-out
        information enters the permutation step. 'train': the training-fold
        traces only (the strict option; a held-out trace whose itemset cannot
        be placed is left unchanged and counted). Use one setting for all
        runs that are compared with each other.
    draw : {'sequential', 'uniform'}
        Fixed mode only: how the positions of a set of 2 or more activities
        are drawn (see ``permute_trace_fixed``).
    """

    def __init__(self, mode="fixed", score_on="test", n_repeats=10,
                 random_state=2023, allowed_from="log", draw="sequential"):
        if mode not in ("fixed", "faithful"):
            raise ValueError("mode must be 'fixed' or 'faithful'")
        if score_on not in ("train", "test"):
            raise ValueError("score_on must be 'train' or 'test'")
        if allowed_from not in ("log", "train"):
            raise ValueError("allowed_from must be 'log' or 'train'")
        if draw not in ("sequential", "uniform"):
            raise ValueError("draw must be 'sequential' or 'uniform'")
        if mode == "faithful" and allowed_from != "log":
            raise ValueError("faithful mode always uses the whole-log positions")
        if mode == "faithful" and draw != "sequential":
            raise ValueError("faithful mode always uses the original's draw")
        self.mode = mode
        self.score_on = score_on
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.allowed_from = allowed_from
        self.draw = draw

    def compute(self, model, log: EventLog, encoder: IndexEncoder,
                train_cases, test_cases, itemsets, *, fold: int) -> pd.DataFrame:
        """Importance of every itemset for one fold.

        Parameters
        ----------
        model : fitted classifier with ``predict``. It must have been fitted
            on ``encoder.transform(...)`` of THIS encoder (same columns, same
            order); see ``check_model_matches_encoder``.
        log, encoder : the event log and its encoder.
        train_cases, test_cases : case ids of this fold. The iteration order
            of the scored list drives the random stream in faithful mode.
        itemsets : sequence of activity collections, or a dict id ->
            collection (checked by ``validate_itemsets``).
        fold : fold number (keyword, required). It is copied into the output
            and is part of the fixed-mode seed, so two folds must not share
            a number.

        Returns
        -------
        Tidy DataFrame, one row per (itemset, repeat): itemset_id, itemset,
        fold, repeat, baseline, permuted, importance (= baseline - permuted,
        weighted F1), n_traces_with_itemset, n_traces_changed,
        n_traces_unchanged_infeasible. Fixed mode adds
        n_other_events_shifted and n_other_events_unobserved (events that
        were not moved but changed position / now sit on a position where
        their activity is not in the position pools), summed over the changed
        traces.

        An (itemset, repeat) that changes no scored trace has importance 0.
        In fixed mode an itemset contained in no scored trace therefore gets
        importance 0 in every repeat and a ``UserWarning`` is issued; faithful
        mode raises ``ValueError`` before any work is done, because the
        original crashes on such an itemset.

        Size-1 itemsets in faithful mode run through this itemset routine,
        which is NOT what the original does for single activities; use
        ``compute_single_activities`` for that.
        """
        itemsets = validate_itemsets(itemsets, log.activities)
        traces, y_true, base_pred = self._scored_fold(
            model, log, encoder, train_cases, test_cases)

        if self.mode == "faithful":
            absent = [
                itemset_label(items) for items in itemsets.values()
                if not any(_is_eligible(trace, items) for trace in traces)
            ]
            if absent:
                raise ValueError(
                    f"no scored trace can be permuted for {absent}; the original "
                    "crashes here too (index_encoding of an empty frame)"
                )
            run = self._run_faithful(encoder, traces, log.allowed_locations, itemsets)
        else:
            if self.allowed_from == "log":
                allowed = log.allowed_locations
            else:
                allowed = observed_locations(log.traces[c] for c in train_cases)
            run = self._run_fixed(encoder, traces, allowed, itemsets, fold)
        table = self._score(model, y_true, base_pred, run, fold)

        if self.mode == "fixed" and len(table):
            empty = table.loc[table["n_traces_with_itemset"] == 0, "itemset"].unique()
            if len(empty):
                warnings.warn(
                    f"fold {fold}: {len(empty)} itemset(s) occur in no scored "
                    f"trace (importance 0): {list(empty[:5])}"
                    + (" ..." if len(empty) > 5 else ""),
                    stacklevel=2,
                )
        return table

    def compute_single_activities(self, model, log: EventLog, encoder: IndexEncoder,
                                  train_cases, test_cases, *, fold: int,
                                  activities=None) -> pd.DataFrame:
        """Importance of single activities for one fold.

        Faithful mode is a port of the original ``trace_permutation_importance``
        with ``constrain=True`` (tools.py:366-448), the routine behind the
        paper's single-activity figure. It differs from the itemset routine:

        * every occurrence of the activity is moved (``shuffle_activity_faithful``),
          with positions clipped to the trace length;
        * an activity with fewer than 2 allowed positions gets importance 0
          and is not permuted;
        * when fewer than 3 scored traces contain the activity, the traces
          are permuted (the permutation stays in the working copy) but the
          importance is 0;
        * permutations accumulate over repeats AND activities, so a value
          depends on every activity processed before it. The original always
          runs all activities of the log in order of first appearance, which
          is the default here; pass ``activities`` only if you know why.

        The output has the columns of ``compute`` (``itemset_id`` and
        ``itemset`` hold the activity name) plus ``guard``: '' or the reason
        for a forced 0 ('one_allowed_position', 'fewer_than_3_cases').

        Fixed mode has no separate single-activity algorithm: the call is
        the same as ``compute`` with one single-activity set per activity.
        """
        activities = list(log.activities if activities is None else activities)
        if self.mode == "fixed":
            return self.compute(model, log, encoder, train_cases, test_cases,
                                {act: [act] for act in activities}, fold=fold)
        validate_itemsets([[act] for act in activities], log.activities)
        traces, y_true, base_pred = self._scored_fold(
            model, log, encoder, train_cases, test_cases)
        run = self._run_faithful_single(
            encoder, traces, log.allowed_locations, activities)
        return self._score(model, y_true, base_pred, run, fold)

    def _scored_fold(self, model, log, encoder, train_cases, test_cases):
        """Traces, labels and baseline predictions of the scored fold."""
        check_model_matches_encoder(model, encoder)
        scored = list(train_cases if self.score_on == "train" else test_cases)
        traces = [log.traces[case] for case in scored]
        y_true = [log.labels[case] for case in scored]
        return traces, y_true, model.predict(encoder.transform(traces))

    @staticmethod
    def _score(model, y_true, base_pred, run, fold) -> pd.DataFrame:
        """Turn the re-encoded rows of every (itemset, repeat) into a table."""
        baseline = weighted_f1(y_true, base_pred)
        records = []
        for itemset_id, items, repeat, rows, new_rows, counts in run:
            # Unchanged rows keep their baseline prediction, so only the
            # re-encoded rows need the model.
            pred = base_pred.copy()
            if len(rows):
                pred[rows] = model.predict(new_rows)
            permuted = weighted_f1(y_true, pred)
            records.append({
                "itemset_id": itemset_id,
                "itemset": itemset_label(items),
                "fold": fold,
                "repeat": repeat,
                "baseline": baseline,
                "permuted": permuted,
                "importance": baseline - permuted,
                **counts,
            })
        return pd.DataFrame.from_records(records)

    def _run_faithful(self, encoder, traces, allowed, itemsets):
        """Yield the re-encoded rows of every (itemset, repeat), original way."""
        rng = np.random.RandomState(self.random_state)  # = np.random.seed(...)
        working = [list(trace) for trace in traces]  # tools.py:507, never reset
        for itemset_id, items in itemsets.items():
            for repeat in range(self.n_repeats):
                rows, n_changed = [], 0
                for row, trace in enumerate(working):
                    if not _is_eligible(trace, items):
                        continue
                    working[row] = shuffle_sequence_faithful(trace, items, allowed, rng)
                    n_changed += working[row] != trace
                    rows.append(row)
                counts = {
                    "n_traces_with_itemset": len(rows),
                    "n_traces_changed": n_changed,
                    "n_traces_unchanged_infeasible": 0,
                }
                new_rows = _encode_shuffled_subset(encoder, working, rows)
                yield itemset_id, items, repeat, rows, new_rows, counts

    def _run_faithful_single(self, encoder, traces, allowed, activities):
        """Yield the re-encoded rows of every (activity, repeat), original way."""
        rng = np.random.RandomState(self.random_state)  # = np.random.seed(...)
        working = [list(trace) for trace in traces]  # tools.py:373, never reset
        nothing = encoder.transform([])
        for act in activities:
            for repeat in range(self.n_repeats):
                counts = {
                    "n_traces_with_itemset": 0,
                    "n_traces_changed": 0,
                    "n_traces_unchanged_infeasible": 0,
                    "guard": "",
                }
                if len(allowed[act]) < 2:  # tools.py:378, nothing is permuted
                    counts["guard"] = "one_allowed_position"
                    yield act, {act}, repeat, [], nothing, counts
                    continue
                rows = []
                for row, trace in enumerate(working):
                    if act not in trace:
                        continue
                    working[row] = shuffle_activity_faithful(trace, act, allowed, rng)
                    counts["n_traces_changed"] += working[row] != trace
                    rows.append(row)
                counts["n_traces_with_itemset"] = len(rows)
                if len(rows) < 3:  # tools.py:408, permuted but not scored
                    counts["guard"] = "fewer_than_3_cases"
                    yield act, {act}, repeat, [], nothing, counts
                    continue
                new_rows = _encode_shuffled_subset(encoder, working, rows)
                yield act, {act}, repeat, rows, new_rows, counts

    def _run_fixed(self, encoder, traces, allowed, itemsets, fold):
        """Yield the re-encoded rows of every (itemset, repeat), paper way."""
        allowed_sets = {act: set(positions) for act, positions in allowed.items()}
        # The default draw keeps the four-argument call, so wrappers written
        # around permute_trace_fixed (tests, verifier) need not know ``draw``.
        options = {} if self.draw == "sequential" else {"draw": self.draw}
        for itemset_id, items in itemsets.items():
            label_hash = zlib.crc32(itemset_label(items).encode("utf-8"))
            # Occurrences depend only on the original traces: find them once.
            eligible = [
                (row, trace, find_occurrences(trace, items))
                for row, trace in enumerate(traces)
                if _is_eligible(trace, items)
            ]
            for repeat in range(self.n_repeats):
                rng = np.random.default_rng(
                    [self.random_state, fold, label_hash, repeat]
                )
                rows, changed = [], []
                n_infeasible = n_shifted = n_unobserved = 0
                for row, trace, occurrences in eligible:
                    new_trace, placements = permute_trace_fixed(
                        trace, occurrences, allowed, rng, **options)
                    if new_trace is None:
                        n_infeasible += 1
                    elif new_trace != trace:
                        rows.append(row)
                        changed.append(new_trace)
                        shifted = count_shifted_events(trace, placements, allowed_sets)
                        n_shifted += shifted[0]
                        n_unobserved += shifted[1]
                counts = {
                    "n_traces_with_itemset": len(eligible),
                    "n_traces_changed": len(rows),
                    "n_traces_unchanged_infeasible": n_infeasible,
                    "n_other_events_shifted": n_shifted,
                    "n_other_events_unobserved": n_unobserved,
                }
                new_rows = encoder.transform(changed)
                yield itemset_id, items, repeat, rows, new_rows, counts


def _is_eligible(trace, items: set) -> bool:
    """The original's rule for "this trace is permuted" (tools.py:521-523)."""
    return items.issubset(trace) and len(items) != len(trace)


def _encode_shuffled_subset(encoder, working, rows) -> np.ndarray:
    """Re-encode the shuffled traces like the original (tools.py:529-531).

    The original re-encodes only the shuffled cases, so padding stops at the
    longest shuffled trace.
    """
    shuffled = [working[row] for row in rows]
    return encoder.transform(shuffled, pad_until=max(len(trace) for trace in shuffled))


# --------------------------------------------------------------------------
# (e) Small demo
# --------------------------------------------------------------------------
def most_frequent_activities(log: EventLog, n: int) -> list[str]:
    """The ``n`` activities contained in the most cases (ties: by name)."""
    support = {act: 0 for act in log.activities}
    for trace in log.traces.values():
        for act in set(trace):
            support[act] += 1
    return sorted(support, key=lambda act: (-support[act], act))[:n]


def main() -> None:
    """Demo: one fold, the 1-, 2- and 3-set of the most frequent activities."""
    from xgboost import XGBClassifier

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="f1")
    parser.add_argument("--mode", choices=["fixed", "faithful"], default="fixed")
    parser.add_argument("--score-on", choices=["train", "test"], default="test")
    parser.add_argument("--allowed-from", choices=["log", "train"], default="log")
    parser.add_argument("--draw", choices=["sequential", "uniform"],
                        default="sequential")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()

    start = time.perf_counter()
    log = EventLog(DATASETS[args.dataset])
    encoder = IndexEncoder(log.traces.values(), log.activities)
    train_cases, test_cases = make_folds(log.labels, k=5, seed=args.seed)[args.fold]
    print(f"{args.dataset}: {len(log.traces)} cases, {len(log.activities)} "
          f"activities, {encoder.n_features} features "
          f"({time.perf_counter() - start:.2f} s)")

    model = XGBClassifier()
    model.fit(encoder.transform(log.traces[c] for c in train_cases),
              [log.labels[c] for c in train_cases])
    top = most_frequent_activities(log, 3)
    itemsets = [top[:1], top[:2], top[:3]]

    start = time.perf_counter()
    engine = LocationPermutationImportance(
        mode=args.mode, score_on=args.score_on, n_repeats=args.repeats,
        random_state=args.seed, allowed_from=args.allowed_from, draw=args.draw)
    table = engine.compute(model, log, encoder, train_cases, test_cases,
                           itemsets, fold=args.fold)
    seconds = time.perf_counter() - start
    print(table.to_string(index=False))
    print(f"{seconds / len(table):.4f} s per (itemset, repeat)")


if __name__ == "__main__":
    main()
