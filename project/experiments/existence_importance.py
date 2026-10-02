"""Existence-importance baseline (prototype for experiment C10).

"Existence importance" is the classical permutation feature importance (PFI)
of features that only say *whether / how often* an activity set occurs in a
case, ignoring *where* it occurs.  It is the blue-box baseline of
Vazifehdoostirani et al. (2024), produced there by ``Classical_Permutation.py``.

This module re-implements that flow as a small class with explicit options:

* ``encoding``  - ``'count'``  : min over the set's activities of the number of
                                 occurrences in the trace (what the original
                                 script computes), or
                  ``'binary'`` : 1 if every activity of the set occurs, else 0
                                 (what the 2024 paper describes).
* ``matching``  - ``'exact'``     : activities are compared as whole tokens, or
                  ``'substring'`` : the original behaviour, ``str.count`` on the
                                    ``'->'``-joined trace (``'370407'`` also
                                    matches inside ``'370407c'``).
* ``scoring``   - ``'accuracy'`` (what the original effectively uses, because it
                  passes no ``scoring`` to sklearn) or ``'f1_weighted'`` (what
                  its axis label claims and what the location importance uses).
* ``score_on``  - ``'train'`` (original) or ``'test'`` (held-out fold).

Everything is seeded, so two runs give identical numbers.

Usage (from the project root)::

    .venv/Scripts/python.exe experiments/run_c10.py
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_2024 = PROJECT_ROOT / "external" / "PermutationLocationImportance"
DATASETS = {
    "f1": REPO_2024 / "datasets" / "BPIC11_f1_trunc36.csv",
    "f2": REPO_2024 / "datasets" / "BPIC11_f2_trunc40.csv",
    "f3": REPO_2024 / "datasets" / "BPIC11_f3_trunc31.csv",
}

ENCODINGS = ("count", "binary")
MATCHINGS = ("exact", "substring")
SCORINGS = ("accuracy", "f1_weighted")
SCORE_ON = ("train", "test")


# --------------------------------------------------------------------------- #
# Data access (thin wrappers around the read-only 2024 repository)
# --------------------------------------------------------------------------- #
def load_data_manager_class():
    """Return the 2024 repo's ``DataManager`` class without touching sys.path.

    The 2023 and the 2024 repository both ship a module called ``tools.py``;
    loading the file under the private name ``tools_2024`` avoids any clash.
    """
    spec = importlib.util.spec_from_file_location("tools_2024", REPO_2024 / "tools.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DataManager


@dataclass
class CaseLog:
    """An event log reduced to what the existence baseline needs.

    Attributes
    ----------
    traces : pd.Series
        index = case id, value = list of activity names ordered by ``event_nr``.
    labels : pd.Series
        index = case id, value = 0/1 outcome label.
    """

    traces: pd.Series
    labels: pd.Series

    @classmethod
    def from_data_manager(cls, data_manager) -> "CaseLog":
        """Build the case-level view from a loaded 2024 ``DataManager``."""
        data = data_manager.data.sort_values([data_manager.case_id, "event_nr"])
        grouped = data.groupby(data_manager.case_id, sort=True)
        traces = grouped[data_manager.activity].apply(list)
        labels = grouped[data_manager.outcome].first().astype(int)
        return cls(traces=traces, labels=labels)


def load_log(dataset: str, frq_threshold: int = 2):
    """Load one BPIC11 log through the original ``DataManager``.

    ``frq_threshold=2`` is the location-importance setting (cases that contain
    an activity occurring only once in the log are dropped); ``1`` is what
    ``Classical_Permutation.py`` uses (no case is dropped).

    Returns ``(CaseLog, data_manager)``.
    """
    data_manager_class = load_data_manager_class()
    data_manager = data_manager_class(str(DATASETS[dataset]), 2, None, frq_threshold,
                                      L_max_perc=0.8)
    return CaseLog.from_data_manager(data_manager), data_manager


def apriori_itemsets(data_manager, min_support: float = 0.5, top_k: int = 10):
    """Top-k frequent activity sets of size > 1, exactly as the 2024 code picks them.

    Returns ``(itemsets, supports)``: a list of alphabetically sorted tuples (in
    the original support order) and the matching list of supports.  Sorting the
    activities inside a set only makes the labels reproducible (Python's set
    order changes between processes).
    """
    _, selected = data_manager.frequent_activity_sets(min_support, int(top_k))
    itemsets = [tuple(sorted(itemset)) for itemset in selected["itemsets"]]
    return itemsets, selected["support"].tolist()


def itemset_label(itemset) -> str:
    """Readable, reproducible column label for an activity set."""
    return "{" + ", ".join(itemset) + "}"


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #
def occurrence_count(trace, activity: str, matching: str = "exact") -> int:
    """Number of occurrences of ``activity`` in ``trace`` (a list of names)."""
    if matching == "exact":
        return trace.count(activity)             # list.count -> whole tokens
    return "->".join(trace).count(activity)      # original: substring count


def encode_existence(traces: pd.Series, itemsets, encoding: str = "count",
                     matching: str = "exact") -> pd.DataFrame:
    """Encode every case as one integer feature per activity set.

    ``count``  : min over the set's activities of the occurrence count
                 (0 as soon as one activity is missing).
    ``binary`` : 1 if that min is > 0, i.e. the whole set occurs in the case.

    Columns are 0..n-1 (as in the original); the row index is the case id.
    """
    if encoding not in ENCODINGS:
        raise ValueError("encoding must be one of %s" % (ENCODINGS,))
    if matching not in MATCHINGS:
        raise ValueError("matching must be one of %s" % (MATCHINGS,))
    rows = []
    for trace in traces:
        counts = [min(occurrence_count(trace, activity, matching) for activity in itemset)
                  for itemset in itemsets]
        rows.append(counts)
    encoded = pd.DataFrame(rows, index=traces.index, columns=range(len(itemsets)), dtype=int)
    if encoding == "binary":
        encoded = (encoded > 0).astype(int)
    return encoded


# --------------------------------------------------------------------------- #
# The baseline itself
# --------------------------------------------------------------------------- #
@dataclass
class ExistenceResult:
    """Output of :meth:`ExistenceImportance.run`.

    Attributes
    ----------
    importances : pd.DataFrame
        One row per (fold, repeat), one column per activity set; each value is
        ``baseline score - score after permuting that column``.
    fold_scores : pd.DataFrame
        Per fold: accuracy and weighted F1 of the model on its train and test
        part, and the share of test cases it predicts as label 1.
    """

    importances: pd.DataFrame
    fold_scores: pd.DataFrame

    def mean_importance(self) -> pd.Series:
        """Mean importance per activity set over all folds and repeats."""
        return self.importances.mean()

    def ranking(self) -> pd.Series:
        """Rank per activity set (1 = most important; ties share the best rank)."""
        return self.mean_importance().rank(ascending=False, method="min").astype(int)


class ExistenceImportance:
    """Classical permutation importance of activity-set existence features.

    For every fold of a seeded stratified K-fold split: fit a default
    ``XGBClassifier`` on the existence features of the training cases (the
    activity-set columns are the *only* features, as in the original script),
    then call ``sklearn.inspection.permutation_importance`` on the training or
    the held-out cases.

    Parameters
    ----------
    encoding, matching, scoring, score_on : str
        See the module docstring.  The original script corresponds to
        ``('count', 'substring', 'accuracy', 'train')``.
    n_splits : int
        Number of CV folds (original: 5).
    n_repeats : int
        Permutations per feature and fold (original: 20).
    fold_seed : int
        ``random_state`` of ``StratifiedKFold`` (the original has none).
    permutation_seed : int
        ``random_state`` of ``permutation_importance`` (original: 42).
    model_seed : int
        ``random_state`` of the classifier.
    """

    def __init__(self, encoding: str = "count", scoring: str = "f1_weighted",
                 matching: str = "exact", score_on: str = "test", n_splits: int = 5,
                 n_repeats: int = 20, fold_seed: int = 2023, permutation_seed: int = 42,
                 model_seed: int = 0):
        if scoring not in SCORINGS:
            raise ValueError("scoring must be one of %s" % (SCORINGS,))
        if score_on not in SCORE_ON:
            raise ValueError("score_on must be one of %s" % (SCORE_ON,))
        self.encoding = encoding
        self.scoring = scoring
        self.matching = matching
        self.score_on = score_on
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.fold_seed = fold_seed
        self.permutation_seed = permutation_seed
        self.model_seed = model_seed

    def encode(self, traces: pd.Series, itemsets) -> pd.DataFrame:
        """Existence features of all cases (see :func:`encode_existence`)."""
        return encode_existence(traces, itemsets, self.encoding, self.matching)

    def run(self, log: CaseLog, itemsets) -> ExistenceResult:
        """Compute the existence importance of ``itemsets`` on ``log``."""
        features = self.encode(log.traces, itemsets)
        labels = log.labels.loc[features.index]
        splitter = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                                   random_state=self.fold_seed)
        names = [itemset_label(itemset) for itemset in itemsets]

        importances, fold_scores = [], []
        for fold, (train_idx, test_idx) in enumerate(splitter.split(features, labels)):
            train_x, train_y = features.iloc[train_idx], labels.iloc[train_idx]
            test_x, test_y = features.iloc[test_idx], labels.iloc[test_idx]

            model = XGBClassifier(random_state=self.model_seed, n_jobs=1)
            model.fit(train_x, train_y)
            fold_scores.append(self._fold_scores(fold, model, train_x, train_y, test_x, test_y))

            score_x, score_y = (train_x, train_y) if self.score_on == "train" else (test_x, test_y)
            result = permutation_importance(model, score_x, score_y, scoring=self.scoring,
                                            n_repeats=self.n_repeats,
                                            random_state=self.permutation_seed)
            fold_frame = pd.DataFrame(result.importances.T, columns=names)
            fold_frame.index = pd.MultiIndex.from_product([[fold], range(self.n_repeats)],
                                                          names=["fold", "repeat"])
            importances.append(fold_frame)

        return ExistenceResult(importances=pd.concat(importances),
                               fold_scores=pd.DataFrame(fold_scores).set_index("fold"))

    @staticmethod
    def _fold_scores(fold, model, train_x, train_y, test_x, test_y) -> dict:
        """Accuracy / weighted F1 on train and test, and share of test cases predicted 1."""
        train_pred, test_pred = model.predict(train_x), model.predict(test_x)
        return {
            "fold": fold,
            "train_accuracy": accuracy_score(train_y, train_pred),
            "train_f1_weighted": f1_score(train_y, train_pred, average="weighted"),
            "test_accuracy": accuracy_score(test_y, test_pred),
            "test_f1_weighted": f1_score(test_y, test_pred, average="weighted"),
            "test_share_predicted_1": float(test_pred.mean()),
        }
