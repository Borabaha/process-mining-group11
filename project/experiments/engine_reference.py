"""Access to the ORIGINAL 2024 code as a reference for engine.py (experiment C2).

The original repository (``external/PermutationLocationImportance``) is
read-only and is loaded here under the module name ``tools_2024`` so that it
cannot be confused with the ``tools.py`` of the 2023 repository. The only
change is the int cast that pandas >= 2 needs (``pd.get_dummies`` returns
bool there), applied in a subclass exactly like ``scripts/scout/smoke_2024c.py``.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import warnings
from pathlib import Path

import pandas as pd

ORIGINAL_REPO = (
    Path(__file__).resolve().parents[1] / "external" / "PermutationLocationImportance"
)


def load_original_tools():
    """Import the original ``tools.py`` as module ``tools_2024`` (cached)."""
    if "tools_2024" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "tools_2024", ORIGINAL_REPO / "tools.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["tools_2024"] = module
        spec.loader.exec_module(module)
    return sys.modules["tools_2024"]


def make_original_manager(csv_path):
    """The original ``DataManager(path, 2, None, L_max_perc=0.8)`` + int cast."""
    tools = load_original_tools()

    class DataManagerIntFeatures(tools.DataManager):
        """pandas >= 2 workaround: cast the one-hot feature columns to int."""

        def index_encoding(self, data):
            encoded = super().index_encoding(data)
            keys = (self.case_id, self.outcome)
            features = [col for col in encoded.columns if col not in keys]
            encoded[features] = encoded[features].astype(int)
            return encoded

    return DataManagerIntFeatures(str(csv_path), 2, None, L_max_perc=0.8)


@contextlib.contextmanager
def quiet():
    """Silence the original's per-iteration prints and pandas warnings."""
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)
        warnings.simplefilter("ignore", FutureWarning)
        yield


def original_encoding(manager) -> pd.DataFrame:
    """``index_encoding`` of the whole log (features + case id + label)."""
    with quiet():
        return manager.index_encoding(manager.data)


def original_fold_data(manager, encoded: pd.DataFrame, cases):
    """Feature frame and label list of ``cases``, as CVPP.py:77-80 builds them."""
    part = encoded[encoded[manager.case_id].isin(cases)]
    labels = part[manager.outcome].tolist()
    return part.drop([manager.case_id, manager.outcome], axis=1), labels


def original_importance(manager, model, features, labels, cases, itemsets, n_repeats):
    """Run the original ``itemset_permutation_importance`` (seed 2023).

    ``itemsets`` is a dict id -> list of activities. Returns the original
    result frame: one column per itemset id, one row per repeat.
    """
    with quiet():
        return manager.itemset_permutation_importance(
            model, features, labels, cases, itemsets, constrain=True,
            n_repeats=n_repeats,
        )


def original_single_importance(manager, model, features, labels, cases, n_repeats):
    """Run the original ``trace_permutation_importance`` (seed 2023).

    This is the single-activity routine (CrossValidation_ProcessPermutation.py
    with ``Multi_activity`` False). It always processes every activity of the
    log. Returns the original result frame: one column per activity, one row
    per repeat.
    """
    with quiet():
        return manager.trace_permutation_importance(
            model, features, labels, cases, constrain=True, n_repeats=n_repeats,
        )
