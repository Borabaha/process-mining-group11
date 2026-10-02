"""Experiment C12, worker: run the ORIGINAL 2023 case-distance code under one scipy version.

Started by ``c12_compare_case_distance.py`` once per scipy version:

    python experiments/c12_scipy_worker.py --in-dir DIR --out FILE.npz --numeric-cols Age [--prepend-path DIR]

``--prepend-path`` puts a ``pip install --target`` folder (scipy 1.11.4 +
numpy 1.26.4) in front of ``sys.path`` so that it shadows the installed scipy.
The 2023 repository is put on ``sys.path`` as well, because the worker calls
its functions ``calculate_pairwise_case_distance`` and
``similarity_measuring_patterns`` unchanged (IMIPD.py; the GUI is not imported).

Input files (written by the driver into ``--in-dir``)
    attributes_<name>.csv        one row per case, index = case id, case attributes
    patterns_sample200_f1.csv    per-case pattern counts for the 200-case sample

Output: one ``.npz`` with condensed distance vectors (scipy ``pdist`` order).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_2023 = PROJECT_ROOT / "external" / "InteractivePatternDetection"
TINY_PAIRS = {
    "pair_A [[0,1,2],[0,1,3]]": [[0, 1, 2], [0, 1, 3]],
    "pair_B [[0,1,2],[1,1,3]]": [[0, 1, 2], [1, 1, 3]],
    "pair_C [[0,1,2],[1,1,2]]": [[0, 1, 2], [1, 1, 2]],
}


def label_codes(table, categorical_cols):
    """Integer codes per column, identical to sklearn's LabelEncoder (sorted categories)."""
    import numpy as np

    return np.column_stack([np.unique(table[col].to_numpy(), return_inverse=True)[1] for col in categorical_cols])


def pair_index(n_cases):
    """Pair list and row offsets exactly as built in GUI_IMPresseD_tool.py:404-411."""
    pair_cases = [(a, b) for a in range(n_cases) for b in range(a + 1, n_cases)]
    start_search_points, i = [], 0
    for k in range(n_cases):
        start_search_points.append(k * n_cases - (i + k))
        i += k
    return pair_cases, start_search_points


def original_pattern_distances(similarity_measuring_patterns, pattern_counts, distances):
    """Case_Distance_Interest of every pattern column with the original function."""
    import pandas as pd

    patient_data = pattern_counts.reset_index(drop=True)
    patterns_data = pd.DataFrame({"patterns": list(patient_data.columns)})
    pair_cases, start_search_points = pair_index(len(patient_data))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scored = similarity_measuring_patterns(patterns_data, patient_data, pair_cases, start_search_points, distances)
    warning_texts = sorted({f"{w.category.__name__}: {w.message}" for w in caught})
    return scored["Case_Distance_Interest"].to_numpy(dtype=float), warning_texts


def run(in_dir, out_path, numeric_cols, prepend_path):
    """Compute all distance vectors with the scipy that is first on sys.path."""
    sys.dont_write_bytecode = True  # the 2023 repository is read-only: no __pycache__ in it
    sys.path.insert(0, str(REPO_2023))
    if prepend_path:
        sys.path.insert(0, str(Path(prepend_path).resolve()))
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    from scipy.spatial.distance import pdist

    from IMIPD import calculate_pairwise_case_distance, similarity_measuring_patterns  # 2023 code, unchanged

    arrays, timings = {}, {}
    tiny = {name: {metric: float(pdist(rows, metric)[0]) for metric in ("jaccard", "hamming")}
            for name, rows in TINY_PAIRS.items()}

    for path in sorted(Path(in_dir).glob("attributes_*.csv")):
        name = path.stem.replace("attributes_", "")
        table = pd.read_csv(path, index_col=0)
        categorical_cols = [c for c in table.columns if c not in numeric_cols]
        start = time.perf_counter()
        codes = label_codes(table, categorical_cols)
        arrays[f"jaccard_on_codes|{name}"] = pdist(codes, "jaccard")
        arrays[f"hamming_on_codes|{name}"] = pdist(codes, "hamming")
        # The original function label-encodes its argument in place, hence the copies.
        arrays[f"original_combined|{name}"] = calculate_pairwise_case_distance(table.copy(), numeric_cols)
        arrays[f"original_numeric|{name}"] = calculate_pairwise_case_distance(table[numeric_cols].copy(), numeric_cols)
        arrays[f"original_categorical|{name}"] = calculate_pairwise_case_distance(table[categorical_cols].copy(), [])
        timings[name] = round(time.perf_counter() - start, 2)

    pattern_counts = pd.read_csv(Path(in_dir) / "patterns_sample200_f1.csv", index_col=0)
    start = time.perf_counter()
    values, warning_texts = original_pattern_distances(
        similarity_measuring_patterns, pattern_counts, arrays["original_combined|sample200_f1"])
    arrays["original_pattern_distance|sample200_f1"] = values
    timings["original similarity_measuring_patterns (sample)"] = round(time.perf_counter() - start, 2)

    info = {
        "scipy": scipy.__version__,
        "scipy_file": str(Path(scipy.__file__).parent),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "python": sys.version.split()[0],
        "tiny_pairs": tiny,
        "pattern_names": list(pattern_counts.columns),
        "original_pattern_warnings": warning_texts,
        "seconds": timings,
    }
    np.savez_compressed(out_path, info=json.dumps(info), **arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--numeric-cols", nargs="+", required=True)
    parser.add_argument("--prepend-path")
    args = parser.parse_args()
    run(args.in_dir, args.out, args.numeric_cols, args.prepend_path)


if __name__ == "__main__":
    main()
