"""Experiment C11: paretoset 1.2.0 (the 2023 repo's pin) versus the installed 1.2.5.

The script has two modes.

Driver (default)
    python experiments/c11_paretoset_versions.py
    Starts one worker process per paretoset version, compares the two result
    files key by key and writes ``results/experiments/C11/c11_results.json``
    and ``results/experiments/C11/c11_table.md``.

Worker (started by the driver)
    python experiments/c11_paretoset_versions.py --worker --out FILE [--prepend-path DIR]
    Runs every hand-made case with whichever ``paretoset`` is imported first.
    ``--prepend-path`` puts a ``pip install --target`` folder in front of
    ``sys.path`` so that the old version shadows the installed one.

paretoset 1.2.0 must be installed once into the throw-away folder:
    python -m pip install paretoset==1.2.0 --no-deps \
        --target results/experiments/C11/tmp/paretoset_1_2_0
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results" / "experiments" / "C11"
OLD_VERSION_DIR = RESULT_DIR / "tmp" / "paretoset_1_2_0"
SEED = 2023
NAN = float("nan")

# Column order used by IMPresseD: outcome interest (information gain),
# frequency interest (case coverage), case distance.
OBJECTIVES = ["Outcome_Interest", "Frequency_Interest", "Case_Distance_Interest"]

SENSE_ROWS = {
    "P1": (0.10, 0.6, 0.30),
    "P2": (0.25, 0.3, 0.30),
    "P3": (0.05, 0.3, 0.35),  # dominated by P1 and by P2
    "P4": (0.30, 0.1, 0.50),  # best information gain
    "P5": (0.20, 0.2, 0.20),  # best case distance
    "P6": (0.20, 0.2, 0.40),  # dominated by P5
}
SENSE_SPELLINGS = {
    "lower": ["max", "max", "min"],
    "Capitalised": ["Max", "Max", "Min"],
    "UPPER": ["MAX", "MAX", "MIN"],
}

DUPLICATE_CASES = {
    "D1_guide_example": {
        "rows": {"a": (1, 1), "b": (1, 1), "c": (2, 0), "d": (0, 2)},
        "sense": ["max", "max"],
    },
    "D2_three_objectives": {
        "rows": {
            "A": (0.2, 0.5, 0.3),
            "B": (0.2, 0.5, 0.3),  # duplicate of A (on the front)
            "C": (0.1, 0.9, 0.3),
            "D": (0.1, 0.4, 0.4),  # dominated by A
            "E": (0.1, 0.4, 0.4),  # duplicate of D (dominated)
            "F": (0.2, 0.5, 0.3),  # third copy of A, not adjacent
        },
        "sense": ["max", "max", "min"],
    },
    "D3_all_identical": {
        "rows": {"a": (0.3, 0.3, 0.3), "b": (0.3, 0.3, 0.3), "c": (0.3, 0.3, 0.3), "d": (0.3, 0.3, 0.3)},
        "sense": ["max", "max", "min"],
    },
    "D4_tie_on_two_of_three": {
        "rows": {"a": (0.2, 0.5, 0.3), "b": (0.2, 0.5, 0.4)},  # b is worse on the third only
        "sense": ["max", "max", "min"],
    },
}

NAN_CASES = {
    "N1_nan_first_objective": {
        "rows": {"a": (1, 1), "b": (NAN, 0), "c": (2, 2)},
        "sense": ["min", "min"],
    },
    "N2_nan_second_objective": {
        "rows": {"a": (1, 1), "b": (0, NAN), "c": (2, 2)},
        "sense": ["min", "min"],
    },
    "N3_all_nan_row": {
        "rows": {"a": (NAN, NAN), "b": (1, 1), "c": (2, 2)},
        "sense": ["min", "min"],
    },
    "N4_pattern_in_all_cases": {
        # p4 occurs in every case: coverage 1.0, case distance undefined (NaN).
        "rows": {
            "p1": (0.10, 0.6, 0.30),
            "p2": (0.25, 0.3, 0.30),
            "p3": (0.05, 0.3, 0.35),
            "p4": (0.00, 1.0, NAN),
            "p5": (0.02, 0.9, 0.50),
        },
        "sense": ["max", "max", "min"],
    },
    "N5_nan_row_worse_elsewhere": {
        # q2 is worse than q1 on both objectives that are not NaN.
        "rows": {"q1": (0.3, 0.5, 0.2), "q2": (0.1, 0.4, NAN), "q3": (0.2, 0.45, 0.6)},
        "sense": ["max", "max", "min"],
    },
    "N6_nan_row_hides_better_distance": {
        # r_all occurs in every case (case distance NaN). r_x and r_y have a
        # real, small case distance and belong on the front.
        "rows": {"r_all": (0.20, 1.0, NAN), "r_x": (0.20, 0.8, 0.10), "r_y": (0.05, 0.5, 0.05)},
        "sense": ["max", "max", "min"],
    },
}


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------
def reference_front(costs, sense):
    """Brute-force Pareto front (textbook definition, keeps duplicates).

    Row ``i`` is dominated when some row ``j`` is no worse on every objective
    and strictly better on at least one. Only valid for NaN-free input.
    """
    import numpy as np

    signs = np.array([-1.0 if s.lower() == "max" else 1.0 for s in sense])
    values = np.asarray(costs, dtype=float) * signs  # everything becomes "minimise"
    keep = np.ones(len(values), dtype=bool)
    for i, row in enumerate(values):
        no_worse = (values <= row).all(axis=1)
        better = (values < row).any(axis=1)
        keep[i] = not (no_worse & better).any()
    return keep


def first_occurrence_only(mask, costs):
    """Reduce a front mask to the first row of every group of identical rows."""
    import numpy as np

    seen, reduced = set(), np.zeros(len(mask), dtype=bool)
    for i, row in enumerate(map(tuple, np.asarray(costs, dtype=float))):
        if mask[i] and row not in seen:
            seen.add(row)
            reduced[i] = True
    return reduced


def call(function, *args, **kwargs):
    """Call ``function`` and return a JSON-friendly result or the error text."""
    try:
        result = function(*args, **kwargs)
    except Exception as error:  # noqa: BLE001 - the error itself is the finding
        return f"ERROR {type(error).__name__}: {str(error)[:120]}"
    return [bool(x) if result.dtype == bool else float(x) for x in result]


def kept_labels(labels, mask):
    """Names of the rows a mask keeps, or the error text."""
    if isinstance(mask, str):
        return mask
    return [label for label, keep in zip(labels, mask) if keep]


def run_sense_cases(paretoset, results):
    """Mixed max/min senses written as 'max', 'Max' and 'MAX'."""
    import numpy as np
    import pandas as pd

    labels = list(SENSE_ROWS)
    frame = pd.DataFrame(list(SENSE_ROWS.values()), index=labels, columns=OBJECTIVES)
    expected = reference_front(frame.to_numpy(), SENSE_SPELLINGS["lower"])
    results["S_expected_front(brute force)"] = kept_labels(labels, expected)
    for name, sense in SENSE_SPELLINGS.items():
        for kind, costs in (("DataFrame", frame), ("ndarray", frame.to_numpy())):
            mask = call(paretoset, costs, sense=sense, distinct=False)
            results[f"S_{name}_{kind}"] = kept_labels(labels, mask)
    results["S_sense_None(all min)"] = kept_labels(labels, call(paretoset, frame, distinct=False))

    # Does the call change the caller's data (max columns are negated internally)?
    array = frame.to_numpy()
    before_frame, before_array = frame.copy(), array.copy()
    paretoset(frame, sense=SENSE_SPELLINGS["lower"])
    paretoset(array, sense=SENSE_SPELLINGS["lower"])
    results["S_input_left_unchanged"] = bool(frame.equals(before_frame) and np.array_equal(array, before_array))


def run_duplicate_cases(paretoset, paretorank, results):
    """Identical objective rows with distinct=True and distinct=False."""
    import pandas as pd

    for case_id, case in DUPLICATE_CASES.items():
        labels = list(case["rows"])
        frame = pd.DataFrame(list(case["rows"].values()), index=labels)
        for distinct in (True, False):
            for use_numba in (True, False):
                mask = call(paretoset, frame, sense=case["sense"], distinct=distinct, use_numba=use_numba)
                results[f"{case_id}|distinct={distinct}|numba={use_numba}"] = kept_labels(labels, mask)
            ranks = call(paretorank, frame, sense=case["sense"], distinct=distinct)
            if not isinstance(ranks, str):
                ranks = {label: int(rank) for label, rank in zip(labels, ranks)}
            results[f"{case_id}|paretorank|distinct={distinct}"] = ranks


def run_nan_cases(paretoset, results):
    """Rows containing NaN, in every row order, plus two ways of removing the NaN first."""
    import numpy as np
    import pandas as pd

    for case_id, case in NAN_CASES.items():
        labels = list(case["rows"])
        frame = pd.DataFrame(list(case["rows"].values()), index=labels, dtype=float)
        for distinct in (True, False):
            for use_numba in (True, False):
                key = f"{case_id}|distinct={distinct}|numba={use_numba}"
                mask = call(paretoset, frame, sense=case["sense"], distinct=distinct, use_numba=use_numba)
                results[key] = kept_labels(labels, mask)
                # Same rows in every possible order: is the kept SET always the same?
                outcomes = set()
                for order in itertools.permutations(labels):
                    shuffled = frame.loc[list(order)]
                    mask = call(paretoset, shuffled, sense=case["sense"], distinct=distinct, use_numba=use_numba)
                    kept = kept_labels(order, mask)
                    outcomes.add(kept if isinstance(kept, str) else ",".join(sorted(kept)))
                results[key + "|all_row_orders"] = sorted(outcomes)

        # Remedies applied BEFORE the call (NaN only occurs in minimised columns here).
        complete = frame.dropna()
        mask = call(paretoset, complete, sense=case["sense"], distinct=False)
        results[f"{case_id}|remedy=drop_nan_rows"] = kept_labels(list(complete.index), mask)
        for name, worst in (("fill_1.0", 1.0), ("fill_inf", np.inf)):
            if all(s == "min" for s in case["sense"]) and name == "fill_1.0":
                continue  # 1.0 is only "worst" for the [0, 1] interest functions
            filled = frame.fillna(worst)
            mask = call(paretoset, filled, sense=case["sense"], distinct=False)
            agrees = bool(np.array_equal(mask, reference_front(filled.to_numpy(), case["sense"])))
            results[f"{case_id}|remedy={name}"] = kept_labels(labels, mask)
            results[f"{case_id}|remedy={name}|equals_brute_force"] = agrees


def run_random_cases(paretoset, paretorank, results):
    """Seeded random data with many ties, checked against the brute-force front."""
    import numpy as np

    rng = np.random.default_rng(SEED)
    sense = ["max", "max", "min"]
    for decimals, n_rows in ((1, 500), (2, 2000)):
        costs = np.round(rng.random((n_rows, 3)), decimals)
        reference = reference_front(costs, sense)
        n_duplicate_rows = int(n_rows - len(np.unique(costs, axis=0)))
        tag = f"X_random_{n_rows}x3_round{decimals}"
        results[f"{tag}|duplicate_rows"] = n_duplicate_rows
        results[f"{tag}|brute_force_front_size"] = int(reference.sum())
        for use_numba in (True, False):
            keep_all = paretoset(costs, sense=sense, distinct=False, use_numba=use_numba)
            keep_first = paretoset(costs, sense=sense, distinct=True, use_numba=use_numba)
            results[f"{tag}|numba={use_numba}|distinct=False|front_size"] = int(keep_all.sum())
            results[f"{tag}|numba={use_numba}|distinct=False|equals_brute_force"] = bool(
                np.array_equal(keep_all, reference))
            results[f"{tag}|numba={use_numba}|distinct=True|front_size"] = int(keep_first.sum())
            results[f"{tag}|numba={use_numba}|distinct=True|equals_first_occurrence_of_brute_force"] = bool(
                np.array_equal(keep_first, first_occurrence_only(reference, costs)))
            results[f"{tag}|numba={use_numba}|mask_sha1"] = hashlib.sha1(
                keep_all.tobytes() + keep_first.tobytes()).hexdigest()
        for distinct in (True, False):
            ranks = paretorank(costs, sense=sense, distinct=distinct)
            results[f"{tag}|paretorank|distinct={distinct}|n_layers"] = int(ranks.max())
            results[f"{tag}|paretorank|distinct={distinct}|layer1_equals_paretoset"] = bool(
                np.array_equal(ranks == 1, paretoset(costs, sense=sense, distinct=distinct)))
            results[f"{tag}|paretorank|distinct={distinct}|ranks_sha1"] = hashlib.sha1(
                ranks.astype("int64").tobytes()).hexdigest()


def run_unused_feature_cases(results):
    """The only two code paths that differ between 1.2.0 and 1.2.5 (we use neither)."""
    import numpy as np
    import pandas as pd
    from paretoset import paretoset
    from paretoset.algorithms_numpy import crowding_distance

    frame = pd.DataFrame({"group": ["g1", "g1", "g2", "g2"], "cost": [1.0, 2.0, 3.0, 4.0]})
    results["V_sense_diff_with_DataFrame"] = call(paretoset, frame, sense=["diff", "min"])
    results["V_crowding_distance"] = call(crowding_distance, np.array([[1.0], [3.0], [5.0], [9.0], [11.0]]))


def run_worker(out_path, prepend_path):
    """Run all cases with the paretoset version that is first on sys.path."""
    if prepend_path:
        sys.path.insert(0, str(Path(prepend_path).resolve()))
    import numba
    import numpy as np
    import pandas as pd
    import paretoset as paretoset_package
    from paretoset import paretorank, paretoset

    results = {}
    start = time.perf_counter()
    paretoset(np.array([[1.0, 2.0], [2.0, 1.0]]))
    first_call_seconds = time.perf_counter() - start  # includes the numba compilation

    run_sense_cases(paretoset, results)
    run_duplicate_cases(paretoset, paretorank, results)
    run_nan_cases(paretoset, results)
    run_random_cases(paretoset, paretorank, results)
    run_unused_feature_cases(results)

    environment = {
        "paretoset": paretoset_package.__version__,
        "paretoset_file": str(Path(paretoset_package.__file__).parent),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "numba": numba.__version__,
        "first_call_seconds(numba compile)": round(first_call_seconds, 2),
        "all_cases_seconds": round(time.perf_counter() - start, 2),
    }
    Path(out_path).write_text(json.dumps({"environment": environment, "results": results}, indent=1))


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def start_worker(out_path, prepend_path=None):
    """Run this script in worker mode in a fresh process and load its result file."""
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--out", str(out_path)]
    if prepend_path:
        command += ["--prepend-path", str(prepend_path)]
    start = time.perf_counter()
    done = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(f"worker {out_path.name}: exit code {done.returncode}, {time.perf_counter() - start:.1f} s")
    if done.returncode != 0:
        raise RuntimeError(done.stderr[-2000:])
    return json.loads(out_path.read_text())


def markdown_table(old, new):
    """One row per case: result under 1.2.0, result under 1.2.5, equal or not."""
    lines = ["| case | paretoset 1.2.0 | paretoset 1.2.5 | same |", "|---|---|---|---|"]
    for key in new["results"]:
        left, right = old["results"].get(key, "<missing>"), new["results"][key]
        lines.append(f"| `{key}` | {json.dumps(left)} | {json.dumps(right)} | {'yes' if left == right else '**NO**'} |")
    return "\n".join(lines) + "\n"


def run_driver():
    """Run both versions, compare, write the JSON and the markdown table."""
    if not OLD_VERSION_DIR.exists():
        raise SystemExit(f"paretoset 1.2.0 is not installed in {OLD_VERSION_DIR}; see the module docstring.")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    old = start_worker(RESULT_DIR / "worker_paretoset_1_2_0.json", OLD_VERSION_DIR)
    new = start_worker(RESULT_DIR / "worker_paretoset_1_2_5.json")
    if old["environment"]["paretoset"] == new["environment"]["paretoset"]:
        raise SystemExit("both workers imported the same paretoset version - comparison is meaningless")

    different = [key for key in new["results"] if old["results"].get(key) != new["results"][key]]
    summary = {
        "environment_old": old["environment"],
        "environment_new": new["environment"],
        "n_cases": len(new["results"]),
        "n_different": len(different),
        "different_keys": different,
    }
    (RESULT_DIR / "c11_results.json").write_text(json.dumps(summary, indent=1))
    (RESULT_DIR / "c11_table.md").write_text(markdown_table(old, new), encoding="utf-8")
    print(json.dumps(summary, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--worker", action="store_true", help="run the cases in this process")
    parser.add_argument("--out", help="worker: JSON file to write")
    parser.add_argument("--prepend-path", help="worker: folder to put first on sys.path")
    args = parser.parse_args()
    if args.worker:
        run_worker(args.out, args.prepend_path)
    else:
        run_driver()


if __name__ == "__main__":
    main()
