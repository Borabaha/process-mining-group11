"""Check that two runs of ``run_original_impressed.py`` produced the same result.

Used in experiment C3 to prove that the performance patches in
``experiments/original_impressed_patched`` do not change any output.

Usage::

    .venv/Scripts/python.exe experiments/compare_impressed_runs.py \
        results/experiments/C3/smoke120 results/experiments/C3/smoke120_patched
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

COMPARED_FIELDS = ["id", "labels", "edges", "on_front", "Pattern_Frequency", "Case_Support",
                   "Outcome_Interest", "Frequency_Interest", "Case_Distance_Interest"]


def compare_patterns(run_a: Path, run_b: Path) -> bool:
    """Compare pattern IDs, graphs, interest values and front flags step by step (exact equality)."""
    a = json.loads((run_a / "patterns.json").read_text(encoding="utf-8"))["steps"]
    b = json.loads((run_b / "patterns.json").read_text(encoding="utf-8"))["steps"]
    same = a.keys() == b.keys()
    for step in sorted(set(a) & set(b)):
        rows_a = [[p[f] for f in COMPARED_FIELDS] for p in a[step]["patterns"]]
        rows_b = [[p[f] for f in COMPARED_FIELDS] for p in b[step]["patterns"]]
        equal = rows_a == rows_b
        same &= equal
        print("step %s: %d vs %d candidates, front %s vs %s -> %s"
              % (step, len(rows_a), len(rows_b), a[step]["front_size"], b[step]["front_size"],
                 "IDENTICAL" if equal else "DIFFERENT"))
    return same


def compare_encoded_logs(run_a: Path, run_b: Path) -> bool:
    """Compare the two CSV files the tool itself writes."""
    same = True
    for name in ("training_encoded_log.csv", "testing_encoded_log.csv"):
        frame_a = pd.read_csv(run_a / "impressed_output" / name)
        frame_b = pd.read_csv(run_b / "impressed_output" / name)
        equal = frame_a.equals(frame_b)
        same &= equal
        print("%s: %s vs %s -> %s" % (name, frame_a.shape, frame_b.shape, "IDENTICAL" if equal else "DIFFERENT"))
    return same


def compare_pattern_files(run_a: Path, run_b: Path) -> bool:
    """Compare the per-pattern JSON files (node-link graphs) byte by byte."""
    files_a = {p.name: p.read_bytes() for p in (run_a / "impressed_output").glob("*.json")}
    files_b = {p.name: p.read_bytes() for p in (run_b / "impressed_output").glob("*.json")}
    equal = files_a == files_b
    print("pattern JSON files: %d vs %d -> %s" % (len(files_a), len(files_b), "IDENTICAL" if equal else "DIFFERENT"))
    return equal


def main() -> int:
    run_a, run_b = Path(sys.argv[1]), Path(sys.argv[2])
    results = [compare_patterns(run_a, run_b), compare_encoded_logs(run_a, run_b),
               compare_pattern_files(run_a, run_b)]
    print("OVERALL:", "IDENTICAL" if all(results) else "DIFFERENT")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
