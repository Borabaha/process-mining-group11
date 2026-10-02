"""Experiment C4, part (g): run the chain-only IMPresseD selector on BPIC11 f1, f2, f3.

Run from anywhere (about 1 minute for the three logs):
    python experiments/c4_run_selector.py

Per log it writes to ``results/experiments/C4/``:

* ``impressed_sets_<log>.json``    - the selected activity sets per length with
  their interest values, source pattern and step, plus front sizes and timings;
* ``impressed_patterns_<log>.csv`` - every evaluated pattern;
* ``impressed_all_sets_<log>.csv`` - every candidate activity set with rank and layer;

and one ``selector_summary.md`` with the tables used in ``RESULT.md``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from apriori_selector import AprioriSelector  # noqa: E402
from case_distance import (  # noqa: E402
    BPIC11_CATEGORICAL_COLS,
    BPIC11_NUMERIC_COLS,
    build_case_attribute_table,
    pairwise_case_distance,
)
from engine import CASE_COL, DATASETS, POSITION_COL, EventLog  # noqa: E402
from impressed_chain import ImpressedChainSelector, count_instances  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results" / "experiments" / "C4"
N_RECOUNT_CHECK = 300  # patterns per log whose counts are re-derived by a plain scan
SEED = 2023
# Working defaults for the Apriori baseline (Guide Part 6, Table B); only used for the overlap count.
APRIORI_MIN_SUPPORT = {"f1": 0.5, "f2": 0.5, "f3": 0.4}


def load_selector_inputs(log_name: str):
    """Return ``(log, dist_matrix, case_table)`` for one BPIC11 log.

    ``log`` is the engine's :class:`EventLog` (same cleaning and rare-activity
    filter as the 2024 ``DataManager``).  The case attributes are re-read from
    the raw CSV (the loader drops them) and the distance matrix is in the order
    of ``log.case_ids``.
    """
    log = EventLog(DATASETS[log_name])
    raw = pd.read_csv(DATASETS[log_name])
    raw[CASE_COL] = raw[CASE_COL].astype(str)
    attributes = BPIC11_NUMERIC_COLS + BPIC11_CATEGORICAL_COLS
    case_table = build_case_attribute_table(raw, attributes, CASE_COL, POSITION_COL).loc[log.case_ids]
    dist_matrix = pairwise_case_distance(case_table, BPIC11_NUMERIC_COLS, BPIC11_CATEGORICAL_COLS)
    return log, dist_matrix, case_table


def recount_check(selector: ImpressedChainSelector, traces, n_patterns: int, seed: int) -> dict:
    """Re-derive the per-case counts of random evaluated patterns with ``count_instances``."""
    patterns = list(selector.counts_)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(patterns), size=min(n_patterns, len(patterns)), replace=False)
    trace_list = [traces[case] for case in selector.case_ids_]
    n_different = 0
    for index in chosen:
        pattern = patterns[index]
        scanned = np.array([count_instances(pattern, trace, selector.max_gap) for trace in trace_list])
        n_different += int(not np.array_equal(scanned, selector.counts_[pattern]))
    return {"patterns_checked": int(len(chosen)), "patterns_with_different_counts": n_different}


def set_support(itemset, traces) -> float:
    """Share of cases whose trace contains ALL activities of the set (the Apriori support)."""
    wanted = set(itemset)
    return sum(wanted <= set(trace) for trace in traces.values()) / len(traces)


def apriori_overlap(log_name: str, traces, selection: dict, k: int) -> dict:
    """Per size: how many of our sets are also among Apriori's ``k`` most frequent sets."""
    apriori = AprioriSelector(APRIORI_MIN_SUPPORT[log_name], max_len=max(int(s) for s in selection), top_k=k)
    frequent = apriori.select(traces)
    return {size: len({tuple(items) for items in sets} & {tuple(items) for items in frequent.get(int(size), [])})
            for size, sets in selection.items()}


def set_records(table: pd.DataFrame, traces) -> dict:
    """Selected sets as JSON-friendly records, grouped by size."""
    records: dict[str, list] = {}
    for row in table.to_dict("records"):
        records.setdefault(str(row["size"]), []).append({
            "activities": list(row["itemset"]), "rank": int(row["rank"]), "layer": int(row["layer"]),
            "IG": float(row["IG"]), "coverage": float(row["coverage"]), "CD": float(row["CD"]),
            "set_support": set_support(row["itemset"], traces),
            "source_pattern": row["source_pattern"],
            "source_pattern_labels": list(row["pattern"].labels),
            "source_pattern_edges": list(row["pattern"].edges),
            "step": int(row["step"]), "n_patterns_for_set": int(row["n_patterns"]),
        })
    return records


def run_log(log_name: str, selector_kwargs: dict) -> dict:
    """Run the selector on one log, write its files and return the summary record."""
    start = time.perf_counter()
    log, dist_matrix, _ = load_selector_inputs(log_name)
    load_seconds = time.perf_counter() - start

    start = time.perf_counter()
    selector = ImpressedChainSelector(**selector_kwargs).fit(log.traces, log.labels, dist_matrix)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    all_sets = selector.full_table()
    projection_seconds = time.perf_counter() - start
    selected = all_sets[all_sets["rank"] <= selector.k]
    first_front = all_sets[all_sets["layer"] == 1]

    front_patterns = selector.patterns_.loc[selector.patterns_["on_front"], "pattern"]
    front_only_sets = {pattern.activity_set for pattern in front_patterns}

    sizes = selector.lengths
    summary = {
        "log": log_name, "n_cases": len(log.traces), "n_activities": len(log.activities),
        "settings": dict(selector_kwargs),
        "seconds": {"load_log_and_distance": load_seconds, "fit": fit_seconds, "projection": projection_seconds},
        "steps": selector.step_summary_.to_dict("records"),
        "n_patterns_evaluated": int(len(selector.patterns_)),
        "n_candidate_sets": {str(s): int((all_sets["size"] == s).sum()) for s in sizes},
        "first_front_size": {str(s): int((first_front["size"] == s).sum()) for s in sizes},
        "layers_used": {str(s): int(selected.loc[selected["size"] == s, "layer"].max())
                        for s in sizes if (selected["size"] == s).any()},
        "n_candidate_sets_front_patterns_only": {str(s): sum(len(items) == s for items in front_only_sets)
                                                 for s in sizes},
        "recount_check": recount_check(selector, log.traces, N_RECOUNT_CHECK, SEED),
        "selection": {str(s): [list(items) for items in selected.loc[selected["size"] == s, "itemset"]]
                      for s in sizes},
        "sets": set_records(selected, log.traces),
    }
    summary["sets_shared_with_apriori_top_k"] = apriori_overlap(log_name, log.traces, summary["selection"],
                                                                selector.k)
    summary["apriori_min_support"] = APRIORI_MIN_SUPPORT[log_name]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_DIR / f"impressed_sets_{log_name}.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1)
    selector.patterns_.drop(columns="pattern").to_csv(RESULT_DIR / f"impressed_patterns_{log_name}.csv", index=False)
    all_sets.drop(columns="pattern").to_csv(RESULT_DIR / f"impressed_all_sets_{log_name}.csv", index=False)
    print(f"[{log_name}] load {load_seconds:.1f} s, fit {fit_seconds:.1f} s, projection {projection_seconds:.1f} s; "
          f"steps {summary['steps']}", flush=True)
    return summary


def markdown_table(frame: pd.DataFrame) -> str:
    """Small pipe table (floats with 4 decimals, tiny non-zero floats in scientific notation)."""
    def cell(value):
        if not isinstance(value, float):
            return str(value)
        return f"{value:.1e}" if 0 < abs(value) < 1e-4 else f"{value:.4f}"
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def write_summary(summaries: list[dict]) -> None:
    """Write ``selector_summary.md``: steps, sets per length and the selected sets."""
    parts = ["# C4 selector runs (generated by c4_run_selector.py)\n"]
    step_rows, length_rows = [], []
    for summary in summaries:
        for step in summary["steps"]:
            step_rows.append({"log": summary["log"], **step})
        for size in summary["n_candidate_sets"]:
            length_rows.append({
                "log": summary["log"], "length": size,
                "candidate sets": summary["n_candidate_sets"][size],
                "first front": summary["first_front_size"][size],
                "selected": len(summary["sets"].get(size, [])),
                "layers used": summary["layers_used"].get(size, 0),
                "sets if only front patterns": summary["n_candidate_sets_front_patterns_only"][size],
                "selected sets with pattern in < 3 cases": sum(
                    round(r["coverage"] * summary["n_cases"]) < 3 for r in summary["sets"].get(size, [])),
                "median pattern coverage": float(np.median([r["coverage"] for r in summary["sets"].get(size, [])])),
                "median set support": float(np.median([r["set_support"] for r in summary["sets"].get(size, [])])),
                "also in Apriori top k": summary["sets_shared_with_apriori_top_k"][size],
            })
    parts += ["## Steps\n", markdown_table(pd.DataFrame(step_rows)), "\n## Sets per length\n",
              markdown_table(pd.DataFrame(length_rows)), "\n## Runtime (seconds)\n",
              markdown_table(pd.DataFrame([{"log": s["log"], **s["seconds"],
                                            "patterns evaluated": s["n_patterns_evaluated"],
                                            **s["recount_check"]} for s in summaries]))]
    for summary in summaries:
        parts.append(f"\n## Selected sets, {summary['log']}\n")
        rows = [{"length": size, "rank": r["rank"], "layer": r["layer"], "activities": ", ".join(r["activities"]),
                 "IG": r["IG"], "coverage": r["coverage"], "CD": r["CD"], "set support": r["set_support"],
                 "source pattern": r["source_pattern"], "step": r["step"]}
                for size, records in summary["sets"].items() for r in records]
        parts.append(markdown_table(pd.DataFrame(rows)))
    (RESULT_DIR / "selector_summary.md").write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--logs", nargs="+", default=["f1", "f2", "f3"], choices=sorted(DATASETS))
    parser.add_argument("--max-gap", type=int, default=3)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    selector_kwargs = {"max_gap": args.max_gap, "steps": args.steps, "k": args.k,
                       "objectives": ["IG", "coverage", "CD"], "distinct": False}
    start = time.perf_counter()
    summaries = [run_log(log_name, selector_kwargs) for log_name in args.logs]
    write_summary(summaries)
    print(f"total {time.perf_counter() - start:.1f} s; results in {RESULT_DIR}")


if __name__ == "__main__":
    main()
