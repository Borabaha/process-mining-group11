"""Experiment C7 - Apriori per-size counts vs ``min_support`` and a per-size selector.

For BPIC11 f1, f2, f3 this script

1. loads each log through the ORIGINAL ``DataManager`` (same filtering and
   normalisation as the 2024 code) and checks the case / activity counts;
2. checks that :func:`apriori_selector.original_top10` returns the same itemsets as
   the original ``DataManager.frequent_activity_sets`` (at several supports) and
   shows where the original top-10 is decided by ties;
3. sweeps ``min_support`` and counts the frequent activity sets of size 1, 2, 3
   (``max_len=3``), and counts what the original, uncapped Apriori mines;
4. recommends, per log, the largest grid support with >= ``TOP_K`` sets for every
   size and saves the resulting top-``TOP_K``-per-size lists as JSON;
5. re-counts everything by brute force (no mlxtend) as an independent check.

Run from anywhere (all paths are resolved from this file)::

    python experiments/run_c7_apriori_sweep.py

Outputs go to ``results/experiments/C7/``. Nothing here is random (no seed needed).
"""
from __future__ import annotations

import itertools
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_REPO = PROJECT_ROOT / "external" / "PermutationLocationImportance"
DATASET_DIR = ORIGINAL_REPO / "datasets"
OUT_DIR = PROJECT_ROOT / "results" / "experiments" / "C7"

# The original repo is only imported (never edited); its tools.py must be first on sys.path.
sys.path.insert(0, str(ORIGINAL_REPO))
from tools import DataManager  # noqa: E402  (original 2024 code)

from apriori_selector import AprioriSelector, original_top10  # noqa: E402

# log name -> (file, expected cases, expected activities after the rare-activity filter)
LOGS = {
    "f1": ("BPIC11_f1_trunc36.csv", 1130, 164),
    "f2": ("BPIC11_f2_trunc40.csv", 1130, 207),
    "f3": ("BPIC11_f3_trunc31.csv", 1111, 156),
}
SIZES = (1, 2, 3)
MAX_LEN = 3
TOP_K = 10
# Sweep of the per-size selector (Apriori capped at MAX_LEN).
SUPPORT_GRID = (0.5, 0.49, 0.45, 0.4, 0.35, 0.3)
TIMING_REPEATS = 3
# Supports at which the replica is compared with the original code.
ORIGINAL_SUPPORTS = (0.55, 0.53, 0.52, 0.51, 0.5, 0.49, 0.48, 0.47)
# Supports at which the uncapped (original) Apriori is counted. The number of sets
# explodes below ~0.47 on f1/f2 (minutes, gigabytes), hence the short lists there.
UNCAPPED_GRID = {
    "f1": (0.5, 0.49, 0.47, 0.45),
    "f2": (0.5, 0.49, 0.47),
    "f3": (0.5, 0.49, 0.47, 0.45, 0.4),
}
F3_LISTING_SUPPORTS = (0.5, 0.49)

LOG = logging.getLogger("c7")


def setup_logging() -> None:
    """Log to the console and to ``results/experiments/C7/run.log``."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout),
                logging.FileHandler(OUT_DIR / "run.log", mode="w", encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers)


def braces(itemset) -> str:
    """Format an activity set as ``{a, b, c}`` with sorted names."""
    return "{" + ", ".join(sorted(itemset)) + "}"


def load_original(log_name: str) -> tuple[DataManager, dict[str, list[str]]]:
    """Load one log with the original DataManager and return it with its traces.

    The traces are ``{case_id: [activity, ...]}`` in ``event_nr`` order, taken from the
    filtered, normalised ``DataManager.data`` (already sorted by case, event_nr).
    """
    file_name, n_cases, n_activities = LOGS[log_name]
    manager = DataManager(str(DATASET_DIR / file_name), 2, None, L_max_perc=0.8)
    data = manager.data
    traces = data.groupby(manager.case_id, sort=False)[manager.activity].apply(list).to_dict()
    found = (len(traces), data[manager.activity].nunique())
    expected = (n_cases, n_activities)
    assert found == expected, f"{log_name}: expected {expected}, got {found}"
    return manager, traces


def compare_with_original(manager: DataManager, traces: dict, min_support: float) -> dict:
    """Compare ``original_top10`` with ``DataManager.frequent_activity_sets``."""
    start = time.perf_counter()
    theirs_lists, theirs_table = manager.frequent_activity_sets(min_support, TOP_K)
    seconds_original = time.perf_counter() - start
    start = time.perf_counter()
    ours_lists, ours_table = original_top10(traces, min_support, TOP_K)
    seconds_replica = time.perf_counter() - start
    theirs_sets = [frozenset(x) for x in theirs_lists]
    ours_sets = [frozenset(x) for x in ours_lists]
    return {
        "min_support": min_support,
        "n_itemsets_original": len(theirs_lists),
        "n_itemsets_replica": len(ours_lists),
        "same_itemsets_same_rank": theirs_sets == ours_sets,
        "same_supports": theirs_table["support"].tolist() == ours_table["support"].tolist(),
        "same_inner_activity_order": theirs_lists == ours_lists,
        "seconds_original": round(seconds_original, 3),
        "seconds_replica": round(seconds_replica, 3),
        "itemsets": [{"itemset": sorted(items), "support": float(support)}
                     for items, support in zip(ours_lists, ours_table["support"])],
    }


def timed_full_table(traces: dict, min_support: float, max_len: int | None,
                     repeats: int) -> tuple[pd.DataFrame, float]:
    """Mine the full table ``repeats`` times; return it with the best wall time."""
    selector = AprioriSelector(min_support, max_len=max_len, top_k=TOP_K)
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        table = selector.full_table(traces)
        best = min(best, time.perf_counter() - start)
    return table, best


def sweep_log(log_name: str, traces: dict) -> tuple[list[dict], dict[float, pd.DataFrame]]:
    """Count the frequent sets per size (Apriori capped at MAX_LEN) for every grid support."""
    rows, tables = [], {}
    for min_support in SUPPORT_GRID:
        table, seconds = timed_full_table(traces, min_support, MAX_LEN, TIMING_REPEATS)
        tables[min_support] = table
        row = {"log": log_name, "min_support": min_support}
        for size in SIZES:
            of_size = table[table["size"] == size]
            row[f"n_size{size}"] = len(of_size)
            # support of the TOP_K-th set of this size (None when fewer than TOP_K exist)
            row[f"support_10th_size{size}"] = (
                float(of_size["support"].iloc[TOP_K - 1]) if len(of_size) >= TOP_K else None)
        row["enough_for_every_size"] = all(row[f"n_size{size}"] >= TOP_K for size in SIZES)
        row[f"seconds_best_of_{TIMING_REPEATS}"] = round(seconds, 3)
        rows.append(row)
    return rows, tables


def uncapped_counts(log_name: str, traces: dict) -> tuple[list[dict], dict[float, pd.DataFrame]]:
    """Count frequent sets WITHOUT a size cap (what the original code mines)."""
    rows, tables = [], {}
    for min_support in UNCAPPED_GRID[log_name]:
        table, seconds = timed_full_table(traces, min_support, None, repeats=1)
        tables[min_support] = table
        sizes = table["size"].value_counts().sort_index()
        rows.append({
            "log": log_name, "min_support": min_support, "n_total": len(table),
            "n_size_gt1": int((table["size"] > 1).sum()),
            "max_size": int(table["size"].max()) if len(table) else 0,
            "counts_by_size": "/".join(str(int(n)) for n in sizes),
            "seconds": round(seconds, 3),
        })
    return rows, tables


def brute_force_case_counts(traces: dict) -> dict[int, list[int]]:
    """Count, WITHOUT mlxtend, the cases containing each candidate set of size 1-3.

    Independent check of the Apriori numbers: candidates are all combinations of the
    activities whose own support reaches the lowest grid support (a set cannot be
    more frequent than its members). Returns ``{size: [case counts, descending]}``.
    """
    activities = sorted({activity for trace in traces.values() for activity in trace})
    contains = np.array([[activity in set(trace) for activity in activities]
                         for trace in traces.values()])
    frequent = np.flatnonzero(contains.mean(axis=0) >= min(SUPPORT_GRID))
    return {
        size: sorted((int(contains[:, list(combo)].all(axis=1).sum())
                      for combo in itertools.combinations(frequent, size)), reverse=True)
        for size in SIZES
    }


def check_against_brute_force(traces: dict, sweep_rows: list[dict],
                              selected: pd.DataFrame) -> None:
    """Assert that the sweep counts and the selected supports match the brute-force counts."""
    counts = brute_force_case_counts(traces)
    n_cases = len(traces)
    for row in sweep_rows:
        for size in SIZES:
            expected = sum(count / n_cases >= row["min_support"] for count in counts[size])
            assert row[f"n_size{size}"] == expected, (row, size, expected)
    for size in SIZES:
        chosen = selected.loc[selected["size"] == size, "count"].tolist()
        assert chosen == counts[size][:TOP_K], (size, chosen, counts[size][:TOP_K])


def recommend_support(rows: list[dict]) -> float | None:
    """Largest grid support with >= TOP_K sets for every size (None if no grid value works)."""
    good = [row["min_support"] for row in rows if row["enough_for_every_size"]]
    return max(good) if good else None


def rank10_boundary(pool: pd.DataFrame, n_cases: int) -> dict | None:
    """Describe the rank-``TOP_K`` boundary of a pool of candidate sets.

    ``threshold_support`` is the support of the TOP_K-th most frequent set: any
    ``min_support`` at or below it yields >= TOP_K sets. ``sets_above`` have a strictly
    higher support (always selected); ``free_slots`` = TOP_K - sets_above are filled
    from ``tied_sets`` (the sets with exactly the threshold support). When there are
    more tied sets than free slots, a tie-break decides which sets are selected.
    """
    if len(pool) < TOP_K:
        return None
    counts = pool["count"].sort_values(ascending=False)
    count_10th = int(counts.iloc[TOP_K - 1])
    sets_above = int((pool["count"] > count_10th).sum())
    tied = pool.loc[pool["count"] == count_10th, "itemset"]
    return {
        "threshold_count": count_10th,
        "threshold_support": count_10th / n_cases,
        "sets_above": sets_above,
        "free_slots": TOP_K - sets_above,
        "tied_sets": [list(itemset) for itemset in tied],
    }


def tie_choices(checks: list[dict], boundary: dict | None) -> list[dict]:
    """Per support: which of the tied sets the original code put into its top-``TOP_K``."""
    if boundary is None:
        return []
    tied = {frozenset(itemset) for itemset in boundary["tied_sets"]}
    rows = []
    for check in checks:
        if check["n_itemsets_original"] < TOP_K:
            continue
        chosen = sorted(braces(entry["itemset"]) for entry in check["itemsets"]
                        if frozenset(entry["itemset"]) in tied)
        rows.append({"min_support": check["min_support"], "tied_sets_chosen": "; ".join(chosen)})
    return rows


def sets_payload(log_name: str, min_support: float, selected: pd.DataFrame) -> dict:
    """Build the JSON document with the top-``TOP_K`` sets per size and their supports."""
    file_name, n_cases, n_activities = LOGS[log_name]
    return {
        "log": log_name,
        "dataset_file": file_name,
        "n_cases": n_cases,
        "n_activities": n_activities,
        "min_support": min_support,
        "max_len": MAX_LEN,
        "top_k": TOP_K,
        "ranking": ("per size: support descending, ties broken lexicographically "
                    "on sorted activity names"),
        "transactions": ("set of distinct activities per case (whole log, after the "
                         "original rare-activity filter)"),
        "sets": {
            str(size): [
                {"rank": int(row["rank"]), "itemset": list(row["itemset"]),
                 "support": float(row["support"]), "count": int(row["count"])}
                for _, row in selected[selected["size"] == size].iterrows()
            ]
            for size in SIZES
        },
    }


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table (no extra dependency)."""
    def cell(value) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "-"
        if isinstance(value, float):
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return str(value)

    header = "| " + " | ".join(str(col) for col in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(cell(value) for value in row) + " |"
            for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *body])


def f3_listing(uncapped_tables: dict[float, pd.DataFrame]) -> list[dict]:
    """List every f3 set of size > 1 at the listing supports, most frequent first."""
    rows = []
    for min_support in F3_LISTING_SUPPORTS:
        table = uncapped_tables[min_support]
        larger = table[table["size"] > 1].sort_values(
            ["count", "itemset"], ascending=[False, True], kind="mergesort")
        for rank, row in enumerate(larger.itertuples(index=False), start=1):
            rows.append({"min_support": min_support, "rank": rank,
                         "itemset": braces(row.itemset), "size": row.size,
                         "support": row.support, "count": row.count})
    return rows


def run_log(log_name: str) -> dict:
    """Run every step of the experiment for one log; return its tables and summary."""
    start = time.perf_counter()
    manager, traces = load_original(log_name)
    n_cases = len(traces)
    LOG.info("\n=== %s: %d cases, %d activities (original DataManager, %.1f s)",
             log_name, n_cases, LOGS[log_name][2], time.perf_counter() - start)

    # (1) replica of the original selection vs the original code
    checks = [compare_with_original(manager, traces, s) for s in ORIGINAL_SUPPORTS]
    for check in checks:
        LOG.info("  original vs replica @%.2f: %2d vs %2d itemsets | same sets+rank=%s"
                 " | same supports=%s | same inner order=%s | %.2f s vs %.2f s",
                 check["min_support"], check["n_itemsets_original"],
                 check["n_itemsets_replica"], check["same_itemsets_same_rank"],
                 check["same_supports"], check["same_inner_activity_order"],
                 check["seconds_original"], check["seconds_replica"])
        assert check["same_itemsets_same_rank"] and check["same_supports"], \
            "replica differs from original"

    # (2) what the original (uncapped) Apriori mines; where its top-10 is decided by ties
    uncapped_rows, uncapped_tables = uncapped_counts(log_name, traces)
    lowest_uncapped = uncapped_tables[min(UNCAPPED_GRID[log_name])]
    original_boundary = rank10_boundary(lowest_uncapped[lowest_uncapped["size"] > 1], n_cases)
    tie_rows = [{"log": log_name, **row} for row in tie_choices(checks, original_boundary)]

    # (3) sweep with max_len = 3
    sweep_rows, tables = sweep_log(log_name, traces)
    recommended = recommend_support(sweep_rows)
    lowest = tables[min(SUPPORT_GRID)]
    per_size_boundary = {size: rank10_boundary(lowest[lowest["size"] == size], n_cases)
                         for size in SIZES}

    # (4) top-k per size at the recommended support, and its stability at lower supports
    selector = AprioriSelector(recommended, max_len=MAX_LEN, top_k=TOP_K)
    selected = selector.select_table(traces)
    selection = selector.select(traces)
    stable = all(AprioriSelector(s, max_len=MAX_LEN, top_k=TOP_K).select(traces) == selection
                 for s in SUPPORT_GRID if s <= recommended)
    distinct_activities = {size: len({act for itemset in selection[size] for act in itemset})
                           for size in SIZES}
    payload = sets_payload(log_name, recommended, selected)
    out_file = OUT_DIR / f"apriori_sets_{log_name}.json"
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOG.info("  recommended min_support = %s | same top-%d lists at every lower grid support:"
             " %s | distinct activities per size: %s",
             recommended, TOP_K, stable, distinct_activities)

    # (5) independent check of the counts and of the selected supports (no mlxtend)
    start = time.perf_counter()
    check_against_brute_force(traces, sweep_rows, selected)
    LOG.info("  brute-force check of all sweep counts and of the selected supports: passed"
             " (%.1f s)", time.perf_counter() - start)

    summary = {
        "n_cases": n_cases,
        "n_activities": LOGS[log_name][2],
        "recommended_min_support": recommended,
        "selection_stable_at_lower_supports": stable,
        "distinct_activities_in_selection_per_size": distinct_activities,
        "per_size_rank10_boundary": per_size_boundary,
        "original_rank10_boundary_size_gt1": original_boundary,
        "original_vs_replica": checks,
    }
    return {"sweep": sweep_rows, "uncapped": uncapped_rows, "ties": tie_rows,
            "f3_listing": f3_listing(uncapped_tables) if log_name == "f3" else [],
            "summary": summary}


def log_boundaries(summary: dict) -> None:
    """Log the rank-10 boundary of the original pool and of every per-size pool."""
    for log_name, entry in summary.items():
        boundary = entry["original_rank10_boundary_size_gt1"]
        LOG.info("\n--- %s original pool (size > 1): rank-10 support %.4f (%d cases), "
                 "%d sets above, %d free slot(s), %d tied set(s): %s",
                 log_name, boundary["threshold_support"], boundary["threshold_count"],
                 boundary["sets_above"], boundary["free_slots"], len(boundary["tied_sets"]),
                 "; ".join(braces(s) for s in boundary["tied_sets"]))
        for size, per_size in entry["per_size_rank10_boundary"].items():
            LOG.info("    per-size pool, size %s: rank-10 support %.4f (%d cases), %d above, "
                     "%d free slot(s), %d tied set(s): %s",
                     size, per_size["threshold_support"], per_size["threshold_count"],
                     per_size["sets_above"], per_size["free_slots"],
                     len(per_size["tied_sets"]),
                     "; ".join(braces(s) for s in per_size["tied_sets"]))


def main() -> None:
    """Run the whole experiment and write the outputs."""
    setup_logging()
    started = time.perf_counter()
    results = {log_name: run_log(log_name) for log_name in LOGS}

    def collect(key: str) -> pd.DataFrame:
        return pd.DataFrame([row for result in results.values() for row in result[key]])

    sweep, uncapped, ties, f3_sets = (
        collect(key) for key in ("sweep", "uncapped", "ties", "f3_listing"))
    sweep.to_csv(OUT_DIR / "sweep_counts.csv", index=False)
    uncapped.to_csv(OUT_DIR / "uncapped_counts.csv", index=False)
    ties.to_csv(OUT_DIR / "original_tie_choices.csv", index=False)
    f3_sets.to_csv(OUT_DIR / "f3_itemsets_size_gt1.csv", index=False)
    summary = {log_name: result["summary"] for log_name, result in results.items()}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    LOG.info("\n--- sweep (max_len=3) ---\n%s", markdown_table(sweep))
    LOG.info("\n--- uncapped Apriori (original behaviour) ---\n%s", markdown_table(uncapped))
    log_boundaries(summary)
    LOG.info("\n--- tied sets chosen by the ORIGINAL code, per min_support ---\n%s",
             markdown_table(ties))
    LOG.info("\n--- f3 itemsets of size > 1 (uncapped) ---\n%s", markdown_table(f3_sets))
    for log_name in LOGS:
        out_file = OUT_DIR / f"apriori_sets_{log_name}.json"
        payload = json.loads(out_file.read_text(encoding="utf-8"))
        rows = [{"size": int(size), **entry, "itemset": braces(entry["itemset"])}
                for size, entries in payload["sets"].items() for entry in entries]
        LOG.info("\n--- %s top-%d per size at min_support=%s ---\n%s", log_name, TOP_K,
                 payload["min_support"], markdown_table(pd.DataFrame(rows)))
    LOG.info("\nTOTAL %.1f s", time.perf_counter() - started)


if __name__ == "__main__":
    main()
