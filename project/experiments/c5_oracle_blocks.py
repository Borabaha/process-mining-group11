"""Experiment C5a: what the original IMPresseD trace conversion does with ``delta_time = 0``.

Run from anywhere (about 1 minute for the three logs):
    python experiments/c5_oracle_blocks.py

Background.  The 2023 code turns a trace into a graph with
``IMIPD.Trace_graph_generator`` (IMIPD.py:331-400): two CONSECUTIVE events whose
timestamps differ by at most ``delta_time`` seconds are flagged ``parallel``
(IMIPD.py:347), and a maximal run of such events becomes one concurrent block.
The BPIC11 timestamps are day-level, so with ``delta_time = 0`` every run of
events of one case on one day is one block.  This script measures how large
these blocks are on BPIC11 f1, f2, f3:

* share of consecutive event pairs with equal timestamps;
* distribution of the block sizes (a "block" of size 1 is an ordinary
  sequential event);
* share of events that sit in blocks of at least 10 events;
* as a cross-check, the trace graphs of the ORIGINAL ``Trace_graph_generator``
  with ``delta_time = 0``: nodes flagged parallel and number of edges, compared
  with the chain (``delta_time < 0``);
* how many nodes the patterns of the original direct-preceding, direct-following
  and concurrent rules would have on these graphs (the node sets are built as in
  ``IMIPD.Pattern_extension``; the pattern dictionary itself is not built).

Outputs in ``results/experiments/C5/``: ``c5a_blocks.json``, ``c5a_tables.md``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # IMIPD imports pyplot; never open a window

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
ORIGINAL_DIR = PROJECT_ROOT / "external" / "InteractivePatternDetection"
RESULT_DIR = PROJECT_ROOT / "results" / "experiments" / "C5"

sys.path.insert(0, str(EXPERIMENT_DIR))

from c4_run_selector import markdown_table  # noqa: E402
from engine import ACTIVITY_COL, CASE_COL, DATASETS, POSITION_COL, EventLog  # noqa: E402

TIME_COL = "time:timestamp"
BIG_BLOCK = 10
# Block-size classes for the distribution table: (label, smallest size, largest size).
SIZE_BINS = [("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-5", 4, 5), ("6-9", 6, 9),
             ("10-19", 10, 19), ("20+", 20, 10**9)]


def load_events(log_name: str, filtered: bool) -> pd.DataFrame:
    """Events of one log, sorted by case and ``event_nr``, with parsed timestamps.

    ``filtered=True`` keeps only the cases that survive the rare-activity
    filter of the 2024 ``DataManager`` (the cases of ``engine.EventLog``).
    """
    events = pd.read_csv(DATASETS[log_name], usecols=[CASE_COL, ACTIVITY_COL, TIME_COL, POSITION_COL])
    events[CASE_COL] = events[CASE_COL].astype(str)
    if filtered:
        events = events[events[CASE_COL].isin(set(EventLog(DATASETS[log_name]).case_ids))]
    events = events.sort_values([CASE_COL, POSITION_COL], kind="stable").reset_index(drop=True)
    events["time"] = pd.to_datetime(events[TIME_COL], format="mixed")
    return events


def block_ids(events: pd.DataFrame) -> pd.Series:
    """Number the same-timestamp blocks: a new block starts at a new case or a new timestamp.

    Equivalent to the original rule with ``delta_time = 0``: consecutive events
    are in the same block exactly when their timestamps are equal.
    """
    new_case = events[CASE_COL] != events[CASE_COL].shift()
    new_time = events["time"] != events["time"].shift()
    return (new_case | new_time).cumsum()


def pair_statistics(events: pd.DataFrame) -> dict:
    """Consecutive event pairs inside a case: how many have equal / decreasing timestamps."""
    difference = events.groupby(CASE_COL, sort=False)["time"].diff().dropna()
    return {
        "n_cases": int(events[CASE_COL].nunique()),
        "n_events": int(len(events)),
        "n_consecutive_pairs": int(len(difference)),
        "n_equal_timestamp_pairs": int((difference == pd.Timedelta(0)).sum()),
        "share_equal_timestamp_pairs": float((difference == pd.Timedelta(0)).mean()),
        "n_decreasing_timestamp_pairs": int((difference < pd.Timedelta(0)).sum()),
        "distinct_time_of_day_values": sorted(events["time"].dt.strftime("%H:%M").unique().tolist()),
    }


def block_statistics(events: pd.DataFrame) -> dict:
    """Sizes of the same-timestamp blocks and the share of events in large blocks."""
    blocks = block_ids(events)
    sizes = blocks.map(blocks.value_counts())           # per event: size of its block
    block_sizes = blocks.value_counts().to_numpy()      # per block: its size
    n_events = len(events)
    parallel_events = int((sizes >= 2).sum())
    big_events = int((sizes >= BIG_BLOCK).sum())
    blocks_per_case = events.assign(block=blocks).groupby(CASE_COL, sort=False)["block"].nunique()
    trace_length = events.groupby(CASE_COL, sort=False).size()

    distribution = []
    for label, low, high in SIZE_BINS:
        in_bin = (block_sizes >= low) & (block_sizes <= high)
        distribution.append({
            "block size": label, "blocks": int(in_bin.sum()),
            "share of blocks": float(in_bin.mean()),
            "events": int(block_sizes[in_bin].sum()),
            "share of events": float(block_sizes[in_bin].sum() / n_events),
        })
    return {
        "n_blocks": int(len(block_sizes)),
        "block_size_mean": float(block_sizes.mean()),
        "block_size_median": float(np.median(block_sizes)),
        "block_size_p90": float(np.quantile(block_sizes, 0.9)),
        "block_size_max": int(block_sizes.max()),
        "event_weighted_median_block_size": float(sizes.median()),
        "n_events_in_blocks_ge_2": parallel_events,
        "share_events_in_blocks_ge_2": parallel_events / n_events,
        f"n_events_in_blocks_ge_{BIG_BLOCK}": big_events,
        f"share_events_in_blocks_ge_{BIG_BLOCK}": big_events / n_events,
        f"share_of_parallel_events_in_blocks_ge_{BIG_BLOCK}": big_events / parallel_events,
        "blocks_per_case_median": float(blocks_per_case.median()),
        "trace_length_median": float(trace_length.median()),
        "n_cases_that_are_one_single_block": int(((blocks_per_case == 1) & (trace_length > 1)).sum()),
        f"n_cases_with_a_block_ge_{BIG_BLOCK}": int(events.loc[sizes >= BIG_BLOCK, CASE_COL].nunique()),
        "distribution": distribution,
    }


def step1_pattern_sizes(graph) -> list[tuple[str, int, int]]:
    """Sizes of the patterns the original step-1 extension builds around every node of one trace graph.

    Same node sets as ``IMIPD.Pattern_extension`` (IMIPD.py:181-242), without
    the expensive dictionary update: for a node ``n``

    * direct preceding:  all predecessors of ``n`` plus ``n``   (IMIPD.py:185-190);
    * direct following:  all successors of ``n`` plus ``n``     (IMIPD.py:202-207);
    * concurrent:        ``n`` plus every node flagged parallel that has the
      same predecessors and successors, only if ``n`` is parallel (IMIPD.py:219-231).

    Returns one ``(rule, number of nodes, number of distinct activities)`` per
    pattern that would be built.
    """
    value = {node: data["value"] for node, data in graph.nodes(data=True)}
    neighbours = {node: (frozenset(graph.pred[node]), frozenset(graph.succ[node])) for node in graph.nodes}
    same_context: dict[tuple, set] = {}
    for node, data in graph.nodes(data=True):
        if data["parallel"]:
            same_context.setdefault(neighbours[node], set()).add(node)

    sizes = []
    for node in graph.nodes:
        before, after = neighbours[node]
        members = {"direct preceding": set(before) | {node} if before else set(),
                   "direct following": set(after) | {node} if after else set(),
                   "concurrent": same_context[neighbours[node]] if graph.nodes[node]["parallel"] else set()}
        for rule, nodes in members.items():
            if nodes:
                sizes.append((rule, len(nodes), len({value[member] for member in nodes})))
    return sizes


def original_graph_statistics(events: pd.DataFrame, delta_time: float) -> dict:
    """Build every trace graph with the ORIGINAL ``Trace_graph_generator`` and summarise it.

    The 2023 repository is put on ``sys.path`` only here (it ships a module
    named ``tools.py``, like the 2024 repository).
    """
    if str(ORIGINAL_DIR) not in sys.path:
        sys.path.insert(0, str(ORIGINAL_DIR))
    import IMIPD  # noqa: E402  (the original code, unchanged)

    data = events[[CASE_COL, ACTIVITY_COL, "time"]]
    colours = dict.fromkeys(data[ACTIVITY_COL].unique(), "#000000")
    blocks = block_ids(events)
    # Expected parallel flag: member of a block of >= 2 events (delta 0); never for a negative delta.
    expected_parallel = (blocks.map(blocks.value_counts()) >= 2) & (delta_time >= 0)
    n_nodes = n_parallel = n_edges = n_chain_edges = max_out_degree = n_flag_mismatch = n_no_edges = 0
    pattern_sizes: list[tuple[str, int, int]] = []
    start = time.perf_counter()
    for case, case_events in data.groupby(CASE_COL, sort=False):
        graph = IMIPD.Trace_graph_generator(data, delta_time, case, colours, CASE_COL, ACTIVITY_COL, "time")
        flags = [graph.nodes[node]["parallel"] for node in range(len(case_events))]
        n_flag_mismatch += int(flags != expected_parallel.loc[case_events.index].tolist())
        n_nodes += graph.number_of_nodes()
        n_parallel += sum(flags)
        n_edges += graph.number_of_edges()
        n_no_edges += int(graph.number_of_edges() == 0 and graph.number_of_nodes() > 1)
        n_chain_edges += len(case_events) - 1
        max_out_degree = max(max_out_degree, max(dict(graph.out_degree()).values()))
        pattern_sizes.extend(step1_pattern_sizes(graph))

    sizes = pd.DataFrame(pattern_sizes, columns=["rule", "n_nodes", "n_activities"])
    per_rule = []
    for rule, group in sizes.groupby("rule", sort=True):
        per_rule.append({
            "rule": rule, "pattern instances": int(len(group)),
            "median nodes": float(group["n_nodes"].median()), "largest": int(group["n_nodes"].max()),
            "share with > 3 nodes": float((group["n_nodes"] > 3).mean()),
            "share with > 3 distinct activities": float((group["n_activities"] > 3).mean()),
            "share with >= 10 nodes": float((group["n_nodes"] >= BIG_BLOCK).mean()),
        })
    return {
        "delta_time": delta_time, "n_nodes": int(n_nodes), "n_nodes_flagged_parallel": int(n_parallel),
        "share_nodes_flagged_parallel": n_parallel / n_nodes,
        "n_edges": int(n_edges), "n_edges_of_the_chains": int(n_chain_edges),
        "edges_per_chain_edge": n_edges / n_chain_edges, "max_out_degree": int(max_out_degree),
        "n_graphs_without_any_edge": int(n_no_edges),
        "cases_where_parallel_flags_differ_from_block_rule": int(n_flag_mismatch),
        "step1_pattern_sizes": per_rule,
        "seconds": time.perf_counter() - start,
    }


def run_log(log_name: str, with_original: bool) -> dict:
    """All C5a measurements for one log."""
    raw, events = load_events(log_name, filtered=False), load_events(log_name, filtered=True)
    record = {"log": log_name, "raw_log": pair_statistics(raw), "filtered_log": pair_statistics(events),
              "blocks_filtered_log": block_statistics(events)}
    if with_original:
        record["original_graphs_delta_0"] = original_graph_statistics(events, delta_time=0)
        record["original_graphs_delta_minus_1"] = original_graph_statistics(events, delta_time=-1)
    print(f"[{log_name}] equal-timestamp pairs {record['filtered_log']['share_equal_timestamp_pairs']:.4f}, "
          f"events in blocks >= {BIG_BLOCK}: "
          f"{record['blocks_filtered_log'][f'share_events_in_blocks_ge_{BIG_BLOCK}']:.4f}", flush=True)
    return record


def write_tables(records: list[dict]) -> None:
    """Write ``c5a_tables.md`` with the tables used in RESULT.md."""
    pairs, blocks, graphs, rules = [], [], [], []
    parts = ["# C5a tables (generated by c5_oracle_blocks.py)\n"]
    for record in records:
        raw, kept, stats = record["raw_log"], record["filtered_log"], record["blocks_filtered_log"]
        pairs.append({
            "log": record["log"], "cases (raw / filtered)": f"{raw['n_cases']} / {kept['n_cases']}",
            "events (filtered)": kept["n_events"], "consecutive pairs": kept["n_consecutive_pairs"],
            "pairs with equal timestamp": kept["n_equal_timestamp_pairs"],
            "share (filtered)": kept["share_equal_timestamp_pairs"],
            "share (raw log)": raw["share_equal_timestamp_pairs"],
            "pairs with decreasing timestamp": kept["n_decreasing_timestamp_pairs"],
        })
        blocks.append({
            "log": record["log"], "blocks": stats["n_blocks"], "mean size": stats["block_size_mean"],
            "median size": stats["block_size_median"], "90 % quantile": stats["block_size_p90"],
            "largest": stats["block_size_max"],
            "median size seen by an event": stats["event_weighted_median_block_size"],
            "events in blocks >= 2": stats["share_events_in_blocks_ge_2"],
            f"events in blocks >= {BIG_BLOCK}": stats[f"share_events_in_blocks_ge_{BIG_BLOCK}"],
            f"parallel events in blocks >= {BIG_BLOCK}":
                stats[f"share_of_parallel_events_in_blocks_ge_{BIG_BLOCK}"],
            f"cases with a block >= {BIG_BLOCK}": stats[f"n_cases_with_a_block_ge_{BIG_BLOCK}"],
            "median blocks per case": stats["blocks_per_case_median"],
            "median trace length": stats["trace_length_median"],
        })
        for key in ("original_graphs_delta_0", "original_graphs_delta_minus_1"):
            if key in record:
                summary = {name: value for name, value in record[key].items() if name != "step1_pattern_sizes"}
                graphs.append({"log": record["log"], **summary})
                rules += [{"log": record["log"], "delta_time": record[key]["delta_time"], **row}
                          for row in record[key]["step1_pattern_sizes"]]
    parts += ["## Consecutive event pairs\n", markdown_table(pd.DataFrame(pairs)),
              "\n## Same-timestamp blocks (filtered log)\n", markdown_table(pd.DataFrame(blocks))]
    for record in records:
        parts += [f"\n## Block-size distribution, {record['log']}\n",
                  markdown_table(pd.DataFrame(record["blocks_filtered_log"]["distribution"]))]
    if graphs:
        parts += ["\n## Trace graphs of the original Trace_graph_generator\n", markdown_table(pd.DataFrame(graphs)),
                  "\n## Size of the step-1 patterns the original extension rules would build (every event as core)\n",
                  markdown_table(pd.DataFrame(rules))]
    (RESULT_DIR / "c5a_tables.md").write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--logs", nargs="+", default=["f1", "f2", "f3"], choices=sorted(DATASETS))
    parser.add_argument("--skip-original", action="store_true",
                        help="do not build the trace graphs with the original 2023 code")
    args = parser.parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    records = [run_log(log_name, not args.skip_original) for log_name in args.logs]
    with open(RESULT_DIR / "c5a_blocks.json", "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=1)
    write_tables(records)
    print(f"total {time.perf_counter() - start:.1f} s; results in {RESULT_DIR}")


if __name__ == "__main__":
    main()
