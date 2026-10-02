"""Experiments C5b + C6: how sensitive is the IMPresseD side to its settings?

Run from anywhere (a few minutes for the three logs):
    python -u experiments/c5_c6_sensitivity.py

Uses the verified chain-only selector ``impressed_chain.ImpressedChainSelector``
(experiment C4) on BPIC11 f1, f2, f3 and changes ONE setting at a time around
the working defaults of Guide Part 6, Table B (``max_gap = 3``, objectives
information gain + coverage + case distance, ``distinct=False``, 2 extension
steps, k = 10 sets per length, mining on the whole log):

* C5b - ``max_gap`` in {1, 2, 3, 5}: candidates and front size per step, the
  selected sets per length and the Jaccard overlap of the selections between gaps;
* C6  - three objectives (IG, coverage, CD) versus two (IG, coverage);
  ``distinct=True`` versus ``False``; how many patterns of each step have
  1, 2, 3 or more than 3 distinct activities; how many candidate sets exist per
  length (is k = 10 reachable?);
* an extra variant that rounds the interest values to 12 decimals before every
  Pareto computation (the C4 verification found that mathematically tied
  patterns differ in the last bits, so ``distinct`` never sees them as ties);
* the selected sets next to the Apriori top 10 per size of experiment C7.

Two kinds of comparison are reported for the objectives and for ``distinct``:

* "free run"       - the whole mining is repeated with the changed setting
  (the step fronts decide which patterns are extended, so the candidates change);
* "same candidates" - the patterns of the default run are kept and only the
  front / ranking rule is changed.

Outputs in ``results/experiments/C5/``: ``c5c6_results.json``,
``c5c6_tables.md``, ``sets_vs_apriori.md``.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
RESULT_DIR = PROJECT_ROOT / "results" / "experiments" / "C5"
APRIORI_DIR = PROJECT_ROOT / "results" / "experiments" / "C7"
C4_DIR = PROJECT_ROOT / "results" / "experiments" / "C4"

sys.path.insert(0, str(EXPERIMENT_DIR))

import impressed_chain  # noqa: E402
from c4_run_selector import load_selector_inputs, markdown_table  # noqa: E402
from engine import DATASETS  # noqa: E402
from impressed_chain import EVENTUAL, ImpressedChainSelector, pareto_front  # noqa: E402

THREE_OBJECTIVES = ("IG", "coverage", "CD")
TWO_OBJECTIVES = ("IG", "coverage")
BASE_SETTINGS = {"max_gap": 3, "steps": 2, "objectives": THREE_OBJECTIVES, "distinct": False,
                 "k": 10, "lengths": (1, 2, 3)}
GAPS = (1, 2, 3, 5)
ROUND_DECIMALS = 12
RARE_CASES = 10  # a set / pattern in fewer cases than this is called "rare" in the tables
# name -> (settings that differ from BASE_SETTINGS, decimals to round the interest values to or None)
CONFIGS = {
    "gap1": ({"max_gap": 1}, None),
    "gap2": ({"max_gap": 2}, None),
    "gap3 (default)": ({}, None),
    "gap5": ({"max_gap": 5}, None),
    "two objectives": ({"objectives": TWO_OBJECTIVES}, None),
    "distinct=True": ({"distinct": True}, None),
    "two objectives, distinct=True": ({"objectives": TWO_OBJECTIVES, "distinct": True}, None),
    "rounded": ({}, ROUND_DECIMALS),
    "rounded, distinct=True": ({"distinct": True}, ROUND_DECIMALS),
}
DEFAULT = "gap3 (default)"
GAP_CONFIGS = {1: "gap1", 2: "gap2", 3: DEFAULT, 5: "gap5"}
ACTIVITY_CLASSES = ("1", "2", "3", ">3")


# --------------------------------------------------------------------------
# Running the selector
# --------------------------------------------------------------------------
@contextlib.contextmanager
def rounded_scores(decimals: int | None):
    """Round IG, coverage and CD to ``decimals`` inside ``impressed_chain`` while the block runs.

    ``ImpressedChainSelector.fit`` calls the module-level ``score_patterns``;
    it is wrapped here (and restored afterwards) so that the experiment does
    not need a modified copy of the verified module.  ``None`` changes nothing.
    """
    if decimals is None:
        yield
        return
    original = impressed_chain.score_patterns

    def score_and_round(*args, **kwargs):
        return original(*args, **kwargs).round(decimals)

    impressed_chain.score_patterns = score_and_round
    try:
        yield
    finally:
        impressed_chain.score_patterns = original


def fit_selector(log, dist_matrix, overrides: dict, decimals: int | None) -> tuple[ImpressedChainSelector, float]:
    """Fit one selector with ``BASE_SETTINGS`` changed by ``overrides``; returns it and the seconds."""
    start = time.perf_counter()
    with rounded_scores(decimals):
        selector = ImpressedChainSelector(**{**BASE_SETTINGS, **overrides}).fit(log.traces, log.labels, dist_matrix)
    return selector, time.perf_counter() - start


def with_rule(selector: ImpressedChainSelector, **attributes) -> ImpressedChainSelector:
    """Shallow copy of a fitted selector with other projection settings (same mined patterns)."""
    clone = copy.copy(selector)
    for name, value in attributes.items():
        setattr(clone, name, value)
    return clone


# --------------------------------------------------------------------------
# Summaries of one fitted selector
# --------------------------------------------------------------------------
def activity_class(pattern) -> str:
    """'1', '2', '3' or '>3': the number of distinct activities of a pattern."""
    n_activities = len(pattern.activity_set)
    return str(n_activities) if n_activities <= 3 else ">3"


def step_composition(selector: ImpressedChainSelector) -> list[dict]:
    """Per step: candidates and front patterns by number of distinct activities."""
    rows = []
    for table in selector.step_tables_:
        classes = table["pattern"].map(activity_class)
        row = {"step": int(table["step"].iloc[0]), "candidates": int(len(table)),
               "front": int(table["on_front"].sum()),
               "front extendable": int(sum(p.extendable for p in table.loc[table["on_front"], "pattern"]))}
        for name in ACTIVITY_CLASSES:
            row[f"candidates with {name} activities"] = int((classes == name).sum())
        for name in ACTIVITY_CLASSES:
            row[f"front with {name} activities"] = int(((classes == name) & table["on_front"]).sum())
        rows.append(row)
    return rows


def case_count(itemset, traces) -> int:
    """Number of cases whose trace contains ALL activities of the set."""
    wanted = set(itemset)
    return sum(wanted <= set(trace) for trace in traces.values())


def length_summary(selector: ImpressedChainSelector, traces) -> tuple[list[dict], dict, dict]:
    """Per length: candidate sets, first front, layers used and facts about the selected sets.

    Returns ``(rows, selection, ranking)``: ``selection[size]`` = the selected
    sets (tuples of sorted activity names) in rank order; ``ranking[itemset]``
    = ``(rank, layer)`` of EVERY candidate set.
    """
    all_sets = selector.full_table()
    front_pool_sets = with_rule(selector, candidate_pool="front").full_table()
    pattern_cases = dict(zip(selector.patterns_["name"], selector.patterns_["n_cases"]))
    ranking = {tuple(itemset): (int(rank), int(layer))
               for itemset, rank, layer in zip(all_sets["itemset"], all_sets["rank"], all_sets["layer"])}
    rows, selection = [], {}
    for size in selector.lengths:
        of_size = all_sets[all_sets["size"] == size]
        chosen = of_size[of_size["rank"] <= selector.k]
        selection[int(size)] = [tuple(itemset) for itemset in chosen["itemset"]]
        rows.append({
            "length": int(size),
            "candidate sets": int(len(of_size)),
            "sets if only step-front patterns": int((front_pool_sets["size"] == size).sum()),
            "first front": int((of_size["layer"] == 1).sum()),
            "selected": int(len(chosen)),
            "layers used": int(chosen["layer"].max()) if len(chosen) else 0,
            "selected with an eventual edge in the source pattern":
                int(sum(EVENTUAL in pattern.edges for pattern in chosen["pattern"])),
            f"selected, source pattern in < {RARE_CASES} cases":
                int(sum(pattern_cases[name] < RARE_CASES for name in chosen["source_pattern"])),
            f"selected, set in < {RARE_CASES} cases":
                int(sum(case_count(itemset, traces) < RARE_CASES for itemset in chosen["itemset"])),
        })
    return rows, selection, ranking


def summarise(selector: ImpressedChainSelector, traces, seconds: float) -> tuple[dict, dict]:
    """Everything that is stored for one configuration, plus the ranking of all its candidate sets."""
    lengths, selection, ranking = length_summary(selector, traces)
    summary = {
        "seconds_fit": seconds,
        "n_patterns_evaluated": int(len(selector.patterns_)),
        "steps": step_composition(selector),
        "lengths": lengths,
        "selection": {str(size): [list(itemset) for itemset in sets] for size, sets in selection.items()},
    }
    return summary, ranking


def same_candidates_step_fronts(selector: ImpressedChainSelector) -> list[dict]:
    """Front sizes of the default run's step candidates under the four front rules.

    Also counts the candidates whose three interest values are exactly equal
    to those of another candidate of the step (the rows ``distinct`` acts on).
    """
    rows = []
    for table in selector.step_tables_:
        tied = table.duplicated(list(THREE_OBJECTIVES), keep=False)
        tied_rounded = table[list(THREE_OBJECTIVES)].round(ROUND_DECIMALS).duplicated(keep=False)
        row = {"step": int(table["step"].iloc[0]), "candidates": int(len(table))}
        for objectives, label in ((THREE_OBJECTIVES, "3 obj"), (TWO_OBJECTIVES, "2 obj")):
            for distinct in (False, True):
                row[f"front {label}, distinct={distinct}"] = int(pareto_front(table, objectives, distinct).sum())
        row["candidates with an exact twin (IG, coverage, CD)"] = int(tied.sum())
        row[f"same after rounding to {ROUND_DECIMALS} decimals"] = int(tied_rounded.sum())
        rows.append(row)
    return rows


def same_candidates_sets(selector: ImpressedChainSelector) -> dict:
    """Re-rank the default run's patterns with other rules: first fronts and selected sets per length."""
    result = {}
    for objectives, label in ((THREE_OBJECTIVES, "3 obj"), (TWO_OBJECTIVES, "2 obj")):
        for distinct in (False, True):
            table = with_rule(selector, objectives=objectives, distinct=distinct).full_table()
            result[f"{label}, distinct={distinct}"] = {
                str(size): {
                    "first_front": int(((table["size"] == size) & (table["layer"] == 1)).sum()),
                    "layers_used": int(table.loc[(table["size"] == size) & (table["rank"] <= selector.k),
                                                 "layer"].max()),
                    "selection": [list(itemset) for itemset in
                                  table.loc[(table["size"] == size) & (table["rank"] <= selector.k), "itemset"]],
                }
                for size in selector.lengths
            }
    return result


# --------------------------------------------------------------------------
# Comparing selections
# --------------------------------------------------------------------------
def as_sets(selection_of_size: list[list[str]]) -> set[tuple]:
    """Selected sets of one length as a Python set of tuples."""
    return {tuple(itemset) for itemset in selection_of_size}


def jaccard(first: set, second: set) -> float:
    """|intersection| / |union| (1.0 for two empty collections)."""
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def compare_selections(first: dict, second: dict) -> dict:
    """Per length and overall: number of shared sets and Jaccard index of two selections."""
    result = {}
    for size in first:
        a, b = as_sets(first[size]), as_sets(second[size])
        result[size] = {"shared": len(a & b), "jaccard": jaccard(a, b)}
    everything_a = set().union(*(as_sets(sets) for sets in first.values()))
    everything_b = set().union(*(as_sets(sets) for sets in second.values()))
    result["all"] = {"shared": len(everything_a & everything_b), "jaccard": jaccard(everything_a, everything_b)}
    return result


def activities_of(selection_of_size: list[list[str]]) -> set[str]:
    """All activities that occur in the selected sets of one length."""
    return {activity for itemset in selection_of_size for activity in itemset}


def gap_stability(selections: dict, rankings: dict, k: int) -> dict:
    """What the gap changes in the selection, per length.

    ``selections`` / ``rankings`` map a gap to the selection and to the
    ``{itemset: (rank, layer)}`` ranking of that gap's run.  Returns two lists
    of rows: ``across_gaps`` (sets and activities shared by all gaps) and
    ``versus_gap_3`` (where the sets that are new under another gap stand in
    the gap-3 run).
    """
    across, versus = [], []
    for size in selections[GAPS[0]]:
        per_gap = {gap: as_sets(selections[gap][size]) for gap in GAPS}
        acts = {gap: activities_of(selections[gap][size]) for gap in GAPS}
        across.append({"length": size,
                       "sets selected under every gap": len(set.intersection(*per_gap.values())),
                       "sets selected under at least one gap": len(set.union(*per_gap.values())),
                       "activities used under every gap": len(set.intersection(*acts.values())),
                       "activities used under at least one gap": len(set.union(*acts.values()))})
        for gap in GAPS:
            if gap == 3:
                continue
            ranks = [rankings[3].get(itemset) for itemset in per_gap[gap] - per_gap[3]]
            versus.append({
                "length": size, "gap": gap, "sets not in the gap-3 selection": len(ranks),
                f"of these: rank {k + 1}-{3 * k} in the gap-3 run": sum(
                    rank is not None and rank[0] <= 3 * k for rank in ranks),
                f"rank > {3 * k} in the gap-3 run": sum(rank is not None and rank[0] > 3 * k for rank in ranks),
                "not a candidate set in the gap-3 run": sum(rank is None for rank in ranks),
                "on the first front of the gap-3 run": sum(rank is not None and rank[1] == 1 for rank in ranks),
            })
    return {"across_gaps": across, "versus_gap_3": versus}


def equals_c4_selection(log_name: str, selection: dict) -> bool | None:
    """Is the default selection the one stored by experiment C4? ``None`` when that file is missing."""
    path = C4_DIR / f"impressed_sets_{log_name}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["selection"] == selection


def load_apriori(log_name: str) -> dict | None:
    """Top-10 Apriori sets per size from experiment C7, or ``None`` when the file is missing."""
    path = APRIORI_DIR / f"apriori_sets_{log_name}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        stored = json.load(handle)
    return {"min_support": stored["min_support"],
            "selection": {size: [sorted(row["itemset"]) for row in rows] for size, rows in stored["sets"].items()},
            "support": {size: [row["support"] for row in rows] for size, rows in stored["sets"].items()}}


# --------------------------------------------------------------------------
# One log
# --------------------------------------------------------------------------
def run_log(log_name: str) -> dict:
    """Run every configuration on one log and compare the selections."""
    log, dist_matrix, _ = load_selector_inputs(log_name)
    record = {"log": log_name, "n_cases": len(log.traces), "n_activities": len(log.activities), "configs": {}}
    rankings = {}
    for name, (overrides, decimals) in CONFIGS.items():
        selector, seconds = fit_selector(log, dist_matrix, overrides, decimals)
        record["configs"][name], rankings[name] = summarise(selector, log.traces, seconds)
        record["configs"][name]["settings"] = {**{key: list(value) if isinstance(value, tuple) else value
                                                  for key, value in {**BASE_SETTINGS, **overrides}.items()},
                                               "round_decimals": decimals}
        if name == DEFAULT:
            record["same_candidates_step_fronts"] = same_candidates_step_fronts(selector)
            record["same_candidates_sets"] = same_candidates_sets(selector)
            record["default_selected_table"] = [
                {"length": int(row["size"]), "rank": int(row["rank"]), "layer": int(row["layer"]),
                 "itemset": list(row["itemset"]), "source_pattern": row["source_pattern"],
                 "IG": float(row["IG"]), "coverage": float(row["coverage"]), "CD": float(row["CD"]),
                 "set_cases": case_count(row["itemset"], log.traces)}
                for row in selector.select_table().to_dict("records")]
        steps = ", ".join(f"{row['candidates']}->{row['front']}" for row in record["configs"][name]["steps"])
        print(f"[{log_name}] {name}: fit {seconds:.1f} s; steps {steps}", flush=True)

    selections = {name: config["selection"] for name, config in record["configs"].items()}
    record["gap_pairs"] = {
        f"{low} vs {high}": compare_selections(selections[GAP_CONFIGS[low]], selections[GAP_CONFIGS[high]])
        for low, high in combinations(GAPS, 2)}
    record["gap_stability"] = gap_stability({gap: selections[name] for gap, name in GAP_CONFIGS.items()},
                                            {gap: rankings[name] for gap, name in GAP_CONFIGS.items()},
                                            BASE_SETTINGS["k"])
    record["default_equals_C4_selection"] = equals_c4_selection(log_name, selections[DEFAULT])
    record["versus_default"] = {name: compare_selections(selections[DEFAULT], selection)
                                for name, selection in selections.items() if name != DEFAULT}
    record["same_candidates_versus_default"] = {
        rule: compare_selections(selections[DEFAULT], {size: cell["selection"] for size, cell in cells.items()})
        for rule, cells in record["same_candidates_sets"].items()}
    apriori = load_apriori(log_name)
    record["apriori"] = apriori
    if apriori is not None:
        record["versus_apriori"] = {name: compare_selections(apriori["selection"], selection)
                                    for name, selection in selections.items()}
    return record


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def overlap_cells(comparison: dict) -> dict:
    """'shared (Jaccard)' strings for lengths 1-3 and for all 30 sets."""
    return {("all sets" if size == "all" else f"length {size}"): f"{cell['shared']} ({cell['jaccard']:.2f})"
            for size, cell in comparison.items()}


def show(itemset) -> str:
    """An activity set as text."""
    return "{" + ", ".join(itemset) + "}"


def write_tables(records: list[dict]) -> None:
    """Write ``c5c6_tables.md`` (all numbers) and ``sets_vs_apriori.md`` (the set lists)."""
    parts = ["# C5b + C6 tables (generated by c5_c6_sensitivity.py)\n",
             "Overlap cells are 'number of shared sets (Jaccard index)'.\n"]

    step_rows, length_rows, time_rows = [], [], []
    for record in records:
        for name, config in record["configs"].items():
            for row in config["steps"]:
                step_rows.append({"log": record["log"], "config": name, **row})
            for row in config["lengths"]:
                length_rows.append({"log": record["log"], "config": name, **row})
            time_rows.append({"log": record["log"], "config": name, "fit seconds": config["seconds_fit"],
                              "patterns evaluated": config["n_patterns_evaluated"]})
    parts += ["## 1. Steps: candidates, front size and number of distinct activities (free runs)\n",
              markdown_table(pd.DataFrame(step_rows)),
              "\n## 2. Sets per length (free runs)\n", markdown_table(pd.DataFrame(length_rows)),
              "\n## 3. Runtime\n", markdown_table(pd.DataFrame(time_rows))]

    gap_rows = [{"log": record["log"], "gaps": pair, **overlap_cells(comparison)}
                for record in records for pair, comparison in record["gap_pairs"].items()]
    default_rows = [{"log": record["log"], "config": name, **overlap_cells(comparison)}
                    for record in records for name, comparison in record["versus_default"].items()]
    across_rows = [{"log": record["log"], **row} for record in records
                   for row in record["gap_stability"]["across_gaps"]]
    versus_rows = [{"log": record["log"], **row} for record in records
                   for row in record["gap_stability"]["versus_gap_3"]]
    parts += ["\n## 4. Overlap of the selected sets between gaps\n", markdown_table(pd.DataFrame(gap_rows)),
              "\n## 4b. Sets and activities shared by all four gaps\n", markdown_table(pd.DataFrame(across_rows)),
              "\n## 4c. Sets that are selected under another gap but not under gap 3: where are they in the "
              "gap-3 run?\n", markdown_table(pd.DataFrame(versus_rows)),
              "\n## 5. Overlap of every free run with the default run\n", markdown_table(pd.DataFrame(default_rows))]

    same_step_rows = [{"log": record["log"], **row} for record in records
                      for row in record["same_candidates_step_fronts"]]
    same_set_rows = []
    for record in records:
        for rule, cells in record["same_candidates_sets"].items():
            overlap = record["same_candidates_versus_default"][rule]
            for size, cell in cells.items():
                same_set_rows.append({"log": record["log"], "rule": rule, "length": size,
                                      "first front": cell["first_front"], "layers used": cell["layers_used"],
                                      "shared with default selection": overlap[size]["shared"]})
    parts += ["\n## 6. Same candidates (default run), other front rules: step fronts\n",
              markdown_table(pd.DataFrame(same_step_rows)),
              "\n## 7. Same patterns (default run), other ranking rules: sets per length\n",
              markdown_table(pd.DataFrame(same_set_rows))]

    apriori_rows = [{"log": record["log"], "config": name, **overlap_cells(comparison)}
                    for record in records for name, comparison in record.get("versus_apriori", {}).items()]
    if apriori_rows:
        parts += ["\n## 8. Overlap with the Apriori top 10 per size (experiment C7)\n",
                  markdown_table(pd.DataFrame(apriori_rows))]

    parts.append("\n## 9. Selected sets per gap\n")
    for record in records:
        for size in ("1", "2", "3"):
            rows = []
            for rank in range(BASE_SETTINGS["k"]):
                row = {"rank": rank + 1}
                for gap, name in GAP_CONFIGS.items():
                    sets = record["configs"][name]["selection"][size]
                    row[f"gap {gap}"] = show(sets[rank]) if rank < len(sets) else "-"
                rows.append(row)
            parts += [f"\n### {record['log']}, length {size}\n", markdown_table(pd.DataFrame(rows))]
    (RESULT_DIR / "c5c6_tables.md").write_text("\n".join(parts) + "\n", encoding="utf-8")

    parts = ["# IMPresseD sets (default run) next to the Apriori top 10 per size "
             "(generated by c5_c6_sensitivity.py)\n",
             "`*` marks a set that is in both lists. Apriori: experiment C7, ranked by support.\n"]
    for record in records:
        apriori = record["apriori"]
        if apriori is None:
            parts.append(f"\n## {record['log']}: no Apriori file found in {APRIORI_DIR.name}\n")
            continue
        selected = pd.DataFrame(record["default_selected_table"])
        for size in ("1", "2", "3"):
            ours = selected[selected["length"] == int(size)].to_dict("records")
            theirs = [tuple(itemset) for itemset in apriori["selection"][size]]
            shared = {tuple(row["itemset"]) for row in ours} & set(theirs)
            rows = []
            for rank in range(max(len(ours), len(theirs))):
                row = {"rank": rank + 1, "IMPresseD set": "-", "layer": "-", "source pattern": "-",
                       "cases with the set": "-", "Apriori set": "-", "support": "-"}
                if rank < len(ours):
                    mine = ours[rank]
                    row.update({"IMPresseD set": show(mine["itemset"]) + ("*" if tuple(mine["itemset"]) in shared
                                                                         else ""),
                                "layer": mine["layer"], "source pattern": mine["source_pattern"],
                                "cases with the set": mine["set_cases"]})
                if rank < len(theirs):
                    row.update({"Apriori set": show(theirs[rank]) + ("*" if theirs[rank] in shared else ""),
                                "support": f"{apriori['support'][size][rank]:.3f}"})
                rows.append(row)
            parts += [f"\n## {record['log']}, length {size}: {len(shared)} shared "
                      f"(Apriori min_support {apriori['min_support']})\n", markdown_table(pd.DataFrame(rows))]
    (RESULT_DIR / "sets_vs_apriori.md").write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--logs", nargs="+", default=["f1", "f2", "f3"], choices=sorted(DATASETS))
    args = parser.parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    module_hash = hashlib.sha256(Path(impressed_chain.__file__).read_bytes()).hexdigest()
    print(f"impressed_chain.py sha256 {module_hash}", flush=True)
    records = [run_log(log_name) for log_name in args.logs]
    output = {"impressed_chain_sha256": module_hash, "base_settings": {
        key: list(value) if isinstance(value, tuple) else value for key, value in BASE_SETTINGS.items()},
        "logs": records}
    with open(RESULT_DIR / "c5c6_results.json", "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=1)
    write_tables(records)
    print(f"total {time.perf_counter() - start:.1f} s; results in {RESULT_DIR}")


if __name__ == "__main__":
    main()
