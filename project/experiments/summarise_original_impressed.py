"""Print Markdown tables for one run of ``run_original_impressed.py``.

Usage::

    .venv/Scripts/python.exe experiments/summarise_original_impressed.py \
        results/experiments/C3/f1_step1/timings.json results/experiments/C3/original_patterns_f1.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def timing_table(timings: dict) -> str:
    """Stage wall time with the inclusive time of every wrapped function."""
    lines = ["| stage | wall s | function (inclusive) | calls | seconds | share of stage |",
             "|---|---|---|---|---|---|"]
    for stage, wall in timings["stage_wall_seconds"].items():
        functions = timings["inclusive_seconds_per_function"].get(stage.replace(" (open)", ""), {})
        calls = timings["calls_per_function"].get(stage.replace(" (open)", ""), {})
        if not functions:
            lines.append("| %s | %.1f | - | - | - | - |" % (stage, wall))
        for name, seconds in functions.items():
            lines.append("| %s | %.1f | %s | %d | %.1f | %.0f%% |"
                         % (stage, wall, name, calls[name], seconds, 100 * seconds / wall if wall else 0))
    return "\n".join(lines)


def edge_kind(pattern: dict) -> str:
    """'single', 'directly', 'eventually' or 'mixed' from the edge types of a pattern."""
    kinds = {edge["type"] for edge in pattern["edges"]}
    if not kinds:
        return "single"
    return kinds.pop() if len(kinds) == 1 else "mixed"


def pattern_table(export: dict) -> str:
    """Candidates and front per step, split by node count, distinct activities and edge type."""
    lines = ["| step | candidates | front | candidates by #nodes | front by #nodes | front by #distinct activities "
             "| front by edge type | distinct activity sets on front | candidates with 0 train cases |",
             "|---|---|---|---|---|---|---|---|---|"]
    for step, block in export["steps"].items():
        patterns = block["patterns"]
        front = [p for p in patterns if p["on_front"]]
        lines.append("| %s | %d | %s | %s | %s | %s | %s | %d | %d |" % (
            step, block["n_candidates"], block["front_size"],
            dict(sorted(Counter(p["n_nodes"] for p in patterns).items())),
            dict(sorted(Counter(p["n_nodes"] for p in front).items())),
            dict(sorted(Counter(len(p["activity_set"]) for p in front).items())),
            dict(sorted(Counter(edge_kind(p) for p in front).items())),
            len({tuple(p["activity_set"]) for p in front}),
            sum((p["Case_Support"] or 0) == 0 for p in patterns)))
    return "\n".join(lines)


def front_listing(export: dict, step: str, limit: int = 15) -> str:
    """The front patterns of one step, ordered by information gain."""
    front = [p for p in export["steps"][step]["patterns"] if p["on_front"]]
    front.sort(key=lambda p: -(p["Outcome_Interest"] or 0))
    lines = ["| id | labels (trace order) | edge types | IG | coverage | case distance | train cases |",
             "|---|---|---|---|---|---|---|"]
    for p in front[:limit]:
        lines.append("| %s | %s | %s | %.4f | %.3f | %.4f | %d |" % (
            p["id"], " , ".join(p["labels"]), "/".join(e["type"] for e in p["edges"]) or "-",
            p["Outcome_Interest"], p["Frequency_Interest"], p["Case_Distance_Interest"], p["Case_Support"]))
    return "\n".join(lines)


def main() -> None:
    timings_path, patterns_path = Path(sys.argv[1]), Path(sys.argv[2])
    meta = json.loads(timings_path.read_text(encoding="utf-8"))
    export = json.loads(patterns_path.read_text(encoding="utf-8"))
    print("completed:", meta["completed"], "| cases:", meta.get("n_cases"), "| activities:", meta.get("n_activities"))
    print("preparation seconds:", {k: round(v, 1) for k, v in meta["preparation_seconds"].items()})
    print()
    print(timing_table(meta["timings"]))
    print()
    print(pattern_table(export))
    for step in export["steps"]:
        print("\nFront of step %s (top 15 by information gain)\n" % step)
        print(front_listing(export, step))
    print("\nunscored patterns from an unfinished step:", len(export["unscored_patterns_from_unfinished_step"]))


if __name__ == "__main__":
    main()
