"""Experiment C8, check: why does accumulation make the importance climb?

One shuffle moves only the COMPLETE occurrences of an itemset (the j-th
occurrence of every activity, ``find_occurrences``). If one activity of the
set occurs more often in a trace than another, the extra events ("surplus
events") stay where they are. When permutations accumulate, the next repeat
finds the occurrences in the already permuted trace, so sooner or later the
surplus events are moved as well: the trace drifts further from the original
with every repeat. An itemset without surplus events cannot drift that way.

This script counts, per itemset of ``run_<dataset>.json`` (written by
``c8_run.py``) and over the whole log: the traces that contain the itemset,
the complete occurrences per trace, and the surplus events.

Run:  python c8_surplus_events.py

Writes ``results/experiments/C8/surplus_events.csv``.
"""

from __future__ import annotations

import json

import pandas as pd

from c8_run import RESULT_DIR
from engine import DATASETS, EventLog, find_occurrences, itemset_label


def surplus_table(dataset: str) -> pd.DataFrame:
    """One row per itemset of ``dataset`` with occurrence and surplus counts."""
    log = EventLog(DATASETS[dataset])
    run_info = json.loads(
        (RESULT_DIR / f"run_{dataset}.json").read_text(encoding="utf-8"))
    records = []
    for rank, items in run_info["itemsets"].items():
        items = set(items)
        # The original's rule for "this trace is permuted" (tools.py:521-523).
        traces = [trace for trace in log.traces.values()
                  if items.issubset(trace) and len(items) != len(trace)]
        occurrences = [len(find_occurrences(trace, items)) for trace in traces]
        surplus = [sum(trace.count(act) - n for act in items)
                   for trace, n in zip(traces, occurrences)]
        records.append({
            "dataset": dataset,
            "itemset_id": int(rank),
            "itemset": itemset_label(items),
            "n_traces": len(traces),
            "complete_occurrences_per_trace": sum(occurrences) / len(traces),
            "share_of_traces_with_surplus_events":
                sum(count > 0 for count in surplus) / len(traces),
            "surplus_events_per_trace": sum(surplus) / len(traces),
        })
    return pd.DataFrame.from_records(records)


def main() -> None:
    table = pd.concat([surplus_table(dataset) for dataset in sorted(DATASETS)
                       if (RESULT_DIR / f"run_{dataset}.json").exists()],
                      ignore_index=True)
    table.to_csv(RESULT_DIR / "surplus_events.csv", index=False, float_format="%.4f")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
