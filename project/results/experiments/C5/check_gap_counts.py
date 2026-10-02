"""Sanity check for C5b: are the per-case counts right for gaps other than 3?

Run:  python results/experiments/C5/check_gap_counts.py

For every gap in {1, 2, 5} and every log, 300 random evaluated patterns are
recounted with the plain scan ``impressed_chain.count_instances`` (independent
of the extension code) and compared with the counts the selector stored.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from c4_run_selector import load_selector_inputs, recount_check  # noqa: E402
from impressed_chain import EVENTUAL, ImpressedChainSelector  # noqa: E402

for log_name in ("f1", "f2", "f3"):
    log, dist_matrix, _ = load_selector_inputs(log_name)
    for gap in (1, 2, 5):
        selector = ImpressedChainSelector(max_gap=gap).fit(log.traces, log.labels, dist_matrix)
        n_eventual = sum(EVENTUAL in pattern.edges for pattern in selector.counts_)
        print(log_name, "gap", gap, recount_check(selector, log.traces, 300, seed=2023),
              "patterns with an eventual edge:", n_eventual, "of", len(selector.counts_), flush=True)
