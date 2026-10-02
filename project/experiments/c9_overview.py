"""Experiment C9, part 3: one table across all studies (overview).

Collects the key numbers of every ``summary_<study>.json`` written by
``c9_analysis.py`` into one markdown table, so that the logs, fold counts and
engine settings can be compared side by side.

Run:  python c9_overview.py

Writes results/experiments/C9/overview.md.
"""

from __future__ import annotations

import json

from c9_analysis import fmt, md_table
from c9_seed_stability import RESULT_DIR

STUDIES = ("f1_original_k5", "f1_original_k5_first10", "f1_original_k10",
           "f2_original_k5", "f3_original_k5")
HEADER = ["study", "setting", "fold seeds", "F1 of the scored fold",
          "Spearman of two seeds: 3 / 5 / 10 / 30 repeats",
          "Kendall of two seeds: 3 / 5 / 10 / 30 repeats",
          "Spearman, same folds, 5 repeats",
          "Spearman, two groups of 5 seeds x 10 repeats",
          "ranking noise (10 repeats)", "spread between itemsets"]


def series(between: dict, key: str) -> str:
    """'a / b / c / d' for 3, 5, 10 and 30 repeats ('-' where not run)."""
    cells = []
    for size in ("3", "5", "10", "30"):
        result = between.get(size)
        cells.append(fmt(result["all_seeds_all_blocks"][key], 2) if result else "-")
    return " / ".join(cells)


def study_rows(stem: str) -> list[list]:
    """The table rows (one per engine setting) of one study."""
    summary = json.loads(
        (RESULT_DIR / f"summary_{stem}.json").read_text(encoding="utf-8"))
    rows = []
    for setting, numbers in summary["settings"].items():
        same = numbers["same_folds"].get("5")
        pooled = numbers["pooled_seeds"].get("5x10")
        noise = numbers["noise_and_signal"]["10"]
        rows.append([
            stem, setting, numbers["n_seeds"],
            fmt(numbers["baseline_mean_all_seeds"]),
            series(numbers["between_seeds"], "spearman_mean"),
            series(numbers["between_seeds"], "kendall_mean"),
            fmt(same["spearman_mean"], 2) if same else "-",
            fmt(pooled["spearman_mean"], 2) if pooled else "-",
            fmt(noise["ranking_noise_std_of_centred_seed_mean"], 5),
            fmt(noise["signal_std_between_itemsets"], 5),
        ])
    return rows


def main() -> None:
    rows = [row for stem in STUDIES
            if (RESULT_DIR / f"summary_{stem}.json").exists()
            for row in study_rows(stem)]
    text = "# C9 overview of all studies\n\n" + md_table(HEADER, rows)
    (RESULT_DIR / "overview.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
