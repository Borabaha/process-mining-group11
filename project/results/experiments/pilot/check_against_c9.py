"""Cross-check of the pilot values against the stored values of experiment C9.

C9 (``c9_seed_stability.py --itemsets project``) computed the same 60 sets
per log with the same engine settings, but read the sets through another
code path (Apriori mined live at support 0.45, IMPresseD via
``c1_common.load_itemsets``). If the pilot is wired correctly, every value
with the same (scored fold, fold seed, fold, set, repeat) must be identical.

Run:  python check_against_c9.py      (writes check_against_c9.json)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PILOT_DIR = Path(__file__).resolve().parent
C9_RAW_DIR = PILOT_DIR.parent / "C9" / "raw"
KEYS = ["score_on", "fold_seed", "fold", "itemset", "repeat"]
VALUES = ["baseline", "importance", "n_traces_with_itemset", "n_traces_changed"]


def load_c9(log_name: str, fold_seeds) -> pd.DataFrame:
    """C9's values for the given fold seeds, with the pilot's key names."""
    folder = C9_RAW_DIR / f"{log_name}_project_k5"
    frames = [pd.read_csv(folder / f"seed_{seed}.csv", dtype={"itemset": str})
              for seed in fold_seeds if (folder / f"seed_{seed}.csv").exists()]
    table = pd.concat(frames, ignore_index=True)
    table["score_on"] = table["setting"].str.replace("fixed_", "")
    table = table.rename(columns={"seed": "fold_seed"})
    return table.drop_duplicates(KEYS)[KEYS + VALUES]


def check_log(log_name: str) -> dict:
    """Compare one log; returns counts and the largest absolute differences."""
    pilot = pd.read_csv(PILOT_DIR / f"importance_{log_name}.csv", dtype={"itemset": str})
    pilot = pilot.drop_duplicates(KEYS)[KEYS + VALUES]
    reference = load_c9(log_name, sorted(pilot["fold_seed"].unique()))
    both = pilot.merge(reference, on=KEYS, suffixes=("_pilot", "_c9"))
    result = {"log": log_name, "pilot_rows": len(pilot), "c9_rows": len(reference),
              "matched_rows": len(both)}
    for value in VALUES:
        difference = (both[f"{value}_pilot"] - both[f"{value}_c9"]).abs()
        result[f"max_abs_diff_{value}"] = float(difference.max())
    return result


def main() -> None:
    results = [check_log(path.stem.split("_")[1])
               for path in sorted(PILOT_DIR.glob("importance_*.csv"))]
    (PILOT_DIR / "check_against_c9.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
