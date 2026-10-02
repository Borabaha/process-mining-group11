"""Pilot: location importance of Apriori sets vs IMPresseD sets, end to end.

For every log this script

1. reads the ``k`` activity sets per length of both strategies (Apriori sets
   of experiment C7, IMPresseD sets of experiment C4),
2. splits the cases into seeded stratified folds (one split per fold seed),
3. fits ONE XGBoost model per fold, shared by all sets and both strategies,
4. computes the location permutation importance of every set with
   ``engine.py`` on the scored fold(s), and
5. writes a tidy CSV and box plots (two strategies side by side).

A set selected by both strategies is computed once and written once per
strategy, so it has exactly the same values in both.

Run (from this folder; one process per log can run side by side):

    python run_pilot.py                 # all logs of pilot.yaml
    python run_pilot.py --logs f1       # one log
    python run_pilot.py --figures-only  # redraw the plots from the stored CSV

Then ``python compare.py`` turns the CSV files into the comparison tables.

Output in ``results/experiments/pilot/`` (``output_dir`` of the config):

``sets_<log>.csv``        the selected sets (strategy, length, rank, support)
``importance_<log>.csv``  one row per strategy x set x scored fold x fold seed
                          x fold x repeat
``box_<log>_len<length>_<scored fold>.png``
``timing_<log>.json``     seconds per step
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib
import pandas as pd
import yaml
from xgboost import XGBClassifier

from engine import (
    DATASETS,
    EventLog,
    IndexEncoder,
    LocationPermutationImportance,
    itemset_label,
    make_folds,
)

matplotlib.use("Agg")  # write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().with_name("pilot.yaml")

STRATEGIES = ("apriori", "impressed")
STRATEGY_TITLE = {"apriori": "Apriori (most frequent sets)",
                  "impressed": "IMPresseD (outcome-oriented sets)"}
STRATEGY_COLOUR = {"apriori": "#2a78d6", "impressed": "#eb6834"}
FOLD_TITLE = {"test": "held-out fold", "train": "training fold"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def say(text: str) -> None:
    """Progress line with a wall-clock stamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def load_config(path: Path) -> dict:
    """Read the YAML configuration and check the values the script relies on."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(config["score_on"], str):
        config["score_on"] = [config["score_on"]]
    unknown = set(config["logs"]) - set(DATASETS)
    if unknown:
        raise ValueError(f"unknown logs in the config: {sorted(unknown)}")
    if config["mode"] not in ("fixed", "faithful"):
        raise ValueError("mode must be 'fixed' or 'faithful'")
    return config


# --------------------------------------------------------------------------
# Activity sets
# --------------------------------------------------------------------------
def read_ranked_sets(path: Path, strategy: str) -> dict[int, list[list[str]]]:
    """``{length: [activities, ...]}`` in rank order from a C7 or C4 file."""
    stored = json.loads(path.read_text(encoding="utf-8"))["sets"]
    key = "itemset" if strategy == "apriori" else "activities"
    return {
        int(length): [entry[key] for entry in sorted(entries, key=lambda e: e["rank"])]
        for length, entries in stored.items()
    }


def load_selected_sets(log_name: str, log: EventLog, config: dict) -> pd.DataFrame:
    """The sets of both strategies for one log.

    Columns: log, strategy, length, rank, itemset (label ``'a, b'``),
    support (share of all cases that contain every activity of the set) and
    selected_by_both. Raises ``ValueError`` when a strategy has fewer than
    ``k`` sets of a length or a set has the wrong number of activities.
    """
    records = []
    for strategy in STRATEGIES:
        path = PROJECT_ROOT / config[f"{strategy}_sets"].format(log=log_name)
        ranked = read_ranked_sets(path, strategy)
        for length in config["lengths"]:
            chosen = ranked.get(length, [])[: config["k"]]
            if len(chosen) < config["k"]:
                raise ValueError(f"{log_name}/{strategy}: only {len(chosen)} sets "
                                 f"of length {length} in {path.name}")
            for rank, activities in enumerate(chosen, start=1):
                if len(set(activities)) != length:
                    raise ValueError(f"{log_name}/{strategy}: {activities} is not "
                                     f"a set of {length} activities")
                records.append({"log": log_name, "strategy": strategy, "length": length,
                                "rank": rank, "itemset": itemset_label(activities),
                                "support": case_support(log, activities)})
    sets = pd.DataFrame.from_records(records)
    sets["selected_by_both"] = sets.groupby("itemset")["strategy"].transform("nunique") == 2
    return sets


def case_support(log: EventLog, activities) -> float:
    """Share of the log's cases that contain every activity of the set."""
    wanted = set(activities)
    return sum(wanted.issubset(trace) for trace in log.traces.values()) / len(log.traces)


# --------------------------------------------------------------------------
# Importance
# --------------------------------------------------------------------------
def make_engine(config: dict, score_on: str, fold_seed: int) -> LocationPermutationImportance:
    """The engine for one scored fold and one fold seed.

    Fixed mode adds the fold seed to the permutation seed, because the
    engine seeds its random streams by fold NUMBER: without this, two fold
    seeds would reuse the same streams. Faithful mode keeps the constant
    seed of the original code.
    """
    if config["mode"] == "faithful":
        return LocationPermutationImportance(
            mode="faithful", score_on=score_on, n_repeats=config["repeats"],
            random_state=config["seed"])
    return LocationPermutationImportance(
        mode="fixed", score_on=score_on, n_repeats=config["repeats"],
        random_state=config["seed"] + fold_seed,
        allowed_from=config["allowed_from"], draw=config["draw"])


def run_fold_seed(log, encoder, itemsets: dict, fold_seed: int, config: dict):
    """Importance of all sets for the folds of one fold seed.

    ``itemsets`` maps a set label to its activities (every distinct set
    once). Returns ``(table, seconds)``: the engine's rows plus the columns
    ``fold_seed``, ``score_on`` and ``n_scored`` (cases in the scored fold),
    and the seconds spent on fitting and on each scored fold.
    """
    tables = []
    seconds = {"fit": 0.0, **{score_on: 0.0 for score_on in config["score_on"]}}
    folds = make_folds(log.labels, k=config["folds"], seed=fold_seed)
    for fold, (train, test) in enumerate(folds):
        start = time.perf_counter()
        model = XGBClassifier(n_jobs=config["xgb_n_jobs"]).fit(
            encoder.transform(log.traces[case] for case in train),
            [log.labels[case] for case in train])
        seconds["fit"] += time.perf_counter() - start

        for score_on in config["score_on"]:
            start = time.perf_counter()
            engine = make_engine(config, score_on, fold_seed)
            with warnings.catch_warnings():
                # "itemset occurs in no scored trace": kept as a column below.
                warnings.simplefilter("ignore", UserWarning)
                table = engine.compute(model, log, encoder, train, test,
                                       itemsets, fold=fold)
            seconds[score_on] += time.perf_counter() - start
            table.insert(0, "n_scored", len(test) if score_on == "test" else len(train))
            table.insert(0, "score_on", score_on)
            table.insert(0, "fold_seed", fold_seed)
            tables.append(table)
    return pd.concat(tables, ignore_index=True), seconds


def run_log(log_name: str, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """All fold seeds for one log: ``(sets, tidy importance table, timing)``.

    The tidy table has one row per strategy x set x scored fold x fold seed
    x fold x repeat. ``measurable`` is False where no scored trace contains
    the set; the engine reports importance 0 there, but the analysis must
    treat such rows as missing.
    """
    start = time.perf_counter()
    log = EventLog(DATASETS[log_name])
    encoder = IndexEncoder(log.traces.values(), log.activities)
    sets = load_selected_sets(log_name, log, config)
    itemsets = {label: label.split(", ") for label in sets["itemset"].unique()}
    say(f"{log_name}: {len(log.traces)} cases, {len(sets)} selected sets, "
        f"{len(itemsets)} distinct")

    tables = []
    timing = {"setup": time.perf_counter() - start, "per_fold_seed": {}}
    for fold_seed in config["fold_seeds"]:
        table, seconds = run_fold_seed(log, encoder, itemsets, fold_seed, config)
        tables.append(table)
        timing["per_fold_seed"][fold_seed] = seconds
        say(f"{log_name}: fold seed {fold_seed} done ("
            + ", ".join(f"{name} {value:.1f} s" for name, value in seconds.items()) + ")")

    values = pd.concat(tables, ignore_index=True).drop(columns="itemset_id")
    values["measurable"] = values["n_traces_with_itemset"] > 0
    keys = ["log", "strategy", "length", "rank", "itemset", "selected_by_both"]
    tidy = sets[keys].merge(values, on="itemset", how="left")
    timing["total"] = time.perf_counter() - start
    return sets, tidy, timing


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fold_means(tidy: pd.DataFrame) -> pd.DataFrame:
    """Mean over the repeats per set, fold seed and fold (measurable rows)."""
    keys = ["score_on", "strategy", "length", "rank", "itemset", "selected_by_both",
            "fold_seed", "fold"]
    return (tidy[tidy["measurable"]].groupby(keys, sort=False)["importance"]
            .mean().reset_index())


def plot_boxes(means: pd.DataFrame, log_name: str, length: int, score_on: str,
               config: dict, path: Path) -> None:
    """One figure: the sets of one length, Apriori left, IMPresseD right.

    Every box holds the fold means of one set (fold seeds x folds values,
    each the mean of the repeats); the diamond is their mean. Rank 1 of the
    strategy's own selection is at the top. Sets selected by both strategies
    are marked with '(both)'.
    """
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True,
                                constrained_layout=True)
    for axis, strategy in zip(axes, STRATEGIES):
        chosen = means[(means["strategy"] == strategy) & (means["length"] == length)
                       & (means["score_on"] == score_on)]
        groups = [(rank, group) for rank, group in chosen.groupby("rank")]
        values = [group["importance"].to_numpy() for _, group in groups]
        labels = [group["itemset"].iloc[0].replace(", ", "  +  ")
                  + ("  (both)" if group["selected_by_both"].iloc[0] else "")
                  for _, group in groups]
        positions = [rank for rank, _ in groups]
        colour = STRATEGY_COLOUR[strategy]
        boxes = axis.boxplot(
            values, positions=positions, orientation="horizontal", widths=0.6,
            patch_artist=True, showmeans=True,
            medianprops={"color": INK, "linewidth": 1.4},
            meanprops={"marker": "D", "markerfacecolor": INK,
                       "markeredgecolor": "white", "markersize": 5},
            whiskerprops={"color": colour, "linewidth": 1},
            capprops={"color": colour, "linewidth": 1},
            flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": colour,
                        "markeredgecolor": "none", "alpha": 0.6})
        for patch in boxes["boxes"]:
            patch.set(facecolor=colour, alpha=0.55, edgecolor=colour, linewidth=1)
        axis.axvline(0, color=MUTED, linewidth=1)
        axis.set_yticks(positions, labels, fontsize=8.5, color=INK)
        axis.set_ylim(config["k"] + 0.6, 0.4)  # rank 1 at the top
        axis.set_title(STRATEGY_TITLE[strategy], fontsize=10.5, color=INK, loc="left")
        axis.grid(axis="x", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(colors=MUTED, labelsize=8.5, length=0)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color(GRID)
    figure.supxlabel("importance = drop in weighted F1 after permuting the set's "
                     "location (same scale in both panels)", fontsize=9, color=MUTED)
    noun ="activity" if length == 1 else "activities"
    figure.suptitle(
        f"BPIC11 {log_name}: location importance of sets of {length} {noun}, "
        f"{config['mode']} mode, scored on the {FOLD_TITLE[score_on]}\n"
        f"box = fold means over {len(config['fold_seeds'])} fold seeds x "
        f"{config['folds']} folds ({config['repeats']} repeats each); "
        "diamond = mean; top row = rank 1 of the strategy's selection",
        fontsize=10, color=INK, x=0.01, ha="left")
    figure.savefig(path, dpi=150, facecolor="white")
    plt.close(figure)


def write_figures(tidy: pd.DataFrame, log_name: str, config: dict, out_dir: Path) -> list[str]:
    """All box plots of one log; returns the file names."""
    means = fold_means(tidy)
    names = []
    for score_on in config["score_on"]:
        for length in config["lengths"]:
            name = f"box_{log_name}_len{length}_{score_on}.png"
            plot_boxes(means, log_name, length, score_on, config, out_dir / name)
            names.append(name)
    return names


# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--logs", nargs="+", choices=sorted(DATASETS),
                        help="override the logs of the config")
    parser.add_argument("--output-dir", type=Path,
                        help="override output_dir of the config")
    parser.add_argument("--figures-only", action="store_true",
                        help="redraw the box plots from the stored importance_<log>.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.logs:
        config["logs"] = args.logs
    out_dir = args.output_dir or PROJECT_ROOT / config["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    say(f"config {args.config.name}: {config}")

    for log_name in config["logs"]:
        if args.figures_only:
            tidy = pd.read_csv(out_dir / f"importance_{log_name}.csv",
                               dtype={"itemset": str})
            say(f"{log_name}: redrew {write_figures(tidy, log_name, config, out_dir)}")
            continue
        sets, tidy, timing = run_log(log_name, config)
        sets.to_csv(out_dir / f"sets_{log_name}.csv", index=False, float_format="%.10g")
        tidy.to_csv(out_dir / f"importance_{log_name}.csv", index=False,
                    float_format="%.10g")
        timing["figures"] = write_figures(tidy, log_name, config, out_dir)
        (out_dir / f"timing_{log_name}.json").write_text(
            json.dumps({"config": config, **timing}, indent=2), encoding="utf-8")
        say(f"{log_name}: {len(tidy)} rows, {len(timing['figures'])} figures, "
            f"{timing['total']:.0f} s")


if __name__ == "__main__":
    main()
