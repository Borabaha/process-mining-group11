"""Experiment C1, part 4: tables and extrapolations from the measured timings.

Reads the JSON files written by ``c1_original_vs_engine.py``,
``c1_engine_grid.py`` and ``c1_original_single.py`` and writes

* results/experiments/C1/tables.md          (markdown tables for RESULT.md)
* results/experiments/C1/extrapolation.json (the numbers behind the tables)

Every extrapolation uses the MEDIAN seconds per (itemset, repeat) of the
measured group. The formulas (S = setup of one log, F = one model fit plus
the baseline prediction, t = seconds per (itemset, repeat)):

(a) original defaults    T = S + 5 F + 5 folds x 10 repeats x sum of t over
                             the original's itemsets (t taken by set size)
(b) full project grid    T = S + 5 F + 5 folds x 10 repeats x 10 sets x
                             sum of t over 2 strategies x 3 sizes
(c) single activities    T = S + 5 F + 5 folds x 10 repeats x (seconds of one
                             pass over all activities)

Run:  python c1_report.py
"""

from __future__ import annotations

import json
import statistics

import pandas as pd

from apriori_selector import original_top10
from c1_common import RESULT_DIR, SIZES, TOP_K
from engine import DATASETS, EventLog

LOGS = ("f1", "f2", "f3")
N_FOLDS, N_REPEATS = 5, 10
STRATEGIES = ("apriori", "impressed")
ENGINE_ONE_FOLD = {
    "engine_faithful_train": "engine faithful, training fold",
    "engine_fixed_train": "engine fixed, training fold",
    "engine_fixed_test": "engine fixed, held-out fold",
}
ENGINE_GRID = {
    "faithful_train": "engine faithful, training fold",
    "fixed_train": "engine fixed, training fold",
    "fixed_test": "engine fixed, held-out fold",
}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def read_json(name: str) -> dict | None:
    """Content of ``results/experiments/C1/<name>`` or None if it is missing."""
    path = RESULT_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def duration(seconds: float | None) -> str:
    """Human-readable time: seconds below 2 minutes, then minutes, then hours."""
    if seconds is None:
        return "n/a"
    if seconds < 120:
        return f"{seconds:.1f} s" if seconds >= 10 else f"{seconds:.2f} s"
    if seconds < 7200:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def median_of(records, key="seconds") -> float:
    """Median of ``key`` over a list of LoadMeter dicts."""
    return statistics.median(record[key] for record in records)


def markdown_table(header, rows) -> str:
    """A markdown table from a header list and a list of row lists."""
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def group_of(one_fold: dict, strategy: str, size: int) -> dict:
    """The measured group (strategy, size) of a one-fold result."""
    for group in one_fold["groups"]:
        if group["strategy"] == strategy and group["size"] == size:
            return group
    raise KeyError((strategy, size))


def original_setup_and_fit(one_fold: dict) -> tuple[float, float]:
    """(S, F) of the original: load + encode, and fit + baseline prediction."""
    one_off, per_fold = one_fold["one_off_seconds"], one_fold["per_fold_seconds"]
    setup = (one_off["original DataManager()"]["seconds"]
             + one_off["original index_encoding(whole log) + int cast"]["seconds"])
    fit = (median_of(per_fold["fit on original DataFrame (int64)"])
           + median_of(per_fold["baseline predict, original DataFrame, training fold"]))
    return setup, fit


def engine_setup_and_fit(one_fold: dict) -> tuple[float, float]:
    """(S, F) of the engine: load + encoder, and fit + baseline prediction."""
    one_off, per_fold = one_fold["one_off_seconds"], one_fold["per_fold_seconds"]
    setup = (one_off["engine EventLog()"]["seconds"]
             + one_off["engine IndexEncoder()"]["seconds"]
             + one_off["engine IndexEncoder.transform(whole log)"]["seconds"])
    fit = (median_of(per_fold["fit on engine matrix (int8)"])
           + median_of(per_fold["baseline predict, engine matrix, training fold"]))
    return setup, fit


def grid_loop_seconds(grid: dict | None, part: str, variant: str):
    """(seconds, failed calls) of all ``compute`` calls of one part and variant."""
    if grid is None:
        return None, 0
    calls = [c for c in grid["calls"] if c["part"] == part and c["variant"] == variant]
    failed = sum(c["error"] is not None for c in calls)
    return sum(c["seconds"] for c in calls if c["error"] is None), failed


def grid_setup_and_fits(grid: dict) -> float:
    """S + 5 F as measured inside a full-grid run of the engine."""
    return (grid["seconds_load_log_and_build_encoder"]["seconds"]
            + sum(record["seconds"] for record in grid["fit_per_fold"]))


def summarise(seconds) -> dict:
    """Median, mean, min, max and count of a list of seconds."""
    return {"n": len(seconds), "median": statistics.median(seconds),
            "mean": statistics.fmean(seconds), "min": min(seconds),
            "max": max(seconds)}


def extra_runs(log: str) -> dict[str, dict]:
    """Repeated one-fold runs of a log: ``{tag: content}`` (files with a tag)."""
    return {
        path.stem.removeprefix(f"one_fold_{log}_"): json.loads(
            path.read_text(encoding="utf-8"))
        for path in sorted(RESULT_DIR.glob(f"one_fold_{log}_*.json"))
    }


def pool_runs(first: dict, others: list[dict]) -> dict:
    """Pool the timings of repeated one-fold runs of the same log.

    Per group the iteration times of the original and the timing runs of the
    engine are concatenated and summarised again; the per-fold records are
    concatenated too. One-off steps keep the values of the first run.
    """
    pooled = json.loads(json.dumps(first))  # deep copy
    for group in pooled["groups"]:
        group["original"]["others_percent_runs"] = [
            group["original"]["load"]["others_percent"]]
    for other in others:
        for name, records in other["per_fold_seconds"].items():
            pooled["per_fold_seconds"][name] += records
        for group in pooled["groups"]:
            twin = group_of(other, group["strategy"], group["size"])
            if twin["itemsets"] != group["itemsets"]:
                raise ValueError("repeated runs used different itemsets")
            original = group["original"]
            original["durations"] += twin["original"]["durations"]
            original["seconds_per_iteration"] = summarise(original["durations"])
            original["others_percent_runs"].append(
                twin["original"]["load"]["others_percent"])
            for label in ENGINE_ONE_FOLD:
                group[label]["runs"] += twin[label]["runs"]
                group[label]["seconds_per_iteration"] = summarise(group[label]["runs"])
            differences = [group["faithful_vs_original_max_abs_diff"],
                           twin["faithful_vs_original_max_abs_diff"]]
            group["faithful_vs_original_max_abs_diff"] = (
                None if None in differences else max(differences))
    return pooled


def original_itemset_sizes(log_name: str) -> list[int]:
    """Sizes of the itemsets the original selects (min_support 0.5, top 10)."""
    log = EventLog(DATASETS[log_name])
    itemsets, _ = original_top10(log.traces, min_support=0.5, top_k=TOP_K)
    return sorted(len(items) for items in itemsets)


# --------------------------------------------------------------------------
# Tables of measurements
# --------------------------------------------------------------------------
def table_one_off(results: dict) -> str:
    """Setup and per-fold costs per log."""
    header = ["Step", *LOGS]
    step_names = list(results[LOGS[0]]["one_off_seconds"])
    rows = [[name] + [duration(results[log]["one_off_seconds"][name]["seconds"])
                      for log in LOGS] for name in step_names]
    for name in results[LOGS[0]]["per_fold_seconds"]:
        cells = []
        for log in LOGS:
            seconds = [record["seconds"]
                       for record in results[log]["per_fold_seconds"][name]]
            cells.append(f"{duration(statistics.median(seconds))} "
                         f"({min(seconds):.2f}-{max(seconds):.2f})")
        rows.append([f"{name}: median of 5 folds (min-max)"] + cells)
    return markdown_table(header, rows)


def per_iteration_row(log: str, group: dict) -> list:
    """One row of the seconds-per-(itemset, repeat) table."""
    original = group["original"]["seconds_per_iteration"]
    engine = {label: group[label]["seconds_per_iteration"]["median"]
              for label in ENGINE_ONE_FOLD}
    speedups = [f"{original['median'] / engine[label]:.0f}x"
                if engine[label] and original["median"] else "n/a"
                for label in ENGINE_ONE_FOLD]
    diff = group["faithful_vs_original_max_abs_diff"]
    cases = group["train_cases_with_itemset"]
    original_cell = "failed" if original["n"] == 0 else (
        f"{original['median']:.2f} s ({original['min']:.2f}-{original['max']:.2f}), "
        f"{original['mean']:.2f} s" + (f" [n = {original['n']}]"
                                       if original["n"] != 10 else ""))
    others = group["original"].get(
        "others_percent_runs", [group["original"]["load"]["others_percent"]])
    return [
        log, group["strategy"], group["size"], f"{min(cases)}-{max(cases)}",
        original_cell, " and ".join(f"{value:.0f} %" for value in others),
        *[f"{engine[label]:.4f} s" if engine[label] else "error"
          for label in ENGINE_ONE_FOLD],
        " / ".join(speedups), "n/a" if diff is None else f"{diff:g}",
    ]


def table_per_iteration(results: dict) -> str:
    """Seconds per (itemset, repeat) for every log, strategy and size."""
    header = ["Log", "Sets", "Size", "training cases containing a set (min-max)",
              "original: median (min-max), mean", "others' CPU load",
              "engine faithful, train", "engine fixed, train",
              "engine fixed, held-out",
              "speed-up (faithful / fixed train / fixed held-out)",
              "max abs diff faithful vs original"]
    rows = [per_iteration_row(log, group)
            for log in LOGS for group in results[log]["groups"]]
    return markdown_table(header, rows)


def table_repeat_runs(first_runs: dict) -> str:
    """The single runs of every log that was measured more than once."""
    header = ["Log", "Run", "Sets", "Size", "original: median (min-max)",
              "others' CPU load", "engine faithful, train", "engine fixed, train",
              "engine fixed, held-out"]
    rows = []
    for log in LOGS:
        runs = {"run1": first_runs[log], **extra_runs(log)}
        if len(runs) == 1:
            continue
        for name, content in runs.items():
            for group in content["groups"]:
                original = group["original"]["seconds_per_iteration"]
                rows.append([
                    log, name, group["strategy"], group["size"],
                    f"{original['median']:.2f} s ({original['min']:.2f}-"
                    f"{original['max']:.2f})",
                    f"{group['original']['load']['others_percent']:.0f} %",
                    *[f"{group[label]['seconds_per_iteration']['median']:.4f} s"
                      for label in ENGINE_ONE_FOLD],
                ])
    return markdown_table(header, rows) if rows else "(no repeated runs)"


def table_engine_per_iteration(grids: dict) -> str:
    """Engine seconds per (itemset, repeat) in the measured full grid."""
    header = ["Log", "Part", "Sets", "Size", *ENGINE_GRID.values()]
    rows = []
    for log in LOGS:
        if log not in grids:
            continue
        calls = pd.DataFrame(grids[log]["calls"])
        calls = calls[calls["error"].isna()].copy()
        calls["per_iteration"] = calls["seconds"] / calls["n_iterations"]
        for (part, strategy, size), group in calls.groupby(
                ["part", "strategy", "size"], sort=False):
            cells = []
            for variant in ENGINE_GRID:
                values = group.loc[group["variant"] == variant, "per_iteration"]
                cells.append(f"{values.median():.4f} s" if len(values) == N_FOLDS
                             else f"{len(values)} of {N_FOLDS} folds ran")
            rows.append([log, part, strategy, size, *cells])
    return markdown_table(header, rows)


def table_loads(results: dict, grid_sets: dict, singles: dict) -> str:
    """CPU load of the rest of the machine during every run.

    ``grid_sets`` maps a row label to the engine-grid results of that kind.
    """
    header = ["Run", "Log", "system load just before the start",
              "others' load during the run", "this process (cores used)"]
    rows = []
    for log in LOGS:
        loads = [group["original"]["load"] for group in results[log]["groups"]]
        others = []
        for group in results[log]["groups"]:
            original = group["original"]
            others += original.get("others_percent_runs",
                                   [original["load"]["others_percent"]])
        own = statistics.median(load["own_cores"] for load in loads)
        rows.append(["original itemset routine (groups of 10 iterations)", log,
                     "", f"{min(others):.0f}-{max(others):.0f} %", f"{own:.2f}"])
        for name, grids in grid_sets.items():
            if log in grids:
                grid = grids[log]
                rows.append([name, log,
                             f"{grid['system_load_before_percent']:.0f} %",
                             f"{grid['whole_run']['others_percent']:.0f} %",
                             f"{grid['whole_run']['own_cores']:.2f}"])
        if log in singles:
            single = singles[log]
            rows.append(["original single-activity pass", log,
                         f"{single['system_load_before_percent']:.0f} %",
                         f"{single['load_during_run']['others_percent']:.0f} %",
                         f"{single['load_during_run']['own_cores']:.2f}"])
    return markdown_table(header, rows)


# --------------------------------------------------------------------------
# Extrapolations
# --------------------------------------------------------------------------
def extrapolate_defaults(results: dict) -> tuple[str, dict]:
    """(a) the original defaults: 5 folds x the original's itemsets x 10 repeats."""
    header = ["Log", "itemsets at min_support 0.5", "iterations",
              "t original (size 2 / size 3)", "original: S + 5 F",
              "original: total", "original, if 10 itemsets (500 iterations)",
              *ENGINE_ONE_FOLD.values()]
    rows, numbers = [], {}
    for log in LOGS:
        one_fold = results[log]
        sizes = original_itemset_sizes(log)
        setup, fit = original_setup_and_fit(one_fold)
        per_size = {size: group_of(one_fold, "apriori", size)["original"]
                    ["seconds_per_iteration"]["median"] for size in SIZES}
        loop = N_FOLDS * N_REPEATS * sum(per_size[size] for size in sizes)
        total = setup + N_FOLDS * fit + loop
        # The same size mix scaled to ten itemsets (matters for f3 only).
        total_ten = setup + N_FOLDS * fit + loop * TOP_K / len(sizes)

        engine_setup, engine_fit = engine_setup_and_fit(one_fold)
        engine_totals = {}
        for label in ENGINE_ONE_FOLD:
            per_size_engine = {
                size: group_of(one_fold, "apriori", size)[label]
                ["seconds_per_iteration"]["median"] for size in SIZES}
            engine_totals[label] = (
                engine_setup + N_FOLDS * engine_fit
                + N_FOLDS * N_REPEATS * sum(per_size_engine[size] for size in sizes))
        rows.append([
            log,
            f"{len(sizes)} ({sizes.count(2)} of size 2, {sizes.count(3)} of size 3)",
            N_FOLDS * N_REPEATS * len(sizes),
            f"{per_size[2]:.2f} s / {per_size[3]:.2f} s",
            duration(setup + N_FOLDS * fit), duration(total), duration(total_ten),
            *[duration(engine_totals[label]) for label in ENGINE_ONE_FOLD],
        ])
        numbers[log] = {
            "itemset_sizes": sizes, "original_setup_seconds": setup,
            "original_fit_and_baseline_seconds": fit,
            "original_total_seconds": total,
            "original_total_seconds_if_10_itemsets": total_ten,
            "engine_total_seconds": engine_totals,
        }
    return markdown_table(header, rows), numbers


def extrapolate_project_grid(results: dict, grids: dict) -> tuple[str, str, dict]:
    """(b) 2 strategies x 3 sizes x 10 sets x 5 folds x 10 repeats.

    Returns the main table, a table that checks the extrapolation formula
    against the measured engine runs, and the numbers.
    """
    n_iterations = len(STRATEGIES) * len(SIZES) * TOP_K * N_FOLDS * N_REPEATS
    header = ["Log", "iterations", "original: total (extrapolated)",
              "original: Apriori half / IMPresseD half"]
    header += [f"{label}: loop MEASURED" for label in ENGINE_GRID.values()]
    header += ["engine: S + 5 F (measured)"]
    check_header = ["Log"] + [f"{label}: extrapolated / measured"
                              for label in ENGINE_GRID.values()]
    rows, check_rows, numbers = [], [], {}
    sums = {"original": 0.0, **{variant: 0.0 for variant in ENGINE_GRID}}
    for log in LOGS:
        one_fold, grid = results[log], grids.get(log)
        setup, fit = original_setup_and_fit(one_fold)
        halves = {}
        for strategy in STRATEGIES:
            medians = [group_of(one_fold, strategy, size)["original"]
                       ["seconds_per_iteration"]["median"] for size in SIZES]
            halves[strategy] = (None if None in medians
                                else N_FOLDS * N_REPEATS * TOP_K * sum(medians))
        original_total = (None if None in halves.values()
                          else setup + N_FOLDS * fit + sum(halves.values()))
        if original_total is not None:
            sums["original"] += original_total
        overhead = None if grid is None else grid_setup_and_fits(grid)
        row = [log, n_iterations, duration(original_total),
               " / ".join(duration(halves[s]) for s in STRATEGIES)]
        check_row = [log]
        numbers[log] = {"original_total_seconds": original_total,
                        "original_loop_seconds_by_strategy": halves,
                        "engine_setup_and_fits_seconds": overhead, "engine": {}}
        for variant in ENGINE_GRID:
            measured, failed = grid_loop_seconds(grid, "grid", variant)
            medians = [group_of(one_fold, strategy, size)[f"engine_{variant}"]
                       ["seconds_per_iteration"]["median"]
                       for strategy in STRATEGIES for size in SIZES]
            predicted = (None if None in medians
                         else N_FOLDS * N_REPEATS * TOP_K * sum(medians))
            row.append(duration(measured)
                       + (f" ({failed} of 30 calls failed)" if failed else ""))
            check_row.append(f"{duration(predicted)} / {duration(measured)}"
                             + (f" = {predicted / measured:.2f}"
                                if predicted and measured else ""))
            numbers[log]["engine"][variant] = {
                "measured_loop_seconds": measured, "failed_calls": failed,
                "extrapolated_loop_seconds": predicted}
            if measured is not None:
                sums[variant] += measured + overhead
        row.append(duration(overhead))
        rows.append(row)
        check_rows.append(check_row)
    rows.append(["all three (incl. S + 5 F)", 3 * n_iterations,
                 duration(sums["original"]), ""]
                + [duration(sums[variant]) for variant in ENGINE_GRID] + [""])
    numbers["all_logs_total_seconds"] = sums
    return (markdown_table(header, rows),
            markdown_table(check_header, check_rows), numbers)


def extrapolate_single(results: dict, grids: dict, singles: dict) -> tuple[str, dict]:
    """(c) all activities x 5 folds x 10 repeats."""
    header = ["Log", "activities", "iterations",
              "original: one pass (1 fold, 1 repeat), measured",
              "original: median per scored activity",
              "original: S + 5 F + 50 passes (extrapolated)"]
    header += [f"{label}: MEASURED" for label in ENGINE_GRID.values()]
    rows, numbers = [], {}
    for log in LOGS:
        one_fold, grid, single = results[log], grids.get(log), singles.get(log)
        setup, fit = original_setup_and_fit(one_fold)
        n_activities = one_fold["n_activities"]
        one_pass = median_scored = total = None
        note = ""
        if single is not None:
            one_pass = single["seconds_sum_measured"]
            median_scored = single["seconds_per_activity_by_kind"]["scored"]["median"]
            if not single["complete"]:
                # Unfinished pass: scale by the share of activities covered.
                one_pass *= n_activities / single["activities_covered"]
                note = (f" (scaled from the first {single['activities_covered']} "
                        "activities)")
            total = setup + N_FOLDS * fit + N_FOLDS * N_REPEATS * one_pass
        row = [log, n_activities, n_activities * N_FOLDS * N_REPEATS,
               duration(one_pass) + note, duration(median_scored), duration(total)]
        numbers[log] = {"original_one_pass_seconds": one_pass,
                        "original_total_seconds": total, "engine": {}}
        for variant in ENGINE_GRID:
            measured, failed = grid_loop_seconds(grid, "single", variant)
            if failed:
                measured = None
            row.append(duration(measured))
            numbers[log]["engine"][variant] = measured
        rows.append(row)
    return markdown_table(header, rows), numbers


def table_threads(threads: dict) -> str:
    """Engine speed and cores used for several XGBoost thread counts."""
    header = ["Log", "XGBoost n_jobs", "one model fit"]
    header += [f"{label}: s per iteration (cores used)"
               for label in ENGINE_GRID.values()]
    header += ["importance values equal to the default's"]
    rows = []
    for log, content in threads.items():
        by_jobs: dict = {}
        for record in content["records"]:
            by_jobs.setdefault(record["n_jobs"], {})[record["variant"]] = record
        for n_jobs, variants in by_jobs.items():
            first = next(iter(variants.values()))
            rows.append([
                log, n_jobs, duration(first["fit_seconds"]),
                *[f"{variants[v]['seconds_per_iteration_median']:.4f} s "
                  f"({variants[v]['own_cores_median']:.1f})" for v in ENGINE_GRID],
                "yes" if all(r["same_importance_as_default"]
                             for r in variants.values()) else "NO",
            ])
    return markdown_table(header, rows) if rows else "(not run)"


def table_presets(grids: dict, demos: dict, parallel: dict) -> tuple[str, dict]:
    """Engine wall time of a 'demo' and a 'full' setting.

    demo  1 fold seed, 5 folds, 2 repeats (measured: runs with tag _demo)
    full  10 fold seeds, 5 folds, 10 repeats = 10 x the measured one-seed run
    Both: T = n_seeds x (S + 5 F + sum of the loops that are switched on).
    ``parallel`` holds the runs in which the three logs ran at the same time
    with one XGBoost thread each; there the wall time is the slowest log.
    """
    header = ["Setting", "What is computed", *LOGS, "all three logs"]
    components = {
        "project grid, fixed, held-out fold": [("grid", "fixed_test")],
        "project grid, fixed, both folds": [
            ("grid", "fixed_test"), ("grid", "fixed_train")],
        "project grid, fixed both folds + faithful": [
            ("grid", variant) for variant in ENGINE_GRID],
        "project grid + single-activity analysis, all three settings": [
            (part, variant) for part in ("grid", "single") for variant in ENGINE_GRID],
    }
    presets = (
        # name, measured runs, number of fold seeds, logs run at the same time?
        ("demo: 1 fold seed x 5 folds x 2 repeats; default threads, logs one "
         "after the other (measured)", demos, 1, False),
        ("1 fold seed x 5 folds x 10 repeats; default threads, logs one after "
         "the other (measured)", grids, 1, False),
        ("1 fold seed x 5 folds x 10 repeats; n_jobs=1, three logs at the same "
         "time (measured)", parallel, 1, True),
        ("full: 10 fold seeds x 5 folds x 10 repeats; default threads, logs one "
         "after the other (10 x measured)", grids, 10, False),
        ("full: 10 fold seeds x 5 folds x 10 repeats; n_jobs=1, three logs at "
         "the same time (10 x measured)", parallel, 10, True),
    )
    rows, numbers = [], {}
    for name, source, n_seeds, at_same_time in presets:
        if any(log not in source for log in LOGS):
            continue
        for label, parts in components.items():
            seconds = {}
            for log in LOGS:
                loops = sum(grid_loop_seconds(source[log], part, variant)[0]
                            for part, variant in parts)
                seconds[log] = n_seeds * (grid_setup_and_fits(source[log]) + loops)
            wall = max(seconds.values()) if at_same_time else sum(seconds.values())
            rows.append([name, label, *[duration(seconds[log]) for log in LOGS],
                         duration(wall)])
            numbers[f"{name} | {label}"] = {**seconds, "all_three_wall": wall}
    return markdown_table(header, rows), numbers


def main() -> None:
    first_runs = {log: read_json(f"one_fold_{log}.json") for log in LOGS}
    missing = [log for log, content in first_runs.items() if content is None]
    if missing:
        raise SystemExit(f"run c1_original_vs_engine.py first for {missing}")
    # A log that was measured more than once enters the tables pooled.
    results = {log: pool_runs(first_runs[log], list(extra_runs(log).values()))
               for log in LOGS}
    grids = {log: content for log in LOGS
             if (content := read_json(f"engine_grid_{log}.json")) is not None}
    demos = {log: content for log in LOGS
             if (content := read_json(f"engine_grid_{log}_demo.json")) is not None}
    parallel = {log: content for log in LOGS
                if (content := read_json(f"engine_grid_{log}_nj1.json")) is not None}
    singles = {log: content for log in LOGS
               if (content := read_json(f"original_single_{log}.json")) is not None}
    threads = {log: content for log in LOGS
               if (content := read_json(f"threads_{log}.json")) is not None}

    defaults_table, defaults = extrapolate_defaults(results)
    grid_table, grid_check, grid_numbers = extrapolate_project_grid(results, grids)
    single_table, single_numbers = extrapolate_single(results, grids, singles)
    preset_table, preset_numbers = table_presets(grids, demos, parallel)

    sections = [
        ("T1. One-off steps and per-fold costs", table_one_off(results)),
        ("T2. Seconds per (itemset, repeat): one fold, 5 sets x 2 repeats per row",
         table_per_iteration(results)),
        ("T3. Engine seconds per (itemset, repeat) in the measured full grid "
         "(10 sets x 10 repeats per call; median over the 5 folds)",
         table_engine_per_iteration(grids)),
        ("T4. (a) Original defaults", defaults_table),
        ("T5. (b) Full project grid", grid_table),
        ("T5b. Check of the extrapolation formula on the engine (500 x sum of "
         "the six one-fold medians vs the measured loop)", grid_check),
        ("T6. (c) Single-activity analysis", single_table),
        ("T7. Machine load", table_loads(results, {
            "engine grid, 10 repeats, default threads": grids,
            "engine grid, 2 repeats (demo), default threads": demos,
            "engine grid, 10 repeats, n_jobs=1, three logs at the same time": parallel,
        }, singles)),
        ("T8. Engine wall time of a demo and a full setting", preset_table),
        ("T9. Logs measured twice: the single runs (T2 pools them)",
         table_repeat_runs(first_runs)),
        ("T10. XGBoost threads: fold 0, ten Apriori sets of size 2 x 10 repeats",
         table_threads(threads)),
    ]
    text = "\n\n".join(f"### {title}\n\n{table}" for title, table in sections)
    (RESULT_DIR / "tables.md").write_text(text + "\n", encoding="utf-8")
    (RESULT_DIR / "extrapolation.json").write_text(json.dumps({
        "original_defaults": defaults, "project_grid": grid_numbers,
        "single_activity": single_numbers, "presets": preset_numbers,
    }, indent=2), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
