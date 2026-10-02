"""Experiment C8, part 2: tables and figures from the values of ``c8_run.py``.

Reads ``results/experiments/C8/values_<dataset>.csv`` (settings A-D) and, when
they exist, ``values_E_<dataset>.csv`` and ``values_F_<dataset>.csv`` (the
settings of ``c8_supplement.py``) and writes, in the same folder:

per_itemset_<dataset>.csv        mean, std and box statistics per setting and
                                 itemset (main fold seed and all fold seeds)
rank_correlations_<dataset>.csv  Spearman and Kendall between the settings
order_dependence_<dataset>.csv   setting A against setting D per itemset
box_ranges_by_fold_seed_<dataset>.csv  where the boxes lie, per fold seed
membership_<dataset>.csv         itemsets with / without an activity
summary_<dataset>.json           all summary numbers
tables_<dataset>.md              the same as Markdown tables
box_<dataset>_<setting>.png      horizontal box plot per setting (as Fig. 5 of
                                 the paper: sorted by mean, outliers hidden)
box_<dataset>_all_settings.png   all settings on one importance axis

Run:  python c8_report.py --dataset f1 [--main-fold-seed 0]

"Main fold seed" = the single 5-fold split that the per-itemset table and
the figures show (50 values per itemset = 5 folds x 10 repeats, as in the
original). The other fold seeds are used to see how much of a difference
between two settings is just the choice of the split.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write PNG files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cbook import boxplot_stats
from scipy import stats

RESULT_DIR = Path(__file__).resolve().parents[1] / "results" / "experiments" / "C8"
C2_DIR = RESULT_DIR.parent / "C2"
SETTING_TITLES = {
    "A_faithful_train": "A  faithful, training fold (original behaviour)",
    "B_fixed_train": "B  fixed, training fold",
    "C_fixed_test": "C  fixed, held-out fold",
    "D_faithful_train_reversed": "D  faithful, training fold, reversed itemset order",
    "E_original_shuffle_no_accumulation":
        "E  original shuffle without accumulation, training fold (supplement)",
    "F_fixed_shuffle_with_accumulation":
        "F  fixed shuffle with accumulation, training fold (supplement)",
}
# Settings of C2's full_grid_<dataset>.csv that must equal ours (fold seed 0).
C2_VARIANTS = {
    "A_faithful_train": "faithful_train",
    "B_fixed_train": "fixed_train",
    "C_fixed_test": "fixed_test",
}
BOX_COLOUR = "#2a78d6"
BOX_FILL = "#cfe0f6"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def load_values(dataset: str) -> pd.DataFrame:
    """The importance values of all settings that were computed for ``dataset``."""
    paths = [RESULT_DIR / f"values_{dataset}.csv"]
    paths += [RESULT_DIR / f"values_{key}_{dataset}.csv" for key in ("E", "F")]
    return pd.concat([pd.read_csv(path) for path in paths if path.exists()],
                     ignore_index=True)


def check_first_iteration(values: pd.DataFrame) -> dict | None:
    """Setting E against setting A on the first iteration of every fold.

    First itemset, repeat 0 is the only iteration in which setting A has not
    accumulated anything yet, so E (no accumulation) must give the same value.
    Returns None when setting E was not computed.
    """
    first = values[(values["position"] == 1) & (values["repeat"] == 0)]
    wide = first.pivot_table(index=["fold_seed", "fold"], columns="setting",
                             values="importance")
    if "E_original_shuffle_no_accumulation" not in wide.columns:
        return None
    diff = (wide["A_faithful_train"] - wide["E_original_shuffle_no_accumulation"]).abs()
    return {"folds_compared": len(diff), "max_abs_diff": float(diff.max())}


def itemset_means(values: pd.DataFrame) -> pd.DataFrame:
    """Mean importance: rows = itemsets (original rank), columns = settings."""
    return values.pivot_table(index="itemset_id", columns="setting",
                              values="importance", aggfunc="mean")[
        [label for label in SETTING_TITLES if label in set(values["setting"])]]


def per_itemset_table(values: pd.DataFrame) -> pd.DataFrame:
    """Mean, std and box statistics of the importance per setting and itemset.

    ``std`` is the sample standard deviation of all values of the itemset
    (folds x repeats pooled, as the original pools them for its box plot).
    ``whisker_low`` / ``whisker_high`` are the whisker ends of a box plot
    with ``whis=1.5`` (the convention of the paper's Fig. 5).
    """
    records = []
    for (setting, itemset_id), part in values.groupby(["setting", "itemset_id"]):
        box = boxplot_stats(part["importance"].to_numpy(), whis=1.5)[0]
        records.append({
            "setting": setting,
            "itemset_id": itemset_id,
            "itemset": part["itemset"].iloc[0],
            "position": int(part["position"].iloc[0]),
            "n_values": len(part),
            "mean": part["importance"].mean(),
            "std": part["importance"].std(ddof=1),
            "min": part["importance"].min(),
            "whisker_low": box["whislo"],
            "q1": box["q1"],
            "median": box["med"],
            "q3": box["q3"],
            "whisker_high": box["whishi"],
            "max": part["importance"].max(),
            "share_above_zero": (part["importance"] > 0).mean(),
            "mean_traces_changed": part["n_traces_changed"].mean(),
        })
    table = pd.DataFrame.from_records(records)
    table["rank"] = (table.groupby("setting")["mean"]
                     .rank(ascending=False, method="min").astype(int))
    return table


def correlate(first: pd.Series, second: pd.Series) -> dict:
    """Spearman and Kendall (tau-b) correlation of two itemset-mean vectors."""
    spearman = stats.spearmanr(first, second)
    kendall = stats.kendalltau(first, second)
    top3 = set(first.nlargest(3).index) & set(second.nlargest(3).index)
    return {
        "spearman": float(spearman.statistic), "spearman_p": float(spearman.pvalue),
        "kendall": float(kendall.statistic), "kendall_p": float(kendall.pvalue),
        "top3_in_common": len(top3),
    }


def rank_correlations(means: pd.DataFrame) -> pd.DataFrame:
    """Rank correlation of every pair of settings (one row per pair)."""
    return pd.DataFrame.from_records([
        {"first": a, "second": b, **correlate(means[a], means[b])}
        for a, b in itertools.combinations(means.columns, 2)
    ])


def correlations_per_fold_seed(values: pd.DataFrame) -> pd.DataFrame:
    """``rank_correlations`` computed separately inside every fold seed."""
    tables = []
    for fold_seed, part in values.groupby("fold_seed"):
        table = rank_correlations(itemset_means(part))
        table.insert(0, "fold_seed", fold_seed)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def agreement_between_fold_seeds(values: pd.DataFrame) -> pd.DataFrame:
    """Per setting: do two fold seeds rank the itemsets alike?

    Mean, minimum and maximum of the Spearman / Kendall correlation over all
    pairs of fold seeds. This is the noise reference: two SETTINGS cannot be
    expected to agree better than two SPLITS of the same setting do.
    """
    records = []
    for setting, part in values.groupby("setting"):
        means = part.pivot_table(index="itemset_id", columns="fold_seed",
                                 values="importance", aggfunc="mean")
        pairs = [correlate(means[a], means[b])
                 for a, b in itertools.combinations(means.columns, 2)]
        if not pairs:
            continue
        frame = pd.DataFrame.from_records(pairs)
        records.append({
            "setting": setting, "n_pairs_of_fold_seeds": len(pairs),
            "spearman_mean": frame["spearman"].mean(),
            "spearman_min": frame["spearman"].min(),
            "spearman_max": frame["spearman"].max(),
            "kendall_mean": frame["kendall"].mean(),
            "kendall_min": frame["kendall"].min(),
            "kendall_max": frame["kendall"].max(),
        })
    return pd.DataFrame.from_records(records)


def order_dependence(table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Setting A against setting D (same code, itemsets in reverse order).

    Returns the per-itemset comparison and, per setting, the Spearman
    correlation between the processing position (1 = processed first) and
    the mean importance. Without accumulation the processing order cannot
    matter: A and D would be identical and the correlation with the position
    would have opposite signs in A and D only by chance.
    """
    wide = table.pivot(index="itemset_id", columns="setting")
    comparison = pd.DataFrame({
        "itemset": wide["itemset"]["A_faithful_train"],
        "position_A": wide["position"]["A_faithful_train"],
        "mean_A": wide["mean"]["A_faithful_train"],
        "rank_A": wide["rank"]["A_faithful_train"],
        "position_D": wide["position"]["D_faithful_train_reversed"],
        "mean_D": wide["mean"]["D_faithful_train_reversed"],
        "rank_D": wide["rank"]["D_faithful_train_reversed"],
    })
    comparison["D_minus_A"] = comparison["mean_D"] - comparison["mean_A"]
    position_trend = {}
    for setting, part in table.groupby("setting"):
        result = stats.spearmanr(part["position"], part["mean"])
        position_trend[setting] = {"spearman_position_vs_mean": float(result.statistic),
                                   "p": float(result.pvalue)}
    return comparison.reset_index(), position_trend


def importance_by_repeat(values: pd.DataFrame) -> pd.DataFrame:
    """Mean importance by repeat number of the itemset each setting processes first.

    In faithful mode only repeat 0 of the first itemset starts from
    unpermuted traces; the climb over the repeats is the accumulation.
    """
    first = values[values["position"] == 1]
    return first.pivot_table(index="setting", columns="repeat", values="importance",
                             aggfunc="mean")


def importance_by_position(values: pd.DataFrame) -> pd.DataFrame:
    """Mean importance by processing position (1 = first), per setting."""
    return values.pivot_table(index="setting", columns="position",
                              values="importance", aggfunc="mean")


def box_ranges(table: pd.DataFrame) -> pd.DataFrame:
    """Per setting: where the boxes of the box plot lie (extremes over itemsets)."""
    return table.groupby("setting").agg(
        lowest_whisker=("whisker_low", "min"), lowest_q1=("q1", "min"),
        lowest_median=("median", "min"), highest_median=("median", "max"),
        highest_q3=("q3", "max"), highest_whisker=("whisker_high", "max"),
        lowest_mean=("mean", "min"), highest_mean=("mean", "max"))


def box_ranges_by_fold_seed(values: pd.DataFrame) -> pd.DataFrame:
    """``box_ranges`` of every fold seed (one row per fold seed and setting)."""
    tables = [box_ranges(per_itemset_table(part)).reset_index().assign(fold_seed=seed)
              for seed, part in values.groupby("fold_seed")]
    table = pd.concat(tables, ignore_index=True)
    return table[["fold_seed"] + [col for col in table.columns if col != "fold_seed"]]


def share_of_equal_values(values: pd.DataFrame, first: str, second: str) -> float:
    """Share of (fold seed, fold, itemset, repeat) cells equal in two settings."""
    keys = ["fold_seed", "fold", "itemset_id", "repeat"]
    wide = values[values["setting"].isin([first, second])].pivot(
        index=keys, columns="setting", values="importance")
    return float(np.isclose(wide[first], wide[second], rtol=0, atol=1e-12).mean())


def membership_effect(table: pd.DataFrame) -> pd.DataFrame:
    """Do itemsets that contain a certain activity score higher?

    Per setting and per activity that is in some but not all itemsets: the
    mean of the itemset means with the activity, without it, and the
    difference. The last rows do the same for the set size (3 against 2).
    A descriptive view of "what the ranking is about" in each setting.
    """
    members = table["itemset"].str.split(", ")
    groups = {f"contains {act}": members.apply(lambda items, act=act: act in items)
              for act in sorted(set(members.explode()))}
    groups["size 3 (against size 2)"] = members.str.len() == 3
    records = []
    for name, inside in groups.items():
        if inside.all() or not inside.any():
            continue
        for setting in table["setting"].unique():
            of_setting = table["setting"] == setting
            with_it = table.loc[of_setting & inside, "mean"]
            without = table.loc[of_setting & ~inside, "mean"]
            records.append({
                "group": name, "setting": setting, "n_itemsets_in_group": len(with_it),
                "mean_in_group": with_it.mean(), "mean_other_itemsets": without.mean(),
                "difference": with_it.mean() - without.mean(),
            })
    return pd.DataFrame.from_records(records)


def compare_with_c2(values: pd.DataFrame, dataset: str) -> dict | None:
    """Cross-check with C2's ``full_grid_<dataset>.csv`` (fold seed 0).

    C2 ran the same engine with default XGBoost threads; the values must be
    equal wherever the itemset and (in faithful mode) everything processed
    before it are the same. Returns None when there is nothing to compare.
    """
    path = C2_DIR / f"full_grid_{dataset}.csv"
    ours = values[values["fold_seed"] == 0]
    if not path.exists() or ours.empty:
        return None
    theirs = pd.read_csv(path)
    theirs = theirs[theirs["variant"].isin(C2_VARIANTS.values())]
    ours = ours.assign(variant=ours["setting"].map(C2_VARIANTS))
    keys = ["variant", "itemset_id", "itemset", "fold", "repeat"]
    merged = ours.merge(theirs, on=keys, suffixes=("", "_c2"))
    if merged.empty:
        return None
    merged["diff"] = (merged["importance"] - merged["importance_c2"]).abs()
    return {
        "rows_compared": len(merged),
        "rows_compared_per_setting": merged.groupby("setting").size().to_dict(),
        "max_abs_diff_importance_per_setting":
            merged.groupby("setting")["diff"].max().to_dict(),
        "max_abs_diff_baseline": float(
            (merged["baseline"] - merged["baseline_c2"]).abs().max()),
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def draw_boxes(ax, values: pd.DataFrame, order: list[int]) -> None:
    """Horizontal boxes of the importance; ``order[0]`` is drawn at the bottom."""
    data = [values.loc[values["itemset_id"] == itemset_id, "importance"].to_numpy()
            for itemset_id in order]
    labels = [values.loc[values["itemset_id"] == itemset_id, "itemset"].iloc[0]
              for itemset_id in order]
    ax.boxplot(
        data, orientation="horizontal", whis=1.5, showfliers=False, widths=0.6,
        patch_artist=True, showmeans=True,
        boxprops={"facecolor": BOX_FILL, "edgecolor": BOX_COLOUR, "linewidth": 1.2},
        medianprops={"color": BOX_COLOUR, "linewidth": 2},
        whiskerprops={"color": BOX_COLOUR, "linewidth": 1.2},
        capprops={"color": BOX_COLOUR, "linewidth": 1.2},
        meanprops={"marker": "o", "markerfacecolor": INK, "markeredgecolor": "white",
                   "markersize": 5},
    )
    ax.set_yticks(range(1, len(order) + 1), ["{" + label + "}" for label in labels])
    ax.axvline(0, color=MUTED, linestyle="--", linewidth=1)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def plot_setting(values: pd.DataFrame, setting: str, dataset: str, note: str) -> Path:
    """Box plot of one setting, itemsets sorted by mean (largest at the bottom).

    The sort order and the hidden outliers follow the original plotting code
    (CrossValidation_ProcessPermutation.py:104-105, Plotting_results.py).
    """
    part = values[values["setting"] == setting]
    order = part.groupby("itemset_id")["importance"].mean().sort_values(
        ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=(9, 5))
    draw_boxes(ax, part, order)
    ax.set_title(f"BPIC11 {dataset} - {SETTING_TITLES[setting]}", loc="left",
                 fontsize=11, color=INK)
    ax.set_xlabel(f"Decrease in weighted F1 (location importance)\n{note}",
                  fontsize=9, color=MUTED)
    fig.tight_layout()
    path = RESULT_DIR / f"box_{dataset}_{setting}.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def plot_all_settings(values: pd.DataFrame, dataset: str, note: str) -> Path:
    """All settings on one importance axis, same itemset order everywhere.

    Row order = the original's processing order (first processed at the top),
    so that setting D, which processes the itemsets bottom-up, can be read
    against setting A.
    """
    settings = [label for label in SETTING_TITLES if label in set(values["setting"])]
    order = sorted(values["itemset_id"].unique(), reverse=True)
    fig, axes = plt.subplots(len(settings), 1, sharex=True,
                             figsize=(9, 3.1 * len(settings)))
    for ax, setting in zip(np.atleast_1d(axes), settings):
        draw_boxes(ax, values[values["setting"] == setting], order)
        ax.set_title(SETTING_TITLES[setting], loc="left", fontsize=10, color=INK)
    np.atleast_1d(axes)[-1].set_xlabel(
        f"Decrease in weighted F1 (location importance)\n{note}",
        fontsize=9, color=MUTED)
    fig.suptitle(f"BPIC11 {dataset} - the same itemsets, folds and models in "
                 f"{len(settings)} settings (one common axis)", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.985))  # room for the figure title
    path = RESULT_DIR / f"box_{dataset}_all_settings.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    """A DataFrame as a Markdown table (floats with ``digits`` decimals)."""
    def cell(value) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}"
        return str(value)

    lines = ["| " + " | ".join(str(col) for col in frame.columns) + " |",
             "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |"
              for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def mean_std_table(table: pd.DataFrame) -> pd.DataFrame:
    """Wide table: one row per itemset, 'mean (std) [rank]' per setting."""
    table = table.assign(cell=[
        f"{mean:.4f} ({std:.4f}) [{rank}]"
        for mean, std, rank in zip(table["mean"], table["std"], table["rank"])])
    wide = table.pivot(index=["itemset_id", "itemset"], columns="setting",
                       values="cell")
    return wide[[s for s in SETTING_TITLES if s in wide.columns]].reset_index()


def summarise_seed_correlations(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Mean / min / max over the fold seeds of the between-setting correlations."""
    grouped = per_seed.groupby(["first", "second"], sort=False)
    return grouped.agg(
        spearman_mean=("spearman", "mean"), spearman_min=("spearman", "min"),
        spearman_max=("spearman", "max"), kendall_mean=("kendall", "mean"),
        kendall_min=("kendall", "min"), kendall_max=("kendall", "max"),
        top3_in_common_mean=("top3_in_common", "mean"),
    ).reset_index()


def analyse(values: pd.DataFrame, main_fold_seed: int) -> dict[str, pd.DataFrame]:
    """Every table of the report, keyed by a short name."""
    main_values = values[values["fold_seed"] == main_fold_seed]
    main_table = per_itemset_table(main_values)
    pooled_table = per_itemset_table(values)
    order_main, trend_main = order_dependence(main_table)
    order_pooled, trend_pooled = order_dependence(pooled_table)
    by_position = importance_by_position(values)
    level = values.groupby("setting")["importance"].agg(["mean", "std"]).join(
        (values["importance"] > 0).groupby(values["setting"]).mean()
        .rename("share_above_zero")).loc[list(by_position.index)]
    return {
        "main_table": main_table,
        "pooled_table": pooled_table,
        "main_corr": rank_correlations(itemset_means(main_values)),
        "pooled_corr": rank_correlations(itemset_means(values)),
        "seed_corr": summarise_seed_correlations(correlations_per_fold_seed(values)),
        "between_seeds": agreement_between_fold_seeds(values),
        "order_main": order_main,
        "order_pooled": order_pooled,
        "trend_main": pd.DataFrame(trend_main).T,
        "trend_pooled": pd.DataFrame(trend_pooled).T,
        "by_repeat": importance_by_repeat(values),
        "by_position": by_position,
        "level": level,
        "boxes_main": box_ranges(main_table),
        "boxes_by_seed": box_ranges_by_fold_seed(values),
        "membership": membership_effect(pooled_table),
    }


def write_csv_files(tables: dict, dataset: str, main_fold_seed: int) -> None:
    """Write the machine-readable tables of the report."""
    main_scope = f"fold seed {main_fold_seed}"
    pairs = {
        "per_itemset": ("main_table", "pooled_table"),
        "rank_correlations": ("main_corr", "pooled_corr"),
        "order_dependence": ("order_main", "order_pooled"),
    }
    for name, (main_key, pooled_key) in pairs.items():
        pd.concat([tables[main_key].assign(scope=main_scope),
                   tables[pooled_key].assign(scope="all fold seeds")]).to_csv(
            RESULT_DIR / f"{name}_{dataset}.csv", index=False, float_format="%.6f")
    tables["boxes_by_seed"].to_csv(
        RESULT_DIR / f"box_ranges_by_fold_seed_{dataset}.csv", index=False,
        float_format="%.6f")
    tables["membership"].to_csv(
        RESULT_DIR / f"membership_{dataset}.csv", index=False, float_format="%.6f")


def build_summary(tables: dict, values: pd.DataFrame, models: pd.DataFrame,
                  dataset: str, main_fold_seed: int, figures: list[Path]) -> dict:
    """The summary numbers of the report as a JSON-friendly dict."""
    quality = ["train_f1", "test_f1"]
    main_models = models[models["fold_seed"] == main_fold_seed]

    def records(key: str) -> list[dict]:
        return tables[key].to_dict(orient="records")

    return {
        "dataset": dataset,
        "fold_seeds": sorted(values["fold_seed"].unique().tolist()),
        "main_fold_seed": main_fold_seed,
        "n_folds": int(values["fold"].max()) + 1,
        "n_repeats": int(values["repeat"].max()) + 1,
        "model_f1_main_fold_seed": main_models[quality].mean().to_dict(),
        "model_f1_all_fold_seeds": models[quality].mean().to_dict(),
        "cross_check_with_C2": compare_with_c2(values, dataset),
        "check_E_equals_A_on_first_iteration": check_first_iteration(values),
        "level_all_fold_seeds": tables["level"].to_dict(orient="index"),
        "box_ranges_main_fold_seed": tables["boxes_main"].to_dict(orient="index"),
        "rank_correlations_main_fold_seed": records("main_corr"),
        "rank_correlations_all_fold_seeds_pooled": records("pooled_corr"),
        "rank_correlations_per_fold_seed": records("seed_corr"),
        "agreement_between_fold_seeds": records("between_seeds"),
        "position_trend_main_fold_seed": tables["trend_main"].to_dict(orient="index"),
        "position_trend_all_fold_seeds": tables["trend_pooled"].to_dict(orient="index"),
        "share_of_A_values_equal_to_D": share_of_equal_values(
            values, "A_faithful_train", "D_faithful_train_reversed"),
        "figures": [path.name for path in figures],
    }


def markdown_report(tables: dict, dataset: str, main_fold_seed: int,
                    n_fold_seeds: int) -> str:
    """All tables as one Markdown text."""
    main = f"fold seed {main_fold_seed}"

    def trend(key: str) -> str:
        return markdown_table(tables[key].rename_axis("setting").reset_index(), 3)

    sections = [
        (f"Per itemset, {main}: mean (std) [rank]",
         markdown_table(mean_std_table(tables["main_table"]))),
        (f"Per itemset, all {n_fold_seeds} fold seeds: mean (std) [rank]",
         markdown_table(mean_std_table(tables["pooled_table"]))),
        (f"Box statistics per setting, {main}",
         markdown_table(tables["boxes_main"].reset_index())),
        ("Box statistics per setting and fold seed",
         markdown_table(tables["boxes_by_seed"])),
        ("Level per setting, all fold seeds",
         markdown_table(tables["level"].reset_index())),
        (f"Rank correlations between settings, {main}",
         markdown_table(tables["main_corr"], 3)),
        ("Rank correlations between settings, itemset means over all fold seeds",
         markdown_table(tables["pooled_corr"], 3)),
        ("Rank correlations between settings inside one fold seed "
         "(mean / min / max over the fold seeds)",
         markdown_table(tables["seed_corr"], 3)),
        ("Agreement between two fold seeds inside one setting (noise reference)",
         markdown_table(tables["between_seeds"], 3)),
        (f"Order dependence (A vs D), {main}", markdown_table(tables["order_main"])),
        ("Order dependence (A vs D), all fold seeds",
         markdown_table(tables["order_pooled"])),
        (f"Spearman between processing position and mean importance, {main}",
         trend("trend_main")),
        ("Spearman between processing position and mean importance, all fold seeds",
         trend("trend_pooled")),
        ("Mean importance by processing position, all fold seeds",
         markdown_table(tables["by_position"].reset_index())),
        ("Mean importance by repeat of the itemset processed first, all fold seeds",
         markdown_table(tables["by_repeat"].reset_index())),
        ("Itemsets with / without an activity (itemset means over all fold seeds)",
         markdown_table(tables["membership"])),
    ]
    return f"# C8 tables - BPIC11 {dataset}\n\n" + "\n\n".join(
        f"## {title}\n\n{body}" for title, body in sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="f1")
    parser.add_argument("--main-fold-seed", type=int, default=0)
    args = parser.parse_args()
    dataset, main_fold_seed = args.dataset, args.main_fold_seed

    values = load_values(dataset)
    models = pd.read_csv(RESULT_DIR / f"models_{dataset}.csv")
    main_values = values[values["fold_seed"] == main_fold_seed]
    n_folds = int(values["fold"].max()) + 1
    n_repeats = int(values["repeat"].max()) + 1
    tables = analyse(values, main_fold_seed)

    note = (f"{n_folds} folds x {n_repeats} repeats = {n_folds * n_repeats} values "
            f"per itemset, fold seed {main_fold_seed}\n"
            "box = quartiles, whiskers = 1.5 IQR, outliers hidden (as in the "
            "paper's Fig. 5), dot = mean")
    figures = [plot_setting(main_values, setting, dataset, note)
               for setting in SETTING_TITLES if setting in set(values["setting"])]
    figures.append(plot_all_settings(main_values, dataset, note))

    write_csv_files(tables, dataset, main_fold_seed)
    summary = build_summary(tables, values, models, dataset, main_fold_seed, figures)
    (RESULT_DIR / f"summary_{dataset}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    text = markdown_report(tables, dataset, main_fold_seed,
                           values["fold_seed"].nunique())
    (RESULT_DIR / f"tables_{dataset}.md").write_text(text, encoding="utf-8")
    print(text)
    print(json.dumps({key: summary[key] for key in (
        "model_f1_main_fold_seed", "cross_check_with_C2",
        "check_E_equals_A_on_first_iteration", "share_of_A_values_equal_to_D")},
        indent=2))


if __name__ == "__main__":
    main()
