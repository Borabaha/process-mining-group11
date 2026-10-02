"""Compare two strategies for selecting activity sets (pilot, prototype).

The project question is: does the importance of activities and of their
locations change when the activity sets come from IMPresseD instead of
Apriori? This module holds the measures used to answer it.

Part 1 - measures on plain Python objects (no files, no pandas needed):

``jaccard``             Jaccard similarity of two collections of activity sets
``overlap_at_k``        number of sets in both top-k lists
``mean_best_match``     soft similarity: how close is the nearest set of the
                        other strategy (1 = every set has an identical twin)
``rank_agreement``      Spearman and Kendall between two rankings
``activity_importance`` mean and max importance of the sets that contain an
                        activity

Part 2 - summary tables for the tidy files written by ``run_pilot.py``
(``importance_<log>.csv`` and ``sets_<log>.csv``).

Run:  python compare.py [--result-dir ../results/experiments/pilot]

Writes ``tables.md`` and one CSV per table into the result folder.

Three things to know before reading the numbers
-----------------------------------------------
* A set's importance is reported as the mean over the fold-seed means
  (experiment C9: never rank from one split).
* Both strategies are scored with the same models, folds and random streams.
  A set selected by BOTH strategies therefore has exactly the same importance
  in both, so "ranking agreement on the shared sets" can only compare the
  strategies' own selection ranks. The informative comparison of importance
  is at the activity level, where the sets around an activity differ.
* A set that occurs in no scored trace of a fold is "not measurable" there
  and is left out of the means (it is not counted as importance 0).
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = PROJECT_ROOT / "results" / "experiments" / "pilot"

STRATEGIES = ("apriori", "impressed")
SEPARATOR = ", "            # between activity names in a set label
MIN_TRACES_CHANGED = 3      # below this a set's importance rests on 1-2 traces
SET_KEYS = ["log", "score_on", "strategy", "length", "rank", "itemset"]


# --------------------------------------------------------------------------
# Part 1: measures
# --------------------------------------------------------------------------
def to_set(itemset) -> frozenset[str]:
    """An activity set as a frozenset; a label ``'a, b'`` is split."""
    if isinstance(itemset, str):
        return frozenset(itemset.split(SEPARATOR))
    return frozenset(itemset)


def set_label(itemset) -> str:
    """Canonical text form of an activity set: sorted names joined by ', '."""
    return SEPARATOR.join(sorted(to_set(itemset)))


def shared_sets(sets_a: Iterable, sets_b: Iterable) -> list[frozenset[str]]:
    """The sets that occur in both collections, in the order of ``sets_a``."""
    in_b = {to_set(items) for items in sets_b}
    return [to_set(items) for items in sets_a if to_set(items) in in_b]


def jaccard(sets_a: Iterable, sets_b: Iterable) -> float:
    """Jaccard similarity |A and B| / |A or B| of two collections of sets.

    Every activity set counts as one element, so {a, b} and {a, c} do not
    overlap at all. Returns NaN when both collections are empty.
    """
    a = {to_set(items) for items in sets_a}
    b = {to_set(items) for items in sets_b}
    union = a | b
    return len(a & b) / len(union) if union else float("nan")


def overlap_at_k(ranked_a: Iterable, ranked_b: Iterable, k: int) -> int:
    """Number of sets that are in the first ``k`` of both ranked lists."""
    return len(shared_sets(list(ranked_a)[:k], list(ranked_b)[:k]))


def activity_vocabulary(sets: Iterable) -> set[str]:
    """All activities that occur in at least one of the sets."""
    return {activity for items in sets for activity in to_set(items)}


def mean_best_match(sets_a: Iterable, sets_b: Iterable) -> float:
    """Soft similarity of two collections of sets, between 0 and 1.

    For every set, take the largest activity-level Jaccard with any set of
    the other collection; average these best matches over both collections.
    1 means every set has an identical twin, 0 means no set shares a single
    activity with the other collection.
    """
    a = [to_set(items) for items in sets_a]
    b = [to_set(items) for items in sets_b]
    if not a or not b:
        return float("nan")

    def best(one, others):
        return max(len(one & other) / len(one | other) for other in others)

    scores = [best(one, b) for one in a] + [best(one, a) for one in b]
    return float(np.mean(scores))


def rank_agreement(x: pd.Series, y: pd.Series) -> dict:
    """Spearman's rho and Kendall's tau-b between two rankings.

    ``x`` and ``y`` are indexed by the ranked objects (sets or activities);
    only objects present in both, with a value in both, are used. With fewer
    than 3 common objects, or when one side is constant, the coefficients
    are NaN.
    """
    both = pd.concat([x.rename("x"), y.rename("y")], axis=1, join="inner").dropna()
    result = {"n": len(both), "spearman": np.nan, "spearman_p": np.nan,
              "kendall": np.nan, "kendall_p": np.nan}
    if len(both) < 3 or both["x"].nunique() < 2 or both["y"].nunique() < 2:
        return result
    rho = stats.spearmanr(both["x"], both["y"])
    tau = stats.kendalltau(both["x"], both["y"])
    result.update(spearman=float(rho.statistic), spearman_p=float(rho.pvalue),
                  kendall=float(tau.statistic), kendall_p=float(tau.pvalue))
    return result


def activity_importance(set_importance: pd.Series) -> pd.DataFrame:
    """Aggregate set importances to activities.

    ``set_importance`` is indexed by set labels (``'a, b'``). Returns one row
    per activity (index) with ``n_sets`` (sets that contain it and have a
    value), ``mean`` and ``max`` of their importances, sorted by ``mean``
    (descending), ties by name.
    """
    rows = [
        (activity, value)
        for label, value in set_importance.dropna().items()
        for activity in to_set(label)
    ]
    long = pd.DataFrame(rows, columns=["activity", "importance"])
    table = long.groupby("activity")["importance"].agg(["count", "mean", "max"])
    table = table.rename(columns={"count": "n_sets"}).reset_index()
    table = table.sort_values(["mean", "activity"], ascending=[False, True])
    return table.set_index("activity")


# --------------------------------------------------------------------------
# Part 2: tables for the pilot files
# --------------------------------------------------------------------------
def load_pilot(result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read all ``importance_<log>.csv`` and ``sets_<log>.csv`` of a folder."""
    importance_files = sorted(result_dir.glob("importance_*.csv"))
    if not importance_files:
        raise FileNotFoundError(f"no importance_<log>.csv in {result_dir}")
    text_columns = {"log": str, "itemset": str}
    importance = pd.concat(
        [pd.read_csv(path, dtype=text_columns) for path in importance_files],
        ignore_index=True)
    sets = pd.concat(
        [pd.read_csv(path, dtype=text_columns)
         for path in sorted(result_dir.glob("sets_*.csv"))],
        ignore_index=True)
    return importance, sets


def seed_means(importance: pd.DataFrame) -> pd.DataFrame:
    """One row per set and fold seed: the mean over folds and repeats.

    Columns: the set keys, ``fold_seed``, ``importance`` (mean over the
    measurable rows; NaN if the set is in no scored trace of any fold),
    ``share_changed`` (changed traces / scored traces, all rows),
    ``traces_changed`` (mean number of changed traces per fold) and
    ``folds_absent`` (folds in which no scored trace contains the set).
    """
    frame = importance.copy()
    frame["value"] = frame["importance"].where(frame["measurable"])
    frame["share_changed"] = frame["n_traces_changed"] / frame["n_scored"]
    frame["absent"] = ~frame["measurable"]
    grouped = frame.groupby(SET_KEYS + ["fold_seed"], sort=False)
    table = grouped.agg(
        importance=("value", "mean"),
        share_changed=("share_changed", "mean"),
        traces_changed=("n_traces_changed", "mean"),
    )
    # A set is absent from a fold in all repeats or in none.
    absent_folds = frame[frame["absent"]].groupby(SET_KEYS + ["fold_seed"])["fold"]
    table["folds_absent"] = absent_folds.nunique().reindex(table.index, fill_value=0)
    return table.reset_index()


def set_summary(seed_table: pd.DataFrame) -> pd.DataFrame:
    """One row per set: mean, spread and support over the fold seeds.

    ``importance`` is the mean of the seed means, ``sd`` / ``se`` their
    standard deviation / standard error over the seeds, ``seeds_positive``
    the number of seeds with a mean above 0, ``importance_rank`` the rank
    inside the strategy's list of this log, length and scored fold
    (1 = most important) and ``few_traces`` flags sets whose value rests on
    fewer than ``MIN_TRACES_CHANGED`` changed traces per fold on average.
    """
    grouped = seed_table.groupby(SET_KEYS, sort=False)
    table = grouped.agg(
        importance=("importance", "mean"),
        sd=("importance", "std"),
        n_seeds=("importance", "count"),
        seeds_positive=("importance", lambda values: int((values > 0).sum())),
        share_changed=("share_changed", "mean"),
        traces_changed=("traces_changed", "mean"),
        folds_absent=("folds_absent", "sum"),
    ).reset_index()
    table["se"] = table["sd"] / np.sqrt(table["n_seeds"])
    table["few_traces"] = table["traces_changed"] < MIN_TRACES_CHANGED
    list_keys = ["log", "score_on", "strategy", "length"]
    table["importance_rank"] = (
        table.groupby(list_keys)["importance"].rank(ascending=False, method="min"))
    return table.sort_values(list_keys + ["rank"]).reset_index(drop=True)


def selection_overlap(sets: pd.DataFrame, ks=(5, 10)) -> pd.DataFrame:
    """Per log and length: how similar are the two selected lists?"""
    records = []
    for (log_name, length), group in sets.groupby(["log", "length"]):
        ranked = {
            strategy: group[group["strategy"] == strategy]
            .sort_values("rank")["itemset"].tolist()
            for strategy in STRATEGIES
        }
        a, b = ranked["apriori"], ranked["impressed"]
        vocabulary_a, vocabulary_b = activity_vocabulary(a), activity_vocabulary(b)
        record = {
            "log": log_name, "length": length,
            "n_apriori": len(a), "n_impressed": len(b),
            "n_shared": len(shared_sets(a, b)),
            "jaccard": jaccard(a, b),
        }
        record.update({f"overlap_at_{k}": overlap_at_k(a, b, k) for k in ks})
        record.update({
            "mean_best_match": mean_best_match(a, b),
            "activities_apriori": len(vocabulary_a),
            "activities_impressed": len(vocabulary_b),
            "activities_shared": len(vocabulary_a & vocabulary_b),
        })
        records.append(record)
    return pd.DataFrame.from_records(records)


def shared_set_table(set_table: pd.DataFrame) -> pd.DataFrame:
    """The sets selected by both strategies, with both sides' ranks.

    ``rank_*`` is the strategy's own selection rank, ``importance_rank_*``
    the rank of the set's importance inside that strategy's list of 10. The
    importance itself is the same number on both sides (same models).
    """
    keys = ["log", "score_on", "length", "itemset"]
    columns = keys + ["rank", "importance_rank", "importance", "se"]
    sides = {
        strategy: set_table.loc[set_table["strategy"] == strategy, columns]
        for strategy in STRATEGIES
    }
    merged = sides["apriori"].merge(
        sides["impressed"].drop(columns=["importance", "se"]),
        on=keys, suffixes=("_apriori", "_impressed"))
    ordered = keys + ["rank_apriori", "rank_impressed", "importance_rank_apriori",
                      "importance_rank_impressed", "importance", "se"]
    return merged[ordered].sort_values(keys[:3] + ["rank_apriori"]).reset_index(drop=True)


def shared_rank_agreement(shared: pd.DataFrame) -> pd.DataFrame:
    """Agreement of the two SELECTION rankings on the shared sets."""
    records = []
    # Selection ranks do not depend on the scored fold: keep each set once.
    one_fold = shared.drop_duplicates(["log", "length", "itemset"])
    for (log_name, length), group in one_fold.groupby(["log", "length"]):
        group = group.set_index("itemset")
        agreement = rank_agreement(group["rank_apriori"], group["rank_impressed"])
        records.append({"log": log_name, "length": length, **agreement})
    return pd.DataFrame.from_records(records)


def strategy_summary(importance: pd.DataFrame, seed_table: pd.DataFrame) -> pd.DataFrame:
    """Mean importance of each strategy's sets, and the paired difference.

    Per log, scored fold and length (plus ``length='all'``): the mean over
    the strategy's sets is taken inside every fold seed; reported are the
    mean and standard deviation of these seed values, and for the difference
    IMPresseD - Apriori (paired by fold seed, because both strategies share
    the folds and models) the mean, a 95 % t-interval over the seeds and the
    number of seeds in which IMPresseD is higher. The interval covers the
    randomness of the split and of the permutation only, not that of the
    data. ``*_share_positive`` is the share of single (fold, repeat) values
    above 0; ``*_share_changed`` the mean share of scored traces a
    permutation changes; ``*_per_changed`` = mean importance / that share.
    """
    pooled = seed_table.assign(length="all")
    seed_all = pd.concat([seed_table.astype({"length": object}), pooled])
    raw = importance[importance["measurable"]]
    raw_all = pd.concat([raw.astype({"length": object}), raw.assign(length="all")])
    positive = (raw_all.assign(positive=raw_all["importance"] > 0)
                .groupby(["log", "score_on", "length", "strategy"])["positive"].mean())

    records = []
    for (log_name, score_on, length), group in seed_all.groupby(
            ["log", "score_on", "length"], sort=False):
        per_seed = group.groupby(["fold_seed", "strategy"])["importance"].mean().unstack()
        share = group.groupby("strategy")["share_changed"].mean()
        record = {"log": log_name, "score_on": score_on, "length": length,
                  "n_seeds": len(per_seed)}
        for strategy in STRATEGIES:
            mean = per_seed[strategy].mean()
            record[f"{strategy}_mean"] = mean
            record[f"{strategy}_sd"] = per_seed[strategy].std()
            record[f"{strategy}_share_positive"] = positive.get(
                (log_name, score_on, length, strategy), np.nan)
            record[f"{strategy}_share_changed"] = share[strategy]
            record[f"{strategy}_per_changed"] = mean / share[strategy]
        difference = per_seed["impressed"] - per_seed["apriori"]
        record.update(_paired_interval(difference))
        records.append(record)
    return pd.DataFrame.from_records(records)


def _paired_interval(difference: pd.Series) -> dict:
    """Mean, 95 % t-interval and sign count of per-seed differences."""
    n = len(difference)
    mean, sd = difference.mean(), difference.std()
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
    return {"diff_mean": mean, "diff_sd": sd, "diff_ci_low": mean - half,
            "diff_ci_high": mean + half,
            "seeds_impressed_higher": int((difference > 0).sum())}


def _half_means(seed_table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean importance per ``keys`` for the first and second half of the seeds."""
    seeds = sorted(seed_table["fold_seed"].unique())
    first = set(seeds[: len(seeds) // 2])
    half = np.where(seed_table["fold_seed"].isin(first), "first", "second")
    return (seed_table.assign(half=half)
            .groupby(keys + ["half"])["importance"].mean().unstack("half"))


def set_ranking_checks(seed_table: pd.DataFrame, set_table: pd.DataFrame) -> pd.DataFrame:
    """Per strategy list: how stable is the importance ranking, what drives it?

    ``stability_*``: agreement between the ranking from the first half of
    the fold seeds and the ranking from the second half (the yardstick for
    every other correlation in this report). ``selection_*``: agreement
    between the strategy's selection rank and the importance rank (positive =
    sets ranked first by the strategy are also the more important ones).
    ``changed_spearman``: Spearman between the share of traces a permutation
    changes and the importance.
    """
    list_keys = ["log", "score_on", "strategy", "length"]
    halves = _half_means(seed_table, list_keys + ["itemset"])
    records = []
    for keys, group in set_table.groupby(list_keys, sort=False):
        group = group.set_index("itemset")
        half = halves.loc[keys]
        stability = rank_agreement(half["first"], half["second"])
        selection = rank_agreement(-group["rank"], group["importance"])
        changed = rank_agreement(group["share_changed"], group["importance"])
        records.append({
            **dict(zip(list_keys, keys)),
            "n_sets": len(group),
            "sets_few_traces": int(group["few_traces"].sum()),
            "stability_spearman": stability["spearman"],
            "stability_kendall": stability["kendall"],
            "selection_spearman": selection["spearman"],
            "selection_kendall": selection["kendall"],
            "changed_spearman": changed["spearman"],
        })
    return pd.DataFrame.from_records(records)


def activity_table(seed_table: pd.DataFrame, seeds=None) -> pd.DataFrame:
    """Activity-level importance per log, scored fold, strategy and scope.

    The set importances are first averaged over the fold seeds (all, or the
    given ``seeds``) and then aggregated with ``activity_importance``.
    ``scope`` is a length (1, 2, 3) or ``'all'`` (the sets of all lengths).
    """
    if seeds is not None:
        seed_table = seed_table[seed_table["fold_seed"].isin(list(seeds))]
    means = (seed_table.groupby(SET_KEYS, sort=False)["importance"].mean()
             .reset_index())
    scoped = pd.concat([means.assign(scope=means["length"].astype(str)),
                        means.assign(scope="all")])
    tables = []
    for keys, group in scoped.groupby(["log", "score_on", "strategy", "scope"], sort=False):
        table = activity_importance(group.set_index("itemset")["importance"]).reset_index()
        for name, value in zip(["scope", "strategy", "score_on", "log"], reversed(keys)):
            table.insert(0, name, value)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def activity_agreement(seed_table: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Per log, scored fold and scope: do the strategies rank activities alike?

    Compared on the activities that occur in sets of BOTH strategies:
    Spearman / Kendall of the mean-aggregated and of the max-aggregated
    importance, and the number of activities in both top-``top_n`` lists
    (by mean). ``stability_*`` is each strategy's own agreement between the
    two halves of the fold seeds (mean aggregation). At scope 1 an activity
    is a set, so the shared activities carry identical values on both sides
    and the correlation is 1 by construction.
    """
    seeds = sorted(seed_table["fold_seed"].unique())
    whole = activity_table(seed_table)
    first = activity_table(seed_table, seeds[: len(seeds) // 2])
    second = activity_table(seed_table, seeds[len(seeds) // 2:])
    keys = ["log", "score_on", "scope"]
    index = keys + ["strategy", "activity"]
    halves = pd.concat([first.set_index(index)["mean"].rename("first"),
                        second.set_index(index)["mean"].rename("second")], axis=1)
    records = []
    for group_keys, group in whole.groupby(keys, sort=False):
        side = {strategy: group[group["strategy"] == strategy].set_index("activity")
                for strategy in STRATEGIES}
        a, b = side["apriori"], side["impressed"]
        by_mean = rank_agreement(a["mean"], b["mean"])
        by_max = rank_agreement(a["max"], b["max"])
        record = {
            **dict(zip(keys, group_keys)),
            "activities_apriori": len(a), "activities_impressed": len(b),
            "activities_shared": by_mean["n"],
            "activity_jaccard": by_mean["n"] / len(a.index.union(b.index)),
            "spearman_mean": by_mean["spearman"], "kendall_mean": by_mean["kendall"],
            "spearman_max": by_max["spearman"], "kendall_max": by_max["kendall"],
            f"top{top_n}_shared": len(set(a.index[:top_n]) & set(b.index[:top_n])),
        }
        for strategy in STRATEGIES:
            half = halves.xs((*group_keys, strategy),
                             level=["log", "score_on", "scope", "strategy"])
            record[f"stability_{strategy}"] = rank_agreement(
                half["first"], half["second"])["spearman"]
        records.append(record)
    return pd.DataFrame.from_records(records)


def shared_activity_table(activities: pd.DataFrame, scope: str = "all") -> pd.DataFrame:
    """Activities that occur in sets of both strategies, side by side.

    One row per log, scored fold and activity with the number of sets and
    the mean / max importance under each strategy (suffix ``_apriori`` /
    ``_impressed``). These rows are the basis of ``activity_agreement``.
    """
    keys = ["log", "score_on", "activity"]
    scoped = activities[activities["scope"] == scope].drop(columns="scope")
    sides = [scoped[scoped["strategy"] == strategy].drop(columns="strategy")
             for strategy in STRATEGIES]
    merged = sides[0].merge(sides[1], on=keys, suffixes=("_apriori", "_impressed"))
    return merged.sort_values(keys[:2] + ["mean_apriori"],
                              ascending=[True, True, False]).reset_index(drop=True)


def top_activities(activities: pd.DataFrame, scope: str = "all", top_n: int = 5) -> pd.DataFrame:
    """The ``top_n`` activities per log, scored fold and strategy (by mean)."""
    scoped = activities[activities["scope"] == scope]
    top = scoped.groupby(["log", "score_on", "strategy"], sort=False).head(top_n).copy()
    top.insert(4, "position", top.groupby(["log", "score_on", "strategy"]).cumcount() + 1)
    return top.reset_index(drop=True)


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def to_markdown(frame: pd.DataFrame, digits: int = 4) -> str:
    """A DataFrame as a GitHub-style markdown table."""
    def cell(value) -> str:
        if isinstance(value, (bool, np.bool_)):
            return "yes" if value else "no"
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return ""
            return f"{value:.0f}" if float(value).is_integer() else f"{value:.{digits}f}"
        return str(value)

    lines = ["| " + " | ".join(str(column) for column in frame.columns) + " |",
             "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |"
              for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def build_tables(importance: pd.DataFrame, sets: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """All summary tables of the pilot, keyed by file stem."""
    seed_table = seed_means(importance)
    set_table = set_summary(seed_table)
    shared = shared_set_table(set_table)
    activities = activity_table(seed_table)
    return {
        "selection_overlap": selection_overlap(sets),
        "strategy_importance": strategy_summary(importance, seed_table),
        "shared_sets": shared,
        "shared_selection_rank_agreement": shared_rank_agreement(shared),
        "set_ranking_checks": set_ranking_checks(seed_table, set_table),
        "activity_agreement": activity_agreement(seed_table),
        "top_activities": top_activities(activities),
        "shared_activities": shared_activity_table(activities),
        "activity_importance": activities,
        "set_importance": set_table,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR,
                        help="folder with importance_<log>.csv and sets_<log>.csv")
    args = parser.parse_args()

    importance, sets = load_pilot(args.result_dir)
    tables = build_tables(importance, sets)
    sections = []
    for name, table in tables.items():
        table.to_csv(args.result_dir / f"{name}.csv", index=False, float_format="%.10g")
        sections.append(f"## {name}\n\n{to_markdown(table)}\n")
    (args.result_dir / "tables.md").write_text("\n".join(sections), encoding="utf-8")
    logs = ", ".join(sorted(importance["log"].unique()))
    print(f"{len(tables)} tables for {logs} written to {args.result_dir}")


if __name__ == "__main__":
    main()
