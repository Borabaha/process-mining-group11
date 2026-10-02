"""Experiment C12: our explicit case distance versus scipy pdist('jaccard') in two scipy versions.

Run from anywhere:
    python experiments/c12_compare_case_distance.py

Steps
1. Build the per-case attribute table of BPIC11 f1, f2, f3 (first value per
   case) and report missing values and distinct values per attribute.
2. Draw 200 random f1 cases (seed 2023).
3. Run the ORIGINAL 2023 distance code in two separate processes: with the
   installed scipy (1.18.1) and with scipy 1.11.4 + numpy 1.26.4 from the
   throw-away folder ``results/experiments/C12/tmp/scipy_1_11_4``.
4. Compare both with ``case_distance.pairwise_case_distance`` (pair level) and
   with ``case_distance.pattern_case_distance`` (pattern level, length-1
   patterns = single activities).

Outputs (``results/experiments/C12/``): ``c12_results.json``, ``c12_tables.md``,
``case_attributes_<log>.csv`` and the worker files in ``work/``.

scipy 1.11.4 must be installed once into the throw-away folder:
    python -m pip install scipy==1.11.4 numpy==1.26.4 --no-deps --only-binary=:all: \
        --target results/experiments/C12/tmp/scipy_1_11_4
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from paretoset import paretoset
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_classif

sys.path.insert(0, str(Path(__file__).resolve().parent))

from c12_scipy_worker import TINY_PAIRS, label_codes  # noqa: E402
from case_distance import (  # noqa: E402
    BPIC11_CATEGORICAL_COLS,
    BPIC11_NUMERIC_COLS,
    build_case_attribute_table,
    categorical_mismatch_share,
    numeric_minmax_euclidean,
    pairwise_case_distance,
    pattern_case_distance,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "external" / "PermutationLocationImportance" / "datasets"
RESULT_DIR = PROJECT_ROOT / "results" / "experiments" / "C12"
WORK_DIR = RESULT_DIR / "work"
OLD_SCIPY_DIR = RESULT_DIR / "tmp" / "scipy_1_11_4"
LOGS = {"f1": "BPIC11_f1_trunc36.csv", "f2": "BPIC11_f2_trunc40.csv", "f3": "BPIC11_f3_trunc31.csv"}
CASE, ACTIVITY, LABEL, ORDER = "case:concept:name", "concept:name", "label", "event_nr"
ATTRIBUTES = BPIC11_NUMERIC_COLS + BPIC11_CATEGORICAL_COLS
SEED = 2023
SAMPLE_SIZE = 200
N_SAMPLE_PATTERNS = 10
RARE_ACTIVITY_THRESHOLD = 2  # DataManager(frq_threshold=2) in the 2024 code
VARIANTS = ["explicit mismatch share (ours)", "original code @ scipy 1.11.4", "original code @ scipy 1.18.1",
            "one-hot Jaccard (alternative reading)"]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_log(csv_path):
    """Read one BPIC11 file the way the 2024 ``DataManager._load_df`` does.

    Activity names are lower-cased and stripped of ' ', '-', '_'; cases that
    contain an activity with fewer than ``RARE_ACTIVITY_THRESHOLD`` events in
    the whole file are removed (2024 tools.py:60-64).

    Returns the raw events and the events of the kept cases.
    """
    raw = pd.read_csv(csv_path)
    raw[CASE] = raw[CASE].astype(str)
    names = raw[ACTIVITY].str.lower()
    for character in (" ", "-", "_"):
        names = names.str.replace(character, "", regex=False)
    raw[ACTIVITY] = names
    raw[LABEL] = raw[LABEL].replace({"deviant": 1, "regular": 0}).astype(int)
    n_events = raw.groupby(ACTIVITY)[CASE].count()
    rare = n_events[n_events < RARE_ACTIVITY_THRESHOLD].index
    cases_to_drop = raw.loc[raw[ACTIVITY].isin(rare), CASE].unique()
    return raw, raw[~raw[CASE].isin(cases_to_drop)]


def attribute_report(raw, kept, table):
    """Missing values, distinct values and per-case constancy of every attribute."""
    rows = []
    for col in ATTRIBUTES:
        is_numeric = col in BPIC11_NUMERIC_COLS
        first_category = sorted(table[col].unique())[0]  # LabelEncoder gives code 0 to the first in sort order
        rows.append({
            "attribute": col,
            "kind": "numeric" if is_numeric else "categorical",
            "missing events (whole file)": int(raw[col].isna().sum()),
            "missing cases (table)": int(table[col].isna().sum()),
            "distinct values (whole file)": int(raw[col].nunique()),
            "distinct values (table)": int(table[col].nunique()),
            "max distinct values inside one case": int(kept.groupby(CASE)[col].nunique(dropna=False).max()),
            "category encoded as 0": "-" if is_numeric else str(first_category),
            "share of cases with code 0": "-" if is_numeric else round(float((table[col] == first_category).mean()), 4),
        })
    return rows


# --------------------------------------------------------------------------
# Emulations of the two scipy behaviours (to prove we understand them)
# --------------------------------------------------------------------------
def emulate_jaccard(codes, booleanise):
    """Condensed pdist(codes, 'jaccard') re-implemented for one scipy behaviour.

    ``booleanise=False`` (scipy 1.11.4): positions where both codes are 0 are
    ignored; among the others, the share with different codes.
    ``booleanise=True`` (scipy >= 1.15, here 1.18.1): codes are first turned
    into ``code != 0``, then the ordinary boolean Jaccard distance is taken.
    """
    values = (codes != 0) if booleanise else codes
    n_cases = len(values)
    n_different, n_not_both_zero = np.zeros((n_cases, n_cases)), np.zeros((n_cases, n_cases))
    for column, nonzero in zip(values.T, (codes != 0).T):
        n_different += column[:, None] != column[None, :]
        n_not_both_zero += nonzero[:, None] | nonzero[None, :]
    with np.errstate(invalid="ignore"):
        distances = np.where(n_not_both_zero > 0, n_different / n_not_both_zero, 0.0)
    return squareform(distances, checks=False)


# --------------------------------------------------------------------------
# Workers
# --------------------------------------------------------------------------
def start_worker(tag, prepend_path=None):
    """Run c12_scipy_worker.py in a fresh process and load its arrays."""
    out_path = WORK_DIR / f"worker_{tag}.npz"
    command = [sys.executable, str(Path(__file__).with_name("c12_scipy_worker.py")),
               "--in-dir", str(WORK_DIR), "--out", str(out_path), "--numeric-cols", *BPIC11_NUMERIC_COLS]
    if prepend_path:
        command += ["--prepend-path", str(prepend_path)]
    start = time.perf_counter()
    done = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(f"worker {tag}: exit code {done.returncode}, {time.perf_counter() - start:.1f} s")
    (WORK_DIR / f"worker_{tag}.stderr.txt").write_text(done.stderr, encoding="utf-8")
    if done.returncode != 0:
        raise RuntimeError(done.stderr[-3000:])
    with np.load(out_path) as archive:
        arrays = {key: archive[key] for key in archive.files if key != "info"}
        info = json.loads(str(archive["info"]))
    return info, arrays


# --------------------------------------------------------------------------
# Comparisons
# --------------------------------------------------------------------------
def max_abs_diff(a, b):
    """Largest absolute difference, NaN counted as equal to NaN."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return float("inf")
    return float(np.nanmax(np.abs(a - b), initial=0.0))


def compare_vectors(ours, other):
    """How close are two condensed distance vectors over the same case pairs?"""
    difference = np.abs(ours - other)
    return {
        "mean ours": round(float(ours.mean()), 4),
        "mean other": round(float(other.mean()), 4),
        "share of pairs identical": round(float((difference < 1e-12).mean()), 4),
        "mean abs diff": round(float(difference.mean()), 4),
        "max abs diff": round(float(difference.max()), 4),
        "Pearson r": round(float(pearsonr(ours, other)[0]), 4),
        "Spearman rho": round(float(spearmanr(ours, other)[0]), 4),
    }


def pair_level_comparison(table, name, new, old):
    """Compare our distance with both scipy runs for one case table.

    Returns the table rows, the consistency checks and two condensed combined
    distances: ours, and the one obtained when "Jaccard" is read as the set
    Jaccard distance of the one-hot encoded attributes, which equals
    ``2s / (1 + s)`` for a mismatch share ``s``.
    """
    codes = label_codes(table, BPIC11_CATEGORICAL_COLS)
    ours_categorical = squareform(categorical_mismatch_share(table, BPIC11_CATEGORICAL_COLS), checks=False)
    ours_numeric = squareform(numeric_minmax_euclidean(table, BPIC11_NUMERIC_COLS), checks=False)
    ours_combined = squareform(pairwise_case_distance(table, BPIC11_NUMERIC_COLS, BPIC11_CATEGORICAL_COLS),
                               checks=False)
    n_categorical = len(BPIC11_CATEGORICAL_COLS)
    one_hot_categorical = 2 * ours_categorical / (1 + ours_categorical)
    one_hot_combined = (n_categorical * one_hot_categorical + ours_numeric) / (1 + n_categorical)
    one_hot = pd.get_dummies(table[BPIC11_CATEGORICAL_COLS]).to_numpy(dtype=bool)

    rows = []
    for part, ours in (("categorical part", ours_categorical), ("combined distance", ours_combined)):
        key = "original_categorical" if part.startswith("categorical") else "original_combined"
        for label, arrays in (("scipy 1.11.4", old), ("scipy 1.18.1", new)):
            rows.append({"table": name, "quantity": part, "other": f"original code @ {label}",
                         **compare_vectors(ours, arrays[f"{key}|{name}"])})

    checks = {  # every entry is a maximum absolute difference and should be ~0
        "ours categorical vs pdist hamming @1.18.1": max_abs_diff(ours_categorical, new[f"hamming_on_codes|{name}"]),
        "ours categorical vs pdist hamming @1.11.4": max_abs_diff(ours_categorical, old[f"hamming_on_codes|{name}"]),
        "ours numeric vs original numeric @1.18.1": max_abs_diff(ours_numeric, new[f"original_numeric|{name}"]),
        "ours numeric vs original numeric @1.11.4": max_abs_diff(ours_numeric, old[f"original_numeric|{name}"]),
        "emulated 1.11.4 jaccard vs pdist jaccard @1.11.4": max_abs_diff(
            emulate_jaccard(codes, booleanise=False), old[f"jaccard_on_codes|{name}"]),
        "emulated 1.18.1 jaccard vs pdist jaccard @1.18.1": max_abs_diff(
            emulate_jaccard(codes, booleanise=True), new[f"jaccard_on_codes|{name}"]),
        "original categorical == pdist jaccard @1.11.4": max_abs_diff(
            old[f"original_categorical|{name}"], old[f"jaccard_on_codes|{name}"]),
        "original categorical == pdist jaccard @1.18.1": max_abs_diff(
            new[f"original_categorical|{name}"], new[f"jaccard_on_codes|{name}"]),
        # The combination formula: plug the original's own Jaccard into OUR numeric part and formula.
        "(m*jaccard + ours numeric)/(1+m) vs original combined @1.11.4": max_abs_diff(
            (n_categorical * old[f"jaccard_on_codes|{name}"] + ours_numeric) / (1 + n_categorical),
            old[f"original_combined|{name}"]),
        "(m*jaccard + ours numeric)/(1+m) vs original combined @1.18.1": max_abs_diff(
            (n_categorical * new[f"jaccard_on_codes|{name}"] + ours_numeric) / (1 + n_categorical),
            new[f"original_combined|{name}"]),
        "2s/(1+s) vs pdist jaccard on one-hot booleans @1.18.1": max_abs_diff(
            one_hot_categorical, pdist(one_hot, "jaccard")),
    }
    return rows, checks, {VARIANTS[0]: ours_combined, VARIANTS[3]: one_hot_combined}


def pattern_distances(dist_matrix, counts):
    """Case distance of every pattern column of ``counts`` (NaN when undefined)."""
    return np.array([pattern_case_distance(dist_matrix, counts[col].to_numpy() > 0) for col in counts.columns])


def pattern_level_comparison(name, counts, labels, distances_by_variant):
    """Length-1 patterns on one whole log: case distance and Pareto front per distance variant."""
    info_gain = mutual_info_classif(counts, labels, discrete_features=True)
    coverage = (counts > 0).mean().to_numpy()
    case_distance, fronts, front_sizes_distinct = {}, {}, {}
    for variant, condensed in distances_by_variant.items():
        case_distance[variant] = pattern_distances(squareform(condensed), counts)
        objectives = pd.DataFrame({"Outcome_Interest": info_gain, "Frequency_Interest": coverage,
                                   "Case_Distance_Interest": np.nan_to_num(case_distance[variant], nan=1.0)})
        sense = ["max", "max", "min"]
        fronts[variant] = set(counts.columns[paretoset(objectives, sense=sense, distinct=False)])
        front_sizes_distinct[variant] = int(paretoset(objectives, sense=sense, distinct=True).sum())

    ours = VARIANTS[0]
    result = {
        "log": name,
        "patterns (activities)": counts.shape[1],
        "patterns in all cases (distance undefined)": int(np.isnan(case_distance[ours]).sum()),
        "duplicate objective rows (ours)": int(pd.DataFrame(
            {"ig": info_gain, "cc": coverage, "cd": case_distance[ours]}).duplicated().sum()),
    }
    for variant in VARIANTS:
        values = case_distance[variant]
        result[f"{variant}: CD min / median / max"] = " / ".join(
            f"{x:.4f}" for x in (np.nanmin(values), np.nanmedian(values), np.nanmax(values)))
        result[f"{variant}: front size (distinct=False)"] = len(fronts[variant])
        result[f"{variant}: front size (distinct=True)"] = front_sizes_distinct[variant]
    for variant in VARIANTS[1:]:
        valid = ~np.isnan(case_distance[ours])
        result[f"ours vs {variant}: Spearman of pattern CD"] = round(float(
            spearmanr(case_distance[ours][valid], case_distance[variant][valid])[0]), 4)
        shared = fronts[ours] & fronts[variant]
        result[f"ours vs {variant}: front overlap"] = (
            f"{len(shared)} shared, Jaccard {len(shared) / len(fronts[ours] | fronts[variant]):.3f}")
    result["front members (ours)"] = ", ".join(sorted(fronts[ours]))
    return result


def markdown(rows):
    """Render a list of dicts as a markdown table."""
    header = list(rows[0])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(row[key]) for key in header) + " |" for row in rows]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    if not OLD_SCIPY_DIR.exists():
        raise SystemExit(f"scipy 1.11.4 is not installed in {OLD_SCIPY_DIR}; see the module docstring.")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    results, tables = {}, []
    start_all = time.perf_counter()

    # 1. attribute tables -------------------------------------------------
    case_tables, counts_by_log, labels_by_log, attribute_rows, log_rows = {}, {}, {}, [], []
    for name, file_name in LOGS.items():
        raw, kept = load_log(DATA_DIR / file_name)
        table = build_case_attribute_table(kept, ATTRIBUTES, CASE, ORDER)
        table.to_csv(RESULT_DIR / f"case_attributes_{name}.csv")
        table.to_csv(WORK_DIR / f"attributes_{name}.csv")
        case_tables[name] = table
        counts_by_log[name] = pd.crosstab(kept[CASE], kept[ACTIVITY]).loc[table.index]
        labels_by_log[name] = build_case_attribute_table(kept, [LABEL], CASE, ORDER)[LABEL].loc[table.index]
        ages = table["Age"]
        log_rows.append({
            "log": name, "events (file)": len(raw), "cases (file)": raw[CASE].nunique(),
            "cases after rare-activity filter": len(table), "activities after filter": kept[ACTIVITY].nunique(),
            "label=1 cases": int(labels_by_log[name].sum()),
            "Age min-max": f"{ages.min()}-{ages.max()}",
            "attribute values missing": int(table.isna().sum().sum()),
        })
        attribute_rows += [{"log": name, **row} for row in attribute_report(raw, kept, table)]
    results["logs"], results["attributes"] = log_rows, attribute_rows
    tables += ["### Logs\n", markdown(log_rows), "### Case attributes\n", markdown(attribute_rows)]

    # 2. the 200-case sample of f1 ----------------------------------------
    rng = np.random.default_rng(SEED)
    f1_table = case_tables["f1"]
    sample_rows = np.sort(rng.choice(len(f1_table), size=SAMPLE_SIZE, replace=False))
    sample = f1_table.iloc[sample_rows]
    sample.to_csv(WORK_DIR / "attributes_sample200_f1.csv")
    sample_counts = counts_by_log["f1"].loc[sample.index]
    support = (sample_counts > 0).sum().sort_values(ascending=False, kind="stable")
    frequent = list(support[support < SAMPLE_SIZE].index[:N_SAMPLE_PATTERNS])
    in_all = list(support[support == SAMPLE_SIZE].index[:1])  # distance undefined: no case without it
    in_none = list(support[support == 0].index[:1])  # distance undefined: no case with it
    sample_counts[frequent + in_all + in_none].to_csv(WORK_DIR / "patterns_sample200_f1.csv")

    # 3. original code under both scipy versions ---------------------------
    new_info, new = start_worker("scipy_installed")
    old_info, old = start_worker("scipy_1_11_4", OLD_SCIPY_DIR)
    if new_info["scipy"] == old_info["scipy"]:
        raise SystemExit("both workers imported the same scipy version - comparison is meaningless")
    results["worker_environments"] = {"installed": new_info, "throw-away": old_info}

    # 4a. tiny pairs --------------------------------------------------------
    tiny_rows = []
    for pair_name, rows in TINY_PAIRS.items():
        frame = pd.DataFrame(rows, columns=["a", "b", "c"])
        tiny_rows.append({
            "pair": pair_name,
            f"pdist jaccard, scipy {old_info['scipy']}": round(old_info["tiny_pairs"][pair_name]["jaccard"], 4),
            f"pdist jaccard, scipy {new_info['scipy']}": round(new_info["tiny_pairs"][pair_name]["jaccard"], 4),
            "explicit mismatch share (ours)": round(float(pairwise_case_distance(frame, [], list(frame))[0, 1]), 4),
            f"pdist hamming, scipy {old_info['scipy']}": round(old_info["tiny_pairs"][pair_name]["hamming"], 4),
            f"pdist hamming, scipy {new_info['scipy']}": round(new_info["tiny_pairs"][pair_name]["hamming"], 4),
            "emulate_jaccard(booleanise=False)": round(float(emulate_jaccard(np.array(rows), False)[0]), 4),
            "emulate_jaccard(booleanise=True)": round(float(emulate_jaccard(np.array(rows), True)[0]), 4),
        })
    results["tiny_pairs"] = tiny_rows
    tables += ["### Tiny pairs\n", markdown(tiny_rows)]

    # 4b. pair level: sample and whole logs ---------------------------------
    pair_rows, check_rows, computed = [], [], {}
    for name, table in {"sample200_f1": sample, **case_tables}.items():
        start = time.perf_counter()
        rows, checks, computed[name] = pair_level_comparison(table, name, new, old)
        print(f"pair-level comparison {name}: {len(table)} cases, {time.perf_counter() - start:.1f} s")
        pair_rows += rows
        check_rows += [{"table": name, "check (max abs difference, expected 0)": key, "value": f"{value:.2e}"}
                       for key, value in checks.items()]
    results["pair_level"], results["checks"] = pair_rows, check_rows
    tables += ["### Pair level\n", markdown(pair_rows), "### Consistency checks\n", markdown(check_rows)]

    # 4c. pattern level on the sample: original function vs ours ------------
    pattern_counts = pd.read_csv(WORK_DIR / "patterns_sample200_f1.csv", index_col=0)
    sample_pattern_rows = []
    variant_inputs = {VARIANTS[0]: computed["sample200_f1"][VARIANTS[0]],
                      VARIANTS[1]: old["original_combined|sample200_f1"],
                      VARIANTS[2]: new["original_combined|sample200_f1"]}
    ours_by_variant = {v: pattern_distances(squareform(d), pattern_counts) for v, d in variant_inputs.items()}
    for i, pattern in enumerate(pattern_counts.columns):
        sample_pattern_rows.append({
            "pattern (activity)": pattern,
            "cases with it (of 200)": int((pattern_counts[pattern] > 0).sum()),
            "original function @1.11.4": round(float(old["original_pattern_distance|sample200_f1"][i]), 6),
            "pattern_case_distance on the same matrix": round(float(ours_by_variant[VARIANTS[1]][i]), 6),
            "original function @1.18.1": round(float(new["original_pattern_distance|sample200_f1"][i]), 6),
            "pattern_case_distance on the same matrix ": round(float(ours_by_variant[VARIANTS[2]][i]), 6),
            "explicit distance (ours)": round(float(ours_by_variant[VARIANTS[0]][i]), 6),
        })
    results["sample_patterns"] = sample_pattern_rows
    results["sample_pattern_checks"] = {
        "max abs diff, pattern_case_distance vs original similarity_measuring_patterns @1.11.4": max_abs_diff(
            ours_by_variant[VARIANTS[1]], old["original_pattern_distance|sample200_f1"]),
        "max abs diff, pattern_case_distance vs original similarity_measuring_patterns @1.18.1": max_abs_diff(
            ours_by_variant[VARIANTS[2]], new["original_pattern_distance|sample200_f1"]),
        "warnings of the original function": new_info["original_pattern_warnings"],
    }
    tables += ["### Pattern level, 200-case sample\n", markdown(sample_pattern_rows)]

    # 4d. pattern level on the whole logs ------------------------------------
    whole_log_rows, timing_rows = [], []
    for name, table in case_tables.items():
        start = time.perf_counter()
        dist_matrix = pairwise_case_distance(table, BPIC11_NUMERIC_COLS, BPIC11_CATEGORICAL_COLS)
        seconds_matrix = time.perf_counter() - start
        start = time.perf_counter()
        pattern_distances(dist_matrix, counts_by_log[name])
        seconds_patterns = time.perf_counter() - start
        timing_rows.append({"log": name, "cases": len(table), "patterns": counts_by_log[name].shape[1],
                            "pairwise_case_distance (s)": round(seconds_matrix, 3),
                            "pattern_case_distance, all patterns (s)": round(seconds_patterns, 3),
                            "matrix size (MB)": round(dist_matrix.nbytes / 1e6, 1)})
        whole_log_rows.append(pattern_level_comparison(
            name, counts_by_log[name], labels_by_log[name],
            {VARIANTS[0]: computed[name][VARIANTS[0]], VARIANTS[1]: old[f"original_combined|{name}"],
             VARIANTS[2]: new[f"original_combined|{name}"], VARIANTS[3]: computed[name][VARIANTS[3]]}))
    results["whole_log_patterns"], results["timings"] = whole_log_rows, timing_rows
    transposed = [{"quantity": key, **{row["log"]: row[key] for row in whole_log_rows}}
                  for key in whole_log_rows[0] if key != "log"]
    tables += ["### Pattern level, whole logs (length-1 patterns)\n", markdown(transposed),
               "### Timings\n", markdown(timing_rows)]

    results["total_seconds"] = round(time.perf_counter() - start_all, 1)
    (RESULT_DIR / "c12_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    (RESULT_DIR / "c12_tables.md").write_text("\n".join(tables), encoding="utf-8")
    print("\n".join(tables))
    print(json.dumps(results["sample_pattern_checks"], indent=1))
    print(f"total {results['total_seconds']} s")


if __name__ == "__main__":
    main()
