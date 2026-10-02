# experiments/ — prototype code of Group 11

This folder holds the prototype modules and the experiment scripts of 2 October 2026.
It is **prototype code**: read it, check it, adapt it for the final package.

- What the experiments found: `results/experiments/RESULTS.md` (overview) and `results/experiments/<id>/RESULT.md` (one full report per experiment).
- This file only says how to run the tests and the experiments.

## 1. Before you start

- Run every command from the project root (`project/`).
- Always use the project's Python 3.12: `.venv/Scripts/python.exe`. Never the machine's default `python` (3.14).
- The commands below are for Git Bash:

  ```bash
  PY=.venv/Scripts/python.exe
  ```

  In PowerShell write `.\.venv\Scripts\python.exe` instead of `$PY`, and run the body of each `for` loop by hand.
- Packages are pinned in `requirements.txt` (numpy 2.5.3, pandas 2.3.3, scipy 1.18.1, scikit-learn 1.9.1, xgboost 3.4.1, mlxtend 0.25.0, networkx 3.7, paretoset 1.2.5, pm4py 2.7.23.8, matplotlib 3.11.2, PyYAML 6.0.3). Tested on Windows 10 only. Not yet tested on macOS.
- Data: `external/PermutationLocationImportance/datasets/BPIC11_f1_trunc36.csv`, `BPIC11_f2_trunc40.csv`, `BPIC11_f3_trunc31.csv`. In the scripts they are called `f1`, `f2`, `f3`.
- `external/` is read-only. Never edit the two original repositories.
- Both original repositories have a module named `tools.py`. Do not import them by hand. Use `engine_reference.py` (2024 code, loaded as `tools_2024`) and the drivers `run_original_impressed.py` / `c4_crosscheck_original.py` (2023 code).
- Never import `GUI_IMPresseD_tool.py`; it opens a window.
- `pytest` is not installed. The tests are `unittest` files or plain scripts.

## 2. Run the tests

```bash
$PY -m unittest experiments/test_impressed_chain.py -v      # 34 tests, 5 s
$PY -m unittest experiments/test_case_distance.py -v        # 18 tests, < 1 s
$PY experiments/test_engine.py --quick                      # 18 checks, 3 min; ends with "18 passed, 0 failed"
```

Slow checks that run the original 2024 code:

```bash
$PY experiments/test_engine.py                              # all 23 checks, about 40 min
$PY experiments/test_engine.py --full-single --only faithful_single_full_f1   # about 70 min
```

`test_engine.py` writes its result file into `results/experiments/C2/` (`test_engine_results.json`, with `--quick`: `test_engine_results_quick.json`). Exit code 1 means a check failed.

The independent reviewers' scripts are next to the results:

```bash
$PY results/experiments/C2/verify_engine.py --part fixed --dataset f1   # parts: encoder, faithful, single, fixed, toy, deadend, noise, hashseed
$PY results/experiments/C4/verify_impressed.py                          # about 6 min
```

Run the three fast test commands after every change to `engine.py`, `impressed_chain.py` or `case_distance.py`.

## 3. A two-minute tour

```bash
$PY experiments/engine.py --dataset f1 --mode fixed --score-on test   # small demo: one fold, three sets, prints a table, writes nothing
```

The pieces fit together like this (put the script in `experiments/`, or add this folder to `sys.path`):

```python
from xgboost import XGBClassifier

from apriori_selector import AprioriSelector
from c4_run_selector import load_selector_inputs
from engine import DATASETS, EventLog, IndexEncoder, LocationPermutationImportance, make_folds
from impressed_chain import ImpressedChainSelector

log = EventLog(DATASETS["f3"])                       # same cleaning and filter as the 2024 DataManager
encoder = IndexEncoder(log.traces.values(), log.activities)

# activity sets: {1: [[...], ...], 2: [...], 3: [...]} from both strategies
apriori_sets = AprioriSelector(min_support=0.45, max_len=3, top_k=10).select(log.traces)
_, dist_matrix, _ = load_selector_inputs("f3")       # explicit case distance, in the order of log.traces
selector = ImpressedChainSelector(max_gap=3, steps=2, k=10).fit(log.traces, log.labels, dist_matrix)
impressed_sets = selector.select()

# one fold of one seeded 5-fold split, one model for all sets
fold_seed, fold = 0, 0
train_cases, test_cases = make_folds(log.labels, k=5, seed=fold_seed)[fold]
model = XGBClassifier(n_jobs=1)
model.fit(encoder.transform(log.traces[c] for c in train_cases), [log.labels[c] for c in train_cases])

engine = LocationPermutationImportance(mode="fixed", score_on="test", n_repeats=10,
                                       random_state=2023 + fold_seed)
table = engine.compute(model, log, encoder, train_cases, test_cases,
                       apriori_sets[2] + impressed_sets[2], fold=fold)
print(table.groupby("itemset")["importance"].mean())
```

This snippet was run once for this README (f3, 2 repeats) and works as written.

## 4. What is in this folder

### Modules to reuse

| File | Content | Checked how |
|---|---|---|
| `engine.py` | `EventLog` (loader), `IndexEncoder`, `make_folds`, `LocationPermutationImportance` (`compute` for sets, `compute_single_activities`), modes `faithful` and `fixed` | 23 checks in `test_engine.py`; passed an independent review (round 2) |
| `engine_reference.py` | loads the original 2024 `tools.py` read-only as `tools_2024`, with the int cast pandas 2 needs | used by all equality checks |
| `apriori_selector.py` | `AprioriSelector(min_support, max_len=3, top_k=10)` and `original_top10(traces, min_support, top_k)` | equal to the original function in 24 of 24 comparisons |
| `case_distance.py` | `build_case_attribute_table`, `pairwise_case_distance`, `pattern_case_distance` | 18 unit tests; equal to the original's numeric part and pattern mean |
| `impressed_chain.py` | `Pattern`, `extend_pattern(s)`, `score_patterns`, `pareto_front`, `pareto_layers`, `ImpressedChainSelector` | 34 unit tests; identical to the original 2023 code; passed an independent review |
| `existence_importance.py` | `ExistenceImportance` (existence baseline), `encode_existence`, `load_log`, `apriori_itemsets` | no unit tests, no second reviewer |
| `compare.py` | Jaccard, overlap@k, best match, Spearman / Kendall, activity-level means; builds the pilot tables | toy checks only |
| `run_pilot.py`, `pilot.yaml` | the end-to-end pilot and its settings | values equal those of C9 (largest difference 5e-11) |
| `original_impressed_patched/` | copy of the 2023 code with three speed patches; see `patches.diff` | output identical to the original where both ran |

### Tests

`test_engine.py`, `test_impressed_chain.py`, `test_case_distance.py` (section 2).

### Experiment scripts

| Experiment | Scripts |
|---|---|
| C1 runtime | `c1_common.py`, `c1_original_vs_engine.py`, `c1_engine_grid.py`, `c1_original_single.py`, `c1_threads.py`, `c1_report.py` |
| C2 engine | `engine_benchmark.py`, `engine_full_grid.py`, `engine_stability.py`, `engine_fixed_options.py` |
| C3 original IMPresseD | `run_original_impressed.py`, `summarise_original_impressed.py`, `compare_impressed_runs.py` |
| C4 chain-only IMPresseD | `c4_crosscheck_original.py`, `c4_run_selector.py` |
| C5 / C6 sensitivity | `c5_oracle_blocks.py`, `c5_c6_sensitivity.py` |
| C7 Apriori | `run_c7_apriori_sweep.py` |
| C8 faithful vs fixed | `c8_run.py`, `c8_supplement.py`, `c8_surplus_events.py`, `c8_report.py` |
| C9 seed stability | `c9_seed_stability.py`, `c9_analysis.py`, `c9_project_sets.py`, `c9_overview.py` |
| C10 existence baseline | `run_c10.py` |
| C11 / C12 Pareto library, case distance | `c11_paretoset_versions.py`, `c12_scipy_worker.py`, `c12_compare_case_distance.py` |
| Pilot | `run_pilot.py`, `pilot.yaml`, `compare.py` |

## 5. Re-run the experiments

**A re-run overwrites the files in `results/experiments/<id>/`.** Copy the folder first if you want to keep the old numbers.
Times were measured on a shared machine; expect them to differ.

**Order.** Run C7 before anything that needs Apriori set files, and C4 before anything that needs IMPresseD set files (C1, C5, C9 with `--itemsets project`, pilot). The pilot cross-check needs the C9 raw files. The C4 reviewer script needs the C3 exports.

```bash
# ---- C7: Apriori sweep and Apriori set lists (1-2 min) -------------------
$PY experiments/run_c7_apriori_sweep.py

# ---- C11 / C12: paretoset versions, case distance (30 s + 40 s) -----------
# The two pip lines are only needed if the tmp folders were deleted. They install
# old versions into a throw-away folder, not into the venv.
$PY -m pip install paretoset==1.2.0 --target results/experiments/C11/tmp/paretoset_1_2_0 --no-deps
$PY -m pip install scipy==1.11.4 numpy==1.26.4 --target results/experiments/C12/tmp/scipy_1_11_4 --no-deps --only-binary=:all:
$PY experiments/c11_paretoset_versions.py
$PY experiments/c12_compare_case_distance.py

# ---- C2: engine speed, grids, stability, options -------------------------
$PY experiments/engine_benchmark.py --dataset f1 --repeats 2 --timing-runs 3    # about 3 min
for d in f1 f2 f3; do $PY experiments/engine_full_grid.py --dataset $d; done
for d in f1 f2 f3; do $PY experiments/engine_stability.py --dataset $d --seeds 10 --repeats 30; done       # 36-53 min per log
for d in f1 f2 f3; do $PY experiments/engine_fixed_options.py --dataset $d --seeds 10 --repeats 10; done   # 11-27 min per log

# ---- C3: the original IMPresseD code -------------------------------------
# original code, full f1, one extension step (27 min)
$PY -u experiments/run_original_impressed.py --max-extension-step 1 --run-name f1_step1 \
    --time-limit-min 45 --patterns-json results/experiments/C3/original_patterns_f1.json \
    > results/experiments/C3/logs/f1_step1.log 2>&1
# original code, 120 / 300 cases, two steps (3 / 6 min)
$PY -u experiments/run_original_impressed.py --max-extension-step 2 --max-cases 120 --run-name smoke120
$PY -u experiments/run_original_impressed.py --max-extension-step 2 --max-cases 300 --run-name scale300_step2 --time-limit-min 30
# patched copy: same commands plus --code-dir (full raw f1, two steps: 2.6 min)
$PY -u experiments/run_original_impressed.py --max-extension-step 2 --max-cases 120 --run-name smoke120_patched \
    --code-dir experiments/original_impressed_patched
$PY -u experiments/run_original_impressed.py --max-extension-step 2 --run-name f1_step2_patched --time-limit-min 40 \
    --code-dir experiments/original_impressed_patched
# patched copy on the 2024-filtered logs (f1 shown; same for BPIC11_f2_trunc40.csv and BPIC11_f3_trunc31.csv)
$PY -u experiments/run_original_impressed.py \
    --dataset external/PermutationLocationImportance/datasets/BPIC11_f1_trunc36.csv --log-prep dm2024 \
    --max-extension-step 2 --run-name f1_dm2024_step2_patched --code-dir experiments/original_impressed_patched \
    --time-limit-min 15 --patterns-json results/experiments/C3/patched_patterns_f1_dm2024.json
# equality check and tables
$PY experiments/compare_impressed_runs.py results/experiments/C3/smoke120 results/experiments/C3/smoke120_patched
$PY experiments/summarise_original_impressed.py results/experiments/C3/f1_step1/timings.json results/experiments/C3/original_patterns_f1.json

# ---- C4: chain-only IMPresseD ---------------------------------------------
$PY -u experiments/c4_crosscheck_original.py --full-core ac370000 --full-front > results/experiments/C4/crosscheck_run.log 2>&1   # 15 min
$PY -u experiments/c4_run_selector.py > results/experiments/C4/selector_run.log 2>&1                                             # 24 s, writes the IMPresseD set lists

# ---- C5 / C6: sensitivity of the IMPresseD side ---------------------------
$PY -u experiments/c5_oracle_blocks.py  > results/experiments/C5/c5a_run.log  2>&1    # 1.5 min
$PY -u experiments/c5_c6_sensitivity.py > results/experiments/C5/c5c6_run.log 2>&1    # 5 min
$PY results/experiments/C5/check_gap_counts.py

# ---- C10: existence baseline (9 min) -------------------------------------
$PY experiments/run_c10.py > results/experiments/C10/run.log 2>&1

# ---- C1: runtime ---------------------------------------------------------
for d in f1 f2 f3; do $PY experiments/c1_original_vs_engine.py --dataset $d; done                        # 26 min in total
$PY experiments/c1_original_vs_engine.py --dataset f2 --tag _run2                                        # second f2 measurement, 12 min
for d in f1 f2 f3; do $PY experiments/c1_engine_grid.py --dataset $d; done                               # 22 min in total
for d in f1 f2 f3; do $PY experiments/c1_engine_grid.py --dataset $d --repeats 2 --tag _demo; done       # 4 min
for d in f1 f2 f3; do $PY experiments/c1_engine_grid.py --dataset $d --n-jobs 1 --tag _nj1; done         # may be started in three terminals at once
$PY experiments/c1_threads.py --dataset f1
$PY experiments/c1_threads.py --dataset f3
for d in f1 f2 f3; do $PY experiments/c1_original_single.py --dataset $d; done                           # 15 / 30 / 11 min
$PY experiments/c1_report.py

# ---- C8: faithful against fixed ------------------------------------------
for d in f1 f2 f3; do $PY experiments/c8_run.py --dataset $d > results/experiments/C8/run_$d.log 2>&1; done                       # 17-29 min per log
for d in f1 f2 f3; do $PY experiments/c8_supplement.py --dataset $d > results/experiments/C8/run_supplement_$d.log 2>&1; done     # 8-12 min per log
$PY experiments/c8_surplus_events.py > results/experiments/C8/surplus_events.log 2>&1
for d in f1 f2 f3; do $PY experiments/c8_report.py --dataset $d > results/experiments/C8/report_$d.log 2>&1; done

# ---- C9: seed stability --------------------------------------------------
# main study, f1, 20 fold seeds (the four calls may run side by side; about 165 s per fold seed)
for S in 0 5 10 15; do $PY experiments/c9_seed_stability.py --dataset f1 --first-seed $S --n-seeds 5; done
# f1 with 10 folds
for S in 0 5; do $PY experiments/c9_seed_stability.py --dataset f1 --folds 10 --first-seed $S --n-seeds 5 --repeats 10 --faithful-repeats 0; done
# f2 and f3, original sets
for d in f2 f3; do for S in 0 5; do $PY experiments/c9_seed_stability.py --dataset $d --first-seed $S --n-seeds 5 --repeats 10 --faithful-repeats 10; done; done
# the project's 60 sets (needs the C7 and C4 set lists); f1 with 20 seeds, f2 and f3 with 10
for S in 0 5 10 15; do $PY experiments/c9_seed_stability.py --dataset f1 --itemsets project --first-seed $S --n-seeds 5 --repeats 10 --faithful-repeats 0; done
for d in f2 f3; do for S in 0 5; do $PY experiments/c9_seed_stability.py --dataset $d --itemsets project --first-seed $S --n-seeds 5 --repeats 10 --faithful-repeats 0; done; done
# analysis
$PY experiments/c9_analysis.py --dataset f1
$PY experiments/c9_analysis.py --dataset f1 --max-seeds 10
$PY experiments/c9_analysis.py --dataset f1 --folds 10
$PY experiments/c9_analysis.py --dataset f2
$PY experiments/c9_analysis.py --dataset f3
$PY experiments/c9_project_sets.py --dataset f1
$PY experiments/c9_project_sets.py --dataset f1 --max-seeds 10
$PY experiments/c9_project_sets.py --dataset f2
$PY experiments/c9_project_sets.py --dataset f3
$PY experiments/c9_overview.py

# ---- Pilot: Apriori sets against IMPresseD sets --------------------------
for d in f1 f2 f3; do $PY -u experiments/run_pilot.py --logs $d > results/experiments/pilot/run_$d.log 2>&1; done   # 16 / 22 / 13 min
$PY experiments/run_pilot.py --logs f1 f2 f3 --figures-only    # redraw the box plots, 5 s
$PY experiments/compare.py                                     # tables, 2 s
$PY results/experiments/pilot/check_against_c9.py              # needs the C9 raw files
```

Notes:
- Long runs: start them in a separate terminal and send the output to a log file, as shown.
- The three logs can run at the same time when XGBoost uses one thread (`n_jobs=1`, the default in `c8_run.py`, `c9_seed_stability.py` and `pilot.yaml`). Do not start several processes with XGBoost's default thread count; each would try to use all cores.
- The pilot's settings are in `pilot.yaml` (logs, lengths, k, folds, fold seeds, repeats, mode, scored folds, where the set files are). Change the file, or pass `--config other.yaml`.
- The reports of C1, C2, C8, C9 and the pilot give their commands from inside `experiments/`. The commands above are the same calls written from the project root. The scripts find their data from their own location, so both ways work.

## 6. Conventions used in the code

- Paths are built with `pathlib` from the script's own location. No absolute paths in code.
- Seeds are fixed: fold seeds through `make_folds(labels, k=5, seed=...)`; permutation seed 2023 (faithful mode) or 2023 + fold seed (fixed mode); samples in the checks use seed 2023.
- "Importance" always means: weighted F1 of the scored fold before the permutation minus after it.
- Activity names are lower case without spaces, `-` and `_` (`engine.normalise_activity`), as in the 2024 code.
- A model must be fitted on the matrix of the same `IndexEncoder` object that is later passed to `compute`.

## 7. Known traps (not yet fixed in the code)

`engine.py`
1. `LocationPermutationImportance(mode='faithful')` defaults to `score_on='test'`. The original code scores the training fold. Always write `mode='faithful', score_on='train'`.
2. A set that occurs in no scored trace gets importance 0 and a warning. Treat such rows as "not measurable" in the analysis (`n_traces_with_itemset == 0`).
3. The random streams of fixed mode depend on (`random_state`, fold number, set, repeat), not on the fold seed. With several fold seeds pass `random_state = 2023 + fold_seed`.
4. Size-1 sets passed to `compute` in faithful mode go through the set routine. For the original's single-activity behaviour use `compute_single_activities`.
5. `EventLog` does not support logs with a `lifecycle:transition` column (the bpic2012 files).

`impressed_chain.py`
1. `fit()` does not check the size or the case order of `dist_matrix`. When you fit on a subset of cases, build the matrix for exactly those cases, in the same order.
2. The interest values are not rounded. Ranks 9–10 can depend on differences near 1e-18 (2 of 90 selected sets).
3. There is no minimum number of cases. Very rare sets can be selected.
4. A length-1 set can be represented by a self-loop pattern (f2: `{ac419100}` by `ac419100 ~> ac419100`).
5. `select()` before `fit()` gives a bare `AttributeError`.

`existence_importance.py`
1. Nested sets can produce identical feature columns; the second one always gets importance 0.
2. On f2 and f3 the existence-only model is no better than the majority class; do not interpret its ranking.

Other
- Never call the original `calculate_pairwise_case_distance` (2023 code) under scipy 1.18.1; use `case_distance.py`.
- Always call `paretoset` / `paretorank` with `distinct=False` and without NaN.
- `results/experiments/C11/tmp` and `results/experiments/C12/tmp` (216 MB) are throw-away installs; delete them if you do not plan to re-run C11 / C12.
