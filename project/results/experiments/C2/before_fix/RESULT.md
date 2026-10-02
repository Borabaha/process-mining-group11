# Experiment C2 — a fast, verified permutation engine

Date: 2 Oct 2026. Machine: Windows 10, 12 logical cores, **shared with other experiments running at the same time** (timings are therefore noisy; see section 5). Interpreter: `project/.venv/Scripts/python.exe` (Python 3.12, pandas 2.3.3, numpy 2.5.3, scikit-learn 1.9.1, xgboost 3.4.1).

## 1. Answer in five lines

1. **Yes, the permutation loop can be made much faster with identical semantics.** `experiments/engine.py` in `mode='faithful'` returns the original `itemset_permutation_importance` values **bit for bit** (max abs difference 0.0 on 38 compared values: f1 3 itemsets x 3 repeats, f3 3 x 3, f1 10 x 2) and is about **134x faster** (0.10 s instead of 13.6 s per (itemset, repeat), measured in the same session).
2. `mode='fixed'` (the paper's method: no accumulation, both constraints enforced, seeded) is **231x faster** when scored on the training fold and **about 800x faster** when scored on the held-out fold. Target was 10x.
3. The whole original grid (5 folds x the Apriori itemsets x 10 repeats) takes **4-47 s per log and setting** with the engine; the original needs roughly 1-2 hours for the 500 iterations of f1 (extrapolated, not run).
4. All 16 checks in `experiments/test_engine.py` pass (preprocessing, encoder, re-encoding, occurrence finder, shuffle, importance equality, 3000 property-checked permutations, determinism, no accumulation).
5. Side finding the group should look at before fixing the design (section 6): the size of the published importance values is mostly produced by the **accumulation** of permutations in the original code. Without accumulation the values on the training fold are about half (f1, f2) or one eighth (f3), and on the held-out fold they are close to zero.

## 2. What was run

All commands from `C:/Users/Bohabara/Desktop/process mining/project/experiments`, with `PY = ../.venv/Scripts/python.exe`:

| Command | Purpose | Output in `results/experiments/C2/` |
|---|---|---|
| `PY test_engine.py` | 16 verification checks (about 5 min, most of it the slow original) | `test_engine.log`, `test_engine_results.json`, `faithful_equality_f1.csv`, `faithful_equality_f3.csv` |
| `PY engine_benchmark.py --dataset f1 --repeats 2 --timing-runs 3` | speed, original vs engine, f1 fold 0, 10 Apriori itemsets x 2 repeats | `engine_benchmark_f1.log`, `speed_f1.json`, `speed_importances_f1.csv` |
| `PY engine_full_grid.py --dataset f1` (also `f2`, `f3`) | full grid with the engine only: 5 folds x itemsets x 10 repeats, three settings | `full_grid_<d>.log`, `full_grid_<d>.json`, `full_grid_<d>.csv` |
| `PY engine.py --dataset f1 --mode fixed --score-on test` | small demo (CLI block of the engine) | prints a table |

Files written (prototype code for the students to read, verify and adapt):

| File | Content |
|---|---|
| `experiments/engine.py` | `EventLog`, `IndexEncoder`, `make_folds`, `find_occurrences`, `shuffle_sequence_faithful`, `permute_trace_fixed`, `LocationPermutationImportance`, demo CLI. Does not import the original code. |
| `experiments/engine_reference.py` | loads the ORIGINAL `tools.py` read-only under the module name `tools_2024` (no clash with the 2023 `tools.py`) and adds the int-cast subclass from `scripts/scout/smoke_2024c.py`. Used only by the test and benchmark scripts. |
| `experiments/test_engine.py` | the checks (plain script, no pytest needed; `--quick` skips the slow original runs) |
| `experiments/engine_benchmark.py` | speed measurement |
| `experiments/engine_full_grid.py` | full-grid timing and importance summary |

Settings common to everything: `DataManager(path, 2, None, L_max_perc=0.8)` semantics (frq_threshold 2), folds from `make_folds(labels, k=5, seed=0)` (seeded `StratifiedKFold`), `XGBClassifier()` defaults, weighted F1, permutation seed 2023, itemsets = the original's `frequent_activity_sets(0.5, 10)` (10 sets on f1 and f2, 5 sets on f3).

## 3. Equality with the original code

### 3.1 Preprocessing and encoding (f1, f2, f3)

| Check | f1 | f2 | f3 |
|---|---|---|---|
| cases / activities after the rare-activity filter (asserted on load) | 1130 / 164 | 1130 / 207 | 1111 / 156 |
| events / longest trace / `L_max` | 23,853 / 36 / 24.0 | 30,929 / 40 / 31.0 | 20,227 / 31 / 22.0 |
| traces, labels, activity list, `Allowed_locations`, `L_max` equal to `DataManager` | yes | yes | yes |
| encoded matrix shape | 1130 x 5939 | 1130 x 8319 | 1111 x 4866 |
| same columns, same column order, same row order, same values as `index_encoding` + int cast | yes | yes | yes |
| columns that are ever 1 / never 1 | 2450 / 3489 | 3289 / 5030 | 2049 / 2817 |

Details that matter for the re-implementation:

- **Column order of the original depends on the Python hash seed.** `index_encoding` appends the never-observed (position, activity) columns by iterating a Python `set` (tools.py:359-362), so their order changes from one Python process to the next. `IndexEncoder(missing_order='hash')` reproduces the original order column for column inside the same process (this is what the equality checks use). The default `missing_order='sorted'` has the same columns and values, the same order for the observed block, and a deterministic order for the never-observed block.
- This does not change the model: three XGBoost models fitted on fold 0 of f1 (original int64 frame; engine matrix in 'hash' order; engine matrix in 'sorted' order) give **identical predicted probabilities for all 1130 cases** (check `same_model_f1`).
- Re-encoding only the changed traces with `IndexEncoder.write_rows` (three vectorised array writes, no pandas) equals the original `index_encoding` of a log in which 300 traces were shuffled (check `reencoding_f1`).
- **A quirk of the original that faithful mode has to copy:** the original re-encodes only the shuffled cases (tools.py:529-531). If the longest shuffled trace is shorter than the longest trace of the log, the padding columns `e{j}_0` beyond that length are set to 0 instead of 1. `write_rows(..., pad_until=...)` reproduces this (verified on a 126-trace subset with longest trace 20 vs 36 in the log). Fixed mode always pads correctly.
- Encoding the whole f1 log: original 8.0 s in this session on the shared machine (the scout measured about 3 s on a quieter machine; it also emits thousands of pandas PerformanceWarnings), engine 0.007 s (+ 0.011 s to build the encoder).

### 3.2 Occurrences and shuffle

- `find_occurrences` (the j-th match is the j-th occurrence of every itemset activity) equals the original brute-force `find_itemset_indexes` on 3000 random synthetic traces and 3000 real (f1 trace, Apriori itemset) pairs.
- `shuffle_sequence_faithful` equals the original `shuffle_sequence` trace by trace with the same NumPy seed: 6381 (trace, itemset) pairs on f1 (10 itemsets), 2829 on f3 (5 itemsets), all equal.

### 3.3 Importance values, faithful mode vs original (the proof asked for)

Same fitted model (XGBoost fitted on the original's training frame), same fold (fold 0 of the seeded split, same case list passed to both), `np.random.seed(2023)` in the original and `RandomState(2023)` in the engine, 3 repeats. Baseline = weighted F1 on the training fold.

**f1** (904 training cases, baseline 0.9966821542445573):

| itemset | repeat | original importance | engine importance (faithful) | abs diff | traces with itemset |
|---|---|---|---|---|---|
| 370407, ac370000 | 0 | 0.041057613659320591 | 0.041057613659320591 | 0 | 538 |
| 370407, ac370000 | 1 | 0.053496553704118766 | 0.053496553704118766 | 0 | 538 |
| 370407, ac370000 | 2 | 0.071458023742885679 | 0.071458023742885679 | 0 | 538 |
| ac370000, ac370419, ac370443 | 0 | 0.07373908411310659 | 0.07373908411310659 | 0 | 516 |
| ac370000, ac370419, ac370443 | 1 | 0.07498401676771016 | 0.07498401676771016 | 0 | 516 |
| ac370000, ac370419, ac370443 | 2 | 0.077115275978364317 | 0.077115275978364317 | 0 | 516 |
| ac370419, ac370443 | 0 | 0.073594948890896572 | 0.073594948890896572 | 0 | 516 |
| ac370419, ac370443 | 1 | 0.074733428634295551 | 0.074733428634295551 | 0 | 516 |
| ac370419, ac370443 | 2 | 0.073548694689846372 | 0.073548694689846372 | 0 | 516 |

**f3** (888 training cases, baseline 0.9966130042772379):

| itemset | repeat | original importance | engine importance (faithful) | abs diff | traces with itemset |
|---|---|---|---|---|---|
| 370407, ac370000 | 0 | 0.043124679577934422 | 0.043124679577934422 | 0 | 468 |
| 370407, ac370000 | 1 | 0.11072216127145074 | 0.11072216127145074 | 0 | 468 |
| 370407, ac370000 | 2 | 0.17913406711705537 | 0.17913406711705537 | 0 | 468 |
| 370407, ac370000, ac370419 | 0 | 0.22110537909057959 | 0.22110537909057959 | 0 | 447 |
| 370407, ac370000, ac370419 | 1 | 0.24703524734811333 | 0.24703524734811333 | 0 | 447 |
| 370407, ac370000, ac370419 | 2 | 0.23394874345334438 | 0.23394874345334438 | 0 | 447 |
| 370407, ac370419 | 0 | 0.23884290517966034 | 0.23884290517966034 | 0 | 447 |
| 370407, ac370419 | 1 | 0.24056008372890447 | 0.24056008372890447 | 0 | 447 |
| 370407, ac370419 | 2 | 0.22996339535801891 | 0.22996339535801891 | 0 | 447 |

Max absolute difference: **0.0** in both tables (requirement: < 1e-12). The speed benchmark compared a further 20 values (f1, all 10 Apriori itemsets x 2 repeats): max absolute difference **0.0** (`speed_f1.json`, `speed_importances_f1.csv`). No residual to explain.

Why exact equality is possible: the predictions are identical row by row, and weighted F1 is computed from integer counts.

## 4. Fixed mode: what it does and the property tests

`permute_trace_fixed` (paper section 3.2), per trace that contains the itemset:

1. occurrences = greedy non-overlapping complete occurrences (same rule as the original; leftover activities are not moved);
2. for every occurrence, each activity gets a new position drawn from the positions at which that activity was observed (`allowed_from='log'`: whole log, as the original; `'train'`: training fold only), not beyond the trace length and not already taken by another moved event of the same trace;
3. the activities of one occurrence are drawn one after the other, each uniformly from the positions that still leave room for the activities after it, so their relative order is always preserved and no redraw loop is needed (this is the original's window logic, tools.py:459-467, without its bookkeeping bug);
4. all moved events are placed at their drawn positions in the final trace; every other event keeps its relative order and fills the remaining positions;
5. if some occurrence has no feasible assignment the trace is left unchanged and counted (`n_traces_unchanged_infeasible`);
6. every repeat starts from the original traces; the working traces are never stored.

Seeding: each (fold, itemset, repeat) gets its own generator, `default_rng([random_state, fold, crc32(itemset label), repeat])`, so a result does not depend on which other itemsets are computed, in which order, or how many repeats are requested.

Output table columns: `itemset_id, itemset, fold, repeat, baseline, permuted, importance, n_traces_with_itemset, n_traces_changed, n_traces_unchanged_infeasible`.

Property tests on f1 (`check_fixed_properties`; random traces, itemsets drawn half from the 10 Apriori sets and half as random 1-3 activity subsets of the trace):

| Property | pools from whole log, all cases | pools from training fold, held-out cases |
|---|---|---|
| feasible permutations checked | 2000 (size 1: 483, size 2: 966, size 3: 551) | 1000 (size 1: 287, size 2: 457, size 3: 256) |
| of which traces with more than one occurrence | 280 | 152 |
| moved events checked | 4655 | 2310 |
| relative order inside every occurrence preserved | 100 % | 100 % |
| every moved activity sits on an allowed position, inside the trace | 100 % | 100 % |
| multiset of activities unchanged | 100 % | 100 % |
| trace length unchanged | 100 % | 100 % |
| all other events keep their relative order | 100 % | 100 % |
| input trace not mutated | 100 % | 100 % |
| permutation identical to the original trace (allowed; counted as "not changed") | 156 | 80 |
| infeasible (trace left unchanged) | 0 | 4 (all 4 confirmed infeasible by brute force) |

Other checks:

- **Determinism:** two runs with the same seed give identical tables (fixed mode 9 rows, faithful mode 9 rows, exact comparison); a different seed gives different draws. `engine_full_grid.py --dataset f3` run twice in separate Python processes gave identical summaries.
- **No accumulation:** in 1104 permute calls (3 itemsets x 3 repeats on the f1 held-out fold) every input trace was one of the original trace objects, and the log's traces were unchanged afterwards. The last itemset computed alone gives exactly the same rows as when computed after the two others; a 2-repeat run equals the first 2 repeats of a 3-repeat run.
- Contrast, faithful mode (same model, training fold, f1): itemset {ac370419, ac370443} computed alone gives importances 0.0244, 0.0244, 0.0199; computed after two other itemsets it gives 0.0736, 0.0747, 0.0735. The original result of an itemset depends on what was permuted before it.
- Infeasible traces in the full grids (Apriori itemsets, pools from the whole log): 0 on f1, f2 and f3.

## 5. Speed

### 5.1 Benchmark: f1, fold 0, the 10 original Apriori itemsets x 2 repeats (20 iterations), same model for all rows

| Variant | seconds per (itemset, repeat) | speed-up vs original (13.58 s, same session) | speed-up vs the best earlier original timing (6.7 s) |
|---|---|---|---|
| original `itemset_permutation_importance` (int cast), training fold | 13.58 (one run, 271.6 s total) | 1x | - |
| engine faithful, training fold | 0.101 (runs: 0.096, 0.105, 0.101) | 134x | 66x |
| engine fixed, training fold, pools from log | 0.059 (runs: 0.059, 0.063, 0.045) | 231x | 114x |
| engine fixed, held-out fold, pools from log | 0.017 (runs: 0.020, 0.017, 0.017) | 810x | 400x |
| engine fixed, held-out fold, pools from training fold | 0.018 (runs: 0.018, 0.016, 0.018) | 773x | 381x |

Engine values are the median of 3 timing runs; each run includes encoding the scored fold and the baseline prediction. Training fold = 904 cases, held-out fold = 226 cases (which is why held-out scoring is faster).

Honesty note on the original's timing: other experiments were using the CPU while this ran. The original took 13.6 s per iteration here, 13.5 s (f1) and 9.9 s (f3) in the final test run, and between 9.7 and 17.1 s in two earlier test runs of this session; the scout measured 6.7-7.2 s on a quieter machine. The last column therefore uses 6.7 s as the most favourable number for the original. Even then fixed mode is more than 100x faster.

One-time costs (f1): original `DataManager()` 0.58 s and `index_encoding` of the whole log 7.98 s; engine `EventLog()` 0.25 s, `IndexEncoder()` 0.011 s, `transform` of the whole log 0.007 s. One `XGBClassifier().fit` on the original frame took 5.3 s in the benchmark; fitting on the engine's int8 matrix took about 1 s per fold in the full grids (5.3 s for 5 fits on f1).

Where the time went in the original (read from the code, not profiled): one pandas filter + sort per case and repeat (tools.py:513), `itertools.combinations` over all C(len, k) index tuples per trace (tools.py:481), one `.loc` write per shuffled case (tools.py:526), a `pivot_table` + per-column `get_dummies` re-encoding (tools.py:529) and a full-frame copy (tools.py:535). The engine keeps traces as Python lists, finds occurrences directly, writes the changed rows into a NumPy matrix and asks the model only for the changed rows.

### 5.2 Full grid with the engine (5 seeded folds x itemsets x 10 repeats)

| Log | iterations per setting | faithful, train | fixed, train | fixed, held-out | 5 model fits |
|---|---|---|---|---|---|
| f1 (10 itemsets) | 500 | 38.8 s (0.078 s/it) | 20.8 s (0.042 s/it) | 8.3 s (0.017 s/it) | 5.3 s |
| f2 (10 itemsets) | 500 | 47.3 s (0.095 s/it) | 26.4 s (0.053 s/it) | 9.4 s (0.019 s/it) | 5.5 s |
| f3 (5 itemsets) | 250 | 16.9 s (0.068 s/it) | 9.4 s (0.038 s/it) | 4.0 s (0.016 s/it) | 4.4 s |

For comparison, 500 iterations of the original at 6.7-13.6 s are 56-113 minutes per log (extrapolation, the original full grid was not run).

Not measured: the single-activity grid (all 164 / 207 / 156 activities) and itemsets longer than 3.

## 6. What the numbers look like (side finding, feeds Table C item C8 and questions B2/B3)

Mean importance over 5 folds x 10 repeats (50 values per itemset), same folds, same models, same itemsets; standard deviation in brackets.

**f1** (mean held-out F1 of the models 0.909, mean training F1 0.994)

| itemset | faithful, train (= original behaviour) | fixed, train | fixed, held-out |
|---|---|---|---|
| 370407, ac370000 | 0.0650 (0.0147) | 0.0395 (0.0043) | 0.0064 (0.0112) |
| ac370000, ac370419 | 0.0687 (0.0104) | 0.0365 (0.0054) | 0.0023 (0.0131) |
| ac370000, ac370443 | 0.0626 (0.0095) | 0.0340 (0.0056) | 0.0045 (0.0115) |
| ac370000, ac370419, ac370443 | 0.0614 (0.0090) | 0.0406 (0.0066) | 0.0067 (0.0134) |
| ac370419, ac370443 | 0.0615 (0.0091) | 0.0226 (0.0034) | -0.0019 (0.0085) |
| ac370000, ac370442 | 0.0601 (0.0079) | 0.0351 (0.0055) | 0.0088 (0.0106) |
| 370407, ac370000, ac370419 | 0.0631 (0.0092) | 0.0426 (0.0055) | 0.0111 (0.0149) |
| 370407, ac370419 | 0.0623 (0.0100) | 0.0279 (0.0040) | -0.0001 (0.0097) |
| ac370442, ac370443 | 0.0615 (0.0082) | 0.0219 (0.0040) | 0.0016 (0.0074) |
| ac370000, ac370419, ac370442 | 0.0617 (0.0070) | 0.0407 (0.0061) | 0.0086 (0.0123) |

**f2** (mean held-out F1 0.890, mean training F1 0.995)

| itemset | faithful, train | fixed, train | fixed, held-out |
|---|---|---|---|
| ac370000, ac379999 | 0.1278 (0.0187) | 0.0894 (0.0081) | 0.0156 (0.0219) |
| ac370000, ac419100 | 0.1378 (0.0089) | 0.0797 (0.0074) | 0.0095 (0.0219) |
| ac370000, ac379999, ac419100 | 0.1456 (0.0103) | 0.0835 (0.0071) | 0.0148 (0.0192) |
| ac379999, ac419100 | 0.1347 (0.0087) | 0.0628 (0.0059) | 0.0132 (0.0135) |
| 370407, ac370000 | 0.1405 (0.0109) | 0.0916 (0.0120) | 0.0337 (0.0264) |
| 370407, ac370000, ac379999 | 0.1487 (0.0111) | 0.0932 (0.0114) | 0.0325 (0.0219) |
| 370407, ac379999 | 0.1450 (0.0085) | 0.0527 (0.0075) | 0.0057 (0.0129) |
| ac370000, ac370419 | 0.1627 (0.0145) | 0.0930 (0.0116) | 0.0368 (0.0210) |
| ac370000, ac370443 | 0.1751 (0.0134) | 0.0852 (0.0118) | 0.0280 (0.0220) |
| ac370419, ac379999 | 0.1725 (0.0123) | 0.0548 (0.0077) | 0.0094 (0.0161) |

**f3** (mean held-out F1 0.931, mean training F1 0.995; only 5 itemsets reach support 0.5)

| itemset | faithful, train | fixed, train | fixed, held-out |
|---|---|---|---|
| 370407, ac370000 | 0.2129 (0.0696) | 0.0389 (0.0061) | 0.0119 (0.0130) |
| ac370000, ac370419 | 0.2635 (0.0096) | 0.0379 (0.0073) | 0.0125 (0.0133) |
| 370407, ac370000, ac370419 | 0.2485 (0.0128) | 0.0384 (0.0095) | 0.0130 (0.0154) |
| 370407, ac370419 | 0.2539 (0.0106) | 0.0184 (0.0039) | 0.0014 (0.0082) |
| ac370000, ac370443 | 0.2578 (0.0118) | 0.0304 (0.0042) | 0.0107 (0.0126) |

| Log | mean importance faithful, train | fixed, train | fixed, held-out | share of held-out values > 0 |
|---|---|---|---|---|
| f1 | 0.0628 | 0.0341 | 0.0048 | 57.8 % |
| f2 | 0.1490 | 0.0786 | 0.0199 | 78.8 % |
| f3 | 0.2473 | 0.0328 | 0.0099 | 75.2 % |

Accumulation made visible: the first itemset of every fold is the only one that starts from unpermuted traces. Its faithful importance, averaged over the 5 folds, by repeat number:

| Log (first itemset) | repeat 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | fixed, train (mean of all repeats) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| f1 {370407, ac370000} | 0.0416 | 0.0540 | 0.0648 | 0.0701 | 0.0695 | 0.0671 | 0.0702 | 0.0715 | 0.0714 | 0.0697 | 0.0395 |
| f2 {ac370000, ac379999} | 0.0881 | 0.1066 | 0.1236 | 0.1327 | 0.1361 | 0.1337 | 0.1363 | 0.1357 | 0.1408 | 0.1440 | 0.0894 |
| f3 {370407, ac370000} | 0.0439 | 0.1320 | 0.1992 | 0.2309 | 0.2409 | 0.2491 | 0.2591 | 0.2545 | 0.2561 | 0.2630 | 0.0389 |

Reading:

- Repeat 0 of the first itemset (no accumulation yet) agrees with fixed mode on the training fold (0.042 vs 0.040, 0.088 vs 0.089, 0.044 vs 0.039). From repeat 1 on, the original permutes already-permuted traces and the "importance" climbs to a plateau that later itemsets inherit. In the original output the itemsets therefore all end up at a similar level (f1 0.060-0.069, f3 0.21-0.26), i.e. they are hard to tell apart.
- The faithful levels match what the guide read off the paper's Fig. 5 by eye (f1 about 0.06-0.10, f3 about 0.20-0.29; Guide section 3.5), so faithful mode with seeded folds appears to reproduce the published magnitudes. This is a comparison with by-eye values, not a numeric reproduction of the figure.
- On the held-out fold the importance of these frequent itemsets is small (means 0.005 / 0.020 / 0.010) and its spread over folds and repeats (0.007-0.026) is as large as or larger than the mean. Rankings of itemsets in fixed/held-out mode will be noisy with 10 repeats.
- These tables are a by-product of the timing run with one fold seed. They are not the full C8 experiment (no Spearman, no box plots, no second fold seed).

## 7. What this means for the project decisions

- **B10 / C2 (re-implementation, runtime):** settled. A clean re-implementation is both exact (faithful mode) and fast. Runtime is no longer a constraint: the whole planned grid (3 logs x 2 strategies x 3 lengths x 10 sets x 5 folds x 10 repeats = 9000 iterations per setting) is in the order of minutes (extrapolated from 0.02-0.10 s per iteration; not run). More repeats or several fold seeds are affordable.
- **Reproduction part of the paper:** use `mode='faithful', score_on='train'`; it is the original, 134x faster, and it can be shown to be bit-identical with `test_engine.py`.
- **Variant / comparison part:** `mode='fixed'` is ready with both `score_on` options and both `allowed_from` options. Because held-out importances are near the noise level, the group should decide (and probably ask the supervisor, B2/B3) whether to report fixed/train, fixed/held-out or both, and should plan for more than 10 repeats or several fold seeds in fixed/held-out mode.
- **Limitations section of the paper:** section 6 gives concrete numbers for discrepancy rows 1 and 2 of the guide's paper-vs-code table (training-fold scoring, accumulation).
- **Encoder:** use `IndexEncoder` with the default `missing_order='sorted'`; it gives the same model as the original encoding and is reproducible across runs.

## 8. Design choices in fixed mode that the paper does not pin down (to confirm)

1. **Simultaneous placement.** All moved events of a trace are placed at their drawn positions in the final trace. The original moves them one by one, so later moves shift earlier ones. The paper's worked example (trace 1, both occurrences end on observed positions in the final trace) fits the simultaneous reading.
2. **Sequential windowed draw**, as in the original code. It is not uniform over all feasible position tuples (the first activity is drawn first, the later ones are squeezed behind it). A uniform draw over all order-preserving tuples is an alternative; not implemented.
3. **Per-activity position pools** (`Allowed_locations`), as in the code and in the paper's feasibility sentence. The paper's example also shows joint tuples OL(L, (ER, CR)); joint pools are not implemented.
4. **Positions are clipped to the trace length** and a position already taken by another moved event of the same trace is excluded. The original does neither.
5. **All-or-nothing per trace:** if any occurrence cannot be placed, the whole trace stays unchanged and is counted. This never happened for the Apriori itemsets (0 traces in all full grids); it happened 4 times in 1004 random trials with training-fold pools.
6. A trace whose length equals the itemset size is skipped (same rule as the original, tools.py:522).
7. If no scored trace contains the itemset, fixed mode returns importance 0 with `n_traces_with_itemset = 0`; faithful mode raises `ValueError` because the original crashes there too. The "fewer than 3 cases" guard mentioned in the guide is not implemented.

## 9. Limits and things that were not done

- Faithful mode covers `itemset_permutation_importance` only. The original's single-activity routine `trace_permutation_importance` (tools.py:366-448, used for Fig. 3) is a different algorithm and is **not** ported. Passing size-1 itemsets to faithful mode runs the itemset routine on them, which is not what the original single-activity mode does. Fixed mode handles size 1 properly (property-tested).
- Exact column-order equality with the original is only checkable inside one Python process (hash-seed dependence, section 3.1).
- Equality was proven on fold 0 of f1 and f3 with the listed itemsets (plus 20 values on f1 in the benchmark); f2 importance equality was not run against the original (its preprocessing and encoding equality were). The shuffle itself was compared on all f1 and f3 traces for all Apriori itemsets.
- `EventLog` supports logs without a `lifecycle:transition` column (all BPIC11 files). It raises `NotImplementedError` for the bpic2012 files.
- Timings were taken on a shared machine; the original's per-iteration time varied between 9.7 and 17.1 s within this session.
- No profiling (`cProfile`) of the original was done; the explanation of where its time goes comes from reading the code.
- Nothing failed in the final run. During development one check (`reencoding_f1`) failed once because the test compared rows in a different order than the original's sorted index; the test was fixed, the engine was not changed.
