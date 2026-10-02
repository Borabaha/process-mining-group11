# Experiment C2 — a fast, verified permutation engine (revision 2)

Date: 2 Oct 2026. Machine: Windows 10, 12 logical cores, shared with other experiments (timings are noisy; section 5 says which numbers were taken on a quiet and which on a loaded machine). Interpreter: `project/.venv/Scripts/python.exe` (Python 3.12, pandas 2.3.3, numpy 2.5.3, scikit-learn 1.9.1, xgboost 3.4.1).

Revision 2 repairs the 11 issues an independent verifier found in revision 1 (section 0). The first revision's report and outputs are kept in `results/experiments/C2/before_fix/`.

## 0. What changed in revision 2

| # | Severity | Issue found by the verifier | What was done | Evidence |
|---|---|---|---|---|
| 1 | high | Fixed mode crashed whenever an (itemset, repeat) changed no trace (`IndexEncoder.transform([])`), and one such itemset lost the results of the whole call. Revision 1 of this report wrongly said such a case "returns importance 0". | `IndexEncoder.write_rows` returns at once for an empty input; `transform([])` gives a (0, n_features) matrix. The statement in the report is now true. | Checks `edge_cases_f1`, `single_activity_grid_f1/f3`. All 164 (f1) and 156 (f3) single activities run on the held-out fold. The verifier's own `verify_engine.py --part edge` now reports 0 of 164 crashing (was 31). |
| 2 | medium | `compute()` did not validate itemsets: an unknown or un-normalised activity name, or a bare string instead of a list, would silently give importance 0. | New `validate_itemsets`: `TypeError` for a string, `ValueError` for an empty set or unknown names (with a "did you mean" hint). New helper `normalise_activity`. A `UserWarning` lists itemsets that occur in no scored trace. | Check `input_validation_f1` (messages in section 4.3). |
| 3 | medium | Faithful mode could not reproduce the original **single-activity** results (Fig. 3): `trace_permutation_importance` was not ported. | New method `compute_single_activities`; in faithful mode it is a port of `trace_permutation_importance` (tools.py:366-448) with its guards and its index bug. | Section 3.4: 328 values on the whole training fold of f1 and 312 values on f3, max abs diff 0.0. |
| 4 | medium | Held-out importances in fixed mode are at noise level; rankings from 10 repeats are unreliable. A decision is needed. | Not a code bug. New script `engine_stability.py` measures it with 10 fold seeds x 5 folds x 30 repeats. | Section 6.2 (numbers) and section 7 (what it means for the decision). |
| 5 | low | A model fitted on the original DataFrame used with the default encoder gives wrong predictions without an error (column order of the never-observed block). | `check_model_matches_encoder` compares the model's feature count and feature names with the encoder and raises on a mismatch. Docstrings say the model must be fitted on this encoder's matrix. | Check `input_validation_f1`. |
| 6 | low | `fold` defaulted to 0; forgetting it gave the same random streams in every fold. | `fold` is a required keyword argument of `compute` and `compute_single_activities`. | Check `input_validation_f1`. |
| 7 | low | `allowed_from='log'` uses positions seen only in held-out cases; should be a stated choice. | Stated in the class docstring. Measured how much it matters (`engine_fixed_options.py`). | Section 6.3: no measurable effect on the importance. |
| 8 | low | Fixed mode constrains only the moved events; other events are shifted and can land on never-observed positions. | Stated in the docstring of `permute_trace_fixed`. Fixed mode now counts it per (itemset, repeat): columns `n_other_events_shifted`, `n_other_events_unobserved`. | Section 4.4: 3.5 % / 4.3 % / 3.7 % of the shifted events (f1 / f2 / f3). |
| 9 | low | `write_rows` failed with confusing messages for a generator, a too long trace or an unknown activity. | `traces = list(traces)` first; explicit `ValueError`s. | Check `edge_cases_f1`. |
| 10 | low | `EventLog` checked the log with `assert` against a table keyed by file name. | Removed from `engine.py`; the expected (cases, activities) are asserted in `test_engine.py` (`check_event_log`). | Checks `event_log_f1/f2/f3`. |
| 11 | low | For sets of 2 and 3 the sequential draw is not uniform over all order-preserving position tuples. | Kept as the default (it is the original's procedure) and stated. New option `draw='uniform'` samples every feasible tuple with the same probability. Measured how much it matters. | Section 4.2 (chi-square) and section 6.3: the draw changes the level of the importance by 9-14 % and, on f2, the ranking. |

Nothing that revision 1 reported as a number changed: all 3750 rows of the revision-1 grids (f1, f2, f3; faithful and fixed) are reproduced exactly by the repaired engine (max abs diff 0.0 on importance and baseline, section 6.1).

## 1. Answer in short

1. **The permutation loop can be made much faster with identical semantics.** `mode='faithful'` returns the original values bit for bit, for itemsets (`itemset_permutation_importance`: 18 values in the equality tables, 20 more in the benchmark, 17 on f2 by the verifier) and now also for single activities (`trace_permutation_importance`: 640 values), max abs difference 0.0 everywhere.
2. Speed on a quiet machine, f1, per (itemset, repeat): original 6.78 s; faithful 0.053 s (128x); fixed on the training fold 0.036 s (189x); fixed on the held-out fold 0.012 s (578x). Target was 10x.
3. The whole original grid (5 folds x Apriori itemsets x 10 repeats) takes 3-34 s per log and setting.
4. All 23 checks of `test_engine.py` pass, plus the whole-fold single-activity proof on f1. The verifier's script was re-run on the repaired engine: no crash left, 51,027 permuted traces without a constraint violation.
5. Facts for the open design decisions (sections 6.2, 6.3, 7):
   - The **original's ranking of itemsets is not trustworthy**: on f1 two different 5-fold splits agree with Spearman 0.52 only, and on f2 and f3 it is unrelated to the ranking obtained without accumulation (Spearman 0.07 and -0.10).
   - **Fixed mode on the training fold** gives a stable ranking (Spearman between two splits 0.94-0.97).
   - **Fixed mode on the held-out fold** needs several fold seeds: with one split and 10 repeats two splits agree with Spearman 0.62 / 0.90 / 0.75 (f1 / f2 / f3). With 10 fold seeds the 95 % interval of an itemset mean is about ±0.003 and two halves of the seeds agree with 0.90-0.98. More repeats help little; more fold seeds help.
   - Training-fold and held-out rankings of fixed mode agree (Spearman 0.87 / 0.94 / 1.00).
   - Position pools from the log or from the training fold: no measurable difference. Sequential or uniform draw: the uniform draw lowers the training-fold importance by 9-14 % and on f2 it also changes the ranking (Spearman 0.52 with the sequential ranking).

## 2. What was run

All commands from `C:/Users/Bohabara/Desktop/process mining/project/experiments`, with `PY = ../.venv/Scripts/python.exe`:

| Command | Purpose | Output in `results/experiments/C2/` |
|---|---|---|
| `PY test_engine.py` | 23 verification checks (38 min on the loaded machine; 26 min of that is the original's single-activity routine) | `test_engine.log`, `test_engine_results.json`, `faithful_equality_f1.csv`, `faithful_equality_f3.csv`, `faithful_single_equality_f3_200cases.csv` |
| `PY test_engine.py --quick` | the 18 checks that do not run the slow original importance (3 min) | `test_engine_quick.log`, `test_engine_results_quick.json` |
| `PY test_engine.py --full-single --only faithful_single_full_f1` | single-activity proof on the whole training fold of f1 (69 min, almost all of it the original) | `test_engine_single_full_f1.log`, `test_engine_results_faithful_single_full_f1.json`, `faithful_single_equality_f1_904cases.csv` |
| `PY engine_benchmark.py --dataset f1 --repeats 2 --timing-runs 3` | speed, original vs engine, f1 fold 0, 10 Apriori itemsets x 2 repeats | `engine_benchmark_f1.log`, `speed_f1.json`, `speed_importances_f1.csv` |
| `PY engine_full_grid.py --dataset f1` (also `f2`, `f3`) | full grid with the engine only: 5 folds x itemsets x 10 repeats, five settings | `full_grid_<d>.log`, `.json`, `.csv` |
| `PY engine_stability.py --dataset f1 --seeds 10 --repeats 30` (also `f2`, `f3`) | rank stability over 10 fold seeds, three settings (`--reuse` recomputes only the summary) | `stability_<d>.csv`, `stability_<d>_itemsets.csv`, `stability_<d>.json`, `stability_<d>.log`, `stability_<d>_summary.log` |
| `PY engine_fixed_options.py --dataset f1 --seeds 10 --repeats 10` (also `f2`, `f3`) | effect of the two open fixed-mode options over 10 fold seeds | `fixed_options_<d>.csv`, `.json`, `.log` |
| `PY engine.py --dataset f1 --mode fixed --score-on test` | small demo (CLI block of the engine) | prints a table |
| `PY ../results/experiments/C2/verify_engine.py --part edge` (also `grid`, `fixed`, `hashseed`, `encoder`, `faithful --dataset f2`) | the verifier's own script, re-run on the repaired engine | `verify_<part>.json`, `.log` |

Files (prototype code for the students to read, verify and adapt):

| File | Content |
|---|---|
| `experiments/engine.py` | `EventLog`, `IndexEncoder`, `make_folds`, `find_occurrences`, `shuffle_sequence_faithful`, `shuffle_activity_faithful`, `permute_trace_fixed`, `count_shifted_events`, `validate_itemsets`, `normalise_activity`, `check_model_matches_encoder`, `LocationPermutationImportance` (`compute`, `compute_single_activities`), demo CLI. Does not import the original code. |
| `experiments/engine_reference.py` | loads the ORIGINAL `tools.py` read-only under the module name `tools_2024` (no clash with the 2023 `tools.py`) and adds the int-cast subclass from `scripts/scout/smoke_2024c.py`. Used only by the test and benchmark scripts. |
| `experiments/test_engine.py` | the checks (plain script, no pytest needed; `--quick`, `--only`, `--full-single`) |
| `experiments/engine_benchmark.py` | speed measurement |
| `experiments/engine_full_grid.py` | full-grid timing and importance summary |
| `experiments/engine_stability.py` | new: rank stability over fold seeds |
| `experiments/engine_fixed_options.py` | new: effect of `allowed_from` and `draw` |
| `results/experiments/C2/verify_engine.py` | the verifier's independent script (not changed) |
| `results/experiments/C2/before_fix/` | revision-1 report and outputs, and the verifier's outputs on the unrepaired engine |

Settings common to everything: `DataManager(path, 2, None, L_max_perc=0.8)` semantics (frq_threshold 2), folds from `make_folds(labels, k=5, seed=...)` (seeded `StratifiedKFold`; seed 0 unless a section says otherwise), `XGBClassifier()` defaults, weighted F1, permutation seed 2023, itemsets = the original's `frequent_activity_sets(0.5, 10)` (10 sets on f1 and f2, 5 sets on f3; all of size 2 or 3).

## 3. Equality with the original code

### 3.1 Preprocessing and encoding (f1, f2, f3)

| Check | f1 | f2 | f3 |
|---|---|---|---|
| cases / activities after the rare-activity filter (asserted in `test_engine.py`) | 1130 / 164 | 1130 / 207 | 1111 / 156 |
| events / longest trace / `L_max` | 23,853 / 36 / 24.0 | 30,929 / 40 / 31.0 | 20,227 / 31 / 22.0 |
| traces, labels, activity list, `Allowed_locations`, `L_max` equal to `DataManager` | yes | yes | yes |
| encoded matrix shape | 1130 x 5939 | 1130 x 8319 | 1111 x 4866 |
| same columns, same column order, same row order, same values as `index_encoding` + int cast | yes | yes | yes |
| columns that are ever 1 / never 1 | 2450 / 3489 | 3289 / 5030 | 2049 / 2817 |

Details that matter for the re-implementation:

- **Column order of the original depends on the Python hash seed.** `index_encoding` appends the never-observed (position, activity) columns by iterating a Python `set` (tools.py:359-362), so their order changes from one Python process to the next. `IndexEncoder(missing_order='hash')` reproduces the original order column for column inside the same process (this is what the equality checks use). The default `missing_order='sorted'` has the same columns and values, the same order for the observed block, and a deterministic order for the never-observed block.
- This does not change the model: three XGBoost models fitted on fold 0 of f1 (original int64 frame; engine matrix in 'hash' order; engine matrix in 'sorted' order) give **identical predicted probabilities for all 1130 cases** (check `same_model_f1`).
- **It does matter for predicting.** The engine predicts on a bare NumPy matrix. A model fitted on the original DataFrame must therefore be used with the 'hash' encoder of the same process, never with the default one. Since revision 2 the engine refuses the wrong pairing when the model knows its feature names (`check_model_matches_encoder`). A model fitted on a bare matrix carries no names; there the rule "fit the model on `encoder.transform(...)` of the encoder you pass to `compute`" cannot be checked by the engine.
- Re-encoding only the changed traces with `IndexEncoder.write_rows` (three vectorised array writes, no pandas) equals the original `index_encoding` of a log in which 300 traces were shuffled (check `reencoding_f1`).
- **A quirk of the original that faithful mode has to copy:** the original re-encodes only the shuffled cases (tools.py:529-531). If the longest shuffled trace is shorter than the longest trace of the log, the padding columns `e{j}_0` beyond that length are set to 0 instead of 1. `write_rows(..., pad_until=...)` reproduces this (verified on a 126-trace subset with longest trace 20 vs 36 in the log). Fixed mode always pads correctly.
- Encoding the whole f1 log: original 3.8 s (quiet machine; it also emits thousands of pandas PerformanceWarnings), engine 0.004 s (+ 0.005 s to build the encoder).

### 3.2 Occurrences and shuffle

- `find_occurrences` (the j-th match is the j-th occurrence of every itemset activity) equals the original brute-force `find_itemset_indexes` on 3000 random synthetic traces and 3000 real (f1 trace, Apriori itemset) pairs.
- `shuffle_sequence_faithful` equals the original `shuffle_sequence` trace by trace with the same NumPy seed: 6381 (trace, itemset) pairs on f1 (10 itemsets), 2829 on f3 (5 itemsets), all equal.

### 3.3 Itemset importance, faithful mode vs original

Same fitted model (XGBoost fitted on the original's training frame), same fold (fold 0 of the seeded split, same case list passed to both), `np.random.seed(2023)` in the original and `RandomState(2023)` in the engine, 3 repeats. Baseline = weighted F1 on the training fold. Re-run on the repaired engine; the numbers are the same as in revision 1.

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

Max absolute difference: **0.0** in both tables (requirement: < 1e-12). Further comparisons with the original, all with max abs diff 0.0: 20 values in the speed benchmark (f1, all 10 Apriori itemsets x 2 repeats; `speed_f1.json`); by the verifier 17 values on **f2** (fold 2 of fold seed 7, re-run on the repaired engine, `verify_faithful_f2.json`) and on f1 with another fold and other itemsets (`verify_faithful_f1.json`, run on revision 1).

Why exact equality is possible: the predictions are identical row by row, and weighted F1 is computed from integer counts.

### 3.4 Single-activity importance, faithful mode vs original (new)

`LocationPermutationImportance(mode='faithful').compute_single_activities(...)` is a port of `trace_permutation_importance(constrain=True)`, the routine the original runs when `Multi_activity` is False. It is a different algorithm from the itemset routine:

- every occurrence of the activity is moved, one after the other, to a position drawn from the activity's allowed positions that fit in the trace (`i < len(trace) + 1`);
- the occurrence indexes are collected before the first move and never updated, so from the second occurrence on the event that is moved can be a different activity (an index bug of the original, ported as it is);
- an activity with fewer than 2 allowed positions gets importance 0 and is not permuted (output column `guard = 'one_allowed_position'`);
- when fewer than 3 scored traces contain the activity, the traces are permuted but the importance is set to 0 (`guard = 'fewer_than_3_cases'`); the permutation stays in the working copy;
- one working copy for all activities and repeats: permutations accumulate over repeats **and over activities**, so every value depends on all activities processed before it. The original always processes all activities of the log in order of first appearance; that is the default of the method.

Proof (same fitted model, fold 0, seed 2023, 2 repeats, all activities of the log):

| Log | scored cases | values compared | of which non-zero in the original | forced zeros (one position / fewer than 3 cases) | largest original value | max abs diff |
|---|---|---|---|---|---|---|
| f1 | whole training fold, 904 | 328 (164 activities x 2) | 243 | 8 / 34 | 0.0732 | **0.0** |
| f3 | first 200 cases of the training fold | 312 (156 activities x 2) | 110 | 6 / 172 | 0.2492 | **0.0** |

The f3 check uses 200 cases because the original needs 2.5-4.8 s per (activity, repeat) even then; the whole-fold run on f1 took the original 4100 s (12.5 s per (activity, repeat) on the loaded machine) and the engine 2.5 s. Largest single-activity values on f1 (whole training fold): ac370000 0.073 and 0.069, ac379999 0.060, 370407 0.057, ac370419 0.056.

Not done: the whole training fold of f3 and anything on f2 for the single-activity routine; a numeric comparison with the paper's Fig. 3 (the original script uses unseeded folds, so the figure cannot be matched value by value anyway).

Size-1 sets passed to `compute()` in faithful mode still run through the **itemset** routine (the verifier confirmed that this equals the original itemset routine exactly, diff 0.0). That is not the original's single-activity behaviour; for faithful single-activity numbers use `compute_single_activities`.

## 4. Fixed mode: what it does and the tests

### 4.1 The permutation

`permute_trace_fixed` (paper section 3.2), per trace that contains the itemset:

1. occurrences = greedy non-overlapping complete occurrences (same rule as the original; leftover activities are not moved);
2. for every occurrence, each activity gets a new position drawn from the positions at which that activity was observed (`allowed_from='log'`: whole log, as the original; `'train'`: training fold only), not beyond the trace length and not already taken by another moved event of the same trace;
3. `draw='sequential'` (default): the activities of one occurrence are drawn one after the other, each uniformly from the positions that still leave room for the activities after it, so their relative order is always preserved and no redraw loop is needed (this is the original's window logic, tools.py:459-467, without its bookkeeping bug). `draw='uniform'` (new): every order-preserving tuple of allowed positions of the occurrence is equally likely;
4. all moved events are placed at their drawn positions in the final trace; every other event keeps its relative order and fills the remaining positions;
5. if some occurrence has no feasible assignment the trace is left unchanged and counted (`n_traces_unchanged_infeasible`);
6. every repeat starts from the original traces; the working traces are never stored.

Seeding: each (fold, itemset, repeat) gets its own generator, `default_rng([random_state, fold, crc32(itemset label), repeat])`, so a result does not depend on which other itemsets are computed, in which order, or how many repeats are requested. Because the fold number is part of the seed, `fold` is a required keyword argument.

Output table columns: `itemset_id, itemset, fold, repeat, baseline, permuted, importance, n_traces_with_itemset, n_traces_changed, n_traces_unchanged_infeasible`, and in fixed mode `n_other_events_shifted, n_other_events_unobserved`. `compute_single_activities` in faithful mode adds `guard`.

### 4.2 Property tests

On f1 (`check_fixed_properties`, `check_uniform_draw`; random traces, itemsets drawn half from the 10 Apriori sets and half as random 1-3 activity subsets of the trace):

| Property | sequential draw, pools from whole log, all cases | sequential draw, pools from training fold, held-out cases | uniform draw, pools from whole log, all cases |
|---|---|---|---|
| feasible permutations checked | 2000 (size 1: 483, size 2: 966, size 3: 551) | 1000 (size 1: 287, size 2: 457, size 3: 256) | 2000 (size 1: 512, size 2: 904, size 3: 584) |
| of which traces with more than one occurrence | 280 | 152 | 270 |
| moved events checked | 4655 | 2310 | 4649 |
| relative order inside every occurrence preserved | 100 % | 100 % | 100 % |
| every moved activity sits on an allowed position, inside the trace | 100 % | 100 % | 100 % |
| multiset of activities unchanged, trace length unchanged | 100 % | 100 % | 100 % |
| all other events keep their relative order | 100 % | 100 % | 100 % |
| input trace not mutated | 100 % | 100 % | 100 % |
| permutation identical to the original trace (allowed; counted as "not changed") | 156 | 80 | 142 |
| infeasible (trace left unchanged) | 0 | 4 (all 4 confirmed infeasible by brute force) | 0 |

Distribution of the draw (one f1 trace per size, all feasible position tuples enumerated by brute force, chi-square against the uniform distribution):

| Set size | feasible tuples | draws | chi-square, `draw='uniform'` | chi-square, `draw='sequential'` | degrees of freedom |
|---|---|---|---|---|---|
| 1 | 18 | 20,000 | 13.2 | 13.2 | 17 |
| 2 | 598 | 20,000 | 588.4 | 23,130 | 597 |
| 3 | 6759 | 135,180 | 6718.7 | 1,641,519 | 6758 |

For a single activity both draws are uniform over the pool. For 2 and 3 activities the sequential draw is far from uniform (13 of the 6759 tuples of the size-3 example were never drawn); the uniform draw passes. With several occurrences in one trace both draws place the occurrences one after the other, so neither is uniform over all joint placements.

Other checks:

- **Determinism:** two runs with the same seed give identical tables (fixed and faithful mode, exact comparison); a different seed gives different draws. The verifier's hash-seed test (two fresh interpreters with different `PYTHONHASHSEED`) gives one digest, also on the repaired engine.
- **No accumulation:** in 1104 permute calls (3 itemsets x 3 repeats on the f1 held-out fold) every input trace was one of the original trace objects, and the log's traces were unchanged afterwards. The last itemset computed alone gives exactly the same rows as when computed after the two others; a 2-repeat run equals the first 2 repeats of a 3-repeat run.
- Contrast, faithful mode (same model, training fold, f1): itemset {ac370419, ac370443} computed alone gives importances 0.0244, 0.0244, 0.0199; computed after two other itemsets it gives 0.0736, 0.0747, 0.0735. The original result of an itemset depends on what was permuted before it.
- The verifier's independent constraint checks, re-run on the repaired engine (`verify_fixed.json`): 51,027 permuted traces on f1 and f3, sizes 1-3, 0 violations.
- Infeasible traces in the full grids (Apriori itemsets, pools from the whole log or from the training fold): 0 on f1, f2 and f3.

### 4.3 Edge cases and input checks (new in revision 2)

Check `edge_cases_f1` and `single_activity_grid_f1/f3` (fold 0 of fold seed 0, model fitted on the engine's matrix):

| Situation | Result |
|---|---|
| `encoder.transform([])` | (0, 5939) matrix |
| traces passed as a generator to `transform` / `write_rows` | same matrix as a list |
| itemset that occurs in no trace of the held-out fold | importance 0 in every repeat, `n_traces_with_itemset = 0`, one `UserWarning` per `compute` call that lists such itemsets |
| activity with a single allowed position, scored on the training fold | importance 0, `n_traces_changed = 0` |
| absent itemset in the same call as a normal itemset | the normal itemset's rows are identical to computing it alone |
| all 164 single activities of f1 on the held-out fold, 2 repeats | 328 rows, no crash; 27 activities occur in no held-out trace; 68 rows without a changed trace, all with importance exactly 0; 2.1 s (quiet machine) |
| the same with pools from the training fold | 67 rows without a changed trace; 20 traces infeasible (left unchanged) |
| all 156 single activities of f3 on the held-out fold, 2 repeats | 312 rows; 31 activities occur in no held-out trace; 83 rows without a changed trace (72 with training-fold pools, 8 infeasible traces) |

Errors the engine now raises (check `input_validation_f1`, messages copied from the log):

| Input | Error |
|---|---|
| `itemsets` is a string | `TypeError: itemsets must be a collection of activity collections` |
| one itemset is a string, e.g. `["ac410100"]` instead of `[["ac410100"]]` | `TypeError: itemset 0 is the string 'ac410100'; pass a collection of activity names, e.g. ['ac410100']` |
| unknown activity | `ValueError: not activities of the log: 'no_such_activity'. Activity names are lower case without spaces, '-' and '_' (see normalise_activity).` |
| name not normalised | `ValueError: not activities of the log: 'AC410100' (did you mean 'ac410100'?). ...` |
| empty itemset | `ValueError: itemset 0 is empty` |
| `fold` missing or passed positionally | `TypeError: ... missing 1 required keyword-only argument: 'fold'` |
| faithful mode, itemset in no scored trace | `ValueError: no scored trace can be permuted for [...]; the original crashes here too`, raised before any itemset is computed |
| model fitted on the original frame, default encoder | `ValueError: the model's feature names differ from encoder.feature_names (same columns, different order); ...` |
| model fitted on another number of features | `ValueError: the model was fitted on 50 features, the encoder produces 5939` |
| trace longer than the encoder's longest trace | `ValueError: a trace has 37 events; the encoder was built for traces of at most 36 events` |
| activity unknown to the encoder | `ValueError: activity 'no_such_activity' is unknown to the encoder` |

IMPresseD output must be mapped to the engine's activity names with `normalise_activity` (lower case, spaces, '-' and '_' removed) before it is passed to `compute`.

### 4.4 Limitation: events that are not moved (new in revision 2)

The two constraints of the paper hold for the moved events only. Moving them shifts other events by one or more places, and a shifted event can land on a position where its activity was never observed. The paper acknowledges this. Fixed mode now counts it. Full grid, held-out fold, fold seed 0, pools from the whole log:

| Log | changed traces | other events shifted | of which on a position outside their activity's pool | share |
|---|---|---|---|---|
| f1 | 63,633 | 1,097,604 | 38,070 | 3.5 % |
| f2 | 69,324 | 1,546,505 | 67,232 | 4.3 % |
| f3 | 28,134 | 448,958 | 16,516 | 3.7 % |

The training fold gives the same shares (3.5 % / 4.3 % / 3.7 %). The verifier measured 4.7 % (f1) and 5.2 % (f3) with other itemsets (sizes 1-3) on another fold. So roughly one shifted event in 25 sits on a position that the log never showed for its activity.

## 5. Speed

### 5.1 Benchmark: f1, fold 0, the 10 original Apriori itemsets x 2 repeats (20 iterations), same model for all rows

Re-measured on the repaired engine when no other experiment was running:

| Variant | seconds per (itemset, repeat) | speed-up vs original |
|---|---|---|
| original `itemset_permutation_importance` (int cast), training fold | 6.78 (one run, 135.6 s total) | 1x |
| engine faithful, training fold | 0.053 (runs: 0.053, 0.052, 0.055) | 128x |
| engine fixed, training fold, pools from log | 0.036 (runs: 0.038, 0.036, 0.034) | 189x |
| engine fixed, held-out fold, pools from log | 0.012 (runs: 0.0117, 0.0115, 0.0118) | 578x |
| engine fixed, held-out fold, pools from training fold | 0.012 (runs: 0.0121, 0.0119, 0.0123) | 560x |

Engine values are the median of 3 timing runs; each run includes input validation, encoding the scored fold and the baseline prediction. Training fold = 904 cases, held-out fold = 226 cases (which is why held-out scoring is faster). The faithful run of the benchmark again equals the original on all 20 values (max abs diff 0.0).

Honesty note on timings: the original's time per iteration depends strongly on the machine load. It was 6.78 s in this run and 6.7-7.2 s in the scout's run (quiet machine), 13.6 s in revision 1 and 15.5-22.4 s inside the test run of revision 2 (both with other experiments running). Revision 1 reported 134x / 231x / 810x from a loaded machine; the ratios above are the cleaner ones. The repairs did not slow the engine down measurably.

One-time costs (f1, quiet machine): original `DataManager()` 0.26 s and `index_encoding` of the whole log 3.80 s; engine `EventLog()` 0.12 s, `IndexEncoder()` 0.005 s, `transform` of the whole log 0.004 s. One `XGBClassifier().fit` on the original frame took 2.7 s; fitting on the engine's int8 matrix took about 0.5 s per fold in the full grids (2.7 s for 5 fits on f1).

Single activities: the original's `trace_permutation_importance` took 12.5 s per (activity, repeat) on the whole training fold of f1 (loaded machine), the engine's faithful port 0.0075 s. Fixed mode needs 2.1 s for all 164 activities x 2 repeats on the held-out fold of f1.

Where the time went in the original (read from the code, not profiled): one pandas filter + sort per case and repeat (tools.py:513), `itertools.combinations` over all C(len, k) index tuples per trace (tools.py:481), one `.loc` write per shuffled case (tools.py:526), a `pivot_table` + per-column `get_dummies` re-encoding (tools.py:529) and a full-frame copy (tools.py:535). The engine keeps traces as Python lists, finds occurrences directly, writes the changed rows into a NumPy matrix and asks the model only for the changed rows.

### 5.2 Full grid with the engine (5 seeded folds x itemsets x 10 repeats), quiet machine

| Log | iterations per setting | faithful, train | fixed, train | fixed, held-out | fixed, held-out, training-fold pools | fixed, held-out, uniform draw | 5 model fits |
|---|---|---|---|---|---|---|---|
| f1 (10 itemsets) | 500 | 26.3 s (0.053 s/it) | 16.9 s (0.034) | 6.0 s (0.012) | 5.9 s (0.012) | 12.4 s (0.025) | 2.7 s |
| f2 (10 itemsets) | 500 | 33.7 s (0.068) | 22.4 s (0.045) | 7.9 s (0.016) | 8.2 s (0.016) | 20.0 s (0.040) | 3.4 s |
| f3 (5 itemsets) | 250 | 10.9 s (0.044) | 7.2 s (0.029) | 2.8 s (0.011) | 2.7 s (0.011) | 4.9 s (0.019) | 2.5 s |

The uniform draw costs about twice the time of the sequential draw (it counts the feasible completions for every position). For comparison, 500 iterations of the original at 6.8 s are 57 minutes per log (extrapolation, the original full grid was not run).

The larger studies of section 6 ran with three logs in parallel on the loaded machine: `engine_stability.py` (10 fold seeds, 3 settings, 35,000 rows for f1) took 50 / 53 / 36 minutes for f1 / f2 / f3; `engine_fixed_options.py` took 20 / 27 / 11 minutes.

Not measured: itemsets longer than 3.

## 6. What the numbers look like

### 6.1 One fold seed (seed 0): mean importance over 5 folds x 10 repeats

50 values per itemset, same folds, same models, same itemsets; standard deviation in brackets. The three columns are identical to revision 1: the repaired engine reproduces all 3750 rows of the revision-1 grid files exactly (max abs diff 0.0 on importance and baseline, same `n_traces_changed`; compared with `before_fix/full_grid_<d>.csv`).

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

- Repeat 0 of the first itemset (no accumulation yet) agrees with fixed mode on the training fold (0.042 vs 0.040, 0.088 vs 0.089, 0.044 vs 0.039). From repeat 1 on, the original permutes already-permuted traces and the "importance" climbs to a plateau that later itemsets inherit. In the original output the itemsets therefore all end up at a similar level (f1 0.060-0.069, f3 0.21-0.26).
- The faithful levels match what the guide read off the paper's Fig. 5 by eye (f1 about 0.06-0.10, f3 about 0.20-0.29; Guide section 3.5), so faithful mode with seeded folds appears to reproduce the published magnitudes. This is a comparison with by-eye values, not a numeric reproduction of the figure.

### 6.2 Ten fold seeds: how reliable are the rankings? (new, issue 4)

`engine_stability.py`: fold seeds 0-9, 5 folds each (50 models per log), the same Apriori itemsets. Faithful mode with 10 repeats (as the original), fixed mode with 30 repeats. "Seed mean" = mean importance of an itemset over the 5 folds and all repeats of one fold seed. The 95 % interval is a t-interval over the 10 seed means (the seeds re-split the same log, so they are not fully independent; read the intervals as approximate).

| Quantity | Log | faithful, train | fixed, train | fixed, held-out |
|---|---|---|---|---|
| mean importance | f1 | 0.0632 | 0.0330 | 0.0023 |
| | f2 | 0.1499 | 0.0809 | 0.0235 |
| | f3 | 0.2458 | 0.0324 | 0.0105 |
| spread between the itemsets (std of the itemset means) | f1 | 0.0022 | 0.0072 | 0.0022 |
| | f2 | 0.0138 | 0.0155 | 0.0119 |
| | f3 | 0.0198 | 0.0084 | 0.0050 |
| half-width of the 95 % interval of an itemset mean (10 seeds) | f1 | 0.0053 | 0.0007 | 0.0027 |
| | f2 | 0.0019 | 0.0012 | 0.0034 |
| | f3 | 0.0039 | 0.0012 | 0.0024 |
| itemsets whose interval lies above 0 | f1 | 10 of 10 | 10 of 10 | **5 of 10** |
| | f2 | 10 of 10 | 10 of 10 | 10 of 10 |
| | f3 | 5 of 5 | 5 of 5 | 5 of 5 |
| Spearman between the rankings of two fold seeds, first 10 repeats | f1 | **0.52** | 0.97 | **0.62** |
| | f2 | 0.96 | 0.96 | 0.90 |
| | f3 | 0.91 | 0.94 | 0.75 |
| the same with 30 repeats | f1 | - | 0.97 | 0.73 |
| | f2 | - | 0.99 | 0.93 |
| | f3 | - | 0.94 | 0.84 |
| Spearman between seeds 0-4 pooled and seeds 5-9 pooled | f1 | 0.75 | 0.96 | 0.98 |
| | f2 | 1.00 | 0.99 | 0.94 |
| | f3 | 1.00 | 0.90 | 0.90 |
| share of itemset pairs whose order is significant (paired t over seeds, 5 %, uncorrected) | f1 | 51 % | 93 % | 73 % |
| | f2 | 96 % | 96 % | 84 % |
| | f3 | 90 % | 90 % | 90 % |

Where the noise of the held-out values comes from (fixed, held-out; mean over itemsets):

| Log | std between repeats, same fold and model | std between the fold means of one seed | std between seed means |
|---|---|---|---|
| f1 | 0.0080 | 0.0074 | 0.0037 |
| f2 | 0.0140 | 0.0119 | 0.0048 |
| f3 | 0.0078 | 0.0102 | 0.0034 |

Do the settings rank the itemsets alike? Spearman between the itemset means over all 10 seeds:

| Log | faithful/train vs fixed/train | faithful/train vs fixed/held-out | fixed/train vs fixed/held-out |
|---|---|---|---|
| f1 | 0.62 | 0.43 | 0.87 |
| f2 | 0.07 | 0.24 | 0.94 |
| f3 | -0.10 | -0.10 | 1.00 |

Reading:

- **Held-out, one split, 10 repeats is not enough.** Two splits agree on the ranking with Spearman 0.62 (f1), 0.90 (f2), 0.75 (f3). On f1 the itemsets differ by 0.002 while one seed mean has a standard deviation of 0.004; even with 10 seeds only 5 of the 10 itemsets are separable from zero.
- **More repeats help little, more fold seeds help.** The mean of 10 repeats in one fold has a repeat noise of about 0.008 / sqrt(10) = 0.0025 on f1, clearly below the spread between folds (0.0074). The spread between folds (a different model and a different held-out sample) is what dominates a seed mean. Going from 10 to 30 repeats raises the two-seed Spearman only from 0.62 to 0.73 on f1; pooling 5 seeds raises the agreement of two independent halves to 0.90-0.98.
- **Fixed mode on the training fold is precise** (interval about ±0.001, two-seed Spearman 0.94-0.97) and ranks the itemsets like the held-out fold does (0.87 / 0.94 / 1.00), at a 3 to 14 times higher level.
- **The original's ranking is weak evidence.** On f1 all ten itemsets lie between 0.061 and 0.069 with an interval of ±0.005; two splits agree with Spearman 0.52 and only half of the pairs can be ordered. On f2 and f3 the faithful ranking is stable across splits (0.96, 0.91) but unrelated to the ranking without accumulation (0.07 and -0.10 against fixed/train). On f2 it follows the order in which the itemsets are processed (Spearman 0.87 between processing position and faithful importance; -0.13 for fixed/train), which is what accumulation would produce; on f1 and f3 no such trend is visible (-0.14, 0.30).
- These runs use the original's Apriori itemsets (sizes 2 and 3, support at least 0.5). Rarer sets, such as low-coverage IMPresseD sets or most single activities, change fewer traces and will be noisier on the held-out fold. That was not measured here.

### 6.3 The two open options of fixed mode (new, issues 7 and 11)

`engine_fixed_options.py`: fold seeds 0-9, 5 folds, 10 repeats, same itemsets. Mean importance over everything with a 95 % interval over the 10 seed means; "difference" = paired difference to the default (sequential draw, pools from the whole log) per seed.

| Log | Scored on | Setting | mean importance [95 % interval] | difference to default [95 % interval] | Spearman with the default ranking |
|---|---|---|---|---|---|
| f1 | train | default | 0.0330 [0.0324, 0.0336] | | |
| f1 | train | uniform draw | 0.0292 [0.0286, 0.0297] | -0.0038 [-0.0041, -0.0035] | 0.93 |
| f1 | held-out | default | 0.0024 [-0.0002, 0.0050] | | |
| f1 | held-out | pools from training fold | 0.0025 [0.0000, 0.0051] | +0.0001 [-0.0000, +0.0003] | 0.96 |
| f1 | held-out | uniform draw | -0.0002 [-0.0024, 0.0020] | -0.0026 [-0.0034, -0.0019] | 0.24 |
| f2 | train | default | 0.0810 [0.0800, 0.0820] | | |
| f2 | train | uniform draw | 0.0694 [0.0685, 0.0702] | -0.0116 [-0.0120, -0.0113] | **0.52** |
| f2 | held-out | default | 0.0234 [0.0203, 0.0265] | | |
| f2 | held-out | pools from training fold | 0.0233 [0.0202, 0.0265] | -0.0001 [-0.0002, +0.0000] | 1.00 |
| f2 | held-out | uniform draw | 0.0175 [0.0145, 0.0206] | -0.0059 [-0.0069, -0.0049] | 0.84 |
| f3 | train | default | 0.0325 [0.0317, 0.0333] | | |
| f3 | train | uniform draw | 0.0297 [0.0291, 0.0303] | -0.0028 [-0.0033, -0.0023] | 0.90 |
| f3 | held-out | default | 0.0103 [0.0082, 0.0125] | | |
| f3 | held-out | pools from training fold | 0.0104 [0.0082, 0.0125] | +0.0000 [-0.0002, +0.0003] | 1.00 |
| f3 | held-out | uniform draw | 0.0085 [0.0067, 0.0103] | -0.0018 [-0.0026, -0.0010] | 0.70 |

Background numbers:

- **Position pools.** Of the (activity, position) pairs of the whole log, the share that occurs only in held-out cases is 6-10 % per fold on f1 (149-247 of 2415), 8-10 % on f2 (256-310 of 3250) and 7-11 % on f3 (140-224 of 2019). With pools from the training fold no trace became infeasible for the Apriori itemsets (0 in all runs); for single activities on the held-out fold 20 (f1) and 8 (f3) traces did (section 4.3). With training-fold pools more of the shifted events sit outside their pool (4.3 % instead of 3.5 % on f1), simply because the pools are smaller.
- **Draw.** The sequential draw puts the first activity of a set early more often and so moves the set further on average; the uniform draw shifts about 15 % fewer other events (f1 held-out, seed 0: 930,053 instead of 1,097,604).

Reading:

- `allowed_from` does not change the result measurably (difference between -0.0001 and +0.0001, every interval contains 0, Spearman with the default ranking 0.96-1.00). The choice can be made on principle: `'train'` is the strict option and costs nothing here.
- `draw` does change the result. The uniform draw gives a lower importance on the training fold (f1 -12 %, f2 -14 %, f3 -9 %) and on the held-out fold (every interval of the difference excludes 0). On f1 and f3 the training-fold ranking stays similar (Spearman 0.93 and 0.90), on **f2 it does not (0.52)**: there the draw also decides which itemsets come out on top. The low held-out Spearman on f1 (0.24) says little, because the held-out ranking on f1 is mostly noise in both settings (section 6.2). The paper does not say which distribution is meant, so this is a question for the supervisor; whatever is chosen must be the same for the Apriori and the IMPresseD runs.

## 7. What this means for the project decisions

- **B10 / C2 (re-implementation, runtime):** settled. A clean re-implementation is exact (faithful mode, itemsets and single activities) and fast. Runtime is no constraint: the stability study (10 fold seeds x 5 folds, three settings, up to 30 repeats, 35,000 importance values for f1) ran in under an hour per log on a loaded machine.
- **Reproduction part of the paper:** use `mode='faithful', score_on='train'`: `compute` for the itemset figure, `compute_single_activities` for the single-activity figure. Both are bit-identical with the original for the same folds and model. Report next to it that the original's ranking is unstable on f1 and reflects accumulation on f2 and f3 (section 6.2).
- **Variant / comparison part (Apriori vs IMPresseD) — needs a group decision and probably the supervisor (questions B2/B3):** which fold is scored. The measurements suggest this design, to be confirmed:
  1. compute fixed mode on both folds; they rank the Apriori itemsets alike (0.87-1.00);
  2. for held-out values use several fold seeds (10 seeds x 5 folds x 10 repeats is affordable) and report the mean with an interval over seeds, or the rank stability, never a ranking from one split;
  3. do not interpret differences between sets whose intervals overlap; on f1 the held-out importance of half of the Apriori itemsets is not distinguishable from zero.
- **Position pools (`allowed_from`):** pick one value and use it for both strategies. `'train'` avoids held-out information in the permutation step and made no measurable difference.
- **Draw (`draw`):** ask the supervisor whether drawing the new locations at random means the sequential procedure of the code or a uniform choice among all order-preserving placements. It is not a cosmetic choice (section 6.3). Until then keep the default `'sequential'` (it is what the original code does) and say so in the method section.
- **Limitations section of the paper:** sections 4.4, 6.1 and 6.2 give concrete numbers for the limits of the method (events that are shifted onto unobserved positions, training-fold scoring, accumulation, instability of the original ranking).
- **Encoder:** use `IndexEncoder` with the default `missing_order='sorted'` and fit the model on its matrix; it gives the same model as the original encoding and is reproducible across runs.
- **IMPresseD sets:** normalise the activity names with `normalise_activity` before passing them to the engine; the engine now rejects names it does not know instead of returning 0.

## 8. Design choices in fixed mode that the paper does not pin down (to confirm)

1. **Simultaneous placement.** All moved events of a trace are placed at their drawn positions in the final trace. The original moves them one by one, so later moves shift earlier ones. The paper's worked example (trace 1, both occurrences end on observed positions in the final trace) fits the simultaneous reading.
2. **Sequential windowed draw** by default, as in the original code. It is not uniform over all feasible position tuples (section 4.2). `draw='uniform'` is available and changes the importance level (section 6.3).
3. **Per-activity position pools** (`Allowed_locations`), as in the code and in the paper's feasibility sentence. The paper's example also shows joint tuples OL(L, (ER, CR)); joint pools are not implemented.
4. **Positions are clipped to the trace length** and a position already taken by another moved event of the same trace is excluded. The original does neither in the itemset routine.
5. **All-or-nothing per trace:** if any occurrence cannot be placed, the whole trace stays unchanged and is counted. This never happened for the Apriori itemsets; it happens for some held-out traces with training-fold pools (4 of 1004 random trials; 20 traces in the f1 single-activity grid).
6. A trace whose length equals the itemset size is skipped (same rule as the original, tools.py:522).
7. **Nothing to permute means importance 0.** If no scored trace contains the itemset, or no trace is changed in a repeat, fixed mode returns importance exactly 0 (with `n_traces_with_itemset` / `n_traces_changed` showing why) and warns about itemsets that occur in no scored trace. Faithful `compute` raises `ValueError` before it starts, because the original crashes there. The original's "fewer than 3 shuffled cases gives 0" guard exists only in its single-activity routine and is ported only there; fixed mode has no such guard, so an itemset contained in one or two scored traces gets whatever those traces produce. The table columns allow filtering such rows afterwards.
8. **Only the moved events are constrained** (section 4.4).

## 9. Limits and things that were not done

- The single-activity proof against the original covers the whole training fold of f1 and a 200-case subset of f3, with 2 repeats. f2 and the whole fold of f3 were not run (about an hour of the original's runtime each).
- The unconstrained variants of the original (`constrain=False`) are not ported.
- Exact column-order equality with the original is only checkable inside one Python process (hash-seed dependence, section 3.1).
- Itemset equality against the original: fold 0 of f1 and f3 by the builder, another fold of f1 and f2 by the verifier; not every fold of every log.
- The stability and option studies use the original's Apriori itemsets only (sizes 2 and 3, 10 / 10 / 5 sets). The project's planned sets (10 per length 1-3, Apriori and IMPresseD) were not available to this experiment; their noise level on the held-out fold has to be measured when they exist. The intervals treat the 10 fold seeds as independent replicates, which is an approximation.
- Faithful mode was run with 10 repeats in the stability study, fixed mode with 30; the "10 repeats" row uses the first 10 repeats of the fixed runs.
- `EventLog` supports logs without a `lifecycle:transition` column (all BPIC11 files). It raises `NotImplementedError` for the bpic2012 files.
- The uniform draw is uniform per occurrence, not over the joint placement of several occurrences in one trace.
- Timings were taken on a shared machine; section 5 says which ones were taken while it was quiet. No profiling (`cProfile`) of the original was done.
- The complete `test_engine.py` run (23 checks) was made before one small edit of the test file (a deterministic choice of the example itemset in `uniform_draw_f1`). The 18 quick checks were re-run on the final files and pass (the chi-square table of section 4.2 comes from that run); the 5 slow checks compare with the original and were not re-run. `engine.py` was not changed after any of the test runs reported here.
- The verifier's `verify_edge.json` still labels two cases "CRASH": an unknown activity name and a string instead of a list. These are now the intended `ValueError` / `TypeError` of the input validation; the script calls every exception a crash.
- Nothing failed in the final runs. During the work one new check failed once (`uniform_draw_f1` could not find an example trace because of a too strict selection rule in the test); the test was corrected, the engine was not changed.
