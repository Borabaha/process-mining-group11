# Experiment C1 — runtime of the original code and of the engine

Date: 2 Oct 2026, 14:48–16:43. Machine: AMD Ryzen 5 5600X (6 cores / 12 logical cores), 24 GB RAM, Windows 10. Interpreter: `project/.venv/Scripts/python.exe` (Python 3.12.10, pandas 2.3.3, numpy 2.5.3, scikit-learn 1.9.1, xgboost 3.4.1, mlxtend 0.25.0).

Feeds Guide Part 6: open questions **B14** (may we reduce folds/repeats?) and **B16** (single-activity analysis over all activities?), the work-plan dates, and the sizes of `config/demo.yaml` and `config/full.yaml`.

"One iteration" below always means one (itemset, repeat): the activity set is permuted once in every scored case that contains it and the model scores the result once.

## 1. Answer in short

1. **The original code needs 5–13 s per iteration** on the training fold (904 / 904 / 888 cases). For the Apriori sets: f1 6.4–7.3 s, f2 10.3–12.7 s, f3 5.1–5.8 s. The time hardly depends on the set: size 3 costs 0.4–2.2 s more than size 1 or 2, and sets that occur in few cases are within 5 % at sizes 1 and 2 and at most 19 % cheaper at size 3. The time per iteration does not grow from one iteration to the next, so the total is linear in folds x sets x repeats.
2. **Extrapolated totals for the original code** (formulas in section 6):

   | | f1 | f2 | f3 | all three |
   |---|---|---|---|---|
   | (a) original defaults, 5 folds x 10 itemsets x 10 repeats | 56 min | 92 min | 23 min (f3 has only 5 itemsets at support 0.5; 46 min with 10) | 2.9 h |
   | (b) full project grid, 2 strategies x 3 lengths x 10 sets x 5 folds x 10 repeats | 5.3 h | 9.2 h | 4.4 h | 18.9 h |
   | (c) single-activity analysis, all 164 / 207 / 156 activities x 5 folds x 10 repeats | 12.8 h | 25.1 h | 9.1 h | 47 h |

3. **The engine does the same work in seconds to minutes, and this was measured, not extrapolated** (one fold seed, 5 folds, 10 repeats):

   | | f1 | f2 | f3 | all three |
   |---|---|---|---|---|
   | (b) full project grid, fixed mode, held-out fold | 30 s | 40 s | 24 s | 1.7 min |
   | (b) full project grid, fixed mode, training fold | 80 s | 108 s | 59 s | 4.3 min |
   | (b) full project grid, faithful mode, training fold | 120 s | 159 s | 87 s | 6.3 min |
   | (c) single-activity analysis, fixed held-out / fixed train / faithful train | 37 / 72 / 78 s | 49 / 97 / 102 s | 29 / 53 / 56 s | 1.9 / 3.7 / 3.9 min |

   Per iteration the engine needs 0.004–0.08 s. On the same fold, model and sets it is 100–960 times faster than the original (section 5.2), and its faithful mode returned exactly the original's importance values in all 240 compared iterations (max abs difference 0).
4. **Use `XGBClassifier(n_jobs=1)`.** With XGBoost's default the engine keeps all 12 logical cores busy (8–10 cores of CPU time) without being faster. With one thread it is as fast (f1: 0.044 vs 0.050 s, 0.033 vs 0.034 s, 0.010 vs 0.012 s per iteration; f3 likewise), gives identical importance values (all 106,050 rows of the three logs), and the three logs can run at the same time: project grid plus single-activity analysis in all three settings for all three logs took **8.7 minutes** of wall time.
5. **Proposed settings** (section 7): `demo` = 1 fold seed, 5 folds, 2 repeats, fixed mode on both folds: 77 s for the three logs (measured). `full` = 10 fold seeds, 5 folds, 10 repeats: about 25 minutes for the project grid in fixed mode on both folds, about 1.4 hours for everything (project grid and single-activity analysis in fixed held-out, fixed train and faithful mode), three logs in parallel.
6. **Decisions:** runtime is no reason to reduce folds, repeats or the number of sets (B14), and the single-activity analysis over all activities is affordable for all three logs (B16), provided the engine is used. With the original code the project grid alone would take 19 hours and the single-activity analysis two days.

## 2. What was run

All commands from `C:/Users/Bohabara/Desktop/process mining/project/experiments`, with `PY=../.venv/Scripts/python.exe`. Nothing ran in parallel unless stated.

| Time | Command | Purpose | Output in `results/experiments/C1/` |
|---|---|---|---|
| 14:48–15:14 | `for d in f1 f2 f3; do $PY c1_original_vs_engine.py --dataset $d; done` | one fold: original vs engine per set size and strategy; one-off steps; model fits | `one_fold_<d>.json`, `one_fold_<d>_iterations.csv`, `one_fold_run.log` |
| 15:15–15:37 | `for d in f1 f2 f3; do $PY c1_engine_grid.py --dataset $d; done` | full project grid and single-activity analysis with the engine, 10 repeats | `engine_grid_<d>.json` (timings), `engine_grid_<d>.csv` (importance values, a by-product), `engine_grid_run.log` |
| 15:37–15:41 | `for d in f1 f2 f3; do $PY c1_engine_grid.py --dataset $d --repeats 2 --tag _demo; done` | the same with 2 repeats (demo size) | `engine_grid_<d>_demo.json`, `.csv` |
| 15:42–15:54 | `$PY c1_original_vs_engine.py --dataset f2 --tag _run2` | second measurement of f2 (the load changed during the first) | `one_fold_f2_run2.json`, `one_fold_f2_run2_iterations.csv`, `one_fold_run2.log` |
| 15:54–15:56 and 16:42–16:43 | `$PY c1_threads.py --dataset f1`, later `--dataset f3` | effect of XGBoost's thread count | `threads_f1.json`, `threads_f3.json`, `threads_run.log` |
| 15:56–16:05 | `$PY c1_engine_grid.py --dataset f1 --n-jobs 1 --tag _nj1` and the same for f2 and f3, **all three started at the same time** | full grid with one XGBoost thread, three logs in parallel (520 s wall for all) | `engine_grid_<d>_nj1.json`, `engine_grid_nj1_<d>.log`, `engine_grid_nj1_parallel.log` (the `.csv` files were deleted after they proved identical to `engine_grid_<d>.csv`) |
| 16:05–16:36 | `$PY c1_original_single.py --dataset f2`, **at the same time as** `$PY c1_original_single.py --dataset f1` followed by `--dataset f3` | one complete pass of the original single-activity routine (all activities, 1 fold, 1 repeat) | `original_single_<d>.json`, `original_single_<d>.csv` (one row per activity), `original_single_<d>.log` |
| after each step | `$PY c1_report.py` | tables and extrapolations | `tables.md`, `extrapolation.json` |

Prototype code (in `experiments/`, for the students to read, check and adapt):

| File | Content |
|---|---|
| `c1_common.py` | `LoadMeter` (wall time, CPU time of the process, CPU load of the whole machine through the Windows call `GetSystemTimes`), `IterationClock` (timestamps the progress lines the original prints, so every iteration of the unmodified original can be timed), `load_itemsets` (Apriori and IMPresseD sets per size) |
| `c1_original_vs_engine.py` | part 1: one fold, original and engine on the same model and sets |
| `c1_engine_grid.py` | part 2: the whole project grid and the single-activity analysis with the engine; options `--repeats`, `--fold-seed`, `--n-jobs`, `--tag` |
| `c1_original_single.py` | part 3: the original single-activity routine, timed per activity |
| `c1_threads.py` | part 5: XGBoost thread count |
| `c1_report.py` | part 4: all tables and extrapolations of this report from the JSON files |

The scripts use `engine.py`, `engine_reference.py` (experiment C2) and `apriori_selector.py` (C7) unchanged. The original repository was not edited; it is loaded read-only by `engine_reference.py` with the int-cast subclass.

## 3. Settings

| Setting | Value |
|---|---|
| Logs | BPIC11 f1 (1130 cases, 164 activities, 5939 feature columns), f2 (1130, 207, 8319), f3 (1111, 156, 4866) after the original's rare-activity filter |
| Folds | `make_folds(labels, k=5, seed=0)` of the engine (seeded `StratifiedKFold`); one-fold measurements use fold 0: 904 / 904 / 888 training cases, 226 / 226 / 223 held-out cases |
| Model | `XGBClassifier()` with default parameters. In part 1 one model per log, fitted on the original's training frame, is shared by the original and the engine. In part 2 one model per fold, fitted on the engine's matrix |
| Apriori sets | `AprioriSelector(min_support=0.45, max_len=3, top_k=10)` (C7): the ten most frequent sets of size 1, 2 and 3. 0.45 is the largest value on the C7 grid at which all three logs have at least ten sets per size; the top ten do not change for lower values |
| IMPresseD sets | the ten sets per length selected in C4 (`results/experiments/C4/impressed_sets_<d>.json`). They are that experiment's working selection, used here only as a realistic workload |
| Part 1 (one fold) | per log, strategy and size: the first 5 sets x 2 repeats = 10 iterations. Original: `itemset_permutation_importance(constrain=True)` on the training fold, as the original script calls it; time per iteration from the timestamps of its own progress line. Engine: `compute` on the same 5 sets x 2 repeats, 5 timing runs, each run includes encoding the scored fold and the baseline prediction; median of the runs |
| Part 2 (full grid) | 2 strategies x 3 sizes x 10 sets x 5 folds x 10 repeats = 3000 iterations per log and setting; single-activity analysis = all activities x 5 folds x 10 repeats; settings: faithful on the training fold, fixed on the training fold, fixed on the held-out fold (`allowed_from='log'`, `draw='sequential'`) |
| Part 3 (original, single activities) | `trace_permutation_importance(constrain=True, n_repeats=1)` on the training fold of fold 0, every activity of the log |
| Seeds | permutation seed 2023 (the original's default; `c1_engine_grid.py` uses 2023 + fold seed in fixed mode, which is 2023 here); fold seed 0 |
| Statistic | median, as asked; minimum, maximum and mean are shown next to it |

Size-1 sets were run through the original's **itemset** routine in part 1 (it accepts sets of one activity). The original's own single-activity routine is a different function and was timed separately in part 3.

## 4. Machine load during the measurements

No other Python process was running when the process list was checked (14:46 and 15:12), but the machine was never idle: a sample at 15:15 showed Discord, the Claude desktop app, OBS and browser components, and over the afternoon other programs used between 1 and 4 logical cores.

| Run | Log | system load just before the start | others' load during the run | this process (cores used) |
|---|---|---|---|---|
| original itemset routine (groups of 10 iterations) | f1 |  | 11-13 % | 1.19 |
| engine grid, 10 repeats, default threads | f1 | 17 % | 34 % | 7.88 |
| engine grid, 2 repeats (demo), default threads | f1 | 13 % | 15 % | 10.10 |
| engine grid, 10 repeats, n_jobs=1, three logs at the same time | f1 | 26 % | 37 % | 1.02 |
| original single-activity pass | f1 | 13 % | 25 % | 1.16 |
| original itemset routine (groups of 10 iterations) | f2 |  | 12-24 % | 1.12 |
| engine grid, 10 repeats, default threads | f2 | 32 % | 29 % | 8.50 |
| engine grid, 2 repeats (demo), default threads | f2 | 10 % | 28 % | 8.66 |
| engine grid, 10 repeats, n_jobs=1, three logs at the same time | f2 | 27 % | 30 % | 1.01 |
| original single-activity pass | f2 | 13 % | 23 % | 1.10 |
| original itemset routine (groups of 10 iterations) | f3 |  | 15-24 % | 1.20 |
| engine grid, 10 repeats, default threads | f3 | 31 % | 18 % | 9.80 |
| engine grid, 2 repeats (demo), default threads | f3 | 21 % | 18 % | 9.81 |
| engine grid, 10 repeats, n_jobs=1, three logs at the same time | f3 | 26 % | 39 % | 1.02 |
| original single-activity pass | f3 | 20 % | 23 % | 1.21 |

How to read the table: "others' load" is the CPU use of the whole machine minus the CPU time of the measuring process, in percent of the 12 logical cores (8.3 % = one core).

- The original is single-threaded apart from the XGBoost prediction (1.1–1.2 cores). Other programs used 11–24 % during its runs.
- The engine with XGBoost's default threads used 8–10 cores of CPU time itself and the machine was at 100 % for the whole run. Its timings were therefore taken on a saturated machine.
- In the rows where several measuring processes ran at the same time (the `n_jobs=1` grid, the single-activity passes), "others' load" contains the sibling processes (8–10 % each).

## 5. Measurements

### 5.1 One-off steps and model fits

| Step | f1 | f2 | f3 |
|---|---|---|---|
| original DataManager() | 0.25 s | 0.35 s | 0.22 s |
| original index_encoding(whole log) + int cast | 3.73 s | 6.45 s | 2.79 s |
| engine EventLog() | 0.11 s | 0.14 s | 0.11 s |
| engine IndexEncoder() | 0.00 s | 0.01 s | 0.00 s |
| engine IndexEncoder.transform(whole log) | 0.01 s | 0.01 s | 0.00 s |
| Apriori per size (max_len 3, min_support 0.45) | 0.02 s | 0.02 s | 0.01 s |
| fit on original DataFrame (int64): median of 5 folds (min-max) | 2.24 s (2.18-2.35) | 3.18 s (3.03-3.43) | 2.00 s (1.88-2.09) |
| fit on engine matrix (int8): median of 5 folds (min-max) | 0.41 s (0.37-0.41) | 0.60 s (0.54-0.88) | 0.39 s (0.38-0.45) |
| baseline predict, original DataFrame, training fold: median of 5 folds (min-max) | 0.39 s (0.38-0.40) | 0.54 s (0.51-0.58) | 0.33 s (0.33-0.34) |
| baseline predict, engine matrix, training fold: median of 5 folds (min-max) | 0.01 s (0.01-0.01) | 0.01 s (0.01-0.02) | 0.01 s (0.01-0.01) |

- S (setup of one log) is 4.0 / 6.8 / 3.0 s for the original and 0.1–0.15 s for the engine.
- F (one model fit plus the baseline prediction on the training fold) is 2.6 / 3.7 / 2.3 s for the original's DataFrame and 0.4–0.6 s for the engine's int8 matrix.
- Both are negligible next to the permutation loop of the original, and a few percent of the engine's total.
- The f2 column pools the two f2 runs (10 folds).

### 5.2 Seconds per iteration: original and engine on the same fold, model and sets

| Log | Sets | Size | training cases containing a set (min-max) | original: median (min-max), mean | others' CPU load | engine faithful, train | engine fixed, train | engine fixed, held-out | speed-up (faithful / fixed train / fixed held-out) | max abs diff faithful vs original |
|---|---|---|---|---|---|---|---|---|---|---|
| f1 | apriori | 1 | 517-636 | 6.44 s (6.34-7.02), 6.49 s | 11 % | 0.0453 s | 0.0338 s | 0.0135 s | 142x / 190x / 477x | 0 |
| f1 | apriori | 2 | 514-538 | 6.50 s (6.39-7.09), 6.54 s | 11 % | 0.0498 s | 0.0352 s | 0.0128 s | 131x / 185x / 509x | 0 |
| f1 | apriori | 3 | 509-516 | 7.25 s (7.17-7.54), 7.29 s | 11 % | 0.0623 s | 0.0393 s | 0.0138 s | 116x / 184x / 525x | 0 |
| f1 | impressed | 1 | 117-636 | 6.17 s (5.67-6.82), 6.20 s | 12 % | 0.0366 s | 0.0259 s | 0.0103 s | 169x / 239x / 599x | 0 |
| f1 | impressed | 2 | 245-492 | 6.21 s (5.81-6.72), 6.25 s | 13 % | 0.0394 s | 0.0266 s | 0.0104 s | 158x / 234x / 596x | 0 |
| f1 | impressed | 3 | 25-236 | 5.89 s (5.59-6.46), 5.92 s | 13 % | 0.0211 s | 0.0136 s | 0.0063 s | 280x / 433x / 940x | 0 |
| f2 | apriori | 1 | 527-856 | 10.30 s (9.60-11.65), 10.41 s [n = 20] | 12 % and 14 % | 0.0635 s | 0.0496 s | 0.0182 s | 162x / 208x / 567x | 0 |
| f2 | apriori | 2 | 539-629 | 10.50 s (9.84-11.27), 10.52 s [n = 20] | 12 % and 15 % | 0.0698 s | 0.0508 s | 0.0188 s | 150x / 207x / 558x | 0 |
| f2 | apriori | 3 | 514-589 | 12.74 s (11.95-14.41), 13.02 s [n = 20] | 24 % and 15 % | 0.0816 s | 0.0584 s | 0.0197 s | 156x / 218x / 647x | 0 |
| f2 | impressed | 1 | 119-856 | 10.76 s (9.71-11.77), 10.69 s [n = 20] | 20 % and 18 % | 0.0531 s | 0.0380 s | 0.0145 s | 202x / 283x / 744x | 0 |
| f2 | impressed | 2 | 242-334 | 10.33 s (9.99-11.38), 10.46 s [n = 20] | 17 % and 23 % | 0.0371 s | 0.0284 s | 0.0118 s | 278x / 363x / 874x | 0 |
| f2 | impressed | 3 | 210-273 | 11.22 s (10.48-11.85), 11.14 s [n = 20] | 15 % and 24 % | 0.0391 s | 0.0295 s | 0.0117 s | 287x / 380x / 963x | 0 |
| f3 | apriori | 1 | 445-597 | 5.08 s (4.91-5.47), 5.13 s | 15 % | 0.0393 s | 0.0290 s | 0.0117 s | 129x / 175x / 436x | 0 |
| f3 | apriori | 2 | 443-468 | 5.40 s (4.96-5.60), 5.31 s | 20 % | 0.0445 s | 0.0312 s | 0.0124 s | 121x / 173x / 436x | 0 |
| f3 | apriori | 3 | 437-447 | 5.82 s (5.65-6.16), 5.89 s | 20 % | 0.0569 s | 0.0361 s | 0.0146 s | 102x / 161x / 400x | 0 |
| f3 | impressed | 1 | 30-597 | 4.99 s (4.53-6.28), 5.10 s | 22 % | 0.0308 s | 0.0240 s | 0.0097 s | 162x / 208x / 514x | 0 |
| f3 | impressed | 2 | 160-419 | 5.27 s (4.78-6.10), 5.32 s | 24 % | 0.0399 s | 0.0292 s | 0.0121 s | 132x / 180x / 437x | 0 |
| f3 | impressed | 3 | 19-151 | 4.82 s (4.57-5.38), 4.89 s | 23 % | 0.0175 s | 0.0122 s | 0.0063 s | 275x / 393x / 763x | 0 |

- f2 was measured twice (section 5.3); its rows pool both runs (20 iterations of the original, 10 timing runs of the engine).
- **Original.** 5.9–7.3 s on f1, 10.3–12.7 s on f2, 4.8–5.8 s on f3. Set size 1 and 2 cost the same; size 3 adds 0.75 s on f1, 2.2 s on f2 and 0.4 s on f3 (the original enumerates all index triples of every trace that contains the set, and f2 has the longest traces, up to 40 events). The measured IMPresseD sets occur in 19–856 training cases, the Apriori sets in 437–856; the original is within 5 % on both at sizes 1 and 2 and 12–19 % faster on the IMPresseD sets at size 3.
- **Where the original's time goes** (from part 3, section 5.5): an activity that occurs in fewer than 3 training cases goes through the same loop over all cases but is not re-encoded or scored; it costs 0.85 / 0.94 / 0.75 s. A scored activity costs 6.2 / 10.0 / 4.6 s. So about 85–90 % of an iteration is the re-encoding of the shuffled cases, the rebuilding of the full feature frame and the prediction. This part grows only slowly with the amount that was shuffled (f1: about 6.2 s plus 1.5 ms per moved event; an activity with 3 events costs 5.9 s, the most frequent one with 2059 events 8.8 s). The original's time therefore grows with the size of the log's feature matrix (5939 / 8319 / 4866 columns) rather than with the set. It was not profiled further.
- **Linear scaling.** Within the groups of 10 iterations the first iteration is 0.3–0.7 s slower (it contains the baseline prediction); iterations 2–5 and 6–10 have the same median (f1 6.39 and 6.44 s; f2, first run, 10.48 and 10.51 s; f3 5.23 and 5.04 s). Repeat 1 costs the same as repeat 0. There is no growth over iterations, so folds x sets x repeats can be multiplied.
- **Engine.** 0.006–0.08 s per iteration. Its time does depend on the set: it is roughly proportional to the number of scored cases that contain the set (IMPresseD size 3: 0.006–0.04 s; Apriori size 3: 0.014–0.08 s), and the held-out fold (a quarter of the cases) is 2.5 to 3 times cheaper than the training fold.
- **Speed-up** on identical work: 102–287 times (faithful, training fold), 161–433 times (fixed, training fold), 400–963 times (fixed, held-out fold; the original scores the training fold, so this last ratio compares different folds).
- **Equality.** In all 18 groups, and in the 6 groups of the second f2 run, faithful mode returned the original's importance values exactly (240 values, max abs difference 0). This extends the C2 proof to sets of size 1, to IMPresseD sets and to f2.
- Cross-check with earlier measurements on f1: the scout measured 6.7–7.2 s, C2 6.78 s per iteration for the original's ten itemsets (7 of size 2, 3 of size 3). The medians here give (7 x 6.50 + 3 x 7.25) / 10 = 6.73 s.

### 5.3 Repeatability: f2 measured twice

| Log | Run | Sets | Size | original: median (min-max) | others' CPU load | engine faithful, train | engine fixed, train | engine fixed, held-out |
|---|---|---|---|---|---|---|---|---|
| f2 | run1 | apriori | 1 | 10.10 s (9.60-11.65) | 12 % | 0.0593 s | 0.0479 s | 0.0173 s |
| f2 | run1 | apriori | 2 | 10.19 s (9.84-10.97) | 12 % | 0.0676 s | 0.0503 s | 0.0194 s |
| f2 | run1 | apriori | 3 | 13.90 s (12.67-14.41) | 24 % | 0.0944 s | 0.0599 s | 0.0204 s |
| f2 | run1 | impressed | 1 | 11.18 s (9.85-11.45) | 20 % | 0.0515 s | 0.0375 s | 0.0145 s |
| f2 | run1 | impressed | 2 | 10.23 s (9.99-11.02) | 17 % | 0.0349 s | 0.0280 s | 0.0118 s |
| f2 | run1 | impressed | 3 | 10.68 s (10.48-11.11) | 15 % | 0.0390 s | 0.0283 s | 0.0116 s |
| f2 | run2 | apriori | 1 | 10.55 s (10.26-11.19) | 14 % | 0.0649 s | 0.0497 s | 0.0184 s |
| f2 | run2 | apriori | 2 | 10.68 s (10.42-11.27) | 15 % | 0.0738 s | 0.0510 s | 0.0184 s |
| f2 | run2 | apriori | 3 | 12.34 s (11.95-12.93) | 15 % | 0.0770 s | 0.0527 s | 0.0190 s |
| f2 | run2 | impressed | 1 | 10.24 s (9.71-11.77) | 18 % | 0.0551 s | 0.0408 s | 0.0144 s |
| f2 | run2 | impressed | 2 | 10.62 s (10.20-11.38) | 23 % | 0.0402 s | 0.0302 s | 0.0119 s |
| f2 | run2 | impressed | 3 | 11.52 s (11.33-11.85) | 24 % | 0.0412 s | 0.0318 s | 0.0117 s |

The first f2 run saw the load of other programs rise from 12 % to 24 % during the Apriori size-3 group (13.9 s). The second run gave 12.3 s for that group at 15 % load. The other groups differ by 4–8 % between the two runs, in both directions. Read every time of the original as ± 5–10 %.

### 5.4 The engine in the full grid (measured)

| Log | Part | Sets | Size | engine faithful, training fold | engine fixed, training fold | engine fixed, held-out fold |
|---|---|---|---|---|---|---|
| f1 | grid | apriori | 1 | 0.0410 s | 0.0311 s | 0.0118 s |
| f1 | grid | apriori | 2 | 0.0522 s | 0.0360 s | 0.0130 s |
| f1 | grid | apriori | 3 | 0.0686 s | 0.0397 s | 0.0141 s |
| f1 | grid | impressed | 1 | 0.0312 s | 0.0202 s | 0.0084 s |
| f1 | grid | impressed | 2 | 0.0296 s | 0.0193 s | 0.0081 s |
| f1 | grid | impressed | 3 | 0.0133 s | 0.0095 s | 0.0047 s |
| f1 | single | all activities | 1 | 0.0095 s | 0.0086 s | 0.0046 s |
| f2 | grid | apriori | 1 | 0.0573 s | 0.0423 s | 0.0147 s |
| f2 | grid | apriori | 2 | 0.0677 s | 0.0463 s | 0.0166 s |
| f2 | grid | apriori | 3 | 0.0813 s | 0.0467 s | 0.0174 s |
| f2 | grid | impressed | 1 | 0.0354 s | 0.0252 s | 0.0102 s |
| f2 | grid | impressed | 2 | 0.0433 s | 0.0306 s | 0.0116 s |
| f2 | grid | impressed | 3 | 0.0357 s | 0.0229 s | 0.0095 s |
| f2 | single | all activities | 1 | 0.0098 s | 0.0092 s | 0.0048 s |
| f3 | grid | apriori | 1 | 0.0304 s | 0.0232 s | 0.0093 s |
| f3 | grid | apriori | 2 | 0.0378 s | 0.0257 s | 0.0100 s |
| f3 | grid | apriori | 3 | 0.0484 s | 0.0292 s | 0.0110 s |
| f3 | grid | impressed | 1 | 0.0214 s | 0.0153 s | 0.0072 s |
| f3 | grid | impressed | 2 | 0.0237 s | 0.0151 s | 0.0065 s |
| f3 | grid | impressed | 3 | 0.0116 s | 0.0075 s | 0.0039 s |
| f3 | single | all activities | 1 | 0.0070 s | 0.0068 s | 0.0037 s |

This table is the measured counterpart of the table in section 5.2 for the real workload (10 sets, 10 repeats, every fold). The values are a little lower than there because the sets ranked 6–10 occur in fewer cases than the top five, and because the per-call overhead is spread over 100 iterations instead of 10. A single-activity iteration costs 0.004–0.010 s on average.

By-products of these runs (not analysed here; the importance values are in `engine_grid_<d>.csv`):

- Faithful mode ran without error on all 30 IMPresseD sets of every log and every fold (it would stop where the original crashes).
- On the held-out fold 3 / 3 / 5 (IMPresseD set, fold) pairs of f1 / f2 / f3 have no trace that contains the set; fixed mode returns importance 0 for them. The analysis has to treat these rows as "not measurable", as the C2 verifier already recommended.

### 5.5 The original single-activity routine: one complete pass (measured)

One pass = all activities of the log, one fold (training fold of fold 0), one repeat.

| Log | activities | scored | skipped by the original (fewer than 3 training cases) | skipped at once (one allowed position) | time of the pass | per scored activity: median (min–max) | per skipped activity: median | linear fit over the scored activities |
|---|---|---|---|---|---|---|---|---|
| f1 | 164 | 143 | 17 | 4 | 922.6 s = 15.4 min | 6.19 s (5.90–8.83) | 0.85 s | 6.15 s + 1.5 ms per event of the activity (R² 0.78) |
| f2 | 207 | 176 | 26 | 5 | 1808.0 s = 30.1 min | 10.00 s (8.72–13.93) | 0.94 s | 9.83 s + 2.1 ms per event (R² 0.60) |
| f3 | 156 | 135 | 18 | 3 | 651.6 s = 10.9 min | 4.61 s (3.99–6.60) | 0.75 s | 4.57 s + 1.3 ms per event (R² 0.88) |

- The cost per scored activity is close to the itemset routine's cost per iteration (f1 6.19 s against 6.44 s for size-1 sets).
- The Guide's planned shortcut "time 5 single activities and multiply" would have overestimated the pass by 11–17 % (median of the first five scored activities x number of activities with more than one allowed position: 1037 / 2108 / 721 s), because it ignores the cheap skipped activities. The complete pass was run instead.
- C2 measured 12.5 s per (activity, repeat) for f1 on a heavily loaded machine (4114 s for 2 repeats); here one repeat took 923 s.

### 5.6 XGBoost threads

| Log | XGBoost n_jobs | one model fit | engine faithful, training fold: s per iteration (cores used) | engine fixed, training fold: s per iteration (cores used) | engine fixed, held-out fold: s per iteration (cores used) | importance values equal to the default's |
|---|---|---|---|---|---|---|
| f1 | default (all cores) | 0.70 s | 0.0503 s (8.7) | 0.0337 s (8.7) | 0.0121 s (8.9) | yes |
| f1 | 1 | 1.27 s | 0.0439 s (1.0) | 0.0334 s (1.0) | 0.0095 s (1.0) | yes |
| f1 | 2 | 0.78 s | 0.0412 s (2.0) | 0.0288 s (2.0) | 0.0091 s (2.0) | yes |
| f1 | 4 | 0.58 s | 0.0429 s (3.9) | 0.0293 s (4.0) | 0.0101 s (4.0) | yes |
| f3 | default (all cores) | 0.56 s | 0.0374 s (10.3) | 0.0250 s (10.2) | 0.0095 s (10.2) | yes |
| f3 | 1 | 1.01 s | 0.0319 s (1.0) | 0.0233 s (1.0) | 0.0074 s (1.0) | yes |
| f3 | 2 | 0.58 s | 0.0299 s (2.0) | 0.0207 s (2.0) | 0.0067 s (2.0) | yes |
| f3 | 4 | 0.45 s | 0.0300 s (4.0) | 0.0206 s (4.0) | 0.0073 s (4.0) | yes |

`XGBClassifier()` takes all logical cores. The engine predicts a few hundred rows per iteration, so the extra threads mostly wait and burn CPU. With `n_jobs=1` the loop is not slower on f1 or f3 (it is 1–22 % faster), a model fit takes 1.0–1.3 s instead of 0.6–0.7 s, and the importance values are identical. Two threads are marginally faster than one. The full grid confirms it on all three logs: `engine_grid_<d>_nj1` used 1.0 core per log, ran the three logs side by side in 375 / 515 / 316 s (default threads, one log at a time: 420 / 560 / 310 s), and all 33,600 + 40,050 + 32,400 importance and baseline values equal those of the default-thread run (max abs difference 0).

## 6. Extrapolations

Notation: S = setup of one log (load and encode), F = one model fit plus the baseline prediction, t = median seconds per iteration of the measured group (table of section 5.2), 5 folds, 10 repeats.

### 6.1 (a) The original defaults: 5 folds x the original's itemsets x 10 repeats

Formula: **T = S + 5 F + 5 x 10 x (n2 x t(size 2) + n3 x t(size 3))**, where n2 and n3 are the numbers of itemsets of size 2 and 3 that the original selects at `min_support` 0.5, `top_k` 10, and t comes from the Apriori rows of the table in section 5.2. The engine columns use the same formula with the engine's S, F and t.

| Log | itemsets at min_support 0.5 | iterations | t original (size 2 / size 3) | original: S + 5 F | original: total | original, if 10 itemsets (500 iterations) | engine faithful, training fold | engine fixed, training fold | engine fixed, held-out fold |
|---|---|---|---|---|---|---|---|---|---|
| f1 | 10 (7 of size 2, 3 of size 3) | 500 | 6.50 s / 7.25 s | 17.1 s | 56.3 min | 56.3 min | 29.0 s | 20.4 s | 8.76 s |
| f2 | 10 (8 of size 2, 2 of size 3) | 500 | 10.50 s / 12.74 s | 25.4 s | 91.6 min | 91.6 min | 39.3 s | 29.3 s | 12.7 s |
| f3 | 5 (4 of size 2, 1 of size 3) | 250 | 5.40 s / 5.82 s | 14.6 s | 23.1 min | 46.0 min | 13.9 s | 10.2 s | 5.34 s |

- f3 has only 5 itemsets of size 2 or more at support 0.5, so the original's default run on f3 is 250 iterations (23 min). With ten itemsets of the same size mix it would be 46 min.
- For the engine, C2 measured the same grid directly: 26.3 / 16.9 / 6.0 s (f1), 33.7 / 22.4 / 7.9 s (f2), 10.9 / 7.2 / 2.8 s (f3) for the loop without S and F. The extrapolations here are consistent with that.
- Not included for the original: writing the CSV and the box plot at the end of its script, and the time it spends printing pandas warnings (they were suppressed).

### 6.2 (b) The full project grid: 2 strategies x 3 lengths x 10 sets x 5 folds x 10 repeats

Formula for the original: **T = S + 5 F + 5 x 10 x 10 x (sum of t over the 2 strategies x 3 sizes)** = S + 5 F + 500 x (sum of the log's six medians in the table of section 5.2). For the engine the loop was run and timed (3000 iterations per log and setting), so its columns are measurements; S + 5 F is the measured setup plus five fits of the same run.

| Log | iterations | original: total (extrapolated) | original: Apriori half / IMPresseD half | engine faithful, training fold: loop MEASURED | engine fixed, training fold: loop MEASURED | engine fixed, held-out fold: loop MEASURED | engine: S + 5 F (measured) |
|---|---|---|---|---|---|---|---|
| f1 | 3000 | 5.3 h | 2.8 h / 2.5 h | 119.8 s | 79.6 s | 30.0 s | 3.14 s |
| f2 | 3000 | 9.2 h | 4.7 h / 4.5 h | 2.7 min | 108.0 s | 40.2 s | 4.24 s |
| f3 | 3000 | 4.4 h | 2.3 h / 2.1 h | 86.5 s | 58.9 s | 24.2 s | 2.47 s |
| all three (incl. S + 5 F) | 9000 | 18.9 h |  | 6.3 min | 4.3 min | 104.3 s |  |

How good is the formula? For the engine both numbers exist:

| Log | engine faithful, training fold: extrapolated / measured | engine fixed, training fold: extrapolated / measured | engine fixed, held-out fold: extrapolated / measured |
|---|---|---|---|
| f1 | 2.1 min / 119.8 s = 1.06 | 87.2 s / 79.6 s = 1.10 | 33.5 s / 30.0 s = 1.12 |
| f2 | 2.9 min / 2.7 min = 1.08 | 2.1 min / 108.0 s = 1.18 | 47.3 s / 40.2 s = 1.18 |
| f3 | 114.4 s / 86.5 s = 1.32 | 80.9 s / 58.9 s = 1.37 | 33.3 s / 24.2 s = 1.38 |

The formula overestimates the engine by 6–38 %, because the five measured sets of every group are the most frequent ones and the engine's time follows the number of cases. The original's time hardly depends on the set (section 5.2), so its extrapolation should be closer; its uncertainty is the ± 5–10 % of the load.

### 6.3 (c) Single-activity analysis: all activities x 5 folds x 10 repeats

Formula for the original: **T = S + 5 F + 5 x 10 x P**, with P = the measured time of one complete pass (section 5.5). The engine columns are measured (`compute_single_activities`, all activities, 5 folds, 10 repeats).

| Log | activities | iterations | original: one pass (1 fold, 1 repeat), measured | original: median per scored activity | original: S + 5 F + 50 passes (extrapolated) | engine faithful, training fold: MEASURED | engine fixed, training fold: MEASURED | engine fixed, held-out fold: MEASURED |
|---|---|---|---|---|---|---|---|---|
| f1 | 164 | 8200 | 15.4 min | 6.19 s | 12.8 h | 77.8 s | 72.2 s | 37.4 s |
| f2 | 207 | 10350 | 30.1 min | 10.0 s | 25.1 h | 101.7 s | 96.8 s | 49.3 s |
| f3 | 156 | 7800 | 10.9 min | 4.61 s | 9.1 h | 55.7 s | 53.4 s | 28.9 s |

All three logs: 47 hours with the original; 1.9 / 3.7 / 3.9 minutes with the engine (fixed held-out / fixed train / faithful train, loops without S and F).

## 7. Proposed settings and their wall time on this machine

Engine wall time for several sizes. T = number of fold seeds x (S + 5 F + the loops that are switched on). The rows "measured" are measurements; the "full" rows are 10 times the measured one-seed run (every fold seed repeats exactly the same amount of work).

| Setting | What is computed | f1 | f2 | f3 | all three logs |
|---|---|---|---|---|---|
| demo: 1 fold seed x 5 folds x 2 repeats; default threads, logs one after the other (measured) | project grid, fixed, held-out fold | 8.31 s | 12.1 s | 7.13 s | 27.6 s |
| demo: 1 fold seed x 5 folds x 2 repeats; default threads, logs one after the other (measured) | project grid, fixed, both folds | 22.7 s | 34.7 s | 19.7 s | 77.0 s |
| demo: 1 fold seed x 5 folds x 2 repeats; default threads, logs one after the other (measured) | project grid, fixed both folds + faithful | 43.5 s | 65.3 s | 37.8 s | 2.4 min |
| demo: 1 fold seed x 5 folds x 2 repeats; default threads, logs one after the other (measured) | project grid + single-activity analysis, all three settings | 76.4 s | 115.0 s | 65.8 s | 4.3 min |
| 1 fold seed x 5 folds x 10 repeats; default threads, logs one after the other (measured) | project grid, fixed, held-out fold | 33.2 s | 44.5 s | 26.6 s | 104.3 s |
| 1 fold seed x 5 folds x 10 repeats; default threads, logs one after the other (measured) | project grid, fixed, both folds | 112.7 s | 2.5 min | 85.5 s | 5.8 min |
| 1 fold seed x 5 folds x 10 repeats; default threads, logs one after the other (measured) | project grid, fixed both folds + faithful | 3.9 min | 5.2 min | 2.9 min | 11.9 min |
| 1 fold seed x 5 folds x 10 repeats; default threads, logs one after the other (measured) | project grid + single-activity analysis, all three settings | 7.0 min | 9.3 min | 5.2 min | 21.5 min |
| 1 fold seed x 5 folds x 10 repeats; n_jobs=1, three logs at the same time (measured) | project grid, fixed, held-out fold | 32.0 s | 42.1 s | 27.9 s | 42.1 s |
| 1 fold seed x 5 folds x 10 repeats; n_jobs=1, three logs at the same time (measured) | project grid, fixed, both folds | 107.6 s | 2.5 min | 93.1 s | 2.5 min |
| 1 fold seed x 5 folds x 10 repeats; n_jobs=1, three logs at the same time (measured) | project grid, fixed both folds + faithful | 3.6 min | 4.9 min | 3.0 min | 4.9 min |
| 1 fold seed x 5 folds x 10 repeats; n_jobs=1, three logs at the same time (measured) | project grid + single-activity analysis, all three settings | 6.2 min | 8.6 min | 5.3 min | 8.6 min |
| full: 10 fold seeds x 5 folds x 10 repeats; default threads, logs one after the other (10 x measured) | project grid, fixed, held-out fold | 5.5 min | 7.4 min | 4.4 min | 17.4 min |
| full: 10 fold seeds x 5 folds x 10 repeats; default threads, logs one after the other (10 x measured) | project grid, fixed, both folds | 18.8 min | 25.4 min | 14.2 min | 58.4 min |
| full: 10 fold seeds x 5 folds x 10 repeats; default threads, logs one after the other (10 x measured) | project grid, fixed both folds + faithful | 38.8 min | 52.0 min | 28.7 min | 119.4 min |
| full: 10 fold seeds x 5 folds x 10 repeats; default threads, logs one after the other (10 x measured) | project grid + single-activity analysis, all three settings | 70.0 min | 93.3 min | 51.7 min | 3.6 h |
| full: 10 fold seeds x 5 folds x 10 repeats; n_jobs=1, three logs at the same time (10 x measured) | project grid, fixed, held-out fold | 5.3 min | 7.0 min | 4.6 min | 7.0 min |
| full: 10 fold seeds x 5 folds x 10 repeats; n_jobs=1, three logs at the same time (10 x measured) | project grid, fixed, both folds | 17.9 min | 25.0 min | 15.5 min | 25.0 min |
| full: 10 fold seeds x 5 folds x 10 repeats; n_jobs=1, three logs at the same time (10 x measured) | project grid, fixed both folds + faithful | 35.7 min | 49.3 min | 30.3 min | 49.3 min |
| full: 10 fold seeds x 5 folds x 10 repeats; n_jobs=1, three logs at the same time (10 x measured) | project grid + single-activity analysis, all three settings | 62.5 min | 85.9 min | 52.7 min | 85.9 min |

Proposal:

| Config | Settings | What it computes | Expected wall time |
|---|---|---|---|
| `config/demo.yaml` | f1, f2, f3; both strategies; lengths 1–3; 10 sets; `k_fold: 5`; one fold seed; `n_repeats: 2`; fixed mode scored on the held-out and on the training fold | the whole project grid, small | **77 s** for the permutation part (measured: 23 + 35 + 20 s). Add the set selection (Apriori below 0.1 s per log; IMPresseD 2.4–7.1 s mining plus 0.9–1.7 s projection per log according to C4) and Python start-up: about 2 minutes in total. Held-out fold only: 28 s. f1 only: 23 s |
| `config/full.yaml` | as above with fold seeds 0–9 and `n_repeats: 10`; fixed mode on both folds; `XGBClassifier(n_jobs=1)`; one process per log | the project grid with the 10 fold seeds that C2 found necessary for held-out rankings | **about 25 min** with the three logs in parallel (58 min one after the other with default threads) |
| `config/full.yaml` plus reproduction | additionally faithful mode on the training fold for the same grid, and the single-activity analysis in all three settings | everything | about 49 min without, **about 1.4 h** with the single-activity analysis, three logs in parallel (2.0 h / 3.6 h one after the other). Faithful mode does not need 10 fold seeds for the reproduction; with one seed the faithful part shrinks to a few minutes |
| original code, for comparison | repository defaults | 500 iterations (f3: 250) | 56 / 92 / 23 min per log; not needed, because faithful mode gives the same numbers |

Practical rules that follow from the measurements:

- Keep 5 folds and 10 repeats also in the demo if you like: the one-seed full grid in fixed mode on both folds takes 5.8 minutes for the three logs (2.5 minutes in parallel). Two repeats are only needed if the demo has to finish in about a minute.
- Fit with `XGBClassifier(n_jobs=1)` and start one process per log. Do not start several engine processes with default threads; each of them tries to use all cores.
- Rule of thumb for other sizes: time is proportional to fold seeds x folds x repeats x sets. Per 1000 iterations budget 7–80 s on the training fold and 4–20 s on the held-out fold (table in section 5.4).
- Another computer will give other absolute times. The engine is single-threaded Python, so its time scales with single-core speed; the ratios between the rows stay.

## 8. What this means for the project decisions

- **B14 (may we reduce repeats or folds?).** Not needed. The Guide's estimate "≈ 5.6 h per dataset" for the original code is confirmed for f1 (5.3 h) and is too low for f2 (9.2 h); f3 needs 4.4 h. With the engine the same grid takes 0.4–2.7 minutes per log and setting. The question to the supervisor can be reduced to "may we fix the fold seed" (and, after C2, "may we use several fold seeds").
- **B16 (single-activity analysis over all activities or only the selected length-1 sets?).** Both are affordable. All activities x 5 folds x 10 repeats cost 29–102 s per log and setting with the engine; the Fig. 3 reproduction in faithful mode costs 78 / 102 / 56 s. With the original code it would be 12.8 / 25.1 / 9.1 hours, which is why the working default was "only the selected sets". That restriction is no longer forced by runtime; it is now a question of what the report should show.
- **Work plan.** No overnight runs have to be planned. The complete study (10 fold seeds, both strategies, three settings, single-activity analysis, three logs) fits in 1.4 hours in parallel or 3.6 hours in sequence, so it can be repeated after every change of a design decision.
- **`demo.yaml` / `full.yaml`.** Sizes as proposed in section 7.
- **Implementation.** Set `n_jobs=1` for XGBoost in the pipeline (same results, frees the machine, allows one process per log). Use the engine for everything; keep the original code only as the reference that `test_engine.py` compares against.
- **Reproduction claim.** The paper's pipeline at its defaults takes about 1 hour (f1), 1.5 hours (f2) and 23 minutes (f3) with the original code on this machine. This can be stated in the report as the cost of the baseline implementation.

## 9. Problems, limits and what was not done

- **Shared machine.** Other programs used 11–24 % of the CPU during the runs of the original and up to about a third during some engine runs. The original's times vary by 5–10 % between runs (section 5.3). The engine runs with default threads saturated the machine by themselves.
- **Runs that shared the machine with my own processes.** The single-activity pass of f2 ran at the same time as the passes of f1 and then f3. The time per scored activity (f1 6.19 s) agrees with the itemset routine measured alone (6.44 s for size-1 sets), so the effect is small, but these passes were not measured in isolation. The `n_jobs=1` grids ran three at a time on purpose.
- **The original was not run at full size.** Its numbers for (a), (b) and (c) are extrapolations from 10 (f2: 20) iterations per group on fold 0 and from one single-activity pass on fold 0. Only the training fold was timed, because that is what the original script scores.
- **IMPresseD sets.** The original was timed on the first five sets per length of C4's working selection. If the selection changes, the engine's time changes with the number of cases that contain the sets; the original's time hardly changes.
- **One fold seed.** The engine grid was measured with fold seed 0. The "full" rows of section 7 are 10 times that measurement, not a 10-seed run.
- **Not included in any time:** IMPresseD mining (numbers quoted from C4), plotting, writing result files of the final pipeline, the original's plot and CSV output, the time the original spends printing warnings.
- **Thread test** (`c1_threads.py`) was run on f1 and f3 with the size-2 Apriori sets only. The `n_jobs=1` full grid covers all three logs and all set types and gave identical values.
- **Not measured:** memory use, other computers (the team's Mac), sets longer than 3, the original on the held-out fold, the original's unconstrained mode.
- **Where the original spends its time** was derived from the difference between scored and skipped activities, not from a profiler.
- Tooling: the Windows performance counter (`Get-Counter`) did not work on this system (localised counter names), so the machine load is computed from `GetSystemTimes` inside the scripts. No run crashed and nothing had to be discarded; the first f2 run was kept and pooled with the second.
