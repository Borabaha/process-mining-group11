# C8 — how much do the original code's behaviours change the result?

Date: 2 Oct 2026. Machine: Windows 10, 12 logical cores, shared with other experiments (timings are noisy, values are not affected). Interpreter: `project/.venv/Scripts/python.exe` (Python 3.12, numpy 2.5.3, pandas 2.3.3, scipy 1.18.1, scikit-learn 1.9.1, xgboost 3.4.1, mlxtend 0.25.0, matplotlib 3.11.2).

Feeds: the design of the reproduction part (faithful mode) and of the comparison part (fixed mode) of the project; Guide §3.6 rows 1-3 (training-fold scoring, accumulation, bookkeeping bug) and Part 6 Table B "fixed mode / faithful mode".

## 1. Answer in short

**Fixing the behaviours changes the story, not only the scale.** In the original behaviour the importance of an itemset says almost nothing about the itemset; it says where the itemset stood in the processing queue.

1. **Scale.** Mean importance over the ten original Apriori itemsets (10 fold seeds x 5 folds x 10 repeats):

   | Log | A faithful, train | B fixed, train | C fixed, held-out | A / B | B / C |
   |---|---|---|---|---|---|
   | f1 | 0.0632 | 0.0332 | 0.0022 | 1.9 | 15 |
   | f2 | 0.1499 | 0.0809 | 0.0231 | 1.9 | 3.5 |
   | f3 | 0.2496 | 0.0284 | 0.0086 | 8.8 | 3.3 |

2. **Ranking.** The faithful ranking (A) is unrelated to the fixed ranking (B): Spearman 0.27 / 0.12 / -0.31 on f1 / f2 / f3 for fold seed 0 (Kendall 0.20 / 0.11 / -0.20), and 0.30 / 0.01 / -0.14 on average over ten fold seeds. The two fixed settings agree with each other: B vs C Spearman 0.82 / 0.94 / 0.88 (Kendall 0.69 / 0.82 / 0.73).
3. **Order dependence (setting D).** The same code with the itemsets processed in reverse order gives another ranking. On f2 the ranking is nearly inverted (A vs D Spearman -0.83, Kendall -0.69). On f3 the itemset {ac370442, ac370443} has importance 0.256 when processed last (A) and 0.011 when processed first (D), with the same folds, models and seed. On f1 the level stays the same but the ranks are reshuffled (A vs D Spearman 0.16).
4. **What the fixed ranking shows and the faithful one hides.** Without accumulation the itemsets fall into two groups: sets that contain `ac370000` are 1.5 to 2.4 times as important as sets that do not (B, all fold seeds: f1 0.037 vs 0.024, f2 0.090 vs 0.060, f3 0.034 vs 0.014). In A the two groups are equal (0.064 vs 0.062, 0.149 vs 0.151, 0.249 vs 0.251).
5. **Which behaviour does it (supplement, settings E and F).** Accumulation alone explains the difference. The original shuffle without accumulation (E) gives the fixed result (E vs B: level +2 to +10 %, Spearman 0.95-1.00). The fixed shuffle with accumulation (F) gives the faithful result (F vs A: level -3 to +0.5 %, Spearman 0.81-1.00). The bookkeeping bug of `shuffle_sequence` has a small effect on these itemsets.
6. **Paper's Fig. 5.** The faithful f1 boxes lie where the paper's lie at the lower end (0.05-0.06) but the paper's upper rows are higher (median up to about 0.085, ours at most 0.074 in any of ten fold seeds). f3 agrees (medians 0.245-0.264 vs about 0.22-0.265). Only the accumulating settings (A, D, F) reach the published magnitudes; B and C do not. This is a comparison against values read off a figure, not against the authors' numbers.

10 repeats were used throughout (no reduction was needed). A-C were also run on f2 and f3, and D as well.

## 2. What was run

All commands from `C:/Users/Bohabara/Desktop/process mining/project/experiments`:

```
../.venv/Scripts/python.exe c8_run.py --dataset f1 > ../results/experiments/C8/run_f1.log 2>&1
../.venv/Scripts/python.exe c8_run.py --dataset f2 > ../results/experiments/C8/run_f2.log 2>&1
../.venv/Scripts/python.exe c8_run.py --dataset f3 > ../results/experiments/C8/run_f3.log 2>&1
../.venv/Scripts/python.exe c8_supplement.py --dataset f1 > ../results/experiments/C8/run_supplement_f1.log 2>&1
../.venv/Scripts/python.exe c8_supplement.py --dataset f2 > ../results/experiments/C8/run_supplement_f2.log 2>&1
../.venv/Scripts/python.exe c8_supplement.py --dataset f3 > ../results/experiments/C8/run_supplement_f3.log 2>&1
../.venv/Scripts/python.exe c8_surplus_events.py > ../results/experiments/C8/surplus_events.log 2>&1
../.venv/Scripts/python.exe c8_report.py --dataset f1 > ../results/experiments/C8/report_f1.log 2>&1
../.venv/Scripts/python.exe c8_report.py --dataset f2 > ../results/experiments/C8/report_f2.log 2>&1
../.venv/Scripts/python.exe c8_report.py --dataset f3 > ../results/experiments/C8/report_f3.log 2>&1
```

The three `c8_run.py` calls ran side by side, then the three `c8_supplement.py` calls side by side. All exit codes 0.

| Script | Wall time f1 / f2 / f3 | What it does |
|---|---|---|
| `c8_run.py` | 1313 / 1721 / 1047 s | settings A-D, 10 fold seeds x 5 folds x 10 itemsets x 10 repeats = 5000 values per setting and log |
| `c8_supplement.py` | 599 / 746 / 494 s | settings E and F (beyond the brief), same grid |
| `c8_surplus_events.py` | a few seconds | counts complete occurrences and surplus events per itemset (section 6) |
| `c8_report.py` | not timed (seconds per log) | tables, correlations, figures |

Code (prototype, for the students to read, verify and adapt; all in `project/experiments/`):

| File | Content |
|---|---|
| `c8_run.py` | `select_itemsets`, `fit_model`, `compute_setting`, `run_fold_seed`; uses `engine.py` and `apriori_selector.original_top10` |
| `c8_supplement.py` | `supplement_importance` (original or fixed shuffle, with or without accumulation), built from the public functions of `engine.py` |
| `c8_surplus_events.py` | `surplus_table` |
| `c8_report.py` | `per_itemset_table`, `rank_correlations`, `agreement_between_fold_seeds`, `order_dependence`, `membership_effect`, `compare_with_c2`, box plots |

Outputs in `project/results/experiments/C8/` (`<d>` = f1, f2, f3):

| File | Content |
|---|---|
| `values_<d>.csv` | settings A-D, one row per setting x fold seed x fold x itemset x repeat (20,000 rows) |
| `values_E_<d>.csv`, `values_F_<d>.csv` | settings E and F (5000 rows each) |
| `models_<d>.csv` | training and held-out weighted F1 of the 50 models |
| `run_<d>.json` | settings, itemsets, timings |
| `tables_<d>.md` | every table of this report for all six settings, complete |
| `per_itemset_<d>.csv` | mean, std, quartiles, whiskers, min, max, rank per setting and itemset |
| `rank_correlations_<d>.csv` | Spearman and Kendall for every pair of settings |
| `order_dependence_<d>.csv` | A against D per itemset |
| `box_ranges_by_fold_seed_<d>.csv`, `membership_<d>.csv`, `summary_<d>.json` | further tables |
| `surplus_events.csv` | section 6 |
| `box_<d>_<setting>.png` | one box plot per setting (six per log) |
| `box_<d>_all_settings.png` | all settings of a log on one axis |

## 3. Settings

| Setting | Value |
|---|---|
| Logs | BPIC11 f1 (1130 cases), f2 (1130), f3 (1111), loaded with `engine.EventLog` (verified equal to the original `DataManager` in C2) |
| Itemsets | `original_top10(traces, min_support, 10)` = the original's selection (size > 1, ten most frequent, original processing order). f1 and f2: `min_support` 0.5. **f3: `min_support` 0.49**, the largest support of the C7 grid that yields ten itemsets (0.5 yields five; every value up to 0.494 yields the same ten, C7 §3.3) |
| Folds | `make_folds(labels, k=5, seed=fold_seed)`; **fold seed 0 is the main result**, fold seeds 1-9 are used to tell a difference between settings from the choice of the split |
| Model | `XGBClassifier(n_jobs=1)` fitted on the training fold; one model per fold, shared by all settings (C1: one thread gives the same model as the default) |
| Repeats | 10 per itemset |
| A | `LocationPermutationImportance(mode='faithful', score_on='train', random_state=2023)`: accumulation, original shuffle, training fold |
| B | `mode='fixed', score_on='train'`, `random_state = 2023 + fold seed`, pools from the whole log, sequential draw |
| C | `mode='fixed', score_on='test'`, otherwise as B |
| D | as A, itemsets passed in reverse order |
| E (supplement) | original shuffle and original scoring, every repeat from the unpermuted traces |
| F (supplement) | fixed shuffle, permutations accumulate in one working copy (scored as the original does) |
| Per-itemset mean and std | over the 50 values of one fold seed (5 folds x 10 repeats pooled, as the original pools them for its box plot); std = sample standard deviation |
| Rank correlation | Spearman and Kendall tau-b (`scipy.stats`) between the ten itemset means of two settings |
| Box plots | horizontal, sorted by mean with the largest at the bottom, whiskers 1.5 IQR, outliers hidden: the conventions of the original `Plotting_results.py` behind Fig. 5; the dot is the mean |

Itemsets in the original's processing order (rank 1 is processed first in A, last in D):

| Rank | f1 @ 0.5 | f2 @ 0.5 | f3 @ 0.49 |
|---|---|---|---|
| 1 | {370407, ac370000} | {ac370000, ac379999} | {370407, ac370000} |
| 2 | {ac370000, ac370419} | {ac370000, ac419100} | {ac370000, ac370419} |
| 3 | {ac370000, ac370443} | {ac370000, ac379999, ac419100} | {370407, ac370000, ac370419} |
| 4 | {ac370000, ac370419, ac370443} | {ac379999, ac419100} | {370407, ac370419} |
| 5 | {ac370419, ac370443} | {370407, ac370000} | {ac370000, ac370443} |
| 6 | {ac370000, ac370442} | {370407, ac370000, ac379999} | {ac370000, ac370419, ac370443} |
| 7 | {370407, ac370000, ac370419} | {370407, ac379999} | {ac370419, ac370443} |
| 8 | {370407, ac370419} | {ac370000, ac370419} | {ac370000, ac370442} |
| 9 | {ac370442, ac370443} | {ac370000, ac370443} | {ac370000, ac370442, ac370443} |
| 10 | {ac370000, ac370419, ac370442} | {ac370419, ac379999} | {ac370442, ac370443} |

## 4. Checks (all passed)

| Check | Result |
|---|---|
| Itemsets equal the original function | `select_itemsets` compared with the original `DataManager.frequent_activity_sets(min_support, 10)` (loaded through `engine_reference.py`) in one inline run: same sets in the same rank order on f1, f2, f3 |
| Same values as C2 | fold seed 0, settings A, B, C against `C2/full_grid_<d>.csv`: max abs difference 0.0 on 1500 (f1), 1500 (f2) and 750 (f3, the five itemsets C2 had) values. So `n_jobs=1` and this script reproduce C2 exactly |
| E equals A where it must | first itemset, repeat 0 of every fold is the only iteration in which A has not accumulated anything: E = A with max abs difference 0.0 on 50 folds per log |
| E reproducible | E was computed twice (an earlier one-setting script and the final `c8_supplement.py`): identical CSV files on all three logs |
| Infeasible traces in fixed mode | 0 in all runs |
| Itemset present in every scored fold | yes: at least 492 / 500 / 420 training traces and 115 / 110 / 99 held-out traces contain each itemset (f1 / f2 / f3) |
| Model quality, fold seed 0 (mean of 5 folds) | training F1 0.994 / 0.995 / 0.995, held-out F1 0.909 / 0.890 / 0.931 |

## 5. Results

### 5.1 f1: mean (std) and rank per itemset, fold seed 0

50 values per cell (5 folds x 10 repeats). Rank 1 = most important within the setting.

| Itemset (original order) | A faithful, train | B fixed, train | C fixed, held-out | D faithful, reversed order |
|---|---|---|---|---|
| {370407, ac370000} | 0.0650 (0.0147) [2] | 0.0395 (0.0043) [4] | 0.0064 (0.0112) [5] | 0.0664 (0.0087) [1] |
| {ac370000, ac370419} | 0.0687 (0.0104) [1] | 0.0365 (0.0054) [5] | 0.0023 (0.0131) [7] | 0.0652 (0.0080) [4] |
| {ac370000, ac370443} | 0.0626 (0.0095) [4] | 0.0340 (0.0056) [7] | 0.0045 (0.0115) [6] | 0.0621 (0.0068) [9] |
| {ac370000, ac370419, ac370443} | 0.0614 (0.0090) [9] | 0.0406 (0.0066) [3] | 0.0067 (0.0134) [4] | 0.0630 (0.0066) [6] |
| {ac370419, ac370443} | 0.0615 (0.0091) [8] | 0.0226 (0.0034) [9] | -0.0019 (0.0085) [10] | 0.0644 (0.0078) [5] |
| {ac370000, ac370442} | 0.0601 (0.0079) [10] | 0.0351 (0.0055) [6] | 0.0088 (0.0106) [2] | 0.0661 (0.0058) [2] |
| {370407, ac370000, ac370419} | 0.0631 (0.0092) [3] | 0.0426 (0.0055) [1] | 0.0111 (0.0149) [1] | 0.0655 (0.0070) [3] |
| {370407, ac370419} | 0.0623 (0.0100) [5] | 0.0279 (0.0040) [8] | -0.0001 (0.0097) [9] | 0.0626 (0.0072) [7] |
| {ac370442, ac370443} | 0.0615 (0.0082) [7] | 0.0219 (0.0040) [10] | 0.0016 (0.0074) [8] | 0.0622 (0.0105) [8] |
| {ac370000, ac370419, ac370442} | 0.0617 (0.0070) [6] | 0.0407 (0.0061) [2] | 0.0086 (0.0123) [3] | 0.0604 (0.0117) [10] |

The same table over all ten fold seeds (500 values per cell):

| Itemset | A | B | C | D |
|---|---|---|---|---|
| {370407, ac370000} | 0.0625 (0.0187) [7] | 0.0387 (0.0052) [4] | 0.0010 (0.0120) [7] | 0.0671 (0.0208) [2] |
| {ac370000, ac370419} | 0.0686 (0.0204) [1] | 0.0349 (0.0050) [5] | 0.0011 (0.0115) [6] | 0.0662 (0.0215) [4] |
| {ac370000, ac370443} | 0.0636 (0.0213) [4] | 0.0335 (0.0055) [7] | 0.0033 (0.0114) [5] | 0.0633 (0.0197) [9] |
| {ac370000, ac370419, ac370443} | 0.0634 (0.0236) [5] | 0.0389 (0.0058) [3] | 0.0046 (0.0123) [2] | 0.0650 (0.0219) [7] |
| {ac370419, ac370443} | 0.0616 (0.0211) [8] | 0.0226 (0.0043) [9] | -0.0012 (0.0094) [10] | 0.0653 (0.0211) [6] |
| {ac370000, ac370442} | 0.0607 (0.0194) [10] | 0.0342 (0.0055) [6] | 0.0040 (0.0117) [3] | 0.0664 (0.0198) [3] |
| {370407, ac370000, ac370419} | 0.0638 (0.0234) [2] | 0.0416 (0.0058) [1] | 0.0056 (0.0126) [1] | 0.0680 (0.0222) [1] |
| {370407, ac370419} | 0.0626 (0.0211) [6] | 0.0261 (0.0042) [8] | -0.0006 (0.0106) [9] | 0.0646 (0.0188) [8] |
| {ac370442, ac370443} | 0.0614 (0.0215) [9] | 0.0218 (0.0042) [10] | 0.0004 (0.0099) [8] | 0.0658 (0.0229) [5] |
| {ac370000, ac370419, ac370442} | 0.0637 (0.0225) [3] | 0.0392 (0.0058) [2] | 0.0039 (0.0117) [4] | 0.0611 (0.0218) [10] |

Reading for f1:

- **A is flat.** All ten means lie between 0.060 and 0.069 (fold seed 0), a spread of 0.009 with a standard deviation of 0.007-0.015 per itemset. The order inside this band is not reproducible: two fold seeds agree on the faithful ranking with Spearman 0.52 on average (section 5.4).
- **B has structure.** The means range from 0.022 to 0.043. The three pairs without `ac370000` ({ac370442, ac370443}, {ac370419, ac370443}, {370407, ac370419}) are at 0.022-0.028, everything with `ac370000` at 0.034-0.043, and the three triples are the top three. Two fold seeds agree with Spearman 0.95.
- **C tells B's story with a lot of noise.** Same bottom three, same top itemset, but the level is 0.005 on average for fold seed 0 (0.002 over all ten fold seeds), only 58 % of the values are above zero (52 % over all fold seeds), and every standard deviation is larger than its mean.
- **D differs from A** although only the order changed: only 0.5 % of the 5000 (fold seed, fold, itemset, repeat) values coincide, and the ranks move by up to 8 places ({ac370000, ac370442}: rank 10 in A, rank 2 in D).

Figures: `box_f1_A_faithful_train.png`, `box_f1_B_fixed_train.png`, `box_f1_C_fixed_test.png`, `box_f1_D_faithful_train_reversed.png` (one per setting, as required) and `box_f1_all_settings.png` (all settings on one axis, itemsets in the original processing order; this is the picture that shows scale and story at once).

Where the boxes lie, fold seed 0 (extremes over the ten itemsets):

| Setting | lowest whisker | lowest Q1 | medians | highest Q3 | highest whisker |
|---|---|---|---|---|---|
| A | 0.038 | 0.052 | 0.058-0.071 | 0.077 | 0.100 |
| B | 0.014 | 0.020 | 0.022-0.042 | 0.047 | 0.057 |
| C | -0.023 | -0.009 | -0.000-0.013 | 0.022 | 0.039 |
| D | 0.035 | 0.052 | 0.060-0.067 | 0.072 | 0.085 |

### 5.2 f2: mean (std) and rank per itemset, fold seed 0

| Itemset (original order) | A faithful, train | B fixed, train | C fixed, held-out | D faithful, reversed order |
|---|---|---|---|---|
| {ac370000, ac379999} | 0.1278 (0.0187) [10] | 0.0894 (0.0081) [4] | 0.0156 (0.0219) [5] | 0.1988 (0.0105) [1] |
| {ac370000, ac419100} | 0.1378 (0.0089) [8] | 0.0797 (0.0074) [7] | 0.0095 (0.0219) [8] | 0.1903 (0.0143) [3] |
| {ac370000, ac379999, ac419100} | 0.1456 (0.0103) [5] | 0.0835 (0.0071) [6] | 0.0148 (0.0192) [6] | 0.1927 (0.0131) [2] |
| {ac379999, ac419100} | 0.1347 (0.0087) [9] | 0.0628 (0.0059) [8] | 0.0132 (0.0135) [7] | 0.1811 (0.0126) [4] |
| {370407, ac370000} | 0.1405 (0.0109) [7] | 0.0916 (0.0120) [3] | 0.0337 (0.0264) [2] | 0.1798 (0.0120) [5] |
| {370407, ac370000, ac379999} | 0.1487 (0.0111) [4] | 0.0932 (0.0114) [1] | 0.0325 (0.0219) [3] | 0.1790 (0.0139) [6] |
| {370407, ac379999} | 0.1450 (0.0085) [6] | 0.0527 (0.0075) [10] | 0.0057 (0.0129) [10] | 0.1753 (0.0116) [7] |
| {ac370000, ac370419} | 0.1627 (0.0145) [3] | 0.0930 (0.0116) [2] | 0.0368 (0.0210) [1] | 0.1751 (0.0151) [8] |
| {ac370000, ac370443} | 0.1751 (0.0134) [1] | 0.0852 (0.0118) [5] | 0.0280 (0.0220) [4] | 0.1522 (0.0221) [9] |
| {ac370419, ac379999} | 0.1725 (0.0123) [2] | 0.0548 (0.0077) [9] | 0.0094 (0.0161) [9] | 0.0582 (0.0091) [10] |

Reading for f2: in A the importance grows with the processing position (Spearman between position and mean 0.87); in D it grows with the reversed position (0.99). The ranks in D are almost exactly the processing order read backwards (1, 3, 2, 4, 5, 6, 7, 8, 9, 10). {ac370419, ac379999} is rank 2 in A (processed last) and rank 10 in D (processed first, 0.058); its fixed value is 0.055 (rank 9 in B). B and C agree on the top three and on the bottom two.

### 5.3 f3 (`min_support` 0.49): mean (std) and rank per itemset, fold seed 0

| Itemset (original order) | A faithful, train | B fixed, train | C fixed, held-out | D faithful, reversed order |
|---|---|---|---|---|
| {370407, ac370000} | 0.2129 (0.0696) [10] | 0.0389 (0.0061) [1] | 0.0119 (0.0130) [4] | 0.2612 (0.0115) [2] |
| {ac370000, ac370419} | 0.2635 (0.0096) [1] | 0.0379 (0.0073) [3] | 0.0125 (0.0133) [3] | 0.2625 (0.0133) [1] |
| {370407, ac370000, ac370419} | 0.2485 (0.0128) [9] | 0.0384 (0.0095) [2] | 0.0130 (0.0154) [1] | 0.2497 (0.0150) [7] |
| {370407, ac370419} | 0.2539 (0.0106) [6] | 0.0184 (0.0039) [8] | 0.0014 (0.0082) [8] | 0.2486 (0.0135) [8] |
| {ac370000, ac370443} | 0.2578 (0.0118) [3] | 0.0304 (0.0042) [7] | 0.0107 (0.0126) [6] | 0.2587 (0.0106) [3] |
| {ac370000, ac370419, ac370443} | 0.2520 (0.0163) [7] | 0.0347 (0.0077) [4] | 0.0129 (0.0158) [2] | 0.2545 (0.0149) [5] |
| {ac370419, ac370443} | 0.2502 (0.0105) [8] | 0.0130 (0.0027) [9] | -0.0024 (0.0076) [9] | 0.2508 (0.0139) [6] |
| {ac370000, ac370442} | 0.2600 (0.0105) [2] | 0.0312 (0.0085) [6] | 0.0118 (0.0116) [5] | 0.2583 (0.0105) [4] |
| {ac370000, ac370442, ac370443} | 0.2547 (0.0138) [5] | 0.0322 (0.0081) [5] | 0.0106 (0.0130) [7] | 0.2075 (0.0703) [9] |
| {ac370442, ac370443} | 0.2563 (0.0090) [4] | 0.0113 (0.0020) [10] | -0.0027 (0.0070) [10] | 0.0110 (0.0026) [10] |

Reading for f3: in A nine itemsets sit on one plateau (0.248-0.264) and the first-processed one is lower (0.213) because its first repeats are not yet on the plateau. The itemset that B ranks first is A's last. {ac370442, ac370443} is 0.256 in A and 0.011 in D and B: its faithful value is 23 times its own effect. In D the second-processed itemset takes the "climbing" role (0.208, std 0.070).

Per-setting figures and tables for f2 and f3: `box_f2_*.png`, `box_f3_*.png`, `tables_f2.md`, `tables_f3.md`.

### 5.4 Rank correlations between the settings

Spearman / Kendall tau-b between the ten itemset means. "fold seed 0" is the main result (p-value of Spearman in brackets; with ten itemsets |Spearman| must exceed about 0.65 for p < 0.05). "ten seeds" = the same correlation computed inside each of the ten fold seeds: mean [minimum, maximum] of Spearman.

| Pair | Log | fold seed 0: Spearman / Kendall | ten seeds: Spearman mean [min, max] | itemset means pooled over ten seeds: Spearman / Kendall |
|---|---|---|---|---|
| A vs B | f1 | 0.27 (p 0.45) / 0.20 | 0.30 [-0.04, 0.67] | 0.62 / 0.51 |
| | f2 | 0.12 (p 0.75) / 0.11 | 0.01 [-0.08, 0.14] | 0.07 / 0.07 |
| | f3 | -0.31 (p 0.39) / -0.20 | -0.14 [-0.31, 0.05] | -0.12 / -0.07 |
| A vs C | f1 | -0.04 (p 0.91) / -0.02 | 0.08 [-0.47, 0.45] | 0.39 / 0.29 |
| | f2 | 0.19 (p 0.60) / 0.11 | 0.17 [0.02, 0.37] | 0.15 / 0.11 |
| | f3 | -0.16 (p 0.65) / -0.02 | 0.09 [-0.16, 0.25] | 0.04 / 0.02 |
| B vs C | f1 | 0.82 (p 0.004) / 0.69 | 0.60 [0.12, 0.87] | 0.79 / 0.60 |
| | f2 | 0.94 (p < 0.001) / 0.82 | 0.89 [0.76, 0.99] | 0.95 / 0.87 |
| | f3 | 0.88 (p 0.001) / 0.73 | 0.83 [0.69, 0.95] | 0.96 / 0.91 |
| A vs D | f1 | 0.16 (p 0.65) / 0.07 | -0.07 [-0.35, 0.22] | -0.13 / -0.20 |
| | f2 | -0.83 (p 0.003) / -0.69 | -0.77 [-0.87, -0.70] | -0.79 / -0.64 |
| | f3 | 0.18 (p 0.63) / 0.07 | 0.25 [0.08, 0.46] | 0.26 / 0.24 |
| D vs B | f1 | 0.22 (p 0.53) / 0.16 | 0.18 [-0.08, 0.64] | 0.20 / 0.11 |
| | f2 | 0.10 (p 0.78) / 0.11 | 0.21 [0.10, 0.41] | 0.15 / 0.20 |
| | f3 | 0.54 (p 0.11) / 0.38 | 0.57 [0.49, 0.67] | 0.48 / 0.33 |
| D vs C | f1 | 0.27 (p 0.45) / 0.11 | -0.01 [-0.32, 0.27] | 0.20 / 0.16 |
| | f2 | -0.02 (p 0.96) / 0.02 | -0.01 [-0.19, 0.16] | 0.01 / 0.07 |
| | f3 | 0.54 (p 0.11) / 0.38 | 0.47 [0.10, 0.66] | 0.46 / 0.33 |

Noise reference: how well do two fold seeds of the SAME setting agree on the ranking? (mean Spearman / mean Kendall over the 45 pairs of fold seeds)

| Log | A | B | C | D |
|---|---|---|---|---|
| f1 | 0.52 / 0.39 | 0.95 / 0.87 | 0.65 / 0.50 | 0.55 / 0.43 |
| f2 | 0.96 / 0.89 | 0.97 / 0.91 | 0.94 / 0.83 | 0.97 / 0.90 |
| f3 | 0.83 / 0.68 | 0.98 / 0.94 | 0.77 / 0.64 | 0.97 / 0.91 |

Reading:

- **A vs B is far below the noise reference on every log.** On f2 the faithful ranking is very stable across splits (0.96) and still unrelated to the fixed ranking (0.01): it is stably measuring something else. On f1 the faithful ranking is not even stable against itself (0.52).
- **B vs C is as high as C's own stability allows** (f1 0.60 against 0.65; f2 0.89 against 0.94; f3 0.83 against 0.77). Scoring on the held-out fold changes the scale and adds noise, but not the story.
- **A vs D:** changing only the order of the itemsets gives an inverted ranking on f2 and an unrelated one on f1 and f3. D is in fact closer to B on f3 (0.57) than A is (-0.14), which is an accident of the order: the weakest set of B, {ac370442, ac370443}, happens to be the one D processes first, before anything has accumulated.
- The pooled f1 value of A vs B (0.62) is higher than the per-seed values (0.30) because averaging ten splits removes part of A's noise; it is still lower than what two splits of B share (0.95).
- Three itemsets in common among the top three: A vs B 1 / 1 / 1 (f1 / f2 / f3, fold seed 0); B vs C 2 / 3 / 2.

### 5.5 Order dependence in numbers

The same itemset processed first or last, faithful mode, mean over all ten fold seeds:

| Log | Itemset | processed first | processed last | fixed, train (B) |
|---|---|---|---|---|
| f1 | {370407, ac370000} | 0.0625 (A) | 0.0671 (D) | 0.0387 |
| f1 | {ac370000, ac370419, ac370442} | 0.0611 (D) | 0.0637 (A) | 0.0392 |
| f2 | {ac370000, ac379999} | 0.1338 (A) | 0.1978 (D) | 0.0913 |
| f2 | {ac370419, ac379999} | 0.0611 (D) | 0.1714 (A) | 0.0575 |
| f3 | {370407, ac370000} | 0.2117 (A) | 0.2610 (D) | 0.0387 |
| f3 | {ac370442, ac370443} | 0.0103 (D) | 0.2524 (A) | 0.0103 |

Mean importance by processing position (1 = processed first), all fold seeds:

| Log | Setting | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | Spearman position vs mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f1 | A | 0.0625 | 0.0686 | 0.0636 | 0.0634 | 0.0616 | 0.0607 | 0.0638 | 0.0626 | 0.0614 | 0.0637 | -0.14 |
| f1 | D | 0.0611 | 0.0658 | 0.0646 | 0.0680 | 0.0664 | 0.0653 | 0.0650 | 0.0633 | 0.0662 | 0.0671 | 0.32 |
| f2 | A | 0.1338 | 0.1406 | 0.1467 | 0.1374 | 0.1414 | 0.1499 | 0.1448 | 0.1607 | 0.1725 | 0.1714 | 0.87 |
| f2 | D | 0.0611 | 0.1525 | 0.1737 | 0.1751 | 0.1802 | 0.1789 | 0.1812 | 0.1926 | 0.1913 | 0.1978 | 0.98 |
| f3 | A | 0.2117 | 0.2610 | 0.2498 | 0.2486 | 0.2580 | 0.2523 | 0.2510 | 0.2584 | 0.2526 | 0.2524 | 0.35 |
| f3 | D | 0.0103 | 0.2050 | 0.2543 | 0.2494 | 0.2536 | 0.2572 | 0.2463 | 0.2489 | 0.2614 | 0.2610 | 0.66 |

Reading: on f2 the faithful importance is a function of the position. On f3 everything after the second processed itemset is on a plateau near 0.25, whatever the itemset. On f1 the plateau (about 0.063) is already reached inside the ten repeats of the first itemset, so the position is not visible as a trend; what is left inside the plateau is mostly noise. (In fixed mode the processing order has no influence at all: C2 verified that a reversed order returns identical values.)

## 6. Which behaviour causes it (supplement, beyond the brief)

A and B differ in two things at once. Settings E and F complete a 2 x 2 design, all on the training fold:

| | permutations accumulate | every repeat from the unpermuted traces |
|---|---|---|
| **original shuffle** (bookkeeping bug, positions not clipped, sequential moves) | A | E |
| **fixed shuffle** | F | B |

Mean importance over everything (10 fold seeds):

| Log | A (original, accumulate) | F (fixed, accumulate) | E (original, fresh) | B (fixed, fresh) |
|---|---|---|---|---|
| f1 | 0.0632 | 0.0621 | 0.0344 | 0.0332 |
| f2 | 0.1499 | 0.1506 | 0.0825 | 0.0809 |
| f3 | 0.2496 | 0.2429 | 0.0311 | 0.0284 |

Spearman / Kendall between the itemset means pooled over the ten fold seeds:

| Pair | what differs | f1 | f2 | f3 |
|---|---|---|---|---|
| E vs B | only the shuffle (no accumulation) | 0.95 / 0.87 | 1.00 / 1.00 | 0.99 / 0.96 |
| A vs F | only the shuffle (with accumulation) | 0.86 / 0.64 | 1.00 / 1.00 | 0.81 / 0.73 |
| A vs E | only the accumulation (original shuffle) | 0.62 / 0.47 | 0.07 / 0.07 | -0.14 / -0.11 |
| F vs B | only the accumulation (fixed shuffle) | 0.42 / 0.33 | 0.07 / 0.07 | -0.25 / -0.24 |

Accumulation over the repeats of the first-processed itemset (mean over 10 fold seeds x 5 folds):

| Log, first itemset | Setting | repeat 0 | 1 | 2 | 3 | 5 | 7 | 9 |
|---|---|---|---|---|---|---|---|---|
| f1 {370407, ac370000} | A | 0.0396 | 0.0509 | 0.0596 | 0.0639 | 0.0675 | 0.0688 | 0.0678 |
| | F | 0.0388 | 0.0477 | 0.0543 | 0.0568 | 0.0619 | 0.0642 | 0.0665 |
| | E | 0.0396 | 0.0391 | 0.0393 | 0.0399 | 0.0407 | 0.0408 | 0.0388 |
| | B | 0.0393 | 0.0390 | 0.0392 | 0.0391 | 0.0372 | 0.0386 | 0.0378 |
| f2 {ac370000, ac379999} | A | 0.0933 | 0.1146 | 0.1268 | 0.1357 | 0.1429 | 0.1448 | 0.1504 |
| | F | 0.0923 | 0.1115 | 0.1241 | 0.1317 | 0.1351 | 0.1399 | 0.1442 |
| | E | 0.0933 | 0.0929 | 0.0937 | 0.0927 | 0.0917 | 0.0922 | 0.0945 |
| | B | 0.0926 | 0.0916 | 0.0907 | 0.0901 | 0.0921 | 0.0907 | 0.0903 |
| f3 {370407, ac370000} | A | 0.0442 | 0.1360 | 0.1979 | 0.2316 | 0.2486 | 0.2522 | 0.2572 |
| | F | 0.0377 | 0.1023 | 0.1579 | 0.1927 | 0.2222 | 0.2293 | 0.2323 |
| | E | 0.0442 | 0.0435 | 0.0423 | 0.0445 | 0.0433 | 0.0437 | 0.0439 |
| | B | 0.0382 | 0.0387 | 0.0385 | 0.0380 | 0.0385 | 0.0393 | 0.0391 |
| f3 {ac370442, ac370443} | D (first itemset there) | 0.0102 | 0.0096 | 0.0103 | 0.0101 | 0.0110 | 0.0106 | 0.0106 |

Reading:

- **Accumulation is the cause**, with either shuffle. Repeat 0 of the first itemset is the only faithful value that measures one permutation of the itemset; it is at the level of the values without accumulation (identical to E, within 0.006 of B). Everything after it is measured on traces that are already scrambled.
- **The defects of the original shuffle matter little here**: without accumulation the original shuffle gives 2 to 10 % higher values than the fixed one and the same ranking.
- **Two ways in which accumulation inflates a value.**
  1. Across itemsets: an itemset is scored on traces that the earlier itemsets left permuted. The frequent itemsets share most of their traces and activities, so a later itemset inherits the damage of the earlier ones (f3: {ac370442, ac370443} 0.010 alone, 0.252 after nine others).
  2. Across repeats of one itemset: one shuffle moves only the complete occurrences of the set (the j-th occurrence of each activity). `ac370000` occurs several times per trace and its partners mostly once, so about two "surplus" `ac370000` events stay in place in one shuffle; in the next repeat the occurrences are found in the already permuted trace and other events are moved. `c8_surplus_events.py`: sets with `ac370000` have surplus events in 92-100 % of their traces (2.1-3.8 per trace), {ac370442, ac370443} in 0.5 % (f1) and 0.7 % (f3) of its traces. This fits the table above: the first itemset climbs when it contains `ac370000` and stays flat for {ac370442, ac370443}. It is an explanation that fits the counts, not a separately proven mechanism.

## 7. Comparison of the faithful f1 result with the paper's Fig. 5

This compares our numbers with values read by eye off a figure: the Guide's description (§3.5) and my own reading of a 1200-dpi render of p. 200. It is not a comparison with the authors' numbers (the paper prints none), and a by-eye reading is good to about ±0.005.

| Location boxes | Guide §3.5 | my reading of the render | A, fold seed 0 | A, range over ten fold seeds |
|---|---|---|---|---|
| f1 medians | - | 0.058-0.085 | 0.058-0.071 | lowest 0.052-0.061, highest 0.059-0.074 |
| f1 boxes (lowest Q1 to highest Q3) | 0.06-0.10 | 0.051-0.102 | 0.052-0.077 | Q1 from 0.047, Q3 up to 0.077 |
| f1 whiskers | - | 0.033-0.152 | 0.038-0.100 | 0.031-0.100 |
| f2 medians | - | 0.178-0.22 | 0.134-0.174 | lowest 0.132-0.141, highest 0.168-0.177 |
| f2 boxes | - | 0.168-0.25 | 0.118-0.184 | Q1 from 0.118, Q3 up to 0.185 |
| f3 medians | - | 0.22-0.265 | 0.245-0.264 | lowest 0.225-0.249, highest 0.261-0.269 |
| f3 boxes | 0.20-0.29 | 0.197-0.285 | 0.202-0.269 | Q1 from 0.179, Q3 up to 0.275 |

Reading:

- **f1:** same level at the lower end, and the same picture of ten overlapping boxes without a clear leader. The paper's upper rows are higher and wider than ours (median up to about 0.085 and a box up to about 0.10; ours stay below 0.074 and 0.077 in all ten fold seeds). So faithful mode reproduces the magnitude approximately, not exactly.
- **f3:** agrees within the reading error.
- **f2:** ours is about 0.04 lower than the paper's. Setting D (reversed order) puts nine of the ten f2 itemsets at 0.15-0.20, inside the paper's range, which shows that under accumulation the level itself depends on which itemsets are processed in which order.
- **The paper's itemsets are not exactly ours.** The truncated row labels of the render show five f1 rows starting with `ac370443` (our f1 list has four sets containing it) and four f2 rows starting with `ac419100` (ours has three). At least one itemset per panel therefore differs from the top ten this environment selects at support 0.5 (C7 found that two f1 slots and one f2 slot are decided by ties). Together with the unseeded folds of the original this is enough to explain differences of the size seen; which of them is responsible cannot be told from the figure.
- **Only the accumulating settings reach the published magnitudes.** B (f1 medians 0.022-0.042, f3 0.011-0.039) and C (f1 around 0.00-0.01) are far below Fig. 5. This supports the reading that the published figure was produced with accumulation, as the published code does.

## 8. What this means for the project decisions

- **Reproduction part.** Keep `mode='faithful', score_on='train'` to reproduce the published magnitudes (f1 and f3 match approximately, f2 is lower). Report it as a reproduction of the code's behaviour, and show setting D next to it: the itemset ranking of the original output is not a property of the itemsets.
- **Comparison part (Apriori vs IMPresseD).** It has to use fixed mode. In faithful mode the two strategies cannot be compared at all: the value of a set depends on which sets were processed before it, and a set evaluated after sets containing `ac370000` inherits their plateau. Any difference between Apriori sets and IMPresseD sets would be a difference in list order and list composition.
- **Training fold or held-out fold (Table B, "scored on the held-out fold").** B and C rank the itemsets alike (Spearman 0.82 / 0.94 / 0.88 for one split), so the choice is about scale and noise, not about the story. C is the stricter measure but on f1 it is at noise level (mean 0.002, 52 % of values above zero, two splits agree with 0.65). Practical consequence, in line with C2: compute both, use several fold seeds for C, and do not interpret C differences on f1 between sets of similar size.
- **Paper-vs-code table (Guide §3.6).** Row 2 (accumulation) is the discrepancy that changes results; row 1 (training-fold scoring) changes the level by a factor of 3 to 15; row 3 (bookkeeping bug) changes the level by 2-10 % and not the ranking, for these itemsets. The report can weight them accordingly.
- **Discussion material.** (1) Under accumulation Fig. 5's "all frequent itemsets are important by location, at a similar level" is produced by the procedure: the fixed values are 1.9 (f1, f2) to 8.8 (f3) times smaller and differ between sets by a factor of about 2 (f1, f2) to 3.5 (f3). (2) In fixed mode the importance of a frequent set is largely the importance of `ac370000` (sets with it: 0.037 / 0.090 / 0.034, without: 0.024 / 0.060 / 0.014) and triples score above pairs (0.040 vs 0.030, 0.092 vs 0.078, 0.035 vs 0.026). The ten Apriori sets are built from five to seven activities (C7), so they mostly measure the same thing; this is a concrete argument for comparing with IMPresseD sets and for reporting set size next to importance.
- **Not settled here:** whether the paper's conclusion "location matters more than existence" survives in fixed mode. The existence importance (classical permutation) was not part of this experiment; the location side of that comparison shrinks by the factors above.

## 9. What failed, limits

- Nothing failed in the final runs. During development a first version of the supplement computed only setting E; it was replaced by `c8_supplement.py` (E and F), and E came out identical in both.
- The comparison with Fig. 5 is by eye, against a figure. The paper's fold split is unseeded and its itemsets differ from ours in at least one set per panel (f1, f2), so an exact match cannot be expected or tested.
- p-values of the rank correlations rest on ten itemsets and are approximate. The ten fold seeds re-split the same log, so they are not independent samples.
- The box plots hide outliers (the convention of Fig. 5). Minimum and maximum of every itemset are in `per_itemset_<d>.csv`.
- Settings E and F are additions beyond the brief. F copies the original's scoring convention (only the traces of the current itemset are replaced in the baseline predictions); it is a diagnostic setting, not a proposed method.
- The explanation of the within-itemset climb by surplus events is supported by counts and by the flat curve of {ac370442, ac370443}, but was not isolated in a dedicated test.
- Fixed mode uses the working defaults of C2 (pools from the whole log, sequential draw). C2 showed that the uniform draw lowers the level by 9-14 % and changes the f2 ranking; C8 did not repeat the comparison with that option.
- Only the original's ten frequent itemsets (sizes 2 and 3) were used. Single activities, rarer sets and IMPresseD sets were not measured here.
- Timings were taken with three logs in parallel on a machine shared with other experiments.
- Nothing is needed from the user for this step.
