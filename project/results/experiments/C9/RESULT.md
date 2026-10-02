# C9 — Seed stability: how much do the importance rankings move between fold seeds?

Feeds Guide Part 6: B11 (stability analysis), B14 (repeats / folds / seeds for `full.yaml` and `demo.yaml`), and adds evidence to B3 (which fold is scored).

## 0. Short answer

| Question | Answer (f1, the 10 original Apriori itemsets) |
|---|---|
| Is one fold seed enough on the **held-out** fold? | No. With the design of the brief (5 folds x 5 repeats) two fold seeds agree on the ranking with Spearman 0.50 and Kendall 0.37 on average (20 seeds). The five seeds asked for (0-4) gave 0.71 / 0.55, but three other groups of five seeds gave 0.52, 0.71 and 0.38: five seeds are too few even to measure the stability. |
| Is one fold seed enough on the **training** fold? | Yes. Same design: Spearman 0.95, Kendall 0.86. |
| Do more repeats help (held-out)? | Up to 10, then hardly: 3 / 5 / 10 / 30 repeats give Spearman 0.41 / 0.50 / 0.60 / 0.66. |
| Do more fold seeds help (held-out)? | Yes, this is what helps: two independent studies of 1 / 2 / 3 / 5 / 10 seeds (10 repeats each) agree with Spearman 0.59 / 0.75 / 0.80 / 0.86 / 0.92. |
| Do more folds help? | No. 10 folds instead of 5 did not improve the held-out agreement (0.62 against 0.60-0.68 at 10 repeats) and need twice as many models. |
| Are the paper's 50 values per itemset (5 folds x 10 repeats, one split) enough? | Only when the training fold is scored without accumulation (fixed mode: 0.96 / 0.97 / 0.98 on f1 / f2 / f3). Not for the paper's own procedure on f1 (faithful mode: 0.53; f2 0.96, f3 0.83), and not on the held-out fold of f1 (0.60) and f3 (0.77; f2 0.93). |

**Recommended final configuration:** 10 fold seeds (0-9) x 5 folds x 10 repeats = 500 values per set, fixed mode, both folds scored from the same models, permutation seed = 2023 + fold seed. Measured cost for all three logs with the project's 60 sets per log: 90 minutes of single-thread time, 20 minutes of wall time when run as six processes. Details and reporting rules are in section 5.

Nothing is needed from the user for this step. One point for the supervisor is in section 6.

## 1. What was run

`PY` = `"C:/Users/Bohabara/Desktop/process mining/project/.venv/Scripts/python.exe"`; all commands are run from `project/experiments/`.

### 1.1 Code

| File | Role |
|---|---|
| `experiments/c9_seed_stability.py` | runner: for each fold seed, 5 folds, one XGBoost model per fold, importance of every set with `engine.py` in up to three settings; one CSV per fold seed |
| `experiments/c9_analysis.py` | analysis: per-seed tables, Spearman / Kendall between seeds, repeats, pooling of seeds, noise against signal |
| `experiments/c9_project_sets.py` | add-on analysis for the project's 60 sets per log (groups by strategy and size) |
| `experiments/c9_overview.py` | one table across all studies |

`engine.py` and every file of other experiments are unchanged.

### 1.2 Commands

| Study | Command(s) | Fold seeds | Repeats |
|---|---|---|---|
| **main:** f1, 10 original itemsets, 5 folds | `PY c9_seed_stability.py --dataset f1 --first-seed S --n-seeds 5` for S = 0, 5, 10, 15 (four processes side by side) | 0-19 | fixed 30, faithful 10 |
| f1, 10 folds | `PY c9_seed_stability.py --dataset f1 --folds 10 --first-seed S --n-seeds 5 --repeats 10 --faithful-repeats 0` for S = 0, 5 | 0-9 | fixed 10 |
| f2, f3, original itemsets | `PY c9_seed_stability.py --dataset f2 --first-seed S --n-seeds 5 --repeats 10 --faithful-repeats 10` for S = 0, 5 (same for f3) | 0-9 | fixed 10, faithful 10 |
| project sets, f1 | `PY c9_seed_stability.py --dataset f1 --itemsets project --first-seed S --n-seeds N --repeats 10 --faithful-repeats 0` (seeds 0-19 in six calls) | 0-19 | fixed 10 |
| project sets, f2, f3 | the same with `--dataset f2` / `f3`, S = 0, 5, N = 5 | 0-9 | fixed 10 |
| analysis | `PY c9_analysis.py --dataset f1`; `--dataset f1 --max-seeds 10`; `--dataset f1 --folds 10`; `--dataset f2`; `--dataset f3` | | |
| | `PY c9_project_sets.py --dataset f1` (also `--max-seeds 10`); `--dataset f2`; `--dataset f3` | | |
| | `PY c9_overview.py` | | |

The brief asked for 5 fold seeds x 5 folds x 5 repeats on f1. That design is the "first 5 fold seeds, first 5 repeats" part of the main study and is reported as such (sections 3.1-3.3). The main study was made larger (20 seeds, 30 repeats) because 5 seeds give only 10 seed pairs, which turned out to be too few to measure the stability itself (section 3.4).

### 1.3 Settings

| Item | Value |
|---|---|
| Log, preprocessing | `engine.EventLog` (same as the original `DataManager(path, 2, None, L_max_perc=0.8)`): f1 1130 cases / 164 activities, f2 1130 / 207, f3 1111 / 156 |
| Itemsets, main study | the 10 itemsets the original code selects: `apriori_selector.original_top10(traces, 0.5, 10)` (sizes 2 and 3). Checked equal, in the same order, to the original `frequent_activity_sets(0.5, 10)`. f3 uses support 0.49 (10 itemsets; 0.5 gives only 5, see C7) |
| Itemsets, add-on | the project's working selection, 10 sets per size 1-3: Apriori from C7 (`AprioriSelector(0.45, max_len=3, top_k=10)`), IMPresseD from C4 (`results/experiments/C4/impressed_sets_<log>.json`) |
| Encoding, model | `IndexEncoder` (default column order), `XGBClassifier(n_jobs=1)` with default hyper-parameters as in the original. XGBoost's defaults use all rows and columns, so the model has no random part of its own |
| Folds | `engine.make_folds`: `StratifiedKFold(n_splits=5, shuffle=True, random_state=fold seed)` |
| Settings compared | `fixed_test` = fixed mode, held-out fold scored; `fixed_train` = fixed mode, training fold scored; `faithful_train` = faithful mode on the training fold (what the original code computes; added beyond the brief to answer the "50 values" question for the paper's own procedure) |
| Permutation seed | fixed mode: `2023 + fold seed`, so every fold seed has its own random streams (the engine seeds each (fold number, itemset, repeat) from `random_state`). Faithful mode: constant 2023, as the original |
| Other engine options | defaults: `allowed_from='log'`, `draw='sequential'` |
| Importance | weighted F1 of the scored fold before permuting minus after permuting |

### 1.4 How stability is measured

- **Seed mean:** mean importance of one itemset over the 5 folds and the chosen repeats of one fold seed. With 5 folds x 10 repeats this is the mean of the paper's 50 values.
- **Ranking:** the itemsets ordered by their seed mean, rank 1 = most important.
- **Spearman rho / Kendall tau-b** between the rankings of two fold seeds: 1 = same order, 0 = unrelated. Kendall is the stricter reading: tau = 0.5 means that 75 % of the itemset pairs are in the same order.
- **Top-3 overlap:** share of the three highest itemsets of one ranking that are also among the three highest of the other.
- **Fewer repeats** are studied on subsets of the 30 repeats. This is exact in fixed mode (every repeat has its own random generator; checked in section 2). Numbers marked "all seeds" are averaged over all disjoint blocks of repeats (30 repeats = 10 blocks of 3, 6 blocks of 5, 3 blocks of 10) and all pairs of seeds.
- **Pooled seeds:** two disjoint groups of m fold seeds are drawn at random (300 draws), the seed means are averaged inside each group, and the two group rankings are compared. This is "what would happen if another team repeated our study with other seeds".

## 2. Checks made before trusting the numbers

| Check | Result |
|---|---|
| Fold seed 0 of this run against experiment C2 (`C2/stability_f1.csv`, same fold seed, permutation seed 2023) | 3500 values (1500 fixed held-out, 1500 fixed train, 500 faithful): max abs difference 0.0 |
| A separate run with 5 repeats against repeats 0-4 of the 30-repeat run (f1, seed 0, fold 2, both folds scored) | 100 values, max abs difference 1e-16 (CSV round trip) |
| Itemsets against the original function `frequent_activity_sets(0.5, 10)` on f1 | same 10 sets, same order |
| Completeness | every (setting, seed, fold, itemset, repeat) cell present exactly once; no missing value; on f1 every original itemset occurs in at least 111 scored traces of every fold |
| Four numbers recomputed with plain pandas (`groupby` + `corr`) from the raw CSV files, Spearman / Kendall | identical to the analysis script: held-out, seeds 0-4, 5 repeats 0.71 / 0.55; training, seeds 0-4, 5 repeats 0.92 / 0.80; held-out, 20 seeds, repeats 0-9 0.62 / 0.48 (`pairwise_f1_original_k5.csv`; the tables report the average over all three blocks of 10 repeats, 0.60 / 0.46); faithful, 20 seeds 0.53 / 0.41 |
| The analysis after its last code change | all `tables_*.md` and `overview.md` regenerated and byte-identical to the previous run (fixed random seeds) |

## 3. Results on f1, the 10 original Apriori itemsets

### 3.1 Model quality per fold seed

Weighted F1 of the model on the scored fold before any permutation, mean over the 5 folds of a seed.

| Fold seed | Held-out F1: mean over folds | std over folds | lowest - highest fold | Training F1: mean over folds |
|---|---|---|---|---|
| 0 | 0.9094 | 0.0176 | 0.8853 - 0.9284 | 0.9938 |
| 1 | 0.9023 | 0.0329 | 0.8529 - 0.9334 | 0.9934 |
| 2 | 0.9061 | 0.0273 | 0.8718 - 0.9377 | 0.9954 |
| 3 | 0.8996 | 0.0231 | 0.8756 - 0.9377 | 0.9942 |
| 4 | 0.8952 | 0.0157 | 0.8770 - 0.9204 | 0.9931 |
| seeds 0-4 | 0.9025 (std between seeds 0.0055) | | 0.8529 - 0.9377 | 0.9940 (0.0009) |
| all 20 seeds | 0.9028 (std between seeds 0.0056; seed means 0.8928 - 0.9113) | | 0.8529 - 0.9425 | 0.9936 (0.0009) |

The model quality is stable across seeds (a seed mean moves by about ±0.006); a single fold moves by ±0.02-0.03. The instability below is therefore not caused by bad models in some seeds.

### 3.2 Held-out fold (`fixed_test`), design of the brief: 5 fold seeds x 5 folds x 5 repeats

Mean importance per itemset and fold seed:

| id | itemset | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | std between seeds |
|---|---|---|---|---|---|---|---|---|
| 0 | 370407, ac370000 | 0.0052 | 0.0072 | -0.0013 | 0.0027 | -0.0022 | 0.0023 | 0.0041 |
| 1 | ac370000, ac370419 | 0.0034 | 0.0100 | -0.0008 | -0.0035 | -0.0042 | 0.0010 | 0.0058 |
| 2 | ac370000, ac370443 | 0.0037 | 0.0127 | 0.0046 | 0.0005 | -0.0007 | 0.0042 | 0.0052 |
| 3 | ac370000, ac370419, ac370443 | 0.0043 | 0.0131 | 0.0046 | 0.0025 | -0.0004 | 0.0048 | 0.0050 |
| 4 | ac370419, ac370443 | -0.0040 | 0.0052 | -0.0020 | -0.0029 | -0.0008 | -0.0009 | 0.0036 |
| 5 | ac370000, ac370442 | 0.0088 | 0.0111 | 0.0047 | -0.0011 | 0.0006 | 0.0048 | 0.0052 |
| 6 | 370407, ac370000, ac370419 | 0.0118 | 0.0127 | 0.0050 | 0.0060 | 0.0022 | 0.0075 | 0.0045 |
| 7 | 370407, ac370419 | -0.0022 | 0.0031 | -0.0016 | -0.0039 | -0.0014 | -0.0012 | 0.0026 |
| 8 | ac370442, ac370443 | 0.0019 | 0.0084 | -0.0001 | -0.0032 | -0.0009 | 0.0012 | 0.0044 |
| 9 | ac370000, ac370419, ac370442 | 0.0085 | 0.0132 | 0.0051 | 0.0016 | 0.0017 | 0.0060 | 0.0049 |

Rank per itemset and fold seed (1 = most important):

| id | itemset | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | range | rank of the mean over the 5 seeds |
|---|---|---|---|---|---|---|---|---|
| 0 | 370407, ac370000 | 4 | 8 | 8 | 2 | 9 | 2-9 | 6 |
| 1 | ac370000, ac370419 | 7 | 6 | 7 | 9 | 10 | 6-10 | 8 |
| 2 | ac370000, ac370443 | 6 | 4 | 4 | 5 | 5 | 4-6 | 5 |
| 3 | ac370000, ac370419, ac370443 | 5 | 2 | 5 | 3 | 4 | 2-5 | 4 |
| 4 | ac370419, ac370443 | 10 | 9 | 10 | 7 | 6 | 6-10 | 9 |
| 5 | ac370000, ac370442 | 2 | 5 | 3 | 6 | 3 | 2-6 | 3 |
| 6 | 370407, ac370000, ac370419 | 1 | 3 | 2 | 1 | 1 | 1-3 | 1 |
| 7 | 370407, ac370419 | 9 | 10 | 9 | 10 | 8 | 8-10 | 10 |
| 8 | ac370442, ac370443 | 8 | 7 | 6 | 8 | 7 | 6-8 | 7 |
| 9 | ac370000, ac370419, ac370442 | 3 | 1 | 1 | 4 | 2 | 1-4 | 2 |

Agreement of the rankings of two fold seeds (Spearman above the diagonal, Kendall tau-b below it):

|  | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| **seed 0** |  | 0.70 | 0.82 | 0.76 | 0.66 |
| **seed 1** | 0.56 |  | 0.89 | 0.60 | 0.73 |
| **seed 2** | 0.64 | 0.73 |  | 0.53 | 0.81 |
| **seed 3** | 0.56 | 0.47 | 0.38 |  | 0.58 |
| **seed 4** | 0.51 | 0.51 | 0.60 | 0.51 |  |

Mean over the 10 pairs: Spearman 0.71 (0.53 to 0.89), Kendall 0.55 (0.38 to 0.73), top-3 overlap 0.67.

Reading:

- The level changes with the seed for all itemsets together: seed 1 is high for every itemset (0.003 to 0.013), seeds 3 and 4 are near zero. This common shift does not change a ranking, but it changes every plotted value.
- An itemset mean over the five seeds is -0.001 to 0.008, while one changed prediction moves the weighted F1 of a 226-case held-out fold by about 0.0044 (the most frequent non-zero values are ±0.0043 to ±0.0046, followed by their doubles around ±0.0088). The mean importance of an itemset is therefore worth at most one or two predictions per fold. Over all 30,000 held-out values, 50.9 % are above zero, 5.8 % are exactly zero and 43.3 % are below zero.
- Itemset 6 is first or near first in every seed and itemsets 4 and 7 are last or near last; in between, an itemset can be 2nd in one seed and 9th in another (itemset 0).

### 3.3 Training fold (`fixed_train`), same design

Mean importance per itemset and fold seed:

| id | itemset | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | std between seeds |
|---|---|---|---|---|---|---|---|---|
| 0 | 370407, ac370000 | 0.0405 | 0.0383 | 0.0387 | 0.0398 | 0.0382 | 0.0391 | 0.0010 |
| 1 | ac370000, ac370419 | 0.0361 | 0.0331 | 0.0331 | 0.0348 | 0.0343 | 0.0343 | 0.0013 |
| 2 | ac370000, ac370443 | 0.0340 | 0.0347 | 0.0341 | 0.0348 | 0.0342 | 0.0344 | 0.0003 |
| 3 | ac370000, ac370419, ac370443 | 0.0405 | 0.0386 | 0.0396 | 0.0396 | 0.0380 | 0.0393 | 0.0010 |
| 4 | ac370419, ac370443 | 0.0228 | 0.0221 | 0.0221 | 0.0244 | 0.0231 | 0.0229 | 0.0010 |
| 5 | ac370000, ac370442 | 0.0359 | 0.0336 | 0.0343 | 0.0349 | 0.0326 | 0.0343 | 0.0013 |
| 6 | 370407, ac370000, ac370419 | 0.0427 | 0.0397 | 0.0406 | 0.0408 | 0.0389 | 0.0406 | 0.0014 |
| 7 | 370407, ac370419 | 0.0282 | 0.0269 | 0.0281 | 0.0256 | 0.0258 | 0.0269 | 0.0012 |
| 8 | ac370442, ac370443 | 0.0225 | 0.0222 | 0.0210 | 0.0224 | 0.0212 | 0.0218 | 0.0007 |
| 9 | ac370000, ac370419, ac370442 | 0.0408 | 0.0412 | 0.0374 | 0.0410 | 0.0372 | 0.0395 | 0.0020 |

Rank per itemset and fold seed:

| id | itemset | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | range | rank of the mean over the 5 seeds |
|---|---|---|---|---|---|---|---|---|
| 0 | 370407, ac370000 | 4 | 4 | 3 | 3 | 2 | 2-4 | 4 |
| 1 | ac370000, ac370419 | 5 | 7 | 7 | 6 | 5 | 5-7 | 7 |
| 2 | ac370000, ac370443 | 7 | 5 | 6 | 7 | 6 | 5-7 | 5 |
| 3 | ac370000, ac370419, ac370443 | 3 | 3 | 2 | 4 | 3 | 2-4 | 3 |
| 4 | ac370419, ac370443 | 9 | 10 | 9 | 9 | 9 | 9-10 | 9 |
| 5 | ac370000, ac370442 | 6 | 6 | 5 | 5 | 7 | 5-7 | 6 |
| 6 | 370407, ac370000, ac370419 | 1 | 2 | 1 | 2 | 1 | 1-2 | 1 |
| 7 | 370407, ac370419 | 8 | 8 | 8 | 8 | 8 | 8-8 | 8 |
| 8 | ac370442, ac370443 | 10 | 9 | 10 | 10 | 10 | 9-10 | 10 |
| 9 | ac370000, ac370419, ac370442 | 2 | 1 | 4 | 1 | 4 | 1-4 | 2 |

Agreement of two fold seeds (Spearman above, Kendall below the diagonal):

|  | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| **seed 0** |  | 0.93 | 0.93 | 0.96 | 0.94 |
| **seed 1** | 0.78 |  | 0.90 | 0.94 | 0.87 |
| **seed 2** | 0.82 | 0.78 |  | 0.90 | 0.94 |
| **seed 3** | 0.87 | 0.82 | 0.78 |  | 0.89 |
| **seed 4** | 0.82 | 0.69 | 0.82 | 0.78 |  |

Mean over the 10 pairs: Spearman 0.92 (0.87 to 0.96), Kendall 0.80 (0.69 to 0.87). Every single value is above zero (100 % of 30,000). The ranks move by at most 3 places; the swaps happen inside groups of itemsets whose means differ by less than 0.001 (ids 0, 3, 9 and ids 1, 2, 5).

### 3.4 The five seeds of the brief were a lucky draw

The same design, evaluated on the four disjoint groups of five fold seeds and on all 20 seeds (held-out fold):

| Seeds | 5 repeats: Spearman / Kendall | 10 repeats: Spearman / Kendall |
|---|---|---|
| 0-4 (the brief) | 0.71 / 0.55 | 0.66 / 0.49 |
| 5-9 | 0.52 / 0.38 | 0.65 / 0.48 |
| 10-14 | 0.71 / 0.56 | 0.80 / 0.64 |
| 15-19 | 0.38 / 0.30 | 0.37 / 0.28 |
| all 20 seeds, all repeat blocks | 0.50 / 0.37 | 0.60 / 0.46 |

On the training fold the four groups give 0.92 to 0.94 (5 repeats) and 0.93 to 0.96 (10 repeats).

The stability figure itself is uncertain when it rests on few seeds. A bootstrap over the 20 fold seeds gives a 95 % interval of 0.36 to 0.61 for the held-out Spearman at 5 repeats and 0.45 to 0.72 at 10 repeats. All held-out numbers below should be read with about ±0.1.

### 3.5 Effect of the number of repeats

Agreement of two fold seeds, each with 5 folds and R repeats. First line of each cell: seeds 0-4 with repeats 0..R-1 (10 seed pairs). Second line: all 20 seeds and all disjoint repeat blocks, with the 95 % bootstrap interval of the Spearman mean.

| Repeats R | Held-out: Spearman | Held-out: Kendall | Training: Spearman | Training: Kendall |
|---|---|---|---|---|
| 3 | 0.48 (seeds 0-4); **0.41** [0.29, 0.51] (20 seeds) | 0.36; **0.31** | 0.89; **0.94** [0.93, 0.94] | 0.72; **0.84** |
| 5 | 0.71; **0.50** [0.36, 0.61] | 0.55; **0.37** | 0.92; **0.95** [0.95, 0.95] | 0.80; **0.86** |
| 10 | 0.66; **0.60** [0.45, 0.72] | 0.49; **0.46** | 0.94; **0.96** [0.95, 0.96] | 0.85; **0.88** |
| 30 | 0.74; **0.66** [0.49, 0.80] | 0.56; **0.51** | 0.98; **0.97** [0.96, 0.98] | 0.94; **0.91** |

Single pairs of seeds on the held-out fold range from -0.52 to 0.98 at 10 repeats (570 comparisons): one pair of runs can agree almost perfectly or disagree completely.

How much of the instability is permutation noise? Two runs with the **same** folds and models and different permutations (disjoint repeat blocks of the same fold seed):

| Repeats | Held-out: Spearman / Kendall | Training: Spearman / Kendall |
|---|---|---|
| 3 | 0.55 / 0.41 | 0.95 / 0.86 |
| 5 | 0.67 / 0.52 | 0.96 / 0.89 |
| 10 | 0.80 / 0.65 | 0.97 / 0.91 |

Noise of a seed mean against the differences between the itemsets:

| Scored fold | Repeats | std of a seed mean | the same without the seed's common shift ("ranking noise") | std between the 10 itemset means ("signal") | signal / ranking noise | median gap between neighbouring itemsets |
|---|---|---|---|---|---|---|
| held-out | 3 | 0.00422 | 0.00238 | 0.00219 | 0.92 | 0.00045 |
| held-out | 5 | 0.00401 | 0.00202 | 0.00219 | 1.09 | 0.00045 |
| held-out | 10 | 0.00385 | 0.00170 | 0.00219 | 1.29 | 0.00045 |
| held-out | 30 | 0.00373 | 0.00144 | 0.00219 | 1.52 | 0.00045 |
| training | 3 | 0.00130 | 0.00108 | 0.00704 | 6.52 | 0.00105 |
| training | 5 | 0.00114 | 0.00091 | 0.00704 | 7.76 | 0.00105 |
| training | 10 | 0.00100 | 0.00075 | 0.00704 | 9.43 | 0.00105 |
| training | 30 | 0.00088 | 0.00059 | 0.00704 | 11.85 | 0.00105 |

Other spreads, held-out / training: std between repeats in the same fold 0.0080 / 0.0039; std between the fold means of one seed 0.0073 / 0.0032; std between seed means (30 repeats) 0.0037 / 0.0009.

Reading:

- On the held-out fold the noise of one seed's value is as large as the differences between the itemsets (ratio about 1); on the training fold the differences are 6 to 12 times the noise. This ratio, not the method, is what makes the held-out ranking unstable.
- Repeats remove permutation noise only. From 3 to 10 repeats the held-out agreement rises from 0.41 to 0.60; from 10 to 30 only to 0.66, because the rest of the noise comes from the split (which cases are held out, which model is fitted). Even with identical folds and models, 10 repeats give only 0.80.
- 3 repeats (a typical demo setting) are too few for any held-out interpretation on f1.

### 3.6 Pooling fold seeds

Agreement of two independent studies that each pool m fold seeds (5 folds each; 300 random draws of two disjoint groups from the 20 seeds):

| Seeds per study | Repeats | Values per itemset | Held-out: Spearman / Kendall / top-3 overlap | Training: Spearman / Kendall / top-3 overlap |
|---|---|---|---|---|
| 1 | 5 | 25 | 0.49 / 0.36 / 0.55 | 0.95 / 0.86 / 0.78 |
| 2 | 5 | 50 | 0.69 / 0.53 / 0.63 | 0.96 / 0.88 / 0.78 |
| 3 | 5 | 75 | 0.75 / 0.59 / 0.69 | 0.96 / 0.90 / 0.79 |
| 5 | 5 | 125 | 0.83 / 0.66 / 0.76 | 0.97 / 0.92 / 0.80 |
| 10 | 5 | 250 | 0.89 / 0.77 / 0.88 | 0.98 / 0.93 / 0.85 |
| 1 | 10 | 50 | 0.59 / 0.46 / 0.61 | 0.96 / 0.88 / 0.78 |
| 2 | 10 | 100 | 0.75 / 0.58 / 0.68 | 0.97 / 0.91 / 0.81 |
| 3 | 10 | 150 | 0.80 / 0.64 / 0.75 | 0.97 / 0.92 / 0.82 |
| 5 | 10 | 250 | 0.86 / 0.71 / 0.84 | 0.98 / 0.93 / 0.84 |
| **10** | **10** | **500** | **0.92 / 0.81 / 0.97** | **0.98 / 0.94 / 0.90** |
| 1 | 30 | 150 | 0.64 / 0.50 / 0.65 | 0.97 / 0.91 / 0.80 |
| 5 | 30 | 750 | 0.89 / 0.75 / 0.94 | 0.98 / 0.94 / 0.87 |
| 10 | 30 | 1500 | 0.93 / 0.81 / 1.00 | 0.99 / 0.96 / 0.97 |

Worst single draw on the held-out fold: 0.59 with 5 seeds x 10 repeats, 0.79 with 10 seeds x 10 repeats (with one seed: -0.39).

Reading:

- For the same number of values, seeds are worth more than repeats: 2 seeds x 5 repeats (0.69) beats 1 seed x 10 repeats (0.59); 10 seeds x 5 repeats (0.89) beats 5 seeds x 10 repeats (0.86); 5 seeds x 10 repeats (0.86) nearly equals 5 seeds x 30 repeats (0.89).
- 10 seeds x 10 repeats is the first design in which two independent studies agree with Spearman above 0.9 and name the same top three (overlap 0.97). Going to 30 repeats adds 0.01.
- The low top-3 overlap on the training fold (0.78 to 0.90, despite Spearman 0.95+) is the near-tie between itemsets 0, 3 and 9 for ranks 2 to 4 (means 0.0384, 0.0389, 0.0388): they swap places although nothing else moves.

The best estimate of the "true" order, mean over all 20 seeds and 30 repeats (± standard error over the seed means):

| id | itemset | held-out: mean ± s.e. | rank | training: mean ± s.e. | rank | faithful, training (10 repeats): mean ± s.e. | rank |
|---|---|---|---|---|---|---|---|
| 0 | 370407, ac370000 | 0.00020 ± 0.00084 | 8 | 0.03841 ± 0.00021 | 4 | 0.06189 ± 0.00134 | 7 |
| 1 | ac370000, ac370419 | 0.00047 ± 0.00093 | 6 | 0.03471 ± 0.00019 | 5 | 0.06773 ± 0.00180 | 1 |
| 2 | ac370000, ac370443 | 0.00265 ± 0.00095 | 5 | 0.03345 ± 0.00021 | 7 | 0.06342 ± 0.00164 | 2 |
| 3 | ac370000, ac370419, ac370443 | 0.00424 ± 0.00100 | 2 | 0.03888 ± 0.00019 | 2 | 0.06307 ± 0.00187 | 5 |
| 4 | ac370419, ac370443 | -0.00096 ± 0.00058 | 10 | 0.02292 ± 0.00016 | 9 | 0.06091 ± 0.00169 | 9 |
| 5 | ac370000, ac370442 | 0.00319 ± 0.00093 | 4 | 0.03391 ± 0.00022 | 6 | 0.06055 ± 0.00162 | 10 |
| 6 | 370407, ac370000, ac370419 | 0.00469 ± 0.00095 | 1 | 0.04143 ± 0.00022 | 1 | 0.06330 ± 0.00173 | 4 |
| 7 | 370407, ac370419 | -0.00067 ± 0.00061 | 9 | 0.02608 ± 0.00014 | 8 | 0.06210 ± 0.00162 | 6 |
| 8 | ac370442, ac370443 | 0.00023 ± 0.00060 | 7 | 0.02187 ± 0.00015 | 10 | 0.06151 ± 0.00187 | 8 |
| 9 | ac370000, ac370419, ac370442 | 0.00419 ± 0.00094 | 3 | 0.03884 ± 0.00026 | 3 | 0.06341 ± 0.00171 | 3 |

- Held-out: 5 of the 10 itemsets (ids 2, 3, 5, 6, 9) are clearly above zero (mean minus two standard errors > 0); the other five cannot be told from zero even with 20 seeds. The three triples (ids 3, 6, 9) are the top three.
- The held-out and the training fold agree on the pooled order with Spearman 0.78 (Kendall 0.64) and on the same top three. Faithful mode ranks differently (0.39 against held-out, 0.44 against fixed training fold).
- With one seed and 10 repeats, the pooled winner (itemset 6) is ranked first in 6 of 20 seeds on the held-out fold and in 18 of 20 on the training fold; itemset 3 is first in 7 of 20 held-out seeds.

### 3.7 Number of folds: 5 against 10

Fold seeds 0-9 in both cases (held-out F1 with 10 folds: 0.9084; training F1 0.9926).

| Design | Values per itemset and seed | Held-out: Spearman of two seeds [bootstrap 95 %] | Held-out: two studies of 5 seeds | Training: Spearman of two seeds | Model fits per seed |
|---|---|---|---|---|---|
| 5 folds x 10 repeats | 50 | 0.68 [0.62, 0.75] (20 seeds: 0.60 [0.45, 0.72]) | 0.90 (20 seeds: 0.86) | 0.96 | 5 |
| 10 folds x 5 repeats | 50 | 0.51 [0.42, 0.61] | 0.85 | 0.98 | 10 |
| 10 folds x 10 repeats | 100 | 0.62 [0.52, 0.71] | 0.87 | 0.98 | 10 |

With 10 folds a held-out fold has 113 cases instead of 226. A seed's common level becomes steadier (std of a seed mean 0.0025 instead of 0.0039), but the ranking noise does not improve (0.00151 against 0.00152 for the same seeds at 10 repeats) and neither does the agreement. On the training fold 10 folds are slightly more stable (0.98 against 0.96), partly because two 90 % training folds share more cases. There is no reason to leave the paper's 5 folds.

### 3.8 Are the paper's 50 values per itemset enough?

The paper's design is one split, 5 folds x 10 repeats = 50 values per itemset. Agreement of two such runs that differ in the fold seed (f1: 20 seeds; f2, f3: 10 seeds; Spearman with bootstrap 95 % interval / Kendall):

| Setting | f1 | f2 | f3 |
|---|---|---|---|
| faithful mode, training fold (the paper's own procedure) | **0.53** [0.46, 0.61] / 0.41 | 0.96 [0.95, 0.98] / 0.89 | 0.83 [0.75, 0.89] / 0.68 |
| fixed mode, training fold | 0.96 [0.95, 0.96] / 0.88 | 0.97 [0.95, 0.98] / 0.91 | 0.98 [0.98, 0.99] / 0.94 |
| fixed mode, held-out fold | **0.60** [0.45, 0.72] / 0.46 | 0.93 [0.91, 0.96] / 0.83 | **0.77** [0.71, 0.83] / 0.64 |

Answer:

- **Fixed mode on the training fold: yes.** 50 values rank the 10 itemsets reproducibly on all three logs.
- **The paper's own procedure: not on f1.** Two runs of the original code with different (unseeded) folds agree on the f1 ranking with Spearman 0.53; all ten f1 itemsets lie within 0.061 to 0.068 with a standard error of about 0.002 over 20 seeds. Even 10 pooled seeds give only 0.86 (top-3 overlap 0.58). On f2 the faithful ranking is reproducible (0.96) and on f3 partly (0.83), but on both logs it is unrelated to the fixed-mode ranking of the same training folds (Spearman 0.07 on f2 and -0.12 on f3 between the means over all seeds; `tables_f2_original_k5.md`, `tables_f3_original_k5.md`). C2 found the same and traced the f2 order to the order in which the itemsets are processed. The original code has no fold seed, so every run of it is one such draw.
- **Fixed mode on the held-out fold: not on f1 and f3.** 50 values give 0.60 and 0.77; on f2 they are enough (0.93).

The f2 and f3 numbers agree with C2 (which used one permutation seed for all fold seeds and, on f3, 5 itemsets): held-out 0.90 and 0.75 there, 0.93 and 0.77 here.

## 4. Add-on: other logs and the project's own sets

### 4.1 All logs, original itemsets (`overview.md`)

| Study | Setting | Fold seeds | F1 of the scored fold | Spearman of two seeds: 3 / 5 / 10 / 30 repeats | Kendall: 3 / 5 / 10 / 30 repeats | Spearman, two studies of 5 seeds x 10 repeats | Ranking noise (10 repeats) | Spread between itemsets |
|---|---|---|---|---|---|---|---|---|
| f1, 5 folds | fixed, held-out | 20 | 0.9028 | 0.41 / 0.50 / 0.60 / 0.66 | 0.31 / 0.37 / 0.46 / 0.51 | 0.86 | 0.00170 | 0.00219 |
| f1, 5 folds | fixed, training | 20 | 0.9936 | 0.94 / 0.95 / 0.96 / 0.97 | 0.84 / 0.86 / 0.88 / 0.91 | 0.98 | 0.00075 | 0.00704 |
| f1, 5 folds | faithful, training | 20 | 0.9936 | - / - / 0.53 / - | - / - / 0.41 / - | 0.82 | 0.00135 | 0.00203 |
| f1, 10 folds | fixed, held-out | 10 | 0.9084 | 0.42 / 0.51 / 0.62 / - | 0.31 / 0.38 / 0.46 / - | 0.87 | 0.00151 | 0.00212 |
| f1, 10 folds | fixed, training | 10 | 0.9926 | 0.97 / 0.98 / 0.98 / - | 0.92 / 0.94 / 0.95 / - | 1.00 | 0.00040 | 0.00659 |
| f2, 5 folds | fixed, held-out | 10 | 0.8910 | 0.85 / 0.89 / 0.93 / - | 0.70 / 0.76 / 0.83 / - | 0.98 | 0.00255 | 0.01180 |
| f2, 5 folds | fixed, training | 10 | 0.9939 | 0.94 / 0.96 / 0.97 / - | 0.85 / 0.90 / 0.91 / - | 1.00 | 0.00125 | 0.01562 |
| f2, 5 folds | faithful, training | 10 | 0.9939 | - / - / 0.96 / - | - / - / 0.89 / - | 0.99 | 0.00192 | 0.01375 |
| f3, 5 folds | fixed, held-out | 10 | 0.9313 | 0.70 / 0.72 / 0.77 / - | 0.55 / 0.58 / 0.64 / - | 0.93 | 0.00200 | 0.00581 |
| f3, 5 folds | fixed, training | 10 | 0.9939 | 0.96 / 0.97 / 0.98 / - | 0.89 / 0.91 / 0.94 / - | 0.99 | 0.00107 | 0.01044 |
| f3, 5 folds | faithful, training | 10 | 0.9939 | - / - / 0.83 / - | - / - / 0.68 / - | 0.93 | 0.00223 | 0.01390 |

f1 is the hardest log: its itemsets differ least (spread 0.002 on the held-out fold against 0.012 on f2 and 0.006 on f3) while the noise is similar on all three. A design that is enough for f1 is enough for f2 and f3.

### 4.2 The project's 60 sets per log (10 per size 1-3, Apriori and IMPresseD)

Fixed mode, 5 folds x 10 repeats per fold seed; f1 with 20 fold seeds, f2 and f3 with 10. These are the working selections of C7 and C4, not final ones. A set that occurs in no trace of a held-out fold is treated as "not measurable" (left out of the mean), not as importance 0.

**Held-out fold**, Spearman between the rankings inside a group of sets:

| Group of sets | f1: two seeds | f1: two studies of 5 seeds | f1: two studies of 10 seeds | f1: sets clearly above zero (20 seeds) | f2: two seeds | f2: two studies of 5 seeds | f2: sets above zero (10 seeds) | f3: two seeds | f3: two studies of 5 seeds | f3: sets above zero (10 seeds) |
|---|---|---|---|---|---|---|---|---|---|---|
| all 60 sets | 0.41 | 0.70 | 0.81 | 23 of 60 | 0.97 | 0.99 | 55 of 60 | 0.74 | 0.90 | 29 of 60 |
| Apriori, all sizes (30) | 0.60 | 0.83 | 0.88 | 12 of 30 | 0.94 | 0.98 | 28 of 30 | 0.83 | 0.94 | 19 of 30 |
| Apriori, size 1 (10) | 0.01 | 0.23 | 0.38 | 0 of 10 | 0.71 | 0.88 | 8 of 10 | 0.48 | 0.81 | 3 of 10 |
| Apriori, size 2 (10) | 0.38 | 0.68 | 0.72 | 2 of 10 | 0.91 | 0.98 | 10 of 10 | 0.85 | 0.96 | 7 of 10 |
| Apriori, size 3 (10) | 0.41 | 0.71 | 0.80 | 10 of 10 | 0.82 | 0.89 | 10 of 10 | 0.78 | 0.90 | 9 of 10 |
| IMPresseD, all sizes (30) | 0.24 | 0.59 | 0.74 | 11 of 30 | 0.98 | 0.99 | 27 of 30 | 0.63 | 0.84 | 10 of 30 |
| IMPresseD, size 1 (10) | 0.20 | 0.47 | 0.65 | 2 of 10 | 0.86 | 0.96 | 7 of 10 | 0.43 | 0.71 | 3 of 10 |
| IMPresseD, size 2 (10) | 0.27 | 0.71 | 0.83 | 5 of 10 | 0.96 | 0.99 | 10 of 10 | 0.75 | 0.87 | 4 of 10 |
| IMPresseD, size 3 (10) | 0.19 | 0.66 | 0.80 | 4 of 10 | 0.99 | 1.00 | 10 of 10 | 0.59 | 0.86 | 3 of 10 |

Not measurable (seed, fold, set) cells on the held-out fold, all for IMPresseD sets: f1 87 of 3000, f2 49 of 1500, f3 58 of 1500.

**Training fold**, same groups: two seeds agree with Spearman 0.88 to 1.00 in every group of every log (lowest: Apriori size 3 on f1, 0.88; all 30-set and 60-set groups 0.99 to 1.00). 59 of 60 sets are clearly above zero on every log. Full tables: `tables_<log>_project_k5_groups.md`.

**Level of the importance, Apriori minus IMPresseD** (mean over the 10 sets of a size; mean ± std over the fold seeds; in how many seeds the difference is positive):

| Log | Size | Held-out: difference | Held-out: seeds with Apriori higher | Training: difference | Training: seeds with Apriori higher |
|---|---|---|---|---|---|
| f1 | 1 | -0.0011 ± 0.0016 | 2 of 20 | 0.0025 ± 0.0004 | 20 of 20 |
| f1 | 2 | -0.0007 ± 0.0026 | 6 of 20 | 0.0124 ± 0.0005 | 20 of 20 |
| f1 | 3 | 0.0045 ± 0.0034 | 19 of 20 | 0.0316 ± 0.0007 | 20 of 20 |
| f2 | 1 | -0.0092 ± 0.0016 | 0 of 10 | 0.0036 ± 0.0006 | 10 of 10 |
| f2 | 2 | -0.0493 ± 0.0020 | 0 of 10 | -0.0305 ± 0.0008 | 0 of 10 |
| f2 | 3 | -0.0196 ± 0.0036 | 0 of 10 | 0.0146 ± 0.0015 | 10 of 10 |
| f3 | 1 | 0.0004 ± 0.0007 | 5 of 10 | 0.0034 ± 0.0003 | 10 of 10 |
| f3 | 2 | 0.0003 ± 0.0014 | 5 of 10 | 0.0054 ± 0.0011 | 10 of 10 |
| f3 | 3 | 0.0085 ± 0.0022 | 10 of 10 | 0.0217 ± 0.0012 | 10 of 10 |

Reading:

- **Training fold:** rankings and the Apriori-against-IMPresseD comparison are reproducible with a single fold seed on all three logs.
- **Held-out fold, f2:** stable with one seed for the big groups, and with 5 seeds for every group.
- **Held-out fold, f1 and f3:** the ranking over all 30 sets of a strategy becomes usable with 5 to 10 seeds (f1: 0.74 to 0.88 with 10 seeds; f3: 0.84 to 0.94 with 5). Inside a group of 10 sets of one size it stays weak on f1 even with 10 seeds (0.38 to 0.83). The reason is not too few seeds but too little signal: none of the 10 Apriori single activities and only 2 of the 10 Apriori pairs on f1 have a held-out importance that can be told from zero with 20 seeds. Where there is nothing to rank, more seeds do not produce a ranking.
- **The sign of the comparison can depend on the scored fold.** On f2 sizes 1 and 3, Apriori sets have the higher importance on the training fold and IMPresseD sets on the held-out fold, both in 10 of 10 seeds. This is a stable result, not noise, and has to be reported as such.
- The level comparison is not corrected for how many traces contain a set (frequent sets move more traces). That is a matter of the comparison design, not of this experiment.

## 5. Recommendation for the final runs

| Parameter | `full.yaml` | `demo.yaml` | Why |
|---|---|---|---|
| Fold seeds | **10** (0-9) | 1 (seed 0) | Held-out f1: two independent 10-seed studies agree with Spearman 0.92 / Kendall 0.81 / top-3 overlap 0.97; 5 seeds give 0.86 / 0.71 / 0.84 (worst draw 0.59). 5 seeds is the minimum that should be reported; 10 costs minutes |
| Folds | **5** | 5 | The paper's design; 10 folds gave no better held-out ranking and double the model fits |
| Repeats | **10** | 2 or 3 | The paper's design; below 10 the permutation noise dominates (0.41 at 3), above 10 the gain is 0.01 to 0.06 |
| Values per set | 500 | 10 to 15 | |
| Permutation seed | 2023 + fold seed | 2023 | Independent random streams per fold seed; `engine.py` seeds by fold number, not by fold seed |
| Scored fold | both, from the same models | both | Costs no extra model; the training fold is the stable companion of the held-out fold |
| Mode | fixed (plus one faithful run per log as the reproduction check, Guide B2) | fixed | |

Measured cost of this `full.yaml` design in this experiment (fixed mode, both folds, 60 sets per log, `n_jobs=1`, machine shared with other experiments): f1 26 min, f2 40 min, f3 24 min of single-thread time for 10 seeds, 90 min in total; as six processes side by side the three logs finished in 20 min of wall time. This confirms C1's estimate (about 25 min in parallel).

Reporting rules that follow from the numbers:

1. Report a set's importance as the mean over the 10 seed means with its standard error or interval over the seeds. Never rank from one split.
2. Next to every ranking, give its own stability: Spearman / Kendall between the ranking from seeds 0-4 and the ranking from seeds 5-9 (one line of code with the per-seed table). A stability estimated from 3 seeds is not reliable: groups of five seeds gave between 0.37 and 0.80 for the same quantity.
3. On the held-out fold, first say which sets are clearly above zero, and interpret order only among those and only in coarse terms (top group / bottom group). On f1 do not interpret the order inside a size group.
4. Treat a set that is absent from a held-out fold as missing, not as importance 0.
5. Use `demo.yaml` only to show that the pipeline runs. Its held-out ranking on f1 is noise (Spearman 0.41 between two such runs at 3 repeats).
6. Use the same seeds, folds and repeats for the Apriori and the IMPresseD sets. Then both strategies share the models and the comparison is paired per fold seed.

## 6. What this means for the project decisions

- **B14 (repeats / folds / seeds):** settled by measurement: keep 5 folds and 10 repeats, add fold seeds (10 for the full run). Nothing has to be reduced for runtime reasons.
- **B11 (stability analysis in the report):** "stability over at least 3 fold seeds" should become "10 fold seeds, split-half Spearman / Kendall, and the number of sets clearly above zero". Section 4.2 gives the numbers to expect.
- **B3 (which fold is scored), for the supervisor:** the held-out fold is the more honest test, but on f1 it carries almost no signal for sets of size 1 and 2, with any affordable number of seeds. The numbers support reporting both folds: training fold for the ranking, held-out fold for "does the effect survive on unseen cases". Whether that is acceptable is the supervisor's call; C9 only shows that a held-out-only report would have little to say on f1.
- **Reproduction part:** the paper's f1 ranking (Fig. 5, f1 panel) should not be expected to reproduce set by set. Two runs of the original procedure with different folds agree with Spearman 0.53. Compare levels and the overall picture, and state this number in the report.

## 7. Limits, corrections and things not done

- Fold seeds re-split the same 1130 cases, so they are not independent samples of new data. The bootstrap intervals and standard errors treat them as replicates and are approximate. Nothing here says how the rankings would change on a new log.
- The 10 original f1 itemsets are all subsets of five activities that mostly occur together, so their importances are close and correlated. Low rank agreement partly reflects that there is little to rank.
- Only f1 has 20 fold seeds and 30 repeats. f2 and f3 have 10 seeds and 10 repeats (no 30-repeat column, and "two studies of 5 seeds" uses complementary halves of the same 10 seeds). The 10-fold study has 10 seeds and 10 repeats. 3 folds were not tried.
- The project sets are the working selections of C7 and C4. If the selection changes (support, IMPresseD settings), section 4.2 has to be re-run: `c9_seed_stability.py --itemsets project` reads the sets through `c1_common.load_itemsets`.
- Only the engine defaults were run (`allowed_from='log'`, `draw='sequential'`). C2 showed that the draw changes levels and, on f2, the order; the stability under `draw='uniform'` was not measured.
- In faithful mode "fewer repeats" cannot be studied on subsets, because every repeat depends on the ones before it; only the original's 10 repeats are reported. Faithful mode keeps the constant permutation seed 2023, as the original.
- "Clearly above zero" (mean minus two standard errors) is a rough screen without correction for multiple sets.
- Top-3 overlap is fragile when ranks 3 and 4 are nearly tied (section 3.6).
- Two analysis mistakes were found and corrected during the experiment; the tables above are from the corrected script. (1) The first version of the pooled-seed table always used repeats 0-4, which happened to agree better than the other blocks (0.60 instead of 0.50 for one seed); it now draws the repeat block at random. (2) The first version also applied the repeat subsets to faithful mode, which is not valid.
- Timings were taken on a machine shared with other experiments, with 4 to 6 processes of this experiment side by side. Per fold seed on f1, original itemsets: fits 10 s, fixed held-out 27 s (1500 iterations), fixed training 87 s (1500), faithful 41 s (500).
- No figures were produced; all results are tables.
- `pytest`, `pyflakes` and `pycodestyle` are not installed in the venv. The scripts were checked by running them, by the cross-checks of section 2 and by a line-length / unused-import check.

## 8. Files

Code (`project/experiments/`): `c9_seed_stability.py`, `c9_analysis.py`, `c9_project_sets.py`, `c9_overview.py`.

Results (`project/results/experiments/C9/`):

| File | Content |
|---|---|
| `RESULT.md` | this report |
| `tables_f1_original_k5.md` | all tables of the main study (20 seeds), including faithful mode |
| `tables_f1_original_k5_first10.md`, `tables_f1_original_k10.md` | f1 with fold seeds 0-9 only; f1 with 10 folds |
| `tables_f2_original_k5.md`, `tables_f3_original_k5.md` | the same analysis for f2 and f3 |
| `tables_<log>_project_k5_groups.md` (f1, f2, f3; f1 also `_first10`) | project sets, groups by strategy and size |
| `overview.md` | one table across the studies |
| `summary_*.json` | every number of the tables in machine-readable form |
| `per_seed_<study>.csv` | mean importance and rank per setting, repeats (3, 5, 10, 30), fold seed and itemset |
| `pairwise_<study>.csv` | Spearman, Kendall and top-3 overlap for every pair of fold seeds |
| `raw/<log>_<itemsets>_k<folds>/seed_<s>.csv` and `.json` | one row per setting x fold x itemset x repeat (importance, baseline F1, traces with the itemset, traces changed); timings per fold seed |
| `logs/run_*.log` | progress lines of every runner process |
