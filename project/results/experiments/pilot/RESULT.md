# Pilot — first end-to-end comparison: Apriori sets vs IMPresseD sets

Run on 2 Oct 2026, 17:53–18:17, Windows 10, 12 logical cores shared with other work. Interpreter: `project/.venv/Scripts/python.exe` (Python 3.12, numpy 2.5.3, pandas 2.3.3, scipy 1.18.1, scikit-learn 1.9.1, xgboost 3.4.1, matplotlib 3.11.2, PyYAML 6.0.3).

This is a **pilot with the working defaults** (Guide Part 6, Table B). It shows that the whole pipeline runs and what the comparison looks like; the supervisor may still change the rules.

## 0. Short answer

**Does the importance of activities and their locations change when the sets come from IMPresseD instead of Apriori? Yes, mainly because the two strategies look at different activities.**

1. **The two strategies select almost disjoint sets.** At length 1 they share 3 / 3 / 4 of 10 sets (f1 / f2 / f3). At length 2 they share 0 / 1 / 0, at length 3 none. Apriori's lists of pairs and triples are built from 5–7 activities per log, IMPresseD's from 9–14.
2. **On f2, IMPresseD's sets are clearly more important.** Mean held-out importance over all 30 sets: 0.054 against 0.028 for Apriori, higher in 10 of 10 fold seeds. The reason is one activity, `376400`. Every set that contains it loses 0.11–0.13 weighted F1 when its location is permuted. Apriori never selects it, because it occurs in 30 % of the cases and Apriori needs 50 %.
3. **On f1 and f3, IMPresseD's sets are not more important.** At lengths 1 and 2 the held-out means are equal within the noise. At length 3 Apriori is higher (f1 0.005 vs 0.000, f3 0.008 vs 0.000), because a permutation of an IMPresseD triple changes only 6–7 % of the traces on average and so can change few predictions.
4. **Per affected trace, the location of IMPresseD's sets matters more.** On the training fold the importance divided by the share of changed traces is higher for IMPresseD in all nine log x length cells (for example f2 length 2: 0.29 vs 0.13). Raw importance mixes "how much does location matter" with "how many traces contain the set".
5. **Ranking agreement on shared sets cannot be measured in a useful way.** A shared set has the same importance under both strategies (same models), and there are only 3–4 shared sets. At activity level the two strategies rank the 5–7 common activities with Spearman 0.3–0.7, well below each strategy's own stability (0.80–1.00), but with so few activities no coefficient is significant (p 0.11–0.54).
6. **f1 on the held-out fold stays at noise level**, as C2 and C9 found: only 1 and 2 of Apriori's 10 singles and pairs are clearly above zero. Read f1 from the training-fold companion only.

Nothing is needed from the user. Three points for the supervisor are in section 7.

## 1. What was run

All commands from `C:/Users/Bohabara/Desktop/process mining/project/experiments`, with `PY = ../.venv/Scripts/python.exe`.

```
PY -u run_pilot.py --logs f1 > ../results/experiments/pilot/run_f1.log 2>&1    # 956 s
PY -u run_pilot.py --logs f2 > ../results/experiments/pilot/run_f2.log 2>&1    # 1320 s
PY -u run_pilot.py --logs f3 > ../results/experiments/pilot/run_f3.log 2>&1    # 796 s
PY run_pilot.py --logs f1 f2 f3 --figures-only     # redraw the box plots (5 s)
PY compare.py                                      # tables (2 s)
PY ../results/experiments/pilot/check_against_c9.py
```

f1 was started first. f2 and f3 were started after the first fold seed of f1 had finished, and ran next to it. Wall time for all three logs: 24 minutes.

| File | Role |
|---|---|
| `experiments/compare.py` | measures (Jaccard, overlap@k, best-match similarity, Spearman / Kendall, activity-level aggregation) and the summary tables |
| `experiments/run_pilot.py` | the pipeline: read the sets, seeded folds, one model per fold, importance with `engine.py`, tidy CSV, box plots |
| `experiments/pilot.yaml` | the settings of this run |
| `results/experiments/pilot/check_against_c9.py` | cross-check against the values stored by experiment C9 |

`engine.py`, `apriori_selector.py`, `impressed_chain.py` and both original repositories were not changed.

## 2. Settings

| Setting | Value |
|---|---|
| Logs | BPIC11 f1 (1130 cases), f2 (1130), f3 (1111), loaded by `engine.EventLog` |
| Apriori sets | `results/experiments/C7/apriori_sets_<log>.json`: 10 most frequent sets per size 1, 2, 3 (min support 0.5 / 0.5 / 0.45) |
| IMPresseD sets | `results/experiments/C4/impressed_sets_<log>.json`: 10 sets per length, mined on the whole log (gap 3, objectives IG, coverage, case distance) |
| Sets per log | 60 selected, 57 / 56 / 56 distinct |
| Folds | 5 stratified folds, fold seeds 0–9 (10 splits) |
| Model | `XGBClassifier(n_jobs=1)`, default parameters, **one model per fold, shared by all sets and both strategies** |
| Engine | fixed mode, `allowed_from='log'`, `draw='sequential'`, permutation seed 2023 + fold seed |
| Scored fold | held-out fold (`test`, primary) and training fold (`train`, companion), from the same models |
| Repeats | 10 per set and fold |
| Values per set | 500 per scored fold (10 fold seeds x 5 folds x 10 repeats) |

**Choice of repeats.** The task asked for a number of repeats that lets one log finish in about 30 minutes. C1 showed that the engine needs only about 30 s per log for 5 folds x 10 repeats on the held-out fold, so nothing had to be reduced: repeats stay at the paper's 10. The rest of the budget went into 10 fold seeds and the training-fold companion, which is the design C9 recommends (one split is not enough on the held-out fold). Measured: 16 / 22 / 13 minutes per log.

| Seconds | f1 | f2 | f3 |
|---|---|---|---|
| 50 model fits | 66 | 80 | 58 |
| held-out fold | 219 | 293 | 187 |
| training fold | 671 | 946 | 550 |
| whole run | 956 | 1320 | 796 |

**How a number is computed.** Importance = weighted F1 of the scored fold minus weighted F1 after permuting the set's location. A set's importance is the mean over the 10 fold-seed means. "SD" is the standard deviation of the seed means. A set that occurs in no scored trace of a fold is left out for that fold (column `measurable` in the CSV), not counted as 0.

Model quality (baseline weighted F1, mean over 50 folds): held-out 0.903 / 0.891 / 0.931, training fold 0.994 for all three logs.

## 3. Checks

- **Same values as C9.** C9 computed the same 60 sets per log through another code path. All 57,000 + 56,000 + 56,000 values (both scored folds) agree: largest difference 5e-11, which is the rounding of the CSV files; trace counts are identical (`check_against_c9.json`).
- `compare.py` measures were checked on hand-made toy lists (Jaccard, overlap@k, best match, Spearman 0.8 / Kendall 0.667 on a four-item example, activity aggregation).
- Two table cells were recomputed directly from `importance_f2.csv` (f2, held-out, length 2: 0.0249 and 0.0742) and match.
- No trace was infeasible in any of the 180,000 rows.

## 4. Numbers

Full tables: `tables.md` and one CSV per table (section 6).

### 4.1 How similar are the selected lists?

| log | length | shared sets | Jaccard | overlap@5 | overlap@10 | best-match similarity | activities used: Apriori / IMPresseD / both |
|---|---|---|---|---|---|---|---|
| f1 | 1 | 3 | 0.18 | 2 | 3 | 0.30 | 10 / 10 / 3 |
| f1 | 2 | 0 | 0 | 0 | 0 | 0.10 | 5 / 11 / 1 |
| f1 | 3 | 0 | 0 | 0 | 0 | 0.11 | 5 / 14 / 2 |
| f2 | 1 | 3 | 0.18 | 2 | 3 | 0.30 | 10 / 10 / 3 |
| f2 | 2 | 1 | 0.05 | 0 | 1 | 0.32 | 7 / 10 / 3 |
| f2 | 3 | 0 | 0 | 0 | 0 | 0.24 | 7 / 11 / 3 |
| f3 | 1 | 4 | 0.25 | 2 | 4 | 0.40 | 10 / 10 / 4 |
| f3 | 2 | 0 | 0 | 0 | 0 | 0.13 | 6 / 9 / 1 |
| f3 | 3 | 0 | 0 | 0 | 0 | 0.18 | 5 / 11 / 2 |

- Jaccard counts a set as one item, so {a, b} and {a, c} do not overlap. "Best-match similarity" is softer: for every set, the largest share of common activities with any set of the other list, averaged (1 = identical lists).
- Shared sets: f1 `ac370000`, `ac419100`, `370407`; f2 `ac419100`, `ac370000`, `ac379999` and the pair {`ac379999`, `ac419100`}; f3 `ac419100`, `ac370000`, `370488e`, `370488g`.
- Share of all cases that contain a set (median of the 10 sets): Apriori 0.49–0.59 in every cell. IMPresseD 0.26–0.33 at length 1, 0.19–0.32 at length 2, 0.03 / 0.26 / 0.04 at length 3 (f1 / f2 / f3).

### 4.2 Mean importance of each strategy's sets

**Held-out fold.** "I − A" is IMPresseD minus Apriori, paired by fold seed, with a 95 % t-interval over the 10 seeds.

| log | length | Apriori mean (SD) | IMPresseD mean (SD) | I − A [95 % interval] | seeds with I > A | traces changed A / I | importance per changed share A / I |
|---|---|---|---|---|---|---|---|
| f1 | 1 | -0.0001 (0.0024) | 0.0008 (0.0014) | +0.0009 [-0.0006, +0.0024] | 8 of 10 | 55% / 28% | -0.000 / 0.003 |
| f1 | 2 | 0.0009 (0.0033) | 0.0013 (0.0012) | +0.0004 [-0.0019, +0.0028] | 7 of 10 | 56% / 26% | 0.002 / 0.005 |
| f1 | 3 | 0.0051 (0.0039) | 0.0003 (0.0006) | -0.0049 [-0.0075, -0.0022] | 0 of 10 | 55% / 7% | 0.009 / 0.004 |
| f1 | all | 0.0020 (0.0031) | 0.0008 (0.0010) | -0.0012 [-0.0033, +0.0009] | 4 of 10 | 55% / 20% | 0.004 / 0.004 |
| f2 | 1 | 0.0108 (0.0037) | 0.0200 (0.0026) | +0.0092 [+0.0081, +0.0104] | 10 of 10 | 61% / 33% | 0.018 / 0.061 |
| f2 | 2 | 0.0249 (0.0044) | 0.0742 (0.0046) | +0.0493 [+0.0479, +0.0508] | 10 of 10 | 60% / 38% | 0.041 / 0.195 |
| f2 | 3 | 0.0479 (0.0052) | 0.0674 (0.0027) | +0.0196 [+0.0170, +0.0222] | 10 of 10 | 58% / 23% | 0.083 / 0.288 |
| f2 | all | 0.0278 (0.0041) | 0.0539 (0.0032) | +0.0261 [+0.0247, +0.0274] | 10 of 10 | 60% / 31% | 0.047 / 0.172 |
| f3 | 1 | 0.0153 (0.0012) | 0.0150 (0.0006) | -0.0004 [-0.0008, +0.0001] | 5 of 10 | 50% / 29% | 0.031 / 0.052 |
| f3 | 2 | 0.0059 (0.0021) | 0.0056 (0.0020) | -0.0003 [-0.0013, +0.0007] | 5 of 10 | 50% / 24% | 0.012 / 0.023 |
| f3 | 3 | 0.0084 (0.0023) | -0.0002 (0.0008) | -0.0085 [-0.0101, -0.0070] | 0 of 10 | 49% / 6% | 0.017 / -0.002 |
| f3 | all | 0.0099 (0.0017) | 0.0068 (0.0009) | -0.0031 [-0.0039, -0.0023] | 0 of 10 | 50% / 20% | 0.020 / 0.035 |

**Training fold** (stable companion; the model has seen these traces).

| log | length | Apriori mean (SD) | IMPresseD mean (SD) | I − A [95 % interval] | seeds with I > A | traces changed A / I | importance per changed share A / I |
|---|---|---|---|---|---|---|---|
| f1 | 1 | 0.0220 (0.0005) | 0.0196 (0.0005) | -0.0024 [-0.0026, -0.0022] | 0 of 10 | 55% / 28% | 0.040 / 0.069 |
| f1 | 2 | 0.0285 (0.0006) | 0.0160 (0.0004) | -0.0125 [-0.0128, -0.0122] | 0 of 10 | 56% / 26% | 0.051 / 0.063 |
| f1 | 3 | 0.0370 (0.0007) | 0.0053 (0.0001) | -0.0317 [-0.0322, -0.0312] | 0 of 10 | 55% / 7% | 0.067 / 0.077 |
| f1 | all | 0.0292 (0.0005) | 0.0137 (0.0003) | -0.0155 [-0.0158, -0.0152] | 0 of 10 | 55% / 20% | 0.053 / 0.067 |
| f2 | 1 | 0.0473 (0.0008) | 0.0437 (0.0006) | -0.0036 [-0.0040, -0.0032] | 0 of 10 | 61% / 33% | 0.078 / 0.134 |
| f2 | 2 | 0.0791 (0.0011) | 0.1096 (0.0008) | +0.0305 [+0.0299, +0.0311] | 10 of 10 | 60% / 38% | 0.131 / 0.288 |
| f2 | 3 | 0.1089 (0.0016) | 0.0943 (0.0008) | -0.0146 [-0.0157, -0.0135] | 0 of 10 | 58% / 23% | 0.188 / 0.403 |
| f2 | all | 0.0784 (0.0011) | 0.0826 (0.0006) | +0.0041 [+0.0036, +0.0046] | 10 of 10 | 60% / 31% | 0.131 / 0.263 |
| f3 | 1 | 0.0300 (0.0009) | 0.0266 (0.0007) | -0.0034 [-0.0036, -0.0032] | 0 of 10 | 50% / 29% | 0.060 / 0.093 |
| f3 | 2 | 0.0236 (0.0007) | 0.0181 (0.0013) | -0.0054 [-0.0062, -0.0046] | 0 of 10 | 50% / 24% | 0.047 / 0.075 |
| f3 | 3 | 0.0275 (0.0011) | 0.0058 (0.0002) | -0.0217 [-0.0225, -0.0208] | 0 of 10 | 49% / 6% | 0.056 / 0.090 |
| f3 | all | 0.0270 (0.0006) | 0.0168 (0.0004) | -0.0102 [-0.0105, -0.0098] | 0 of 10 | 50% / 20% | 0.055 / 0.085 |

How to read the last two columns: "traces changed" is the share of scored traces that a permutation actually changes (mean over the strategy's sets). "Importance per changed share" divides the mean importance by that share; it is roughly the F1 drop one would see if every trace contained the set.

Sets clearly above zero on the held-out fold (mean larger than 2 standard errors over the seeds), length 1 / 2 / 3, out of 10 each:

| log | Apriori | IMPresseD |
|---|---|---|
| f1 | 1 / 2 / 10 | 3 / 5 / 4 |
| f2 | 8 / 10 / 10 | 7 / 10 / 10 |
| f3 | 3 / 7 / 9 | 3 / 4 / 3 |

On the training fold all Apriori sets and 9 / 10 / 10 IMPresseD sets are above zero in every log.

### 4.3 Shared sets

| log | set | Apriori rank | IMPresseD rank | held-out importance (SE) | training-fold importance | importance rank in Apriori's / IMPresseD's list, training fold |
|---|---|---|---|---|---|---|
| f1 | ac370000 | 1 | 3 | 0.0032 (0.0014) | 0.0682 | 1 / 1 |
| f1 | ac419100 | 2 | 2 | 0.0003 (0.0012) | 0.0411 | 2 / 2 |
| f1 | 370407 | 3 | 9 | -0.0005 (0.0012) | 0.0209 | 3 / 4 |
| f2 | ac419100 | 1 | 4 | 0.0042 (0.0018) | 0.0490 | 3 / 5 |
| f2 | ac370000 | 2 | 3 | 0.0376 (0.0017) | 0.1155 | 1 / 2 |
| f2 | ac379999 | 3 | 8 | 0.0093 (0.0011) | 0.0599 | 2 / 3 |
| f2 | ac379999, ac419100 | 3 | 10 | 0.0115 (0.0012) | 0.0658 | 8 / 10 |
| f3 | ac419100 | 1 | 1 | -0.0009 (0.0011) | 0.0251 | 2 / 2 |
| f3 | ac370000 | 2 | 2 | 0.1507 (0.0015) | 0.1982 | 1 / 1 |
| f3 | 370488e | 9 | 10 | 0.0004 (0.0004) | 0.0089 | 5 / 4 |
| f3 | 370488g | 10 | 9 | 0.0003 (0.0004) | 0.0074 | 9 / 6 |

- A shared set has one importance value: both strategies are scored with the same models and random streams. An "importance ranking agreement on shared sets" is therefore 1 by construction and says nothing.
- What can be compared is the two **selection** rankings on the shared sets: Spearman 0.5 / 0.5 / 0.8 (f1 / f2 / f3) on 3 / 3 / 4 sets, Kendall 0.33 / 0.33 / 0.67. With so few sets these numbers carry no weight (p 0.2–1.0).

### 4.4 Do the rankings inside one strategy's list hold, and what drives them?

Spearman coefficients over the 10 sets of a list. "Stability" compares the importance ranking from fold seeds 0–4 with the one from seeds 5–9.

| log | strategy | length | stability held-out | stability train | selection rank vs importance, held-out | same, train | share of traces changed vs importance, train |
|---|---|---|---|---|---|---|---|
| f1 | apriori | 1 | 0.64 | 1.00 | +0.38 | +0.84 | +0.75 |
| f1 | apriori | 2 | 0.88 | 0.99 | +0.26 | +0.64 | +0.64 |
| f1 | apriori | 3 | 0.66 | 0.93 | +0.08 | +0.31 | +0.24 |
| f1 | impressed | 1 | 0.76 | 1.00 | +0.19 | +0.36 | +0.93 |
| f1 | impressed | 2 | 0.92 | 0.98 | -0.41 | -0.03 | +0.55 |
| f1 | impressed | 3 | 0.61 | 1.00 | -0.04 | +0.35 | +0.41 |
| f2 | apriori | 1 | 0.83 | 1.00 | +0.04 | +0.96 | +0.93 |
| f2 | apriori | 2 | 0.99 | 1.00 | -0.22 | +0.21 | +0.19 |
| f2 | apriori | 3 | 0.90 | 0.99 | -0.47 | -0.36 | -0.32 |
| f2 | impressed | 1 | 0.82 | 1.00 | +0.79 | +0.72 | +0.81 |
| f2 | impressed | 2 | 0.99 | 1.00 | +0.68 | +0.48 | -0.66 |
| f2 | impressed | 3 | 0.99 | 1.00 | +0.83 | +0.78 | +0.48 |
| f3 | apriori | 1 | 0.92 | 0.99 | +0.09 | +0.81 | +0.82 |
| f3 | apriori | 2 | 0.99 | 1.00 | +0.59 | +0.60 | +0.60 |
| f3 | apriori | 3 | 0.84 | 0.89 | +0.43 | +0.62 | +0.60 |
| f3 | impressed | 1 | 0.81 | 1.00 | -0.30 | +0.32 | +0.94 |
| f3 | impressed | 2 | 0.85 | 0.99 | +0.71 | +0.96 | +0.98 |
| f3 | impressed | 3 | 0.90 | 1.00 | -0.13 | +0.39 | +0.96 |

- With 10 fold seeds the training-fold rankings are stable (0.89–1.00). The held-out rankings are usable on f2 (0.82–0.99) and f3 (0.81–0.99) and weak on f1 (0.61–0.92).
- On f2, IMPresseD's own rank predicts the location importance (0.68–0.83 on the held-out fold): the sets it ranks first are the ones whose location matters most. Apriori's rank (support) does not (+0.04, −0.22, −0.47).
- On f1 and f3, IMPresseD's rank does not predict location importance at lengths 1 and 3. There the importance mostly follows how many traces a permutation changes (0.93–0.94 at length 1).

### 4.5 Activity level

An activity's importance = mean (and max) of the importances of the selected sets that contain it, over all three lengths.

Top 5 activities by mean, held-out fold (mean, number of sets):

| log | strategy | top 5 |
|---|---|---|
| f1 | Apriori | ac370000 (0.0045, 11); ac370442 (0.0032, 11); 370407 (0.0030, 11); ac370443 (0.0028, 11); ac370419 (0.0021, 11) |
| f1 | IMPresseD | ac370403 (0.0033, 1); 370401c (0.0026, 1); ac370000 (0.0021, 4); ac415100 (0.0018, 7); ac10307 (0.0009, 4) |
| f2 | Apriori | ac370442 (0.0438, 5); ac370443 (0.0429, 8); ac370419 (0.0408, 9); ac370000 (0.0401, 16); 370407 (0.0356, 6) |
| f2 | IMPresseD | 370715a (0.1210, 1); 370712b (0.1206, 1); 377498a (0.1151, 5); ac372417 (0.1111, 1); 376400 (0.1107, 12) |
| f3 | Apriori | ac370000 (0.0238, 12); 370407 (0.0065, 10); 370712b (0.0055, 2); ac370419 (0.0054, 11); ac370442 (0.0051, 10) |
| f3 | IMPresseD | ac370000 (0.0291, 7); ac370403 (0.0194, 1); 370401c (0.0192, 1); ac370402 (0.0098, 2); ac379999 (0.0008, 1) |

Training fold:

| log | strategy | top 5 |
|---|---|---|
| f1 | Apriori | ac370000 (0.0411, 11); ac419100 (0.0411, 1); 370407 (0.0324, 11); ac370419 (0.0306, 11); ac370442 (0.0304, 11) |
| f1 | IMPresseD | ac370000 (0.0333, 4); ac370403 (0.0315, 1); 370401c (0.0282, 1); 370407 (0.0209, 1); ac419100 (0.0147, 10) |
| f2 | Apriori | ac370000 (0.1019, 16); ac370443 (0.0955, 8); ac370442 (0.0941, 5); ac370419 (0.0932, 9); 370407 (0.0877, 6) |
| f2 | IMPresseD | 370712b (0.1394, 1); 370715a (0.1381, 1); 377498a (0.1326, 5); 376400 (0.1326, 12); ac372417 (0.1312, 1) |
| f3 | Apriori | ac370000 (0.0474, 12); ac419100 (0.0251, 1); 370407 (0.0249, 10); ac370419 (0.0230, 11); ac370442 (0.0212, 10) |
| f3 | IMPresseD | ac370000 (0.0477, 7); 370401c (0.0393, 1); ac370403 (0.0389, 1); ac370402 (0.0233, 2); ac419100 (0.0103, 10) |

Agreement of the two strategies' activity rankings (all lengths together):

| log | scored fold | activities Apriori / IMPresseD / both | activity Jaccard | Spearman (mean) | Kendall (mean) | Spearman (max) | Kendall (max) | common in both top 5 | own stability Apriori / IMPresseD |
|---|---|---|---|---|---|---|---|---|---|
| f1 | held-out | 10 / 21 / 7 | 0.29 | 0.32 | 0.24 | 0.35 | 0.25 | 1 | 0.96 / 0.87 |
| f1 | train | 10 / 21 / 7 | 0.29 | 0.57 | 0.43 | 0.41 | 0.39 | 3 | 0.99 / 0.99 |
| f2 | held-out | 10 / 20 / 6 | 0.25 | 0.71 | 0.47 | 0.71 | 0.60 | 0 | 0.98 / 0.98 |
| f2 | train | 10 / 20 / 6 | 0.25 | 0.31 | 0.33 | 0.43 | 0.33 | 0 | 1.00 / 1.00 |
| f3 | held-out | 10 / 18 / 5 | 0.22 | 0.40 | 0.40 | 0.10 | 0.00 | 1 | 0.92 / 0.80 |
| f3 | train | 10 / 18 / 5 | 0.22 | 0.70 | 0.60 | 0.40 | 0.40 | 2 | 0.99 / 1.00 |

- The correlations use only the 5–7 activities that occur in sets of both strategies; p-values of the Spearman coefficients are 0.11–0.54. "Own stability" is the Spearman between the activity ranking from seeds 0–4 and from seeds 5–9 inside one strategy.
- Per length, only length 1 has enough common activities, and there the values are identical by construction (an activity is a set). Lengths 2 and 3 have 1–3 common activities (`activity_agreement.csv`).
- The same activity can look very different depending on the sets around it (`shared_activities.csv`). Example f1, training fold: `ac370419` has 0.031 under Apriori (11 sets) and 0.0003 under IMPresseD, where it occurs only in a triple contained in 5 cases. Example f2, held-out: `370712b` has 0.014 under Apriori (as a single) and 0.121 under IMPresseD (in a pair with `376400`).

### 4.6 Sets that can hardly be measured

All of them are IMPresseD sets. (set, fold) pairs in which no scored trace contains the set, out of 500 per cell:

| log | held-out: length 1 / 2 / 3 | training fold: length 1 / 2 / 3 |
|---|---|---|
| f1 | 27 / 0 / 17 | 1 / 0 / 0 |
| f2 | 49 / 0 / 0 | 1 / 0 / 0 |
| f3 | 49 / 0 / 9 | 1 / 0 / 0 |

Sets whose held-out value rests on fewer than 3 changed traces per fold on average: f1 `337419c`, {`376400`, `ac370419`, `ac378452`}, {`ac410500`, `ac411100`, `ac415100`}; f2 `ac337441`, `ac378431`, `378490e`; f3 `ac388170`, `387070a`, {`ac410500`, `ac411100`, `ac415100`}. That is 9 of the 90 IMPresseD sets (column `few_traces` in `set_importance.csv`). Their importance is 0.000–0.002.

## 5. What the pilot suggests for the project question

In plain language.

- **The answer depends on the log.** On f2 the picture changes strongly, on f1 and f3 it changes little in level but the lists show different activities.
- **f2: IMPresseD finds the activity the label is about.** The f2 label is the rule "every CEA test is eventually followed by a squamous-cell-carcinoma test". The scout report (`scripts/scout/labels_report.md`, section 7) identified `376400` as the likely CEA code: it occurs in 92 % of the label-0 cases and 13 % of the label-1 cases. IMPresseD ranks it first (highest information gain) and builds 12 of its 30 sets around it. Apriori cannot select it at a support of 0.5. Moving `376400` in a trace changes whether a later test "follows" it, so its location importance is large (0.12 on the held-out fold; the largest Apriori value is 0.061). This is the clearest result of the pilot and it has an explanation.
- **f1 and f3: outcome-oriented does not mean location-important.** `376400` is also IMPresseD's first choice on f1, and its location importance there is zero (held-out −0.0002, training fold 0.006). The f1 label says whether a tumor-marker test happens later, so the presence of an activity can separate the classes while its position does not matter. This is the difference between existence importance and location importance that the 2024 paper is about, and IMPresseD's information gain is an existence measure (counts per case).
- **Frequency is a confound.** Apriori's sets occur in half of the traces, so a permutation changes about half of the scored fold. IMPresseD's sets change 6–38 %. Raw importance favours frequent sets. Per changed trace IMPresseD's sets are higher on the training fold in every cell. The report should show both views, or state clearly which one it uses.
- **Length 3 with IMPresseD on f1 and f3 is close to empty.** The triples occur in a median of 3–4 % of the cases; their importance is near zero whatever their location effect is.
- **`ac370000` is the one activity both strategies agree on.** It is the most important single activity in f1 and f3 under both, and second or first in f2. Apriori builds most of its pairs and triples around it (11–16 sets), so Apriori's lists are largely "ac370000 plus a frequent companion".
- **What can be claimed.** "The top activities differ between the strategies" is supported in all three logs (0–1 common activities in the held-out top 5, 0–3 on the training fold). "IMPresseD sets have higher location importance" is supported on f2 only. A rank correlation between the strategies is not a useful headline number, because fewer than 8 activities and at most 4 sets are common.

## 6. Output files (all in `results/experiments/pilot/`)

| File | Content |
|---|---|
| `importance_f1.csv`, `importance_f2.csv`, `importance_f3.csv` | tidy values, 60,000 rows each: log, strategy, length, rank, itemset, selected_by_both, fold_seed, score_on, n_scored, fold, repeat, baseline, permuted, importance, trace counts, measurable |
| `sets_f1.csv`, `sets_f2.csv`, `sets_f3.csv` | the selected sets with rank, support and `selected_by_both` |
| `box_<log>_len<1,2,3>_test.png` | 9 figures, held-out fold: Apriori left, IMPresseD right, same scale |
| `box_<log>_len<1,2,3>_train.png` | 9 figures, training fold |
| `tables.md` | all tables below in one file |
| `selection_overlap.csv` | section 4.1 |
| `strategy_importance.csv` | section 4.2 |
| `shared_sets.csv`, `shared_selection_rank_agreement.csv` | section 4.3 |
| `set_ranking_checks.csv` | section 4.4 |
| `activity_importance.csv`, `top_activities.csv`, `shared_activities.csv`, `activity_agreement.csv` | section 4.5 |
| `set_importance.csv` | one row per set and scored fold: mean, SD, SE, seeds above zero, traces changed, `few_traces` |
| `timing_<log>.json`, `run_<log>.log` | settings and seconds per fold seed |
| `check_against_c9.py`, `check_against_c9.json` | cross-check of section 3 |

Figure names in full: `box_f1_len1_test.png`, `box_f1_len2_test.png`, `box_f1_len3_test.png`, `box_f1_len1_train.png`, `box_f1_len2_train.png`, `box_f1_len3_train.png`, and the same twelve names with `f2` and `f3`. In a figure every box holds 50 fold means (10 fold seeds x 5 folds, each the mean of 10 repeats), the diamond is the mean, the top row is rank 1 of the strategy's own selection, and "(both)" marks a set selected by both strategies. The figure to look at first is `box_f2_len2_test.png`.

## 7. Caveats, open points and what was not done

Caveats:

- **Pilot, working defaults.** k = 10, length = number of distinct activities, IMPresseD with gap 3 and three objectives, ranking by non-dominated layers. C5 showed that the gap alone changes 2–6 of the 10 length-2 and length-3 sets. Any change of these rules changes the lists and can change the result.
- **Whole-log mining leaks label information into IMPresseD's selection.** The sets were chosen with the labels of all cases, including the cases that are later held out. The held-out importance of IMPresseD sets is therefore optimistic in principle. For f2 the effect is unlikely to explain the result (`376400` separates the classes in every large subset), but it has not been measured: mining on the training cases only was not run.
- **The 95 % intervals cover only the randomness of the fold split and of the permutation.** The ten fold seeds reuse the same 1130 cases, so the intervals are too narrow as a statement about new data.
- **f1 on the held-out fold is at noise level** (26–61 % of single values above zero; one changed prediction moves the F1 of a fold by about 0.004). Its training-fold values are stable but come from traces the model has memorised (baseline F1 0.994).
- **Activity-level means depend on the mix of sets.** An activity that occurs once, in a strong set, gets a high mean (f2: `370715a`, `370712b` and `ac372417` owe their place to `376400`). The max aggregation has the same issue.
- **Single activities are not comparable with pairs and triples in fixed mode.** For a single activity every occurrence in a trace is moved; for a pair or triple only as many occurrences as the rarest member has. This is a likely reason (not tested) why `ac370000` alone has 0.151 on f3 while the Apriori sets that contain it have 0.010–0.014.
- **Open issues of the IMPresseD selector found by the C4 verifier are still in the lists used here**: rank 9–10 can depend on floating-point noise in 2 of 9 cells, 9 of 90 sets are too rare to measure, and the set-ranking rule differs from the Guide default.
- Position pools come from the whole log (`allowed_from='log'`), as in the original code.

Not done:

- No faithful-mode run. `run_pilot.py` accepts `mode: faithful`; this was only smoke-tested (f3, training fold, 1 fold seed, 1 repeat) and not analysed.
- No sensitivity run (other k, other gap, minimum number of cases for IMPresseD sets, training-fold mining).
- No comparison with existence importance (C10) for the same sets.
- No significance test between strategies beyond the paired interval over fold seeds.

What failed or had to be redone: no run crashed. The first version of the figures clipped the axis label of the right panel; the label was moved to the figure and all 18 figures were redrawn from the CSV files with `--figures-only`.

Points for the supervisor:

1. May the comparison be reported per changed trace next to the raw importance, since Apriori's sets occur in about 50 % of the traces and IMPresseD's in 3–38 %?
2. The two lists share 0–4 sets, so a rank correlation on shared sets is not meaningful. Is a comparison at activity level plus the overlap numbers what is expected?
3. Should IMPresseD sets need a minimum number of cases (for example 10, as in the 2023 user study) before they enter the comparison? Nine of the 90 selected sets cannot be measured on a held-out fold.
