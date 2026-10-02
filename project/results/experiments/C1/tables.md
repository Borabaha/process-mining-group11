### T1. One-off steps and per-fold costs

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

### T2. Seconds per (itemset, repeat): one fold, 5 sets x 2 repeats per row

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

### T3. Engine seconds per (itemset, repeat) in the measured full grid (10 sets x 10 repeats per call; median over the 5 folds)

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

### T4. (a) Original defaults

| Log | itemsets at min_support 0.5 | iterations | t original (size 2 / size 3) | original: S + 5 F | original: total | original, if 10 itemsets (500 iterations) | engine faithful, training fold | engine fixed, training fold | engine fixed, held-out fold |
|---|---|---|---|---|---|---|---|---|---|
| f1 | 10 (7 of size 2, 3 of size 3) | 500 | 6.50 s / 7.25 s | 17.1 s | 56.3 min | 56.3 min | 29.0 s | 20.4 s | 8.76 s |
| f2 | 10 (8 of size 2, 2 of size 3) | 500 | 10.50 s / 12.74 s | 25.4 s | 91.6 min | 91.6 min | 39.3 s | 29.3 s | 12.7 s |
| f3 | 5 (4 of size 2, 1 of size 3) | 250 | 5.40 s / 5.82 s | 14.6 s | 23.1 min | 46.0 min | 13.9 s | 10.2 s | 5.34 s |

### T5. (b) Full project grid

| Log | iterations | original: total (extrapolated) | original: Apriori half / IMPresseD half | engine faithful, training fold: loop MEASURED | engine fixed, training fold: loop MEASURED | engine fixed, held-out fold: loop MEASURED | engine: S + 5 F (measured) |
|---|---|---|---|---|---|---|---|
| f1 | 3000 | 5.3 h | 2.8 h / 2.5 h | 119.8 s | 79.6 s | 30.0 s | 3.14 s |
| f2 | 3000 | 9.2 h | 4.7 h / 4.5 h | 2.7 min | 108.0 s | 40.2 s | 4.24 s |
| f3 | 3000 | 4.4 h | 2.3 h / 2.1 h | 86.5 s | 58.9 s | 24.2 s | 2.47 s |
| all three (incl. S + 5 F) | 9000 | 18.9 h |  | 6.3 min | 4.3 min | 104.3 s |  |

### T5b. Check of the extrapolation formula on the engine (500 x sum of the six one-fold medians vs the measured loop)

| Log | engine faithful, training fold: extrapolated / measured | engine fixed, training fold: extrapolated / measured | engine fixed, held-out fold: extrapolated / measured |
|---|---|---|---|
| f1 | 2.1 min / 119.8 s = 1.06 | 87.2 s / 79.6 s = 1.10 | 33.5 s / 30.0 s = 1.12 |
| f2 | 2.9 min / 2.7 min = 1.08 | 2.1 min / 108.0 s = 1.18 | 47.3 s / 40.2 s = 1.18 |
| f3 | 114.4 s / 86.5 s = 1.32 | 80.9 s / 58.9 s = 1.37 | 33.3 s / 24.2 s = 1.38 |

### T6. (c) Single-activity analysis

| Log | activities | iterations | original: one pass (1 fold, 1 repeat), measured | original: median per scored activity | original: S + 5 F + 50 passes (extrapolated) | engine faithful, training fold: MEASURED | engine fixed, training fold: MEASURED | engine fixed, held-out fold: MEASURED |
|---|---|---|---|---|---|---|---|---|
| f1 | 164 | 8200 | 15.4 min | 6.19 s | 12.8 h | 77.8 s | 72.2 s | 37.4 s |
| f2 | 207 | 10350 | 30.1 min | 10.0 s | 25.1 h | 101.7 s | 96.8 s | 49.3 s |
| f3 | 156 | 7800 | 10.9 min | 4.61 s | 9.1 h | 55.7 s | 53.4 s | 28.9 s |

### T7. Machine load

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

### T8. Engine wall time of a demo and a full setting

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

### T9. Logs measured twice: the single runs (T2 pools them)

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

### T10. XGBoost threads: fold 0, ten Apriori sets of size 2 x 10 repeats

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
