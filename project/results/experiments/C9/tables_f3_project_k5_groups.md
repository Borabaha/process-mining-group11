# C9 add-on: project sets on f3

## Setting `fixed_test` (10 fold seeds, 5 folds, 10 repeats)

Weighted F1 of the scored fold: 0.9313 (std between seed means 0.0032).

### Ranking stability inside a group of sets

| group | two seeds: Spearman mean [min, max] | two seeds: Kendall | two seeds: top-3 overlap | two groups of 5 seeds: Spearman | two groups of 5 seeds: Kendall | two groups of 5 seeds: top-3 overlap | sets above zero (mean - 2 s.e. > 0) | (seed, fold, set) not measurable |
|---|---|---|---|---|---|---|---|---|
| all 60 sets | 0.74 [0.56, 0.90] | 0.58 | 0.76 | 0.90 | 0.76 | 0.71 | 29 of 60 | 58 of 3000 |
| apriori, all sizes (30) | 0.83 [0.73, 0.93] | 0.65 | 0.52 | 0.94 | 0.81 | 0.59 | 19 of 30 | 0 of 1500 |
| apriori, size 1 (10) | 0.48 [-0.10, 0.99] | 0.38 | 0.56 | 0.81 | 0.66 | 0.82 | 3 of 10 | 0 of 500 |
| apriori, size 2 (10) | 0.85 [0.68, 0.98] | 0.71 | 0.65 | 0.96 | 0.89 | 0.93 | 7 of 10 | 0 of 500 |
| apriori, size 3 (10) | 0.78 [0.60, 0.96] | 0.61 | 0.59 | 0.90 | 0.80 | 0.71 | 9 of 10 | 0 of 500 |
| impressed, all sizes (30) | 0.63 [0.31, 0.88] | 0.49 | 0.79 | 0.84 | 0.70 | 0.83 | 10 of 30 | 58 of 1500 |
| impressed, size 1 (10) | 0.43 [-0.21, 0.86] | 0.34 | 0.54 | 0.71 | 0.60 | 0.67 | 3 of 10 | 49 of 500 |
| impressed, size 2 (10) | 0.75 [0.50, 0.98] | 0.60 | 1.00 | 0.87 | 0.70 | 1.00 | 4 of 10 | 0 of 500 |
| impressed, size 3 (10) | 0.59 [0.16, 0.98] | 0.48 | 0.61 | 0.86 | 0.76 | 0.64 | 3 of 10 | 9 of 500 |

### Level of the importance per strategy (mean ± std over fold seeds)

| sets | Apriori | IMPresseD | Apriori - IMPresseD | seeds with a positive difference |
|---|---|---|---|---|
| size 1 | 0.0153 ± 0.0012 | 0.0150 ± 0.0006 | 0.0004 ± 0.0007 | 5 of 10 |
| size 2 | 0.0059 ± 0.0021 | 0.0056 ± 0.0020 | 0.0003 ± 0.0014 | 5 of 10 |
| size 3 | 0.0084 ± 0.0023 | -0.0002 ± 0.0008 | 0.0085 ± 0.0022 | 10 of 10 |
| all sizes | 0.0099 ± 0.0017 | 0.0068 ± 0.0009 | 0.0031 ± 0.0011 | 10 of 10 |

## Setting `fixed_train` (10 fold seeds, 5 folds, 10 repeats)

Weighted F1 of the scored fold: 0.9939 (std between seed means 0.0004).

### Ranking stability inside a group of sets

| group | two seeds: Spearman mean [min, max] | two seeds: Kendall | two seeds: top-3 overlap | two groups of 5 seeds: Spearman | two groups of 5 seeds: Kendall | two groups of 5 seeds: top-3 overlap | sets above zero (mean - 2 s.e. > 0) | (seed, fold, set) not measurable |
|---|---|---|---|---|---|---|---|---|
| all 60 sets | 0.99 [0.99, 1.00] | 0.96 | 0.74 | 1.00 | 0.98 | 0.70 | 59 of 60 | 1 of 3000 |
| apriori, all sizes (30) | 0.99 [0.98, 1.00] | 0.94 | 1.00 | 1.00 | 0.97 | 1.00 | 30 of 30 | 0 of 1500 |
| apriori, size 1 (10) | 0.92 [0.75, 1.00] | 0.84 | 1.00 | 0.99 | 0.95 | 1.00 | 10 of 10 | 0 of 500 |
| apriori, size 2 (10) | 0.99 [0.98, 1.00] | 0.95 | 0.81 | 0.99 | 0.97 | 0.95 | 10 of 10 | 0 of 500 |
| apriori, size 3 (10) | 0.91 [0.75, 0.99] | 0.80 | 0.66 | 0.93 | 0.85 | 0.59 | 10 of 10 | 0 of 500 |
| impressed, all sizes (30) | 0.99 [0.99, 1.00] | 0.96 | 0.88 | 1.00 | 0.99 | 1.00 | 29 of 30 | 1 of 1500 |
| impressed, size 1 (10) | 0.99 [0.96, 1.00] | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 | 9 of 10 | 1 of 500 |
| impressed, size 2 (10) | 0.98 [0.95, 1.00] | 0.94 | 1.00 | 0.99 | 0.98 | 1.00 | 10 of 10 | 0 of 500 |
| impressed, size 3 (10) | 1.00 [0.99, 1.00] | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 10 of 10 | 0 of 500 |

### Level of the importance per strategy (mean ± std over fold seeds)

| sets | Apriori | IMPresseD | Apriori - IMPresseD | seeds with a positive difference |
|---|---|---|---|---|
| size 1 | 0.0300 ± 0.0009 | 0.0266 ± 0.0007 | 0.0034 ± 0.0003 | 10 of 10 |
| size 2 | 0.0236 ± 0.0007 | 0.0181 ± 0.0013 | 0.0054 ± 0.0011 | 10 of 10 |
| size 3 | 0.0275 ± 0.0011 | 0.0058 ± 0.0002 | 0.0217 ± 0.0012 | 10 of 10 |
| all sizes | 0.0270 ± 0.0006 | 0.0168 ± 0.0004 | 0.0102 ± 0.0005 | 10 of 10 |
