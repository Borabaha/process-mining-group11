# C9 add-on: project sets on f1

## Setting `fixed_test` (10 fold seeds, 5 folds, 10 repeats)

Weighted F1 of the scored fold: 0.9031 (std between seed means 0.0053).

### Ranking stability inside a group of sets

| group | two seeds: Spearman mean [min, max] | two seeds: Kendall | two seeds: top-3 overlap | two groups of 5 seeds: Spearman | two groups of 5 seeds: Kendall | two groups of 5 seeds: top-3 overlap | sets above zero (mean - 2 s.e. > 0) | (seed, fold, set) not measurable |
|---|---|---|---|---|---|---|---|---|
| all 60 sets | 0.41 [-0.19, 0.83] | 0.30 | 0.27 | 0.68 | 0.51 | 0.88 | 25 of 60 | 44 of 3000 |
| apriori, all sizes (30) | 0.63 [0.23, 0.85] | 0.47 | 0.50 | 0.84 | 0.67 | 0.94 | 13 of 30 | 0 of 1500 |
| apriori, size 1 (10) | 0.02 [-0.68, 0.70] | 0.02 | 0.37 | 0.36 | 0.29 | 0.50 | 1 of 10 | 0 of 500 |
| apriori, size 2 (10) | 0.46 [-0.37, 0.95] | 0.34 | 0.59 | 0.75 | 0.61 | 0.67 | 2 of 10 | 0 of 500 |
| apriori, size 3 (10) | 0.53 [0.02, 0.89] | 0.40 | 0.69 | 0.81 | 0.69 | 0.93 | 10 of 10 | 0 of 500 |
| impressed, all sizes (30) | 0.27 [-0.35, 0.76] | 0.20 | 0.26 | 0.63 | 0.47 | 0.29 | 12 of 30 | 44 of 1500 |
| impressed, size 1 (10) | 0.22 [-0.44, 0.80] | 0.19 | 0.44 | 0.42 | 0.33 | 0.65 | 3 of 10 | 27 of 500 |
| impressed, size 2 (10) | 0.32 [-0.36, 0.84] | 0.24 | 0.47 | 0.77 | 0.63 | 0.64 | 5 of 10 | 0 of 500 |
| impressed, size 3 (10) | 0.23 [-0.42, 0.76] | 0.18 | 0.44 | 0.75 | 0.59 | 0.74 | 4 of 10 | 17 of 500 |

### Level of the importance per strategy (mean ± std over fold seeds)

| sets | Apriori | IMPresseD | Apriori - IMPresseD | seeds with a positive difference |
|---|---|---|---|---|
| size 1 | -0.0001 ± 0.0024 | 0.0008 ± 0.0014 | -0.0009 ± 0.0021 | 2 of 10 |
| size 2 | 0.0009 ± 0.0033 | 0.0013 ± 0.0012 | -0.0004 ± 0.0033 | 3 of 10 |
| size 3 | 0.0051 ± 0.0039 | 0.0003 ± 0.0006 | 0.0049 ± 0.0037 | 10 of 10 |
| all sizes | 0.0020 ± 0.0031 | 0.0008 ± 0.0010 | 0.0012 ± 0.0029 | 6 of 10 |

## Setting `fixed_train` (10 fold seeds, 5 folds, 10 repeats)

Weighted F1 of the scored fold: 0.9938 (std between seed means 0.0008).

### Ranking stability inside a group of sets

| group | two seeds: Spearman mean [min, max] | two seeds: Kendall | two seeds: top-3 overlap | two groups of 5 seeds: Spearman | two groups of 5 seeds: Kendall | two groups of 5 seeds: top-3 overlap | sets above zero (mean - 2 s.e. > 0) | (seed, fold, set) not measurable |
|---|---|---|---|---|---|---|---|---|
| all 60 sets | 1.00 [0.99, 1.00] | 0.96 | 0.74 | 1.00 | 0.98 | 0.71 | 59 of 60 | 1 of 3000 |
| apriori, all sizes (30) | 0.99 [0.98, 1.00] | 0.93 | 0.65 | 1.00 | 0.97 | 0.84 | 30 of 30 | 0 of 1500 |
| apriori, size 1 (10) | 0.98 [0.95, 1.00] | 0.94 | 1.00 | 1.00 | 0.99 | 1.00 | 10 of 10 | 0 of 500 |
| apriori, size 2 (10) | 0.96 [0.92, 1.00] | 0.87 | 0.79 | 0.99 | 0.96 | 0.97 | 10 of 10 | 0 of 500 |
| apriori, size 3 (10) | 0.89 [0.71, 0.99] | 0.76 | 0.66 | 0.95 | 0.84 | 0.68 | 10 of 10 | 0 of 500 |
| impressed, all sizes (30) | 0.99 [0.99, 1.00] | 0.96 | 0.79 | 1.00 | 0.98 | 0.70 | 29 of 30 | 1 of 1500 |
| impressed, size 1 (10) | 1.00 [1.00, 1.00] | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 9 of 10 | 1 of 500 |
| impressed, size 2 (10) | 0.97 [0.94, 1.00] | 0.90 | 1.00 | 0.97 | 0.90 | 1.00 | 10 of 10 | 0 of 500 |
| impressed, size 3 (10) | 0.99 [0.99, 1.00] | 0.98 | 1.00 | 0.99 | 0.96 | 1.00 | 10 of 10 | 0 of 500 |

### Level of the importance per strategy (mean ± std over fold seeds)

| sets | Apriori | IMPresseD | Apriori - IMPresseD | seeds with a positive difference |
|---|---|---|---|---|
| size 1 | 0.0220 ± 0.0005 | 0.0196 ± 0.0005 | 0.0024 ± 0.0003 | 10 of 10 |
| size 2 | 0.0285 ± 0.0006 | 0.0160 ± 0.0004 | 0.0125 ± 0.0005 | 10 of 10 |
| size 3 | 0.0370 ± 0.0007 | 0.0053 ± 0.0001 | 0.0317 ± 0.0007 | 10 of 10 |
| all sizes | 0.0292 ± 0.0005 | 0.0137 ± 0.0003 | 0.0155 ± 0.0004 | 10 of 10 |
