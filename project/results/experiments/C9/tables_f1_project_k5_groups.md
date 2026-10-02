# C9 add-on: project sets on f1

## Setting `fixed_test` (20 fold seeds, 5 folds, 10 repeats)

Weighted F1 of the scored fold: 0.9028 (std between seed means 0.0056).

### Ranking stability inside a group of sets

| group | two seeds: Spearman mean [min, max] | two seeds: Kendall | two seeds: top-3 overlap | two groups of 5 seeds: Spearman | two groups of 5 seeds: Kendall | two groups of 5 seeds: top-3 overlap | two groups of 10 seeds: Spearman | two groups of 10 seeds: Kendall | two groups of 10 seeds: top-3 overlap | sets above zero (mean - 2 s.e. > 0) | (seed, fold, set) not measurable |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all 60 sets | 0.41 [-0.46, 0.84] | 0.29 | 0.30 | 0.70 | 0.54 | 0.86 | 0.81 | 0.64 | 0.99 | 23 of 60 | 87 of 6000 |
| apriori, all sizes (30) | 0.60 [-0.16, 0.91] | 0.45 | 0.48 | 0.83 | 0.66 | 0.95 | 0.88 | 0.73 | 1.00 | 12 of 30 | 0 of 3000 |
| apriori, size 1 (10) | 0.01 [-0.81, 0.82] | 0.01 | 0.34 | 0.23 | 0.18 | 0.44 | 0.38 | 0.31 | 0.44 | 0 of 10 | 0 of 1000 |
| apriori, size 2 (10) | 0.38 [-0.76, 0.95] | 0.29 | 0.54 | 0.68 | 0.54 | 0.70 | 0.72 | 0.56 | 0.67 | 2 of 10 | 0 of 1000 |
| apriori, size 3 (10) | 0.41 [-0.48, 0.98] | 0.31 | 0.59 | 0.71 | 0.56 | 0.94 | 0.80 | 0.64 | 1.00 | 10 of 10 | 0 of 1000 |
| impressed, all sizes (30) | 0.24 [-0.48, 0.79] | 0.18 | 0.27 | 0.59 | 0.45 | 0.40 | 0.74 | 0.58 | 0.41 | 11 of 30 | 87 of 3000 |
| impressed, size 1 (10) | 0.20 [-0.62, 0.92] | 0.18 | 0.42 | 0.47 | 0.37 | 0.63 | 0.65 | 0.52 | 0.79 | 2 of 10 | 54 of 1000 |
| impressed, size 2 (10) | 0.27 [-0.78, 0.95] | 0.21 | 0.50 | 0.71 | 0.57 | 0.65 | 0.83 | 0.69 | 0.70 | 5 of 10 | 0 of 1000 |
| impressed, size 3 (10) | 0.19 [-0.67, 0.88] | 0.15 | 0.36 | 0.66 | 0.52 | 0.59 | 0.80 | 0.66 | 0.70 | 4 of 10 | 33 of 1000 |

### Level of the importance per strategy (mean ± std over fold seeds)

| sets | Apriori | IMPresseD | Apriori - IMPresseD | seeds with a positive difference |
|---|---|---|---|---|
| size 1 | -0.0006 ± 0.0021 | 0.0005 ± 0.0018 | -0.0011 ± 0.0016 | 2 of 20 |
| size 2 | 0.0005 ± 0.0031 | 0.0011 ± 0.0014 | -0.0007 ± 0.0026 | 6 of 20 |
| size 3 | 0.0048 ± 0.0037 | 0.0003 ± 0.0008 | 0.0045 ± 0.0034 | 19 of 20 |
| all sizes | 0.0016 ± 0.0029 | 0.0006 ± 0.0012 | 0.0009 ± 0.0024 | 13 of 20 |

## Setting `fixed_train` (20 fold seeds, 5 folds, 10 repeats)

Weighted F1 of the scored fold: 0.9936 (std between seed means 0.0009).

### Ranking stability inside a group of sets

| group | two seeds: Spearman mean [min, max] | two seeds: Kendall | two seeds: top-3 overlap | two groups of 5 seeds: Spearman | two groups of 5 seeds: Kendall | two groups of 5 seeds: top-3 overlap | two groups of 10 seeds: Spearman | two groups of 10 seeds: Kendall | two groups of 10 seeds: top-3 overlap | sets above zero (mean - 2 s.e. > 0) | (seed, fold, set) not measurable |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all 60 sets | 1.00 [0.99, 1.00] | 0.96 | 0.76 | 1.00 | 0.98 | 0.76 | 1.00 | 0.99 | 0.67 | 59 of 60 | 2 of 6000 |
| apriori, all sizes (30) | 0.99 [0.97, 1.00] | 0.93 | 0.63 | 1.00 | 0.97 | 0.68 | 1.00 | 0.98 | 0.76 | 30 of 30 | 0 of 3000 |
| apriori, size 1 (10) | 0.98 [0.94, 1.00] | 0.94 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 10 of 10 | 0 of 1000 |
| apriori, size 2 (10) | 0.96 [0.88, 1.00] | 0.88 | 0.79 | 0.99 | 0.95 | 0.92 | 0.99 | 0.97 | 0.98 | 10 of 10 | 0 of 1000 |
| apriori, size 3 (10) | 0.88 [0.66, 0.99] | 0.75 | 0.68 | 0.94 | 0.85 | 0.78 | 0.96 | 0.88 | 0.75 | 10 of 10 | 0 of 1000 |
| impressed, all sizes (30) | 0.99 [0.98, 1.00] | 0.96 | 0.78 | 1.00 | 0.98 | 0.76 | 1.00 | 0.98 | 0.80 | 29 of 30 | 2 of 3000 |
| impressed, size 1 (10) | 1.00 [1.00, 1.00] | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 9 of 10 | 2 of 1000 |
| impressed, size 2 (10) | 0.97 [0.93, 1.00] | 0.90 | 1.00 | 0.97 | 0.92 | 1.00 | 0.98 | 0.93 | 1.00 | 10 of 10 | 0 of 1000 |
| impressed, size 3 (10) | 0.98 [0.95, 1.00] | 0.95 | 1.00 | 0.99 | 0.97 | 1.00 | 0.99 | 0.96 | 1.00 | 10 of 10 | 0 of 1000 |

### Level of the importance per strategy (mean ± std over fold seeds)

| sets | Apriori | IMPresseD | Apriori - IMPresseD | seeds with a positive difference |
|---|---|---|---|---|
| size 1 | 0.0220 ± 0.0006 | 0.0196 ± 0.0004 | 0.0025 ± 0.0004 | 20 of 20 |
| size 2 | 0.0284 ± 0.0006 | 0.0160 ± 0.0003 | 0.0124 ± 0.0005 | 20 of 20 |
| size 3 | 0.0369 ± 0.0007 | 0.0053 ± 0.0001 | 0.0316 ± 0.0007 | 20 of 20 |
| all sizes | 0.0291 ± 0.0006 | 0.0136 ± 0.0002 | 0.0155 ± 0.0005 | 20 of 20 |
