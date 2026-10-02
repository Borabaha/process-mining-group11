# C9 add-on: project sets on f2

## Setting `fixed_test` (10 fold seeds, 5 folds, 10 repeats)

Weighted F1 of the scored fold: 0.8910 (std between seed means 0.0062).

### Ranking stability inside a group of sets

| group | two seeds: Spearman mean [min, max] | two seeds: Kendall | two seeds: top-3 overlap | two groups of 5 seeds: Spearman | two groups of 5 seeds: Kendall | two groups of 5 seeds: top-3 overlap | sets above zero (mean - 2 s.e. > 0) | (seed, fold, set) not measurable |
|---|---|---|---|---|---|---|---|---|
| all 60 sets | 0.97 [0.95, 0.99] | 0.88 | 1.00 | 0.99 | 0.94 | 1.00 | 55 of 60 | 49 of 3000 |
| apriori, all sizes (30) | 0.94 [0.90, 0.98] | 0.81 | 0.59 | 0.98 | 0.89 | 0.56 | 28 of 30 | 0 of 1500 |
| apriori, size 1 (10) | 0.71 [0.37, 0.95] | 0.58 | 0.65 | 0.88 | 0.76 | 0.70 | 8 of 10 | 0 of 500 |
| apriori, size 2 (10) | 0.91 [0.79, 0.99] | 0.80 | 0.81 | 0.98 | 0.94 | 0.96 | 10 of 10 | 0 of 500 |
| apriori, size 3 (10) | 0.82 [0.60, 0.98] | 0.68 | 0.59 | 0.89 | 0.79 | 0.56 | 10 of 10 | 0 of 500 |
| impressed, all sizes (30) | 0.98 [0.96, 0.99] | 0.91 | 1.00 | 0.99 | 0.96 | 1.00 | 27 of 30 | 49 of 1500 |
| impressed, size 1 (10) | 0.86 [0.61, 0.99] | 0.78 | 1.00 | 0.96 | 0.92 | 1.00 | 7 of 10 | 49 of 500 |
| impressed, size 2 (10) | 0.96 [0.90, 1.00] | 0.89 | 0.80 | 0.99 | 0.96 | 1.00 | 10 of 10 | 0 of 500 |
| impressed, size 3 (10) | 0.99 [0.95, 1.00] | 0.95 | 0.93 | 1.00 | 1.00 | 1.00 | 10 of 10 | 0 of 500 |

### Level of the importance per strategy (mean ± std over fold seeds)

| sets | Apriori | IMPresseD | Apriori - IMPresseD | seeds with a positive difference |
|---|---|---|---|---|
| size 1 | 0.0108 ± 0.0037 | 0.0200 ± 0.0026 | -0.0092 ± 0.0016 | 0 of 10 |
| size 2 | 0.0249 ± 0.0044 | 0.0742 ± 0.0046 | -0.0493 ± 0.0020 | 0 of 10 |
| size 3 | 0.0479 ± 0.0052 | 0.0674 ± 0.0027 | -0.0196 ± 0.0036 | 0 of 10 |
| all sizes | 0.0278 ± 0.0041 | 0.0539 ± 0.0032 | -0.0261 ± 0.0019 | 0 of 10 |

## Setting `fixed_train` (10 fold seeds, 5 folds, 10 repeats)

Weighted F1 of the scored fold: 0.9939 (std between seed means 0.0003).

### Ranking stability inside a group of sets

| group | two seeds: Spearman mean [min, max] | two seeds: Kendall | two seeds: top-3 overlap | two groups of 5 seeds: Spearman | two groups of 5 seeds: Kendall | two groups of 5 seeds: top-3 overlap | sets above zero (mean - 2 s.e. > 0) | (seed, fold, set) not measurable |
|---|---|---|---|---|---|---|---|---|
| all 60 sets | 1.00 [1.00, 1.00] | 0.97 | 0.82 | 1.00 | 0.99 | 0.73 | 59 of 60 | 1 of 3000 |
| apriori, all sizes (30) | 0.99 [0.98, 1.00] | 0.94 | 0.68 | 1.00 | 0.98 | 1.00 | 30 of 30 | 0 of 1500 |
| apriori, size 1 (10) | 0.92 [0.71, 1.00] | 0.83 | 1.00 | 0.99 | 0.96 | 1.00 | 10 of 10 | 0 of 500 |
| apriori, size 2 (10) | 0.96 [0.89, 1.00] | 0.89 | 0.88 | 1.00 | 1.00 | 1.00 | 10 of 10 | 0 of 500 |
| apriori, size 3 (10) | 0.91 [0.79, 1.00] | 0.83 | 0.68 | 0.98 | 0.94 | 1.00 | 10 of 10 | 0 of 500 |
| impressed, all sizes (30) | 1.00 [0.99, 1.00] | 0.98 | 0.82 | 1.00 | 0.99 | 0.73 | 29 of 30 | 1 of 1500 |
| impressed, size 1 (10) | 1.00 [0.99, 1.00] | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 9 of 10 | 1 of 500 |
| impressed, size 2 (10) | 0.99 [0.98, 1.00] | 0.96 | 0.84 | 1.00 | 0.99 | 0.99 | 10 of 10 | 0 of 500 |
| impressed, size 3 (10) | 1.00 [1.00, 1.00] | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 10 of 10 | 0 of 500 |

### Level of the importance per strategy (mean ± std over fold seeds)

| sets | Apriori | IMPresseD | Apriori - IMPresseD | seeds with a positive difference |
|---|---|---|---|---|
| size 1 | 0.0473 ± 0.0008 | 0.0437 ± 0.0006 | 0.0036 ± 0.0006 | 10 of 10 |
| size 2 | 0.0791 ± 0.0011 | 0.1096 ± 0.0008 | -0.0305 ± 0.0008 | 0 of 10 |
| size 3 | 0.1089 ± 0.0016 | 0.0943 ± 0.0008 | 0.0146 ± 0.0015 | 10 of 10 |
| all sizes | 0.0784 ± 0.0011 | 0.0826 ± 0.0006 | -0.0041 ± 0.0007 | 0 of 10 |
