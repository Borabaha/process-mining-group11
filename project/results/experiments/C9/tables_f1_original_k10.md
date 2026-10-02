# C9 tables: f1, original itemsets, 10 folds

## Setting `fixed_test`

### Weighted F1 of the scored (test) fold, per fold seed

| fold seed | mean over folds | std over folds | lowest fold | highest fold |
|---|---|---|---|---|
| 0 | 0.9058 | 0.0327 | 0.8547 | 0.9734 |
| 1 | 0.9078 | 0.0316 | 0.8584 | 0.9552 |
| 2 | 0.9042 | 0.0328 | 0.8413 | 0.9555 |
| 3 | 0.9112 | 0.0254 | 0.8836 | 0.9555 |
| 4 | 0.9057 | 0.0384 | 0.8469 | 0.9734 |
| all listed | 0.9069 | 0.0027 (between seeds) | 0.8413 | 0.9734 |

### Mean importance per itemset and fold seed (first 5 fold seeds, 10 folds x 5 repeats)

| id | itemset | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | std between seeds |
|---|---|---|---|---|---|---|---|---|
| 0 | 370407, ac370000 | -0.0026 | 0.0005 | -0.0020 | 0.0014 | 0.0058 | 0.0006 | 0.0034 |
| 1 | ac370000, ac370419 | -0.0039 | -0.0020 | -0.0047 | 0.0018 | 0.0033 | -0.0011 | 0.0035 |
| 2 | ac370000, ac370443 | -0.0022 | 0.0012 | -0.0005 | 0.0031 | 0.0031 | 0.0010 | 0.0023 |
| 3 | ac370000, ac370419, ac370443 | 0.0019 | 0.0020 | 0.0025 | 0.0057 | 0.0049 | 0.0034 | 0.0018 |
| 4 | ac370419, ac370443 | -0.0015 | -0.0028 | -0.0024 | -0.0006 | -0.0006 | -0.0016 | 0.0010 |
| 5 | ac370000, ac370442 | 0.0007 | 0.0017 | 0.0020 | 0.0043 | 0.0018 | 0.0021 | 0.0013 |
| 6 | 370407, ac370000, ac370419 | 0.0029 | 0.0085 | -0.0002 | 0.0050 | 0.0069 | 0.0046 | 0.0034 |
| 7 | 370407, ac370419 | -0.0003 | 0.0005 | -0.0020 | 0.0005 | 0.0008 | -0.0001 | 0.0011 |
| 8 | ac370442, ac370443 | 0.0033 | -0.0014 | 0.0007 | -0.0002 | 0.0055 | 0.0016 | 0.0028 |
| 9 | ac370000, ac370419, ac370442 | 0.0016 | -0.0004 | 0.0073 | 0.0104 | 0.0059 | 0.0049 | 0.0044 |

### Rank per itemset and fold seed (first 5 fold seeds, 10 folds x 5 repeats; 1 = most important)

| id | itemset | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | range | rank of the mean over these seeds |
|---|---|---|---|---|---|---|---|---|
| 0 | 370407, ac370000 | 9 | 5 | 8 | 7 | 3 | 3-9 | 7 |
| 1 | ac370000, ac370419 | 10 | 9 | 10 | 6 | 6 | 6-10 | 9 |
| 2 | ac370000, ac370443 | 8 | 4 | 6 | 5 | 7 | 4-8 | 6 |
| 3 | ac370000, ac370419, ac370443 | 3 | 2 | 2 | 2 | 5 | 2-5 | 3 |
| 4 | ac370419, ac370443 | 7 | 10 | 9 | 10 | 10 | 7-10 | 10 |
| 5 | ac370000, ac370442 | 5 | 3 | 3 | 4 | 8 | 3-8 | 4 |
| 6 | 370407, ac370000, ac370419 | 2 | 1 | 5 | 3 | 1 | 1-5 | 2 |
| 7 | 370407, ac370419 | 6 | 6 | 7 | 8 | 9 | 6-9 | 8 |
| 8 | ac370442, ac370443 | 1 | 8 | 4 | 9 | 4 | 1-9 | 5 |
| 9 | ac370000, ac370419, ac370442 | 4 | 7 | 1 | 1 | 2 | 1-7 | 1 |

### Agreement of the rankings of two fold seeds (first 5 fold seeds, 10 folds x 5 repeats)

Spearman above the diagonal, Kendall tau-b below it.

|  | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| **seed 0** |  | 0.36 | 0.75 | 0.28 | 0.41 |
| **seed 1** | 0.29 |  | 0.49 | 0.64 | 0.36 |
| **seed 2** | 0.56 | 0.38 |  | 0.70 | 0.41 |
| **seed 3** | 0.20 | 0.56 | 0.56 |  | 0.54 |
| **seed 4** | 0.29 | 0.29 | 0.29 | 0.38 |  |

### Effect of the number of repeats on the agreement of two fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| 3 repeats, first 5 seeds, repeats 0..2 | 0.38 [0.02, 0.76] | 0.26 [0.02, 0.60] | 0.50 | 10 |
| 3 repeats, all 10 seeds, all disjoint repeat blocks (95 % bootstrap interval of the Spearman mean: 0.29 to 0.54) | 0.42 [-0.30, 0.89] | 0.31 [-0.16, 0.73] | 0.47 | 135 |
| 5 repeats, first 5 seeds, repeats 0..4 | 0.49 [0.28, 0.75] | 0.38 [0.20, 0.56] | 0.53 | 10 |
| 5 repeats, all 10 seeds, all disjoint repeat blocks (95 % bootstrap interval of the Spearman mean: 0.42 to 0.61) | 0.51 [-0.15, 0.88] | 0.38 [-0.16, 0.73] | 0.53 | 90 |
| 10 repeats, first 5 seeds, repeats 0..9 | 0.57 [0.30, 0.84] | 0.43 [0.24, 0.69] | 0.53 | 10 |
| 10 repeats, all 10 seeds, all disjoint repeat blocks (95 % bootstrap interval of the Spearman mean: 0.52 to 0.71) | 0.62 [0.27, 0.89] | 0.46 [0.24, 0.73] | 0.61 | 45 |

### The same design with other groups of 5 fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| seeds 0-4, repeats 0..4 | 0.49 [0.28, 0.75] | 0.38 [0.20, 0.56] | 0.53 | 10 |
| seeds 0-4, repeats 0..9 | 0.57 [0.30, 0.84] | 0.43 [0.24, 0.69] | 0.53 | 10 |
| seeds 5-9, repeats 0..4 | 0.49 [0.22, 0.71] | 0.36 [0.11, 0.51] | 0.57 | 10 |
| seeds 5-9, repeats 0..9 | 0.61 [0.45, 0.72] | 0.43 [0.33, 0.56] | 0.60 | 10 |

### Same folds and models, only the permutations differ

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| 3 repeats | 0.56 [0.13, 0.88] | 0.41 [0.07, 0.78] | 0.52 | 30 |
| 5 repeats | 0.66 [0.45, 0.90] | 0.51 [0.29, 0.78] | 0.63 | 10 |

### Two independent studies that each pool several fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| 1 seed(s) x 10 folds x 5 repeats = 50 values per itemset | 0.51 [-0.03, 0.88] | 0.38 [-0.07, 0.73] | 0.55 | 300 |
| 2 seed(s) x 10 folds x 5 repeats = 100 values per itemset | 0.69 [0.16, 0.94] | 0.53 [0.07, 0.82] | 0.64 | 300 |
| 3 seed(s) x 10 folds x 5 repeats = 150 values per itemset | 0.78 [0.35, 0.95] | 0.62 [0.24, 0.87] | 0.74 | 300 |
| 5 seed(s) x 10 folds x 5 repeats = 250 values per itemset | 0.85 [0.64, 0.99] | 0.70 [0.42, 0.96] | 0.79 | 300 |
| 1 seed(s) x 10 folds x 10 repeats = 100 values per itemset | 0.61 [0.27, 0.89] | 0.45 [0.24, 0.73] | 0.59 | 300 |
| 2 seed(s) x 10 folds x 10 repeats = 200 values per itemset | 0.80 [0.54, 0.99] | 0.63 [0.33, 0.96] | 0.71 | 300 |
| 3 seed(s) x 10 folds x 10 repeats = 300 values per itemset | 0.83 [0.47, 0.99] | 0.66 [0.33, 0.96] | 0.73 | 300 |
| 5 seed(s) x 10 folds x 10 repeats = 500 values per itemset | 0.87 [0.72, 0.98] | 0.70 [0.56, 0.91] | 0.73 | 300 |

### Noise of one seed mean against the spread between itemsets

| repeats | std of a seed mean (noise) | the same without the seed's common shift (ranking noise) | std between itemset means (signal) | signal / ranking noise | median gap between neighbouring itemsets |
|---|---|---|---|---|---|
| 3 | 0.00303 | 0.00228 | 0.00212 | 0.93 | 0.00052 |
| 5 | 0.00271 | 0.00183 | 0.00212 | 1.16 | 0.00052 |
| 10 | 0.00248 | 0.00151 | 0.00212 | 1.40 | 0.00052 |

| std between repeats (same fold) | std between fold means (same seed) | std between seed means | values > 0 | values = 0 | values < 0 |
|---|---|---|---|---|---|
| 0.01099 | 0.01092 | 0.00248 | 46.9 % | 12.9 % | 40.2 % |

## Setting `fixed_train`

### Weighted F1 of the scored (train) fold, per fold seed

| fold seed | mean over folds | std over folds | lowest fold | highest fold |
|---|---|---|---|---|
| 0 | 0.9929 | 0.0018 | 0.9892 | 0.9961 |
| 1 | 0.9920 | 0.0022 | 0.9892 | 0.9951 |
| 2 | 0.9934 | 0.0021 | 0.9902 | 0.9961 |
| 3 | 0.9922 | 0.0022 | 0.9882 | 0.9961 |
| 4 | 0.9927 | 0.0015 | 0.9911 | 0.9961 |
| all listed | 0.9927 | 0.0006 (between seeds) | 0.9882 | 0.9961 |

### Mean importance per itemset and fold seed (first 5 fold seeds, 10 folds x 5 repeats)

| id | itemset | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | std between seeds |
|---|---|---|---|---|---|---|---|---|
| 0 | 370407, ac370000 | 0.0358 | 0.0356 | 0.0374 | 0.0367 | 0.0363 | 0.0364 | 0.0007 |
| 1 | ac370000, ac370419 | 0.0328 | 0.0324 | 0.0332 | 0.0332 | 0.0320 | 0.0327 | 0.0005 |
| 2 | ac370000, ac370443 | 0.0320 | 0.0304 | 0.0317 | 0.0319 | 0.0315 | 0.0315 | 0.0006 |
| 3 | ac370000, ac370419, ac370443 | 0.0373 | 0.0368 | 0.0374 | 0.0382 | 0.0376 | 0.0375 | 0.0005 |
| 4 | ac370419, ac370443 | 0.0237 | 0.0221 | 0.0229 | 0.0230 | 0.0226 | 0.0229 | 0.0006 |
| 5 | ac370000, ac370442 | 0.0324 | 0.0319 | 0.0335 | 0.0323 | 0.0325 | 0.0325 | 0.0006 |
| 6 | 370407, ac370000, ac370419 | 0.0407 | 0.0399 | 0.0414 | 0.0402 | 0.0403 | 0.0405 | 0.0006 |
| 7 | 370407, ac370419 | 0.0255 | 0.0251 | 0.0253 | 0.0246 | 0.0260 | 0.0253 | 0.0005 |
| 8 | ac370442, ac370443 | 0.0226 | 0.0219 | 0.0222 | 0.0218 | 0.0214 | 0.0220 | 0.0005 |
| 9 | ac370000, ac370419, ac370442 | 0.0379 | 0.0367 | 0.0388 | 0.0377 | 0.0381 | 0.0378 | 0.0007 |

### Rank per itemset and fold seed (first 5 fold seeds, 10 folds x 5 repeats; 1 = most important)

| id | itemset | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | range | rank of the mean over these seeds |
|---|---|---|---|---|---|---|---|---|
| 0 | 370407, ac370000 | 4 | 4 | 3 | 4 | 4 | 3-4 | 4 |
| 1 | ac370000, ac370419 | 5 | 5 | 6 | 5 | 6 | 5-6 | 5 |
| 2 | ac370000, ac370443 | 7 | 7 | 7 | 7 | 7 | 7-7 | 7 |
| 3 | ac370000, ac370419, ac370443 | 3 | 2 | 4 | 2 | 3 | 2-4 | 3 |
| 4 | ac370419, ac370443 | 9 | 9 | 9 | 9 | 9 | 9-9 | 9 |
| 5 | ac370000, ac370442 | 6 | 6 | 5 | 6 | 5 | 5-6 | 6 |
| 6 | 370407, ac370000, ac370419 | 1 | 1 | 1 | 1 | 1 | 1-1 | 1 |
| 7 | 370407, ac370419 | 8 | 8 | 8 | 8 | 8 | 8-8 | 8 |
| 8 | ac370442, ac370443 | 10 | 10 | 10 | 10 | 10 | 10-10 | 10 |
| 9 | ac370000, ac370419, ac370442 | 2 | 3 | 2 | 3 | 2 | 2-3 | 2 |

### Agreement of the rankings of two fold seeds (first 5 fold seeds, 10 folds x 5 repeats)

Spearman above the diagonal, Kendall tau-b below it.

|  | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| **seed 0** |  | 0.99 | 0.98 | 0.99 | 0.99 |
| **seed 1** | 0.96 |  | 0.95 | 1.00 | 0.98 |
| **seed 2** | 0.91 | 0.87 |  | 0.95 | 0.99 |
| **seed 3** | 0.96 | 1.00 | 0.87 |  | 0.98 |
| **seed 4** | 0.96 | 0.91 | 0.96 | 0.91 |  |

### Effect of the number of repeats on the agreement of two fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| 3 repeats, first 5 seeds, repeats 0..2 | 0.97 [0.94, 0.99] | 0.91 [0.82, 0.96] | 0.77 | 10 |
| 3 repeats, all 10 seeds, all disjoint repeat blocks (95 % bootstrap interval of the Spearman mean: 0.97 to 0.98) | 0.97 [0.89, 1.00] | 0.92 [0.78, 1.00] | 0.87 | 135 |
| 5 repeats, first 5 seeds, repeats 0..4 | 0.98 [0.95, 1.00] | 0.93 [0.87, 1.00] | 0.87 | 10 |
| 5 repeats, all 10 seeds, all disjoint repeat blocks (95 % bootstrap interval of the Spearman mean: 0.97 to 0.99) | 0.98 [0.93, 1.00] | 0.94 [0.82, 1.00] | 0.88 | 90 |
| 10 repeats, first 5 seeds, repeats 0..9 | 0.99 [0.98, 1.00] | 0.95 [0.91, 1.00] | 1.00 | 10 |
| 10 repeats, all 10 seeds, all disjoint repeat blocks (95 % bootstrap interval of the Spearman mean: 0.97 to 0.99) | 0.98 [0.95, 1.00] | 0.95 [0.87, 1.00] | 0.93 | 45 |

### The same design with other groups of 5 fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| seeds 0-4, repeats 0..4 | 0.98 [0.95, 1.00] | 0.93 [0.87, 1.00] | 0.87 | 10 |
| seeds 0-4, repeats 0..9 | 0.99 [0.98, 1.00] | 0.95 [0.91, 1.00] | 1.00 | 10 |
| seeds 5-9, repeats 0..4 | 0.98 [0.95, 1.00] | 0.94 [0.87, 1.00] | 0.87 | 10 |
| seeds 5-9, repeats 0..9 | 0.98 [0.95, 1.00] | 0.95 [0.87, 1.00] | 0.87 | 10 |

### Same folds and models, only the permutations differ

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| 3 repeats | 0.98 [0.92, 1.00] | 0.93 [0.78, 1.00] | 0.89 | 30 |
| 5 repeats | 0.98 [0.96, 1.00] | 0.94 [0.91, 1.00] | 0.93 | 10 |

### Two independent studies that each pool several fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| 1 seed(s) x 10 folds x 5 repeats = 50 values per itemset | 0.98 [0.93, 1.00] | 0.94 [0.82, 1.00] | 0.88 | 300 |
| 2 seed(s) x 10 folds x 5 repeats = 100 values per itemset | 0.99 [0.96, 1.00] | 0.96 [0.91, 1.00] | 0.96 | 300 |
| 3 seed(s) x 10 folds x 5 repeats = 150 values per itemset | 0.99 [0.96, 1.00] | 0.98 [0.91, 1.00] | 0.99 | 300 |
| 5 seed(s) x 10 folds x 5 repeats = 250 values per itemset | 1.00 [0.99, 1.00] | 0.99 [0.96, 1.00] | 1.00 | 300 |
| 1 seed(s) x 10 folds x 10 repeats = 100 values per itemset | 0.98 [0.95, 1.00] | 0.95 [0.87, 1.00] | 0.93 | 300 |
| 2 seed(s) x 10 folds x 10 repeats = 200 values per itemset | 0.99 [0.96, 1.00] | 0.98 [0.91, 1.00] | 0.99 | 300 |
| 3 seed(s) x 10 folds x 10 repeats = 300 values per itemset | 1.00 [0.98, 1.00] | 0.99 [0.91, 1.00] | 1.00 | 300 |
| 5 seed(s) x 10 folds x 10 repeats = 500 values per itemset | 1.00 [0.99, 1.00] | 1.00 [0.96, 1.00] | 1.00 | 300 |

### Noise of one seed mean against the spread between itemsets

| repeats | std of a seed mean (noise) | the same without the seed's common shift (ranking noise) | std between itemset means (signal) | signal / ranking noise | median gap between neighbouring itemsets |
|---|---|---|---|---|---|
| 3 | 0.00069 | 0.00062 | 0.00659 | 10.64 | 0.00116 |
| 5 | 0.00060 | 0.00052 | 0.00659 | 12.63 | 0.00116 |
| 10 | 0.00048 | 0.00040 | 0.00659 | 16.66 | 0.00116 |

| std between repeats (same fold) | std between fold means (same seed) | std between seed means | values > 0 | values = 0 | values < 0 |
|---|---|---|---|---|---|
| 0.00359 | 0.00256 | 0.00048 | 100.0 % | 0.0 % | 0.0 % |

## All settings: mean over all fold seeds and repeats

| id | itemset | fixed_test: mean ± s.e. | rank | fixed_train: mean ± s.e. | rank |
|---|---|---|---|---|---|
| 0 | 370407, ac370000 | 0.00178 ± 0.00083 | 7 | 0.03671 ± 0.00017 | 4 |
| 1 | ac370000, ac370419 | 0.00056 ± 0.00105 | 8 | 0.03311 ± 0.00014 | 5 |
| 2 | ac370000, ac370443 | 0.00264 ± 0.00073 | 5 | 0.03179 ± 0.00019 | 7 |
| 3 | ac370000, ac370419, ac370443 | 0.00473 ± 0.00061 | 3 | 0.03749 ± 0.00017 | 3 |
| 4 | ac370419, ac370443 | 0.00052 ± 0.00054 | 9 | 0.02297 ± 0.00013 | 9 |
| 5 | ac370000, ac370442 | 0.00455 ± 0.00094 | 4 | 0.03265 ± 0.00014 | 6 |
| 6 | 370407, ac370000, ac370419 | 0.00534 ± 0.00093 | 2 | 0.04025 ± 0.00018 | 1 |
| 7 | 370407, ac370419 | 0.00033 ± 0.00064 | 10 | 0.02505 ± 0.00009 | 8 |
| 8 | ac370442, ac370443 | 0.00210 ± 0.00062 | 6 | 0.02181 ± 0.00010 | 10 |
| 9 | ac370000, ac370419, ac370442 | 0.00586 ± 0.00096 | 1 | 0.03793 ± 0.00020 | 2 |

### Do two settings rank the itemsets alike?

| settings | Spearman | Kendall tau-b | top-3 overlap |
|---|---|---|---|
| fixed_test vs fixed_train | 0.71 | 0.56 | 1.00 |
