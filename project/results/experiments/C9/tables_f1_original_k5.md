# C9 tables: f1, original itemsets, 5 folds

## Setting `fixed_test`

### Weighted F1 of the scored (test) fold, per fold seed

| fold seed | mean over folds | std over folds | lowest fold | highest fold |
|---|---|---|---|---|
| 0 | 0.9094 | 0.0176 | 0.8853 | 0.9284 |
| 1 | 0.9023 | 0.0329 | 0.8529 | 0.9334 |
| 2 | 0.9061 | 0.0273 | 0.8718 | 0.9377 |
| 3 | 0.8996 | 0.0231 | 0.8756 | 0.9377 |
| 4 | 0.8952 | 0.0157 | 0.8770 | 0.9204 |
| all listed | 0.9025 | 0.0055 (between seeds) | 0.8529 | 0.9377 |

### Mean importance per itemset and fold seed (first 5 fold seeds, 5 folds x 5 repeats)

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

### Rank per itemset and fold seed (first 5 fold seeds, 5 folds x 5 repeats; 1 = most important)

| id | itemset | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | range | rank of the mean over these seeds |
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

### Agreement of the rankings of two fold seeds (first 5 fold seeds, 5 folds x 5 repeats)

Spearman above the diagonal, Kendall tau-b below it.

|  | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| **seed 0** |  | 0.70 | 0.82 | 0.76 | 0.66 |
| **seed 1** | 0.56 |  | 0.89 | 0.60 | 0.73 |
| **seed 2** | 0.64 | 0.73 |  | 0.53 | 0.81 |
| **seed 3** | 0.56 | 0.47 | 0.38 |  | 0.58 |
| **seed 4** | 0.51 | 0.51 | 0.60 | 0.51 |  |

### Effect of the number of repeats on the agreement of two fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| 3 repeats, first 5 seeds, repeats 0..2 | 0.48 [0.24, 0.81] | 0.36 [0.11, 0.64] | 0.47 | 10 |
| 3 repeats, all 20 seeds, all disjoint repeat blocks (95 % bootstrap interval of the Spearman mean: 0.29 to 0.51) | 0.41 [-0.79, 0.96] | 0.31 [-0.60, 0.91] | 0.49 | 1900 |
| 5 repeats, first 5 seeds, repeats 0..4 | 0.71 [0.53, 0.89] | 0.55 [0.38, 0.73] | 0.67 | 10 |
| 5 repeats, all 20 seeds, all disjoint repeat blocks (95 % bootstrap interval of the Spearman mean: 0.36 to 0.61) | 0.50 [-0.67, 0.96] | 0.37 [-0.56, 0.87] | 0.55 | 1140 |
| 10 repeats, first 5 seeds, repeats 0..9 | 0.66 [0.47, 0.85] | 0.49 [0.33, 0.69] | 0.67 | 10 |
| 10 repeats, all 20 seeds, all disjoint repeat blocks (95 % bootstrap interval of the Spearman mean: 0.45 to 0.72) | 0.60 [-0.52, 0.98] | 0.46 [-0.38, 0.91] | 0.60 | 570 |
| 30 repeats, first 5 seeds, repeats 0..29 | 0.74 [0.39, 1.00] | 0.56 [0.20, 1.00] | 0.73 | 10 |
| 30 repeats, all 20 seeds, all disjoint repeat blocks (95 % bootstrap interval of the Spearman mean: 0.49 to 0.80) | 0.66 [-0.31, 1.00] | 0.51 [-0.24, 1.00] | 0.64 | 190 |

### The same design with other groups of 5 fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| seeds 0-4, repeats 0..4 | 0.71 [0.53, 0.89] | 0.55 [0.38, 0.73] | 0.67 | 10 |
| seeds 0-4, repeats 0..9 | 0.66 [0.47, 0.85] | 0.49 [0.33, 0.69] | 0.67 | 10 |
| seeds 5-9, repeats 0..4 | 0.52 [0.14, 0.84] | 0.38 [0.16, 0.73] | 0.43 | 10 |
| seeds 5-9, repeats 0..9 | 0.65 [0.32, 0.93] | 0.48 [0.20, 0.78] | 0.63 | 10 |
| seeds 10-14, repeats 0..4 | 0.71 [0.54, 0.85] | 0.56 [0.38, 0.73] | 0.63 | 10 |
| seeds 10-14, repeats 0..9 | 0.80 [0.65, 0.93] | 0.64 [0.42, 0.82] | 0.80 | 10 |
| seeds 15-19, repeats 0..4 | 0.38 [-0.07, 0.95] | 0.30 [-0.02, 0.87] | 0.43 | 10 |
| seeds 15-19, repeats 0..9 | 0.37 [-0.14, 0.88] | 0.28 [-0.16, 0.73] | 0.50 | 10 |

### Same folds and models, only the permutations differ

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| 3 repeats | 0.55 [-0.49, 0.99] | 0.41 [-0.51, 0.96] | 0.56 | 900 |
| 5 repeats | 0.67 [-0.26, 0.96] | 0.52 [-0.20, 0.91] | 0.63 | 300 |
| 10 repeats | 0.80 [0.50, 0.96] | 0.65 [0.38, 0.87] | 0.73 | 60 |

### Two independent studies that each pool several fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| 1 seed(s) x 5 folds x 5 repeats = 25 values per itemset | 0.49 [-0.54, 0.95] | 0.36 [-0.42, 0.87] | 0.55 | 300 |
| 2 seed(s) x 5 folds x 5 repeats = 50 values per itemset | 0.69 [-0.07, 0.96] | 0.53 [-0.02, 0.87] | 0.63 | 300 |
| 3 seed(s) x 5 folds x 5 repeats = 75 values per itemset | 0.75 [0.36, 0.99] | 0.59 [0.20, 0.96] | 0.69 | 300 |
| 5 seed(s) x 5 folds x 5 repeats = 125 values per itemset | 0.83 [0.55, 1.00] | 0.66 [0.33, 1.00] | 0.76 | 300 |
| 10 seed(s) x 5 folds x 5 repeats = 250 values per itemset | 0.89 [0.75, 1.00] | 0.77 [0.51, 1.00] | 0.88 | 300 |
| 1 seed(s) x 5 folds x 10 repeats = 50 values per itemset | 0.59 [-0.39, 0.98] | 0.46 [-0.29, 0.91] | 0.61 | 300 |
| 2 seed(s) x 5 folds x 10 repeats = 100 values per itemset | 0.75 [0.07, 0.98] | 0.58 [0.07, 0.91] | 0.68 | 300 |
| 3 seed(s) x 5 folds x 10 repeats = 150 values per itemset | 0.80 [0.32, 0.99] | 0.64 [0.20, 0.96] | 0.75 | 300 |
| 5 seed(s) x 5 folds x 10 repeats = 250 values per itemset | 0.86 [0.59, 0.99] | 0.71 [0.38, 0.96] | 0.84 | 300 |
| 10 seed(s) x 5 folds x 10 repeats = 500 values per itemset | 0.92 [0.79, 1.00] | 0.81 [0.60, 1.00] | 0.97 | 300 |
| 1 seed(s) x 5 folds x 30 repeats = 150 values per itemset | 0.64 [-0.31, 1.00] | 0.50 [-0.24, 1.00] | 0.65 | 300 |
| 2 seed(s) x 5 folds x 30 repeats = 300 values per itemset | 0.79 [0.15, 0.99] | 0.63 [0.07, 0.96] | 0.76 | 300 |
| 3 seed(s) x 5 folds x 30 repeats = 450 values per itemset | 0.84 [0.37, 0.99] | 0.68 [0.20, 0.96] | 0.82 | 300 |
| 5 seed(s) x 5 folds x 30 repeats = 750 values per itemset | 0.89 [0.68, 1.00] | 0.75 [0.42, 1.00] | 0.94 | 300 |
| 10 seed(s) x 5 folds x 30 repeats = 1500 values per itemset | 0.93 [0.81, 0.99] | 0.81 [0.60, 0.96] | 1.00 | 300 |

### Noise of one seed mean against the spread between itemsets

| repeats | std of a seed mean (noise) | the same without the seed's common shift (ranking noise) | std between itemset means (signal) | signal / ranking noise | median gap between neighbouring itemsets |
|---|---|---|---|---|---|
| 3 | 0.00422 | 0.00238 | 0.00219 | 0.92 | 0.00045 |
| 5 | 0.00401 | 0.00202 | 0.00219 | 1.09 | 0.00045 |
| 10 | 0.00385 | 0.00170 | 0.00219 | 1.29 | 0.00045 |
| 30 | 0.00373 | 0.00144 | 0.00219 | 1.52 | 0.00045 |

| std between repeats (same fold) | std between fold means (same seed) | std between seed means | values > 0 | values = 0 | values < 0 |
|---|---|---|---|---|---|
| 0.00800 | 0.00729 | 0.00373 | 50.9 % | 5.8 % | 43.3 % |

## Setting `fixed_train`

### Weighted F1 of the scored (train) fold, per fold seed

| fold seed | mean over folds | std over folds | lowest fold | highest fold |
|---|---|---|---|---|
| 0 | 0.9938 | 0.0022 | 0.9911 | 0.9967 |
| 1 | 0.9934 | 0.0028 | 0.9911 | 0.9978 |
| 2 | 0.9954 | 0.0016 | 0.9934 | 0.9978 |
| 3 | 0.9942 | 0.0014 | 0.9934 | 0.9967 |
| 4 | 0.9931 | 0.0030 | 0.9900 | 0.9967 |
| all listed | 0.9940 | 0.0009 (between seeds) | 0.9900 | 0.9978 |

### Mean importance per itemset and fold seed (first 5 fold seeds, 5 folds x 5 repeats)

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

### Rank per itemset and fold seed (first 5 fold seeds, 5 folds x 5 repeats; 1 = most important)

| id | itemset | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | range | rank of the mean over these seeds |
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

### Agreement of the rankings of two fold seeds (first 5 fold seeds, 5 folds x 5 repeats)

Spearman above the diagonal, Kendall tau-b below it.

|  | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| **seed 0** |  | 0.93 | 0.93 | 0.96 | 0.94 |
| **seed 1** | 0.78 |  | 0.90 | 0.94 | 0.87 |
| **seed 2** | 0.82 | 0.78 |  | 0.90 | 0.94 |
| **seed 3** | 0.87 | 0.82 | 0.78 |  | 0.89 |
| **seed 4** | 0.82 | 0.69 | 0.82 | 0.78 |  |

### Effect of the number of repeats on the agreement of two fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| 3 repeats, first 5 seeds, repeats 0..2 | 0.89 [0.83, 0.95] | 0.72 [0.60, 0.87] | 0.77 | 10 |
| 3 repeats, all 20 seeds, all disjoint repeat blocks (95 % bootstrap interval of the Spearman mean: 0.93 to 0.94) | 0.94 [0.79, 1.00] | 0.84 [0.60, 1.00] | 0.77 | 1900 |
| 5 repeats, first 5 seeds, repeats 0..4 | 0.92 [0.87, 0.96] | 0.80 [0.69, 0.87] | 0.73 | 10 |
| 5 repeats, all 20 seeds, all disjoint repeat blocks (95 % bootstrap interval of the Spearman mean: 0.95 to 0.95) | 0.95 [0.85, 1.00] | 0.86 [0.69, 1.00] | 0.78 | 1140 |
| 10 repeats, first 5 seeds, repeats 0..9 | 0.94 [0.87, 1.00] | 0.85 [0.69, 1.00] | 0.80 | 10 |
| 10 repeats, all 20 seeds, all disjoint repeat blocks (95 % bootstrap interval of the Spearman mean: 0.95 to 0.96) | 0.96 [0.87, 1.00] | 0.88 [0.69, 1.00] | 0.78 | 570 |
| 30 repeats, first 5 seeds, repeats 0..29 | 0.98 [0.96, 1.00] | 0.94 [0.91, 1.00] | 0.87 | 10 |
| 30 repeats, all 20 seeds, all disjoint repeat blocks (95 % bootstrap interval of the Spearman mean: 0.96 to 0.98) | 0.97 [0.90, 1.00] | 0.91 [0.73, 1.00] | 0.79 | 190 |

### The same design with other groups of 5 fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| seeds 0-4, repeats 0..4 | 0.92 [0.87, 0.96] | 0.80 [0.69, 0.87] | 0.73 | 10 |
| seeds 0-4, repeats 0..9 | 0.94 [0.87, 1.00] | 0.85 [0.69, 1.00] | 0.80 | 10 |
| seeds 5-9, repeats 0..4 | 0.94 [0.90, 0.99] | 0.84 [0.73, 0.96] | 0.73 | 10 |
| seeds 5-9, repeats 0..9 | 0.96 [0.92, 0.99] | 0.87 [0.78, 0.96] | 0.73 | 10 |
| seeds 10-14, repeats 0..4 | 0.94 [0.92, 0.98] | 0.85 [0.78, 0.91] | 0.87 | 10 |
| seeds 10-14, repeats 0..9 | 0.96 [0.92, 0.99] | 0.88 [0.78, 0.96] | 0.77 | 10 |
| seeds 15-19, repeats 0..4 | 0.94 [0.88, 0.98] | 0.84 [0.73, 0.91] | 0.73 | 10 |
| seeds 15-19, repeats 0..9 | 0.93 [0.88, 0.98] | 0.83 [0.73, 0.91] | 0.73 | 10 |

### Same folds and models, only the permutations differ

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| 3 repeats | 0.95 [0.78, 1.00] | 0.86 [0.64, 1.00] | 0.81 | 900 |
| 5 repeats | 0.96 [0.88, 1.00] | 0.89 [0.73, 1.00] | 0.83 | 300 |
| 10 repeats | 0.97 [0.92, 1.00] | 0.91 [0.78, 1.00] | 0.84 | 60 |

### Two independent studies that each pool several fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| 1 seed(s) x 5 folds x 5 repeats = 25 values per itemset | 0.95 [0.87, 1.00] | 0.86 [0.69, 1.00] | 0.78 | 300 |
| 2 seed(s) x 5 folds x 5 repeats = 50 values per itemset | 0.96 [0.90, 1.00] | 0.88 [0.73, 1.00] | 0.78 | 300 |
| 3 seed(s) x 5 folds x 5 repeats = 75 values per itemset | 0.96 [0.90, 1.00] | 0.90 [0.73, 1.00] | 0.79 | 300 |
| 5 seed(s) x 5 folds x 5 repeats = 125 values per itemset | 0.97 [0.90, 1.00] | 0.92 [0.73, 1.00] | 0.80 | 300 |
| 10 seed(s) x 5 folds x 5 repeats = 250 values per itemset | 0.98 [0.93, 1.00] | 0.93 [0.82, 1.00] | 0.85 | 300 |
| 1 seed(s) x 5 folds x 10 repeats = 50 values per itemset | 0.96 [0.87, 1.00] | 0.88 [0.69, 1.00] | 0.78 | 300 |
| 2 seed(s) x 5 folds x 10 repeats = 100 values per itemset | 0.97 [0.90, 1.00] | 0.91 [0.73, 1.00] | 0.81 | 300 |
| 3 seed(s) x 5 folds x 10 repeats = 150 values per itemset | 0.97 [0.92, 1.00] | 0.92 [0.78, 1.00] | 0.82 | 300 |
| 5 seed(s) x 5 folds x 10 repeats = 250 values per itemset | 0.98 [0.93, 1.00] | 0.93 [0.82, 1.00] | 0.84 | 300 |
| 10 seed(s) x 5 folds x 10 repeats = 500 values per itemset | 0.98 [0.94, 1.00] | 0.94 [0.82, 1.00] | 0.90 | 300 |
| 1 seed(s) x 5 folds x 30 repeats = 150 values per itemset | 0.97 [0.90, 1.00] | 0.91 [0.73, 1.00] | 0.80 | 300 |
| 2 seed(s) x 5 folds x 30 repeats = 300 values per itemset | 0.97 [0.92, 1.00] | 0.92 [0.78, 1.00] | 0.81 | 300 |
| 3 seed(s) x 5 folds x 30 repeats = 450 values per itemset | 0.98 [0.94, 1.00] | 0.93 [0.82, 1.00] | 0.83 | 300 |
| 5 seed(s) x 5 folds x 30 repeats = 750 values per itemset | 0.98 [0.94, 1.00] | 0.94 [0.82, 1.00] | 0.87 | 300 |
| 10 seed(s) x 5 folds x 30 repeats = 1500 values per itemset | 0.99 [0.95, 1.00] | 0.96 [0.87, 1.00] | 0.97 | 300 |

### Noise of one seed mean against the spread between itemsets

| repeats | std of a seed mean (noise) | the same without the seed's common shift (ranking noise) | std between itemset means (signal) | signal / ranking noise | median gap between neighbouring itemsets |
|---|---|---|---|---|---|
| 3 | 0.00130 | 0.00108 | 0.00704 | 6.52 | 0.00105 |
| 5 | 0.00114 | 0.00091 | 0.00704 | 7.76 | 0.00105 |
| 10 | 0.00100 | 0.00075 | 0.00704 | 9.43 | 0.00105 |
| 30 | 0.00088 | 0.00059 | 0.00704 | 11.85 | 0.00105 |

| std between repeats (same fold) | std between fold means (same seed) | std between seed means | values > 0 | values = 0 | values < 0 |
|---|---|---|---|---|---|
| 0.00393 | 0.00318 | 0.00088 | 100.0 % | 0.0 % | 0.0 % |

## Setting `faithful_train`

### Weighted F1 of the scored (train) fold, per fold seed

| fold seed | mean over folds | std over folds | lowest fold | highest fold |
|---|---|---|---|---|
| 0 | 0.9938 | 0.0022 | 0.9911 | 0.9967 |
| 1 | 0.9934 | 0.0028 | 0.9911 | 0.9978 |
| 2 | 0.9954 | 0.0016 | 0.9934 | 0.9978 |
| 3 | 0.9942 | 0.0014 | 0.9934 | 0.9967 |
| 4 | 0.9931 | 0.0030 | 0.9900 | 0.9967 |
| all listed | 0.9940 | 0.0009 (between seeds) | 0.9900 | 0.9978 |

### Mean importance per itemset and fold seed (first 5 fold seeds, 5 folds x 10 repeats)

| id | itemset | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | std between seeds |
|---|---|---|---|---|---|---|---|---|
| 0 | 370407, ac370000 | 0.0650 | 0.0636 | 0.0554 | 0.0606 | 0.0570 | 0.0603 | 0.0041 |
| 1 | ac370000, ac370419 | 0.0687 | 0.0720 | 0.0606 | 0.0655 | 0.0644 | 0.0663 | 0.0043 |
| 2 | ac370000, ac370443 | 0.0626 | 0.0660 | 0.0562 | 0.0619 | 0.0599 | 0.0613 | 0.0036 |
| 3 | ac370000, ac370419, ac370443 | 0.0614 | 0.0634 | 0.0564 | 0.0604 | 0.0588 | 0.0601 | 0.0026 |
| 4 | ac370419, ac370443 | 0.0615 | 0.0648 | 0.0549 | 0.0584 | 0.0592 | 0.0597 | 0.0037 |
| 5 | ac370000, ac370442 | 0.0601 | 0.0619 | 0.0541 | 0.0578 | 0.0576 | 0.0583 | 0.0029 |
| 6 | 370407, ac370000, ac370419 | 0.0631 | 0.0625 | 0.0576 | 0.0598 | 0.0583 | 0.0603 | 0.0025 |
| 7 | 370407, ac370419 | 0.0623 | 0.0636 | 0.0565 | 0.0596 | 0.0579 | 0.0600 | 0.0030 |
| 8 | ac370442, ac370443 | 0.0615 | 0.0597 | 0.0532 | 0.0587 | 0.0586 | 0.0584 | 0.0031 |
| 9 | ac370000, ac370419, ac370442 | 0.0617 | 0.0621 | 0.0589 | 0.0590 | 0.0599 | 0.0603 | 0.0015 |

### Rank per itemset and fold seed (first 5 fold seeds, 5 folds x 10 repeats; 1 = most important)

| id | itemset | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | range | rank of the mean over these seeds |
|---|---|---|---|---|---|---|---|---|
| 0 | 370407, ac370000 | 2 | 5 | 7 | 3 | 10 | 2-10 | 3 |
| 1 | ac370000, ac370419 | 1 | 1 | 1 | 1 | 1 | 1-1 | 1 |
| 2 | ac370000, ac370443 | 4 | 2 | 6 | 2 | 2 | 2-6 | 2 |
| 3 | ac370000, ac370419, ac370443 | 9 | 6 | 5 | 4 | 5 | 4-9 | 6 |
| 4 | ac370419, ac370443 | 8 | 3 | 8 | 9 | 4 | 3-9 | 8 |
| 5 | ac370000, ac370442 | 10 | 9 | 9 | 10 | 9 | 9-10 | 10 |
| 6 | 370407, ac370000, ac370419 | 3 | 7 | 3 | 5 | 7 | 3-7 | 5 |
| 7 | 370407, ac370419 | 5 | 4 | 4 | 6 | 8 | 4-8 | 7 |
| 8 | ac370442, ac370443 | 7 | 10 | 10 | 8 | 6 | 6-10 | 9 |
| 9 | ac370000, ac370419, ac370442 | 6 | 8 | 2 | 7 | 3 | 2-8 | 4 |

### Agreement of the rankings of two fold seeds (first 5 fold seeds, 5 folds x 10 repeats)

Spearman above the diagonal, Kendall tau-b below it.

|  | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| **seed 0** |  | 0.53 | 0.56 | 0.77 | 0.18 |
| **seed 1** | 0.42 |  | 0.41 | 0.65 | 0.49 |
| **seed 2** | 0.38 | 0.24 |  | 0.56 | 0.45 |
| **seed 3** | 0.69 | 0.56 | 0.33 |  | 0.37 |
| **seed 4** | 0.16 | 0.38 | 0.33 | 0.29 |  |

### Effect of the number of repeats on the agreement of two fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| 10 repeats, first 5 seeds, repeats 0..9 | 0.50 [0.18, 0.77] | 0.38 [0.16, 0.69] | 0.53 | 10 |
| 10 repeats, all 20 seeds, all disjoint repeat blocks (95 % bootstrap interval of the Spearman mean: 0.46 to 0.61) | 0.53 [-0.04, 0.95] | 0.41 [-0.07, 0.87] | 0.53 | 190 |

### The same design with other groups of 5 fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| seeds 0-4, repeats 0..9 | 0.50 [0.18, 0.77] | 0.38 [0.16, 0.69] | 0.53 | 10 |
| seeds 5-9, repeats 0..9 | 0.54 [0.21, 0.72] | 0.40 [0.16, 0.56] | 0.50 | 10 |
| seeds 10-14, repeats 0..9 | 0.51 [0.25, 0.89] | 0.40 [0.16, 0.78] | 0.43 | 10 |
| seeds 15-19, repeats 0..9 | 0.62 [0.26, 0.81] | 0.48 [0.29, 0.64] | 0.57 | 10 |

### Two independent studies that each pool several fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| 1 seed(s) x 5 folds x 10 repeats = 50 values per itemset | 0.56 [-0.04, 0.95] | 0.43 [-0.02, 0.87] | 0.53 | 300 |
| 2 seed(s) x 5 folds x 10 repeats = 100 values per itemset | 0.66 [0.14, 0.96] | 0.52 [0.07, 0.91] | 0.58 | 300 |
| 3 seed(s) x 5 folds x 10 repeats = 150 values per itemset | 0.73 [0.24, 0.98] | 0.58 [0.20, 0.91] | 0.61 | 300 |
| 5 seed(s) x 5 folds x 10 repeats = 250 values per itemset | 0.82 [0.37, 0.99] | 0.67 [0.24, 0.96] | 0.63 | 300 |
| 10 seed(s) x 5 folds x 10 repeats = 500 values per itemset | 0.86 [0.50, 0.98] | 0.72 [0.38, 0.91] | 0.58 | 300 |

### Noise of one seed mean against the spread between itemsets

| repeats | std of a seed mean (noise) | the same without the seed's common shift (ranking noise) | std between itemset means (signal) | signal / ranking noise | median gap between neighbouring itemsets |
|---|---|---|---|---|---|
| 10 | 0.00755 | 0.00135 | 0.00203 | 1.50 | 0.00037 |

| std between repeats (same fold) | std between fold means (same seed) | std between seed means | values > 0 | values = 0 | values < 0 |
|---|---|---|---|---|---|
| 0.00475 | 0.01574 | 0.00755 | 100.0 % | 0.0 % | 0.0 % |

## All settings: mean over all fold seeds and repeats

| id | itemset | fixed_test: mean ± s.e. | rank | fixed_train: mean ± s.e. | rank | faithful_train: mean ± s.e. | rank |
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

### Do two settings rank the itemsets alike?

| settings | Spearman | Kendall tau-b | top-3 overlap |
|---|---|---|---|
| fixed_test vs fixed_train | 0.78 | 0.64 | 1.00 |
| fixed_test vs faithful_train | 0.39 | 0.20 | 0.33 |
| fixed_train vs faithful_train | 0.44 | 0.29 | 0.33 |
