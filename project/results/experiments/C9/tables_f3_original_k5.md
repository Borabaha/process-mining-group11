# C9 tables: f3, original itemsets, 5 folds

## Setting `fixed_test`

### Weighted F1 of the scored (test) fold, per fold seed

| fold seed | mean over folds | std over folds | lowest fold | highest fold |
|---|---|---|---|---|
| 0 | 0.9314 | 0.0101 | 0.9165 | 0.9411 |
| 1 | 0.9297 | 0.0218 | 0.9020 | 0.9540 |
| 2 | 0.9317 | 0.0169 | 0.9119 | 0.9499 |
| 3 | 0.9282 | 0.0212 | 0.9021 | 0.9596 |
| 4 | 0.9274 | 0.0160 | 0.9055 | 0.9448 |
| all listed | 0.9297 | 0.0019 (between seeds) | 0.9020 | 0.9596 |

### Mean importance per itemset and fold seed (first 5 fold seeds, 5 folds x 5 repeats)

| id | itemset | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | std between seeds |
|---|---|---|---|---|---|---|---|---|
| 0 | 370407, ac370000 | 0.0116 | 0.0181 | 0.0063 | 0.0088 | 0.0150 | 0.0120 | 0.0047 |
| 1 | ac370000, ac370419 | 0.0111 | 0.0243 | 0.0081 | 0.0102 | 0.0191 | 0.0146 | 0.0069 |
| 2 | 370407, ac370000, ac370419 | 0.0131 | 0.0192 | 0.0081 | 0.0142 | 0.0164 | 0.0142 | 0.0041 |
| 3 | 370407, ac370419 | 0.0006 | 0.0048 | 0.0045 | 0.0022 | -0.0011 | 0.0022 | 0.0025 |
| 4 | ac370000, ac370443 | 0.0090 | 0.0153 | 0.0063 | 0.0099 | 0.0083 | 0.0097 | 0.0034 |
| 5 | ac370000, ac370419, ac370443 | 0.0139 | 0.0192 | 0.0090 | 0.0121 | 0.0145 | 0.0138 | 0.0037 |
| 6 | ac370419, ac370443 | -0.0020 | 0.0013 | 0.0032 | -0.0023 | -0.0027 | -0.0005 | 0.0026 |
| 7 | ac370000, ac370442 | 0.0111 | 0.0150 | 0.0101 | 0.0088 | 0.0110 | 0.0112 | 0.0023 |
| 8 | ac370000, ac370442, ac370443 | 0.0116 | 0.0161 | 0.0115 | 0.0077 | 0.0118 | 0.0117 | 0.0030 |
| 9 | ac370442, ac370443 | -0.0022 | -0.0019 | 0.0011 | -0.0019 | -0.0028 | -0.0015 | 0.0015 |

### Rank per itemset and fold seed (first 5 fold seeds, 5 folds x 5 repeats; 1 = most important)

| id | itemset | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | range | rank of the mean over these seeds |
|---|---|---|---|---|---|---|---|---|
| 0 | 370407, ac370000 | 4 | 4 | 6 | 6 | 3 | 3-6 | 4 |
| 1 | ac370000, ac370419 | 6 | 1 | 5 | 3 | 1 | 1-6 | 1 |
| 2 | 370407, ac370000, ac370419 | 2 | 3 | 4 | 1 | 2 | 1-4 | 2 |
| 3 | 370407, ac370419 | 8 | 8 | 8 | 8 | 8 | 8-8 | 8 |
| 4 | ac370000, ac370443 | 7 | 6 | 7 | 4 | 7 | 4-7 | 7 |
| 5 | ac370000, ac370419, ac370443 | 1 | 2 | 3 | 2 | 4 | 1-4 | 3 |
| 6 | ac370419, ac370443 | 9 | 9 | 9 | 10 | 9 | 9-10 | 9 |
| 7 | ac370000, ac370442 | 5 | 7 | 2 | 5 | 6 | 2-7 | 6 |
| 8 | ac370000, ac370442, ac370443 | 3 | 5 | 1 | 7 | 5 | 1-7 | 5 |
| 9 | ac370442, ac370443 | 10 | 10 | 10 | 9 | 10 | 9-10 | 10 |

### Agreement of the rankings of two fold seeds (first 5 fold seeds, 5 folds x 5 repeats)

Spearman above the diagonal, Kendall tau-b below it.

|  | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| **seed 0** |  | 0.78 | 0.84 | 0.75 | 0.76 |
| **seed 1** | 0.69 |  | 0.61 | 0.84 | 0.95 |
| **seed 2** | 0.73 | 0.51 |  | 0.58 | 0.62 |
| **seed 3** | 0.51 | 0.64 | 0.42 |  | 0.79 |
| **seed 4** | 0.64 | 0.87 | 0.47 | 0.60 |  |

### Effect of the number of repeats on the agreement of two fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| 3 repeats, first 5 seeds, repeats 0..2 | 0.76 [0.59, 0.98] | 0.61 [0.42, 0.91] | 0.53 | 10 |
| 3 repeats, all 10 seeds, all disjoint repeat blocks (95 % bootstrap interval of the Spearman mean: 0.65 to 0.75) | 0.70 [0.35, 0.98] | 0.55 [0.16, 0.91] | 0.50 | 135 |
| 5 repeats, first 5 seeds, repeats 0..4 | 0.75 [0.58, 0.95] | 0.61 [0.42, 0.87] | 0.53 | 10 |
| 5 repeats, all 10 seeds, all disjoint repeat blocks (95 % bootstrap interval of the Spearman mean: 0.68 to 0.78) | 0.72 [0.48, 0.95] | 0.58 [0.29, 0.87] | 0.54 | 90 |
| 10 repeats, first 5 seeds, repeats 0..9 | 0.78 [0.55, 0.96] | 0.67 [0.42, 0.91] | 0.53 | 10 |
| 10 repeats, all 10 seeds, all disjoint repeat blocks (95 % bootstrap interval of the Spearman mean: 0.71 to 0.83) | 0.77 [0.55, 0.96] | 0.64 [0.42, 0.91] | 0.56 | 45 |

### The same design with other groups of 5 fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| seeds 0-4, repeats 0..4 | 0.75 [0.58, 0.95] | 0.61 [0.42, 0.87] | 0.53 | 10 |
| seeds 0-4, repeats 0..9 | 0.78 [0.55, 0.96] | 0.67 [0.42, 0.91] | 0.53 | 10 |
| seeds 5-9, repeats 0..4 | 0.71 [0.59, 0.93] | 0.56 [0.42, 0.82] | 0.57 | 10 |
| seeds 5-9, repeats 0..9 | 0.74 [0.62, 0.95] | 0.60 [0.47, 0.87] | 0.57 | 10 |

### Same folds and models, only the permutations differ

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| 3 repeats | 0.76 [0.41, 0.96] | 0.62 [0.24, 0.87] | 0.62 | 30 |
| 5 repeats | 0.81 [0.49, 0.95] | 0.68 [0.33, 0.87] | 0.60 | 10 |

### Two independent studies that each pool several fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| 1 seed(s) x 5 folds x 5 repeats = 25 values per itemset | 0.71 [0.48, 0.95] | 0.57 [0.29, 0.87] | 0.52 | 300 |
| 2 seed(s) x 5 folds x 5 repeats = 50 values per itemset | 0.77 [0.44, 0.99] | 0.64 [0.33, 0.96] | 0.57 | 300 |
| 3 seed(s) x 5 folds x 5 repeats = 75 values per itemset | 0.82 [0.56, 0.99] | 0.70 [0.42, 0.96] | 0.61 | 300 |
| 5 seed(s) x 5 folds x 5 repeats = 125 values per itemset | 0.88 [0.66, 0.99] | 0.78 [0.56, 0.96] | 0.67 | 300 |
| 1 seed(s) x 5 folds x 10 repeats = 50 values per itemset | 0.77 [0.55, 0.96] | 0.64 [0.42, 0.91] | 0.56 | 300 |
| 2 seed(s) x 5 folds x 10 repeats = 100 values per itemset | 0.83 [0.60, 1.00] | 0.71 [0.47, 1.00] | 0.64 | 300 |
| 3 seed(s) x 5 folds x 10 repeats = 150 values per itemset | 0.87 [0.65, 0.99] | 0.77 [0.51, 0.96] | 0.65 | 300 |
| 5 seed(s) x 5 folds x 10 repeats = 250 values per itemset | 0.93 [0.75, 1.00] | 0.86 [0.64, 1.00] | 0.67 | 300 |

### Noise of one seed mean against the spread between itemsets

| repeats | std of a seed mean (noise) | the same without the seed's common shift (ranking noise) | std between itemset means (signal) | signal / ranking noise | median gap between neighbouring itemsets |
|---|---|---|---|---|---|
| 3 | 0.00349 | 0.00251 | 0.00581 | 2.31 | 0.00090 |
| 5 | 0.00329 | 0.00220 | 0.00581 | 2.64 | 0.00090 |
| 10 | 0.00315 | 0.00200 | 0.00581 | 2.91 | 0.00090 |

| std between repeats (same fold) | std between fold means (same seed) | std between seed means | values > 0 | values = 0 | values < 0 |
|---|---|---|---|---|---|
| 0.00712 | 0.00971 | 0.00315 | 71.1 % | 6.7 % | 22.2 % |

## Setting `fixed_train`

### Weighted F1 of the scored (train) fold, per fold seed

| fold seed | mean over folds | std over folds | lowest fold | highest fold |
|---|---|---|---|---|
| 0 | 0.9946 | 0.0019 | 0.9921 | 0.9966 |
| 1 | 0.9930 | 0.0012 | 0.9921 | 0.9944 |
| 2 | 0.9939 | 0.0015 | 0.9921 | 0.9955 |
| 3 | 0.9943 | 0.0011 | 0.9932 | 0.9955 |
| 4 | 0.9939 | 0.0010 | 0.9932 | 0.9955 |
| all listed | 0.9939 | 0.0006 (between seeds) | 0.9921 | 0.9966 |

### Mean importance per itemset and fold seed (first 5 fold seeds, 5 folds x 5 repeats)

| id | itemset | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | std between seeds |
|---|---|---|---|---|---|---|---|---|
| 0 | 370407, ac370000 | 0.0405 | 0.0376 | 0.0361 | 0.0377 | 0.0396 | 0.0383 | 0.0018 |
| 1 | ac370000, ac370419 | 0.0390 | 0.0376 | 0.0352 | 0.0373 | 0.0396 | 0.0377 | 0.0017 |
| 2 | 370407, ac370000, ac370419 | 0.0389 | 0.0398 | 0.0347 | 0.0426 | 0.0392 | 0.0391 | 0.0028 |
| 3 | 370407, ac370419 | 0.0179 | 0.0175 | 0.0193 | 0.0193 | 0.0191 | 0.0186 | 0.0009 |
| 4 | ac370000, ac370443 | 0.0307 | 0.0294 | 0.0299 | 0.0317 | 0.0290 | 0.0301 | 0.0011 |
| 5 | ac370000, ac370419, ac370443 | 0.0348 | 0.0344 | 0.0323 | 0.0351 | 0.0353 | 0.0344 | 0.0012 |
| 6 | ac370419, ac370443 | 0.0134 | 0.0138 | 0.0129 | 0.0143 | 0.0127 | 0.0134 | 0.0006 |
| 7 | ac370000, ac370442 | 0.0316 | 0.0311 | 0.0291 | 0.0311 | 0.0309 | 0.0308 | 0.0009 |
| 8 | ac370000, ac370442, ac370443 | 0.0331 | 0.0308 | 0.0304 | 0.0323 | 0.0332 | 0.0320 | 0.0013 |
| 9 | ac370442, ac370443 | 0.0112 | 0.0096 | 0.0100 | 0.0100 | 0.0097 | 0.0101 | 0.0006 |

### Rank per itemset and fold seed (first 5 fold seeds, 5 folds x 5 repeats; 1 = most important)

| id | itemset | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | range | rank of the mean over these seeds |
|---|---|---|---|---|---|---|---|---|
| 0 | 370407, ac370000 | 1 | 3 | 1 | 2 | 1 | 1-3 | 2 |
| 1 | ac370000, ac370419 | 2 | 2 | 2 | 3 | 2 | 2-3 | 3 |
| 2 | 370407, ac370000, ac370419 | 3 | 1 | 3 | 1 | 3 | 1-3 | 1 |
| 3 | 370407, ac370419 | 8 | 8 | 8 | 8 | 8 | 8-8 | 8 |
| 4 | ac370000, ac370443 | 7 | 7 | 6 | 6 | 7 | 6-7 | 7 |
| 5 | ac370000, ac370419, ac370443 | 4 | 4 | 4 | 4 | 4 | 4-4 | 4 |
| 6 | ac370419, ac370443 | 9 | 9 | 9 | 9 | 9 | 9-9 | 9 |
| 7 | ac370000, ac370442 | 6 | 5 | 7 | 7 | 6 | 5-7 | 6 |
| 8 | ac370000, ac370442, ac370443 | 5 | 6 | 5 | 5 | 5 | 5-6 | 5 |
| 9 | ac370442, ac370443 | 10 | 10 | 10 | 10 | 10 | 10-10 | 10 |

### Agreement of the rankings of two fold seeds (first 5 fold seeds, 5 folds x 5 repeats)

Spearman above the diagonal, Kendall tau-b below it.

|  | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| **seed 0** |  | 0.94 | 0.99 | 0.95 | 1.00 |
| **seed 1** | 0.82 |  | 0.92 | 0.95 | 0.94 |
| **seed 2** | 0.96 | 0.78 |  | 0.96 | 0.99 |
| **seed 3** | 0.87 | 0.87 | 0.91 |  | 0.95 |
| **seed 4** | 1.00 | 0.82 | 0.96 | 0.87 |  |

### Effect of the number of repeats on the agreement of two fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| 3 repeats, first 5 seeds, repeats 0..2 | 0.97 [0.95, 0.99] | 0.92 [0.87, 0.96] | 1.00 | 10 |
| 3 repeats, all 10 seeds, all disjoint repeat blocks (95 % bootstrap interval of the Spearman mean: 0.95 to 0.97) | 0.96 [0.87, 1.00] | 0.89 [0.69, 1.00] | 0.96 | 135 |
| 5 repeats, first 5 seeds, repeats 0..4 | 0.96 [0.92, 1.00] | 0.88 [0.78, 1.00] | 1.00 | 10 |
| 5 repeats, all 10 seeds, all disjoint repeat blocks (95 % bootstrap interval of the Spearman mean: 0.96 to 0.98) | 0.97 [0.90, 1.00] | 0.91 [0.78, 1.00] | 0.97 | 90 |
| 10 repeats, first 5 seeds, repeats 0..9 | 0.99 [0.98, 1.00] | 0.95 [0.91, 1.00] | 1.00 | 10 |
| 10 repeats, all 10 seeds, all disjoint repeat blocks (95 % bootstrap interval of the Spearman mean: 0.98 to 0.99) | 0.98 [0.96, 1.00] | 0.94 [0.87, 1.00] | 1.00 | 45 |

### The same design with other groups of 5 fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| seeds 0-4, repeats 0..4 | 0.96 [0.92, 1.00] | 0.88 [0.78, 1.00] | 1.00 | 10 |
| seeds 0-4, repeats 0..9 | 0.99 [0.98, 1.00] | 0.95 [0.91, 1.00] | 1.00 | 10 |
| seeds 5-9, repeats 0..4 | 0.96 [0.93, 0.99] | 0.89 [0.82, 0.96] | 1.00 | 10 |
| seeds 5-9, repeats 0..9 | 0.98 [0.96, 1.00] | 0.93 [0.87, 1.00] | 1.00 | 10 |

### Same folds and models, only the permutations differ

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| 3 repeats | 0.96 [0.92, 1.00] | 0.90 [0.78, 1.00] | 0.98 | 30 |
| 5 repeats | 0.97 [0.92, 1.00] | 0.92 [0.78, 1.00] | 0.97 | 10 |

### Two independent studies that each pool several fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| 1 seed(s) x 5 folds x 5 repeats = 25 values per itemset | 0.97 [0.90, 1.00] | 0.91 [0.78, 1.00] | 0.96 | 300 |
| 2 seed(s) x 5 folds x 5 repeats = 50 values per itemset | 0.98 [0.93, 1.00] | 0.94 [0.82, 1.00] | 1.00 | 300 |
| 3 seed(s) x 5 folds x 5 repeats = 75 values per itemset | 0.99 [0.95, 1.00] | 0.96 [0.87, 1.00] | 1.00 | 300 |
| 5 seed(s) x 5 folds x 5 repeats = 125 values per itemset | 0.99 [0.98, 1.00] | 0.96 [0.91, 1.00] | 1.00 | 300 |
| 1 seed(s) x 5 folds x 10 repeats = 50 values per itemset | 0.98 [0.96, 1.00] | 0.94 [0.87, 1.00] | 1.00 | 300 |
| 2 seed(s) x 5 folds x 10 repeats = 100 values per itemset | 0.99 [0.96, 1.00] | 0.95 [0.87, 1.00] | 1.00 | 300 |
| 3 seed(s) x 5 folds x 10 repeats = 150 values per itemset | 0.99 [0.98, 1.00] | 0.96 [0.91, 1.00] | 1.00 | 300 |
| 5 seed(s) x 5 folds x 10 repeats = 250 values per itemset | 0.99 [0.98, 1.00] | 0.95 [0.91, 1.00] | 1.00 | 300 |

### Noise of one seed mean against the spread between itemsets

| repeats | std of a seed mean (noise) | the same without the seed's common shift (ranking noise) | std between itemset means (signal) | signal / ranking noise | median gap between neighbouring itemsets |
|---|---|---|---|---|---|
| 3 | 0.00158 | 0.00130 | 0.01044 | 8.06 | 0.00200 |
| 5 | 0.00145 | 0.00115 | 0.01044 | 9.10 | 0.00200 |
| 10 | 0.00137 | 0.00107 | 0.01044 | 9.79 | 0.00200 |

| std between repeats (same fold) | std between fold means (same seed) | std between seed means | values > 0 | values = 0 | values < 0 |
|---|---|---|---|---|---|
| 0.00353 | 0.00508 | 0.00137 | 100.0 % | 0.0 % | 0.0 % |

## Setting `faithful_train`

### Weighted F1 of the scored (train) fold, per fold seed

| fold seed | mean over folds | std over folds | lowest fold | highest fold |
|---|---|---|---|---|
| 0 | 0.9946 | 0.0019 | 0.9921 | 0.9966 |
| 1 | 0.9930 | 0.0012 | 0.9921 | 0.9944 |
| 2 | 0.9939 | 0.0015 | 0.9921 | 0.9955 |
| 3 | 0.9943 | 0.0011 | 0.9932 | 0.9955 |
| 4 | 0.9939 | 0.0010 | 0.9932 | 0.9955 |
| all listed | 0.9939 | 0.0006 (between seeds) | 0.9921 | 0.9966 |

### Mean importance per itemset and fold seed (first 5 fold seeds, 5 folds x 10 repeats)

| id | itemset | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | std between seeds |
|---|---|---|---|---|---|---|---|---|
| 0 | 370407, ac370000 | 0.2129 | 0.2145 | 0.2105 | 0.2161 | 0.2127 | 0.2133 | 0.0021 |
| 1 | ac370000, ac370419 | 0.2635 | 0.2649 | 0.2596 | 0.2637 | 0.2608 | 0.2625 | 0.0022 |
| 2 | 370407, ac370000, ac370419 | 0.2485 | 0.2553 | 0.2504 | 0.2545 | 0.2505 | 0.2519 | 0.0029 |
| 3 | 370407, ac370419 | 0.2539 | 0.2564 | 0.2444 | 0.2534 | 0.2478 | 0.2512 | 0.0049 |
| 4 | ac370000, ac370443 | 0.2578 | 0.2621 | 0.2582 | 0.2652 | 0.2587 | 0.2604 | 0.0032 |
| 5 | ac370000, ac370419, ac370443 | 0.2520 | 0.2590 | 0.2535 | 0.2589 | 0.2528 | 0.2553 | 0.0034 |
| 6 | ac370419, ac370443 | 0.2502 | 0.2573 | 0.2511 | 0.2582 | 0.2478 | 0.2529 | 0.0046 |
| 7 | ac370000, ac370442 | 0.2600 | 0.2624 | 0.2578 | 0.2645 | 0.2571 | 0.2604 | 0.0031 |
| 8 | ac370000, ac370442, ac370443 | 0.2547 | 0.2571 | 0.2530 | 0.2604 | 0.2549 | 0.2560 | 0.0029 |
| 9 | ac370442, ac370443 | 0.2563 | 0.2561 | 0.2499 | 0.2566 | 0.2541 | 0.2546 | 0.0028 |

### Rank per itemset and fold seed (first 5 fold seeds, 5 folds x 10 repeats; 1 = most important)

| id | itemset | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | range | rank of the mean over these seeds |
|---|---|---|---|---|---|---|---|---|
| 0 | 370407, ac370000 | 10 | 10 | 10 | 10 | 10 | 10-10 | 10 |
| 1 | ac370000, ac370419 | 1 | 1 | 1 | 3 | 1 | 1-3 | 1 |
| 2 | 370407, ac370000, ac370419 | 9 | 9 | 7 | 8 | 7 | 7-9 | 8 |
| 3 | 370407, ac370419 | 6 | 7 | 9 | 9 | 9 | 6-9 | 9 |
| 4 | ac370000, ac370443 | 3 | 3 | 2 | 1 | 2 | 1-3 | 2 |
| 5 | ac370000, ac370419, ac370443 | 7 | 4 | 4 | 5 | 6 | 4-7 | 5 |
| 6 | ac370419, ac370443 | 8 | 5 | 6 | 6 | 8 | 5-8 | 7 |
| 7 | ac370000, ac370442 | 2 | 2 | 3 | 2 | 3 | 2-3 | 3 |
| 8 | ac370000, ac370442, ac370443 | 5 | 6 | 5 | 4 | 4 | 4-6 | 4 |
| 9 | ac370442, ac370443 | 4 | 8 | 8 | 7 | 5 | 4-8 | 6 |

### Agreement of the rankings of two fold seeds (first 5 fold seeds, 5 folds x 10 repeats)

Spearman above the diagonal, Kendall tau-b below it.

|  | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| **seed 0** |  | 0.78 | 0.73 | 0.78 | 0.89 |
| **seed 1** | 0.64 |  | 0.93 | 0.88 | 0.78 |
| **seed 2** | 0.60 | 0.78 |  | 0.94 | 0.89 |
| **seed 3** | 0.60 | 0.69 | 0.82 |  | 0.90 |
| **seed 4** | 0.73 | 0.64 | 0.78 | 0.78 |  |

### Effect of the number of repeats on the agreement of two fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| 10 repeats, first 5 seeds, repeats 0..9 | 0.85 [0.73, 0.94] | 0.71 [0.60, 0.82] | 1.00 | 10 |
| 10 repeats, all 10 seeds, all disjoint repeat blocks (95 % bootstrap interval of the Spearman mean: 0.75 to 0.89) | 0.83 [0.56, 0.98] | 0.68 [0.29, 0.91] | 1.00 | 45 |

### The same design with other groups of 5 fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| seeds 0-4, repeats 0..9 | 0.85 [0.73, 0.94] | 0.71 [0.60, 0.82] | 1.00 | 10 |
| seeds 5-9, repeats 0..9 | 0.79 [0.62, 0.92] | 0.64 [0.47, 0.82] | 1.00 | 10 |

### Two independent studies that each pool several fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| 1 seed(s) x 5 folds x 10 repeats = 50 values per itemset | 0.82 [0.56, 0.98] | 0.68 [0.29, 0.91] | 1.00 | 300 |
| 2 seed(s) x 5 folds x 10 repeats = 100 values per itemset | 0.85 [0.58, 1.00] | 0.73 [0.33, 1.00] | 1.00 | 300 |
| 3 seed(s) x 5 folds x 10 repeats = 150 values per itemset | 0.89 [0.64, 1.00] | 0.78 [0.51, 1.00] | 1.00 | 300 |
| 5 seed(s) x 5 folds x 10 repeats = 250 values per itemset | 0.93 [0.83, 0.99] | 0.82 [0.64, 0.96] | 1.00 | 300 |

### Noise of one seed mean against the spread between itemsets

| repeats | std of a seed mean (noise) | the same without the seed's common shift (ranking noise) | std between itemset means (signal) | signal / ranking noise | median gap between neighbouring itemsets |
|---|---|---|---|---|---|
| 10 | 0.00660 | 0.00223 | 0.01390 | 6.23 | 0.00125 |

| std between repeats (same fold) | std between fold means (same seed) | std between seed means | values > 0 | values = 0 | values < 0 |
|---|---|---|---|---|---|
| 0.01161 | 0.01607 | 0.00660 | 100.0 % | 0.0 % | 0.0 % |

## All settings: mean over all fold seeds and repeats

| id | itemset | fixed_test: mean ± s.e. | rank | fixed_train: mean ± s.e. | rank | faithful_train: mean ± s.e. | rank |
|---|---|---|---|---|---|---|---|
| 0 | 370407, ac370000 | 0.01233 ± 0.00126 | 4 | 0.03870 ± 0.00058 | 2 | 0.21173 ± 0.00115 | 10 |
| 1 | ac370000, ac370419 | 0.01343 ± 0.00155 | 2 | 0.03670 ± 0.00063 | 3 | 0.26095 ± 0.00149 | 1 |
| 2 | 370407, ac370000, ac370419 | 0.01391 ± 0.00113 | 1 | 0.03879 ± 0.00070 | 1 | 0.24982 ± 0.00179 | 8 |
| 3 | 370407, ac370419 | 0.00199 ± 0.00071 | 8 | 0.01869 ± 0.00028 | 8 | 0.24857 ± 0.00206 | 9 |
| 4 | ac370000, ac370443 | 0.01013 ± 0.00108 | 7 | 0.02983 ± 0.00030 | 7 | 0.25801 ± 0.00202 | 3 |
| 5 | ac370000, ac370419, ac370443 | 0.01248 ± 0.00116 | 3 | 0.03423 ± 0.00052 | 4 | 0.25227 ± 0.00259 | 6 |
| 6 | ac370419, ac370443 | 0.00010 ± 0.00064 | 9 | 0.01357 ± 0.00018 | 9 | 0.25095 ± 0.00274 | 7 |
| 7 | ac370000, ac370442 | 0.01093 ± 0.00085 | 6 | 0.03045 ± 0.00049 | 6 | 0.25838 ± 0.00200 | 2 |
| 8 | ac370000, ac370442, ac370443 | 0.01188 ± 0.00106 | 5 | 0.03231 ± 0.00046 | 5 | 0.25262 ± 0.00274 | 4 |
| 9 | ac370442, ac370443 | -0.00080 ± 0.00052 | 10 | 0.01028 ± 0.00021 | 10 | 0.25245 ± 0.00232 | 5 |

### Do two settings rank the itemsets alike?

| settings | Spearman | Kendall tau-b | top-3 overlap |
|---|---|---|---|
| fixed_test vs fixed_train | 0.96 | 0.91 | 0.67 |
| fixed_test vs faithful_train | 0.04 | 0.02 | 0.33 |
| fixed_train vs faithful_train | -0.12 | -0.07 | 0.33 |
