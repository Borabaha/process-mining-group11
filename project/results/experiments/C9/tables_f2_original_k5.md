# C9 tables: f2, original itemsets, 5 folds

## Setting `fixed_test`

### Weighted F1 of the scored (test) fold, per fold seed

| fold seed | mean over folds | std over folds | lowest fold | highest fold |
|---|---|---|---|---|
| 0 | 0.8902 | 0.0269 | 0.8550 | 0.9202 |
| 1 | 0.8921 | 0.0314 | 0.8399 | 0.9143 |
| 2 | 0.8887 | 0.0197 | 0.8628 | 0.9119 |
| 3 | 0.8761 | 0.0290 | 0.8459 | 0.9191 |
| 4 | 0.8882 | 0.0254 | 0.8537 | 0.9184 |
| all listed | 0.8871 | 0.0063 (between seeds) | 0.8399 | 0.9202 |

### Mean importance per itemset and fold seed (first 5 fold seeds, 5 folds x 5 repeats)

| id | itemset | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | std between seeds |
|---|---|---|---|---|---|---|---|---|
| 0 | ac370000, ac379999 | 0.0174 | 0.0227 | 0.0103 | 0.0125 | 0.0170 | 0.0160 | 0.0048 |
| 1 | ac370000, ac419100 | 0.0105 | 0.0177 | 0.0082 | 0.0062 | 0.0151 | 0.0115 | 0.0048 |
| 2 | ac370000, ac379999, ac419100 | 0.0131 | 0.0182 | 0.0134 | 0.0144 | 0.0134 | 0.0145 | 0.0021 |
| 3 | ac379999, ac419100 | 0.0128 | 0.0116 | 0.0112 | 0.0042 | 0.0096 | 0.0099 | 0.0034 |
| 4 | 370407, ac370000 | 0.0352 | 0.0508 | 0.0348 | 0.0397 | 0.0340 | 0.0389 | 0.0070 |
| 5 | 370407, ac370000, ac379999 | 0.0324 | 0.0451 | 0.0264 | 0.0285 | 0.0262 | 0.0317 | 0.0079 |
| 6 | 370407, ac379999 | 0.0076 | 0.0127 | 0.0021 | -0.0020 | 0.0081 | 0.0057 | 0.0057 |
| 7 | ac370000, ac370419 | 0.0392 | 0.0432 | 0.0271 | 0.0317 | 0.0339 | 0.0350 | 0.0063 |
| 8 | ac370000, ac370443 | 0.0242 | 0.0392 | 0.0232 | 0.0277 | 0.0237 | 0.0276 | 0.0067 |
| 9 | ac370419, ac379999 | 0.0113 | 0.0162 | 0.0042 | 0.0050 | 0.0032 | 0.0080 | 0.0056 |

### Rank per itemset and fold seed (first 5 fold seeds, 5 folds x 5 repeats; 1 = most important)

| id | itemset | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | range | rank of the mean over these seeds |
|---|---|---|---|---|---|---|---|---|
| 0 | ac370000, ac379999 | 5 | 5 | 7 | 6 | 5 | 5-7 | 5 |
| 1 | ac370000, ac419100 | 9 | 7 | 8 | 7 | 6 | 6-9 | 7 |
| 2 | ac370000, ac379999, ac419100 | 6 | 6 | 5 | 5 | 7 | 5-7 | 6 |
| 3 | ac379999, ac419100 | 7 | 10 | 6 | 9 | 8 | 6-10 | 8 |
| 4 | 370407, ac370000 | 2 | 1 | 1 | 1 | 1 | 1-2 | 1 |
| 5 | 370407, ac370000, ac379999 | 3 | 2 | 3 | 3 | 3 | 2-3 | 3 |
| 6 | 370407, ac379999 | 10 | 9 | 10 | 10 | 9 | 9-10 | 10 |
| 7 | ac370000, ac370419 | 1 | 3 | 2 | 2 | 2 | 1-3 | 2 |
| 8 | ac370000, ac370443 | 4 | 4 | 4 | 4 | 4 | 4-4 | 4 |
| 9 | ac370419, ac379999 | 8 | 8 | 9 | 8 | 10 | 8-10 | 9 |

### Agreement of the rankings of two fold seeds (first 5 fold seeds, 5 folds x 5 repeats)

Spearman above the diagonal, Kendall tau-b below it.

|  | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| **seed 0** |  | 0.88 | 0.94 | 0.93 | 0.89 |
| **seed 1** | 0.73 |  | 0.84 | 0.96 | 0.93 |
| **seed 2** | 0.82 | 0.73 |  | 0.93 | 0.89 |
| **seed 3** | 0.78 | 0.87 | 0.87 |  | 0.93 |
| **seed 4** | 0.78 | 0.78 | 0.78 | 0.82 |  |

### Effect of the number of repeats on the agreement of two fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| 3 repeats, first 5 seeds, repeats 0..2 | 0.87 [0.78, 0.99] | 0.75 [0.60, 0.96] | 0.80 | 10 |
| 3 repeats, all 10 seeds, all disjoint repeat blocks (95 % bootstrap interval of the Spearman mean: 0.83 to 0.87) | 0.85 [0.67, 1.00] | 0.70 [0.47, 1.00] | 0.82 | 135 |
| 5 repeats, first 5 seeds, repeats 0..4 | 0.91 [0.84, 0.96] | 0.80 [0.73, 0.87] | 1.00 | 10 |
| 5 repeats, all 10 seeds, all disjoint repeat blocks (95 % bootstrap interval of the Spearman mean: 0.87 to 0.91) | 0.89 [0.73, 1.00] | 0.76 [0.51, 1.00] | 0.86 | 90 |
| 10 repeats, first 5 seeds, repeats 0..9 | 0.95 [0.90, 0.99] | 0.87 [0.78, 0.96] | 1.00 | 10 |
| 10 repeats, all 10 seeds, all disjoint repeat blocks (95 % bootstrap interval of the Spearman mean: 0.91 to 0.96) | 0.93 [0.87, 0.99] | 0.83 [0.69, 0.96] | 0.88 | 45 |

### The same design with other groups of 5 fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| seeds 0-4, repeats 0..4 | 0.91 [0.84, 0.96] | 0.80 [0.73, 0.87] | 1.00 | 10 |
| seeds 0-4, repeats 0..9 | 0.95 [0.90, 0.99] | 0.87 [0.78, 0.96] | 1.00 | 10 |
| seeds 5-9, repeats 0..4 | 0.91 [0.83, 1.00] | 0.80 [0.64, 1.00] | 0.87 | 10 |
| seeds 5-9, repeats 0..9 | 0.93 [0.89, 0.99] | 0.82 [0.73, 0.96] | 0.80 | 10 |

### Same folds and models, only the permutations differ

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| 3 repeats | 0.86 [0.70, 0.98] | 0.71 [0.56, 0.91] | 0.83 | 30 |
| 5 repeats | 0.90 [0.82, 0.96] | 0.76 [0.64, 0.91] | 0.80 | 10 |

### Two independent studies that each pool several fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| 1 seed(s) x 5 folds x 5 repeats = 25 values per itemset | 0.89 [0.73, 1.00] | 0.75 [0.51, 1.00] | 0.84 | 300 |
| 2 seed(s) x 5 folds x 5 repeats = 50 values per itemset | 0.93 [0.75, 1.00] | 0.81 [0.56, 1.00] | 0.88 | 300 |
| 3 seed(s) x 5 folds x 5 repeats = 75 values per itemset | 0.95 [0.81, 1.00] | 0.87 [0.60, 1.00] | 0.93 | 300 |
| 5 seed(s) x 5 folds x 5 repeats = 125 values per itemset | 0.97 [0.90, 1.00] | 0.90 [0.73, 1.00] | 0.98 | 300 |
| 1 seed(s) x 5 folds x 10 repeats = 50 values per itemset | 0.94 [0.87, 0.99] | 0.83 [0.69, 0.96] | 0.88 | 300 |
| 2 seed(s) x 5 folds x 10 repeats = 100 values per itemset | 0.96 [0.89, 1.00] | 0.89 [0.78, 1.00] | 0.96 | 300 |
| 3 seed(s) x 5 folds x 10 repeats = 150 values per itemset | 0.97 [0.89, 1.00] | 0.92 [0.78, 1.00] | 0.99 | 300 |
| 5 seed(s) x 5 folds x 10 repeats = 250 values per itemset | 0.98 [0.96, 1.00] | 0.95 [0.91, 1.00] | 1.00 | 300 |

### Noise of one seed mean against the spread between itemsets

| repeats | std of a seed mean (noise) | the same without the seed's common shift (ranking noise) | std between itemset means (signal) | signal / ranking noise | median gap between neighbouring itemsets |
|---|---|---|---|---|---|
| 3 | 0.00603 | 0.00386 | 0.01180 | 3.06 | 0.00232 |
| 5 | 0.00563 | 0.00318 | 0.01180 | 3.71 | 0.00232 |
| 10 | 0.00529 | 0.00255 | 0.01180 | 4.63 | 0.00232 |

| std between repeats (same fold) | std between fold means (same seed) | std between seed means | values > 0 | values = 0 | values < 0 |
|---|---|---|---|---|---|
| 0.01351 | 0.01231 | 0.00529 | 85.7 % | 1.1 % | 13.2 % |

## Setting `fixed_train`

### Weighted F1 of the scored (train) fold, per fold seed

| fold seed | mean over folds | std over folds | lowest fold | highest fold |
|---|---|---|---|---|
| 0 | 0.9947 | 0.0020 | 0.9911 | 0.9956 |
| 1 | 0.9935 | 0.0018 | 0.9922 | 0.9967 |
| 2 | 0.9942 | 0.0022 | 0.9911 | 0.9967 |
| 3 | 0.9940 | 0.0017 | 0.9911 | 0.9956 |
| 4 | 0.9938 | 0.0030 | 0.9900 | 0.9978 |
| all listed | 0.9940 | 0.0004 (between seeds) | 0.9900 | 0.9978 |

### Mean importance per itemset and fold seed (first 5 fold seeds, 5 folds x 5 repeats)

| id | itemset | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | std between seeds |
|---|---|---|---|---|---|---|---|---|
| 0 | ac370000, ac379999 | 0.0894 | 0.0909 | 0.0896 | 0.0954 | 0.0935 | 0.0918 | 0.0026 |
| 1 | ac370000, ac419100 | 0.0791 | 0.0781 | 0.0790 | 0.0824 | 0.0815 | 0.0800 | 0.0018 |
| 2 | ac370000, ac379999, ac419100 | 0.0841 | 0.0880 | 0.0862 | 0.0888 | 0.0898 | 0.0874 | 0.0023 |
| 3 | ac379999, ac419100 | 0.0632 | 0.0640 | 0.0673 | 0.0693 | 0.0656 | 0.0659 | 0.0025 |
| 4 | 370407, ac370000 | 0.0918 | 0.0896 | 0.0899 | 0.0925 | 0.0966 | 0.0921 | 0.0028 |
| 5 | 370407, ac370000, ac379999 | 0.0931 | 0.0942 | 0.0978 | 0.0971 | 0.0978 | 0.0960 | 0.0022 |
| 6 | 370407, ac379999 | 0.0525 | 0.0566 | 0.0547 | 0.0573 | 0.0558 | 0.0554 | 0.0019 |
| 7 | ac370000, ac370419 | 0.0926 | 0.0928 | 0.0912 | 0.0962 | 0.0923 | 0.0930 | 0.0019 |
| 8 | ac370000, ac370443 | 0.0846 | 0.0874 | 0.0847 | 0.0902 | 0.0901 | 0.0874 | 0.0028 |
| 9 | ac370419, ac379999 | 0.0545 | 0.0584 | 0.0574 | 0.0612 | 0.0582 | 0.0580 | 0.0024 |

### Rank per itemset and fold seed (first 5 fold seeds, 5 folds x 5 repeats; 1 = most important)

| id | itemset | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | range | rank of the mean over these seeds |
|---|---|---|---|---|---|---|---|---|
| 0 | ac370000, ac379999 | 4 | 3 | 4 | 3 | 3 | 3-4 | 4 |
| 1 | ac370000, ac419100 | 7 | 7 | 7 | 7 | 7 | 7-7 | 7 |
| 2 | ac370000, ac379999, ac419100 | 6 | 5 | 5 | 6 | 6 | 5-6 | 6 |
| 3 | ac379999, ac419100 | 8 | 8 | 8 | 8 | 8 | 8-8 | 8 |
| 4 | 370407, ac370000 | 3 | 4 | 3 | 4 | 2 | 2-4 | 3 |
| 5 | 370407, ac370000, ac379999 | 1 | 1 | 1 | 1 | 1 | 1-1 | 1 |
| 6 | 370407, ac379999 | 10 | 10 | 10 | 10 | 10 | 10-10 | 10 |
| 7 | ac370000, ac370419 | 2 | 2 | 2 | 2 | 4 | 2-4 | 2 |
| 8 | ac370000, ac370443 | 5 | 6 | 6 | 5 | 5 | 5-6 | 5 |
| 9 | ac370419, ac379999 | 9 | 9 | 9 | 9 | 9 | 9-9 | 9 |

### Agreement of the rankings of two fold seeds (first 5 fold seeds, 5 folds x 5 repeats)

Spearman above the diagonal, Kendall tau-b below it.

|  | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| **seed 0** |  | 0.98 | 0.99 | 0.99 | 0.96 |
| **seed 1** | 0.91 |  | 0.99 | 0.99 | 0.94 |
| **seed 2** | 0.96 | 0.96 |  | 0.98 | 0.95 |
| **seed 3** | 0.96 | 0.96 | 0.91 |  | 0.95 |
| **seed 4** | 0.91 | 0.82 | 0.87 | 0.87 |  |

### Effect of the number of repeats on the agreement of two fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| 3 repeats, first 5 seeds, repeats 0..2 | 0.93 [0.84, 0.98] | 0.85 [0.73, 0.91] | 0.77 | 10 |
| 3 repeats, all 10 seeds, all disjoint repeat blocks (95 % bootstrap interval of the Spearman mean: 0.92 to 0.95) | 0.94 [0.79, 1.00] | 0.85 [0.69, 1.00] | 0.81 | 135 |
| 5 repeats, first 5 seeds, repeats 0..4 | 0.97 [0.94, 0.99] | 0.91 [0.82, 0.96] | 0.73 | 10 |
| 5 repeats, all 10 seeds, all disjoint repeat blocks (95 % bootstrap interval of the Spearman mean: 0.96 to 0.97) | 0.96 [0.90, 1.00] | 0.90 [0.78, 1.00] | 0.82 | 90 |
| 10 repeats, first 5 seeds, repeats 0..9 | 0.98 [0.95, 1.00] | 0.93 [0.87, 1.00] | 0.87 | 10 |
| 10 repeats, all 10 seeds, all disjoint repeat blocks (95 % bootstrap interval of the Spearman mean: 0.95 to 0.98) | 0.97 [0.90, 1.00] | 0.91 [0.82, 1.00] | 0.83 | 45 |

### The same design with other groups of 5 fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| seeds 0-4, repeats 0..4 | 0.97 [0.94, 0.99] | 0.91 [0.82, 0.96] | 0.73 | 10 |
| seeds 0-4, repeats 0..9 | 0.98 [0.95, 1.00] | 0.93 [0.87, 1.00] | 0.87 | 10 |
| seeds 5-9, repeats 0..4 | 0.96 [0.92, 0.99] | 0.90 [0.82, 0.96] | 0.87 | 10 |
| seeds 5-9, repeats 0..9 | 0.97 [0.95, 1.00] | 0.91 [0.87, 1.00] | 0.80 | 10 |

### Same folds and models, only the permutations differ

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| 3 repeats | 0.95 [0.85, 1.00] | 0.87 [0.73, 1.00] | 0.83 | 30 |
| 5 repeats | 0.97 [0.95, 1.00] | 0.92 [0.87, 1.00] | 0.87 | 10 |

### Two independent studies that each pool several fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| 1 seed(s) x 5 folds x 5 repeats = 25 values per itemset | 0.96 [0.90, 1.00] | 0.90 [0.78, 1.00] | 0.81 | 300 |
| 2 seed(s) x 5 folds x 5 repeats = 50 values per itemset | 0.98 [0.90, 1.00] | 0.94 [0.82, 1.00] | 0.86 | 300 |
| 3 seed(s) x 5 folds x 5 repeats = 75 values per itemset | 0.98 [0.94, 1.00] | 0.95 [0.82, 1.00] | 0.90 | 300 |
| 5 seed(s) x 5 folds x 5 repeats = 125 values per itemset | 0.99 [0.95, 1.00] | 0.98 [0.87, 1.00] | 0.98 | 300 |
| 1 seed(s) x 5 folds x 10 repeats = 50 values per itemset | 0.97 [0.90, 1.00] | 0.91 [0.82, 1.00] | 0.83 | 300 |
| 2 seed(s) x 5 folds x 10 repeats = 100 values per itemset | 0.98 [0.94, 1.00] | 0.95 [0.82, 1.00] | 0.89 | 300 |
| 3 seed(s) x 5 folds x 10 repeats = 150 values per itemset | 0.99 [0.95, 1.00] | 0.98 [0.87, 1.00] | 0.95 | 300 |
| 5 seed(s) x 5 folds x 10 repeats = 250 values per itemset | 1.00 [0.99, 1.00] | 1.00 [0.96, 1.00] | 0.99 | 300 |

### Noise of one seed mean against the spread between itemsets

| repeats | std of a seed mean (noise) | the same without the seed's common shift (ranking noise) | std between itemset means (signal) | signal / ranking noise | median gap between neighbouring itemsets |
|---|---|---|---|---|---|
| 3 | 0.00224 | 0.00182 | 0.01562 | 8.59 | 0.00206 |
| 5 | 0.00194 | 0.00151 | 0.01562 | 10.33 | 0.00206 |
| 10 | 0.00170 | 0.00125 | 0.01562 | 12.48 | 0.00206 |

| std between repeats (same fold) | std between fold means (same seed) | std between seed means | values > 0 | values = 0 | values < 0 |
|---|---|---|---|---|---|
| 0.00671 | 0.00540 | 0.00170 | 100.0 % | 0.0 % | 0.0 % |

## Setting `faithful_train`

### Weighted F1 of the scored (train) fold, per fold seed

| fold seed | mean over folds | std over folds | lowest fold | highest fold |
|---|---|---|---|---|
| 0 | 0.9947 | 0.0020 | 0.9911 | 0.9956 |
| 1 | 0.9935 | 0.0018 | 0.9922 | 0.9967 |
| 2 | 0.9942 | 0.0022 | 0.9911 | 0.9967 |
| 3 | 0.9940 | 0.0017 | 0.9911 | 0.9956 |
| 4 | 0.9938 | 0.0030 | 0.9900 | 0.9978 |
| all listed | 0.9940 | 0.0004 (between seeds) | 0.9900 | 0.9978 |

### Mean importance per itemset and fold seed (first 5 fold seeds, 5 folds x 10 repeats)

| id | itemset | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | std between seeds |
|---|---|---|---|---|---|---|---|---|
| 0 | ac370000, ac379999 | 0.1278 | 0.1303 | 0.1338 | 0.1386 | 0.1358 | 0.1332 | 0.0043 |
| 1 | ac370000, ac419100 | 0.1378 | 0.1393 | 0.1407 | 0.1448 | 0.1425 | 0.1410 | 0.0027 |
| 2 | ac370000, ac379999, ac419100 | 0.1456 | 0.1435 | 0.1447 | 0.1503 | 0.1500 | 0.1468 | 0.0032 |
| 3 | ac379999, ac419100 | 0.1347 | 0.1326 | 0.1368 | 0.1430 | 0.1391 | 0.1372 | 0.0040 |
| 4 | 370407, ac370000 | 0.1405 | 0.1400 | 0.1430 | 0.1408 | 0.1437 | 0.1416 | 0.0016 |
| 5 | 370407, ac370000, ac379999 | 0.1487 | 0.1425 | 0.1523 | 0.1491 | 0.1543 | 0.1494 | 0.0045 |
| 6 | 370407, ac379999 | 0.1450 | 0.1408 | 0.1465 | 0.1459 | 0.1503 | 0.1457 | 0.0034 |
| 7 | ac370000, ac370419 | 0.1627 | 0.1586 | 0.1630 | 0.1593 | 0.1636 | 0.1614 | 0.0023 |
| 8 | ac370000, ac370443 | 0.1751 | 0.1699 | 0.1711 | 0.1712 | 0.1738 | 0.1722 | 0.0021 |
| 9 | ac370419, ac379999 | 0.1725 | 0.1689 | 0.1738 | 0.1697 | 0.1732 | 0.1716 | 0.0022 |

### Rank per itemset and fold seed (first 5 fold seeds, 5 folds x 10 repeats; 1 = most important)

| id | itemset | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | range | rank of the mean over these seeds |
|---|---|---|---|---|---|---|---|---|
| 0 | ac370000, ac379999 | 10 | 10 | 10 | 10 | 10 | 10-10 | 10 |
| 1 | ac370000, ac419100 | 8 | 8 | 8 | 7 | 8 | 7-8 | 8 |
| 2 | ac370000, ac379999, ac419100 | 5 | 4 | 6 | 4 | 6 | 4-6 | 5 |
| 3 | ac379999, ac419100 | 9 | 9 | 9 | 8 | 9 | 8-9 | 9 |
| 4 | 370407, ac370000 | 7 | 7 | 7 | 9 | 7 | 7-9 | 7 |
| 5 | 370407, ac370000, ac379999 | 4 | 5 | 4 | 5 | 4 | 4-5 | 4 |
| 6 | 370407, ac379999 | 6 | 6 | 5 | 6 | 5 | 5-6 | 6 |
| 7 | ac370000, ac370419 | 3 | 3 | 3 | 3 | 3 | 3-3 | 3 |
| 8 | ac370000, ac370443 | 1 | 1 | 2 | 1 | 1 | 1-2 | 1 |
| 9 | ac370419, ac379999 | 2 | 2 | 1 | 2 | 2 | 1-2 | 2 |

### Agreement of the rankings of two fold seeds (first 5 fold seeds, 5 folds x 10 repeats)

Spearman above the diagonal, Kendall tau-b below it.

|  | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| **seed 0** |  | 0.99 | 0.98 | 0.95 | 0.99 |
| **seed 1** | 0.96 |  | 0.95 | 0.96 | 0.96 |
| **seed 2** | 0.91 | 0.87 |  | 0.92 | 0.99 |
| **seed 3** | 0.87 | 0.91 | 0.78 |  | 0.93 |
| **seed 4** | 0.96 | 0.91 | 0.96 | 0.82 |  |

### Effect of the number of repeats on the agreement of two fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| 10 repeats, first 5 seeds, repeats 0..9 | 0.96 [0.92, 0.99] | 0.89 [0.78, 0.96] | 1.00 | 10 |
| 10 repeats, all 10 seeds, all disjoint repeat blocks (95 % bootstrap interval of the Spearman mean: 0.95 to 0.98) | 0.96 [0.89, 1.00] | 0.89 [0.78, 1.00] | 1.00 | 45 |

### The same design with other groups of 5 fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| seeds 0-4, repeats 0..9 | 0.96 [0.92, 0.99] | 0.89 [0.78, 0.96] | 1.00 | 10 |
| seeds 5-9, repeats 0..9 | 0.96 [0.94, 1.00] | 0.89 [0.82, 1.00] | 1.00 | 10 |

### Two independent studies that each pool several fold seeds

|  | Spearman mean [min, max] | Kendall tau-b mean [min, max] | top-3 overlap | comparisons |
|---|---|---|---|---|
| 1 seed(s) x 5 folds x 10 repeats = 50 values per itemset | 0.96 [0.89, 1.00] | 0.88 [0.78, 1.00] | 1.00 | 300 |
| 2 seed(s) x 5 folds x 10 repeats = 100 values per itemset | 0.98 [0.93, 1.00] | 0.94 [0.82, 1.00] | 1.00 | 300 |
| 3 seed(s) x 5 folds x 10 repeats = 150 values per itemset | 0.99 [0.96, 1.00] | 0.95 [0.87, 1.00] | 1.00 | 300 |
| 5 seed(s) x 5 folds x 10 repeats = 250 values per itemset | 0.99 [0.98, 1.00] | 0.98 [0.91, 1.00] | 1.00 | 300 |

### Noise of one seed mean against the spread between itemsets

| repeats | std of a seed mean (noise) | the same without the seed's common shift (ranking noise) | std between itemset means (signal) | signal / ranking noise | median gap between neighbouring itemsets |
|---|---|---|---|---|---|
| 10 | 0.00262 | 0.00192 | 0.01375 | 7.17 | 0.00322 |

| std between repeats (same fold) | std between fold means (same seed) | std between seed means | values > 0 | values = 0 | values < 0 |
|---|---|---|---|---|---|
| 0.00874 | 0.00870 | 0.00262 | 100.0 % | 0.0 % | 0.0 % |

## All settings: mean over all fold seeds and repeats

| id | itemset | fixed_test: mean ± s.e. | rank | fixed_train: mean ± s.e. | rank | faithful_train: mean ± s.e. | rank |
|---|---|---|---|---|---|---|---|
| 0 | ac370000, ac379999 | 0.01948 ± 0.00161 | 5 | 0.09130 ± 0.00063 | 4 | 0.13381 ± 0.00099 | 10 |
| 1 | ac370000, ac419100 | 0.01564 ± 0.00199 | 7 | 0.08024 ± 0.00054 | 7 | 0.14059 ± 0.00077 | 8 |
| 2 | ac370000, ac379999, ac419100 | 0.01804 ± 0.00148 | 6 | 0.08730 ± 0.00054 | 6 | 0.14672 ± 0.00069 | 5 |
| 3 | ac379999, ac419100 | 0.01153 ± 0.00125 | 8 | 0.06581 ± 0.00055 | 8 | 0.13737 ± 0.00107 | 9 |
| 4 | 370407, ac370000 | 0.03968 ± 0.00174 | 1 | 0.09444 ± 0.00068 | 2 | 0.14144 ± 0.00062 | 7 |
| 5 | 370407, ac370000, ac379999 | 0.03512 ± 0.00210 | 3 | 0.09650 ± 0.00045 | 1 | 0.14988 ± 0.00097 | 4 |
| 6 | 370407, ac379999 | 0.00975 ± 0.00153 | 10 | 0.05544 ± 0.00050 | 10 | 0.14482 ± 0.00093 | 6 |
| 7 | ac370000, ac370419 | 0.03736 ± 0.00146 | 2 | 0.09285 ± 0.00036 | 3 | 0.16067 ± 0.00078 | 3 |
| 8 | ac370000, ac370443 | 0.03259 ± 0.00171 | 4 | 0.08813 ± 0.00057 | 5 | 0.17254 ± 0.00064 | 1 |
| 9 | ac370419, ac379999 | 0.01150 ± 0.00186 | 9 | 0.05746 ± 0.00053 | 9 | 0.17141 ± 0.00082 | 2 |

### Do two settings rank the itemsets alike?

| settings | Spearman | Kendall tau-b | top-3 overlap |
|---|---|---|---|
| fixed_test vs fixed_train | 0.95 | 0.87 | 1.00 |
| fixed_test vs faithful_train | 0.15 | 0.11 | 0.33 |
| fixed_train vs faithful_train | 0.07 | 0.07 | 0.33 |
