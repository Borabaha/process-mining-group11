# C9 overview of all studies

| study | setting | fold seeds | F1 of the scored fold | Spearman of two seeds: 3 / 5 / 10 / 30 repeats | Kendall of two seeds: 3 / 5 / 10 / 30 repeats | Spearman, same folds, 5 repeats | Spearman, two groups of 5 seeds x 10 repeats | ranking noise (10 repeats) | spread between itemsets |
|---|---|---|---|---|---|---|---|---|---|
| f1_original_k5 | fixed_test | 20 | 0.9028 | 0.41 / 0.50 / 0.60 / 0.66 | 0.31 / 0.37 / 0.46 / 0.51 | 0.67 | 0.86 | 0.00170 | 0.00219 |
| f1_original_k5 | fixed_train | 20 | 0.9936 | 0.94 / 0.95 / 0.96 / 0.97 | 0.84 / 0.86 / 0.88 / 0.91 | 0.96 | 0.98 | 0.00075 | 0.00704 |
| f1_original_k5 | faithful_train | 20 | 0.9936 | - / - / 0.53 / - | - / - / 0.41 / - | - | 0.82 | 0.00135 | 0.00203 |
| f1_original_k5_first10 | fixed_test | 10 | 0.9031 | 0.46 / 0.58 / 0.68 / 0.75 | 0.34 / 0.43 / 0.51 / 0.58 | 0.68 | 0.90 | 0.00152 | 0.00230 |
| f1_original_k5_first10 | fixed_train | 10 | 0.9938 | 0.94 / 0.95 / 0.96 / 0.97 | 0.84 / 0.87 / 0.89 / 0.91 | 0.96 | 0.97 | 0.00075 | 0.00718 |
| f1_original_k5_first10 | faithful_train | 10 | 0.9938 | - / - / 0.52 / - | - / - / 0.39 / - | - | 0.77 | 0.00141 | 0.00220 |
| f1_original_k10 | fixed_test | 10 | 0.9084 | 0.42 / 0.51 / 0.62 / - | 0.31 / 0.38 / 0.46 / - | 0.66 | 0.87 | 0.00151 | 0.00212 |
| f1_original_k10 | fixed_train | 10 | 0.9926 | 0.97 / 0.98 / 0.98 / - | 0.92 / 0.94 / 0.95 / - | 0.98 | 1.00 | 0.00040 | 0.00659 |
| f2_original_k5 | fixed_test | 10 | 0.8910 | 0.85 / 0.89 / 0.93 / - | 0.70 / 0.76 / 0.83 / - | 0.90 | 0.98 | 0.00255 | 0.01180 |
| f2_original_k5 | fixed_train | 10 | 0.9939 | 0.94 / 0.96 / 0.97 / - | 0.85 / 0.90 / 0.91 / - | 0.97 | 1.00 | 0.00125 | 0.01562 |
| f2_original_k5 | faithful_train | 10 | 0.9939 | - / - / 0.96 / - | - / - / 0.89 / - | - | 0.99 | 0.00192 | 0.01375 |
| f3_original_k5 | fixed_test | 10 | 0.9313 | 0.70 / 0.72 / 0.77 / - | 0.55 / 0.58 / 0.64 / - | 0.81 | 0.93 | 0.00200 | 0.00581 |
| f3_original_k5 | fixed_train | 10 | 0.9939 | 0.96 / 0.97 / 0.98 / - | 0.89 / 0.91 / 0.94 / - | 0.97 | 0.99 | 0.00107 | 0.01044 |
| f3_original_k5 | faithful_train | 10 | 0.9939 | - / - / 0.83 / - | - / - / 0.68 / - | - | 0.93 | 0.00223 | 0.01390 |
