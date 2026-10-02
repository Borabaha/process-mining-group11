# C10 generated tables (written by experiments/run_c10.py)

## f1

1130 cases after the rare-activity filter (1140 unfiltered), 164 activities, share of label 1 = 0.402, 10 activity sets (Apriori took 0.1 s).

**Substring vs exact matching on the activity-set features:**

| log version | cases | cells | cells differ | cases differ | 0/1 presence differs (cells) | max count gap | sets affected |
|---|---|---|---|---|---|---|---|
| filtered (frq_threshold=2) | 1130 | 11300 | 1 | 1 | 1 | 2 | {370407, ac370000}: 1 cases |
| unfiltered (frq_threshold=1) | 1140 | 11400 | 1 | 1 | 1 | 2 | {370407, ac370000}: 1 cases |

Activity names that are substrings of another activity name (whole unfiltered alphabet): [('370407', '370407c')]

**Feature redundancy (exact matching, filtered log):**

| encoding | columns | distinct columns | min pairwise r | median pairwise r | identical column pairs |
|---|---|---|---|---|---|
| count | 10 | 8 | 0.822 | 0.940 | {ac370000, ac370419, ac370443} = {ac370419, ac370443}; {370407, ac370000, ac370419} = {370407, ac370419} |
| binary | 10 | 8 | 0.870 | 0.954 | {ac370000, ac370419, ac370443} = {ac370419, ac370443}; {370407, ac370000, ac370419} = {370407, ac370419} |

**Model quality (mean over 5 folds; the activity-set columns are the only features):**

| model | test_accuracy | test_f1_weighted | test_share_predicted_1 | train_accuracy | train_f1_weighted |
|---|---|---|---|---|---|
| count encoding | 0.7336 | 0.7359 | 0.5177 | 0.7478 | 0.7501 |
| binary encoding | 0.7088 | 0.7101 | 0.5655 | 0.7184 | 0.7197 |
| always predict label 0 | 0.5982 | 0.4478 | 0.0000 | nan | nan |

**Mean importance and rank, scored on the train fold (fold seed 2023):**

| activity set | support | count/acc | count/f1w | binary/acc | binary/f1w | rank count/acc | rank count/f1w | rank binary/acc | rank binary/f1w |
|---|---|---|---|---|---|---|---|---|---|
| {370407, ac370000} | 0.5912 | 0.0566 | 0.0632 | 0.0277 | 0.0308 | 3 | 3 | 4 | 4 |
| {ac370000, ac370419} | 0.5770 | 0.0724 | 0.0832 | 0.0522 | 0.0592 | 2 | 2 | 3 | 2 |
| {ac370000, ac370443} | 0.5637 | 0.0298 | 0.0348 | 0.0565 | 0.0585 | 4 | 4 | 2 | 3 |
| {ac370000, ac370419, ac370443} | 0.5619 | 0.0010 | 0.0010 | 0.0003 | 0.0003 | 8 | 8 | 8 | 8 |
| {ac370419, ac370443} | 0.5619 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 9 | 9 | 9 | 9 |
| {ac370000, ac370442} | 0.5593 | 0.1097 | 0.1276 | 0.1084 | 0.1340 | 1 | 1 | 1 | 1 |
| {370407, ac370000, ac370419} | 0.5593 | 0.0019 | 0.0019 | 0.0006 | 0.0004 | 5 | 5 | 5 | 6 |
| {370407, ac370419} | 0.5593 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 9 | 9 | 9 | 9 |
| {ac370442, ac370443} | 0.5566 | 0.0013 | 0.0012 | 0.0005 | 0.0005 | 7 | 7 | 6 | 5 |
| {ac370000, ac370419, ac370442} | 0.5566 | 0.0015 | 0.0017 | 0.0004 | 0.0004 | 6 | 6 | 7 | 7 |

**Spearman rho between the four combinations (train fold):**

| rho | count/acc | count/f1w | binary/acc | binary/f1w |
|---|---|---|---|---|
| count/acc | 1.000 | 1.000 | 0.951 | 0.951 |
| count/f1w | 1.000 | 1.000 | 0.951 | 0.951 |
| binary/acc | 0.951 | 0.951 | 1.000 | 0.976 |
| binary/f1w | 0.951 | 0.951 | 0.976 | 1.000 |

**Mean importance and rank, scored on the test fold (fold seed 2023):**

| activity set | support | count/acc | count/f1w | binary/acc | binary/f1w | rank count/acc | rank count/f1w | rank binary/acc | rank binary/f1w |
|---|---|---|---|---|---|---|---|---|---|
| {370407, ac370000} | 0.5912 | 0.0516 | 0.0613 | 0.0237 | 0.0300 | 3 | 3 | 4 | 4 |
| {ac370000, ac370419} | 0.5770 | 0.0691 | 0.0810 | 0.0526 | 0.0602 | 2 | 2 | 3 | 3 |
| {ac370000, ac370443} | 0.5637 | 0.0259 | 0.0302 | 0.0597 | 0.0616 | 4 | 4 | 2 | 2 |
| {ac370000, ac370419, ac370443} | 0.5619 | 0.0004 | 0.0005 | -0.0013 | -0.0012 | 7 | 7 | 10 | 10 |
| {ac370419, ac370443} | 0.5619 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8 | 8 | 7 | 7 |
| {ac370000, ac370442} | 0.5593 | 0.1043 | 0.1224 | 0.1045 | 0.1295 | 1 | 1 | 1 | 1 |
| {370407, ac370000, ac370419} | 0.5593 | 0.0008 | 0.0010 | 0.0001 | 0.0000 | 6 | 6 | 6 | 6 |
| {370407, ac370419} | 0.5593 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8 | 8 | 7 | 7 |
| {ac370442, ac370443} | 0.5566 | -0.0037 | -0.0037 | -0.0011 | -0.0011 | 10 | 10 | 9 | 9 |
| {ac370000, ac370419, ac370442} | 0.5566 | 0.0013 | 0.0014 | 0.0003 | 0.0004 | 5 | 5 | 5 | 5 |

**Spearman rho between the four combinations (test fold):**

| rho | count/acc | count/f1w | binary/acc | binary/f1w |
|---|---|---|---|---|
| count/acc | 1.000 | 1.000 | 0.890 | 0.890 |
| count/f1w | 1.000 | 1.000 | 0.890 | 0.890 |
| binary/acc | 0.890 | 0.890 | 1.000 | 1.000 |
| binary/f1w | 0.890 | 0.890 | 1.000 | 1.000 |

**Spearman rho between train-fold and test-fold importances:**

| index | count/acc | count/f1w | binary/acc | binary/f1w |
|---|---|---|---|---|
| rho(train, test) | 0.915 | 0.915 | 0.841 | 0.793 |

**Faithful configuration vs project default (Spearman rho = 0.951):**

| activity set | faithful (substring count, acc, train, unfiltered) | rank faithful | default (exact count, f1w, test, filtered) | rank default |
|---|---|---|---|---|
| {370407, ac370000} | 0.0368 | 3 | 0.0613 | 3 |
| {ac370000, ac370419} | 0.0722 | 2 | 0.0810 | 2 |
| {ac370000, ac370443} | 0.0189 | 4 | 0.0302 | 4 |
| {ac370000, ac370419, ac370443} | 0.0020 | 6 | 0.0005 | 7 |
| {ac370419, ac370443} | 0.0000 | 9 | 0.0000 | 8 |
| {ac370000, ac370442} | 0.0777 | 1 | 0.1224 | 1 |
| {370407, ac370000, ac370419} | 0.0017 | 7 | 0.0010 | 6 |
| {370407, ac370419} | 0.0000 | 9 | 0.0000 | 8 |
| {ac370442, ac370443} | 0.0007 | 8 | -0.0037 | 10 |
| {ac370000, ac370419, ac370442} | 0.0136 | 5 | 0.0014 | 5 |

**Fold-seed stability (test fold, seeds [2023, 0, 1, 2, 3]): pairwise Spearman rho between seeds, per combination:**

| combination | min rho | median rho | max rho |
|---|---|---|---|
| count/acc | 0.634 | 0.817 | 0.915 |
| count/f1w | 0.732 | 0.823 | 0.951 |
| binary/acc | 0.732 | 0.806 | 0.915 |
| binary/f1w | 0.732 | 0.805 | 0.963 |

**Mean importance averaged over the 5 seeds, and rank:**

| activity set | count/acc | count/f1w | binary/acc | binary/f1w | rank count/acc | rank count/f1w | rank binary/acc | rank binary/f1w |
|---|---|---|---|---|---|---|---|---|
| {370407, ac370000} | 0.0531 | 0.0621 | 0.0308 | 0.0364 | 3 | 3 | 3 | 3 |
| {ac370000, ac370419} | 0.0838 | 0.0935 | 0.0608 | 0.0676 | 1 | 1 | 2 | 2 |
| {ac370000, ac370443} | 0.0258 | 0.0280 | 0.0229 | 0.0253 | 4 | 4 | 4 | 4 |
| {ac370000, ac370419, ac370443} | 0.0015 | 0.0022 | -0.0014 | -0.0014 | 6 | 6 | 10 | 10 |
| {ac370419, ac370443} | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8 | 8 | 6 | 6 |
| {ac370000, ac370442} | 0.0737 | 0.0865 | 0.0754 | 0.0930 | 2 | 2 | 1 | 1 |
| {370407, ac370000, ac370419} | 0.0001 | 0.0001 | -0.0008 | -0.0009 | 7 | 7 | 9 | 9 |
| {370407, ac370419} | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8 | 8 | 6 | 6 |
| {ac370442, ac370443} | -0.0033 | -0.0034 | -0.0008 | -0.0008 | 10 | 10 | 8 | 8 |
| {ac370000, ac370419, ac370442} | 0.0017 | 0.0025 | 0.0016 | 0.0022 | 5 | 5 | 5 | 5 |

**Spearman rho between the four combinations on the seed-averaged importances:**

| rho | count/acc | count/f1w | binary/acc | binary/f1w |
|---|---|---|---|---|
| count/acc | 1.000 | 1.000 | 0.793 | 0.793 |
| count/f1w | 1.000 | 1.000 | 0.793 | 0.793 |
| binary/acc | 0.793 | 0.793 | 1.000 | 1.000 |
| binary/f1w | 0.793 | 0.793 | 1.000 | 1.000 |

**Default combination (count/f1w) per seed:**

| activity set | seed 2023 | seed 0 | seed 1 | seed 2 | seed 3 |
|---|---|---|---|---|---|
| {370407, ac370000} | 0.0613 | 0.0579 | 0.0650 | 0.0561 | 0.0703 |
| {ac370000, ac370419} | 0.0810 | 0.1139 | 0.0885 | 0.0772 | 0.1068 |
| {ac370000, ac370443} | 0.0302 | 0.0170 | 0.0600 | 0.0148 | 0.0177 |
| {ac370000, ac370419, ac370443} | 0.0005 | 0.0149 | -0.0021 | -0.0003 | -0.0018 |
| {ac370419, ac370443} | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| {ac370000, ac370442} | 0.1224 | 0.0787 | 0.0679 | 0.0420 | 0.1213 |
| {370407, ac370000, ac370419} | 0.0010 | -0.0009 | 0.0017 | 0.0006 | -0.0018 |
| {370407, ac370419} | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| {ac370442, ac370443} | -0.0037 | -0.0044 | -0.0016 | -0.0028 | -0.0043 |
| {ac370000, ac370419, ac370442} | 0.0014 | 0.0161 | -0.0016 | -0.0021 | -0.0013 |

## f2

1130 cases after the rare-activity filter (1140 unfiltered), 207 activities, share of label 1 = 0.784, 10 activity sets (Apriori took 0.2 s).

**Substring vs exact matching on the activity-set features:**

| log version | cases | cells | cells differ | cases differ | 0/1 presence differs (cells) | max count gap | sets affected |
|---|---|---|---|---|---|---|---|
| filtered (frq_threshold=2) | 1130 | 11300 | 3 | 3 | 0 | 3 | {370407, ac370000}: 3 cases |
| unfiltered (frq_threshold=1) | 1140 | 11400 | 6 | 4 | 3 | 3 | {370407, ac370000}: 4 cases; {370407, ac370000, ac379999}: 1 cases; {370407, ac379999}: 1 cases |

Activity names that are substrings of another activity name (whole unfiltered alphabet): [('370407', '370407c')]

**Feature redundancy (exact matching, filtered log):**

| encoding | columns | distinct columns | min pairwise r | median pairwise r | identical column pairs |
|---|---|---|---|---|---|
| count | 10 | 8 | 0.417 | 0.596 | {ac370000, ac379999, ac419100} = {ac379999, ac419100}; {370407, ac370000, ac379999} = {370407, ac379999} |
| binary | 10 | 8 | 0.688 | 0.808 | {ac370000, ac379999, ac419100} = {ac379999, ac419100}; {370407, ac370000, ac379999} = {370407, ac379999} |

**Model quality (mean over 5 folds; the activity-set columns are the only features):**

| model | test_accuracy | test_f1_weighted | test_share_predicted_1 | train_accuracy | train_f1_weighted |
|---|---|---|---|---|---|
| count encoding | 0.7646 | 0.7180 | 0.9221 | 0.8058 | 0.7659 |
| binary encoding | 0.7770 | 0.6873 | 0.9912 | 0.7861 | 0.7000 |
| always predict label 1 | 0.7841 | 0.6892 | 1.0000 | nan | nan |

**Mean importance and rank, scored on the train fold (fold seed 2023):**

| activity set | support | count/acc | count/f1w | binary/acc | binary/f1w | rank count/acc | rank count/f1w | rank binary/acc | rank binary/f1w |
|---|---|---|---|---|---|---|---|---|---|
| {ac370000, ac379999} | 0.6903 | 0.0241 | 0.0290 | 0.0020 | 0.0014 | 2 | 2 | 4 | 5 |
| {ac370000, ac419100} | 0.6469 | 0.0208 | 0.0240 | 0.0019 | 0.0035 | 4 | 5 | 5 | 2 |
| {ac370000, ac379999, ac419100} | 0.6407 | 0.0109 | 0.0220 | 0.0001 | 0.0005 | 7 | 6 | 8 | 6 |
| {ac379999, ac419100} | 0.6407 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 9 | 9 | 9 | 8 |
| {370407, ac370000} | 0.6018 | 0.0298 | 0.0289 | 0.0156 | 0.0005 | 1 | 3 | 2 | 7 |
| {370407, ac370000, ac379999} | 0.5929 | 0.0197 | 0.0043 | 0.0018 | 0.0028 | 5 | 8 | 6 | 3 |
| {370407, ac379999} | 0.5929 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 9 | 9 | 9 | 8 |
| {ac370000, ac370419} | 0.5858 | 0.0120 | 0.0247 | 0.0198 | -0.0004 | 6 | 4 | 1 | 10 |
| {ac370000, ac370443} | 0.5770 | 0.0231 | 0.0302 | 0.0124 | 0.0056 | 3 | 1 | 3 | 1 |
| {ac370419, ac379999} | 0.5761 | 0.0092 | 0.0133 | 0.0003 | 0.0021 | 8 | 7 | 7 | 4 |

**Spearman rho between the four combinations (train fold):**

| rho | count/acc | count/f1w | binary/acc | binary/f1w |
|---|---|---|---|---|
| count/acc | 1.000 | 0.854 | 0.793 | 0.439 |
| count/f1w | 0.854 | 1.000 | 0.841 | 0.354 |
| binary/acc | 0.793 | 0.841 | 1.000 | 0.122 |
| binary/f1w | 0.439 | 0.354 | 0.122 | 1.000 |

**Mean importance and rank, scored on the test fold (fold seed 2023):**

| activity set | support | count/acc | count/f1w | binary/acc | binary/f1w | rank count/acc | rank count/f1w | rank binary/acc | rank binary/f1w |
|---|---|---|---|---|---|---|---|---|---|
| {ac370000, ac379999} | 0.6903 | 0.0077 | 0.0061 | -0.0004 | -0.0029 | 3 | 5 | 8 | 8 |
| {ac370000, ac419100} | 0.6469 | 0.0041 | 0.0033 | -0.0009 | -0.0012 | 5 | 7 | 10 | 6 |
| {ac370000, ac379999, ac419100} | 0.6407 | -0.0027 | 0.0095 | -0.0007 | -0.0016 | 10 | 3 | 9 | 7 |
| {ac379999, ac419100} | 0.6407 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8 | 8 | 5 | 4 |
| {370407, ac370000} | 0.6018 | 0.0201 | 0.0135 | 0.0106 | -0.0068 | 1 | 2 | 2 | 9 |
| {370407, ac370000, ac379999} | 0.5929 | 0.0117 | -0.0038 | 0.0010 | 0.0005 | 2 | 10 | 4 | 3 |
| {370407, ac379999} | 0.5929 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8 | 8 | 5 | 4 |
| {ac370000, ac370419} | 0.5858 | 0.0038 | 0.0182 | 0.0156 | -0.0070 | 6 | 1 | 1 | 10 |
| {ac370000, ac370443} | 0.5770 | 0.0016 | 0.0042 | 0.0104 | 0.0013 | 7 | 6 | 3 | 1 |
| {ac370419, ac379999} | 0.5761 | 0.0058 | 0.0064 | -0.0001 | 0.0007 | 4 | 4 | 7 | 2 |

**Spearman rho between the four combinations (test fold):**

| rho | count/acc | count/f1w | binary/acc | binary/f1w |
|---|---|---|---|---|
| count/acc | 1.000 | 0.098 | 0.244 | -0.146 |
| count/f1w | 0.098 | 1.000 | 0.232 | -0.622 |
| binary/acc | 0.244 | 0.232 | 1.000 | -0.110 |
| binary/f1w | -0.146 | -0.622 | -0.110 | 1.000 |

**Spearman rho between train-fold and test-fold importances:**

| index | count/acc | count/f1w | binary/acc | binary/f1w |
|---|---|---|---|---|
| rho(train, test) | 0.671 | 0.561 | 0.524 | 0.598 |

**Faithful configuration vs project default (Spearman rho = 0.305):**

| activity set | faithful (substring count, acc, train, unfiltered) | rank faithful | default (exact count, f1w, test, filtered) | rank default |
|---|---|---|---|---|
| {ac370000, ac379999} | 0.0195 | 4 | 0.0061 | 5 |
| {ac370000, ac419100} | 0.0209 | 3 | 0.0033 | 7 |
| {ac370000, ac379999, ac419100} | 0.0080 | 8 | 0.0095 | 3 |
| {ac379999, ac419100} | 0.0000 | 9 | 0.0000 | 8 |
| {370407, ac370000} | 0.0778 | 1 | 0.0135 | 2 |
| {370407, ac370000, ac379999} | 0.0157 | 6 | -0.0038 | 10 |
| {370407, ac379999} | 0.0000 | 9 | 0.0000 | 8 |
| {ac370000, ac370419} | 0.0137 | 7 | 0.0182 | 1 |
| {ac370000, ac370443} | 0.0244 | 2 | 0.0042 | 6 |
| {ac370419, ac379999} | 0.0188 | 5 | 0.0064 | 4 |

**Fold-seed stability (test fold, seeds [2023, 0, 1, 2, 3]): pairwise Spearman rho between seeds, per combination:**

| combination | min rho | median rho | max rho |
|---|---|---|---|
| count/acc | 0.293 | 0.683 | 0.890 |
| count/f1w | 0.280 | 0.671 | 0.817 |
| binary/acc | 0.055 | 0.556 | 0.880 |
| binary/f1w | -0.240 | 0.358 | 0.646 |

**Mean importance averaged over the 5 seeds, and rank:**

| activity set | count/acc | count/f1w | binary/acc | binary/f1w | rank count/acc | rank count/f1w | rank binary/acc | rank binary/f1w |
|---|---|---|---|---|---|---|---|---|
| {ac370000, ac379999} | 0.0031 | 0.0075 | 0.0007 | -0.0022 | 5 | 5 | 6 | 7 |
| {ac370000, ac419100} | 0.0081 | 0.0094 | -0.0013 | -0.0014 | 3 | 3 | 10 | 6 |
| {ac370000, ac379999, ac419100} | 0.0006 | 0.0074 | -0.0003 | -0.0002 | 8 | 6 | 9 | 4 |
| {ac379999, ac419100} | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 9 | 9 | 7 | 2 |
| {370407, ac370000} | 0.0299 | 0.0130 | 0.0287 | -0.0050 | 1 | 2 | 1 | 10 |
| {370407, ac370000, ac379999} | 0.0134 | 0.0002 | 0.0055 | -0.0023 | 2 | 8 | 2 | 8 |
| {370407, ac379999} | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 9 | 9 | 7 | 2 |
| {ac370000, ac370419} | 0.0062 | 0.0157 | 0.0030 | -0.0029 | 4 | 1 | 4 | 9 |
| {ac370000, ac370443} | 0.0017 | 0.0079 | 0.0040 | 0.0001 | 6 | 4 | 3 | 1 |
| {ac370419, ac379999} | 0.0015 | 0.0015 | 0.0012 | -0.0002 | 7 | 7 | 5 | 5 |

**Spearman rho between the four combinations on the seed-averaged importances:**

| rho | count/acc | count/f1w | binary/acc | binary/f1w |
|---|---|---|---|---|
| count/acc | 1.000 | 0.671 | 0.561 | -0.817 |
| count/f1w | 0.671 | 1.000 | 0.280 | -0.561 |
| binary/acc | 0.561 | 0.280 | 1.000 | -0.451 |
| binary/f1w | -0.817 | -0.561 | -0.451 | 1.000 |

**Default combination (count/f1w) per seed:**

| activity set | seed 2023 | seed 0 | seed 1 | seed 2 | seed 3 |
|---|---|---|---|---|---|
| {ac370000, ac379999} | 0.0061 | 0.0100 | 0.0094 | 0.0075 | 0.0047 |
| {ac370000, ac419100} | 0.0033 | 0.0146 | 0.0113 | 0.0078 | 0.0099 |
| {ac370000, ac379999, ac419100} | 0.0095 | 0.0078 | 0.0091 | 0.0022 | 0.0083 |
| {ac379999, ac419100} | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| {370407, ac370000} | 0.0135 | 0.0078 | 0.0106 | 0.0324 | 0.0009 |
| {370407, ac370000, ac379999} | -0.0038 | 0.0061 | -0.0030 | -0.0003 | 0.0020 |
| {370407, ac379999} | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| {ac370000, ac370419} | 0.0182 | 0.0195 | 0.0166 | 0.0174 | 0.0066 |
| {ac370000, ac370443} | 0.0042 | 0.0097 | 0.0082 | 0.0047 | 0.0125 |
| {ac370419, ac379999} | 0.0064 | 0.0075 | -0.0105 | 0.0034 | 0.0008 |

## f3

1111 cases after the rare-activity filter (1121 unfiltered), 156 activities, share of label 1 = 0.233, 5 activity sets (Apriori took 0.0 s).

**Substring vs exact matching on the activity-set features:**

| log version | cases | cells | cells differ | cases differ | 0/1 presence differs (cells) | max count gap | sets affected |
|---|---|---|---|---|---|---|---|
| filtered (frq_threshold=2) | 1111 | 5555 | 0 | 0 | 0 | 0 | - |
| unfiltered (frq_threshold=1) | 1121 | 5605 | 0 | 0 | 0 | 0 | - |

Activity names that are substrings of another activity name (whole unfiltered alphabet): none

**Feature redundancy (exact matching, filtered log):**

| encoding | columns | distinct columns | min pairwise r | median pairwise r | identical column pairs |
|---|---|---|---|---|---|
| count | 5 | 4 | 0.869 | 0.933 | {370407, ac370000, ac370419} = {370407, ac370419} |
| binary | 5 | 4 | 0.906 | 0.955 | {370407, ac370000, ac370419} = {370407, ac370419} |

**Model quality (mean over 5 folds; the activity-set columns are the only features):**

| model | test_accuracy | test_f1_weighted | test_share_predicted_1 | train_accuracy | train_f1_weighted |
|---|---|---|---|---|---|
| count encoding | 0.7669 | 0.6797 | 0.0162 | 0.7707 | 0.6853 |
| binary encoding | 0.7678 | 0.6758 | 0.0099 | 0.7689 | 0.6772 |
| always predict label 0 | 0.7669 | 0.6657 | 0.0000 | nan | nan |

**Mean importance and rank, scored on the train fold (fold seed 2023):**

| activity set | support | count/acc | count/f1w | binary/acc | binary/f1w | rank count/acc | rank count/f1w | rank binary/acc | rank binary/f1w |
|---|---|---|---|---|---|---|---|---|---|
| {370407, ac370000} | 0.5275 | 0.0415 | -0.0450 | 0.0413 | -0.0513 | 2 | 5 | 2 | 5 |
| {ac370000, ac370419} | 0.5131 | 0.0022 | 0.0038 | 0.0007 | 0.0027 | 3 | 2 | 3 | 2 |
| {370407, ac370000, ac370419} | 0.5023 | 0.0020 | 0.0008 | 0.0000 | 0.0000 | 4 | 3 | 4 | 3 |
| {370407, ac370419} | 0.5023 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5 | 4 | 4 | 3 |
| {ac370000, ac370443} | 0.5014 | 0.0516 | 0.0363 | 0.0895 | 0.0547 | 1 | 1 | 1 | 1 |

**Spearman rho between the four combinations (train fold):**

| rho | count/acc | count/f1w | binary/acc | binary/f1w |
|---|---|---|---|---|
| count/acc | 1.000 | 0.400 | 0.975 | 0.359 |
| count/f1w | 0.400 | 1.000 | 0.359 | 0.975 |
| binary/acc | 0.975 | 0.359 | 1.000 | 0.368 |
| binary/f1w | 0.359 | 0.975 | 0.368 | 1.000 |

**Mean importance and rank, scored on the test fold (fold seed 2023):**

| activity set | support | count/acc | count/f1w | binary/acc | binary/f1w | rank count/acc | rank count/f1w | rank binary/acc | rank binary/f1w |
|---|---|---|---|---|---|---|---|---|---|
| {370407, ac370000} | 0.5275 | 0.0305 | -0.0569 | 0.0329 | -0.0594 | 2 | 5 | 2 | 5 |
| {ac370000, ac370419} | 0.5131 | 0.0016 | 0.0040 | 0.0009 | 0.0033 | 4 | 2 | 3 | 2 |
| {370407, ac370000, ac370419} | 0.5023 | 0.0020 | 0.0004 | 0.0000 | 0.0000 | 3 | 3 | 4 | 3 |
| {370407, ac370419} | 0.5023 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5 | 4 | 4 | 3 |
| {ac370000, ac370443} | 0.5014 | 0.0446 | 0.0298 | 0.0840 | 0.0497 | 1 | 1 | 1 | 1 |

**Spearman rho between the four combinations (test fold):**

| rho | count/acc | count/f1w | binary/acc | binary/f1w |
|---|---|---|---|---|
| count/acc | 1.000 | 0.300 | 0.821 | 0.205 |
| count/f1w | 0.300 | 1.000 | 0.359 | 0.975 |
| binary/acc | 0.821 | 0.359 | 1.000 | 0.368 |
| binary/f1w | 0.205 | 0.975 | 0.368 | 1.000 |

**Spearman rho between train-fold and test-fold importances:**

| index | count/acc | count/f1w | binary/acc | binary/f1w |
|---|---|---|---|---|
| rho(train, test) | 0.900 | 1.000 | 1.000 | 1.000 |

**Faithful configuration vs project default (Spearman rho = 0.000):**

| activity set | faithful (substring count, acc, train, unfiltered) | rank faithful | default (exact count, f1w, test, filtered) | rank default |
|---|---|---|---|---|
| {370407, ac370000} | 0.0341 | 1 | -0.0569 | 5 |
| {ac370000, ac370419} | 0.0020 | 3 | 0.0040 | 2 |
| {370407, ac370000, ac370419} | 0.0007 | 4 | 0.0004 | 3 |
| {370407, ac370419} | 0.0000 | 5 | 0.0000 | 4 |
| {ac370000, ac370443} | 0.0060 | 2 | 0.0298 | 1 |

**Fold-seed stability (test fold, seeds [2023, 0, 1, 2, 3]): pairwise Spearman rho between seeds, per combination:**

| combination | min rho | median rho | max rho |
|---|---|---|---|
| count/acc | 0.600 | 0.700 | 1.000 |
| count/f1w | 0.600 | 0.900 | 1.000 |
| binary/acc | 0.684 | 0.684 | 1.000 |
| binary/f1w | 0.684 | 1.000 | 1.000 |

**Mean importance averaged over the 5 seeds, and rank:**

| activity set | count/acc | count/f1w | binary/acc | binary/f1w | rank count/acc | rank count/f1w | rank binary/acc | rank binary/f1w |
|---|---|---|---|---|---|---|---|---|
| {370407, ac370000} | 0.0355 | -0.0538 | 0.0355 | -0.0498 | 2 | 5 | 2 | 5 |
| {ac370000, ac370419} | -0.0007 | 0.0002 | 0.0004 | 0.0022 | 5 | 2 | 3 | 2 |
| {370407, ac370000, ac370419} | -0.0001 | -0.0014 | 0.0000 | 0.0000 | 4 | 4 | 4 | 3 |
| {370407, ac370419} | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3 | 3 | 4 | 3 |
| {ac370000, ac370443} | 0.0443 | 0.0277 | 0.0778 | 0.0466 | 1 | 1 | 1 | 1 |

**Spearman rho between the four combinations on the seed-averaged importances:**

| rho | count/acc | count/f1w | binary/acc | binary/f1w |
|---|---|---|---|---|
| count/acc | 1.000 | 0.100 | 0.667 | 0.051 |
| count/f1w | 0.100 | 1.000 | 0.359 | 0.975 |
| binary/acc | 0.667 | 0.359 | 1.000 | 0.368 |
| binary/f1w | 0.051 | 0.975 | 0.368 | 1.000 |

**Default combination (count/f1w) per seed:**

| activity set | seed 2023 | seed 0 | seed 1 | seed 2 | seed 3 |
|---|---|---|---|---|---|
| {370407, ac370000} | -0.0569 | -0.0529 | -0.0559 | -0.0509 | -0.0523 |
| {ac370000, ac370419} | 0.0040 | -0.0001 | 0.0017 | -0.0005 | -0.0039 |
| {370407, ac370000, ac370419} | 0.0004 | -0.0022 | -0.0009 | -0.0026 | -0.0017 |
| {370407, ac370419} | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| {ac370000, ac370443} | 0.0298 | 0.0246 | 0.0277 | 0.0294 | 0.0270 |
