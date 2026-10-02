### Logs

| log | events (file) | cases (file) | cases after rare-activity filter | activities after filter | label=1 cases | Age min-max | attribute values missing |
|---|---|---|---|---|---|---|---|
| f1 | 24176 | 1140 | 1130 | 164 | 454 | 19-99 | 0 |
| f2 | 31235 | 1140 | 1130 | 207 | 886 | 19-99 | 0 |
| f3 | 20534 | 1121 | 1111 | 156 | 259 | 19-99 | 0 |

### Case attributes

| log | attribute | kind | missing events (whole file) | missing cases (table) | distinct values (whole file) | distinct values (table) | max distinct values inside one case | category encoded as 0 | share of cases with code 0 |
|---|---|---|---|---|---|---|---|---|---|
| f1 | Age | numeric | 0 | 0 | 74 | 74 | 1 | - | - |
| f1 | Diagnosis | categorical | 0 | 0 | 105 | 105 | 1 | Adenoca. vagina st II | 0.0009 |
| f1 | Treatment code | categorical | 0 | 0 | 43 | 42 | 1 | TC101 | 0.4044 |
| f1 | Diagnosis code | categorical | 0 | 0 | 11 | 11 | 1 | DC106 | 0.1292 |
| f1 | Specialism code | categorical | 0 | 0 | 3 | 3 | 1 | SC13 | 0.0496 |
| f2 | Age | numeric | 0 | 0 | 74 | 74 | 1 | - | - |
| f2 | Diagnosis | categorical | 0 | 0 | 105 | 104 | 1 | Adenoca. vagina st II | 0.0009 |
| f2 | Treatment code | categorical | 0 | 0 | 43 | 41 | 1 | TC101 | 0.4053 |
| f2 | Diagnosis code | categorical | 0 | 0 | 11 | 11 | 1 | DC106 | 0.1292 |
| f2 | Specialism code | categorical | 0 | 0 | 3 | 3 | 1 | SC13 | 0.0496 |
| f3 | Age | numeric | 0 | 0 | 73 | 73 | 1 | - | - |
| f3 | Diagnosis | categorical | 0 | 0 | 101 | 101 | 1 | Adenoca. vagina st II | 0.0009 |
| f3 | Treatment code | categorical | 0 | 0 | 42 | 42 | 1 | TC101 | 0.4086 |
| f3 | Diagnosis code | categorical | 0 | 0 | 11 | 11 | 1 | DC106 | 0.1278 |
| f3 | Specialism code | categorical | 0 | 0 | 3 | 3 | 1 | SC13 | 0.0495 |

### Tiny pairs

| pair | pdist jaccard, scipy 1.11.4 | pdist jaccard, scipy 1.18.1 | explicit mismatch share (ours) | pdist hamming, scipy 1.11.4 | pdist hamming, scipy 1.18.1 | emulate_jaccard(booleanise=False) | emulate_jaccard(booleanise=True) |
|---|---|---|---|---|---|---|---|
| pair_A [[0,1,2],[0,1,3]] | 0.5 | 0.0 | 0.3333 | 0.3333 | 0.3333 | 0.5 | 0.0 |
| pair_B [[0,1,2],[1,1,3]] | 0.6667 | 0.3333 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.3333 |
| pair_C [[0,1,2],[1,1,2]] | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333 |

### Pair level

| table | quantity | other | mean ours | mean other | share of pairs identical | mean abs diff | max abs diff | Pearson r | Spearman rho |
|---|---|---|---|---|---|---|---|---|---|
| sample200_f1 | categorical part | original code @ scipy 1.11.4 | 0.7464 | 0.7709 | 0.8645 | 0.0244 | 0.5 | 0.9629 | 0.9681 |
| sample200_f1 | categorical part | original code @ scipy 1.18.1 | 0.7464 | 0.2002 | 0.0495 | 0.5465 | 1.0 | 0.3997 | 0.3894 |
| sample200_f1 | combined distance | original code @ scipy 1.11.4 | 0.646 | 0.6655 | 0.8645 | 0.0195 | 0.4 | 0.9648 | 0.9668 |
| sample200_f1 | combined distance | original code @ scipy 1.18.1 | 0.646 | 0.209 | 0.0495 | 0.4372 | 0.8 | 0.4343 | 0.4685 |
| f1 | categorical part | original code @ scipy 1.11.4 | 0.7376 | 0.7641 | 0.8454 | 0.0265 | 0.5 | 0.9651 | 0.9776 |
| f1 | categorical part | original code @ scipy 1.18.1 | 0.7376 | 0.2019 | 0.0494 | 0.5358 | 1.0 | 0.4708 | 0.4727 |
| f1 | combined distance | original code @ scipy 1.11.4 | 0.6353 | 0.6565 | 0.8454 | 0.0212 | 0.4 | 0.9666 | 0.9744 |
| f1 | combined distance | original code @ scipy 1.18.1 | 0.6353 | 0.2068 | 0.0494 | 0.4287 | 0.8 | 0.496 | 0.5403 |
| f2 | categorical part | original code @ scipy 1.11.4 | 0.7377 | 0.7643 | 0.8444 | 0.0267 | 0.5 | 0.9648 | 0.9772 |
| f2 | categorical part | original code @ scipy 1.18.1 | 0.7377 | 0.202 | 0.0489 | 0.5358 | 1.0 | 0.4717 | 0.4736 |
| f2 | combined distance | original code @ scipy 1.11.4 | 0.6354 | 0.6567 | 0.8444 | 0.0213 | 0.4 | 0.9663 | 0.9741 |
| f2 | combined distance | original code @ scipy 1.18.1 | 0.6354 | 0.2068 | 0.0489 | 0.4287 | 0.8 | 0.4973 | 0.5418 |
| f3 | categorical part | original code @ scipy 1.11.4 | 0.7365 | 0.7634 | 0.8427 | 0.027 | 0.5 | 0.9647 | 0.977 |
| f3 | categorical part | original code @ scipy 1.18.1 | 0.7365 | 0.2018 | 0.05 | 0.5348 | 1.0 | 0.471 | 0.473 |
| f3 | combined distance | original code @ scipy 1.11.4 | 0.6346 | 0.6561 | 0.8427 | 0.0216 | 0.4 | 0.9662 | 0.974 |
| f3 | combined distance | original code @ scipy 1.18.1 | 0.6346 | 0.2068 | 0.05 | 0.4279 | 0.8 | 0.4971 | 0.5417 |

### Consistency checks

| table | check (max abs difference, expected 0) | value |
|---|---|---|
| sample200_f1 | ours categorical vs pdist hamming @1.18.1 | 0.00e+00 |
| sample200_f1 | ours categorical vs pdist hamming @1.11.4 | 0.00e+00 |
| sample200_f1 | ours numeric vs original numeric @1.18.1 | 1.11e-16 |
| sample200_f1 | ours numeric vs original numeric @1.11.4 | 1.11e-16 |
| sample200_f1 | emulated 1.11.4 jaccard vs pdist jaccard @1.11.4 | 0.00e+00 |
| sample200_f1 | emulated 1.18.1 jaccard vs pdist jaccard @1.18.1 | 0.00e+00 |
| sample200_f1 | original categorical == pdist jaccard @1.11.4 | 0.00e+00 |
| sample200_f1 | original categorical == pdist jaccard @1.18.1 | 0.00e+00 |
| sample200_f1 | (m*jaccard + ours numeric)/(1+m) vs original combined @1.11.4 | 2.22e-16 |
| sample200_f1 | (m*jaccard + ours numeric)/(1+m) vs original combined @1.18.1 | 1.11e-16 |
| sample200_f1 | 2s/(1+s) vs pdist jaccard on one-hot booleans @1.18.1 | 0.00e+00 |
| f1 | ours categorical vs pdist hamming @1.18.1 | 0.00e+00 |
| f1 | ours categorical vs pdist hamming @1.11.4 | 0.00e+00 |
| f1 | ours numeric vs original numeric @1.18.1 | 1.11e-16 |
| f1 | ours numeric vs original numeric @1.11.4 | 1.11e-16 |
| f1 | emulated 1.11.4 jaccard vs pdist jaccard @1.11.4 | 0.00e+00 |
| f1 | emulated 1.18.1 jaccard vs pdist jaccard @1.18.1 | 0.00e+00 |
| f1 | original categorical == pdist jaccard @1.11.4 | 0.00e+00 |
| f1 | original categorical == pdist jaccard @1.18.1 | 0.00e+00 |
| f1 | (m*jaccard + ours numeric)/(1+m) vs original combined @1.11.4 | 1.11e-16 |
| f1 | (m*jaccard + ours numeric)/(1+m) vs original combined @1.18.1 | 1.11e-16 |
| f1 | 2s/(1+s) vs pdist jaccard on one-hot booleans @1.18.1 | 0.00e+00 |
| f2 | ours categorical vs pdist hamming @1.18.1 | 0.00e+00 |
| f2 | ours categorical vs pdist hamming @1.11.4 | 0.00e+00 |
| f2 | ours numeric vs original numeric @1.18.1 | 1.11e-16 |
| f2 | ours numeric vs original numeric @1.11.4 | 1.11e-16 |
| f2 | emulated 1.11.4 jaccard vs pdist jaccard @1.11.4 | 0.00e+00 |
| f2 | emulated 1.18.1 jaccard vs pdist jaccard @1.18.1 | 0.00e+00 |
| f2 | original categorical == pdist jaccard @1.11.4 | 0.00e+00 |
| f2 | original categorical == pdist jaccard @1.18.1 | 0.00e+00 |
| f2 | (m*jaccard + ours numeric)/(1+m) vs original combined @1.11.4 | 1.11e-16 |
| f2 | (m*jaccard + ours numeric)/(1+m) vs original combined @1.18.1 | 1.11e-16 |
| f2 | 2s/(1+s) vs pdist jaccard on one-hot booleans @1.18.1 | 0.00e+00 |
| f3 | ours categorical vs pdist hamming @1.18.1 | 0.00e+00 |
| f3 | ours categorical vs pdist hamming @1.11.4 | 0.00e+00 |
| f3 | ours numeric vs original numeric @1.18.1 | 1.11e-16 |
| f3 | ours numeric vs original numeric @1.11.4 | 1.11e-16 |
| f3 | emulated 1.11.4 jaccard vs pdist jaccard @1.11.4 | 0.00e+00 |
| f3 | emulated 1.18.1 jaccard vs pdist jaccard @1.18.1 | 0.00e+00 |
| f3 | original categorical == pdist jaccard @1.11.4 | 0.00e+00 |
| f3 | original categorical == pdist jaccard @1.18.1 | 0.00e+00 |
| f3 | (m*jaccard + ours numeric)/(1+m) vs original combined @1.11.4 | 1.11e-16 |
| f3 | (m*jaccard + ours numeric)/(1+m) vs original combined @1.18.1 | 1.11e-16 |
| f3 | 2s/(1+s) vs pdist jaccard on one-hot booleans @1.18.1 | 0.00e+00 |

### Pattern level, 200-case sample

| pattern (activity) | cases with it (of 200) | original function @1.11.4 | pattern_case_distance on the same matrix | original function @1.18.1 | pattern_case_distance on the same matrix  | explicit distance (ours) |
|---|---|---|---|---|---|---|
| ac370000 | 138 | 0.667769 | 0.667769 | 0.257663 | 0.257663 | 0.646427 |
| ac419100 | 135 | 0.685718 | 0.685718 | 0.215526 | 0.215526 | 0.674869 |
| 370407 | 119 | 0.675572 | 0.675572 | 0.244165 | 0.244165 | 0.658744 |
| ac370443 | 115 | 0.677792 | 0.677792 | 0.248813 | 0.248813 | 0.663709 |
| ac370419 | 114 | 0.677295 | 0.677295 | 0.244777 | 0.244777 | 0.662355 |
| ac370442 | 112 | 0.679829 | 0.679829 | 0.247165 | 0.247165 | 0.665935 |
| 370715a | 110 | 0.68081 | 0.68081 | 0.242312 | 0.242312 | 0.666359 |
| 370712b | 108 | 0.682685 | 0.682685 | 0.239985 | 0.239985 | 0.66791 |
| ac370403 | 106 | 0.683082 | 0.683082 | 0.240703 | 0.240703 | 0.669714 |
| ac372417 | 105 | 0.683063 | 0.683063 | 0.241599 | 0.241599 | 0.670144 |
| 302282 | 0 | nan | nan | nan | nan | nan |

### Pattern level, whole logs (length-1 patterns)

| quantity | f1 | f2 | f3 |
|---|---|---|---|
| patterns (activities) | 164 | 207 | 156 |
| patterns in all cases (distance undefined) | 0 | 0 | 0 |
| duplicate objective rows (ours) | 1 | 1 | 1 |
| explicit mismatch share (ours): CD min / median / max | 0.5904 / 0.6617 / 0.7687 | 0.5900 / 0.6662 / 0.7626 | 0.5466 / 0.6640 / 0.7728 |
| explicit mismatch share (ours): front size (distinct=False) | 7 | 7 | 7 |
| explicit mismatch share (ours): front size (distinct=True) | 7 | 7 | 7 |
| original code @ scipy 1.11.4: CD min / median / max | 0.6165 / 0.6749 / 0.7791 | 0.6174 / 0.6787 / 0.7786 | 0.5927 / 0.6777 / 0.7843 |
| original code @ scipy 1.11.4: front size (distinct=False) | 7 | 12 | 7 |
| original code @ scipy 1.11.4: front size (distinct=True) | 7 | 12 | 7 |
| original code @ scipy 1.18.1: CD min / median / max | 0.1511 / 0.2105 / 0.3128 | 0.1504 / 0.2136 / 0.2956 | 0.1514 / 0.2160 / 0.3175 |
| original code @ scipy 1.18.1: front size (distinct=False) | 17 | 33 | 29 |
| original code @ scipy 1.18.1: front size (distinct=True) | 17 | 33 | 28 |
| one-hot Jaccard (alternative reading): CD min / median / max | 0.6654 / 0.7256 / 0.7991 | 0.6649 / 0.7297 / 0.7939 | 0.6319 / 0.7278 / 0.7997 |
| one-hot Jaccard (alternative reading): front size (distinct=False) | 7 | 7 | 6 |
| one-hot Jaccard (alternative reading): front size (distinct=True) | 7 | 7 | 6 |
| ours vs original code @ scipy 1.11.4: Spearman of pattern CD | 0.9497 | 0.9658 | 0.9555 |
| ours vs original code @ scipy 1.11.4: front overlap | 6 shared, Jaccard 0.750 | 7 shared, Jaccard 0.583 | 7 shared, Jaccard 1.000 |
| ours vs original code @ scipy 1.18.1: Spearman of pattern CD | 0.2348 | 0.3297 | 0.3988 |
| ours vs original code @ scipy 1.18.1: front overlap | 3 shared, Jaccard 0.143 | 4 shared, Jaccard 0.111 | 1 shared, Jaccard 0.029 |
| ours vs one-hot Jaccard (alternative reading): Spearman of pattern CD | 0.9856 | 0.9912 | 0.9897 |
| ours vs one-hot Jaccard (alternative reading): front overlap | 5 shared, Jaccard 0.556 | 7 shared, Jaccard 1.000 | 6 shared, Jaccard 0.857 |
| front members (ours) | 337419c, 376400, ac10307, ac355427, ac370000, ac415100, ac419100 | 376400, 378619a, ac10307, ac337441, ac370000, ac415100, ac419100 | 387070a, ac10307, ac355427, ac370000, ac388170, ac415100, ac419100 |

### Timings

| log | cases | patterns | pairwise_case_distance (s) | pattern_case_distance, all patterns (s) | matrix size (MB) |
|---|---|---|---|---|---|
| f1 | 1130 | 164 | 0.079 | 0.182 | 10.2 |
| f2 | 1130 | 207 | 0.066 | 0.159 | 10.2 |
| f3 | 1111 | 156 | 0.066 | 0.123 | 9.9 |
