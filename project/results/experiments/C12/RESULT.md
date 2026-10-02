# Experiments C11 + C12 — paretoset versions and an explicit case distance

Run on 2026-10-02, Windows 10, Python 3.12.10 from `project/.venv` (numpy 2.5.3, pandas 2.3.3,
scipy 1.18.1, scikit-learn 1.9.1, paretoset 1.2.5, numba 0.68.0). All numbers below were produced by
the commands in section 1; nothing is estimated. Raw outputs: `C11/c11_table.md`, `C11/c11_results.json`,
`C12/c12_tables.md`, `C12/c12_results.json`, `C12/run_log.txt`.

## 0. Short answers

| Question | Answer |
|---|---|
| C11: do paretoset 1.2.0 and 1.2.5 agree? | **Yes** on everything we use: 141 of 143 recorded results are identical. The 2 that differ are features we do not use (`sense="diff"` and `crowding_distance`), where 1.2.0 crashes under pandas 2 / numpy 2. The Pareto algorithms are textually identical in both versions. **Pin 1.2.5.** |
| C11: sense spelling | `"max"`, `"Max"`, `"MAX"` give the same (correct) front in both versions. |
| C11: `distinct` | `distinct=True` keeps only the **first** of several identical objective rows; `distinct=False` keeps all. In `paretorank`, `distinct=True` pushes every duplicate into a later layer. **Use `distinct=False`** in both functions. |
| C11: NaN | A NaN is treated as "neither better nor worse", so a NaN row can **remove valid patterns from the front** and the result can depend on the row order. **Never pass NaN**; replace an undefined case distance by the worst value 1.0 first. |
| C12: which behaviour does our distance reproduce? | The **explicit mismatch share** (= `pdist(codes, 'hamming')`, same in every scipy version). It is **not** identical to `pdist(codes, 'jaccard')` in either scipy version: it equals scipy 1.11.4 on 84–86 % of the case pairs and scipy 1.18.1 on 5 %. |
| C12: does the choice matter? | Hugely for scipy 1.18.1 (the installed one): the original code then gives a different quantity (pair-level Pearson r ≈ 0.50, length-1 Pareto fronts of 17/33/29 patterns instead of 7/7/7). Mildly for scipy 1.11.4 (r ≈ 0.97, fronts 7/12/7 vs 7/7/7). |
| C12: BPIC11 case attributes | No missing values; every attribute is constant inside a case; 0 activities occur in all cases (the case distance is never undefined at length 1 on the whole log). |

## 1. What was run

All commands from `C:/Users/Bohabara/Desktop/process mining/project`, with `PY = .venv/Scripts/python.exe`.

```
# throw-away installs (once)
PY -m pip install paretoset==1.2.0 --target results/experiments/C11/tmp/paretoset_1_2_0 --no-deps
PY -m pip install scipy==1.11.4 numpy==1.26.4 --target results/experiments/C12/tmp/scipy_1_11_4 --no-deps --only-binary=:all:

# C11 (about 30 s: two worker processes of ~15 s, of which ~6 s is the numba compilation)
PY experiments/c11_paretoset_versions.py

# C12 unit tests (18 tests, 0.03 s, all pass)
PY -m unittest experiments/test_case_distance.py -v

# C12 comparison (about 40 s)
PY experiments/c12_compare_case_distance.py
```

Both installs succeeded on Python 3.12 (binary wheels `cp312`), so the fallback to the documented values
0.5 / 0.667 was not needed: they were re-measured.

| File | Purpose |
|---|---|
| `experiments/case_distance.py` | `build_case_attribute_table`, `pairwise_case_distance`, `pattern_case_distance` (the module to reuse) |
| `experiments/test_case_distance.py` | 18 unit tests on hand-computed examples |
| `experiments/c11_paretoset_versions.py` | C11 driver + worker (one process per paretoset version) |
| `experiments/c12_scipy_worker.py` | runs the **unchanged** 2023 functions `calculate_pairwise_case_distance` and `similarity_measuring_patterns` under one scipy version |
| `experiments/c12_compare_case_distance.py` | C12 driver: attribute tables, 200-case sample, both scipy versions, comparison |

Settings: seed 2023 (`numpy.random.default_rng`) for the 200-case sample and the random Pareto data;
logs read like the 2024 `DataManager` (activity names lower-cased, " ", "-", "_" removed, cases with an
activity that has < 2 events removed); case attributes `Age` (numeric) and `Diagnosis`, `Treatment code`,
`Diagnosis code`, `Specialism code` (categorical); interest functions for the pattern-level part:
information gain (`mutual_info_classif(..., discrete_features=True)` on per-case activity counts, max),
case coverage (max), case distance (min), scored on the whole log, `paretoset(distinct=False)`.

## 2. C11 — paretoset 1.2.0 vs 1.2.5

Both versions were run in separate processes against the same numpy 2.5.3 / pandas 2.3.3 / numba 0.68.0.

**Source diff** (`diff -r` of the two packages): the files differ only in (a) the version string and a removed
Python-version check, (b) `np.float` / `np.int` → `float` / `int` inside `crowding_distance`, (c) a
`try/except KeyError` around the group key of the `sense="diff"` path, (d) blank lines. The functions that
compute the front (`BNL`, `paretoset_efficient`, `pareto_rank_naive`) are textually identical.

**Measured**: 143 recorded results, 141 identical, 2 different:

| Case | 1.2.0 | 1.2.5 | Relevant for us? |
|---|---|---|---|
| `paretoset(df, sense=["diff","min"])` | `KeyError: ('g1',)` | works | no — IMPresseD only uses max/min |
| `crowding_distance(...)` | `AttributeError: module 'numpy' has no attribute 'float'` | works | no — not called anywhere |

### 2.1 Sense spelling (mixed max/min)

Six patterns with columns (information gain, coverage, case distance) and senses max, max, min. The brute-force
front (textbook dominance definition, written in the script) is {P1, P2, P4, P5}.

| `sense` given as | DataFrame input | ndarray input |
|---|---|---|
| `["max","max","min"]` | P1, P2, P4, P5 | P1, P2, P4, P5 |
| `["Max","Max","Min"]` (what the 2023 GUI passes) | P1, P2, P4, P5 | P1, P2, P4, P5 |
| `["MAX","MAX","MIN"]` | P1, P2, P4, P5 | P1, P2, P4, P5 |
| `None` (control: everything minimised) | P1, P3, P4, P5 | – |

Identical in 1.2.0 and 1.2.5. The caller's DataFrame / array is not modified by the call (checked).

### 2.2 Duplicate objective rows

Identical in both versions and for both internal algorithms (`use_numba=True` / `False`).

| Case (senses) | `distinct=True` | `distinct=False` |
|---|---|---|
| D1: a(1,1), b(1,1), c(2,0), d(0,2) (max, max) | a, c, d | a, b, c, d |
| D2: A=B=F=(0.2,0.5,0.3), C=(0.1,0.9,0.3), D=E=(0.1,0.4,0.4) (max, max, min) | A, C | A, B, C, F |
| D3: four identical rows | a | a, b, c, d |
| D4: a(0.2,0.5,0.3), b(0.2,0.5,0.4) — tie on two objectives only | a | a |

`paretorank` (Pareto layers, rank 1 = front):

| Case | `distinct=True` | `distinct=False` |
|---|---|---|
| D2 | A 1, C 1, **B 2, F 3, D 4, E 5** (5 layers) | A 1, B 1, C 1, F 1, D 2, E 2 (2 layers) |
| 500 random rows × 3 objectives rounded to 1 decimal (103 duplicate rows) | 36 layers | 25 layers |
| 2000 random rows × 3 objectives rounded to 2 decimals (2 duplicate rows) | 29 layers | 29 layers |

On the random data, `paretoset(distinct=False)` equals the brute-force front, `paretoset(distinct=True)` equals
"first occurrence of each distinct front row", and rank 1 of `paretorank` equals the `paretoset` mask — in both
versions and both algorithms (front sizes 6 and 19).

Meaning: with `distinct=True`, which of two patterns with identical scores survives depends only on the row
order, and in layer filling a duplicate is demoted by one layer per copy. Two different patterns can have
exactly identical scores (same per-case counts → same information gain, coverage and case distance); on the
real data there is 1 such duplicate row among the length-1 patterns of each log.

### 2.3 Rows containing NaN

Every case was run with `distinct` True/False × `use_numba` True/False (all four gave the same result) and in
every possible row order. "Remedy" columns: the NaN was removed before the call.

| Case (senses) | paretoset keeps | Over all row orders | Remedy: drop NaN rows | Remedy: NaN → worst value |
|---|---|---|---|---|
| N1: a(1,1), b(NaN,0), c(2,2) (min, min) | **b** | b | a | a, b |
| N2: a(1,1), b(0,NaN), c(2,2) (min, min) | **b** | b | a | a, b |
| N3: a(NaN,NaN), b(1,1), c(2,2) (min, min) | **a** | **"a" or "b", depending on the order** | b | b |
| N4: p4 = (IG 0.00, CC 1.0, CD NaN) among four normal patterns (max, max, min) | p1, p2, p4, p5 | same | p1, p2, p5 | p1, p2, p4, p5 |
| N5: q2 = (0.1, 0.4, NaN) is worse than q1 on IG and CC | q1 | q1 | q1 | q1 |
| N6: r_all(0.20, 1.0, NaN), r_x(0.20, 0.8, 0.10), r_y(0.05, 0.5, 0.05) (max, max, min) | **r_all only** | r_all | r_x, r_y | r_all, r_x, r_y |

What NaN does: comparisons with NaN are false, so a NaN counts as a tie on that objective. A row with a NaN can
therefore dominate rows that should stay on the front (N1, N2, N6: the valid patterns r_x and r_y disappear),
and with an all-NaN row the answer depends on the row order (N3). With the NaN replaced by the worst value
(`inf`, or 1.0 for a distance in [0, 1]) paretoset equals the brute-force front in every case. The results were
the same in both versions, but the package documents "the user is responsible for dealing with NaN values" and
its numba code is compiled with `fastmath=True`, so this behaviour is not guaranteed.

### 2.4 Recommended settings (C11)

1. Keep `paretoset==1.2.5` in `requirements.txt`; state in the README that 1.2.0 (the 2023 pin) gives identical
   fronts (tested) and fails only in two features we do not use.
2. Call `paretoset(objectives, sense=["max", "max", "min"], distinct=False)`; write the senses in lower case
   (any case works).
3. For the k = 10 filling by layers use `paretorank(objectives, sense=..., distinct=False)` (rank 1 = front,
   rank 2 = next front, …) or peel fronts with `paretoset`; both give the same layers. Never `distinct=True`.
4. No NaN in the objective table: compute the case distance with
   `pattern_case_distance(dist, mask, undefined_value=1.0)` (or drop such patterns) and assert
   `objectives.notna().all().all()` before the call.
5. The first call compiles with numba (6.4 s and 6.6 s measured in the final run, 5.7–6.6 s over all runs); later calls are fast.

## 3. C12 — explicit case distance

### 3.1 What the original code does (read in `IMIPD.py`, verified numerically)

* `calculate_pairwise_case_distance` (IMIPD.py:135-165): label-encode the categorical columns →
  `pdist(codes, 'jaccard')` (:140-146); numeric columns → `pdist(values, 'euclid')`, then `MinMaxScaler` over
  the **distances** (:148-155); combination `((len(cat_col) * cat_dist) + numeric_dist) / (1 + len(cat_col))`
  (:158). **The formula `(m*cat + num)/(1+m)` is confirmed**: plugging the original's own Jaccard values into
  our numeric part and formula reproduces the original output to 2.2e-16 in both scipy versions and on all four
  tables.
* `similarity_measuring_patterns` (IMIPD.py:37-60): `np.mean` over all (case with pattern, case without
  pattern) pairs (:57-58). Our `pattern_case_distance` on the same distance matrix reproduces the original
  function to 1.1e-16 (11 patterns on the 200-case sample, both scipy versions). For a pattern in no case the
  original returns NaN with `RuntimeWarning: Mean of empty slice`.

### 3.2 Tiny pairs — measured in both scipy versions

| Pair | `pdist jaccard` scipy 1.11.4 | `pdist jaccard` scipy 1.18.1 | **explicit mismatch share (ours)** | `pdist hamming` (both versions) |
|---|---|---|---|---|
| A: [0,1,2] vs [0,1,3] | 0.5 | 0.0 | **0.3333** | 0.3333 |
| B: [0,1,2] vs [1,1,3] | 0.6667 | 0.3333 | **0.6667** | 0.6667 |
| C: [0,1,2] vs [1,1,2] | 0.3333 | 0.3333 | **0.3333** | 0.3333 |

The documented values 0.5 / 0.667 (scipy 1.11.4) and 0.0 / 0.333 (scipy 1.18.1) are confirmed.

The two scipy rules, re-implemented in `emulate_jaccard` and equal to the real `pdist` output with 0 difference
on the 200-case sample and on all pairs of f1, f2, f3:

* **scipy 1.11.4**: (number of attributes with different codes) / (number of attributes on which the two cases
  are **not both code 0**). An attribute on which both cases carry the alphabetically first category is left
  out of the denominator. Pair A: 1 difference / 2 counted attributes = 0.5.
* **scipy 1.18.1** (since 1.15): each code becomes "code ≠ 0", then boolean Jaccard. It only sees whether a
  case has the alphabetically first category or not; codes 2 and 3 are "equal". Pair A: 0.
* **ours**: (number of attributes with different values) / m. Pair A: 1/3. No label encoding involved.

### 3.3 Which behaviour our function reproduces, and why

`pairwise_case_distance` reproduces the original code **except for the categorical part**, where it uses the
explicit mismatch share. It therefore reproduces **neither** `pdist('jaccard')` behaviour exactly; it is close
to scipy 1.11.4 (the authors' pin) and far from scipy 1.18.1 (our environment). Reasons:

1. Both scipy behaviours depend on which category `LabelEncoder` happens to number 0 (the alphabetically first:
   `TC101` for Treatment code, carried by 40 % of the cases; `DC106`; `SC13`; `Adenoca. vagina st II`). Under
   scipy 1.11.4 two cases that share `TC101` get a **larger** distance than two cases that share `TC102`; under
   scipy 1.18.1 all categories except the first are the same. Renaming a category would change the result. Our
   value does not depend on names or codes (unit test `test_does_not_depend_on_the_names_of_the_categories`).
2. It matches the paper's formula read literally: dist = (F_normal(dist_Euc) + dist_Jac)/(m+1) with dist_Jac
   running over the m categorical features. With dist_Jac = number of mismatching attributes this is exactly
   `(m * share + numeric)/(1+m)`. (Our reading; the paper's reference [8], Cheung & Jia 2013, was not checked.)
3. It is the same number in every scipy version, so the result does not change when the environment changes.

If the supervisor wants the authors' exact numbers instead, `emulate_jaccard(codes, booleanise=False)` in
`c12_compare_case_distance.py` gives the scipy 1.11.4 values without installing the old scipy.

### 3.4 BPIC11 per-case attribute tables

Built with `build_case_attribute_table` (first event per case by `event_nr`); saved as
`case_attributes_f1.csv`, `_f2.csv`, `_f3.csv` (index = case id as string, sorted).

| log | events (file) | cases (file) | cases after rare-activity filter | activities after filter | label = 1 cases | Age range | missing attribute values |
|---|---|---|---|---|---|---|---|
| f1 | 24176 | 1140 | 1130 | 164 | 454 | 19–99 | 0 |
| f2 | 31235 | 1140 | 1130 | 207 | 886 | 19–99 | 0 |
| f3 | 20534 | 1121 | 1111 | 156 | 259 | 19–99 | 0 |

Distinct values per attribute, whole file / per-case table after the filter (missing values: 0 everywhere, in
events and in cases; every attribute has exactly 1 value inside each case):

| attribute | f1 | f2 | f3 | category encoded as 0 (share of cases f1 / f2 / f3) |
|---|---|---|---|---|
| Age (numeric) | 74 / 74 | 74 / 74 | 73 / 73 | – |
| Diagnosis | 105 / 105 | 105 / 104 | 101 / 101 | `Adenoca. vagina st II` (0.0009 / 0.0009 / 0.0009) |
| Treatment code | 43 / 42 | 43 / 41 | 42 / 42 | `TC101` (0.4044 / 0.4053 / 0.4086) |
| Diagnosis code | 11 / 11 | 11 / 11 | 11 / 11 | `DC106` (0.1292 / 0.1292 / 0.1278) |
| Specialism code | 3 / 3 | 3 / 3 | 3 / 3 | `SC13` (0.0496 / 0.0496 / 0.0495) |

Since several cases share an age, the smallest Age distance is 0 and the min-max scaling reduces to
|ΔAge| / 80 on these logs.

### 3.5 Pair level: ours vs the original code under each scipy

"Identical" = absolute difference < 1e-12. 200 cases = 19 900 pairs; f1 = 637 885 pairs.

| table | quantity | compared with original code @ | mean ours | mean other | share of pairs identical | mean abs diff | max abs diff | Pearson r | Spearman ρ |
|---|---|---|---|---|---|---|---|---|---|
| 200 random f1 cases | categorical part | scipy 1.11.4 | 0.7464 | 0.7709 | 0.8645 | 0.0244 | 0.5 | 0.9629 | 0.9681 |
| 200 random f1 cases | categorical part | scipy 1.18.1 | 0.7464 | 0.2002 | 0.0495 | 0.5465 | 1.0 | 0.3997 | 0.3894 |
| 200 random f1 cases | combined distance | scipy 1.11.4 | 0.6460 | 0.6655 | 0.8645 | 0.0195 | 0.4 | 0.9648 | 0.9668 |
| 200 random f1 cases | combined distance | scipy 1.18.1 | 0.6460 | 0.2090 | 0.0495 | 0.4372 | 0.8 | 0.4343 | 0.4685 |
| f1 (1130 cases) | combined distance | scipy 1.11.4 | 0.6353 | 0.6565 | 0.8454 | 0.0212 | 0.4 | 0.9666 | 0.9744 |
| f1 (1130 cases) | combined distance | scipy 1.18.1 | 0.6353 | 0.2068 | 0.0494 | 0.4287 | 0.8 | 0.4960 | 0.5403 |
| f2 (1130 cases) | combined distance | scipy 1.11.4 | 0.6354 | 0.6567 | 0.8444 | 0.0213 | 0.4 | 0.9663 | 0.9741 |
| f2 (1130 cases) | combined distance | scipy 1.18.1 | 0.6354 | 0.2068 | 0.0489 | 0.4287 | 0.8 | 0.4973 | 0.5418 |
| f3 (1111 cases) | combined distance | scipy 1.11.4 | 0.6346 | 0.6561 | 0.8427 | 0.0216 | 0.4 | 0.9662 | 0.9740 |
| f3 (1111 cases) | combined distance | scipy 1.18.1 | 0.6346 | 0.2068 | 0.0500 | 0.4279 | 0.8 | 0.4971 | 0.5417 |

(The categorical-part rows for the whole logs are in `c12_tables.md`.) By the scipy 1.11.4 rule of section 3.2,
ours can differ from it only on pairs in which both cases carry the code-0 category of some attribute (with
the shares of section 3.4 that is mainly two cases that both have `TC101`).

Consistency checks (44 = 11 per table, all ≤ 2.2e-16): ours categorical = `pdist hamming` in both scipy
versions; ours numeric = original numeric part in both versions; emulations = real `pdist jaccard`; original
categorical part = `pdist jaccard`; combination formula; `2s/(1+s)` = boolean Jaccard on one-hot columns.

### 3.6 Pattern level

**200-case sample**, 10 most frequent activities plus one that is in no sampled case:

| activity | cases with it | original function @ scipy 1.11.4 | original function @ scipy 1.18.1 | explicit distance (ours) |
|---|---|---|---|---|
| ac370000 | 138 | 0.667769 | 0.257663 | 0.646427 |
| ac419100 | 135 | 0.685718 | 0.215526 | 0.674869 |
| 370407 | 119 | 0.675572 | 0.244165 | 0.658744 |
| ac370443 | 115 | 0.677792 | 0.248813 | 0.663709 |
| ac370419 | 114 | 0.677295 | 0.244777 | 0.662355 |
| ac370442 | 112 | 0.679829 | 0.247165 | 0.665935 |
| 370715a | 110 | 0.680810 | 0.242312 | 0.666359 |
| 370712b | 108 | 0.682685 | 0.239985 | 0.667910 |
| ac370403 | 106 | 0.683082 | 0.240703 | 0.669714 |
| ac372417 | 105 | 0.683063 | 0.241599 | 0.670144 |
| 302282 | 0 | NaN (+ RuntimeWarning) | NaN (+ RuntimeWarning) | NaN (no warning) |

**Whole logs, all length-1 patterns** (single activities), objectives IG max / coverage max / case distance min:

| quantity | f1 | f2 | f3 |
|---|---|---|---|
| patterns (activities) | 164 | 207 | 156 |
| patterns in all cases (case distance undefined) | 0 | 0 | 0 |
| duplicate objective rows (ours) | 1 | 1 | 1 |
| ours: CD min / median / max | 0.5904 / 0.6617 / 0.7687 | 0.5900 / 0.6662 / 0.7626 | 0.5466 / 0.6640 / 0.7728 |
| original @ scipy 1.11.4: CD min / median / max | 0.6165 / 0.6749 / 0.7791 | 0.6174 / 0.6787 / 0.7786 | 0.5927 / 0.6777 / 0.7843 |
| original @ scipy 1.18.1: CD min / median / max | 0.1511 / 0.2105 / 0.3128 | 0.1504 / 0.2136 / 0.2956 | 0.1514 / 0.2160 / 0.3175 |
| Spearman of pattern CD, ours vs original @ 1.11.4 | 0.9497 | 0.9658 | 0.9555 |
| Spearman of pattern CD, ours vs original @ 1.18.1 | 0.2348 | 0.3297 | 0.3988 |
| **Pareto front size, ours** (`distinct=False` / `True`) | **7** / 7 | **7** / 7 | **7** / 7 |
| Pareto front size, original @ 1.11.4 | 7 / 7 | 12 / 12 | 7 / 7 |
| Pareto front size, original @ 1.18.1 | 17 / 17 | 33 / 33 | 29 / 28 |
| front overlap, ours vs original @ 1.11.4 | 6 shared (Jaccard 0.750) | 7 shared (0.583) | 7 shared (1.000) |
| front overlap, ours vs original @ 1.18.1 | 3 shared (0.143) | 4 shared (0.111) | 1 shared (0.029) |
| front members (ours) | 337419c, 376400, ac10307, ac355427, ac370000, ac415100, ac419100 | 376400, 378619a, ac10307, ac337441, ac370000, ac415100, ac419100 | 387070a, ac10307, ac355427, ac370000, ac388170, ac415100, ac419100 |

Extra sensitivity check — "Jaccard" read as the set Jaccard distance on one-hot encoded attributes, which
equals 2s/(1+s) for a mismatch share s (verified against `pdist` on booleans): pattern-CD Spearman with ours
0.9856 / 0.9912 / 0.9897, front sizes 7 / 7 / 6, shared with ours 5 / 7 / 6.

Timings of our functions (final run): distance matrix of 1111–1130 cases in 0.07–0.08 s (about 10 MB as
float64); case distance of all 156–207 length-1 patterns in 0.12–0.18 s. The original pair-list function
needed 0.19–0.32 s for 11 patterns on only 200 cases. Timings vary by a few hundredths of a second between runs
(other experiments were using the CPU at the same time).

### 3.7 `pattern_case_distance` when a pattern is in all or in no cases

There is no (with, without) pair, so the distance is undefined. The function returns `undefined_value`
(default `NaN`, as the original, but without the warning). Because NaN must not reach paretoset (section 2.3),
the selection code should call it with `undefined_value=1.0`: 1.0 is the largest possible distance, so an
undefined distance can never help a pattern, while the pattern can still reach the front through coverage or
information gain. On BPIC11 this does not occur for patterns mined and scored on the whole log (0 activities
are in all cases; longer patterns cover fewer cases); it can occur if patterns are scored on a training fold.
Missing attribute values make `pairwise_case_distance` raise a `ValueError` (BPIC11 has none).

### 3.8 Behaviours of the original that we kept on purpose

* The numeric distances, not the attributes, are min-max scaled over all pairs, so the matrix depends on which
  cases are in the table (compute it once on all cases and slice it for folds).
* Each categorical attribute weighs as much as all numeric attributes together (weight m vs 1).
* Mean over pairs (code), not the paper's 1/|L|-weighted sum.
* Attributes only — the GUI additionally feeds the activity-count columns as "categorical" attributes; we do not.

## 4. What this means for the project decisions

* **README / requirements (C11, B7)**: keep paretoset 1.2.5; `distinct=False`; no NaN; layers via
  `paretorank(distinct=False)`.
* **B12 / C12 (case distance)**: use `case_distance.py`. In the paper, say: "categorical part = share of
  attributes on which two cases differ; the original call `pdist(..., 'jaccard')` on label codes is
  scipy-version dependent and encoding dependent". Never call the original `calculate_pairwise_case_distance`
  under the project's scipy 1.18.1 — it silently produces a different interest function.
* **B7 / C6 (front size, k = 10)**: with the explicit distance the length-1 front on the whole log has only
  **7 patterns in each of f1, f2, f3**, fewer than k = 10, so layer filling is needed already at length 1. The
  earlier probe values 19 / 32 / 29 are close to what the booleanised scipy 1.18.1 distance gives here
  (17 / 33 / 29); they were probably produced with `pdist` under the venv's scipy (not re-checked how the probe
  was run) and should not be quoted for the explicit distance.
* **Question for the supervisor (still open)**: which scipy version produced the published figures, and whether
  the mismatch share is the intended "Jaccard distance" of reference [8]. Under their pin (1.11.4) our fronts
  share 6/7, 7/12 and 7/7 patterns with the original's at length 1.

## 5. Limits and things that did not get done

* Nothing failed. Not verified: reference [8] (Cheung & Jia, Pattern Recognition 46(8), 2013) was not read; the
  scipy version behind the published figures is unknown.
* The pattern-level comparison covers length-1 patterns scored on the whole log only (no lengths 2–3, no
  train/test split as in `Auto_IMPID.py`).
* In the scipy 1.11.4 process, pandas 2.3.3 and scikit-learn 1.9.1 ran against numpy 1.26.4. That combination
  is not an official one, but it only served to call the original function; the `pdist` values themselves were
  cross-checked by the emulation (0 difference).
* The NaN behaviour of paretoset was observed, not guaranteed (see 2.3).
* The throw-away folders `C11/tmp` (0.1 MB) and `C12/tmp` (216 MB) are only needed to re-run the two drivers
  and can be deleted; `C12/work` (15 MB) holds the worker outputs.
