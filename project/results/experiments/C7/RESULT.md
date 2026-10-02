# C7 — Apriori per-size counts vs `min_support`, and a per-size Apriori selector

Feeds Guide Part 6 decisions **B5** (top-k itemsets per size 1/2/3) and **B6** (`min_support` per log; f3 needs a value below 0.5).
Everything below was measured on 2 Oct 2026 with the project venv (Python 3.12.10, mlxtend 0.25.0, pandas 2.3.3, numpy 2.5.3). Nothing is random in this experiment.

## 1. What was run

```
cd "C:/Users/Bohabara/Desktop/process mining/project"
.venv/Scripts/python.exe experiments/run_c7_apriori_sweep.py
```

- Exit code 0, total wall time **98.6 s** (three complete runs: 72–105 s; the machine was shared with other experiments, so timings are noisy; all counts, supports and itemsets were identical in every run).
- Code:
  - `experiments/apriori_selector.py` — `AprioriSelector(min_support, max_len=3, top_k=10)` with `.select(traces)`, `.select_table(traces)`, `.full_table(traces)`; `original_top10(traces, min_support=0.5, top_k=10)`; `encode_transactions(traces)`.
  - `experiments/run_c7_apriori_sweep.py` — this experiment.
- Outputs in `results/experiments/C7/`:
  - `apriori_sets_f1.json`, `apriori_sets_f2.json`, `apriori_sets_f3.json` — top-10 per size with support and case count (the deliverable).
  - `sweep_counts.csv`, `uncapped_counts.csv`, `original_tie_choices.csv`, `f3_itemsets_size_gt1.csv`, `summary.json`, `run.log`.

## 2. Settings

| Setting | Value |
|---|---|
| Preprocessing | loaded through the ORIGINAL `DataManager(path, 2, None, L_max_perc=0.8)` (rare-activity filter, lower-casing, removal of spaces/`-`/`_`); traces = activities per case in `event_nr` order from `DataManager.data` |
| Case / activity counts (asserted) | f1 1130 / 164, f2 1130 / 207, f3 1111 / 156 |
| Transaction | set of distinct activities of one case (whole log, before any split) — same construction as `tools.py:161-166` (`TransactionEncoder`) |
| Support | fraction of cases containing all activities of the set; criterion `support >= min_support` (mlxtend) |
| Per-size selector | `mlxtend.apriori(..., max_len=3)`, top 10 per size, rank = support descending, ties broken lexicographically on the sorted activity names (ranking done on the integer case count, so ties are exact) |
| Sweep grid | `min_support` ∈ {0.5, 0.49, 0.45, 0.4, 0.35, 0.3}; runtime = best of 3 calls of `full_table` (one-hot encoding + Apriori + sort) |
| Original replica | `original_top10` = the statements of `tools.py:160-177` (no size cap, size > 1, top 10, pandas tie order); compared with `DataManager.frequent_activity_sets(s, 10)` at s ∈ {0.55, 0.53, 0.52, 0.51, 0.5, 0.49, 0.48, 0.47} |
| Uncapped counts | Apriori without `max_len` at f1 {0.5, 0.49, 0.47, 0.45}, f2 {0.5, 0.49, 0.47}, f3 {0.5, 0.49, 0.47, 0.45, 0.4}; one timing each |

## 3. Results

### 3.1 Checks (all passed)

- Preprocessing: the three case/activity counts match the known facts (assert in `load_original`).
- `original_top10` vs `DataManager.frequent_activity_sets`: **24 / 24 comparisons identical** (3 logs × 8 supports) — same itemsets, same rank order, same supports, and (inside one process) the same activity order inside each list. This includes the required f1 and f2 at (0.5, 10).
- Independent brute-force recount without mlxtend (numpy, all combinations of the activities with support ≥ 0.3): all 54 sweep counts (3 logs × 6 supports × 3 sizes) and the case counts of the 90 selected sets are reproduced.
- Toy example of the Guide (§3.3.4: cases {A,B}, {A,B,C}, {A,C}; `min_support` 0.6): selector returns size 1 {A},{B},{C}; size 2 {A,B},{A,C}; size 3 empty; `original_top10` returns {A,B},{A,C}.

### 3.2 Sweep: number of frequent sets of size 1 / 2 / 3 (`max_len=3`)

`10th` = support of the 10th most frequent set of that size (`-` = fewer than 10 sets exist).

| log | min_support | #size 1 | 10th (size 1) | #size 2 | 10th (size 2) | #size 3 | 10th (size 3) | ≥ 10 for every size | seconds (best of 3) |
|---|---|---|---|---|---|---|---|---|---|
| f1 | 0.5 | 16 | 0.5354 | 94 | 0.5442 | 258 | 0.5416 | yes | 0.017 |
| f1 | 0.49 | 20 | 0.5354 | 121 | 0.5442 | 414 | 0.5416 | yes | 0.022 |
| f1 | 0.45 | 23 | 0.5354 | 168 | 0.5442 | 841 | 0.5416 | yes | 0.033 |
| f1 | 0.4 | 23 | 0.5354 | 211 | 0.5442 | 1330 | 0.5416 | yes | 0.034 |
| f1 | 0.35 | 23 | 0.5354 | 212 | 0.5442 | 1330 | 0.5416 | yes | 0.031 |
| f1 | 0.3 | 25 | 0.5354 | 215 | 0.5442 | 1331 | 0.5416 | yes | 0.040 |
| f2 | 0.5 | 23 | 0.5496 | 148 | 0.5735 | 544 | 0.5646 | yes | 0.022 |
| f2 | 0.49 | 25 | 0.5496 | 188 | 0.5735 | 804 | 0.5646 | yes | 0.024 |
| f2 | 0.45 | 26 | 0.5496 | 248 | 0.5735 | 1390 | 0.5646 | yes | 0.037 |
| f2 | 0.4 | 26 | 0.5496 | 286 | 0.5735 | 2036 | 0.5646 | yes | 0.047 |
| f2 | 0.35 | 28 | 0.5496 | 311 | 0.5735 | 2272 | 0.5646 | yes | 0.043 |
| f2 | 0.3 | 30 | 0.5496 | 342 | 0.5735 | 2670 | 0.5646 | yes | 0.060 |
| f3 | 0.5 | 5 | - | 4 | - | 1 | - | **no** | 0.013 |
| f3 | 0.49 | 6 | - | 9 | - | 7 | - | **no** | 0.010 |
| f3 | 0.45 | 17 | 0.4788 | 84 | 0.4887 | 217 | 0.4833 | yes | 0.012 |
| f3 | 0.4 | 20 | 0.4788 | 130 | 0.4887 | 572 | 0.4833 | yes | 0.015 |
| f3 | 0.35 | 21 | 0.4788 | 156 | 0.4887 | 816 | 0.4833 | yes | 0.022 |
| f3 | 0.3 | 22 | 0.4788 | 160 | 0.4887 | 819 | 0.4833 | yes | 0.023 |

- The Guide's Table C numbers are confirmed: 0.5 → f1 16/94/258, f2 23/148/544, f3 5/4/1; 0.4 → f1 23/211/1330, f2 26/286/2036, f3 20/130/572.
- Runtime with `max_len=3` is negligible: 0.010–0.060 s per call in the final run, and never above 0.07 s in any run.

### 3.3 f3: which `min_support` first yields ≥ 10 sets per size

- On the grid: **0.45** is the first (largest) value with ≥ 10 sets for every size (17 / 84 / 217). 0.5 gives 5 / 4 / 1 and 0.49 gives 6 / 9 / 7 — at 0.49 *no* size reaches 10.
- Exact thresholds (support of the 10th set, i.e. any `min_support` at or below it gives ≥ 10 sets of that size): size 1: 532/1111 = 0.4788; size 2: 543/1111 = 0.4887; size 3: 537/1111 = 0.4833. All three sizes together: **≤ 0.4788**.

f3 itemsets of size > 1 (uncapped Apriori, all of them, most frequent first; `count` = number of cases):

| min_support | rank | itemset | size | support | count |
|---|---|---|---|---|---|
| 0.5 | 1 | {370407, ac370000} | 2 | 0.5275 | 586 |
| 0.5 | 2 | {ac370000, ac370419} | 2 | 0.5131 | 570 |
| 0.5 | 3 | {370407, ac370000, ac370419} | 3 | 0.5023 | 558 |
| 0.5 | 4 | {370407, ac370419} | 2 | 0.5023 | 558 |
| 0.5 | 5 | {ac370000, ac370443} | 2 | 0.5014 | 557 |
| 0.49 | 1–5 | the same five sets as at 0.5 | | | |
| 0.49 | 6 | {ac370000, ac370419, ac370443} | 3 | 0.4995 | 555 |
| 0.49 | 7 | {ac370419, ac370443} | 2 | 0.4995 | 555 |
| 0.49 | 8 | {ac370000, ac370442} | 2 | 0.4959 | 551 |
| 0.49 | 9 | {ac370000, ac370442, ac370443} | 3 | 0.4941 | 549 |
| 0.49 | 10 | {ac370442, ac370443} | 2 | 0.4941 | 549 |
| 0.49 | 11 | {ac370000, ac370419, ac370442} | 3 | 0.4932 | 548 |
| 0.49 | 12 | {ac370419, ac370442} | 2 | 0.4932 | 548 |
| 0.49 | 13 | {ac370000, ac370419, ac370442, ac370443} | 4 | 0.4923 | 547 |
| 0.49 | 14 | {ac370419, ac370442, ac370443} | 3 | 0.4923 | 547 |
| 0.49 | 15 | {370407, ac370000, ac370419, ac370443} | 4 | 0.4905 | 545 |
| 0.49 | 16 | {370407, ac370000, ac370443} | 3 | 0.4905 | 545 |
| 0.49 | 17 | {370407, ac370419, ac370443} | 3 | 0.4905 | 545 |
| 0.49 | 18 | {370407, ac370443} | 2 | 0.4905 | 545 |

- At 0.5 the original code returns only **5** itemsets for f3 (silently fewer than `top_k`).
- At 0.49 there are **18** sets of size > 1 (not 10; two of them have size 4). The original code keeps ranks 1–10, and those contain exactly **three sets with ac370442** (ranks 8, 9, 10) — consistent with the 10 rows / three ac370442 rows of the paper's Fig. 5 f3 panel.
- The original top-10 on f3 is the same for **every** `min_support` ≤ 549/1111 = 0.4941 (ranks 9–10 are tied at 549 cases and both are in; rank 11 has 548). So the figure tells us only that the authors' support for f3 was ≤ 0.494, not which value they used. This is still an inference, not something the paper states.

### 3.4 Recommended rule and the saved lists

Rule asked for: *largest support of the grid that yields ≥ 10 sets for every size 1–3*.

| log | recommended `min_support` (grid) | exact threshold (10th support, min over sizes) | counts at the recommended value (size 1/2/3) |
|---|---|---|---|
| f1 | **0.5** | 605/1130 = 0.5354 | 16 / 94 / 258 |
| f2 | **0.5** | 621/1130 = 0.5496 | 23 / 148 / 544 |
| f3 | **0.45** | 532/1111 = 0.4788 | 17 / 84 / 217 |

Important property (verified, `selection_stable_at_lower_supports = true` for all three logs): **the top-10-per-size lists are identical for every grid support at or below the recommended one.** Once at least 10 sets per size exist, `min_support` no longer influences the selection — only "top 10 by support" does. A single global value of 0.45 (or 0.4, the Guide's earlier working default for f3) therefore produces exactly the same lists as the per-log rule.

Saved lists (`apriori_sets_<log>.json`; support, then case count in brackets):

**f1** (`min_support` 0.5)

| rank | size 1 | size 2 | size 3 |
|---|---|---|---|
| 1 | {ac370000} 0.7009 (792) | {370407, ac370000} 0.5912 (668) | {ac370000, ac370419, ac370443} 0.5619 (635) |
| 2 | {ac419100} 0.6451 (729) | {ac370000, ac370419} 0.5770 (652) | {370407, ac370000, ac370419} 0.5593 (632) |
| 3 | {370407} 0.5912 (668) | {ac370000, ac370443} 0.5637 (637) | {ac370000, ac370419, ac370442} 0.5566 (629) |
| 4 | {ac370419} 0.5770 (652) | {ac370419, ac370443} 0.5619 (635) | {ac370000, ac370442, ac370443} 0.5566 (629) |
| 5 | {ac370443} 0.5637 (637) | {370407, ac370419} 0.5593 (632) | {ac370419, ac370442, ac370443} 0.5549 (627) |
| 6 | {ac370442} 0.5593 (632) | {ac370000, ac370442} 0.5593 (632) | {370407, ac370000, ac370443} 0.5487 (620) |
| 7 | {370712b} 0.5416 (612) | {ac370419, ac370442} 0.5566 (629) | {370407, ac370419, ac370443} 0.5487 (620) |
| 8 | {370715a} 0.5389 (609) | {ac370442, ac370443} 0.5566 (629) | {370407, ac370000, ac370442} 0.5442 (615) |
| 9 | {ac370403} 0.5381 (608) | {370407, ac370443} 0.5487 (620) | {370407, ac370419, ac370442} 0.5434 (614) |
| 10 | {370488e} 0.5354 (605) | {370407, ac370442} 0.5442 (615) | {370407, ac370442, ac370443} 0.5416 (612) |

**f2** (`min_support` 0.5)

| rank | size 1 | size 2 | size 3 |
|---|---|---|---|
| 1 | {ac419100} 0.9372 (1059) | {ac370000, ac379999} 0.6903 (780) | {ac370000, ac379999, ac419100} 0.6407 (724) |
| 2 | {ac370000} 0.7018 (793) | {ac370000, ac419100} 0.6469 (731) | {370407, ac370000, ac379999} 0.5929 (670) |
| 3 | {ac379999} 0.6903 (780) | {ac379999, ac419100} 0.6407 (724) | {ac370000, ac370419, ac379999} 0.5761 (651) |
| 4 | {370407} 0.6018 (680) | {370407, ac370000} 0.6018 (680) | {ac370000, ac370419, ac370443} 0.5735 (648) |
| 5 | {ac370419} 0.5858 (662) | {370407, ac379999} 0.5929 (670) | {ac370000, ac370442, ac370443} 0.5708 (645) |
| 6 | {ac410100} 0.5823 (658) | {ac370000, ac370419} 0.5858 (662) | {370407, ac370000, ac370419} 0.5699 (644) |
| 7 | {ac370443} 0.5770 (652) | {ac370000, ac370443} 0.5770 (652) | {ac370000, ac370419, ac370442} 0.5690 (643) |
| 8 | {ac370442} 0.5735 (648) | {ac370419, ac379999} 0.5761 (651) | {ac370000, ac370443, ac379999} 0.5673 (641) |
| 9 | {370712b} 0.5513 (623) | {ac370000, ac370442} 0.5735 (648) | {ac370419, ac370442, ac370443} 0.5673 (641) |
| 10 | {ac411100} 0.5496 (621) | {ac370419, ac370443} 0.5735 (648) | {370407, ac370000, ac370443} 0.5646 (638) |

**f3** (`min_support` 0.45)

| rank | size 1 | size 2 | size 3 |
|---|---|---|---|
| 1 | {ac419100} 0.6805 (756) | {370407, ac370000} 0.5275 (586) | {370407, ac370000, ac370419} 0.5023 (558) |
| 2 | {ac370000} 0.6346 (705) | {ac370000, ac370419} 0.5131 (570) | {ac370000, ac370419, ac370443} 0.4995 (555) |
| 3 | {370407} 0.5275 (586) | {370407, ac370419} 0.5023 (558) | {ac370000, ac370442, ac370443} 0.4941 (549) |
| 4 | {ac370419} 0.5131 (570) | {ac370000, ac370443} 0.5014 (557) | {ac370000, ac370419, ac370442} 0.4932 (548) |
| 5 | {ac370443} 0.5014 (557) | {ac370419, ac370443} 0.4995 (555) | {ac370419, ac370442, ac370443} 0.4923 (547) |
| 6 | {ac370442} 0.4959 (551) | {ac370000, ac370442} 0.4959 (551) | {370407, ac370000, ac370443} 0.4905 (545) |
| 7 | {370712b} 0.4887 (543) | {ac370442, ac370443} 0.4941 (549) | {370407, ac370419, ac370443} 0.4905 (545) |
| 8 | {370715a} 0.4842 (538) | {ac370419, ac370442} 0.4932 (548) | {370407, ac370000, ac370442} 0.4851 (539) |
| 9 | {370488e} 0.4797 (533) | {370407, ac370443} 0.4905 (545) | {370407, ac370419, ac370442} 0.4842 (538) |
| 10 | {370488g} 0.4788 (532) | {370712b, ac370000} 0.4887 (543) | {370407, ac370442, ac370443} 0.4833 (537) |

Tie-break at rank 10 of the per-size lists: it decides membership in exactly **one** place — f2 size 2, where three pairs have 648 cases for two free slots: {ac370000, ac370442} and {ac370419, ac370443} are kept, **{ac410100, ac419100} is left out** by the lexicographic rule. Every other rank-10 boundary (f1 all sizes, f2 sizes 1 and 3, f3 all sizes) is unique.

Number of distinct activities inside the selected lists (measured):

| log | size 1 | size 2 | size 3 |
|---|---|---|---|
| f1 | 10 | 5 | 5 |
| f2 | 10 | 7 | 7 |
| f3 | 10 | 6 | 5 |

On f1 the ten pairs are *all* 10 pairs and the ten triples are *all* 10 triples of the same five activities {370407, ac370000, ac370419, ac370442, ac370443} (10 distinct sets from 5 activities = C(5,2) = C(5,3) = 10); the same holds for the f3 triples. Many supports are also identical between a set and its superset (e.g. f1: {370407} and {370407, ac370000} both 668 cases), i.e. these activities almost always occur together.

### 3.5 The ORIGINAL selection (for reproduction runs)

`original_top10(traces, s, 10)` in this environment, in the rank order the original code produces (the order matters in faithful mode, because the original permutes the training traces in place across itemsets):

| rank | f1 @ 0.5 | f2 @ 0.5 | f3 @ 0.49 (f3 @ 0.5 = ranks 1–5 only) |
|---|---|---|---|
| 1 | {370407, ac370000} 0.5912 | {ac370000, ac379999} 0.6903 | {370407, ac370000} 0.5275 |
| 2 | {ac370000, ac370419} 0.5770 | {ac370000, ac419100} 0.6469 | {ac370000, ac370419} 0.5131 |
| 3 | {ac370000, ac370443} 0.5637 | {ac370000, ac379999, ac419100} 0.6407 | {370407, ac370000, ac370419} 0.5023 |
| 4 | {ac370000, ac370419, ac370443} 0.5619 | {ac379999, ac419100} 0.6407 | {370407, ac370419} 0.5023 |
| 5 | {ac370419, ac370443} 0.5619 | {370407, ac370000} 0.6018 | {ac370000, ac370443} 0.5014 |
| 6 | {ac370000, ac370442} 0.5593 | {370407, ac370000, ac379999} 0.5929 | {ac370000, ac370419, ac370443} 0.4995 |
| 7 | {370407, ac370000, ac370419} 0.5593 | {370407, ac379999} 0.5929 | {ac370419, ac370443} 0.4995 |
| 8 | {370407, ac370419} 0.5593 | {ac370000, ac370419} 0.5858 | {ac370000, ac370442} 0.4959 |
| 9 | {ac370442, ac370443} 0.5566 | {ac370000, ac370443} 0.5770 | {ac370000, ac370442, ac370443} 0.4941 |
| 10 | {ac370000, ac370419, ac370442} 0.5566 | {ac370419, ac379999} 0.5761 | {ac370442, ac370443} 0.4941 |

Three properties of the original selection that the students must know:

1. **The original top-10 is not uniquely defined on f1 and f2 — ties decide, and the winner changes with `min_support`.**
   - f1: 8 sets are above the rank-10 support 0.5566 (629 cases); **4 sets are tied** for the last 2 slots: {ac370419, ac370442}, {ac370442, ac370443}, {ac370000, ac370419, ac370442}, {ac370000, ac370442, ac370443}.
   - f2: 9 sets above 0.5761 (651 cases); **2 sets tied** for the last slot: {ac370419, ac379999}, {ac370000, ac370419, ac379999}.
   - f3 (≤ 0.494): 8 above, 2 tied for 2 slots → unique.
   - Which tied sets the original code (pandas `sort_values`, default non-stable quicksort) keeps, measured per `min_support`:

   | log | min_support | tied sets that made the top-10 |
   |---|---|---|
   | f1 | 0.55, 0.52 | {ac370419, ac370442}; {ac370442, ac370443} |
   | f1 | 0.53, 0.51, 0.5, 0.49, 0.48 | {ac370000, ac370419, ac370442}; {ac370442, ac370443} |
   | f1 | 0.47 | {ac370000, ac370419, ac370442}; {ac370419, ac370442} |
   | f2 | 0.55, 0.53, 0.51, 0.5, 0.49, 0.48, 0.47 | {ac370419, ac379999} |
   | f2 | 0.52 | {ac370000, ac370419, ac379999} |
   | f3 | 0.49, 0.48, 0.47 | {ac370000, ac370442, ac370443}; {ac370442, ac370443} (no choice) |

   The order of tied sets inside the top-10 also changes with `min_support` (e.g. f1 ranks 4/5 swap between 0.5 and 0.49). Not tested: whether the tie outcome also changes with the pandas/numpy version — it is an implementation detail of the sort, so it may; the authors' 2024 environment could have picked other tied sets for the paper's f1 and f2 panels.
2. **The activity order inside each returned list is random between Python processes.** The original returns `list(frozenset)`; with `PYTHONHASHSEED` 0 / 1 / 2 the first f1 list came out as `['ac370000', '370407']`, `['ac370000', '370407']`, `['370407', 'ac370000']` and the fourth as `['ac370443', 'ac370419', 'ac370000']`, `['ac370443', 'ac370000', 'ac370419']`, `['ac370419', 'ac370443', 'ac370000']`. This only changes labels (CSV column names, plot tick labels): `itemset_permutation_importance` converts the list to a `set` (`tools.py:520`) and `shuffle_sequence` takes the order from the trace. Check with e.g.
   `PYTHONHASHSEED=1 .venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'external/PermutationLocationImportance'); from tools import DataManager; dm=DataManager('external/PermutationLocationImportance/datasets/BPIC11_f1_trunc36.csv',2,None,L_max_perc=0.8); print(dm.frequent_activity_sets(0.5,10)[0][:4])"`
3. **Without a size cap the original Apriori explodes just below 0.5 on f1/f2** (measured, uncapped `apriori`, one call):

   | log | min_support | frequent sets (all sizes) | largest size | sets per size 1/2/3/… | seconds |
   |---|---|---|---|---|---|
   | f1 | 0.5 | 1,142 | 8 | 16/94/258/364/277/111/21/1 | 0.05 |
   | f1 | 0.49 | 3,704 | 9 | 20/121/414/838/1044/803/366/89/9 | 0.15 |
   | f1 | 0.47 | 44,662 | 13 | 22/159/806/2773/6374/9982/10775/8036/4086/1352/268/28/1 | 2.8 |
   | f1 | 0.45 | 262,215 | 18 | 23/168/841/3079/8575/18565/31824/43758/48620/43758/31824/18564/8568/3060/816/153/18/1 | 32.0 |
   | f2 | 0.5 | 4,705 | 9 | 23/148/544/1140/1377/982/404/82/5 | 0.28 |
   | f2 | 0.49 | 13,271 | 10 | 25/188/804/2076/3348/3435/2246/916/213/20 | 0.54 |
   | f2 | 0.47 | 154,153 | 13 | 25/215/1160/4504/12633/25002/34880/34629/24439/11981/3876/745/64 | 13.7 |
   | f3 | 0.5 | 10 | 3 | 5/4/1 | 0.01 |
   | f3 | 0.49 | 24 | 4 | 6/9/7/2 | 0.01 |
   | f3 | 0.47 | 102 | 5 | 13/33/38/17/1 | 0.02 |
   | f3 | 0.45 | 855 | 7 | 17/84/217/280/189/62/6 | 0.03 |
   | f3 | 0.4 | 65,568 | 16 | 20/130/572/1826/4369/8008/11440/12870/11440/8008/4368/1820/560/120/16/1 | 3.2 |

   Not measured: uncapped f2 at 0.45 and f1/f2 at 0.4 or lower (skipped on purpose — expected to take minutes and a lot of memory while other experiments were running). With `max_len=3` every support of the grid down to 0.3 costs ≤ 0.07 s.

## 4. What this means for the project decisions

- **B6 (`min_support` per log).** Use **0.5 for f1 and f2 and 0.45 for f3** if a per-log rule is wanted ("largest grid support with ≥ 10 sets for every size 1–3"). Simpler and equivalent: **one global `min_support` = 0.45 with `max_len=3`** — the selected lists are identical (verified down to 0.3), because the selection is really "top 10 per size by support" and `min_support` only has to be low enough (≤ 0.5354 on f1, ≤ 0.5496 on f2, ≤ 0.4788 on f3). The Guide's earlier working default of 0.4 for f3 gives the same f3 lists as 0.45. Do not pass a value equal to one of these exact thresholds (floating-point `>=`); stay on the grid.
- **B5 (top-k per size).** `AprioriSelector(min_support, max_len=3, top_k=10).select(traces)` gives 10 + 10 + 10 sets per log, deterministically; the lists are saved in the three JSON files with supports and case counts. `max_len=3` is required for any support below ~0.49 (see the explosion table), and it does not change sizes 1–3.
- **Reproduction of the paper (faithful mode).** Use `original_top10(traces, 0.5, 10)` for f1 and f2 and `original_top10(traces, 0.49, 10)` for f3 (0.5 gives only 5 sets on f3; any value ≤ 0.494 gives the same 10 as 0.49, and uncapped Apriori is still instant on f3 at 0.49). It is verified equal to the original function. Keep 0.5 exactly for f1/f2: changing the support changes which tied sets enter the top-10.
- **For the report / discussion.**
  - The original "top-10 of size > 1" is partly arbitrary on f1 (2 of 10 slots) and f2 (1 of 10 slots); state this as a limitation of the reproduction and prefer the deterministic tie-break in the group's own variant.
  - Apriori's most frequent sets are very redundant: sizes 2 and 3 are built from only 5–7 activities per log (on f1 literally every pair and every triple of five activities). This is a concrete, measured argument for comparing with IMPresseD-based sets, and the "number of distinct activities covered" is a cheap extra comparison metric between the two strategies.
  - Size-1 Apriori sets (top 10 by support) are *not* the same thing as the original single-activity mode, which permutes every distinct activity (`tools.py:376`) and plots the top 20; decide which of the two the "length 1" comparison uses.
- **Open point for the supervisor (unchanged, B6):** the paper does not state the support used for f3. The data only show it was ≤ 0.494 if the code as published was used.

## 5. What failed / limits

- Nothing failed in the final run. During development nothing crashed either; the only change between runs was added checks.
- Timings are indicative only (shared machine); counts, supports and itemsets are deterministic.
- The comparison with the paper's Fig. 5 f3 panel relies on the Guide's earlier reading of the rendered figure (10 rows, three with ac370442); the figure itself was not re-inspected here, and the f1/f2 panels were not compared with the tie choices above. If someone reads the f1 and f2 panels of Fig. 5 row by row, the table in §3.5 tells which tied sets to look for.
- Transactions are built on the whole log (as in the original, before the train/test split). Mining per training fold was not part of this experiment.
