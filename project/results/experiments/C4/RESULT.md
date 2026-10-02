# Experiment C4 — chain-only re-implementation of IMPresseD's automatic mode

Run on 2026-10-02, Windows 10, Python 3.12 from `project/.venv` (numpy 2.5.3, pandas 2.3.3, scipy 1.18.1,
scikit-learn 1.9.1, networkx 3.7, paretoset 1.2.5). Every number below was measured by the commands in
section 1 (plus one ad-hoc timing check of the numba warm-up, marked where it is used); nothing is estimated.
Other experiments were running on the same machine, so wall-clock times vary between runs (the same original
call took 136.2 s in one run and 188.9 s in another). The original is 2 to 4 orders of magnitude slower than our
code, so this does not change any conclusion.

## 0. Short answers

| Question | Answer |
|---|---|
| Is a tuple key `(labels…, edge types…)` equivalent to the original isomorphism test on chains? | **Yes.** In every comparison the original code and our code produce exactly the same set of patterns: 475 + 461 + 598 + 1,028 + 385 + 1,204 patterns, 0 only in the original, 0 only in ours. `nx.is_isomorphic` (called as the original calls it) agreed with tuple equality on 21,807 tested pairs, 807 of them pairs with the same activities (0 disagreements). |
| Same per-case instance counts? | **Same up to a constant factor per pattern.** 2,957 of 3,123 compared patterns have identical counts; 165 are counted exactly 2× and 1 exactly 4× by the original. The factor never differs between cases, and a simple rule predicts it for all 3,123 patterns (section 3.2). |
| Same interest values? | **Yes, exactly** (largest absolute difference 0.0 for information gain, coverage and case distance) on 461 + 598 patterns scored on the whole subset and on 132 + 566 + 330 patterns scored inside the original automatic mode. The double counting does not change any interest value. |
| How slow is the isomorphism deduplication (Guide Table C, row C4)? | One `Pattern_extension` pass on full f1 for `ac370000`: **30.8 s** with an empty dictionary (385 patterns; 27.5 s in an earlier run), **154.7 s** when four other core activities are already in the shared dictionary, as in the automatic mode. Ours: **0.019 s**. Step 1 for all 7 front activities of f1: original **344.5 s**, ours **0.047 s** (same 1,204 patterns). |
| Does the automatic-mode loop behave like `Auto_IMPID.py`? | **Yes**, checked step by step against a real `AutoStepWise_PPD` run (150 cases, 2 extension steps): same candidates and same values at steps 0, 1, 2, and the original Pareto front is reproduced from our values at every step. With our default rules (`distinct=False`) the front is a superset: 10/23/35 instead of 10/18/18 patterns; all 22 extra patterns are exact ties with a front pattern. |
| Runtime of our selector on the full logs | **f1 7.1 s, f2 3.5 s, f3 2.4 s** for mining + scoring, plus 0.9–1.7 s for the projection. The f1 figure includes the one-off numba compilation of paretoset: in a separate check, f1 took 9.0 s on the first `fit` of a process and 2.9 s and 3.0 s on the second and third. |
| Is k = 10 per length reachable? | Yes, but the first Pareto front alone is not enough at length 1 (7 sets in every log) and at length 3 in f3 (9 sets): the second layer is needed there. The candidate sets must come from **all evaluated patterns**; if only patterns on the step fronts could supply sets, length 1 would have 7 sets and f3 length 3 only 9. |

## 1. What was run

All commands from `C:/Users/Bohabara/Desktop/process mining/project`, with `PY = .venv/Scripts/python.exe`.

```
PY -m unittest experiments/test_impressed_chain.py -v          # 34 tests, about 5 s, all pass
PY -u experiments/c4_crosscheck_original.py --full-core ac370000 --full-front > results/experiments/C4/crosscheck_run.log 2>&1
                                                               # about 15 min (run in the background)
PY -u experiments/c4_run_selector.py > results/experiments/C4/selector_run.log 2>&1   # 24 s
```

| File | Purpose |
|---|---|
| `experiments/impressed_chain.py` | the module to reuse: `Pattern`, `count_instances`, `extend_pattern`, `extend_patterns`, `score_patterns`, `pareto_front`, `pareto_layers`, `ImpressedChainSelector` |
| `experiments/test_impressed_chain.py` | 34 unit tests on toy traces with hand-computed patterns, counts, interest values and projections |
| `experiments/c4_crosscheck_original.py` | part (f): runs the **unchanged** 2023 functions and compares (parts A–G below) |
| `experiments/c4_run_selector.py` | part (g): runs the selector on f1, f2, f3 and writes the JSON/CSV files; also holds `load_selector_inputs` |
| `results/experiments/C4/crosscheck.json`, `crosscheck_tables.md`, `crosscheck_run.log` | raw cross-check results (final run); `crosscheck_run_first.log` is an earlier run of parts A–F |
| `results/experiments/C4/impressed_sets_<log>.json` | selected sets per length with IG, coverage, CD, set support, source pattern, step; front sizes; timings |
| `results/experiments/C4/impressed_patterns_<log>.csv`, `impressed_all_sets_<log>.csv` | every evaluated pattern / every candidate set with rank and layer |
| `results/experiments/C4/selector_summary.md` | all tables of section 4, including the 30 selected sets per log |
| `results/experiments/C4/original_auto_output/` | the 36 pattern JSON files the original automatic mode wrote during part E |

Settings (Guide Part 6, Table B working defaults): logs loaded through `engine.EventLog` (same cleaning and
rare-activity filter as the 2024 `DataManager`: 1130/1130/1111 cases); chain traces ordered by `event_nr`
(`delta_time = -1` for the original code); `max_gap = 3`; 2 extension steps; interest functions information
gain (max), case coverage (max), case distance (min) with the explicit distance of `case_distance.py` on
Age + Diagnosis, Treatment code, Diagnosis code, Specialism code; `distinct=False`; undefined case distance
→ 1.0; k = 10 per length; mining and scoring on the whole log. Cross-check: 150 random f1 cases
(`numpy.random.default_rng(2023)`, 3,025 events, 132 activities); seed activities = the 5 activities that
occur in most of these cases: `ac370000`, `ac419100`, `370488j`, `ac370419`, `370407`.

How to use the selector:

```python
from c4_run_selector import load_selector_inputs      # EventLog + distance matrix in case order
from impressed_chain import ImpressedChainSelector

log, dist_matrix, _ = load_selector_inputs("f1")
selector = ImpressedChainSelector(max_gap=3, steps=2, k=10).fit(log.traces, log.labels, dist_matrix)
selector.select()         # {1: [[...], ...], 2: [...], 3: [...]}  - same format as AprioriSelector.select
selector.select_table()   # itemset, size, rank, layer, IG, coverage, CD, source_pattern, step, n_patterns
selector.patterns_        # every evaluated pattern with its interest values and step
```

## 2. The representation and how it maps to the original graphs

A pattern is `Pattern(labels, edges)`: a tuple of activity names plus one edge type (`'direct'` or
`'eventual'`) between consecutive nodes; printed as `a -> b ~> c`.

* With `delta_time < 0` the original `Trace_graph_generator` returns the path `0 -> 1 -> … -> L-1`
  (checked: 150 of 150 trace graphs are such chains, no node flagged parallel, every edge `eventually=False`).
* Every pattern graph the original then builds is a directed path (our converter raises an error otherwise;
  it never did, on 4,022 original pattern graphs). `labels` = the node `value`s along the path, an edge with
  `eventually=True` is `'eventual'`, otherwise `'direct'`; `parallel` and `color` are dropped.
* A direct edge means adjacent positions. An eventual edge means 2 to `max_gap + 1` positions apart, i.e. 1 to
  `max_gap` events in between (the original window `max(out) < node <= max(out) + Max_gap_between_events`
  starts after the direct successor, IMIPD.py:249-251).
* Patterns without an eventual edge are n-grams, so instances are found by scanning the trace lists; patterns
  with an eventual edge are never extended (Auto_IMPID.py:108-110), so they carry at most one eventual edge,
  at one end.

Extension rules replicated for chains (pattern at positions `start..end`): direct preceding, direct following,
eventually following (`end+2 … end+1+max_gap`), eventually preceding (`start-1-max_gap … start-2`), and — only
when a single activity is extended — the direct context `pred -> activity -> succ`. The concurrent rule yields
nothing on chains. Guide example reproduced as a unit test: extending `b` in `⟨a,b,c,d,e⟩` with gap 2 gives
`a->b`, `b->c`, `b~>d`, `b~>e`, `a->b->c`.

## 3. Cross-check against the original code (part f)

### 3.1 Patterns

| Part | What the original ran | Patterns original / ours | Only in one of them | Original time | Our time |
|---|---|---|---|---|---|
| A | `Pattern_extension`, 5 seed activities, fresh dictionary each (150 cases) | 475 / 475 (176, 127, 58, 56, 58) | 0 | 3.8 s (1.89, 0.88, 0.47, 0.29, 0.30) | 0.007 s |
| B | the same 5 seeds in one shared dictionary, as Auto_IMPID.py:45-73 | 461 / 461 | 0 | 33.7 s | 0.009 s |
| C | `Single_Pattern_Extender` for 16 step-1 parents (step 2) | 598 / 598 | 0 | 217.2 s | 0.021 s |
| E | whole `AutoStepWise_PPD`, 2 extension steps, 120 training cases | step 0: 132 / 132; step 1: 566 / 566; step 2: 333 rows = 330 distinct / 330 | 0 | 188.9 s (136.2 s in an earlier run) | 1.13 s (extension + scoring) |
| F | `Pattern_extension` on **full f1** (1130 cases) for `ac370000`, fresh dictionary | 385 / 385 | 0 | 30.8 s (27.5 s in an earlier run) | 0.019 s |
| G | step 1 on **full f1** for our 7 step-0 front activities, shared dictionary | 1,204 / 1,204 | 0 | 344.5 s | 0.047 s |

Part B has 14 patterns fewer than part A (461 < 475) because a pattern reached from two seeds is stored once
in the shared dictionary; ours merges them the same way.

Original seconds per core in part G: `337419c` 0.0, `376400` 2.8, `ac10307` 2.3, `ac355427` 1.6,
`ac370000` 154.7, `ac415100` 43.8, `ac419100` 139.4. The same core `ac370000` costs 30.8 s with an empty
dictionary and 154.7 s with 4 earlier cores in it: every candidate is compared with every stored pattern.
(Building the 1130 trace graphs takes another 4.9 s; not included above.) The slowest single step-2 parent on
the 150-case subset was `ac370000 -> ac370000`: 64.0 s for 97 children, because the tool recounts all count
columns after every instance (IMIPD.py:741-746).

Isomorphism versus tuple equality (`nx.is_isomorphic` with the original's node and edge matchers): 15,150 pairs
in part A, 3,141 in F, 3,516 in G; of these 150 + 141 + 516 pairs have the same multiset of activities (e.g.
`a -> b` / `a ~> b` / `b -> a`); **0 disagreements**. The number of distinct tuples equals the number of
dictionary entries in every dictionary.

### 3.2 Per-case instance counts: every difference explained

| Part | Patterns | Identical counts | Original = 2 × ours | Original = 4 × ours | Factor differs between cases | Factor predicted by the rule |
|---|---|---|---|---|---|---|
| A (5 seeds) | 475 | 472 | 3 | 0 | 0 | 475 |
| B (shared dictionary, instance lists) | 461 | 444 | 17 | 0 | 0 | 461 |
| B (count columns the tool writes) | 461 | 458 | 3 | 0 | 0 | – |
| C (16 parents, step 2) | 598 | 500 | 97 | 1 | 0 | 598 |
| F (full f1, 1 core) | 385 | 383 | 2 | 0 | 0 | 385 |
| G (full f1, 7 cores) | 1,204 | 1,158 | 46 | 0 | 0 | 1,204 |

The original counts some instances more than once. The rule (derived from reading the original code, then
confirmed for all patterns above; one first guess was wrong, see section 7):

1. **Step 1, repeated activity.** `a -> a` at positions (i, i+1) is found once as "a followed by a" from node i
   and once as "a preceded by a" from node i+1; `a ~> a` likewise. Both land in the same dictionary entry →
   factor 2 (the 3 patterns in A, the 2 in F).
2. **Step 1, shared dictionary.** A 2-node pattern `x -> y` / `x ~> y` is found once from every node whose
   activity is one of the extended core activities, so the *instance list* holds each instance twice when both
   `x` and `y` are cores (17 patterns in B, 46 in G). The *count column* is written when the first core's loop
   ends (Auto_IMPID.py:68-73) and is not updated later, so only the repeated-activity patterns of rule 1 have a
   doubled column (3 columns in B).
3. **Step 2.** `Single_Pattern_Extender` extends every entry of the parent's instance list, so a parent with a
   doubled list doubles all its children (the 96 children of `ac370000 -> ac370000`). In addition, a child made
   of one repeated activity with only direct edges is reached as "preceding" from one parent instance and as
   "following" from another: `a -> a -> a` from `a -> a` is counted 2 × 2 = **4** times, `a -> a -> a -> a` from
   the context `a -> a -> a` 2 times.

Our code keeps the instances of a pattern in a set of position tuples, so each instance is counted once. As a
second, independent check, the counts of 300 random evaluated patterns per full log were recomputed with
the plain scan `count_instances`: 0 of 900 differ.

### 3.3 Interest values

| Compared | Patterns | Scored on | max abs diff IG / coverage / CD | NaN mismatches | Original time | Our time |
|---|---|---|---|---|---|---|
| part B patterns (step 1) | 461 | all 150 cases | 0.0 / 0.0 / 0.0 | 0 | 18.6 s | 1.05 s |
| part C patterns (step 2) | 598 | all 150 cases | 0.0 / 0.0 / 0.0 | 0 | 23.2 s | 0.93 s |
| part E step 0 | 132 | 120 training cases | 0.0 / 0.0 / 0.0 | 0 | inside the 188.9 s | inside the 1.13 s |
| part E step 1 | 566 | 120 training cases | 0.0 / 0.0 / 0.0 | 0 | | |
| part E step 2 | 330 distinct | 120 training cases | 0.0 / 0.0 / 0.0 | 0 | | |

The original values come from the unchanged `create_pattern_attributes` on the count columns the tool itself
wrote (including the doubled ones); ours from `score_patterns` on our counts. Both sides were given the same
distances (our explicit distance; which distance to use was settled in C12). Value ranges in parts B/C, to show
the comparison is not trivial: IG 0.0001–0.2366, coverage 0.0067–0.6, CD 0.5005–0.8124. The differences are
exactly 0 because (i) information gain treats the count as a category, so doubling every count changes nothing,
(ii) coverage and case distance only use "count > 0", and (iii) both sides average the same distances in the
same order. In part E the original case distance is NaN for 14 / 81 / 29 candidates (activities or patterns
absent from the 120 training cases); ours is NaN in exactly the same rows when asked for NaN.

### 3.4 The automatic mode, step by step (part E)

The original `AutoStepWise_PPD` was run unchanged (instrumented from outside) on the 150 cases with
`Max_extension_step=2`, test share 0.2. At every step we extended **the original's own front of the previous
step** with our code and scored on the same 120 training cases.

| Step | Candidates original / ours | Original front | Original front reproduced from our values (`distinct=True`, NaN kept, same row order) | Front with `distinct=True`, NaN → 1.0 | Front by our rules (`distinct=False`, NaN → 1.0) | Extra patterns that tie exactly with a front pattern |
|---|---|---|---|---|---|---|
| 0 | 132 / 132 | 10 | yes | 10 | 10 | 0 of 0 |
| 1 | 566 / 566 | 18 | yes | 18 | 23 | 5 of 5 |
| 2 | 333 (330 distinct) / 330 | 18 | yes | 18 | 35 | 17 of 17 |

* The only difference between the original front and ours is `distinct`: paretoset's default keeps one pattern
  of each group with identical interest values, ours keeps all (decision from C11). Replacing NaN by 1.0 changed
  nothing in this run.
* Step 2 of the original contains 3 duplicate rows: the same graph reached from two parents gets two IDs,
  because the dictionary is per parent from step 2 on (IMIPD.py:617). Ours scores each pattern once.
* Not done: a free-running comparison of whole runs. It would not be like for like, because our selector scores
  on the whole log by design (Table B) while the original scores on an 80 % split.

## 4. The selector on full f1, f2, f3 (part g)

### 4.1 Runtime and front sizes per step

| Log | Cases | Step 0 candidates → front | Step 1 candidates → front (extendable) | Step 2 candidates → front | Patterns evaluated | Load + distance | Fit | Projection |
|---|---|---|---|---|---|---|---|---|
| f1 | 1130 | 164 → 7 | 1,204 → 26 (16) | 851 → 50 | 2,109 | 0.3 s | 7.1 s | 1.1 s |
| f2 | 1130 | 207 → 7 | 1,634 → 44 (21) | 1,175 → 64 | 2,893 | 0.4 s | 3.5 s | 1.7 s |
| f3 | 1111 | 156 → 7 | 1,031 → 23 (11) | 861 → 39 | 1,946 | 0.3 s | 2.4 s | 0.9 s |

"Extendable" = front patterns without an eventual edge; only those are extended in step 2. The f1 fit time
includes the one-off numba compilation of paretoset (separate check: 9.0 s for the first `fit` in a process,
2.9 s and 3.0 s for the next two). "Patterns evaluated" counts each pattern
once (some step-2 children were already scored in step 1 as context patterns). Number of distinct activities
of the evaluated patterns — f1: 175 / 1,085 / 672 / 177 with 1 / 2 / 3 / 4 activities; f2: 218 / 1,454 / 968 / 253;
f3: 167 / 947 / 701 / 131. Patterns with 4 distinct activities are dropped; 4-node patterns with a repeated
activity are kept.

### 4.2 Sets per length

| Log | Length | Candidate sets | First front | Layers needed for k = 10 | Sets if only step-front patterns could supply them | Median coverage of the source pattern | Median support of the set | Selected sets whose source pattern is in < 10 cases | Also in Apriori's top 10 of that size |
|---|---|---|---|---|---|---|---|---|---|
| f1 | 1 | 164 | 7 | 2 | 7 | 0.262 | 0.262 | 1 | 3 |
| f1 | 2 | 460 | 14 | 1 | 17 | 0.191 | 0.239 | 0 | 0 |
| f1 | 3 | 400 | 21 | 1 | 21 | 0.012 | 0.032 | 3 | 0 |
| f2 | 1 | 207 | 7 | 2 | 7 | 0.254 | 0.302 | 2 | 3 |
| f2 | 2 | 647 | 34 | 1 | 27 | 0.250 | 0.318 | 0 | 1 |
| f2 | 3 | 619 | 45 | 1 | 28 | 0.113 | 0.256 | 0 | 0 |
| f3 | 1 | 156 | 7 | 2 | 7 | 0.334 | 0.334 | 2 | 4 |
| f3 | 2 | 378 | 14 | 1 | 13 | 0.115 | 0.190 | 0 | 0 |
| f3 | 3 | 347 | 9 | 2 | 9 | 0.010 | 0.038 | 3 | 0 |

"Support of the set" = share of cases that contain all activities of the set in any order (what Apriori calls
support, and what decides how many cases the 2024 permutation step can shuffle); it is always ≥ the coverage of
the source pattern. Apriori baseline for the last column: `AprioriSelector`, top 10 per size, `min_support`
0.5 (f1, f2) and 0.4 (f3).

The 30 selected sets per log, with all values, are in `selector_summary.md` and `impressed_sets_<log>.json`.
First three per length on f1 (IG / coverage / CD):

| Length | Rank 1 | Rank 2 | Rank 3 |
|---|---|---|---|
| 1 | `376400` (0.368 / 0.296 / 0.662) | `ac419100` (0.336 / 0.645 / 0.661) | `ac370000` (0.245 / 0.701 / 0.629) |
| 2 | `370715a, 376400` from `370715a ~> 376400` (0.297 / 0.250 / 0.669) | `370712b, 376400` from `370712b ~> 376400` (0.291 / 0.247 / 0.667) | `ac411100, ac419100` from `ac411100 -> ac419100` (0.145 / 0.360 / 0.640) |
| 3 | `376400, 377498a, ac372417` from `ac372417 -> 376400 -> 377498a` (0.157 / 0.150 / 0.666) | `ac411100, ac415100, ac419100` from `ac411100 -> ac419100 -> ac415100` (0.030 / 0.079 / 0.584) | `ac10307, ac411100, ac419100` from `ac10307 -> ac411100 -> ac419100 -> ac411100` (0.016 / 0.051 / 0.579) |

Where the selected sets come from (source step): length 1 — step 0 except one set in f2; length 2 — step 1
(9, 10, 9 of 10); length 3 — step 1 for 4, 2, 4 sets and step 2 for 6, 8, 6 sets in f1, f2, f3. So step 2 is
needed for length 3.

### 4.3 Things the group should know before using these sets

1. **Rare sets get selected.** A pattern that occurs in a handful of cases can sit on the Pareto front because
   its case distance is the lowest. Examples: f1 length 1 rank 7 `337419c` (3 cases), f1 length 3 rank 9
   (source pattern in 2 cases, the set in 5), f3 length 3 ranks 7 and 8 (source pattern in 1 case). The 2024
   permutation step only shuffles cases that contain the whole set, so such sets will get an importance near 0
   by construction. Selected sets whose *set support* is below 10 cases: f1 1 (length 1) + 1 (length 3);
   f2 2 (length 1); f3 2 (length 1) + 1 (length 3).
2. **The tie-break favours information gain inside a layer, not across layers.** Ranks 1–7 at length 1 are the
   whole first front (down to IG 0.0014), and `ac411100` (IG 0.170, coverage 0.459) only comes at rank 8
   because it is in layer 2. This is what "fill by layers" means; it is listed here so nobody is surprised.
3. **A set can be represented by a pattern that repeats an activity.** In f2 the set `{ac419100}` is
   represented by `ac419100 ~> ac419100` (IG 0.033, coverage 0.209), not by the single activity (which occurs
   in 93.7 % of the cases and has the lower IG 0.021), because the rule is "best non-dominated pattern of the
   set, highest IG". Five of the 90 selected sets have a source pattern with more nodes than activities.
4. **Almost no overlap with Apriori.** At lengths 2 and 3, 0 or 1 of the 10 sets is also among Apriori's 10 most
   frequent sets; at length 1, 3–4 of 10. The comparison of the two strategies can therefore not be made on
   shared sets; it has to compare the two groups of sets (e.g. distributions of importance).
5. **Coverage of a pattern is not the support of its set** (median columns above): at length 3 in f1 the
   median source pattern is in 1.2 % of the cases while the median set is in 3.2 %.

## 5. What this means for the project decisions

* **B10 / implementation strategy:** use the chain-only re-implementation. On chains it finds exactly the
  patterns of the original code, gives exactly the same interest values, and needs seconds instead of minutes
  (step 1 on full f1: 0.047 s vs 344.5 s). The original code is only needed as a cross-check, which now exists
  as a script.
* **Guide Part 6, Table C, row C4** can be closed with measured values: 30.8 s (empty dictionary) / 154.7 s
  (shared dictionary) for `ac370000` on full f1, 385 patterns (after the 2024 filter, gap 3); the earlier
  "≈160 s with 435 patterns" was not reproduced as such but is of the same order as the shared-dictionary case.
  Tuple-key deduplication is equivalent on chains.
* **Guide §4.5 "cross-core merge … doubling derived step-2 counts"** is confirmed and now quantified (factors 2
  and 4, uniform per pattern), with the addition that it is harmless for the three interest functions; it only
  changes the tool's `Pattern_Frequency` column.
* **B7 (front versus fixed k):** k = 10 per length works with layer filling; report the first-front sizes of
  section 4.2 next to it. Candidate sets must come from all evaluated patterns.
* **Open, needs the group's or the supervisor's decision:** (a) a minimum support for selected sets, or accept
  that rare sets get importance ≈ 0 (point 1); (b) whether a length-1 set should always be represented by the
  plain activity (point 3); (c) how the two strategies are compared when they share almost no sets (point 4).

## 6. Deliberate differences from the original code

Listed in the docstring of `impressed_chain.py`: each instance counted once; one pattern per graph even when
reached from several parents; interest values on all cases passed to `fit` (the original scores on an 80 %
split but extends on the whole log); `distinct=False`; undefined case distance → 1.0 instead of NaN; the
distance matrix is an input; `steps` counts extension steps after step 0 (`steps=2` ≙ `Max_extension_step=2`).
The projection to activity sets and the selection of k sets per length are new (project logic, not IMPresseD).

## 7. What failed, what was not done

* Nothing crashed. One wrong prediction of mine was corrected during the work: the 3-node context
  `a -> a -> a` is stored once, not twice, by the original; the rule in section 3.2 is the corrected one.
* The cross-check uses f1 only (150-case subset for parts A–E, the full log for step 1 in parts F–G). The
  original step 2 was **not** run on a full log by this experiment, and f2/f3 were not run through the original
  code; they use the same code paths.
* Part E compares step by step given the original's fronts; whole free-running runs were not compared (see 3.4).
* The original was always fed our explicit case distance, so this experiment says nothing about
  `pdist('jaccard')` (that is C12).
* Timings were taken while other experiments were running; e.g. `AutoStepWise_PPD` took 136.2 s in the first
  run and 188.9 s in the final one.
* pytest, pycodestyle and pyflakes are not installed in the venv: tests use `unittest`; line length (≤ 120) and
  unused imports were checked with a small script.
