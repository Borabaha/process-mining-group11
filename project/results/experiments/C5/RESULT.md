# Experiments C5 + C6 — sensitivity of the IMPresseD side (oracle, gap, objectives, `distinct`, k)

Run on 2026-10-02, Windows 10, Python 3.12 from `project/.venv` (numpy 2.5.3, pandas 2.3.3, scikit-learn 1.9.1,
networkx 3.7, paretoset 1.2.5). Every number below was produced by the commands in section 1; nothing is
estimated. Other experiments were running on the same machine, so the timings are noisy; all counts and sets were
identical in the three complete runs of the sensitivity script.

Feeds Guide Part 6: Table C rows **C5** and **C6**; decisions **B12** (oracle and gap), **B1 / B7 / B8**
(objectives, front versus k, ranking).

## 0. Short answers

| Question | Answer |
|---|---|
| C5a: what happens with `delta_time = 0`? | 88.8 % / 87.1 % / 88.4 % of consecutive events (f1 / f2 / f3) have the same timestamp. 94 % of all events are flagged concurrent, 73.7 % / 68.4 % / 73.5 % of all events sit in same-day blocks of 10 or more events, and 412 / 110 / 401 traces become one single block with no order at all. |
| C5a: why chain traces? | With `delta_time = 0` the original extension rules build patterns out of whole same-day blocks (the concurrent pattern around an event has a median of 25 / 28 / 22 nodes, and 20–35 % of the direct-following and direct-preceding patterns have more than 3 nodes), so most patterns cannot be turned into sets of 1–3 activities at all. Chain traces (`delta_time < 0`, order = `event_nr`) use exactly the order that the 2024 pipeline already uses as "location", keep every step-1 pattern at 2–3 nodes, and are the setting for which the verified, fast re-implementation exists. |
| C5b: does `max_gap` matter for the final sets? | **Not for length 1** (10 of 10 sets identical for every pair of gaps, except f2 with gap 5: 9 of 10). **Yes for lengths 2 and 3**: compared with gap 3, gap 2 keeps 8 / 6 / 8 of the 10 length-2 sets and 6 / 4 / 8 of the 10 length-3 sets (f1 / f2 / f3); gap 5 keeps 7 / 5 / 7 and 5 / 6 / 7. At lengths 2 and 3 only 3–5 of the 10 sets per log are selected under all four gaps. |
| C6: front sizes with three and with two objectives | Three objectives (IG, coverage, CD), per step 0 / 1 / 2: f1 7 / 26 / 50, f2 7 / 44 / 64, f3 7 / 23 / 39. Two objectives (IG, coverage): f1 3 / 2 / 1, f2 4 / 6 / 5, f3 1 / 1 / 2. With two objectives the mining collapses (f3 evaluates 573 patterns instead of 1,946) and the first front per length holds only 1–8 sets, so k = 10 needs 2–6 layers. |
| C6: step-2 patterns by number of distinct activities | f1 3 / 231 / 440 / 177 with 1 / 2 / 3 / more than 3 activities (851 candidates), f2 3 / 280 / 639 / 253 (1,175), f3 3 / 217 / 510 / 131 (861). So 21 % / 22 % / 15 % of the step-2 candidates have 4 distinct activities and are dropped by the projection. |
| C6: `distinct=True` versus `False` | Fronts shrink (step 2: 50 → 35, 64 → 42, 39 → 31; first front of length-3 sets: 21 → 11, 45 → 24, 9 → 8), but the selected 30 sets stay the same in f1 and f2 and change by one set in f3. Keep `distinct=False`: `True` removes a set only because another set has exactly the same three values. |
| C6: is k = 10 always reachable? | **Yes, if sets may come from all evaluated patterns**: the smallest number of candidate sets in any of the 27 runs is 156 (length 1), 106 (length 2) and 116 (length 3). **No, if only patterns on the step fronts may supply sets**: 7 sets at length 1 in every log, 9 at length 3 in f3 (default run), down to 0 with two objectives. |
| Overlap with Apriori's top 10 per size | f1 3 / 0 / 0, f2 3 / 1 / 0, f3 4 / 0 / 0 shared sets at length 1 / 2 / 3 (default run). No gap value changes this by more than one set. |

## 1. What was run

All commands from `C:/Users/Bohabara/Desktop/process mining/project`, with `PY = .venv/Scripts/python.exe`.

```
PY -u experiments/c5_oracle_blocks.py   > results/experiments/C5/c5a_run.log  2>&1    # C5a, 88.5 s and 94.4 s (two runs)
PY -u experiments/c5_c6_sensitivity.py  > results/experiments/C5/c5c6_run.log 2>&1    # C5b + C6, 260.6 - 292.3 s (three runs)
PY results/experiments/C5/check_gap_counts.py                                         # sanity check, not timed
```

| File | Purpose |
|---|---|
| `experiments/c5_oracle_blocks.py` | C5a: equal-timestamp pairs, same-timestamp blocks, and the trace graphs of the **original** `IMIPD.Trace_graph_generator` with `delta_time = 0` and `-1` |
| `experiments/c5_c6_sensitivity.py` | C5b + C6: 9 configurations of `ImpressedChainSelector` per log, overlap tables, Apriori comparison |
| `results/experiments/C5/c5a_blocks.json`, `c5a_tables.md`, `c5a_run.log` | C5a numbers |
| `results/experiments/C5/c5c6_results.json`, `c5c6_tables.md`, `c5c6_run.log` | C5b + C6 numbers; `c5c6_tables.md` section 9 lists the 10 selected sets per gap, log and length |
| `results/experiments/C5/sets_vs_apriori.md` | IMPresseD sets next to the Apriori top 10 (also in the appendix below) |
| `results/experiments/C5/check_gap_counts.py`, `check_gap_counts.log` | recount of 300 random patterns per log for gaps 1, 2, 5 |

Settings.

* Logs: BPIC11 f1, f2, f3 through `engine.EventLog` (same cleaning and rare-activity filter as the 2024
  `DataManager`: 1130 / 1130 / 1111 cases, 164 / 207 / 156 activities). C5a also reports the pair share on the raw
  files (1140 / 1140 / 1121 cases).
* Selector: `experiments/impressed_chain.py` unchanged (sha256 `1f7164a6…7251`, stored in `c5c6_results.json`),
  inputs from `c4_run_selector.load_selector_inputs` (chain traces in `event_nr` order, explicit case distance on
  Age + Diagnosis, Treatment code, Diagnosis code, Specialism code).
* Default configuration ("gap3 (default)", Guide Part 6 Table B): `max_gap = 3`, 2 extension steps, objectives
  IG (max), coverage (max), CD (min), `distinct=False`, undefined CD → 1.0, k = 10 per length, lengths 1–3,
  candidate sets from all evaluated patterns, mining and scoring on the whole log. Its 30 selected sets per log are
  identical to the ones stored by experiment C4 (checked in the script: `default_equals_C4_selection = true` for
  the three logs).
* One setting is changed at a time: `max_gap` 1, 2, 5; objectives (IG, coverage); `distinct=True`; both of the
  last two together; and "rounded" = the three interest values rounded to 12 decimals before every Pareto
  computation (with `distinct` False and True). The rounding is done by wrapping `impressed_chain.score_patterns`
  from the experiment script, so the verified module was not edited.
* Two kinds of comparison are used for the objectives and for `distinct`:
  **free run** = the whole mining is repeated with the changed setting (the step fronts decide which patterns are
  extended, so the candidates change); **same candidates** = the patterns of the default run are kept and only
  the front / ranking rule is changed.
* Nothing is random in these scripts (the selector has no random part; `check_gap_counts.py` samples with seed 2023).

## 2. C5a — the original conversion with `delta_time = 0`

The original `Trace_graph_generator` (IMIPD.py:331-400) flags two consecutive events as parallel when their
timestamps differ by at most `delta_time` seconds. All BPIC11 timestamps have the time of day 23:00 (checked: one
distinct value in each log), so with `delta_time = 0` every run of events of a case on the same day becomes one
concurrent block.

### 2.1 Consecutive event pairs with equal timestamps

| Log | Cases (raw / filtered) | Events (filtered) | Consecutive pairs | Pairs with equal timestamp | Share (filtered log) | Share (raw log) | Pairs with decreasing timestamp |
|---|---|---|---|---|---|---|---|
| f1 | 1140 / 1130 | 23,853 | 22,723 | 20,185 | **0.8883** | 0.8877 | 0 |
| f2 | 1140 / 1130 | 30,929 | 29,799 | 25,964 | **0.8713** | 0.8711 | 0 |
| f3 | 1121 / 1111 | 20,227 | 19,116 | 16,901 | **0.8841** | 0.8846 | 0 |

The Guide's figures (0.8877 / 0.8711 / 0.8846, §4.3 and B12) are the raw-log values and are reproduced exactly.
Along `event_nr` the timestamps never decrease.

### 2.2 Same-timestamp blocks (filtered logs)

A "block" is a maximal run of consecutive events of one case with the same timestamp; a block of size 1 is an
ordinary sequential event.

| Log | Blocks | Mean size | Median size | 90 % quantile | Largest | Block size seen by the median event | Events in blocks ≥ 2 (flagged parallel) | **Events in blocks ≥ 10** | Parallel events that sit in blocks ≥ 10 | Cases with a block ≥ 10 | Traces that are one single block | Median blocks per case | Median trace length |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f1 | 3,668 | 6.50 | 2 | 25 | 36 | 25 | 94.3 % (22,487) | **73.7 %** (17,583) | 78.2 % | 690 of 1130 | 412 (36.5 %) | 3 | 25 |
| f2 | 4,965 | 6.23 | 2 | 27 | 40 | 26 | 93.9 % (29,042) | **68.4 %** (21,163) | 72.9 % | 713 of 1130 | 110 (9.7 %) | 4 | 40 |
| f3 | 3,326 | 6.08 | 2 | 23 | 31 | 22 | 93.5 % (18,918) | **73.5 %** (14,874) | 78.6 % | 614 of 1111 | 401 (36.1 %) | 2 | 21 |

Distribution of the block sizes (number of blocks; share of all events in brackets):

| Block size | f1 | f2 | f3 |
|---|---|---|---|
| 1 | 1,366 (5.7 %) | 1,887 (6.1 %) | 1,309 (6.5 %) |
| 2 | 925 (7.8 %) | 1,131 (7.3 %) | 906 (9.0 %) |
| 3 | 246 (3.1 %) | 331 (3.2 %) | 191 (2.8 %) |
| 4–5 | 204 (3.8 %) | 332 (4.7 %) | 157 (3.4 %) |
| 6–9 | 198 (5.9 %) | 425 (10.2 %) | 132 (4.8 %) |
| 10–19 | 148 (8.6 %) | 250 (10.5 %) | 123 (8.7 %) |
| 20 or more | 581 (65.1 %) | 609 (57.9 %) | 508 (64.8 %) |

Most blocks are small (median 2), but most *events* live in very large blocks: the median event sits in a block
of 25 / 26 / 22 events. The earlier probe "78 % of the parallel-flagged f1 events are in blocks of ≥ 10 nodes"
(Guide C5, until now marked "not re-run") is reproduced: 78.2 %.

### 2.3 Cross-check with the original code

The trace graphs were built for every case with the unchanged `IMIPD.Trace_graph_generator`.

| Log | `delta_time` | Nodes | Nodes flagged parallel | Edges | Edges of the chains | Largest out-degree | Graphs without any edge | Cases where the flags differ from the block rule of 2.2 |
|---|---|---|---|---|---|---|---|---|
| f1 | 0 | 23,853 | 22,487 (94.3 %) | 37,909 | 22,723 | 32 | 412 | 0 |
| f1 | −1 | 23,853 | 0 | 22,723 | 22,723 | 1 | 0 | 0 |
| f2 | 0 | 30,929 | 29,042 (93.9 %) | 83,410 | 29,799 | 35 | 110 | 0 |
| f2 | −1 | 30,929 | 0 | 29,799 | 29,799 | 1 | 0 | 0 |
| f3 | 0 | 20,227 | 18,918 (93.5 %) | 25,921 | 19,116 | 29 | 401 | 0 |
| f3 | −1 | 20,227 | 0 | 19,116 | 19,116 | 1 | 0 | 0 |

The number of parallel-flagged nodes equals the number of events in blocks ≥ 2 in every case, so the block
statistics describe what the original code really builds. With `delta_time = −1` every graph is the chain.

Size of the patterns the original step-1 rules would build on these graphs, taking every event as the core once.
The node sets are built exactly as in `IMIPD.Pattern_extension` (IMIPD.py:181-242: all predecessors + core, all
successors + core, all parallel nodes with the same neighbours); the pattern dictionary and its isomorphism
tests were **not** run.

| Log | Rule (`delta_time = 0`) | Patterns built | Median number of nodes | Largest | More than 3 nodes | More than 3 distinct activities | 10 or more nodes |
|---|---|---|---|---|---|---|---|
| f1 | concurrent | 22,487 | 25 | 36 | 88.5 % | 87.6 % | 78.2 % |
| f1 | direct following | 11,724 | 3 | 33 | 30.8 % | 28.7 % | 8.3 % |
| f1 | direct preceding | 8,982 | 3 | 36 | 31.5 % | 29.5 % | 12.4 % |
| f2 | concurrent | 29,042 | 28 | 40 | 88.8 % | 87.7 % | 72.9 % |
| f2 | direct following | 24,602 | 3 | 36 | 33.7 % | 31.1 % | 11.1 % |
| f2 | direct preceding | 14,437 | 3 | 40 | 35.4 % | 33.5 % | 17.6 % |
| f3 | concurrent | 18,918 | 22 | 31 | 87.4 % | 86.5 % | 78.6 % |
| f3 | direct following | 10,129 | 2 | 30 | 20.4 % | 17.7 % | 5.2 % |
| f3 | direct preceding | 6,937 | 3 | 31 | 27.4 % | 25.3 % | 10.8 % |

With `delta_time = −1` every direct-following and direct-preceding pattern has exactly 2 nodes and the concurrent
rule builds nothing. The second earlier probe ("31 % of the directly-following instances have more than 3 nodes",
f1) is reproduced: 30.8 %. Note also that with `delta_time = 0` only 11,724 of the 23,853 f1 events have any
successor at all (49 %; f2 80 %, f3 50 %): the events of the last block of a trace have none.

### 2.4 Conclusion (two sentences)

With `delta_time = 0`, 87–89 % of consecutive events share a timestamp, 94 % of the events are flagged concurrent
and 68–74 % of all events sit in same-day blocks of 10 or more events, so the original rules build patterns that
contain whole blocks (median concurrent pattern 22–28 nodes; 20–35 % of the direct patterns have more than 3
nodes) and cannot be projected to activity sets of length 1–3, while 36 % / 10 % / 36 % of the traces lose their
order completely. Chain traces (`delta_time < 0`, events ordered by `event_nr`) use the same order that the 2024
pipeline uses as "location", keep the patterns small, and are the case for which the chain-only re-implementation
was verified against the original code (experiment C4).

## 3. C5b — `max_gap` in {1, 2, 3, 5}

An eventual edge `a ~> b` means that 1 to `max_gap` events lie between `a` and `b`. The value is a parameter of
the 2023 code only (no value is given in the paper or the code; 3 is the project's working default).

### 3.1 Patterns per step and front sizes (three objectives, `distinct=False`)

"a → b (c)" = a candidates, b on the Pareto front of the step, c of them without an eventual edge (only those are
extended in the next step).

| Log | Gap | Step 0 | Step 1 | Step 2 | Patterns evaluated |
|---|---|---|---|---|---|
| f1 | 1 | 164 → 7 | 922 → 33 (21) | 678 → 45 | 1,637 |
| f1 | 2 | 164 → 7 | 1,084 → 26 (15) | 717 → 40 | 1,873 |
| f1 | 3 | 164 → 7 | 1,204 → 26 (16) | 851 → 50 | 2,109 |
| f1 | 5 | 164 → 7 | 1,355 → 27 (16) | 1,054 → 55 | 2,450 |
| f2 | 1 | 207 → 7 | 1,234 → 33 (25) | 875 → 67 | 2,151 |
| f2 | 2 | 207 → 7 | 1,472 → 42 (19) | 947 → 54 | 2,503 |
| f2 | 3 | 207 → 7 | 1,634 → 44 (21) | 1,175 → 64 | 2,893 |
| f2 | 5 | 207 → 7 | 1,869 → 43 (22) | 1,689 → 81 | 3,612 |
| f3 | 1 | 156 → 7 | 779 → 26 (17) | 598 → 38 | 1,429 |
| f3 | 2 | 156 → 7 | 926 → 22 (13) | 733 → 38 | 1,712 |
| f3 | 3 | 156 → 7 | 1,031 → 23 (11) | 861 → 39 | 1,946 |
| f3 | 5 | 156 → 7 | 1,166 → 21 (12) | 943 → 45 | 2,163 |

Step 0 does not depend on the gap. From gap 1 to gap 5 the number of step-1 candidates grows by 47–51 % and the
number of evaluated patterns by 50–68 %; the step fronts stay in the same range (21–44 at step 1, 38–81 at
step 2). One fit took between about 1 and 15 seconds (the first fit of a process includes the numba compilation
of paretoset); the fit times of the last run are in `c5c6_tables.md` section 3.

Sanity check of the counts for the other gaps (`check_gap_counts.py`): for gaps 1, 2 and 5 on each log, 300
random evaluated patterns were recounted with the plain scan `count_instances`; 0 of 2,700 differ.

### 3.2 Candidate sets and first fronts per length

"x / y" = number of candidate sets of that length / size of the first Pareto front among them.

| Log | Gap | Length 1 | Length 2 | Length 3 | Selected sets whose source pattern has an eventual edge (length 2 / 3) |
|---|---|---|---|---|---|
| f1 | 1 | 164 / 7 | 303 / 17 | 349 / 19 | 4 / 4 |
| f1 | 2 | 164 / 7 | 404 / 15 | 371 / 24 | 7 / 5 |
| f1 | 3 | 164 / 7 | 460 / 14 | 400 / 21 | 8 / 3 |
| f1 | 5 | 164 / 7 | 545 / 19 | 482 / 29 | 7 / 7 |
| f2 | 1 | 207 / 7 | 422 / 20 | 527 / 30 | 3 / 4 |
| f2 | 2 | 207 / 7 | 567 / 34 | 569 / 44 | 6 / 7 |
| f2 | 3 | 207 / 7 | 647 / 34 | 619 / 45 | 8 / 8 |
| f2 | 5 | 207 / 6 | 782 / 30 | 821 / 58 | 7 / 9 |
| f3 | 1 | 156 / 7 | 243 / 14 | 285 / 11 | 6 / 4 |
| f3 | 2 | 156 / 7 | 331 / 17 | 330 / 11 | 8 / 6 |
| f3 | 3 | 156 / 7 | 378 / 14 | 347 / 9 | 9 / 5 |
| f3 | 5 | 156 / 7 | 444 / 20 | 388 / 10 | 9 / 8 |

k = 10 is reached in all 36 cells. Layer 2 is needed at length 1 everywhere (first front 6–7) and at length 3 in
f3 with gap 3 (first front 9).

### 3.3 Overlap of the selected sets between gaps

Cells: number of sets selected under both gaps (Jaccard index = shared / union). 10 sets per length, 30 in total.

| Log | Gaps | Length 1 | Length 2 | Length 3 | All 30 sets |
|---|---|---|---|---|---|
| f1 | 1 vs 2 | 10 (1.00) | 9 (0.82) | 6 (0.43) | 25 (0.71) |
| f1 | 1 vs 3 | 10 (1.00) | 8 (0.67) | 7 (0.54) | 25 (0.71) |
| f1 | 1 vs 5 | 10 (1.00) | 7 (0.54) | 6 (0.43) | 23 (0.62) |
| f1 | 2 vs 3 | 10 (1.00) | 8 (0.67) | 6 (0.43) | 24 (0.67) |
| f1 | 2 vs 5 | 10 (1.00) | 7 (0.54) | 7 (0.54) | 24 (0.67) |
| f1 | 3 vs 5 | 10 (1.00) | 7 (0.54) | 5 (0.33) | 22 (0.58) |
| f2 | 1 vs 2 | 10 (1.00) | 6 (0.43) | 6 (0.43) | 22 (0.58) |
| f2 | 1 vs 3 | 10 (1.00) | 5 (0.33) | 3 (0.18) | 18 (0.43) |
| f2 | 1 vs 5 | 9 (0.82) | 5 (0.33) | 3 (0.18) | 17 (0.40) |
| f2 | 2 vs 3 | 10 (1.00) | 6 (0.43) | 4 (0.25) | 20 (0.50) |
| f2 | 2 vs 5 | 9 (0.82) | 4 (0.25) | 5 (0.33) | 18 (0.43) |
| f2 | 3 vs 5 | 9 (0.82) | 5 (0.33) | 6 (0.43) | 20 (0.50) |
| f3 | 1 vs 2 | 10 (1.00) | 8 (0.67) | 7 (0.54) | 25 (0.71) |
| f3 | 1 vs 3 | 10 (1.00) | 7 (0.54) | 7 (0.54) | 24 (0.67) |
| f3 | 1 vs 5 | 10 (1.00) | 6 (0.43) | 5 (0.33) | 21 (0.54) |
| f3 | 2 vs 3 | 10 (1.00) | 8 (0.67) | 8 (0.67) | 26 (0.76) |
| f3 | 2 vs 5 | 10 (1.00) | 7 (0.54) | 6 (0.43) | 23 (0.62) |
| f3 | 3 vs 5 | 10 (1.00) | 7 (0.54) | 7 (0.54) | 24 (0.67) |

Sets and activities that survive every gap:

| Log | Length | Sets selected under all four gaps | Different sets selected under at least one gap | Activities used under all four gaps | Activities used under at least one gap |
|---|---|---|---|---|---|
| f1 | 1 | 10 | 10 | 10 | 10 |
| f1 | 2 | 5 | 14 | 7 | 15 |
| f1 | 3 | 3 | 17 | 7 | 19 |
| f2 | 1 | 9 | 11 | 9 | 11 |
| f2 | 2 | 3 | 21 | 7 | 17 |
| f2 | 3 | 3 | 23 | 6 | 17 |
| f3 | 1 | 10 | 10 | 10 | 10 |
| f3 | 2 | 5 | 16 | 6 | 13 |
| f3 | 3 | 5 | 18 | 9 | 13 |

Where do the sets that another gap selects, but gap 3 does not, stand in the gap-3 run? (Full table:
`c5c6_tables.md` section 4c.) Summed over the three other gaps: at length 2, 29 such sets, of which 17 have rank
11–30 in the gap-3 run and 12 a rank above 30, none is missing; at length 3, 37 such sets, of which 7 have rank
11–30, 17 a rank above 30, and 13 are **not candidate sets at all** in the gap-3 run (f1: 9, f2: 4) because the
gap changes the step fronts and therefore which patterns are extended in step 2.

### 3.4 Reading

* **Length 1 does not depend on the gap**, with one exception that is worth knowing. Step 0 never uses the gap,
  but a set is represented by its "best" pattern, and for `{ac419100}` in f2 that is the self-loop
  `ac419100 ~> ac419100` at gaps 2, 3 and 5 (it has a higher information gain than the plain activity: 0.031 /
  0.033 / 0.033 against 0.021). At gaps 2 and 3 the set is still rank 4 (first front); at gap 5 the self-loop's
  values (IG 0.033, coverage 0.236, CD 0.632) put the set in **layer 3, rank 17**, so the most frequent activity
  of f2 (in 93.7 % of the cases, on the step-0 front) drops out of the top 10. This is an artefact of the
  representation rule, not of the data (checked ad hoc on the fitted selectors; the C4 verification already flagged this rule).
* **Lengths 2 and 3 depend on the gap.** Most selected sets of these lengths come from a pattern with an
  eventual edge (table 3.2), and the per-case counts of such a pattern — hence its information gain and coverage —
  change with the gap. Neighbouring values (2 vs 3, 3 vs 5) share 5–8 of 10 sets at length 2 and 4–8 at length 3.
  f2 is the least stable log (17–22 of 30 sets shared between any two gaps), f3 the most stable (21–26).
* The **activities** inside the sets are more stable than the sets: 6–9 activities appear under every gap at
  lengths 2 and 3, usually the same core activities (`376400`, `ac411100`, `ac419100`, `ac415100`, …) paired with
  different partners.
* The top ranks are not immune: in f1 length 3, rank 1 is a different set under gaps 1, 2 and 3 (section 9 of
  `c5c6_tables.md`).

## 4. C6 — composition of the fronts

### 4.1 Three objectives versus two

Front size per step ("candidates → front"):

| Log | Three objectives (IG, coverage, CD), default run | Two objectives (IG, coverage), free run | Two objectives on the candidates of the default run |
|---|---|---|---|
| f1 | 164 → 7, 1,204 → 26, 851 → 50 | 164 → 3, 774 → 2, 204 → 1 | 3, 2, 2 |
| f2 | 207 → 7, 1,634 → 44, 1,175 → 64 | 207 → 4, 1,190 → 6, 481 → 5 | 4, 6, 5 |
| f3 | 156 → 7, 1,031 → 23, 861 → 39 | 156 → 1, 274 → 1, 168 → 2 | 1, 1, 2 |

Per length (first Pareto front among the candidate sets; "layers" = layers needed to fill k = 10):

| Log | Length | Candidate sets, 3 obj | First front, 3 obj (layers) | Candidate sets, 2 obj free run | First front, 2 obj free run (layers) | First front, 2 obj on the default run's patterns (layers) | Sets shared with the default selection: free run / same patterns |
|---|---|---|---|---|---|---|---|
| f1 | 1 | 164 | 7 (2) | 164 | 3 (5) | 3 (5) | 5 / 5 |
| f1 | 2 | 460 | 14 (1) | 301 | 4 (4) | 4 (4) | 5 / 5 |
| f1 | 3 | 400 | 21 (1) | 166 | 1 (6) | 1 (6) | 2 / 3 |
| f2 | 1 | 207 | 7 (2) | 207 | 3 (4) | 3 (4) | 4 / 4 |
| f2 | 2 | 647 | 34 (1) | 492 | 8 (2) | 8 (2) | 7 / 7 |
| f2 | 3 | 619 | 45 (1) | 385 | 3 (4) | 3 (4) | 4 / 4 |
| f3 | 1 | 156 | 7 (2) | 156 | 1 (4) | 1 (4) | 5 / 5 |
| f3 | 2 | 378 | 14 (1) | 106 | 1 (6) | 2 (6) | 3 / 4 |
| f3 | 3 | 347 | 9 (2) | 116 | 1 (6) | 1 (6) | 2 / 3 |

* The earlier probe "length-1 front 3 / 4 / 1 with two objectives" (Guide C6) is reproduced exactly (step-0 front
  3 / 4 / 1). The earlier "19 / 32 / 29 with three objectives" is **not** what our settings give: the step-0 front
  is 7 / 7 / 7. Experiment C12 explains the difference: fronts of that size (17 / 33 / 29 there) appear when the
  case distance is computed with the original `pdist('jaccard')` under scipy 1.18.1, not with our explicit distance.
* In f2 the step-0 front has 4 activities but the first front of length-1 *sets* has 3, because a set is
  represented by its best pattern, which can be a self-loop from step 1 or 2 (C4, section 4.3 point 3).
* With two objectives the case distance no longer widens the front: 1–4 patterns per step are extended, the
  mining evaluates 1,084 / 1,781 / 573 patterns instead of 2,109 / 2,893 / 1,946, and the "top 10" is filled from
  up to 6 layers. If only step-front patterns could supply sets, lengths 3 of f1 and f3 would have **0** sets.
* The case distance is also what lets rare sets onto the front: selected sets that occur in fewer than 10 cases —
  three objectives 2 / 2 / 3 (f1 / f2 / f3, 7 of 90), two objectives 1 / 0 / 0 (1 of 90). With two objectives the
  length-1 selection moves towards frequent activities (shared with Apriori's top 10: 6 / 6 / 8 instead of
  3 / 3 / 4).

### 4.2 Number of distinct activities of the patterns per step (default run)

| Log | Step | Candidates | with 1 / 2 / 3 / more than 3 distinct activities | Front | Front with 1 / 2 / 3 / more than 3 |
|---|---|---|---|---|---|
| f1 | 0 | 164 | 164 / 0 / 0 / 0 | 7 | 7 / 0 / 0 / 0 |
| f1 | 1 | 1,204 | 9 / 915 / 280 / 0 | 26 | 2 / 17 / 7 / 0 |
| f1 | 2 | 851 | **3 / 231 / 440 / 177** | 50 | 0 / 14 / 26 / 10 |
| f2 | 0 | 207 | 207 / 0 / 0 / 0 | 7 | 7 / 0 / 0 / 0 |
| f2 | 1 | 1,634 | 9 / 1,243 / 382 / 0 | 44 | 3 / 33 / 8 / 0 |
| f2 | 2 | 1,175 | **3 / 280 / 639 / 253** | 64 | 0 / 12 / 36 / 16 |
| f3 | 0 | 156 | 156 / 0 / 0 / 0 | 7 | 7 / 0 / 0 / 0 |
| f3 | 1 | 1,031 | 9 / 786 / 236 / 0 | 23 | 4 / 15 / 4 / 0 |
| f3 | 2 | 861 | **3 / 217 / 510 / 131** | 39 | 0 / 12 / 19 / 8 |

* "More than 3" is always exactly 4 (a 3-node context pattern extended by one node). 177 / 253 / 131 step-2
  candidates (20.8 % / 21.5 % / 15.2 %) and 10 / 16 / 8 step-2 front patterns are dropped by the projection.
* "Step" is not "length": step 1 already yields length-3 sets (280 / 382 / 236 context patterns) and length-1 sets
  (9 self-loops); step 2 still yields 231 / 280 / 217 patterns with 2 activities. So the length must be taken from
  the number of distinct activities, as the working default says.
* The same table for all gaps and the other configurations is in `c5c6_tables.md` section 1 (e.g. gap 5, f2:
  1,689 step-2 candidates, 300 with 4 activities).

### 4.3 `distinct=True` versus `False`

`distinct=True` (paretoset's default, used by the 2023 code) keeps only the first of several rows with identical
objective values on a front and pushes the others to later layers.

How many candidates have an exact twin (same IG, coverage and CD as another candidate of the step), default run:

| Log | Step 0 | Step 1 | Step 2 | The same after rounding to 12 decimals |
|---|---|---|---|---|
| f1 | 2 of 164 | 444 of 1,204 | 333 of 851 | 2 / 455 / 341 |
| f2 | 2 of 207 | 600 of 1,634 | 523 of 1,175 | 2 / 601 / 528 |
| f3 | 2 of 156 | 393 of 1,031 | 363 of 861 | 2 / 412 / 377 |

About a third to almost a half of the step-1 and step-2 candidates share their three values with another
candidate (typically patterns that occur in the same few cases).

Effect on the fronts and on the selection:

| Log | Step fronts, `False` | Step fronts, `True`, same candidates | Step fronts, `True`, free run (step-2 candidates) | First front of the sets per length, `False` | The same, `True` (free run) | Selected sets equal to the default (length 1 / 2 / 3), free run | Same patterns, only the ranking with `True` |
|---|---|---|---|---|---|---|---|
| f1 | 7 / 26 / 50 | 7 / 24 / 35 | 7 / 24 / 35 (838 instead of 851) | 7 / 14 / 21 | 7 / 14 / 11 | 10 / 10 / 10 | 10 / 10 / 10 |
| f2 | 7 / 44 / 64 | 7 / 41 / 42 | 7 / 41 / 42 (1,162 instead of 1,175) | 7 / 34 / 45 | 7 / 33 / 24 | 10 / 10 / 10 | 10 / 10 / 10 |
| f3 | 7 / 23 / 39 | 7 / 23 / 31 | 7 / 23 / 31 (861, unchanged) | 7 / 14 / 9 | 7 / 14 / 8 | 10 / 10 / 9 | 10 / 10 / 9 |

* `distinct=True` removes 0–3 patterns from the step-1 front and 8–22 from the step-2 front, and roughly halves
  the first front of the length-3 sets in f1 and f2. The top 10 is almost unaffected, because the removed twins
  are mostly low-ranked rare patterns.
* The one changed set: in f3, length 3, `{376400, ac370000, ac411100}` (rank 8, first front, the set occurs in
  42 cases) is replaced by `{ac379999, ac411100, ac419100}`. It is pushed to a later layer only because another
  set has exactly the same three values, and which of the two twins survives depends on the row order. This is
  the argument against `distinct=True`.
* With two objectives `distinct` changes nothing at all (same fronts, same 30 sets in every log).

Rounding the interest values to 12 decimals (the C4 verification found that mathematically tied patterns can
differ by about 1e-18, so they are not seen as ties):

| Log | Step fronts, rounded, `False` | Sets equal to the default (length 1 / 2 / 3) | What changes |
|---|---|---|---|
| f1 | 7 / 27 / 51 | 10 / 10 / 9 | length 3: `{ac410500, ac411100, ac415100}` (rank 10) is replaced by `{376400, ac370000, ac378452}` |
| f2 | 7 / 44 / 65 | 10 / 10 / 10 | nothing |
| f3 | 7 / 23 / 39 (step 2 has 869 instead of 861 candidates) | 10 / 9 / 10 | length 2: `{ac10307, ac415100}` is replaced by `{ac355427, ac411100}` |

So 2 of the 90 default sets depend on floating-point noise, both at the bottom of a top 10 (the same two cells
the C4 verification found with rounding at selection time only). Rounding plus `distinct=True` gives 29 / 30 / 28
of 30 sets equal to the default.

### 4.4 How many sets exist per length — is k = 10 reachable?

Default run ("evaluated" = sets may come from every evaluated pattern; "step fronts only" = only from patterns on
a step's Pareto front):

| Log | Length | Candidate sets, evaluated | Candidate sets, step fronts only | First front | Layers for k = 10 |
|---|---|---|---|---|---|
| f1 | 1 | 164 | **7** | 7 | 2 |
| f1 | 2 | 460 | 17 | 14 | 1 |
| f1 | 3 | 400 | 21 | 21 | 1 |
| f2 | 1 | 207 | **7** | 7 | 2 |
| f2 | 2 | 647 | 27 | 34 | 1 |
| f2 | 3 | 619 | 28 | 45 | 1 |
| f3 | 1 | 156 | **7** | 7 | 2 |
| f3 | 2 | 378 | 13 | 14 | 1 |
| f3 | 3 | 347 | **9** | 9 | 2 |

Over all 9 configurations × 3 logs: the smallest number of candidate sets from evaluated patterns is 156
(length 1, f3), 106 (length 2, f3, two objectives) and 116 (length 3, f3, two objectives), so **k = 10 is always
reachable** with `candidate_pool='evaluated'` and all 81 cells select exactly 10 sets. With step-front patterns
only it is not: 7 sets at length 1 in every three-objective run, 7–11 at length 3 in f3 depending on the gap, and
0–7 sets per length with two objectives. In the three-objective runs the first front alone gives 10 or more sets at
lengths 2 and 3 (except f3 length 3: 8–11 depending on the run) but only 6–7 at length 1.

## 5. IMPresseD sets versus the Apriori top 10 per size

Apriori sets from `results/experiments/C7/apriori_sets_<log>.json` (top 10 per size by support; `min_support`
0.5 / 0.5 / 0.45). Number of shared sets (of 10) per length:

| Configuration | f1 (length 1 / 2 / 3) | f2 | f3 |
|---|---|---|---|
| gap 1 | 3 / 0 / 0 | 3 / 1 / 0 | 4 / 0 / 0 |
| gap 2 | 3 / 0 / 0 | 3 / 1 / 0 | 4 / 0 / 0 |
| gap 3 (default) | 3 / 0 / 0 | 3 / 1 / 0 | 4 / 0 / 0 |
| gap 5 | 3 / 1 / 0 | 2 / 1 / 0 | 4 / 1 / 0 |
| two objectives | 6 / 0 / 0 | 6 / 0 / 0 | 8 / 0 / 0 |
| `distinct=True` | 3 / 0 / 0 | 3 / 1 / 0 | 4 / 0 / 0 |
| rounded | 3 / 0 / 0 | 3 / 1 / 0 | 4 / 0 / 0 |

The default values equal the ones experiment C4 reported. Whatever the gap, the two strategies share 2–4
single activities, at most 1 pair and no triple, so the comparison in the paper has to be made between two
*groups* of sets, not on shared sets. The full side-by-side lists are in the appendix.

## 6. What this means for the project decisions

* **B12, oracle (`delta_time`)** — use chain traces. The numbers of section 2 can be quoted in the paper
  (88.8 / 87.1 / 88.4 % equal-timestamp pairs on the filtered logs; 73.7 / 68.4 / 73.5 % of the events in blocks
  ≥ 10; 412 / 110 / 401 traces without any order). The two "not re-run" probes of Guide row C5 are now re-run
  and confirmed (78.2 % and 30.8 % on f1).
* **B12, gap** — the gap is **not** a harmless technical parameter: it leaves length 1 untouched but changes
  2–5 of the 10 length-2 sets and 2–7 of the 10 length-3 sets relative to gap 3. Consequences:
  (1) ask the supervisor which value the 2023 experiments used (the paper gives none) — this question is worth
  keeping in B12; (2) state the chosen value (3) in the paper as a parameter of the method; (3) because the
  selector needs only seconds, a robustness check with one other gap (2 or 5) is cheap on the selection side —
  the cost is the permutation importance for the sets that differ (4–10 new sets per log for one extra gap); at
  minimum report the overlap table 3.3 as a limitation.
* **B1 (objectives)** — keep the three objectives. With IG and coverage only, the fronts have 1–8 sets, the
  mining collapses to a handful of extended patterns, and k = 10 is filled from up to 6 layers, which is no longer
  a "Pareto front" selection. The price of the case distance is the rare, noise-driven sets (7 of 90 selected
  sets occur in fewer than 10 cases); that is the open min-support question already raised by C4, not solved here.
* **B7 / B8 (front versus k, ranking)** — k = 10 per length is always reachable, but only when sets may come from
  all evaluated patterns and when layers below the first front are used (always at length 1). The paper should say
  so explicitly. `distinct` must stay `False`; `True` changes at most 1 of 30 sets but does so arbitrarily.
* **Robustness of ranks 9–10** — 2 of 90 default sets flip when the interest values are rounded to 12 decimals.
  Rounding inside `score_patterns` (the fix proposed by the C4 verification) costs nothing and makes the selection
  independent of the numpy / scikit-learn build; if it is adopted, the default sets change in f1 length 3 and f3
  length 2 as listed in section 4.3.
* **Length-1 representation** — represent a length-1 set by the plain activity (the fix proposed by the C4
  verification). Section 3.4 shows what the current rule can do: at gap 5 the activity `ac419100` leaves the f2
  top 10 only because its self-loop pattern is used as its representative.
* **Length definition** — confirmed on all runs: steps and lengths do not coincide (step 1 yields sets of length
  1, 2 and 3; 15–22 % of the step-2 candidates have 4 activities and are dropped).

## 7. What failed, what was not done, caveats

* Nothing crashed. A first version of the C5a script compared the parallel flags of the `delta_time = −1` graphs
  with the block rule for `delta_time = 0` and therefore reported meaningless mismatches; it was corrected and the
  script was re-run (the table in 2.3 is from the corrected run).
* The original pattern extension with `delta_time = 0` was **not** run (only the trace graphs and the node sets
  of the step-1 rules). So there is no measured runtime or pattern count for the original code on concurrent
  traces; the statement that it would be much slower is an expectation, not a measurement.
* The chain-only selector is verified against the original code for gap 3 only (C4). For gaps 1, 2 and 5 the
  counts were checked against the independent plain scan (0 of 2,700 patterns differ) but not against the original
  2023 code.
* `max_gap = 0` (no eventual edges), more than 2 extension steps, other k values and train-only mining were not
  part of this experiment.
* The "free run" and "same candidates" comparisons answer different questions; where they differ (two objectives,
  f3 length 2 and the length-3 overlaps) both numbers are given.
* All selections use the current set-ranking rule of `impressed_chain.py` ("one representative pattern per set,
  then non-dominated sorting of the sets"). The C4 verification pointed out that this differs from the Guide
  wording; the alternative rule was not tested here, and the overlap numbers between gaps could differ under it.
* The rounding variant wraps `impressed_chain.score_patterns` at run time; if the module is later changed to
  round by itself, the "rounded" configurations of this script become redundant.
* Timings were taken while other experiments were running (the three runs of the sensitivity script took 279.5 s,
  292.3 s and 260.6 s).
* pytest, pycodestyle and pyflakes are not installed in the venv; the two scripts were checked with a small
  script for unused imports, absolute paths (none) and line length (none over 120 characters).

## Appendix — selected IMPresseD sets (default run) next to the Apriori top 10 per size

`*` marks a set that is in both lists. "Cases with the set" = number of cases that contain all activities of the
set in any order. Generated by `c5_c6_sensitivity.py` (same content as `sets_vs_apriori.md`).

### f1, length 1: 3 shared (Apriori min_support 0.5)

| rank | IMPresseD set | layer | source pattern | cases with the set | Apriori set | support |
|---|---|---|---|---|---|---|
| 1 | {376400} | 1 | 376400 | 334 | {ac370000}* | 0.701 |
| 2 | {ac419100}* | 1 | ac419100 | 729 | {ac419100}* | 0.645 |
| 3 | {ac370000}* | 1 | ac370000 | 792 | {370407}* | 0.591 |
| 4 | {ac415100} | 1 | ac415100 | 258 | {ac370419} | 0.577 |
| 5 | {ac10307} | 1 | ac10307 | 141 | {ac370443} | 0.564 |
| 6 | {ac355427} | 1 | ac355427 | 36 | {ac370442} | 0.559 |
| 7 | {337419c} | 1 | 337419c | 3 | {370712b} | 0.542 |
| 8 | {ac411100} | 2 | ac411100 | 519 | {370715a} | 0.539 |
| 9 | {370407}* | 2 | 370407 | 668 | {ac370403} | 0.538 |
| 10 | {ac378449} | 2 | ac378449 | 24 | {370488e} | 0.535 |

### f1, length 2: 0 shared (Apriori min_support 0.5)

| rank | IMPresseD set | layer | source pattern | cases with the set | Apriori set | support |
|---|---|---|---|---|---|---|
| 1 | {370715a, 376400} | 1 | 370715a ~> 376400 | 299 | {370407, ac370000} | 0.591 |
| 2 | {370712b, 376400} | 1 | 370712b ~> 376400 | 299 | {ac370000, ac370419} | 0.577 |
| 3 | {ac411100, ac419100} | 1 | ac411100 -> ac419100 | 481 | {ac370000, ac370443} | 0.564 |
| 4 | {370401c, ac370000} | 1 | ac370000 ~> 370401c | 591 | {ac370419, ac370443} | 0.562 |
| 5 | {ac370000, ac370403} | 1 | ac370000 ~> ac370403 | 608 | {370407, ac370419} | 0.559 |
| 6 | {ac411100, ac415100} | 1 | ac411100 ~> ac415100 | 216 | {ac370000, ac370442} | 0.559 |
| 7 | {ac415100, ac419100} | 1 | ac419100 -> ac415100 | 241 | {ac370419, ac370442} | 0.557 |
| 8 | {ac10307, ac419100} | 1 | ac10307 ~> ac419100 | 119 | {ac370442, ac370443} | 0.557 |
| 9 | {ac10307, ac411100} | 1 | ac10307 -> ac411100 ~> ac411100 | 132 | {370407, ac370443} | 0.549 |
| 10 | {ac355201, ac415100} | 1 | ac355201 ~> ac415100 | 33 | {370407, ac370442} | 0.544 |

### f1, length 3: 0 shared (Apriori min_support 0.5)

| rank | IMPresseD set | layer | source pattern | cases with the set | Apriori set | support |
|---|---|---|---|---|---|---|
| 1 | {376400, 377498a, ac372417} | 1 | ac372417 -> 376400 -> 377498a | 288 | {ac370000, ac370419, ac370443} | 0.562 |
| 2 | {ac411100, ac415100, ac419100} | 1 | ac411100 -> ac419100 -> ac415100 | 201 | {370407, ac370000, ac370419} | 0.559 |
| 3 | {ac10307, ac411100, ac419100} | 1 | ac10307 -> ac411100 -> ac419100 -> ac411100 | 119 | {ac370000, ac370419, ac370442} | 0.557 |
| 4 | {ac356134, ac411100, ac419100} | 1 | ac411100 -> ac419100 ~> ac356134 | 63 | {ac370000, ac370442, ac370443} | 0.557 |
| 5 | {ac355201, ac415100, ac419100} | 1 | ac355201 ~> ac419100 -> ac415100 | 33 | {ac370419, ac370442, ac370443} | 0.555 |
| 6 | {ac355201, ac411100, ac419100} | 1 | ac355201 -> ac411100 -> ac419100 | 39 | {370407, ac370000, ac370443} | 0.549 |
| 7 | {ac355427, ac411100, ac419100} | 1 | ac355427 -> ac411100 -> ac419100 | 25 | {370407, ac370419, ac370443} | 0.549 |
| 8 | {376400, ac370000, ac411100} | 1 | ac411100 -> ac370000 -> 376400 | 21 | {370407, ac370000, ac370442} | 0.544 |
| 9 | {376400, ac370419, ac378452} | 1 | ac370419 ~> 376400 -> ac378452 | 5 | {370407, ac370419, ac370442} | 0.543 |
| 10 | {ac410500, ac411100, ac415100} | 1 | ac410500 -> ac415100 -> ac411100 | 15 | {370407, ac370442, ac370443} | 0.542 |

### f2, length 1: 3 shared (Apriori min_support 0.5)

| rank | IMPresseD set | layer | source pattern | cases with the set | Apriori set | support |
|---|---|---|---|---|---|---|
| 1 | {376400} | 1 | 376400 | 339 | {ac419100}* | 0.937 |
| 2 | {378619a} | 1 | 378619a | 413 | {ac370000}* | 0.702 |
| 3 | {ac370000}* | 1 | ac370000 | 793 | {ac379999}* | 0.690 |
| 4 | {ac419100}* | 1 | ac419100 ~> ac419100 | 1059 | {370407} | 0.602 |
| 5 | {ac10307} | 1 | ac10307 | 145 | {ac370419} | 0.586 |
| 6 | {ac415100} | 1 | ac415100 | 344 | {ac410100} | 0.582 |
| 7 | {ac337441} | 1 | ac337441 | 3 | {ac370443} | 0.577 |
| 8 | {ac379999}* | 2 | ac379999 | 780 | {ac370442} | 0.573 |
| 9 | {ac378431} | 2 | ac378431 | 14 | {370712b} | 0.551 |
| 10 | {378490e} | 2 | 378490e | 4 | {ac411100} | 0.550 |

### f2, length 2: 1 shared (Apriori min_support 0.5)

| rank | IMPresseD set | layer | source pattern | cases with the set | Apriori set | support |
|---|---|---|---|---|---|---|
| 1 | {376400, 377498a} | 1 | 376400 -> 377498a | 294 | {ac370000, ac379999} | 0.690 |
| 2 | {376400, 378619a} | 1 | 376400 ~> 378619a | 331 | {ac370000, ac419100} | 0.647 |
| 3 | {370712b, 376400} | 1 | 370712b ~> 376400 | 309 | {ac379999, ac419100}* | 0.641 |
| 4 | {370715a, 376400} | 1 | 370715a ~> 376400 | 307 | {370407, ac370000} | 0.602 |
| 5 | {378619a, ac379999} | 1 | 378619a -> ac379999 | 412 | {370407, ac379999} | 0.593 |
| 6 | {378619a, ac419100} | 1 | 378619a ~> ac419100 | 385 | {ac370000, ac370419} | 0.586 |
| 7 | {370401c, ac370000} | 1 | ac370000 ~> 370401c | 604 | {ac370000, ac370443} | 0.577 |
| 8 | {ac370000, ac370403} | 1 | ac370000 ~> ac370403 | 619 | {ac370419, ac379999} | 0.576 |
| 9 | {376400, ac379999} | 1 | 376400 ~> ac379999 | 334 | {ac370000, ac370442} | 0.573 |
| 10 | {ac379999, ac419100}* | 1 | ac379999 ~> ac419100 | 724 | {ac370419, ac370443} | 0.573 |

### f2, length 3: 0 shared (Apriori min_support 0.5)

| rank | IMPresseD set | layer | source pattern | cases with the set | Apriori set | support |
|---|---|---|---|---|---|---|
| 1 | {376400, 377498a, 378619a} | 1 | 376400 -> 377498a ~> 378619a | 288 | {ac370000, ac379999, ac419100} | 0.641 |
| 2 | {376400, 377498a, ac372417} | 1 | ac372417 -> 376400 -> 377498a | 292 | {370407, ac370000, ac379999} | 0.593 |
| 3 | {376400, 377498a, ac379999} | 1 | 376400 -> 377498a ~> ac379999 | 290 | {ac370000, ac370419, ac379999} | 0.576 |
| 4 | {376400, 378619a, ac379999} | 1 | 376400 ~> 378619a -> ac379999 | 330 | {ac370000, ac370419, ac370443} | 0.573 |
| 5 | {376400, 377498a, ac370606} | 1 | ac370606 ~> 376400 -> 377498a | 259 | {ac370000, ac370442, ac370443} | 0.571 |
| 6 | {378619a, ac379999, ac419100} | 1 | 378619a -> ac379999 ~> ac419100 | 385 | {370407, ac370000, ac370419} | 0.570 |
| 7 | {378619a, 387042a, ac379999} | 1 | 378619a -> ac379999 ~> 387042a | 148 | {ac370000, ac370419, ac370442} | 0.569 |
| 8 | {376400, 378619a, ac411100} | 1 | 376400 -> 378619a ~> ac411100 | 88 | {ac370000, ac370443, ac379999} | 0.567 |
| 9 | {387042a, ac410100, ac419100} | 1 | ac410100 -> ac419100 -> 387042a | 154 | {ac370419, ac370442, ac370443} | 0.567 |
| 10 | {378619a, ac370000, ac379999} | 1 | 378619a -> ac379999 ~> ac370000 | 412 | {370407, ac370000, ac370443} | 0.565 |

### f3, length 1: 4 shared (Apriori min_support 0.45)

| rank | IMPresseD set | layer | source pattern | cases with the set | Apriori set | support |
|---|---|---|---|---|---|---|
| 1 | {ac419100}* | 1 | ac419100 | 756 | {ac419100}* | 0.680 |
| 2 | {ac370000}* | 1 | ac370000 | 705 | {ac370000}* | 0.635 |
| 3 | {ac415100} | 1 | ac415100 | 232 | {370407} | 0.527 |
| 4 | {ac10307} | 1 | ac10307 | 145 | {ac370419} | 0.513 |
| 5 | {ac355427} | 1 | ac355427 | 34 | {ac370443} | 0.501 |
| 6 | {ac388170} | 1 | ac388170 | 4 | {ac370442} | 0.496 |
| 7 | {387070a} | 1 | 387070a | 2 | {370712b} | 0.489 |
| 8 | {ac370402} | 2 | ac370402 | 510 | {370715a} | 0.484 |
| 9 | {370488g}* | 2 | 370488g | 532 | {370488e}* | 0.480 |
| 10 | {370488e}* | 2 | 370488e | 533 | {370488g}* | 0.479 |

### f3, length 2: 0 shared (Apriori min_support 0.45)

| rank | IMPresseD set | layer | source pattern | cases with the set | Apriori set | support |
|---|---|---|---|---|---|---|
| 1 | {ac370000, ac370402} | 1 | ac370000 ~> ac370402 | 510 | {370407, ac370000} | 0.527 |
| 2 | {370401c, ac370000} | 1 | ac370000 ~> 370401c | 523 | {ac370000, ac370419} | 0.513 |
| 3 | {ac370000, ac370403} | 1 | ac370000 ~> ac370403 | 531 | {370407, ac370419} | 0.502 |
| 4 | {ac411100, ac419100} | 1 | ac411100 -> ac419100 | 478 | {ac370000, ac370443} | 0.501 |
| 5 | {ac411100, ac415100} | 1 | ac411100 ~> ac415100 | 201 | {ac370419, ac370443} | 0.500 |
| 6 | {ac415100, ac419100} | 1 | ac415100 ~> ac419100 | 221 | {ac370000, ac370442} | 0.496 |
| 7 | {ac10307, ac411100} | 1 | ac10307 -> ac411100 ~> ac411100 | 136 | {ac370442, ac370443} | 0.494 |
| 8 | {ac10307, ac419100} | 1 | ac10307 ~> ac419100 | 129 | {ac370419, ac370442} | 0.493 |
| 9 | {ac10307, ac415100} | 1 | ac10307 ~> ac415100 | 59 | {370407, ac370443} | 0.491 |
| 10 | {ac355201, ac415100} | 1 | ac355201 ~> ac415100 | 29 | {370712b, ac370000} | 0.489 |

### f3, length 3: 0 shared (Apriori min_support 0.45)

| rank | IMPresseD set | layer | source pattern | cases with the set | Apriori set | support |
|---|---|---|---|---|---|---|
| 1 | {ac411100, ac415100, ac419100} | 1 | ac411100 -> ac419100 -> ac415100 | 192 | {370407, ac370000, ac370419} | 0.502 |
| 2 | {ac10307, ac411100, ac419100} | 1 | ac10307 ~> ac419100 -> ac411100 | 129 | {ac370000, ac370419, ac370443} | 0.500 |
| 3 | {ac370000, ac411100, ac415100} | 1 | ac411100 -> ac415100 -> ac370000 | 120 | {ac370000, ac370442, ac370443} | 0.494 |
| 4 | {ac355201, ac415100, ac419100} | 1 | ac355201 ~> ac419100 -> ac415100 | 29 | {ac370000, ac370419, ac370442} | 0.493 |
| 5 | {ac355201, ac411100, ac419100} | 1 | ac355201 -> ac411100 -> ac419100 | 30 | {ac370419, ac370442, ac370443} | 0.492 |
| 6 | {ac355427, ac415100, ac419100} | 1 | ac355427 ~> ac419100 -> ac415100 | 20 | {370407, ac370000, ac370443} | 0.491 |
| 7 | {370407, ac10307, ac370000} | 1 | ac10307 -> ac370000 -> 370407 | 42 | {370407, ac370419, ac370443} | 0.491 |
| 8 | {376400, ac370000, ac411100} | 1 | ac411100 ~> ac411100 -> ac370000 -> 376400 | 42 | {370407, ac370000, ac370442} | 0.485 |
| 9 | {ac410500, ac411100, ac415100} | 1 | ac410500 -> ac415100 -> ac411100 | 8 | {370407, ac370419, ac370442} | 0.484 |
| 10 | {ac379999, ac415100, ac419100} | 2 | ac379999 ~> ac419100 -> ac415100 | 129 | {370407, ac370442, ac370443} | 0.483 |
