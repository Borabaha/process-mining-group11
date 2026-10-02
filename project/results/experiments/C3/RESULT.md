# C3 — Does the ORIGINAL IMPresseD automatic mode run end-to-end on BPIC11 f1 with our pinned packages?

Run on 2 Oct 2026, Windows 10, Python 3.12 venv (numpy 2.5.3, pandas 2.3.3, scipy 1.18.1, scikit-learn 1.9.1,
networkx 3.7, pm4py 2.7.23.8, paretoset 1.2.5). Other experiments were running on the same machine and some
C3 runs overlapped each other, so all wall times are "busy machine" times (single-threaded code; expect them
to be up to roughly a third lower on an idle machine).

## 1. Answer

**Yes.** `Auto_IMPID.AutoStepWise_PPD` from `external/InteractivePatternDetection` runs end-to-end on
`BPIC11_f1_trunc36.csv` **unmodified, without any crash**, under the pinned packages. No patch is needed for
correctness. It only emits warnings (section 6).

It is slow, for three reasons that are bugs of the "works, but quadratic" kind:

| | original code, full f1 | same code + 3 result-preserving performance patches |
|---|---|---|
| step 0 + step 1 (`Max_extension_step = 1`) | **1632 s = 27.2 min** (measured) | 83 s (measured) |
| step 2 on top (`Max_extension_step = 2`) | **not run; extrapolated 3.7-5.1 h** (section 5) | 70 s (measured) |
| whole run with 2 extension steps | about 4-5.5 h (extrapolated) | **153 s = 2.6 min** (measured) |

The patched copy gives **bit-identical results** to the original wherever both were run (120 cases and 300 cases
with 2 extension steps; the full f1 log for steps 0 and 1) — section 4.

## 2. What was run

Driver: `experiments/run_original_impressed.py`. It rebuilds the GUI's input preparation headlessly (each function
names the GUI lines it mirrors) and calls `AutoStepWise_PPD` unchanged; the original functions are only wrapped at
run time for timing. `GUI_IMPresseD_tool.py` was read as text, never imported. Nothing in `external/` was edited or
written to.

Commands (Git Bash, from the project root `C:/Users/Bohabara/Desktop/process mining/project`):

```
# main run: original code, full f1, Max_extension_step = 1  (27.2 min)
.venv/Scripts/python.exe -u experiments/run_original_impressed.py --max-extension-step 1 --run-name f1_step1 \
    --time-limit-min 45 --patterns-json results/experiments/C3/original_patterns_f1.json \
    > results/experiments/C3/logs/f1_step1.log 2>&1

# scaling runs: original code, first 120 / 300 cases, Max_extension_step = 2  (3.1 / 6.3 min)
.venv/Scripts/python.exe -u experiments/run_original_impressed.py --max-extension-step 2 --max-cases 120 --run-name smoke120
.venv/Scripts/python.exe -u experiments/run_original_impressed.py --max-extension-step 2 --max-cases 300 --run-name scale300_step2 --time-limit-min 30

# patched copy (add: --code-dir experiments/original_impressed_patched)
... --max-extension-step 2 --max-cases 120 --run-name smoke120_patched        --code-dir experiments/original_impressed_patched
... --max-extension-step 2 --max-cases 300 --run-name scale300_step2_patched  --code-dir experiments/original_impressed_patched
... --max-extension-step 2 --run-name f1_step2_patched --time-limit-min 40    --code-dir experiments/original_impressed_patched
# patched copy on the 2024-filtered logs (same lower-casing and rare-activity filter as DataManager), f1 / f2 / f3
... --dataset external/PermutationLocationImportance/datasets/BPIC11_f1_trunc36.csv --log-prep dm2024 --max-extension-step 2 \
    --run-name f1_dm2024_step2_patched --code-dir experiments/original_impressed_patched --time-limit-min 15 \
    --patterns-json results/experiments/C3/patched_patterns_f1_dm2024.json        (same for f2_trunc40, f3_trunc31)

# equality check and tables
.venv/Scripts/python.exe experiments/compare_impressed_runs.py results/experiments/C3/smoke120 results/experiments/C3/smoke120_patched
.venv/Scripts/python.exe experiments/summarise_original_impressed.py results/experiments/C3/f1_step1/timings.json results/experiments/C3/original_patterns_f1.json
```

Settings (what a user would enter in the GUI):

| GUI field | value | source |
|---|---|---|
| case id / activity / timestamp / outcome | `case:concept:name` / `concept:name` / `time:timestamp` / `label` | task |
| outcome type | `binary` (information gain via `mutual_info_classif`) | task |
| delta time | `-1` s, so no two events are concurrent: every trace is a chain in file order (= `event_nr` order, checked) | task |
| Max gap between events | 3 | working default |
| Max extension step | 1 (main run), 2 (scaling and patched runs) | task |
| interest functions | `Outcome_Interest` Max, `Frequency_Interest` Max, `Case_Distance_Interest` Min (GUI defaults, GUI:273/287/301) | GUI |
| numerical attributes | `Age` | task |
| categorical attributes | `Diagnosis`, `Treatment code`, `Diagnosis code`, `Specialism code` | task |
| testing percentage | 0.2 — **the GUI has no default; this is our choice** | assumption |
| split | inside the tool: `train_test_split(test_size=0.2, random_state=42, stratify=label)` on the case table -> 912 train / 228 test cases (labels 546/366 and 136/92) | Auto_IMPID.py:16-18 |
| log | raw CSV as the GUI reads it: 24 176 events, 1140 cases, 176 activities (no 2024 rare-activity filter) | GUI:178 |

Differences from a real GUI session, all outside the original code: the distance pickle is written to the run
folder (the GUI writes `<csv folder>/dist/pairwise_case_distances.pkl`, i.e. into the dataset folder, and re-uses it
for every other CSV in that folder); `random.seed(2023)` before the node colours are drawn; pandas
`PerformanceWarning` silenced (the first run printed it 1163 times on 120 cases).

## 3. Results of the original code

### 3.1 Wall time per stage (full f1, original code, `Max_extension_step = 1`)

| stage | wall time | where the time goes (inclusive times of wrapped functions) |
|---|---|---|
| read CSV, format columns, colours | 0.1 s | |
| case table + activity counts (`VariantSelection` through pm4py, GUI:989-1000) | 12.6 s | |
| pairwise case distances (`calculate_pairwise_case_distance`, 649 230 pairs) | **0.5 s** | plus building the `pair_cases` list |
| **step 0**: score 176 activities on the 912 train cases, Pareto front | **96.7 s** | `similarity_measuring_patterns` 90.7 s (94 %); `paretoset` 5.8 s (numba compilation on first call); information gain 0.2 s |
| **step 1 extension**: 13 core activities, 1132 trace graphs, 3688 `Pattern_extension` calls | **1072.5 s** | `update_pattern_dict` **1029.7 s** (44 345 calls, 23 ms each: one `nx.is_isomorphic` call per stored pattern); trace graphs 4.9 s; embedded graphs 2.7 s; rest (pandas filtering, count columns) about 30 s |
| **step 1 scoring**: 2109 candidate patterns | **462.7 s** | `similarity_measuring_patterns` **458.3 s**; information gain 2.9 s; frequency 1.5 s |
| writing 51 pattern JSON files, building `train_X` / `test_X` | 0.1 s | |
| **`AutoStepWise_PPD` total** | **1632.0 s (27.2 min)** | isomorphism scan 63 %, case-distance scoring 34 % (549 s over both steps) |

So 97 % of the run is spent in two loops:

1. `tools.update_pattern_dict` (tools.py:85-87): every candidate graph is compared with `nx.is_isomorphic` against
   every pattern found so far, in one dictionary shared by all core activities (2109 patterns at the end).
2. `IMIPD.similarity_measuring_patterns` (IMIPD.py:40-58): for every pattern it copies the whole (ever wider) case
   table twice and then finds every in x out case pair with `pair_cases.index(...)`, a linear scan of a Python list
   (8.1 million pairs in step 0, 17.4 million in step 1, about 11 microseconds each).

Scaling of the original code (measured):

| log | cases | activities | step 0 | step 1 | step 2 | `AutoStepWise_PPD` total |
|---|---|---|---|---|---|---|
| first 120 cases | 120 | 127 | 6.1 s | 73.3 s | 107.2 s | 186.6 s |
| first 300 cases | 300 | 152 | 9.1 s | 89.7 s | 280.3 s | 379.1 s |
| full f1 | 1140 | 176 | 96.7 s | 1535.2 s | not run (section 5) | 1632.0 s for steps 0-1 |

In step 2 the time goes somewhere else: `Single_Pattern_Extender` takes 86 % of step 2 on 300 cases (240.7 s), and
only 7.2 s of that is the isomorphism scan. The rest is the count-column block at IMIPD.py:741-746, which is
indented inside the per-instance loop: after every single instance of the parent pattern all new count columns are
reset to 0 and refilled with one `DataFrame.loc` assignment per (pattern, case).

### 3.2 Number of patterns and Pareto-front size per step (full f1)

Steps 0 and 1 are from the original code. Step 2 is from the patched copy, which is identical to the original on
steps 0 and 1 of this very log (section 4).

| step | candidates | on the Pareto front | candidates by node count | front by node count | front by number of distinct activities | front by edge type | candidates with 0 train cases |
|---|---|---|---|---|---|---|---|
| 0 | 176 activities | **13** | 1: 176 | 1: 13 | 1: 13 | - | 4 |
| 1 | **2109** (393 direct pairs, 1080 eventually pairs, 636 three-node "context" patterns) | **51** | 2: 1473, 3: 636 | 2: 38, 3: 13 | 1: 2, 2: 39, 3: 10 (46 distinct sets) | 22 directly only, 29 eventually | 190 |
| 2 (patched copy) | **1298** (children of the 22 front patterns without an eventually edge) | **58** | 3: 851, 4: 447 | 3: 31, 4: 27 | 1: 1, 2: 7, 3: 28, 4: 22 (56 distinct sets) | 18 directly only, 40 mixed | 72 |

Step-0 front (the 13 core activities, by information gain): 376400, AC419100, AC370000, AC379999, AC410100,
AC386002, AC387090, AC10107, AC389190, 376480A, AC355111, 370737C, 330001B.

Step-1 front, top 10 by information gain (train cases out of 912):

| id | activities in trace order | edge | IG | coverage | case distance | train cases |
|---|---|---|---|---|---|---|
| 376400_7 | 370715A , 376400 | eventually | 0.2982 | 0.251 | 0.6702 | 229 |
| 376400_6 | 370712B , 376400 | eventually | 0.2912 | 0.247 | 0.6665 | 225 |
| 376400_1 | AC372417 , 376400 | directly | 0.2826 | 0.241 | 0.6637 | 220 |
| 376400_4 | 376400 , AC378607 | eventually | 0.2401 | 0.213 | 0.6487 | 194 |
| 376400_2 | 376400 , 377498A | directly | 0.1740 | 0.163 | 0.6216 | 149 |
| 376400_5 | AC370606 , 376400 | eventually | 0.1645 | 0.156 | 0.6138 | 142 |
| AC370000_2 | AC370000 , AC370000 | directly | 0.1492 | 0.641 | 0.8622 | 585 |
| AC419100_14 | AC411100 , AC419100 | directly | 0.1370 | 0.346 | 0.7900 | 316 |
| AC370000_3 | AC370000 , AC370403 | eventually | 0.1097 | 0.529 | 0.8353 | 482 |
| AC370000_42 | AC370000 , AC370401 | eventually | 0.1050 | 0.498 | 0.8212 | 454 |

Distinct projected activity sets that are on a front of any step (steps 0-2), by set size:

| log (code) | size 1 | size 2 | size 3 | size 4 (to be dropped) |
|---|---|---|---|---|
| f1 raw, 1140 cases (original for steps 0-1, patched for step 2) | 13 | 36 | 35 | 22 |
| f1 with the 2024 filter, 1130 cases / 164 activities (patched) | 13 | 33 | 35 | 19 |
| f2 with the 2024 filter, 1130 / 207 (patched) | 31 | 64 | 73 | 54 |
| f3 with the 2024 filter, 1111 / 156 (patched) | 14 | 36 | 20 | 23 |

### 3.3 Output files the tool writes (formats)

In the folder the user selects (here `results/experiments/C3/f1_step1/impressed_output/`), 53 files for the main run:

* `<patternID>.json`, one per front pattern of step 1 and later (51 files; the step-0 activities get no file).
  networkx node-link JSON, written with `json_graph.node_link_data` (Auto_IMPID.py:88-93), one line, e.g.
  `{"directed": true, "multigraph": false, "graph": {}, "nodes": [{"value": "AC370000", "parallel": false, "color": "#AB1F00", "id": 8}, ...], "edges": [{"eventually": false, "source": 8, "target": 9}, ...]}`.
  Node `id` is the position (0-based) of the event in the trace where the pattern was first seen, `value` is the
  activity label, `eventually` is the edge type. Pattern IDs are `<core activity>_<n>` (step 1) and `<parent ID>_<n>`
  (step 2), which is why the GUI replaces "_" in activity labels by "-" (BPIC11 labels contain no "_").
* `training_encoded_log.csv` (912 x 66) and `testing_encoded_log.csv` (228 x 66), written by the GUI code
  (GUI:1016-1017), reproduced by the driver: one integer count column per front pattern of every step (13 activities +
  51 step-1 patterns), then `Case_ID` and `Outcome`.
* `dist/pairwise_case_distances.pkl`: pickle of the condensed `pdist` vector (numpy float64, 649 230 values, 5.2 MB).
  The GUI puts it next to the CSV; the driver puts it in the run folder.

The tool does not save the interest values or the non-front candidates, and the pattern dictionary is a local
variable. The driver captures them and writes, per run folder: `pattern_attributes_step<k>.csv` (the tool's own
interest table plus an `on_front` column), `timings.json`, and the pattern export below.

### 3.4 Pattern export for the chain-only prototype

`results/experiments/C3/original_patterns_f1.json` (original code, raw f1, steps 0 and 1; 1.4 MB):

```
{"meta": {settings, package versions, n_cases, timings, ...},
 "steps": {"0": {"n_candidates": 176, "front_size": 13, "patterns": [...]},
           "1": {"n_candidates": 2109, "front_size": 51, "patterns": [
               {"id": "AC410100_1", "labels": ["AC410100", "AC419100"],            # activity labels in trace order
                "edges": [{"source": 0, "target": 1, "type": "directly"}],          # indices into "labels"; "directly" | "eventually"
                "n_nodes": 2, "activity_set": ["AC410100", "AC419100"], "step": 1, "on_front": true,
                "n_instances_whole_log": 762, "Pattern_Frequency": 316.0, "Case_Support": 284.0,   # the last two on the 912 train cases
                "Outcome_Interest": 0.0520, "Frequency_Interest": 0.3114, "Case_Distance_Interest": 0.6673}, ...]}},
 "unscored_patterns_from_unfinished_step": []}
```

All candidates are included (not only the front), so the prototype can be compared on the candidate universe, on
the interest values and on front membership separately. Same format, steps 0-2, from the patched copy:
`f1_step2_patched/patterns.json` (raw f1) and `patched_patterns_f{1,2,3}_dm2024.json` (2024-filtered logs,
lower-case labels — the log the 2024 pipeline and the recipe in Guide 4.3 use).

## 4. Crashes and patches

**Crashes: none**, in any of the runs (120 cases, 300 cases, full f1 with the original; full f1, f2, f3 with the
patched copy). No correctness patch was needed, so there is no traceback to report.

Because the original cannot do extension step 2 on the full log inside the time box, three **performance-only**
patches were applied in a copy, `experiments/original_impressed_patched/` (`IMIPD.py`, `tools.py`, unchanged
`Auto_IMPID.py`; full diff in `patches.diff`, about 25 changed lines):

| patch | where | change |
|---|---|---|
| P1 | `IMIPD.similarity_measuring_patterns` | take the row index without copying the whole frame; compute the position of pair (a, b) in the condensed distance vector directly as `start_search_points[a] + b - a - 1` instead of `pair_cases.index(...)` |
| P2 | `IMIPD.Single_Pattern_Extender` | de-indent the count-column block (:741-746) so it runs once per parent pattern instead of once per instance |
| P3 | `tools.update_pattern_dict` | store a signature (sorted activity labels, number of edges) with every pattern and call `nx.is_isomorphic` only for stored patterns with the same signature |

Proof that the patches do not change results (`experiments/compare_impressed_runs.py`; pattern IDs, graphs, all
interest values with exact float equality, front flags, encoded CSVs, pattern JSON files byte for byte):

| comparison | result |
|---|---|
| 120 cases, steps 0-2: original vs patched | IDENTICAL (127 / 805 / 324 candidates, fronts 12 / 32 / 22, both CSVs, 54 JSON files) |
| 300 cases, steps 0-2: original vs patched | IDENTICAL (152 / 781 / 445 candidates, fronts 10 / 25 / 28, both CSVs, 53 JSON files) |
| full f1, steps 0-1: original vs patched | IDENTICAL (176 / 2109 candidates, fronts 13 / 51); the 66 columns of the original CSVs and the 51 JSON files are identical inside the patched step-2 run |
| full f1, step 2 | patched only; the original was not run (section 5) |

Measured effect on full f1: `update_pattern_dict` in step 1 1029.7 s -> 16.6 s; `similarity_measuring_patterns` in
step 1 458.3 s -> 12.4 s; steps 0-1 together 1632 s -> 83 s.

Patched copy, 2 extension steps, wall time of `AutoStepWise_PPD`:

| log | cases / activities | step 0 | step 1 | step 2 | total | candidates per step | front per step |
|---|---|---|---|---|---|---|---|
| f1 raw (GUI) | 1140 / 176 | 12.1 s | 70.9 s | 69.9 s | **153.1 s** | 176 / 2109 / 1298 | 13 / 51 / 58 |
| f1, 2024 filter | 1130 / 164 | 7.9 s | 63.2 s | 72.2 s | **143.4 s** | 164 / 2041 / 1178 | 13 / 48 / 57 |
| f2, 2024 filter | 1130 / 207 | 9.6 s | 413.6 s | 324.0 s | **747.4 s** | 207 / 4996 / 2603 | 31 / 93 / 125 |
| f3, 2024 filter | 1111 / 156 | 8.9 s | 25.9 s | 22.4 s | **57.3 s** | 156 / 1161 / 874 | 14 / 42 / 52 |

f2 and f3 were not run with the original code.

## 5. What could not be finished, and the extrapolation

**Not finished: the original code with `Max_extension_step = 2` on the full f1 log.** Steps 0-1 alone take 27 min
and a step-2 run repeats them, so it does not fit the 60-minute box. How far the original got: step 2 completed on
120 cases (107 s) and on 300 cases (280 s); on the full log step 1 completed, step 2 was never started.

Extrapolation of the original step 2 on full f1:

* A scratch copy of the patched code (not in the project) counted what the original per-instance block would
  execute: 7415 parent instances, **15 550 859 `.loc` assignments** and 7415 resets of on average 121 columns.
  The same count is 131 777 for 120 cases and 278 909 for 300 cases.
* Cost per operation, micro-benchmark on a frame shaped like the case table in step 2 (1140 rows x 2415 columns):
  0.83 ms per `.loc` assignment, 6.0 ms per reset -> 12 900 s = **3.6 h**.
* Cross-check on 300 cases: the same method predicts 170 s for the block; measured (original minus patched step-2
  extender time) 230 s, i.e. the micro-benchmark is 1.36 times too optimistic for a run on the busy machine ->
  upper estimate **4.9 h**.
* Plus scoring 1298 step-2 candidates with the original `similarity_measuring_patterns` (17.3 million pairs, as in
  step 1: about 7-8 min) and the unpatched isomorphism scan (about 1 min).

So the original with two extension steps on f1 needs roughly **27 min + 3.7 to 5.1 h, about 4 to 5.5 hours**. This is
an extrapolation, not a measurement. For f2 (4996 step-1 candidates instead of 2109) it would be clearly longer; no
number was derived for f2 or f3.

Compute used: about 37 min of original-code runs plus about 22 min of patched and counting runs, partly in
parallel (wall clock 11:28-12:07).

## 6. Things the run showed about the original code (relevant when comparing against it)

Warnings only: `SettingWithCopyWarning` at Auto_IMPID.py:40-43 and :97-100 (harmless, the frames are only returned);
`RuntimeWarning: Mean of empty slice` 194 times in the main run (see point 3); pandas `PerformanceWarning`
(fragmented frame) at every column insertion.

1. **`Max_extension_step = 1` already yields patterns with 2 and 3 nodes** (pairs, plus the pred -> core -> succ
   "context" pattern); `= 2` yields 3 and 4 nodes. Step 2 extends only front patterns without an eventually edge:
   22 of the 51 on f1.
2. **The case distance of the GUI is mostly a control-flow distance.** The GUI passes the whole case table minus id
   and outcome, so 176 activity-count columns are treated as categorical next to the 4 chosen attributes (181
   feature columns, 1 numerical). On f1 the tool's distance vector correlates 0.97 with the plain Jaccard distance
   of activity presence and only 0.14 with an attribute-only distance (Age + explicit mismatch share of the 4
   attributes). A prototype with the project default (attributes only) will therefore produce different
   `Case_Distance_Interest` values and different fronts than this file.
3. **Step 1 extends on the whole log but scores on the train cases**: 190 of 2109 step-1 candidates (72 of 1298 in
   step 2) occur only in test cases, get coverage 0 and a NaN case distance. None of them is on a front here.
4. **Counts are inflated in two ways.** (a) A self-pattern such as `AC370000 -> AC370000` is found once as
   "following" and once as "preceding" pattern of the same core: tool count 3072 vs 1536 real adjacent occurrences
   (number of cases is right: 726). (b) If both activities of a pair are core activities, the second core appends
   the same occurrences to the stored instance list again: 4 of the 51 front patterns have exactly twice as many
   stored instances as occurrences (e.g. `AC410100_1`: 762 vs 381); their count columns are right, but step-2
   children are derived from the doubled list. Information gain is computed on these counts
   (`mutual_info_classif`, discrete), coverage and case distance only on presence.
5. No duplicated graphs among the step-2 candidates of different parents in this run (1298 candidates, 1298 distinct
   label/edge signatures).
6. The distance cache of the GUI is keyed by folder, not by file: all BPIC11 CSVs are in one folder, so a GUI
   session on f2 after f1 would silently load f1's distances (not triggered here because the driver uses its own
   folder).

## 7. What this means for the project

* **Table C row C3 is closed: the original runs under the pinned versions.** The environment is not the obstacle;
  the run time is (27 min for one extension step on f1, hours for two).
* **Decision B10 (wrap the original vs reimplement chain-only).** The original, unpatched, is not usable for the
  variant, which needs `Max_extension_step = 2` on three logs. With the three small patches it is usable as a
  cross-check on the full logs (f1 2.6 min, f2 12.5 min, f3 1 min), not only on a small log as the guide assumed.
  The chain-only reimplementation stays the recommended main path, because the project defaults deliberately differ
  from the tool (attribute-only case distance, explicit Jaccard, mining on the whole log, clean counts) and the tool's
  quirks in section 6 would otherwise have to be explained in the paper.
* **How to compare the prototype with `original_patterns_f1.json`.** Compare in this order, each level needs more
  emulation of the tool: (1) step-1 candidate universe as (labels, edge types) when the prototype is given the same
  13 core activities — independent of scoring; (2) `Case_Support` on the tool's 912 train cases (split:
  `train_test_split(test_size=0.2, random_state=42, stratify=label)` on the case table sorted by case id as a
  string); (3) information gain, which needs the tool's count conventions (section 6 point 4); (4) case distance,
  which needs the GUI-style distance including activity counts and scipy 1.18.1 `pdist('jaccard')`; (5) front
  membership. Expect differences from level 3 on unless the prototype has "emulate the tool" switches.
* **k = 10 per length is always available, and the first front is already larger than 10.** On f1/f2/f3 the
  fronts contain 13-31 single activities, 33-64 distinct pairs and 20-73 distinct triples (table in 3.2; tool
  settings, 80 % train split). With k = 10, "fill by non-dominated layers" therefore never leaves the first layer;
  the selected sets are decided entirely by the tie-break (information gain). This should be stated when the rule
  is described (Guide Part 6, B7). The counts will change under the project's own case distance and whole-log
  scoring, so re-check them with the prototype.
* **Length mapping confirmed on real data**: step-1 front patterns project to sets of size 1 (2 patterns, A -> A),
  2 (39) and 3 (10); step-2 front patterns to size 1-4 (1 / 7 / 28 / 22). Size-4 sets must be dropped and repeats
  collapse, as the recipe in Guide 4.3 says.

## 8. Files

| path | content |
|---|---|
| `experiments/run_original_impressed.py` | headless driver (GUI preparation, timing wrappers, pattern export, time limit) |
| `experiments/summarise_original_impressed.py` | Markdown tables from one run |
| `experiments/compare_impressed_runs.py` | equality check of two runs |
| `experiments/original_impressed_patched/` | copy of `IMIPD.py`, `tools.py`, `Auto_IMPID.py` with patches P1-P3; `patches.diff` |
| `results/experiments/C3/original_patterns_f1.json` | original code, raw f1, steps 0-1, all candidates with interest values and front flags |
| `results/experiments/C3/f1_step1/` | main run: `impressed_output/` (51 pattern JSONs, 2 CSVs), `dist/`, `pattern_attributes_step{0,1}.csv`, `timings.json`, `patterns.json` (copy of the export) |
| `results/experiments/C3/f1_step2_patched/` | patched copy, raw f1, steps 0-2 (`patterns.json` in the same format) |
| `results/experiments/C3/patched_patterns_f{1,2,3}_dm2024.json`, `f{1,2,3}_dm2024_step2_patched/` | patched copy on the 2024-filtered logs, steps 0-2 |
| `results/experiments/C3/smoke120*`, `scale300_step2*` | scaling and equality runs (original and patched) |
| `results/experiments/C3/logs/` | console logs of all runs |
