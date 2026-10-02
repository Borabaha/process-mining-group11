# Group 11 — what the experiments of 2 October 2026 tell us

This file sums up twelve experiments (C1–C5/C6, C7–C12, C18) and one pilot run.
They were run on 2 Oct 2026 on one Windows 10 PC (Ryzen 5 5600X, 12 logical cores, 24 GB RAM) with the project's Python 3.12 venv.
Each experiment has its own full report: `results/experiments/<id>/RESULT.md`.
Every number below was checked against those reports. Where two reports disagree, section 6 says which number to use.

**Words used in this file**

| Word | Meaning |
|---|---|
| iteration | one (activity set, repeat): the set is moved once in every scored trace that contains it, and the model scores the result once |
| faithful mode | our engine behaves exactly like the original 2024 code, bugs included |
| fixed mode | our engine does what the 2024 paper describes: every repeat starts from the unchanged traces, the order inside a set is kept, positions are allowed ones |
| training fold / held-out fold | the 4/5 of the cases the model was fitted on / the 1/5 it never saw |
| fold seed | the random seed of the 5-fold split; one fold seed = one way to cut the log into 5 folds |
| length of a set | number of distinct activities in the set (1, 2 or 3) |
| Spearman, Kendall | rank agreement between two rankings; 1 = same order, 0 = unrelated, −1 = reversed |

---

## 1. One-page summary: what we know now that we did not know yesterday

1. **Runtime is no longer a problem.** The original code needs 5–13 s per iteration. The full project grid would take 5.3 / 9.2 / 4.4 hours on f1 / f2 / f3 (extrapolated). Our engine needs 0.004–0.08 s per iteration and ran the same grid in 30–159 s per log and setting (measured). *Why it matters: we do not have to cut folds, repeats or sets, and we can re-run everything after each design change.*
2. **The engine gives the original's numbers exactly.** Faithful mode equals the original code with a largest difference of 0.0 on every value compared (more than 350 set values on f1, f2 and f3, and 952 single-activity values on f1 and f3). *Why it matters: we can reproduce the paper's pipeline without running the slow code, and we can trust the engine as the base of the variant.*
3. **The published importance values come mostly from a side effect of the code.** The original code never resets the traces, so permutations pile up. Mean importance, original behaviour / fixed on training fold / fixed on held-out fold: f1 0.063 / 0.033 / 0.002, f2 0.150 / 0.081 / 0.023, f3 0.250 / 0.028 / 0.009. The original ranking is unrelated to the fixed one (Spearman 0.27 / 0.12 / −0.31). Processing the same sets in reverse order nearly inverts the f2 ranking (−0.83). *Why it matters: the Apriori-versus-IMPresseD comparison must use fixed mode. In faithful mode a set's value depends on which sets were processed before it.*
4. **On the held-out fold one split is not enough, and f1 is close to noise.** On f1 two fold seeds agree on the ranking with Spearman 0.60 (10 repeats). More repeats and more folds do not help. More fold seeds do: two studies with 10 seeds each agree with 0.92. On the training fold one seed is enough (0.96). *Why it matters: the final run needs 10 fold seeds × 5 folds × 10 repeats, and we should report both folds.*
5. **Apriori is settled.** Take the 10 most frequent sets per size 1, 2, 3 with `max_len=3`. Use `min_support` 0.5 for f1 and f2 and 0.45 for f3 (a single 0.45 gives the same lists). f3 at 0.5 has only 5 / 4 / 1 sets of size 1 / 2 / 3. Apriori's pairs and triples are built from only 5–7 activities per log. *Why it matters: the baseline is now deterministic and the same size as the IMPresseD side.*
6. **The original IMPresseD code runs, but far too slowly.** One extension step on f1 took 27.2 minutes; two steps would take about 4–5.5 hours (extrapolated). Our chain-only re-implementation finds exactly the same patterns and the same interest values (difference 0.0) in 2.4–7.1 s per log. *Why it matters: we use our own implementation and keep the original only as a cross-check.*
7. **Chain traces are the right input, and the gap is a real parameter.** 88.8 / 87.1 / 88.4 % of consecutive events have the same timestamp, so with `delta_time = 0` most of a trace becomes one unordered block. `max_gap` does not change the length-1 sets, but compared with gap 3 another gap changes 2–5 of the 10 length-2 sets and 2–7 of the 10 length-3 sets. *Why it matters: we must state gap = 3 as a method parameter and ask the supervisor which value the 2023 experiments used.*
8. **k = 10 per length needs two rules that were not in the plan.** The first Pareto front has only 7 sets at length 1 in every log, so lower layers must be used. And the sets must be taken from all evaluated patterns, not only from patterns on a step front. Three objectives are needed: with information gain and coverage only, the fronts hold 1–8 sets. *Why it matters: these rules decide which sets we call "IMPresseD sets", so they must be written into the paper.*
9. **The original case distance depends on the scipy version.** For the same pair of cases, `pdist('jaccard')` gives 0.5 under scipy 1.11.4 and 0.0 under our scipy 1.18.1. We use our own explicit distance (0.333 for that pair). With the original call under scipy 1.18.1 the length-1 front would have 17 / 33 / 29 patterns instead of 7 / 7 / 7. *Why it matters: calling the original function in our environment would silently give a different interest function.*
10. **The existence baseline of the paper measures accuracy on the training fold, not F1.** This is certain now (checked in scikit-learn 0.22 to 1.9.1). On f1 accuracy and weighted F1 give the same ranking (Spearman 1.000). On f2 and f3 the existence-only model is no better than always predicting the majority class (accuracy 0.765 vs 0.784 and 0.767 vs 0.767), so its ranking is noise. *Why it matters: we can only interpret existence importance on f1.*
11. **Pilot result: yes, the picture changes, mainly because the two strategies pick different activities.** The lists share 3 / 3 / 4 of 10 sets at length 1, 0 / 1 / 0 at length 2 and none at length 3. On f2 the IMPresseD sets are clearly more important on the held-out fold (0.054 vs 0.028, in 10 of 10 fold seeds), because IMPresseD finds activity `376400`, which Apriori cannot select (it occurs in 30 % of the cases). On f1 and f3 the IMPresseD sets are not more important.
12. **Two things can mislead the comparison.** Apriori sets change 50–60 % of the scored traces, IMPresseD sets 6–38 %, so raw importance favours frequent sets. And 9 of the 90 IMPresseD sets are so rare that fewer than 3 traces per held-out fold change. *Why it matters: we need a rule for rare sets and should show importance per changed trace next to raw importance.*

---

## 2. The experiments, one by one

### C1 — How long does everything take?

**Question.** How many seconds does the original code need per iteration? Does the time grow linearly? How long are f2, f3 and the single-activity analysis?

**What we ran.** The unchanged original code on one training fold: 5 sets × 2 repeats for each strategy and set size, on all three logs. One complete pass of the original single-activity routine per log. Then the engine on the whole project grid (2 strategies × 3 lengths × 10 sets × 5 folds × 10 repeats = 3000 iterations per log) and on all single activities, in three settings.

**Result.**

| | f1 | f2 | f3 |
|---|---|---|---|
| Original: seconds per iteration (measured) | 5.9–7.3 | 10.3–12.7 | 4.8–5.8 |
| Original: paper defaults, 5 folds × 10 sets × 10 repeats (extrapolated) | 56 min | 92 min | 23 min (only 5 sets at support 0.5) |
| Original: full project grid (extrapolated) | 5.3 h | 9.2 h | 4.4 h |
| Original: all single activities × 5 folds × 10 repeats (extrapolated) | 12.8 h | 25.1 h | 9.1 h |
| Engine: full project grid, fixed mode, held-out / training fold (measured) | 30 / 80 s | 40 / 108 s | 24 / 59 s |
| Engine: full project grid, faithful mode (measured) | 120 s | 159 s | 87 s |
| Engine: all single activities, fixed held-out / fixed train / faithful (measured) | 37 / 72 / 78 s | 49 / 97 / 102 s | 29 / 53 / 56 s |

- The original's time per iteration does not grow from one iteration to the next. So folds × sets × repeats can be multiplied.
- On the same fold, model and sets the engine is 102–963 times faster. Its faithful mode returned the original's values exactly in all 240 compared iterations.
- `XGBClassifier(n_jobs=1)` gives identical values (all 106,050 rows) and is not slower. With one thread the three logs ran side by side and finished everything in 8.7 minutes.

**What we decide.**
- Use the engine for everything. Keep the original code only as the reference in the tests.
- Fit XGBoost with `n_jobs=1` and start one process per log.
- `demo` config: 1 fold seed × 5 folds × 2 repeats, fixed mode, both folds: 77 s for the three logs (measured).
- `full` config: 10 fold seeds × 5 folds × 10 repeats: about 25 minutes in parallel for fixed mode on both folds; about 1.4 hours with faithful mode and the single-activity analysis added.
- No overnight runs are needed.

**What could not be settled.** The original was never run at full size; its totals are extrapolations (read them as ± 5–10 %, the machine was shared). The "full" times are 10 times a measured one-seed run. Memory use and other computers (the team's Mac) were not measured.

### C2 — A fast engine that we can trust

**Question.** Can the permutation loop be made much faster and still give the same numbers as the original? Can we also build a corrected version that really keeps the paper's two rules (order kept, allowed positions)?

**What we ran.** `experiments/engine.py` with two modes (faithful, fixed). `test_engine.py` holds 23 checks. A second, independent reviewer wrote their own script and attacked the engine in two rounds. Round 1 found 11 issues; all were repaired (revision 2). Round 2 passed with no high-severity issue.

**Result.**

| Check | Result |
|---|---|
| Faithful mode vs original, activity sets | largest difference 0.0 on every value: 38 (builder), 57 (reviewer round 2; f1, f2, f3, also on the held-out fold and with unusual sets), 240 (C1) |
| Faithful mode vs original, single activities | 0.0 on 640 values (builder) and 312 values (reviewer) |
| Encoding vs original | identical matrices: 1130 × 5939 (f1), 1130 × 8319 (f2), 1111 × 4866 (f3) |
| Fixed mode: order kept, allowed position, same activities, same length | 100 % in 5000 random permutations (builder); 0 violations in 132,315 permuted traces (reviewer) |
| Fixed mode: no piling up, same seed → same result | confirmed by builder and reviewer |
| Speed on f1, quiet machine, per iteration | original 6.78 s; faithful 0.053 s (128×); fixed on training fold 0.036 s (189×); fixed on held-out fold 0.012 s (578×) |

Extra facts that matter for the method section:
- **How positions are drawn matters.** The paper does not say how the new positions of a pair or triple are drawn. The code draws them one after the other ("sequential"). Drawing every valid placement with the same chance ("uniform") lowers the training-fold importance by 12 % / 14 % / 9 % (f1 / f2 / f3) and changes the f2 ranking (Spearman 0.52 with the sequential ranking).
- **Where the allowed positions come from does not matter.** Whole log or training fold only: differences between −0.0001 and +0.0001.
- **Other events get shifted.** Moving a set pushes other events one or more places. 3.5 % / 4.3 % / 3.7 % of these shifted events land on a position where their activity was never seen. The paper admits this limit.

**What we decide.**
- Reproduction part: `mode='faithful', score_on='train'`.
- Comparison part: fixed mode, same settings for both strategies.
- Keep `draw='sequential'` (it is what the original code does) until the supervisor answers, and say so in the paper.
- Use the engine's own encoder (`IndexEncoder`, default order) and fit the model on its matrix.

**Open issues the reviewer found that are still in `engine.py`** (small, but fix them before the final run):
1. `mode='faithful'` still defaults to `score_on='test'`. The original scores the training fold. Until fixed, always pass `score_on='train'` with faithful mode.
2. A set that occurs in no scored trace gets importance 0. It should be "not measurable" (NaN). The pilot and C9 already treat it that way in their analysis.
3. The random streams of fixed mode do not include the fold seed. Pass `random_state = 2023 + fold seed` (C8, C9 and the pilot do).
4. If a trace holds several occurrences of a set, the sequential placement can give up although a valid placement exists (25 % on a toy trace; 0 of 21,950 calls on the real logs).
5. There is no "at least N traces" guard for rare sets.

**What could not be settled.** Sequential or uniform draw (supervisor). Which fold to report (supervisor, see C9). The single-activity proof covers the whole training fold of f1 and a 200-case subset of f3, not f2.

### C3 — Does the original IMPresseD code run in our environment?

**Question.** Does the original automatic mode (`Auto_IMPID.AutoStepWise_PPD`) run end to end on BPIC11 with our pinned packages, and how long does it take?

**What we ran.** A headless driver that prepares the inputs as the GUI does and calls the original function unchanged, on f1. Then a copy with three performance-only patches (about 25 changed lines), on f1, f2 and f3.

**Result.**

| Run | Time | Patterns per step (0 / 1 / 2) | On the Pareto front |
|---|---|---|---|
| Original, full raw f1, one extension step | 27.2 min (measured) | 176 / 2109 / – | 13 / 51 / – |
| Original, full f1, two extension steps | about 4–5.5 h (extrapolated, not run) | – | – |
| Patched copy, raw f1, two steps | 2.6 min | 176 / 2109 / 1298 | 13 / 51 / 58 |
| Patched copy, filtered f1 / f2 / f3, two steps | 2.4 / 12.5 / 1.0 min | 164 / 2041 / 1178; 207 / 4996 / 2603; 156 / 1161 / 874 | 13 / 48 / 57; 31 / 93 / 125; 14 / 42 / 52 |

- No crash, only warnings. The environment is not the obstacle; the run time is.
- 63 % of the original's time is the graph-isomorphism scan, 34 % the case-distance scoring.
- The patched copy gives identical results wherever both ran: 120 and 300 cases with two steps, and the full f1 log for steps 0–1.
- The original tool has habits that matter when we compare with it: its GUI feeds 176 activity-count columns into the case distance (the tool's distance correlates 0.97 with a plain activity-presence distance and only 0.14 with an attribute-only distance); it extends on the whole log but scores on an 80 % split; it counts some instances twice.

**What we decide.** The unpatched original is not usable for the variant. Keep our chain-only implementation as the main path. Use the patched copy as a cross-check on full logs.

**What could not be settled.** The original with two steps was not run on a full log (the 4–5.5 h is a calculation). f2 and f3 were only run with the patched copy. The front sizes above use the tool's own settings, not the project defaults (see section 6, note 2).

### C4 — Our own chain-only IMPresseD

**Question.** Does our simple re-implementation for traces without concurrency find the same patterns and the same interest values as the original code? Is comparing patterns as tuples as good as the original graph-isomorphism test?

**What we ran.** `experiments/impressed_chain.py`, 34 unit tests on hand-computed toy traces, and a cross-check against the unchanged 2023 functions on f1. An independent reviewer then compared it with the stored C3 runs on the full f1, f2 and f3 logs, steps 0–2.

**Result.**

| Compared with the original code | Patterns original / ours | Original time | Our time |
|---|---|---|---|
| 5 seed activities, one shared dictionary, 150 cases | 461 / 461 | 33.7 s | 0.009 s |
| 16 step-2 parents, 150 cases | 598 / 598 | 217.2 s | 0.021 s |
| Whole automatic mode, 2 steps, 120 training cases | 132 / 132, 566 / 566, 330 / 330 | 188.9 s | 1.13 s |
| Step 1 on full f1, 7 front activities | 1,204 / 1,204 | 344.5 s | 0.047 s |

- Nothing was found by only one side. Information gain, coverage and case distance match with a largest difference of exactly 0.0.
- Tuple equality agreed with the isomorphism test on all 21,807 tested pairs.
- The original counts some instances 2× or 4× (165 and 1 of 3,123 patterns). This does not change any interest value. We count each instance once.
- Our selector on the full logs: 7.1 / 3.5 / 2.4 s to mine and score (f1 / f2 / f3).

| Log | Candidate sets, length 1 / 2 / 3 | Sets on the first Pareto front | Also in Apriori's top 10 |
|---|---|---|---|
| f1 | 164 / 460 / 400 | 7 / 14 / 21 | 3 / 0 / 0 |
| f2 | 207 / 647 / 619 | 7 / 34 / 45 | 3 / 1 / 0 |
| f3 | 156 / 378 / 347 | 7 / 14 / 9 | 4 / 0 / 0 |

**What we decide.**
- Use `ImpressedChainSelector` (gap 3, 2 extension steps, k = 10 per length) as the IMPresseD variant.
- Fill k = 10 by Pareto layers and report the first-front sizes next to it.
- Take candidate sets from all evaluated patterns. With step-front patterns only, length 1 would have 7 sets and f3 length 3 only 9.
- Two extension steps are needed: 6–8 of the 10 length-3 sets per log come from step 2.

**Open issues the reviewer found that are still in `impressed_chain.py`** (the reviewer's verdict was "pass"; these are things to settle before the final run):
1. `fit()` does not check that the distance matrix matches the cases. A matrix of the wrong size or order is accepted silently. This is dangerous when fitting on a training fold.
2. Ranks 9–10 can depend on floating-point noise (differences near 1e-18) in 2 of 9 log/length cells. Fix: round the three interest values to 12 decimals.
3. Rare sets get selected because a low case distance alone puts a pattern on the front. Some selected sets occur in only 2–5 cases.
4. The set-ranking rule is not the Guide's default. The code first picks one representative pattern per set, then sorts the sets. Compared with "sort the patterns, then remove duplicate sets" the top 10 share 9 / 10 / 9 (f1), 8 / 10 / 10 (f2) and 5 / 8 / 7 (f3) sets at length 1 / 2 / 3.

**What could not be settled.** Minimum support for IMPresseD sets. Which set-ranking rule is meant. Whether a length-1 set must be represented by the plain activity (in f2 `{ac419100}` is represented by the self-loop `ac419100 ~> ac419100`). All three go to the supervisor.

### C5 / C6 — How sensitive is the IMPresseD side to its settings?

**Question.** What happens with the original time-based conversion (`delta_time = 0`)? Does `max_gap` change the selected sets? What do the Pareto fronts look like with three or two objectives, and with `distinct=True`? Is k = 10 always reachable?

**What we ran.** Block statistics and the original trace graphs for `delta_time = 0` on all three logs. Then nine configurations of the verified selector per log (gap 1, 2, 3, 5; two objectives; `distinct=True`; rounding).

**Result.**

| | f1 | f2 | f3 |
|---|---|---|---|
| Consecutive events with equal timestamp | 88.8 % | 87.1 % | 88.4 % |
| Events in same-day blocks of 10 or more | 73.7 % | 68.4 % | 73.5 % |
| Traces that become one single block (no order left) | 412 | 110 | 401 |
| Sets shared with gap 3, length 1 / 2 / 3 (of 10): gap 1 | 10 / 8 / 7 | 10 / 5 / 3 | 10 / 7 / 7 |
| … gap 2 | 10 / 8 / 6 | 10 / 6 / 4 | 10 / 8 / 8 |
| … gap 5 | 10 / 7 / 5 | 9 / 5 / 6 | 10 / 7 / 7 |
| Step fronts (steps 0 / 1 / 2), three objectives | 7 / 26 / 50 | 7 / 44 / 64 | 7 / 23 / 39 |
| Step fronts, two objectives (IG + coverage) | 3 / 2 / 1 | 4 / 6 / 5 | 1 / 1 / 2 |
| Step-2 candidates with 4 distinct activities (dropped) | 177 of 851 (21 %) | 253 of 1,175 (22 %) | 131 of 861 (15 %) |
| Selected sets unchanged with `distinct=True` | 30 of 30 | 30 of 30 | 29 of 30 |
| Selected sets unchanged after rounding to 12 decimals | 29 of 30 | 30 of 30 | 29 of 30 |

- With `delta_time = 0` the original rules would build "concurrent" patterns with a median of 25 / 28 / 22 nodes. Such patterns cannot become sets of 1–3 activities.
- Only 3–5 of the 10 length-2 and length-3 sets per log are selected under all four gaps. The activities inside the sets are more stable than the sets.
- k = 10 was reached in all 27 runs when sets may come from all evaluated patterns (at least 106 candidate sets per length).
- "Step" is not "length": step 1 already gives sets of length 1, 2 and 3.

**What we decide.**
- Chain traces (`delta_time < 0`, order by `event_nr`).
- Keep gap 3 as the default, state it as a method parameter, and show the overlap table as a limitation or run one other gap as a robustness check.
- Keep three objectives. Keep `distinct=False`.
- Adopt rounding to 12 decimals and represent a length-1 set by the plain activity (both are code changes still to make).

**What could not be settled.** Which gap the 2023 experiments used (the paper gives none). The original pattern extension with `delta_time = 0` was not run, so "much slower" is an expectation. The selector was checked against the original code only for gap 3.

### C7 — Apriori: which support, which sets?

**Question.** How many frequent sets of size 1, 2, 3 exist at each `min_support`? Which value gives at least 10 per size?

**What we ran.** A sweep of `min_support` ∈ {0.5, 0.49, 0.45, 0.4, 0.35, 0.3} with `max_len=3` on the three logs, a copy of the original selection rule, and a brute-force recount without mlxtend.

**Result (number of sets of size 1 / 2 / 3).**

| min_support | f1 | f2 | f3 |
|---|---|---|---|
| 0.5 | 16 / 94 / 258 | 23 / 148 / 544 | **5 / 4 / 1** |
| 0.49 | 20 / 121 / 414 | 25 / 188 / 804 | **6 / 9 / 7** |
| 0.45 | 23 / 168 / 841 | 26 / 248 / 1390 | 17 / 84 / 217 |
| 0.4 | 23 / 211 / 1330 | 26 / 286 / 2036 | 20 / 130 / 572 |

- Once 10 sets per size exist, `min_support` no longer matters: the lists are simply "top 10 by support". This was checked down to 0.3.
- Our copy of the original rule matches the original function in 24 of 24 comparisons.
- The original "top 10 of size > 1" is partly decided by sort ties: on f1 four sets tie for the last 2 places, on f2 two sets tie for the last place. Which ones win changes with `min_support`.
- On f3 the original returns only 5 sets at 0.5. At 0.49 it returns 10, with exactly three sets containing `ac370442`, as in the paper's Fig. 5.
- Without a size cap Apriori explodes just below 0.5: 262,215 sets on f1 at 0.45 (32 s).
- Apriori's pairs and triples use only 5 / 7 / 6 and 5 / 7 / 5 distinct activities (f1 / f2 / f3). On f1 they are every pair and every triple of the same five activities.

**What we decide.**
- `AprioriSelector(min_support, max_len=3, top_k=10)` with 0.5 / 0.5 / 0.45 (or 0.45 for all). Ties are broken by name. The lists are saved in `results/experiments/C7/apriori_sets_<log>.json`.
- Reproduction runs: `original_top10(traces, 0.5, 10)` for f1 and f2, `original_top10(traces, 0.49, 10)` for f3.
- Report the tie problem of the original top 10 as a limitation.
- Use "number of distinct activities covered" as a cheap extra measure when comparing the strategies.

**What could not be settled.** Which support the authors used for f3 (the data only show it was at most 0.494). Whether "length 1" should use Apriori's top-10 single activities or all activities, as the original single-activity mode does.

### C8 — How much do the original code's behaviours change the result?

**Question.** The original code (1) lets permutations pile up, (2) has a bookkeeping bug in the shuffle, (3) scores the training fold. How much does each change the result?

**What we ran.** The ten original Apriori sets per log, 10 fold seeds × 5 folds × 10 repeats (5000 values per setting and log), in six settings: A = faithful on the training fold; B = fixed on the training fold; C = fixed on the held-out fold; D = like A with the sets in reverse order; E = original shuffle without piling up; F = fixed shuffle with piling up.

**Result.**

| | f1 | f2 | f3 |
|---|---|---|---|
| Mean importance A / B / C (ten fold seeds) | 0.0632 / 0.0332 / 0.0022 | 0.1499 / 0.0809 / 0.0231 | 0.2496 / 0.0284 / 0.0086 |
| Ranking A vs B, Spearman (fold seed 0) | 0.27 | 0.12 | −0.31 |
| Ranking B vs C, Spearman (fold seed 0) | 0.82 | 0.94 | 0.88 |
| Ranking A vs D, only the order of the sets changed (fold seed 0) | 0.16 | −0.83 | 0.18 |
| Ranking E vs B, only the shuffle differs, no piling up (means over ten fold seeds) | 0.95 | 1.00 | 0.99 |
| Ranking A vs E, only the piling up differs (means over ten fold seeds) | 0.62 | 0.07 | −0.14 |
| Fixed mode (B): sets with `ac370000` vs without (ten fold seeds) | 0.037 vs 0.024 | 0.090 vs 0.060 | 0.034 vs 0.014 |

- One example: on f3 the set {ac370442, ac370443} has importance 0.256 when it is processed last and 0.011 when it is processed first. Same folds, same models, same seed.
- Piling up alone explains the difference. The bookkeeping bug changes the level by only 2–10 % on these sets.
- Held-out scoring changes the scale and the noise, not the story (B and C agree).
- Faithful mode matches the paper's Fig. 5 only roughly: f3 agrees, f1 is lower at the top (our highest median 0.074, the paper's about 0.085), f2 is about 0.04 lower. This is a by-eye reading of a figure.

**What we decide.**
- The comparison of the two strategies uses fixed mode.
- Faithful mode is only for showing that we can reproduce the published magnitudes. Show the reversed-order result next to it.
- In the paper-versus-code table, piling up is the difference that changes results; training-fold scoring changes the level by a factor of 3 to 15; the bookkeeping bug is minor.

**What could not be settled.** An exact match with Fig. 5 cannot be tested: the paper's folds are unseeded and at least one set per panel differs from ours. Whether "location matters more than existence" still holds in fixed mode was not tested.

### C9 — How much do the rankings move between fold seeds?

**Question.** Is one 5-fold split enough? How many repeats, folds and fold seeds does the final run need?

**What we ran.** f1 with the ten original sets: 20 fold seeds × 5 folds × 30 repeats in fixed mode (both folds) and 10 repeats in faithful mode. A 10-fold study. f2 and f3 with 10 seeds. The project's own 60 sets per log (f1 20 seeds, f2 and f3 10 seeds).

**Result (f1, ten original sets, Spearman between two rankings).**

| Design | Held-out fold | Training fold |
|---|---|---|
| Two single fold seeds, 3 / 5 / 10 / 30 repeats | 0.41 / 0.50 / 0.60 / 0.66 | 0.94 / 0.95 / 0.96 / 0.97 |
| Two studies that each pool 1 / 2 / 3 / 5 / 10 seeds (10 repeats) | 0.59 / 0.75 / 0.80 / 0.86 / 0.92 | 0.96 / 0.97 / 0.97 / 0.98 / 0.98 |
| 10 folds instead of 5 (10 repeats) | 0.62 (5 folds on the same seeds: 0.68) | 0.98 (0.96) |

| The paper's design (one split, 5 folds × 10 repeats) | f1 | f2 | f3 |
|---|---|---|---|
| Faithful mode, training fold | 0.53 | 0.96 | 0.83 |
| Fixed mode, training fold | 0.96 | 0.97 | 0.98 |
| Fixed mode, held-out fold | 0.60 | 0.93 | 0.77 |

- The models are fine in every seed (held-out F1 on f1: 0.903 on average, 0.893–0.911 per seed). The unstable ranking is not caused by bad models.
- On the f1 held-out fold the sets differ by about 0.002, and one seed's value has a ranking noise of 0.0017. One changed prediction moves the F1 of a 226-case fold by about 0.0044. Only 50.9 % of the single values are above zero.
- Even with 20 seeds, 5 of the 10 original f1 sets cannot be told from zero on the held-out fold.
- Project sets, held-out fold: f2 is stable with one seed (0.94–0.98 for the big groups). On f1, none of the 10 Apriori single activities is clearly above zero. Clearly above zero: 23 of 60 sets on f1, 55 of 60 on f2, 29 of 60 on f3.
- The sign of the comparison can depend on the fold: on f2, sizes 1 and 3, Apriori sets are higher on the training fold and IMPresseD sets on the held-out fold, both in 10 of 10 seeds.

**What we decide.**
- Final run: fold seeds 0–9 × 5 folds × 10 repeats, fixed mode, both folds scored from the same models, permutation seed = 2023 + fold seed. Five seeds is the minimum worth reporting.
- Report a set's importance as the mean over the 10 seed means, with its standard error.
- Give the stability of every ranking next to it (ranking from seeds 0–4 against ranking from seeds 5–9).
- On the held-out fold first say which sets are clearly above zero, then interpret order only among those.
- A demo with 1 seed and 2–3 repeats only shows that the pipeline runs.

**What could not be settled.** Which fold the report should lead with (supervisor). A held-out-only report would have little to say on f1. The fold seeds re-split the same cases, so intervals are approximate. Stability with the uniform draw was not measured.

### C10 / C18 — The existence-importance baseline

**Question.** C18: does "no `scoring` argument" in scikit-learn's `permutation_importance` really mean accuracy, also in old versions? C10: do the scorer (accuracy or weighted F1) and the encoding (count or 0/1) change the existence ranking?

**What we ran.** We read the scikit-learn source of every release line from 0.22 to 1.9.1. We re-implemented `Classical_Permutation.py` as class `ExistenceImportance` and ran every combination (2 encodings × 2 scorers × 2 folds) on f1, f2, f3, plus 5 fold seeds.

**Result.**

| | f1 | f2 | f3 |
|---|---|---|---|
| Accuracy vs weighted F1, Spearman of the rankings | 1.000 | no meaning | no meaning (sign flips for one set) |
| Count vs 0/1 encoding, Spearman (held-out / training fold) | 0.89 / 0.95 | no meaning | no meaning |
| Between two fold seeds, same combination (median) | 0.81–0.82 | 0.36–0.68 | 0.68–1.00 |
| Test accuracy, existence columns only (count) | 0.7336 | 0.7646 | 0.7669 |
| Test accuracy, always predict the majority class | 0.5982 | 0.7841 | 0.7669 |

- C18: `scoring=None` means `estimator.score`, which is accuracy for a classifier, in every version checked. In our venv the result is identical to `scoring='accuracy'`. So the original existence importance is a drop in training accuracy, although its plot says "Decrease in f1 score".
- The original script uses the unfiltered log (1140 cases on f1 instead of 1130), a substring count as feature, and unseeded folds.
- Substring instead of exact matching changes 1 of 11,300 feature values on f1 (name `370407` inside `370407c`).
- Nested sets can give identical columns (2 pairs in f1). The second twin then always gets exactly 0.

**What we decide.**
- Existence baseline = count encoding with exact tokens, `scoring='f1_weighted'`, scored on the held-out fold, mean over at least 5 fold seeds. Keep 0/1 encoding as a switch.
- Always print the existence model's score next to the majority-class score. On f2 and f3 the baseline has no skill and its ranking must not be interpreted.
- Detect identical columns and report them as one group.
- Use the same filtered case set as the location importance.

**What could not be settled.** Whether the baseline is required at all, and count or 0/1 (the paper's text says "binary encoding"; the code counts). The original script itself was not run (hard-coded paths), so our "faithful" configuration is a re-implementation, not a bit-identical copy. The scikit-learn history was read through a web tool that returns excerpts.

### C11 / C12 — Pareto library versions and the case distance

**Question.** C11: do paretoset 1.2.0 (the 2023 pin) and 1.2.5 (ours) give the same fronts? What do `distinct` and NaN do? C12: what exactly does the original case distance compute, and can we write it down explicitly?

**What we ran.** Both paretoset versions on 143 test cases. `experiments/case_distance.py` with 18 unit tests. The unchanged 2023 distance functions under scipy 1.11.4 and scipy 1.18.1, compared with ours on all case pairs of the three logs.

**Result.**

| Pair of cases (label codes of 3 attributes) | `pdist('jaccard')`, scipy 1.11.4 | `pdist('jaccard')`, scipy 1.18.1 | Ours (share of attributes that differ) |
|---|---|---|---|
| [0,1,2] vs [0,1,3] | 0.5 | 0.0 | 0.3333 |
| [0,1,2] vs [1,1,3] | 0.6667 | 0.3333 | 0.6667 |
| [0,1,2] vs [1,1,2] | 0.3333 | 0.3333 | 0.3333 |

| Length-1 patterns on the whole log | f1 | f2 | f3 |
|---|---|---|---|
| Pareto front size, our distance | 7 | 7 | 7 |
| Pareto front size, original code under scipy 1.11.4 | 7 | 12 | 7 |
| Pareto front size, original code under scipy 1.18.1 | 17 | 33 | 29 |
| Front members shared, ours vs 1.11.4 | 6 | 7 | 7 |
| Front members shared, ours vs 1.18.1 | 3 | 4 | 1 |

- paretoset: 141 of 143 results are identical. The 2 differences are features we never call.
- `distinct=True` keeps only the first of several rows with identical values. Which one survives depends on the row order.
- A NaN is treated as a tie. A NaN row can push valid patterns off the front.
- Both scipy behaviours depend on which category happens to get code 0 (for example `TC101`, carried by 40 % of the cases). Ours does not.
- The combination formula `(m × categorical + numeric) / (1 + m)` of the original is confirmed to 2.2e-16.
- BPIC11 attributes: no missing values, every attribute is constant inside a case.

**What we decide.**
- Keep `paretoset==1.2.5`. Always `distinct=False`. Never pass NaN: an undefined case distance becomes 1.0, the worst value.
- Use `case_distance.py`. Never call the original `calculate_pairwise_case_distance` under scipy 1.18.1.
- Compute the distance matrix once on all cases and slice it for folds.

**What could not be settled.** Which scipy version produced the published 2023 figures. What the paper's reference [8] (Cheung & Jia 2013) really defines; nobody has read it yet. The pattern-level comparison covers length 1 only.

### Pilot — the first end-to-end comparison

**Question.** Does the importance of activities and their locations change when the sets come from IMPresseD instead of Apriori?

**What we ran.** `run_pilot.py` with the working defaults: 60 sets per log (10 per length and strategy), fixed mode, 10 fold seeds × 5 folds × 10 repeats, one XGBoost model per fold shared by all sets and both strategies, held-out and training fold. Wall time 24 minutes for the three logs (956 / 1320 / 796 s per log).

**Result (held-out fold, mean importance, Apriori vs IMPresseD; in brackets: fold seeds in which IMPresseD is higher).**

| Length | f1 | f2 | f3 |
|---|---|---|---|
| 1 | −0.0001 vs 0.0008 (8 of 10) | 0.0108 vs 0.0200 (10 of 10) | 0.0153 vs 0.0150 (5 of 10) |
| 2 | 0.0009 vs 0.0013 (7 of 10) | 0.0249 vs 0.0742 (10 of 10) | 0.0059 vs 0.0056 (5 of 10) |
| 3 | 0.0051 vs 0.0003 (0 of 10) | 0.0479 vs 0.0674 (10 of 10) | 0.0084 vs −0.0002 (0 of 10) |
| Shared sets, length 1 / 2 / 3 | 3 / 0 / 0 | 3 / 1 / 0 | 4 / 0 / 0 |
| Traces changed by a permutation, Apriori / IMPresseD | 55 % / 20 % | 60 % / 31 % | 50 % / 20 % |
| Importance per changed share, training fold, Apriori / IMPresseD | 0.053 / 0.067 | 0.131 / 0.263 | 0.055 / 0.085 |

- **f2:** every set that contains `376400` loses 0.11–0.13 weighted F1 when its location is permuted. As a single activity it has 0.1196; the largest Apriori value on f2 is 0.0608. Apriori cannot select it at support 0.5.
- **f1 and f3:** IMPresseD's sets are not more important. An IMPresseD triple changes only 6–7 % of the traces.
- `376400` is also IMPresseD's first choice on f1, and its location importance there is zero (−0.0002). A high information gain says that the presence of an activity separates the classes. It does not say that its position matters.
- At activity level the two strategies rank the 5–7 common activities with Spearman 0.3–0.7. None of these values is significant.
- The pilot values equal the values C9 stored for the same sets (169,000 values, largest difference 5e-11).

**What we decide.**
- Compare two groups of sets, not shared sets. A shared set has the same importance under both strategies anyway, because the models are shared.
- Report raw importance together with the share of traces changed and importance per changed share.
- Read f1 from the training-fold companion.

**What could not be settled.** The IMPresseD sets were mined on the whole log, so the labels of held-out cases influenced the selection. Mining on training cases only was not run. That `376400` is the CEA test named in the f2 label rule is an inference from our earlier label report, not confirmed by a source. No faithful run and no existence comparison were analysed in the pilot. The selector issues of C4 are still in the lists used.

---

## 3. Decisions and open questions

### 3.1 Decisions we can now take without the supervisor

These are implementation choices, or choices where the measurements leave no real doubt.

1. **Use our engine, not the original script.** Faithful mode is bit-identical; the engine is 100–960 times faster (C1, C2).
2. **XGBoost with `n_jobs=1`, one process per log.** Identical values, the machine stays usable (C1).
3. **Final configuration: fold seeds 0–9 × 5 folds × 10 repeats, both folds scored from the same models, permutation seed 2023 + fold seed.** About 25 minutes for the three logs (C1, C9; the pilot measured 24 minutes).
4. **The comparison of the two strategies runs in fixed mode.** Faithful mode is kept for the reproduction check only (C8).
5. **Apriori side:** `AprioriSelector(min_support=0.45, max_len=3, top_k=10)`; reproduction sets from `original_top10` at 0.5 / 0.5 / 0.49 (C7).
6. **IMPresseD side:** our chain-only implementation on chain traces ordered by `event_nr`; the patched original as a cross-check (C3, C4, C5).
7. **Case distance:** our explicit function; distance matrix computed once on all cases (C12).
8. **Pareto:** paretoset 1.2.5, `distinct=False`, no NaN, undefined distance = 1.0 (C11).
9. **Sets come from all evaluated patterns and k = 10 is filled by layers.** We report the first-front sizes next to it (C4, C5/C6). The rule itself is also put to the supervisor (B7, B8).
10. **Existence baseline:** count encoding, exact tokens, weighted F1, held-out fold, at least 5 fold seeds, always with the majority-class score next to it (C10).
11. **A set that is absent from a scored fold is "not measurable", not 0** (C2 reviewer, C9, pilot).
12. **Code fixes before the final run** (none changes a result we reported, all remove traps):
    - `engine.py`: make `score_on` depend on the mode (faithful → train); return NaN for sets in no scored trace; add the fold seed to the random stream; retry a dead-end placement.
    - `impressed_chain.py`: check the distance-matrix shape in `fit()`; round IG, coverage and CD to 12 decimals; represent a length-1 set by the plain activity; add an optional `min_cases`; make the set-ranking rule a parameter.
    - After each fix re-run the tests in section 5.

### 3.2 Questions that remain for the supervisor

Office hour 2 is on Mon 26 Oct; questions are due Sun 25 Oct 15:00. If some of these were already answered in office hour 1 (2 Oct), the answer wins over our default.

1. **Fixed or faithful as the headline, and which fold? (B2, B3)** "Your code lets permutations accumulate. With the same folds and models, mean importance is 0.063 / 0.150 / 0.250 (f1 / f2 / f3) with accumulation and 0.033 / 0.081 / 0.028 without. The rankings are unrelated (Spearman 0.27 / 0.12 / −0.31), and reversing the order of the itemsets nearly inverts the f2 ranking (−0.83). On the held-out fold the values drop to 0.002 / 0.023 / 0.009 and on f1 only about half of them are above zero (52 %). We plan: fixed mode as the main result, both folds, 10 fold seeds; one faithful run as a reproduction check. Do you agree?"
2. **How should new positions be drawn? (new, belongs to B2)** "For pairs and triples your code draws the positions one after the other. Drawing every order-preserving placement with equal chance lowers the importance by 9–14 % and changes the f2 ranking (Spearman 0.52). Which one is meant in the paper?"
3. **Support for f3 (B6).** "At 0.5, f3 has 5 itemsets of size > 1. At any value up to 0.494 the top 10 are the same, with three `ac370442` sets as in your Fig. 5. Which value did you use?"
4. **What counts as an IMPresseD set? (B7, B8)** "With IG, coverage and case distance the first front has only 7 single activities per log, so we fill k = 10 from the next layer. We take sets from all evaluated patterns, not only from front patterns; otherwise length 1 has 7 sets and f3 length 3 has 9. Is that acceptable? Should we rank patterns and then drop duplicate sets, or rank one representative per set? The two rules share 5–10 of 10 sets."
5. **Minimum support for IMPresseD sets (B7, pilot).** "9 of the 90 selected IMPresseD sets change fewer than 3 traces per held-out fold. May we require a minimum number of cases, for example 10 as in your user study?"
6. **Gap and case distance (B12).** "Which `Max_gap_between_events` did the 2023 experiments use? Against gap 3, other gaps change 2–5 of 10 length-2 sets and 2–7 of 10 length-3 sets. Which scipy version produced the 2023 figures? `pdist('jaccard')` on label codes gives 0.5 in scipy 1.11.4 and 0.0 in 1.18.1 for the same pair; we use the share of differing attributes (0.333). Did your runs include the activity-count columns in the case distance, as the GUI does?"
7. **How to compare the strategies (B11).** "The two lists share 3–4 single activities, at most 1 pair and no triple. We plan to compare the two groups of sets (overlap, importance distributions per length, number of sets clearly above zero, activity-level means) and to show importance per changed trace, because Apriori sets change 50–60 % of the traces and IMPresseD sets 6–38 %. Is that what you expect?"
8. **Where to mine (B9).** Unchanged question. New fact: fitting on one random 80 % subset of f1 keeps 8 / 7 / 8 of the 10 whole-log sets per length (one subset only, reviewer's check).
9. **Existence baseline (B13).** "Your `Classical_Permutation.py` measures a drop in training accuracy with a count feature. On f2 and f3 a model on these features is no better than the majority class (0.765 vs 0.784; 0.767 vs 0.767). Is the baseline required? Count or 0/1?"
10. **Activity names (B15).** "Is `376400` the CEA test? It drives our f2 result: 0.12 held-out location importance, and Apriori cannot select it (30 % of cases)."
11. **Length 1 and the single-activity figure (B5, B16).** "All activities × 5 folds × 10 repeats now cost 29–102 s per log. Should length 1 be compared on the top-10 single activities of each strategy, on all activities, or both?"
12. **Versions and scripts (B17).** "Which xgboost version did you use, and can you share the scripts behind Fig. 3 and Fig. 4? Our faithful run matches Fig. 5 on f3, is lower at the top on f1 and about 0.04 lower on f2; at least one itemset per panel differs from ours."
13. **Short confirmations:** length = number of distinct activities (B1); f4 excluded (B4); own re-implementation, automatic mode only (B10); seeded folds with 10 fold seeds (B14).

### 3.3 Table B of the Guide, question by question

"Default" is the recommended default of Guide Part 6, Table B.

| # | Topic | Did the experiments change the default? | New facts and numbers to quote | Still ask? |
|---|---|---|---|---|
| B1 | What "length 1, 2, 3" means | **No, confirmed.** | Step is not length: step-1 candidates on f1 have 9 / 915 / 280 patterns with 1 / 2 / 3 distinct activities. 21 % / 22 % / 15 % of step-2 candidates have 4 activities and are dropped. Apriori per size works (C7). New detail: a length-1 set should be represented by the plain activity, not by a self-loop pattern (C5: `ac419100` falls out of the f2 top 10 at gap 5 otherwise). | Yes, short confirmation. |
| B2 | Reproduce the code as it is, or fix it | **No, but now strongly backed.** | Accumulation is the behaviour that changes results: A / B ratio 1.9 / 1.9 / 8.8; rankings unrelated (0.27 / 0.12 / −0.31); order dependence −0.83 on f2; f3 set 0.256 vs 0.011. The bookkeeping bug alone: +2 to +10 % in level, same ranking. Unseeded folds: two runs of the original procedure agree on the f1 ranking with Spearman 0.53. New sub-question: sequential or uniform draw. | Yes (question 1 and 2). |
| B3 | Training fold or held-out fold | **Yes.** Old: held-out in fixed mode, both only for f1. New: both folds for all logs, 10 fold seeds; held-out is the honest test, the training fold the stable companion. | f1 held-out: mean 0.002, about half of the values above zero (50.9 % in C9 with 20 seeds, 52 % in C8 with 10), two seeds agree with 0.60; none of the 10 Apriori single activities is clearly above zero. Fixed training vs held-out rankings agree (0.82 / 0.94 / 0.88). On f2 sizes 1 and 3 the sign of Apriori-minus-IMPresseD flips between the folds (10 of 10 seeds). | Yes (question 1). |
| B4 | Is f4 excluded | **No.** Not touched by any experiment. | The cost argument in the Guide was for the original code. With the engine f1–f3 take 0.4–2.7 minutes per log and setting; f4 was not run. | Yes, short confirmation. |
| B5 | Apriori top k per size | **No, confirmed and implemented.** | `AprioriSelector` gives 10 + 10 + 10 sets per log; lists saved. The name tie-break decides membership in one place only (f2 size 2: three pairs with 648 cases for two places). Pairs and triples use 5–7 activities per log. | Only the length-1 point (question 11). |
| B6 | `min_support` per log | **Yes, slightly.** f3: 0.45 instead of 0.4 (largest grid value with ≥ 10 sets per size). The lists are the same for 0.45, 0.4 and lower. | Exact thresholds for ≥ 10 sets per size: f1 ≤ 0.5354, f2 ≤ 0.5496, f3 ≤ 0.4788. Original selection on f3: 5 sets at 0.5, 18 at 0.49; same top 10 for any value ≤ 0.4941. | Yes (question 3). |
| B7 | Whole front or fixed k | **Yes, the details.** k = 10 by layers and `distinct=False` stay. New: sets come from all evaluated patterns; lower layers are needed at length 1. | First-front sizes per length: f1 7 / 14 / 21, f2 7 / 34 / 45, f3 7 / 14 / 9. The probe numbers 19 / 32 / 29 in the old question are wrong for our distance (they match the scipy 1.18.1 behaviour: 17 / 33 / 29); with two objectives 3 / 4 / 1 is confirmed. `distinct=True` changes 1 of 90 sets, rounding 2 of 90. 7 of 90 selected sets occur in fewer than 10 cases. | Yes (questions 4 and 5). |
| B8 | Select then project, or re-score sets | **Sharpened.** The prototype does a third thing: one representative pattern per set, then layers of sets per length. | Against "layers of patterns, then remove duplicate sets" the top 10 share 9 / 10 / 9 (f1), 8 / 10 / 10 (f2), 5 / 8 / 7 (f3). Only 5–8 of 10 selected sets per cell have their source pattern on a step front. | Yes (question 4). |
| B9 | Where pattern discovery runs | **No.** Whole log, leakage stated as a limitation. | Labels steer which patterns are extended (1204 step-1 candidates on f1 with real labels, 1618 with shuffled labels). One 80 % subset of f1 keeps 8 / 7 / 8 of 10 sets. The planned train-only sensitivity run was not done; it needs the `fit()` shape check first. | Yes (question 8). |
| B10 | Own re-implementation, automatic mode only | **No, confirmed by measurement.** | Engine: difference 0.0 to the 2024 code. Chain-only IMPresseD: identical patterns and values to the 2023 code. Original IMPresseD: 27.2 min for one step on f1, about 4–5.5 h for two (extrapolated). | Yes, short confirmation. |
| B11 | Which analyses | **Yes.** Drop "rank correlation on shared sets": there are 0–4 shared sets and a shared set has identical importance under both strategies. "At least 3 fold seeds" becomes 10. | Use instead: overlap (Jaccard 0.18 / 0.18 / 0.25 at length 1, 0–0.05 above), importance distributions per length, sets clearly above zero, importance per changed share, activity level (5–7 common activities, Spearman 0.3–0.7, not significant), distinct activities covered. Location vs existence per set is only meaningful on f1. | Yes (question 7). |
| B12 | IMPresseD configuration | **Partly.** Chain traces and three objectives confirmed. Gap 3 kept but is now known to matter. Case distance: explicit function instead of `pdist`. | Equal-timestamp pairs 88.8 / 87.1 / 88.4 % (filtered logs); 73.7 / 68.4 / 73.5 % of events in blocks ≥ 10; both old "not re-run" probes confirmed (78.2 %, 30.8 %). Gap: see C5 table. Two objectives: step fronts 3 / 2 / 1, 4 / 6 / 5, 1 / 1 / 2. The GUI's distance is mostly a control-flow distance (0.97 vs 0.14). | Yes (question 6). |
| B13 | Existence baseline | **No, confirmed; one tag removed.** The "accuracy, not F1" statement no longer needs the assumption tag (C18). | f1: scorer Spearman 1.000, encoding 0.89, fold seed 0.81–0.82. f2 and f3: no skill; on f3 one set is +0.0305 with accuracy and −0.0569 with weighted F1. Report over ≥ 5 seeds. | Yes (question 9). |
| B14 | Runtime, seeds, fewer repeats | **Yes.** No reduction and no overnight run. Instead: 10 fold seeds. | Original grid: 5.3 / 9.2 / 4.4 h (the Guide's 5.6 h holds for f1, is too low for f2). Engine: 30–159 s per log and setting. Full design: about 25 min. | Only "may we fix the seeds and use 10 fold seeds". |
| B15 | Label meaning, activity names | **No.** Not tested (experiment C13 was not run). | `376400` now carries the f2 result (0.1196 alone on the held-out fold). On f1 its location importance is −0.0002 although it has the highest information gain. | Yes (question 10). |
| B16 | Single-activity analysis over all activities | **Yes.** Cost is no longer the reason to restrict it. | Engine: 29–102 s per log and setting; original: 12.8 / 25.1 / 9.1 h. The faithful single-activity routine equals the original on 952 values. On a held-out fold 27 of 164 f1 activities do not occur at all. | Yes (question 11): it is now a question of what to show. |
| B17 | xgboost version, scripts of Fig. 3 / 4 | **No.** Not tested. | Fig. 5 comparison by eye: f3 agrees, f1 lower at the top, f2 about 0.04 lower. Paper's panels show five f1 rows starting with `ac370443` (we have four such sets) and four f2 rows starting with `ac419100` (we have three). | Yes (question 12). |

---

## 4. What still needs a person

**Things to find or download (the user offered to help with these).**
1. **The paper behind the case distance:** Cheung & Jia, "Categorical-and-numerical-attribute data clustering based on a unified similarity metric without knowing cluster number", Pattern Recognition 46(8), 2013. It is reference [8] of the 2023 paper. Nobody has read it. It decides whether our "share of differing attributes" is the intended distance.
2. **The original BPIC 2011 log (XES) from 4TU and the Teinemaa benchmark folder** (links in Guide Part 6, row C13). We need the activity names behind the codes, above all for `376400`, and the meaning of the labels. This download was not done.
3. **The supervisor's answers from office hour 1 (2 Oct).** The experiments ran the same day and do not know them.

**Another machine.**
4. **macOS install (Guide row C15).** Nobody has tested the pinned requirements on the Mac. Steps: Python 3.12 venv, `pip install -r requirements.txt`, `pip check`, then the three test commands of section 5. All timings in this file are from one Windows PC.

**Human decisions (group first, then supervisor where marked in 3.2).**
5. Which fold leads the report, and sequential or uniform draw.
6. Minimum number of cases for an IMPresseD set; the set-ranking rule; whether to adopt rounding and the plain-activity rule for length 1. These change the IMPresseD lists, so decide them before the final run.
7. Whether length 1 is compared on top-10 single activities or on all activities.
8. Whether to run one extra gap (2 or 5) as a robustness check.
9. Whether to add development tools to the venv. `pytest` and a linter are not installed, so the tests are plain scripts or `unittest`, and code style was checked by hand only. Installing packages is a change to the pinned environment and needs a group decision.

**Work nobody has done yet.**
10. The code fixes of section 3.1, item 12, and a re-run of the tests after them.
11. Train-only mining as a sensitivity run on f1 (B9).
12. An analysed faithful run and the existence importance for the project's 60 sets.
13. Guide rows C13, C15, C16, C17 (data provenance, Mac, pm4py API names, literature look-ups) were not part of this batch.
14. Reading the f1 and f2 panels of the paper's Fig. 5 row by row, to see which tied sets the authors had. This needs a person with the PDF.

**Housekeeping.**
15. `results/experiments/C12/tmp` (216 MB) and `results/experiments/C11/tmp` hold throw-away installs of scipy 1.11.4 and paretoset 1.2.0. They are only needed to re-run C11 and C12 and can be deleted. The project venv was not changed by them.
16. `results/experiments/C2/RESULT.md`, section 2, still names `verify_engine.py --part edge` and `--part grid`. Those parts now live in `verify_engine_round1.py`; the current `verify_engine.py` is the second reviewer's script.

---

## 5. Prototype code: modules, verification status, how to re-run

All prototype code is in `experiments/`. It is prototype code: read it, check it, adapt it. `experiments/README.md` repeats the run instructions.

### 5.1 Modules to reuse

| File | What it is | Verification status |
|---|---|---|
| `engine.py` | Loader (`EventLog`), encoder (`IndexEncoder`), seeded folds (`make_folds`), and `LocationPermutationImportance` with faithful and fixed mode, for sets and for single activities | **Independently verified.** 23 builder checks pass; a second reviewer passed it in round 2 with own checks. The 18 quick checks were re-run while writing this report: 18 passed. Five small open issues (section 2, C2). |
| `engine_reference.py` | Loads the original 2024 `tools.py` read-only under the name `tools_2024`, with the int cast that pandas 2 needs | Used by every equality check. No tests of its own. |
| `apriori_selector.py` | `AprioriSelector` (top k per size) and `original_top10` (the original rule) | **Self-checked.** Equal to the original function in 24 of 24 comparisons; brute-force recount of 54 counts and 90 supports. No second reviewer. |
| `case_distance.py` | Explicit case distance and the pattern-level case distance | **Self-checked.** 18 unit tests (re-run for this report: pass); equal to the original numeric part and pattern mean to 2.2e-16. The C4 reviewer re-computed the pairwise distance with an own loop: difference 0.0. |
| `impressed_chain.py` | Chain-only IMPresseD: patterns, extension, scoring, Pareto layers, `ImpressedChainSelector` | **Independently verified.** 34 unit tests (re-run for this report: pass); identical to the original code on every comparison; second reviewer: 53 checks passed, 0 failed. Four medium open issues (section 2, C4). |
| `existence_importance.py` | `ExistenceImportance`: the existence baseline with switches for encoding, matching, scorer and fold | **Not independently verified.** No unit tests. It is a re-implementation; the original script could not be run on BPIC11. |
| `compare.py` | Measures for comparing two lists of sets (Jaccard, overlap@k, best match, Spearman / Kendall, activity-level means) and the pilot tables | **Self-checked** on hand-made toy lists; two table cells recomputed by hand. No second reviewer. |
| `run_pilot.py`, `pilot.yaml` | The pilot pipeline and its settings | **Cross-checked** against C9: 169,000 values, largest difference 5e-11. |
| `original_impressed_patched/` | Copy of the 2023 code with three performance patches (`patches.diff`) | **Verified identical** to the original on 120 and 300 cases (steps 0–2) and on full f1 (steps 0–1). |

### 5.2 Tests

| File | What it checks | Time |
|---|---|---|
| `test_engine.py` | 23 checks of the engine against the original code and against its own rules (`--quick`: the 18 that do not run the slow original) | quick 3 min; all about 40 min; `--full-single` about 70 min |
| `test_impressed_chain.py` | 34 unit tests on toy traces | 5 s |
| `test_case_distance.py` | 18 unit tests on hand-computed examples | under 1 s |

### 5.3 Experiment scripts (one line each)

| File | Experiment | What it does |
|---|---|---|
| `c1_common.py` | C1 | timing helpers and `load_itemsets` (Apriori and IMPresseD sets per size; also used by C9) |
| `c1_original_vs_engine.py` | C1 | one fold: seconds per iteration, original against engine, same model and sets |
| `c1_engine_grid.py` | C1 | whole project grid and single-activity analysis with the engine, timed |
| `c1_original_single.py` | C1 | one full pass of the original single-activity routine, timed per activity |
| `c1_threads.py` | C1 | effect of the XGBoost thread count |
| `c1_report.py` | C1 | tables and extrapolations from the timing files |
| `engine_benchmark.py` | C2 | speed of the engine against the original on f1 |
| `engine_full_grid.py` | C2 | the original's grid with the engine in five settings |
| `engine_stability.py` | C2 | rank stability over 10 fold seeds |
| `engine_fixed_options.py` | C2 | effect of `allowed_from` and `draw` |
| `run_original_impressed.py` | C3 | runs the original IMPresseD automatic mode without the GUI |
| `summarise_original_impressed.py` | C3 | tables for one such run |
| `compare_impressed_runs.py` | C3 | checks that two runs gave the same output |
| `c4_crosscheck_original.py` | C4 | compares `impressed_chain.py` with the unchanged 2023 functions |
| `c4_run_selector.py` | C4 | runs the selector on f1, f2, f3 and writes the IMPresseD set files; holds `load_selector_inputs` |
| `c5_oracle_blocks.py` | C5 | what `delta_time = 0` does to the traces |
| `c5_c6_sensitivity.py` | C5, C6 | nine selector configurations per log, overlap tables |
| `run_c7_apriori_sweep.py` | C7 | support sweep, original replica, saved Apriori lists |
| `c8_run.py` | C8 | settings A–D (faithful, fixed, held-out, reversed order) |
| `c8_supplement.py` | C8 | settings E and F (separate the shuffle from the piling up) |
| `c8_surplus_events.py` | C8 | counts events a shuffle leaves in place |
| `c8_report.py` | C8 | tables, rank correlations, box plots |
| `c9_seed_stability.py` | C9 | importance values for many fold seeds |
| `c9_analysis.py` | C9 | stability tables for the original sets |
| `c9_project_sets.py` | C9 | stability tables for the project's 60 sets |
| `c9_overview.py` | C9 | one table across all C9 studies |
| `run_c10.py` | C10 | the existence-baseline grid, tables and plots |
| `c11_paretoset_versions.py` | C11 | paretoset 1.2.0 against 1.2.5 |
| `c12_scipy_worker.py` | C12 | runs the original distance code under one scipy version |
| `c12_compare_case_distance.py` | C12 | our distance against the original under two scipy versions |

Reviewer scripts live next to the results: `results/experiments/C2/verify_engine.py` (round 2) and `verify_engine_round1.py`, `results/experiments/C4/verify_impressed.py`, `results/experiments/C5/check_gap_counts.py`, `results/experiments/pilot/check_against_c9.py`.

### 5.4 How to re-run everything

Run every command from the project root (`project/`), in Git Bash, with the project's Python 3.12:

```bash
PY=.venv/Scripts/python.exe
```

In PowerShell write `.\.venv\Scripts\python.exe` instead of `$PY` and run loop bodies one by one. Never use the machine's default `python` (it is 3.14).

**Warning:** a re-run overwrites the files in `results/experiments/<id>/`. Copy the folder first if you want to keep the old numbers. Times are from a shared machine.

**Order matters for a few steps.** C7 writes the Apriori lists and C4 writes the IMPresseD lists; C1, C5, C9 (project sets) and the pilot read them. C3's exports are read by the C4 reviewer script. C9's raw files are read by the pilot cross-check.

```bash
# ---- Tests ---------------------------------------------------------------
$PY -m unittest experiments/test_impressed_chain.py -v      # 34 tests, 5 s
$PY -m unittest experiments/test_case_distance.py -v        # 18 tests, < 1 s
$PY experiments/test_engine.py --quick                      # 18 checks, 3 min
$PY experiments/test_engine.py                              # all 23 checks, about 40 min (runs the original code)
$PY experiments/test_engine.py --full-single --only faithful_single_full_f1   # about 70 min

# ---- C7: Apriori sweep and Apriori set lists (1-2 min) -------------------
$PY experiments/run_c7_apriori_sweep.py

# ---- C11 / C12: paretoset versions, case distance (30 s + 40 s) -----------
# The two pip lines are only needed if the tmp folders were deleted. They install
# old versions into a throw-away folder, not into the venv.
$PY -m pip install paretoset==1.2.0 --target results/experiments/C11/tmp/paretoset_1_2_0 --no-deps
$PY -m pip install scipy==1.11.4 numpy==1.26.4 --target results/experiments/C12/tmp/scipy_1_11_4 --no-deps --only-binary=:all:
$PY experiments/c11_paretoset_versions.py
$PY experiments/c12_compare_case_distance.py

# ---- C2: engine speed, grids, stability, options -------------------------
$PY experiments/engine.py --dataset f1 --mode fixed --score-on test             # small demo, prints a table
$PY experiments/engine_benchmark.py --dataset f1 --repeats 2 --timing-runs 3    # about 3 min
for d in f1 f2 f3; do $PY experiments/engine_full_grid.py --dataset $d; done
for d in f1 f2 f3; do $PY experiments/engine_stability.py --dataset $d --seeds 10 --repeats 30; done       # 36-53 min per log
for d in f1 f2 f3; do $PY experiments/engine_fixed_options.py --dataset $d --seeds 10 --repeats 10; done   # 11-27 min per log
# reviewer's script, parts: encoder, faithful, single, fixed, toy, deadend, noise, hashseed
$PY results/experiments/C2/verify_engine.py --part fixed --dataset f1

# ---- C3: the original IMPresseD code -------------------------------------
# original code, full f1, one extension step (27 min)
$PY -u experiments/run_original_impressed.py --max-extension-step 1 --run-name f1_step1 \
    --time-limit-min 45 --patterns-json results/experiments/C3/original_patterns_f1.json \
    > results/experiments/C3/logs/f1_step1.log 2>&1
# original code, 120 / 300 cases, two steps (3 / 6 min)
$PY -u experiments/run_original_impressed.py --max-extension-step 2 --max-cases 120 --run-name smoke120
$PY -u experiments/run_original_impressed.py --max-extension-step 2 --max-cases 300 --run-name scale300_step2 --time-limit-min 30
# patched copy: same commands plus --code-dir (full raw f1, two steps: 2.6 min)
$PY -u experiments/run_original_impressed.py --max-extension-step 2 --max-cases 120 --run-name smoke120_patched \
    --code-dir experiments/original_impressed_patched
$PY -u experiments/run_original_impressed.py --max-extension-step 2 --run-name f1_step2_patched --time-limit-min 40 \
    --code-dir experiments/original_impressed_patched
# patched copy on the 2024-filtered logs (f1 shown; same for BPIC11_f2_trunc40.csv and BPIC11_f3_trunc31.csv)
$PY -u experiments/run_original_impressed.py \
    --dataset external/PermutationLocationImportance/datasets/BPIC11_f1_trunc36.csv --log-prep dm2024 \
    --max-extension-step 2 --run-name f1_dm2024_step2_patched --code-dir experiments/original_impressed_patched \
    --time-limit-min 15 --patterns-json results/experiments/C3/patched_patterns_f1_dm2024.json
# equality check and tables
$PY experiments/compare_impressed_runs.py results/experiments/C3/smoke120 results/experiments/C3/smoke120_patched
$PY experiments/summarise_original_impressed.py results/experiments/C3/f1_step1/timings.json results/experiments/C3/original_patterns_f1.json

# ---- C4: chain-only IMPresseD ---------------------------------------------
$PY -u experiments/c4_crosscheck_original.py --full-core ac370000 --full-front > results/experiments/C4/crosscheck_run.log 2>&1   # 15 min
$PY -u experiments/c4_run_selector.py > results/experiments/C4/selector_run.log 2>&1                                             # 24 s, writes the IMPresseD set lists
$PY results/experiments/C4/verify_impressed.py                                                                                   # reviewer's script, about 6 min

# ---- C5 / C6: sensitivity of the IMPresseD side ---------------------------
$PY -u experiments/c5_oracle_blocks.py  > results/experiments/C5/c5a_run.log  2>&1    # 1.5 min
$PY -u experiments/c5_c6_sensitivity.py > results/experiments/C5/c5c6_run.log 2>&1    # 5 min
$PY results/experiments/C5/check_gap_counts.py

# ---- C10: existence baseline (9 min) -------------------------------------
$PY experiments/run_c10.py > results/experiments/C10/run.log 2>&1

# ---- C1: runtime ---------------------------------------------------------
for d in f1 f2 f3; do $PY experiments/c1_original_vs_engine.py --dataset $d; done                        # 26 min in total
$PY experiments/c1_original_vs_engine.py --dataset f2 --tag _run2                                        # second f2 measurement, 12 min
for d in f1 f2 f3; do $PY experiments/c1_engine_grid.py --dataset $d; done                               # 22 min in total
for d in f1 f2 f3; do $PY experiments/c1_engine_grid.py --dataset $d --repeats 2 --tag _demo; done       # 4 min
for d in f1 f2 f3; do $PY experiments/c1_engine_grid.py --dataset $d --n-jobs 1 --tag _nj1; done         # may be started in three terminals at once
$PY experiments/c1_threads.py --dataset f1
$PY experiments/c1_threads.py --dataset f3
for d in f1 f2 f3; do $PY experiments/c1_original_single.py --dataset $d; done                           # 15 / 30 / 11 min
$PY experiments/c1_report.py

# ---- C8: faithful against fixed ------------------------------------------
for d in f1 f2 f3; do $PY experiments/c8_run.py --dataset $d > results/experiments/C8/run_$d.log 2>&1; done                       # 17-29 min per log
for d in f1 f2 f3; do $PY experiments/c8_supplement.py --dataset $d > results/experiments/C8/run_supplement_$d.log 2>&1; done     # 8-12 min per log
$PY experiments/c8_surplus_events.py > results/experiments/C8/surplus_events.log 2>&1
for d in f1 f2 f3; do $PY experiments/c8_report.py --dataset $d > results/experiments/C8/report_$d.log 2>&1; done

# ---- C9: seed stability --------------------------------------------------
# main study, f1, 20 fold seeds (the four calls may run side by side; about 165 s per fold seed)
for S in 0 5 10 15; do $PY experiments/c9_seed_stability.py --dataset f1 --first-seed $S --n-seeds 5; done
# f1 with 10 folds
for S in 0 5; do $PY experiments/c9_seed_stability.py --dataset f1 --folds 10 --first-seed $S --n-seeds 5 --repeats 10 --faithful-repeats 0; done
# f2 and f3, original sets
for d in f2 f3; do for S in 0 5; do $PY experiments/c9_seed_stability.py --dataset $d --first-seed $S --n-seeds 5 --repeats 10 --faithful-repeats 10; done; done
# the project's 60 sets (needs the C7 and C4 set lists); f1 with 20 seeds, f2 and f3 with 10
for S in 0 5 10 15; do $PY experiments/c9_seed_stability.py --dataset f1 --itemsets project --first-seed $S --n-seeds 5 --repeats 10 --faithful-repeats 0; done
for d in f2 f3; do for S in 0 5; do $PY experiments/c9_seed_stability.py --dataset $d --itemsets project --first-seed $S --n-seeds 5 --repeats 10 --faithful-repeats 0; done; done
# analysis
$PY experiments/c9_analysis.py --dataset f1
$PY experiments/c9_analysis.py --dataset f1 --max-seeds 10
$PY experiments/c9_analysis.py --dataset f1 --folds 10
$PY experiments/c9_analysis.py --dataset f2
$PY experiments/c9_analysis.py --dataset f3
$PY experiments/c9_project_sets.py --dataset f1
$PY experiments/c9_project_sets.py --dataset f1 --max-seeds 10
$PY experiments/c9_project_sets.py --dataset f2
$PY experiments/c9_project_sets.py --dataset f3
$PY experiments/c9_overview.py

# ---- Pilot: Apriori sets against IMPresseD sets --------------------------
for d in f1 f2 f3; do $PY -u experiments/run_pilot.py --logs $d > results/experiments/pilot/run_$d.log 2>&1; done   # 16 / 22 / 13 min
$PY experiments/run_pilot.py --logs f1 f2 f3 --figures-only    # redraw the box plots, 5 s
$PY experiments/compare.py                                     # tables, 2 s
$PY results/experiments/pilot/check_against_c9.py              # needs the C9 raw files
```

Notes on the commands:
- The reports of C1, C2, C8, C9 and the pilot give their commands from inside `experiments/`. The commands above are the same calls written from the project root. The scripts find their data from their own location, so both ways work. `test_engine.py --quick` and the `engine.py` demo were run from the project root for this report.
- In C9 the project sets of f1 were originally run in six calls that cover fold seeds 0–19; the four calls above cover the same seeds.
- The first figure to open: `results/experiments/pilot/box_f2_len2_test.png`. For C8: `results/experiments/C8/box_f1_all_settings.png`.

---

## 6. Where the reports disagree, and which number to use

1. **Engine speed (C2).** An early version of the C2 report gave 134× / 231× / 810× against an original at 13.58 s per iteration. That was measured on a loaded machine. The final report gives 128× / 189× / 578× against 6.78 s on a quiet machine. Use the second set. C1 measured 102–963× on all three logs.
2. **"k = 10 never leaves the first front" (C3) against "the first front has only 7 sets at length 1" (C4, C5, C12).** C3 used the original tool's settings (GUI-style distance under scipy 1.18.1, 80 % training split). C4 uses the project settings. For the project, C4 is right.
3. **Length-1 front sizes 19 / 32 / 29 (old probe in the Guide, question B7).** Not reproduced. Our distance gives 7 / 7 / 7. The original distance call under scipy 1.18.1 gives 17 / 33 / 29. Do not quote the old numbers.
4. **Apriori support for f3: 0.4 (Guide, C4's overlap count) or 0.45 (C7, C1, C5, C9, pilot).** Both give the same top-10 lists (checked in C7).
5. **How many sets the gap changes.** The pilot report says "2–6 of 10". C5's own table gives 2–5 of 10 at length 2 and 2–7 of 10 at length 3, compared with gap 3. Use C5.
6. **f3 means in C2 and C8 differ a little** (C2: 0.2473 / 0.0328 / 0.0099; C8: 0.2496 / 0.0284 / 0.0086). C2 used the 5 sets f3 has at support 0.5 and one fold seed; C8 used 10 sets at support 0.49 and ten fold seeds. Both are right for their setting. For fold seed 0 and the shared sets the two experiments agree exactly.
7. **Cost of the final design.** C9 measured 90 minutes of single-thread time (20 minutes wall as six processes); the pilot measured 51 minutes of single-thread time (24 minutes wall as three processes). Same design, both on a shared machine. Plan with "about half an hour".
8. **Where the three-objective evidence is filed.** The C5 report files it under "B1". In the Guide, B1 is the meaning of "length"; the interest functions belong to B12 and B7. Section 3.3 follows the Guide.
