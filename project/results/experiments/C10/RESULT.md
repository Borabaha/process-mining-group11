# C10 + C18 — The existence-importance baseline

Run on 2 Oct 2026, Windows 10, Python 3.12 venv (scikit-learn 1.9.1, xgboost 3.4.1, pandas 2.3.3).
Feeds decision **B13** (Guide Part 6, Table B) and closes the `[ASSUMPTION: older-sklearn scorer]` tag in Guide §3.3.7.

## 0. Short answers

| Question | Answer |
|---|---|
| C18: does "no `scoring` argument" mean accuracy, now and in the past? | **Yes.** In every release line checked from 0.22 (where the function was introduced) to 1.9.1, `scoring=None` resolves to `estimator.score`, which is accuracy for a classifier. The original existence importance is therefore a drop in **accuracy**, not in F1. |
| C10: does accuracy vs weighted F1 change the existence ranking? | On **f1: no** (Spearman rho = 1.000, train and test fold). On f2 and f3 it does, but there the existence model is no better than always predicting the majority class, so every ranking is noise. |
| C10: does count vs 0/1 encoding change the ranking? | On **f1: a little** (rho = 0.89 on the test fold, 0.95 on the train fold; same four sets on top). Less than changing the fold seed (median rho 0.81-0.82 between seeds). |
| Substring vs exact matching on f1 | Differs in **1 of 11,300** feature values (1 case, set {370407, ac370000}). |
| Which combination should we use? | **count encoding with exact tokens + `scoring='f1_weighted'`, scored on the held-out fold, averaged over several fold seeds.** Keep 0/1 encoding as a config switch. See section 6. |

## 1. What was run

```
cd "C:/Users/Bohabara/Desktop/process mining/project"
.venv/Scripts/python.exe experiments/run_c10.py > results/experiments/C10/run.log 2>&1
```

Wall time 540 s (other experiments were running on the machine at the same time). One `ExistenceImportance.run` takes 2-13 s.

Code (prototype, read and adapt):

- `experiments/existence_importance.py` — class `ExistenceImportance`, `encode_existence`, `load_log`, `apriori_itemsets`.
- `experiments/run_c10.py` — the experiment grid, diagnostics, tables, plots.

Settings

| Setting | Value |
|---|---|
| Logs | BPIC11 f1, f2, f3, loaded with the original `DataManager(path, 2, None, L_max_perc=0.8)` (rare-activity filter on: 1130 / 1130 / 1111 cases) |
| Activity sets | `DataManager.frequent_activity_sets(0.5, 10)` (int `top_k`): 10 sets for f1 and f2, 5 for f3 |
| Features | one column per activity set, nothing else (as in the original) |
| Encoding | `count` = min over the set's activities of the exact-token occurrence count; `binary` = set fully present or not |
| Model | `XGBClassifier(random_state=0, n_jobs=1)`, otherwise defaults |
| Folds | `StratifiedKFold(5, shuffle=True, random_state=2023)`; identical folds for all combinations |
| Permutation | `sklearn.inspection.permutation_importance(..., n_repeats=20, random_state=42)` -> 5 x 20 = 100 values per set |
| Scoring | `accuracy` or `f1_weighted`, on the train fold (original) or the held-out fold |
| Seed check | the four combinations (held-out fold) repeated with fold seeds 2023, 0, 1, 2, 3 |
| "Faithful" configuration | unfiltered cases (`frq_threshold=1`), substring count, accuracy, train fold (folds seeded, which the original does not do) |

Output files in this folder: `tables.md` (every table, generated), `importances_long.csv` (all raw values), `mean_importance_<log>_<fold>.csv`, `spearman_<log>_<fold>.csv`, `seed_means_<log>.csv`, `faithful_vs_default_<log>.csv`, `existence_<log>_<fold>.png`, `run.log`.

Two runs of the script gave identical numbers for the seed-2023 tables (checked between the first and the final run).

## 2. C18 — what `scoring=None` means

### Installed version (scikit-learn 1.9.1, read in the venv)

- `sklearn/inspection/_permutation_importance.py:143` — signature has `scoring=None`.
- `:285-286` — `scorer = check_scoring(estimator, scoring=scoring)`, then `baseline_score = _weights_scorer(scorer, estimator, X, y, sample_weight)`.
- `sklearn/metrics/_scorer.py:1042-1044` — `if scoring is None: if hasattr(estimator, "score"): return _PassthroughScorer(estimator)`; `:536-537` — the passthrough scorer returns `estimator.score(*args, **kwargs)`.
- `sklearn/base.py:602-628` — `ClassifierMixin.score` returns `accuracy_score(y, self.predict(X), sample_weight=sample_weight)`.
- Run in the venv: `XGBClassifier.score.__qualname__` is `ClassifierMixin.score`; `check_scoring(XGBClassifier(), scoring=None)` prints as `XGBClassifier.score`.
- Numerical check on the f1 existence features: `permutation_importance(model, X, y, n_repeats=20, random_state=42)` returns an array **identical** to the call with `scoring='accuracy'`, and different from `scoring='f1_weighted'` (max absolute difference 0.0145). `n_jobs=1` and `n_jobs=2` also give identical arrays.

### History (source read at GitHub release branches through WebFetch)

| Branch | `permutation_importance` code that picks the scorer when `scoring=None` | Docstring |
|---|---|---|
| 0.21.X | function does not exist (`sklearn/inspection/__init__.py` exports only partial dependence) | - |
| 0.22.X | `scorer = check_scoring(estimator, scoring=scoring)` | "If None, the estimator's default scorer is used." |
| 0.23.X | same | same |
| 0.24.X | same (adds `sample_weight`) | same |
| 1.0.X, 1.1.X, 1.2.X, 1.3.X, 1.4.X | `elif scoring is None or isinstance(scoring, str): scorer = check_scoring(estimator, scoring=scoring)` (new branches only for callables and multi-metric input) | same |
| 1.5.X, 1.6.X | `scorer = check_scoring(estimator, scoring=scoring)` | same |
| 1.7.X, 1.8.X, 1.9.1 | same | "`None`: the estimator's default evaluation criterion is used." |

`check_scoring` with `scoring=None` returns a passthrough to `estimator.score` in the three older branches read (0.22.X and 1.0.X: `_passthrough_scorer`; 1.3.X: `_PassthroughScorer(estimator)`) and in 1.9.1. `ClassifierMixin.score` is `accuracy_score(...)` in 0.22.X and in 1.9.1. For xgboost 1.7.6, `class XGBClassifier(XGBModel, XGBClassifierBase)` and `compat.py` has `from sklearn.base import ClassifierMixin as XGBClassifierBase`.

### Conclusion and limits

"No scoring argument = `estimator.score` = accuracy for a classifier" has held since the function was introduced in 0.22. `Classical_Permutation.py:108-109` therefore measures the drop in training **accuracy**; its axis label "Decrease in f1 score" (`:143`) is wrong. Guide §3.3.7 and B13 can state this without the assumption tag.

Not checked: `check_scoring` in the branches between those listed (0.23, 0.24, 1.1, 1.2, 1.4-1.8); whether xgboost 1.7.6 defines its own `XGBClassifier.score` (the fetch tool cut the long file off; in 3.4.1 it does not). WebFetch returns excerpts through a summarising model, not the raw bytes. Which versions the authors ran is unknown (the repo has no requirements file).

## 3. C10 — what `Classical_Permutation.py` does

| Aspect | What the script does | Line |
|---|---|---|
| Inputs | Hard-coded bpic2012 paths; needs the location-importance CSV of an earlier run. Cannot run as shipped on BPIC11. | 19-21 |
| Loading | `DataManager(address, 2, None, 1)`: the fourth positional argument is `frq_threshold=1`, so **no case is dropped** (f1: 1140 cases instead of the 1130 of the location run). | 18, 25-26 |
| Activity sets | Read from the **header** of the location CSV with `eval`, i.e. in the order of mean location importance. | 29-31, 56 |
| Encoding | Trace joined to one string with `'->'`; feature = `min(trace.count(act) for act in set)`, a **substring count**. Not the binary encoding of the paper (p.197), and the flag `Binary_Existence = True` is unused in this branch. | 44-58 |
| Features | Only the activity-set columns, named 0..n-1. | 62 |
| Folds | `cross_split_test_train(5)` = `StratifiedKFold(5, shuffle=True)` **without `random_state`**: different folds on every run. | 78; tools.py:231 |
| Model | `XGBClassifier()` with defaults, one per fold, fitted on the train fold. | 91-92 |
| Test fold | Used only to print the weighted F1. | 93-96 |
| SHAP | `TreeExplainer` mean absolute SHAP per feature on the train fold; saved, not in the paper. | 99-106 |
| Importance | `permutation_importance(model, train_x, train_y, n_repeats=20, random_state=42, n_jobs=2)` on the **training** fold, no `scoring` -> accuracy. | 108-109 |
| Output | 5 folds x 20 repeats = 100 values per set, columns sorted by mean; CSV and a box plot (`whis=10`) labelled "Decrease in f1 score" and titled "..._Prefix_Binary Encoding". | 112-148 |
| Seeds | Only `random_state=42` for the permutations (same for every fold). | 109 |

### Substring vs exact matching

A "cell" is one (case, activity set) feature value.

| Log | Version | Cases | Cells | Cells that differ | Cases that differ | Cells where presence (0/1) flips | Sets affected |
|---|---|---|---|---|---|---|---|
| f1 | filtered | 1130 | 11300 | 1 | 1 | 1 | {370407, ac370000} |
| f1 | unfiltered (original) | 1140 | 11400 | 1 | 1 | 1 | {370407, ac370000} |
| f2 | filtered | 1130 | 11300 | 3 | 3 | 0 | {370407, ac370000} |
| f2 | unfiltered (original) | 1140 | 11400 | 6 | 4 | 3 | the three sets containing 370407 |
| f3 | both | 1111 / 1121 | 5555 / 5605 | 0 | 0 | 0 | - |

The only pair of activity names in the three alphabets where one is a substring of another is `370407` inside `370407c` (f1 and f2; `370407c` does not occur in f3). The effect on the numbers is negligible, but it is a real bug: use exact tokens.

## 4. C10 — results on f1

Model quality with the ten existence columns as the only features (mean over 5 folds):

| Model | Test accuracy | Test weighted F1 |
|---|---|---|
| count encoding | 0.7336 | 0.7359 |
| binary encoding | 0.7088 | 0.7101 |
| always predict label 0 | 0.5982 | 0.4478 |

Mean importance and rank, **held-out fold**, fold seed 2023 (100 values per cell):

| Activity set | Support | count/acc | count/f1w | binary/acc | binary/f1w | Ranks (same order) |
|---|---|---|---|---|---|---|
| {370407, ac370000} | 0.5912 | 0.0516 | 0.0613 | 0.0237 | 0.0300 | 3 / 3 / 4 / 4 |
| {ac370000, ac370419} | 0.5770 | 0.0691 | 0.0810 | 0.0526 | 0.0602 | 2 / 2 / 3 / 3 |
| {ac370000, ac370443} | 0.5637 | 0.0259 | 0.0302 | 0.0597 | 0.0616 | 4 / 4 / 2 / 2 |
| {ac370000, ac370419, ac370443} | 0.5619 | 0.0004 | 0.0005 | -0.0013 | -0.0012 | 7 / 7 / 10 / 10 |
| {ac370419, ac370443} | 0.5619 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8 / 8 / 7 / 7 |
| {ac370000, ac370442} | 0.5593 | 0.1043 | 0.1224 | 0.1045 | 0.1295 | 1 / 1 / 1 / 1 |
| {370407, ac370000, ac370419} | 0.5593 | 0.0008 | 0.0010 | 0.0001 | 0.0000 | 6 / 6 / 6 / 6 |
| {370407, ac370419} | 0.5593 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8 / 8 / 7 / 7 |
| {ac370442, ac370443} | 0.5566 | -0.0037 | -0.0037 | -0.0011 | -0.0011 | 10 / 10 / 9 / 9 |
| {ac370000, ac370419, ac370442} | 0.5566 | 0.0013 | 0.0014 | 0.0003 | 0.0004 | 5 / 5 / 5 / 5 |

Spearman rho between the four rankings:

| | count/acc | count/f1w | binary/acc | binary/f1w |
|---|---|---|---|---|
| **Held-out fold** count/acc | 1.000 | 1.000 | 0.890 | 0.890 |
| count/f1w | 1.000 | 1.000 | 0.890 | 0.890 |
| binary/acc | 0.890 | 0.890 | 1.000 | 1.000 |
| binary/f1w | 0.890 | 0.890 | 1.000 | 1.000 |
| **Train fold** count/acc | 1.000 | 1.000 | 0.951 | 0.951 |
| count/f1w | 1.000 | 1.000 | 0.951 | 0.951 |
| binary/acc | 0.951 | 0.951 | 1.000 | 0.976 |
| binary/f1w | 0.951 | 0.951 | 0.976 | 1.000 |

Other comparisons on f1:

| Comparison | Spearman rho |
|---|---|
| Train fold vs held-out fold, same combination | 0.915 (count, both scorers), 0.841 (binary/acc), 0.793 (binary/f1w) |
| Faithful configuration vs default (count/f1w, held-out, filtered) | 0.951; ranks 1-5 identical |
| Between fold seeds, same combination (10 seed pairs) | median 0.805-0.823, minimum 0.634-0.732 |
| count vs binary after averaging over the 5 seeds | 0.793 |
| accuracy vs weighted F1 after averaging over the 5 seeds | 1.000 |

Reading:

- **Scorer.** The ranking is the same. Weighted-F1 importances are about 1.17-1.19 times the accuracy ones for the three largest sets under count encoding.
- **Encoding.** The same four sets are on top in every combination and the same six are near zero. Within the top four the order moves: with seed 2023 the sets ranked 2-4 are permuted; averaged over 5 seeds, count puts {ac370000, ac370419} first (0.0935 vs 0.0865 for {ac370000, ac370442}, weighted F1) and binary puts {ac370000, ac370442} first (0.0930 vs 0.0676).
- **Fold seed.** Under count/f1w, {ac370000, ac370419} is first for 3 of the 5 seeds and {ac370000, ac370442} for the other 2. The importance of {ac370000, ac370442} ranges from 0.042 to 0.122 across seeds. The fold seed moves the ranking at least as much as the encoding does.
- **Paper.** The paper names {ac370000, ac370419} as the f1 set with the highest existence importance (p.199, Fig. 6b). The seed-averaged count encoding agrees; the binary encoding ranks it second.

## 5. C10 — results on f2 and f3

Default combination (count, weighted F1, held-out fold), mean over 100 values, fold seed 2023, and the mean over 5 fold seeds:

| Log | Activity set | Support | Seed 2023 | Rank | Mean of 5 seeds | Rank |
|---|---|---|---|---|---|---|
| f2 | {ac370000, ac379999} | 0.6903 | 0.0061 | 5 | 0.0075 | 5 |
| f2 | {ac370000, ac419100} | 0.6469 | 0.0033 | 7 | 0.0094 | 3 |
| f2 | {ac370000, ac379999, ac419100} | 0.6407 | 0.0095 | 3 | 0.0074 | 6 |
| f2 | {ac379999, ac419100} | 0.6407 | 0.0000 | 8 | 0.0000 | 9 |
| f2 | {370407, ac370000} | 0.6018 | 0.0135 | 2 | 0.0130 | 2 |
| f2 | {370407, ac370000, ac379999} | 0.5929 | -0.0038 | 10 | 0.0002 | 8 |
| f2 | {370407, ac379999} | 0.5929 | 0.0000 | 8 | 0.0000 | 9 |
| f2 | {ac370000, ac370419} | 0.5858 | 0.0182 | 1 | 0.0157 | 1 |
| f2 | {ac370000, ac370443} | 0.5770 | 0.0042 | 6 | 0.0079 | 4 |
| f2 | {ac370419, ac379999} | 0.5761 | 0.0064 | 4 | 0.0015 | 7 |
| f3 | {370407, ac370000} | 0.5275 | -0.0569 | 5 | -0.0538 | 5 |
| f3 | {ac370000, ac370419} | 0.5131 | 0.0040 | 2 | 0.0002 | 2 |
| f3 | {370407, ac370000, ac370419} | 0.5023 | 0.0004 | 3 | -0.0014 | 4 |
| f3 | {370407, ac370419} | 0.5023 | 0.0000 | 4 | 0.0000 | 3 |
| f3 | {ac370000, ac370443} | 0.5014 | 0.0298 | 1 | 0.0277 | 1 |

The existence model has no skill on these two logs:

| Log | Model | Test accuracy | Test weighted F1 | Share of test cases predicted 1 |
|---|---|---|---|---|
| f2 | count encoding | 0.7646 | 0.7180 | 0.922 |
| f2 | binary encoding | 0.7770 | 0.6873 | 0.991 |
| f2 | always predict label 1 | 0.7841 | 0.6892 | 1.000 |
| f3 | count encoding | 0.7669 | 0.6797 | 0.016 |
| f3 | binary encoding | 0.7678 | 0.6758 | 0.010 |
| f3 | always predict label 0 | 0.7669 | 0.6657 | 0.000 |

Consequences (all four combinations were also run on f2 and f3; full tables in `tables.md`):

| Log | rho between the four combinations, held-out fold | rho between fold seeds (median per combination) | rho faithful vs default |
|---|---|---|---|
| f2 | -0.622 to 0.244 | 0.358 to 0.683 | 0.305 |
| f3 (5 sets) | accuracy vs weighted F1: 0.205 to 0.368; count vs binary: 0.821 (accuracy), 0.975 (weighted F1) | 0.684 to 1.000 | 0.000 |

- **f2.** The largest mean held-out importance is 0.0201 with fold seed 2023 (0.058 over all 5 seeds and 4 combinations), and the rankings disagree with each other and across seeds. There is nothing to rank.
- **f3.** The two scorers disagree in **sign** for {370407, ac370000}: +0.0305 with accuracy, -0.0569 with weighted F1 (count encoding; negative for all 5 seeds, -0.051 to -0.057). Checked on the pooled test folds: the fitted model predicts label 1 for 18 of 1111 cases (confusion matrix [[843, 9], [250, 9]]); after one permutation of that column it predicts label 1 for 263 cases ([[702, 150], [146, 113]]). Accuracy falls, minority recall rises from 9/259 to 113/259, so weighted F1 rises. The "importance" here measures how the column suppresses minority predictions, not predictive value.

## 6. Recommendation for B13

Use **count encoding with exact token matching, `scoring='f1_weighted'`, importance on the held-out fold**, and report the mean over at least 5 fold seeds. Keep `encoding='binary'` as a config switch and show it once as a sensitivity check.

Why:

1. **Scorer = weighted F1.** The location importance is a drop in weighted F1 (`tools.py:503-504`, `:538-539`) and the paper puts both importances on one "decrease in f1-score" axis. On f1, the only BPIC11 log where the baseline has signal, weighted F1 gives exactly the ranking that the original's accuracy gives (rho = 1.000), so nothing of the original result is lost.
2. **Encoding = count.** It is what the code behind the paper's figures computed (rho = 0.951 with the faithful configuration on f1, ranks 1-5 identical), the model is slightly better (test accuracy 0.734 vs 0.709 on f1), and its seed-averaged top set is the one the paper names. The paper's text says "binary encoding", so this stays a documented deviation from the text until the supervisor answers B13.
3. **Exact tokens.** Substring matching is a bug with almost no numerical effect (1 cell on f1, 3 on f2, 0 on f3 in the filtered logs).

How much the choice matters:

- On f1: the scorer not at all for the ranking; the encoding only inside the top four (rho 0.89-0.95), less than the fold seed (median rho 0.81-0.82). A single-seed existence ranking should not be interpreted beyond "these four sets matter, these six do not".
- On f2 and f3: the choice changes the ranking completely, and on f3 even the sign, but only because the existence model is at the majority-class level. No combination fixes that. Report the existence importance there together with the model-vs-majority table and say that it carries no information; this matches the paper's remark that the existence of the frequent sets "does not exhibit much importance".

## 7. Findings the implementation has to handle

- **Identical columns.** Nested sets can give the same column: f1 {ac370000, ac370419, ac370443} = {ac370419, ac370443} and {370407, ac370000, ac370419} = {370407, ac370419}; f2 two pairs; f3 one pair (count and binary alike). XGBoost splits only on the first of two identical columns, so the second gets exactly 0.0000 in every run. Which twin gets the credit depends on column order; the original takes that order from the location-importance CSV. Detect identical columns and report them as one group.
- **Correlated columns.** Median pairwise correlation between the ten f1 columns is 0.94 (count) and 0.95 (binary). Permutation importance spreads credit over such columns differently in every fold: the 100 held-out values of {ac370000, ac370442} on f1 span -0.009 to 0.283 under count/f1w (mean 0.122, standard deviation 0.098; `existence_f1_test.png`).
- **Different case sets.** The original existence run uses all cases (1140 on f1), the location run 1130. Our default uses the filtered log for both.
- **Label skew.** f2 has 78.4 % label 1, f3 23.3 %; print the majority-class scores next to every existence result.

## 8. Not done / limits

- The SHAP part of the original script was not reproduced (`shap` is not installed in the venv; it is not in the paper).
- `Classical_Permutation.py` itself was not executed (hard-coded bpic2012 paths, needs a location CSV, unseeded folds). The "faithful" configuration is a re-implementation of its logic with seeded folds and Apriori column order, so it is not bit-identical to an original run.
- Spearman on 10 sets (5 on f3) is coarse and several sets tie at exactly 0; ties get average ranks.
- Seed stability used 5 fold seeds; the permutation seed (42) and the model seed (0) were not varied.
- C18 limits are listed at the end of section 2.
