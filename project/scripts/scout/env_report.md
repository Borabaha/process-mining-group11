# Environment report — Python environment for the Group 11 PPM project (Windows 10)

Scout run on 2026-10-01. Everything below was actually executed on this machine unless tagged
[UNVERIFIED] / [ASSUMPTION]. Smoke-test scripts are kept next to this file:
`smoke_2024.py`, `smoke_2024b.py`, `smoke_2024c.py`, `smoke_2023.py` (same directory as this report).

## 0. TL;DR

| Item | Result |
|---|---|
| Interpreter | Python **3.12.10** via `py -3.12` (default `py` is 3.14.3 — do not use; relayed from the orchestrator, not re-tested here) |
| venv | `C:/Users/Bohabara/AppData/Local/Temp/claude/C--Users-Bohabara-Desktop-process-mining/05db42ec-3109-4098-847a-cf389d8c070a/scratchpad/venv312` |
| All packages | installed with current wheels, `pip check` = "No broken requirements found" |
| 2024 repo (PermutationLocationImportance) | loads data, Apriori, index encoding and XGBoost fit all work; **the core `itemset_permutation_importance` crashes under pandas 2.x** (bool-vs-int dtype upcast to `object` → XGBoost `ValueError`). Verified one-line workaround: cast encoded feature columns to `int`. |
| 2023 repo (InteractivePatternDetection / IMPresseD) | `import IMIPD`, `import Auto_IMPID` succeed without the GUI; pm4py 2.7.23.8 and paretoset 1.2.5 work |
| Runtime estimate | ~6.7 s per (itemset, repeat) on BPIC11_f1 → ~56 min for the repo's defaults (5 folds × 10 itemsets × 10 repeats) on this laptop [extrapolation, see §3.4] |

## 1. Commands that worked (in order)

### 1.1 Bash form (Git Bash, exactly what was run)

```bash
# 1. venv OUTSIDE the project, with Python 3.12
py -3.12 -m venv "C:/Users/Bohabara/AppData/Local/Temp/claude/C--Users-Bohabara-Desktop-process-mining/05db42ec-3109-4098-847a-cf389d8c070a/scratchpad/venv312"
VP="C:/Users/Bohabara/AppData/Local/Temp/claude/C--Users-Bohabara-Desktop-process-mining/05db42ec-3109-4098-847a-cf389d8c070a/scratchpad/venv312/Scripts/python.exe"
"$VP" --version            # Python 3.12.10
# 2. pip upgrade (25.0.1 -> 26.2.1)
"$VP" -m pip install --upgrade pip
# 3. core packages (unpinned at install time; versions resolved by pip are listed in §2)
"$VP" -m pip install numpy "pandas<3" scikit-learn xgboost mlxtend networkx paretoset scipy matplotlib pyyaml
# 4. pm4py (pulls lxml, graphviz, cvxopt, tqdm, colorama)
"$VP" -m pip install pm4py
# 5. optional, only needed by Pixel_Flipping_Process.py / Classical_Permutation.py of the 2024 repo
"$VP" -m pip install shap
# 6. record + sanity
"$VP" -m pip freeze
"$VP" -m pip check
```

### 1.2 Windows PowerShell form (equivalent; not re-run separately — same executables)

```powershell
py -3.12 -m venv "C:\Users\Bohabara\AppData\Local\Temp\claude\C--Users-Bohabara-Desktop-process-mining\05db42ec-3109-4098-847a-cf389d8c070a\scratchpad\venv312"
$VP = "C:\Users\Bohabara\AppData\Local\Temp\claude\C--Users-Bohabara-Desktop-process-mining\05db42ec-3109-4098-847a-cf389d8c070a\scratchpad\venv312\Scripts\python.exe"
& $VP -m pip install --upgrade pip
& $VP -m pip install numpy "pandas<3" scikit-learn xgboost mlxtend networkx paretoset scipy matplotlib pyyaml
& $VP -m pip install pm4py
& $VP -m pip install shap      # optional
& $VP -m pip freeze
```

Activation scripts exist in `venv312\Scripts\` (`Activate.ps1`, `activate.bat`, `activate`), but calling
`<venv>\Scripts\python.exe -m ...` directly avoids PowerShell execution-policy problems and is what the
README should recommend.

### 1.3 Generic form for the group's README (relative paths, no absolute paths — course rule)

```bash
# Windows (PowerShell or cmd)
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py --config config.yaml
# Linux / macOS
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py --config config.yaml
```
`main.py` / `config.yaml` are placeholders for the group's future entry point [ASSUMPTION].

## 2. `pip freeze` output (venv312, after all installs incl. optional shap)

```
cloudpickle==3.1.2
colorama==0.4.6
contourpy==1.4.0
cvxopt==1.3.3
cycler==0.12.1
fonttools==4.66.1
graphviz==0.21
joblib==1.6.0
kiwisolver==1.5.1
llvmlite==0.50.0
lxml==6.1.3
matplotlib==3.11.2
mlxtend==0.25.0
narwhals==2.26.0
networkx==3.7
numba==0.68.0
numpy==2.5.3
packaging==26.3
pandas==2.3.3
paretoset==1.2.5
pillow==12.3.0
pm4py==2.7.23.8
pyparsing==3.3.3
python-dateutil==2.9.0.post0
pytz==2026.4
PyYAML==6.0.3
scikit-learn==1.9.1
scipy==1.18.1
shap==0.52.0
six==1.17.0
slicer==0.0.8
threadpoolctl==3.7.0
tqdm==4.70.1
tzdata==2026.4
xgboost==3.4.1
```
`pip check`: "No broken requirements found". `tkinter` 8.6 is available in this venv (only the 2023 GUI needs it).
Install was fast (all binary wheels; largest download llvmlite 41.9 MB, pulled in by paretoset→numba).
The finished venv occupies about 719 MB on disk (`du -sh`).

## 3. Smoke test — 2024 repo (PermutationLocationImportance)

Dataset `datasets/BPIC11_f1_trunc36.csv` (24,177 lines incl. header). Constructor call copied from
`CrossValidation_ProcessPermutation.py:38-46`: `DataManager(address, min_prefix_length=2, max_prefix_length=None, L_max_perc=0.8)`.

### 3.1 `smoke_2024.py` (load → Apriori → index encoding → one XGBoost fit)

```
python 3.12.10
numpy 2.5.3 | pandas 2.3.3 | sklearn 1.9.1 | xgboost 3.4.1 | mlxtend 0.25.0

[1] DataManager loaded in 0.24 s
    data shape: (23853, 5)            # columns: case id, activity, timestamp, label, event_nr (tools.py:76)
    #cases: 1130
    #distinct activities: 164
    max event_nr: 36
    L_max (quantile 0.80 of event_nr): 24.0
    label dtype: int64 | value counts (per event row): {0: 12858, 1: 10995}
    label value counts (per case): {0: 676, 1: 454}

[2] frequent_activity_sets(min_support=0.5, top_k=10) in 0.03 s
    10 itemsets returned
    support=0.591 size=2 itemset=['370407', 'ac370000']
    support=0.577 size=2 itemset=['ac370000', 'ac370419']
    support=0.564 size=2 itemset=['ac370000', 'ac370443']
    support=0.562 size=3 itemset=['ac370000', 'ac370419', 'ac370443']
    support=0.562 size=2 itemset=['ac370419', 'ac370443']
    support=0.559 size=2 itemset=['ac370000', 'ac370442']
    support=0.559 size=3 itemset=['370407', 'ac370000', 'ac370419']
    support=0.559 size=2 itemset=['370407', 'ac370419']
    support=0.557 size=2 itemset=['ac370442', 'ac370443']
    support=0.557 size=3 itemset=['ac370000', 'ac370419', 'ac370442']

[2b] frequent_activity_sets(0.5, 10.0) (float top_k as argparse would pass):
    FAILED: TypeError cannot do positional indexing on Index with these indexers [174.0] of type float

[3] index_encoding(dm.data) in 3.14 s
    encoded shape: (1130, 5941) | warnings: 5842 ['PerformanceWarning']
    first 5 columns: ['case:concept:name', 'label', 'e1_330001b', 'e1_339171a', 'e1_339486e']
    dtypes summary: {int64: 3490, bool: 2450, object: 1}

[4] X shape: (1130, 5939) | y len: 1130 | y unique: [0, 1]
    X dtypes: {int64: 3489, bool: 2450}
    XGBClassifier() fit in 2.89 s | train weighted F1 = 0.9920
```
Column count sanity: 36 positions × 164 activities = 5904 one-hot columns; 5939 − 5904 = 35 extra columns are the
padding dummies `e2_0 … e36_0` created by `fill_value=0` in the pivot (tools.py:345-346) [ASSUMPTION from arithmetic;
individual `eN_0` columns were seen in the XGBoost error listing of §3.2]. Activity names are lower-cased and
stripped of spaces/`-`/`_` by `_load_df` (tools.py:39, 48-50), which is why e.g. `AC370000` appears as `ac370000`.

### 3.2 `smoke_2024b.py` — unmodified `itemset_permutation_importance` CRASHES

Fold 0 of `cross_split_test_train(5)` (returns two dicts keyed 0..4, values = pandas Series of case ids;
904 train / 226 test cases per fold), model fitted exactly as `CrossValidation_ProcessPermutation.py:77-88`:

```
[6] fold 0: fit 2.55 s | test weighted F1 = 0.8888 | train weighted F1 = 0.9967
[7] itemsets used: {0: ['ac370000', '370407'], 3: ['ac370443', 'ac370000', 'ac370419']}
    warnings during permutation: {'FutureWarning': 8}
      - FutureWarning : Setting an item of incompatible dtype is deprecated and will raise an error in a
        future version of pandas. Value '[False False ... True ...]' ...
Traceback (most recent call last):
  File ".../tools.py", line 538, in itemset_permutation_importance
    predicted = model.predict(Shuffled_X)
  ...
  File ".../xgboost/data.py", line 360, in _invalid_dataframe_dtype
    raise ValueError(msg)
ValueError: DataFrame.dtypes for data must be int, float, bool or category. ... Invalid columns:
  e1_387042a: object, e1_ac386902: object, e2_0: object, ... (hundreds of columns)
```

Diagnosis (version issue, verified):
- pandas ≥ 2.0 `pd.get_dummies` returns `bool` columns (checked: `pd.get_dummies(...).dtypes → [bool]` in 2.3.3);
  in pandas 1.x it returned `uint8`. So `index_encoding` now yields a mix of `bool` (one-hot) and `int64`
  (the `missing_cols` filled with 0 at tools.py:362) columns, and which activity-position columns are `bool` vs
  `int64` differs between `X` (all cases) and `X_corrupt` (only the shuffled cases, tools.py:529-531).
- tools.py:536 `Shuffled_X[Shuffled_X.index.isin(X_corrupt.index)] = X_corrupt` therefore writes bool values
  into int64 columns (and vice versa) → pandas 2.x `FutureWarning` + upcast to `object` → XGBoost refuses.
- Pinning pandas back is NOT possible on Python 3.12: `pip install --only-binary=:all: "pandas<2"` finds no wheel
  (lowest cp312 wheel is pandas 2.1.1; `pandas==2.0.3` from the 2023 requirements also has none).

### 3.3 `smoke_2024c.py` — same test with the workaround (cast features to int) WORKS

Workaround used (subclass; the group's reimplementation can do the same inline, or pass `dtype=int` to
`pd.get_dummies` at tools.py:353 — the `dtype=int` variant was not executed here [UNVERIFIED]):

```python
class DataManagerIntFeatures(DataManager):
    def index_encoding(self, data):
        enc = super().index_encoding(data)
        feat = [c for c in enc.columns if c not in (self.case_id, self.outcome)]
        enc[feat] = enc[feat].astype(int)
        return enc
```

```
[3'] index_encoding + int cast: 3.66 s, shape (1130, 5941), dtypes {int64: 5940, object: 1}
[6'] fold 0 (904 train / 226 test cases): fit 2.42 s | test weighted F1 = 0.9064
[7'] itemsets used: {0: ['370407', 'ac370000'], 3: ['ac370419', 'ac370443', 'ac370000']}
     itemset_permutation_importance (2 itemsets x 1 repeat) in 13.4 s -> 6.7 s per (itemset,repeat)
     result:
           0         3
     0  0.041058  0.062332
     extrapolation for the repo defaults: 5 folds x 10 itemsets x 10 repeats = 500 iterations -> ~56 min
     warnings during permutation: {}
TOTAL script time 21.7 s
```
The two importance numbers (0.041 for {370407, ac370000}; 0.062 for {ac370000, ac370419, ac370443}) are
single-repeat, single-fold smoke values on the TRAIN fold (baseline = train F1, as in the repo) — not results.

### 3.4 Runtime notes
- One (itemset, repeat) iteration ≈ 6.7 s on BPIC11_f1 with 904 training cases (dominated by the per-case
  pandas filtering loop tools.py:512-526 and one `index_encoding` call ≈ 3 s). Linear extrapolation to the repo
  defaults (5 folds × 10 itemsets × 10 repeats = 500 iterations) ≈ 56 min for f1 on this machine [ASSUMPTION:
  linear scaling; measured on 2 iterations only]. f2 (trunc40) and f4 are larger files, so expect longer [ASSUMPTION].
- Test F1 of fold 0 differed between the two runs (0.8888 vs 0.9064) because `StratifiedKFold(..., shuffle=True)`
  has no `random_state` (tools.py:231) — see problem P3.

## 4. Smoke test — 2023 repo (InteractivePatternDetection / IMPresseD), `smoke_2023.py`

Run with cwd = repo dir and the repo dir first on `sys.path` (IMIPD imports `from tools import ...`, i.e. the
2023 `tools.py`). `GUI_IMPresseD_tool.py` was NOT imported.

```
import pm4py       OK  v=2.7.23.8  (2.68 s)   # prints an AGPL-v3 license banner on import
import paretoset   OK  v=1.2.5  (0.45 s)
import networkx    OK  v=3.7
import scipy       OK  v=1.18.1
import sklearn     OK  v=1.9.1
import matplotlib  OK  v=3.11.2

import IMIPD      OK (2.14 s)
   IMIPD public names: ['Pattern_extension', 'Single_Pattern_Extender', 'Trace_graph_generator',
   'VariantSelection', 'calculate_pairwise_case_distance', 'create_pattern_attributes', 'create_pattern_frame',
   'defining_graph_pos', 'frequency_measuring_patterns', 'plot_dashboard', 'plot_patterns',
   'predictive_measuring_patterns', 'similarity_measuring_patterns']
import Auto_IMPID OK
   has AutoStepWise_PPD: True

paretoset() tiny check -> mask: [False, False, False, True]      # (1,1) dominates (1,3),(2,2),(3,1) under min/min
pm4py tiny check -> EventLog with 2 traces                        # format_dataframe + convert_to_event_log
```
Full pattern discovery was not run (out of scope); whether `AutoStepWise_PPD` runs end-to-end on BPIC11 with
pm4py 2.7.23.8 / networkx 3.7 is therefore [UNVERIFIED].

## 5. Problems found and workarounds

| # | Problem | Evidence | Workaround / decision |
|---|---|---|---|
| P1 | **2024 core function crashes under pandas 2.x** (`itemset_permutation_importance`, tools.py:536→538): bool/int64 column mix from `pd.get_dummies` is upcast to `object`, XGBoost raises `ValueError` | §3.2 traceback; `pd.get_dummies` dtype = bool in pandas 2.3.3 | Cast all feature columns to `int` after `index_encoding` (verified, §3.3) or use `pd.get_dummies(..., dtype=int)` [UNVERIFIED]. Cannot be solved by pinning: no pandas < 2.1.1 wheel for Python 3.12. |
| P2 | **`--top_k` is parsed as `float`** (`CrossValidation_ProcessPermutation.py:27`), and `frequent_activity_sets` does `.head(top_k + n_activities)` (tools.py:168-169) → `TypeError: cannot do positional indexing on Index with these indexers [174.0] of type float`. The repo's main script crashes immediately with default CLI args under pandas 2.3.3. | §3.1 [2b] | Use `type=int` (or `int(top_k)`). Calling the method with an `int` works. |
| P3 | **Folds are not reproducible**: `StratifiedKFold(n_splits=K_fold, shuffle=True)` without `random_state` (tools.py:231). Fold-0 test F1 was 0.8888 in one run and 0.9064 in the next. (`np.random.seed(2023)` at tools.py:501 only seeds the permutation, not the split.) | §3.2 vs §3.3 | Pass `random_state` to the splitter in the reimplementation; the course rubric scores "correctness & reproducibility" (14 pts). |
| P4 | `index_encoding` emits one pandas `PerformanceWarning` per added column (5842 on f1; columns added one at a time at tools.py:352-354 and 362). Harmless but floods stdout. | §3.1 [3] | Build the one-hot frame with a single `pd.concat` / `pd.get_dummies` on the whole frame, or filter the warning. |
| P5 | `prefix_generator` uses `DataFrame.append` (tools.py:92), removed in pandas 2.0 → would crash. Not exercised because `All_prefixes = False` is hard-coded (main:36). | read, not run | Not needed for the project; if prefixes are ever wanted, use `pd.concat`. |
| P6 | `constrain` is accepted by `itemset_permutation_importance` (tools.py:498-499) but never used inside it (lines 500-552); only `trace_permutation_importance` uses it (tools.py:395). `shuffle_sequence` always restricts positions to `Allowed_locations` (tools.py:461-467). | read | Document in the paper that the multi-activity run is always "constrained"; do not expose a dead flag. |
| P7 | **Both repos ship a module named `tools.py`** with different contents. `from tools import DataManager` (2024) vs `from tools import create_embedded_pattern_in_trace, ...` (2023 IMIPD.py:14). Whichever directory is first on `sys.path` wins. | file listings | In the group's package give them distinct module names (e.g. `ppm_tools.py` / `impressed_tools.py`) and import IMIPD functions from there. |
| P8 | 2023 `requirements.txt` is not installable as-is on Python 3.12: it contains `sklearn==0.0` (deprecated meta-package, pip refuses it) [UNVERIFIED that pip refuses on this machine — not attempted], and `pandas==2.0.3` has no cp312 wheel (verified); `xgboost==1.4.2`, `matplotlib==3.6.0`, `PyQt5`, `pyperclip`, `seaborn`, `future`, `Pillow 9.2.0` pins were not tested [UNVERIFIED]. | §3.2 pandas check | Use the fresh pins of §6; the IMIPD/Auto_IMPID imports work with them. |
| P9 | Runtime: ≈ 6.7 s per (itemset, repeat) on f1 → ≈ 56 min for the repo defaults per dataset; the variant doubles the number of runs (Apriori sets + IMPresseD sets) over three datasets. | §3.3 | Plan compute time; make `n_repeats`, `K_fold`, `top_k` YAML-configurable so a short "demo" run exists for the graders. |
| P10 | pm4py prints a license banner (AGPL v3) on every import. pm4py is only required if IMIPD.py code is reused for the variant [ASSUMPTION: the group will reuse `Trace_graph_generator`/`Pattern_extension` etc.]. | §4 | Cosmetic; keep pm4py in requirements. Mention the AGPL license in the technology statement [ASSUMPTION that it is relevant]. |
| P11 | Python 3.14 (the machine default) is unsuitable; pandas 3 breaks `index_encoding`. Relayed from the orchestrator's verified facts; NOT re-tested in this run. | — | Always invoke `py -3.12`. State "Python 3.12" in the README. |
| P12 | `shap` is only imported by `Classical_Permutation.py` (2024 repo); it installed trivially (0.52.0) and is listed as optional. | §2 | Keep optional / commented in requirements. |

## 6. Proposed `requirements.txt` (pinned to the versions that worked, Python 3.12.10, Windows 10 x64)

```
# Python 3.12 (tested with 3.12.10 on Windows 10 x64). Do NOT use Python 3.14 / pandas 3.
numpy==2.5.3
pandas==2.3.3
scipy==1.18.1
scikit-learn==1.9.1
xgboost==3.4.1
mlxtend==0.25.0
networkx==3.7
paretoset==1.2.5
pm4py==2.7.23.8
matplotlib==3.11.2
PyYAML==6.0.3
# optional (only for SHAP-based comparisons, e.g. the 2024 repo's Classical_Permutation.py):
# shap==0.52.0
```
Transitive packages (numba, llvmlite, cvxopt, lxml, graphviz, tqdm, pillow, joblib, …) are resolved automatically;
the full resolved set is the `pip freeze` in §2 if a fully frozen file is preferred.
