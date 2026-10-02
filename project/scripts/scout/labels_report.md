# BPIC11 label definitions (f1–f4): what the labels mean, how 0/1 is encoded, why the files are "trunc36/40/31/40"

Scope: label-definition scout for the four `BPIC11_f*_trunc*.csv` files used by the 2024 repo (Vazifehdoostirani et al.). Primary sources were fetched and quoted directly; the local CSVs were analysed with the default Python 3.14 + pandas 3.0.1 (read-only). Scratch scripts and the extracted paper text are next to this file (`_label_check.py`, `_cut_check.py`, `_code_check.py`, `_confusion.py`, `_tkdd_arxiv.txt`).

Tagging convention: anything not verified in a primary source is marked `[UNVERIFIED]`; anything inferred is marked `[ASSUMPTION]`.

---

## 0. TL;DR (one table)

| file (2024 repo) | benchmark name | LTL rule φ (paper §5.1) | "trace cutting" before (benchmark script) | trunc (Table 6) | # cases (Table 6 / local) | pos-class ratio (Table 6 / local share of label 1) | What label 1 means **in the local CSV** (empirically verified) |
|---|---|---|---|---|---|---|---|
| `BPIC11_f1_trunc36.csv` | bpic2011_1 | F("tumor marker CA-19.9") ∨ F("ca-125 using meia") | `AC379414` (CA-19.9), `378619A` (ca-125) | 36 | 1140 / 1140 | 0.40 / 0.4018 (458 of 1140) | φ **satisfied**: the patient eventually gets a CA-19.9 or CA-125 tumor-marker test (the trace was cut right before that test) |
| `BPIC11_f2_trunc40.csv` | bpic2011_2 | G("CEA - tumor marker using meia" → F("squamous cell carcinoma using eia")) | none (label not known until case ends) | 40 | 1140 / 1140 | 0.78 / 0.7833 (893 of 1140) | φ **satisfied**: every CEA test is eventually followed by a squamous-cell-carcinoma test — including the vacuous case "no CEA test at all" |
| `BPIC11_f3_trunc31.csv` | bpic2011_3 | (¬"histological examination - biopsies nno") U ("squamous cell carcinoma using eia") | `AC356134` (biopsies nno), `376480A` (squamous) | 31 | 1121 / 1121 | 0.23 / 0.2310 (259 of 1121) | φ **satisfied**: a squamous-cell-carcinoma test occurs, and no "histological examination - biopsies nno" happens before it |
| `BPIC11_f4_trunc40.csv` | bpic2011_4 | F("histological examination - big resectiep") | `AC356133` (big resectiep) | 40 | 1140 / 1140 | 0.28 / 0.2798 (319 of 1140) | φ **satisfied**: "histological examination - big resectiep" eventually occurs |

Key caveat (see §2): the paper's text defines the class label as **1 = φ violated**, but the benchmark's preprocessing script **swaps** `deviant`/`regular`, and the local data show **label 1 = φ satisfied**. For the project, trust the data: **label 1 ("deviant", positive) = the LTL rule is satisfied.**

---

## 1. Sources (all fetched 2026-10-01)

- Paper (arXiv v4, 23 Oct 2018, same text as ACM TKDD 13(2) 17:1–17:57, 2019): https://arxiv.org/abs/1707.06766 and PDF https://arxiv.org/pdf/1707.06766 (text extracted with pdftotext to `_tkdd_arxiv.txt`; line numbers below refer to that extraction).
- Benchmark repo root: https://github.com/irhete/predictive-monitoring-benchmark (folders: `bucketers/`, `experiments/`, `preprocessing/`, `transformers/`; README points to the labeled+preprocessed datasets on Google Drive: `https://drive.google.com/open?id=154hcH-HGThlcZJW5zBvCJMZvjOQDsnPR` — **not fetched** (Drive download; not needed because the 2024 repo ships the files)).
- `dataset_confs.py` (note: it lives in `experiments/`, not at the root): https://raw.githubusercontent.com/irhete/predictive-monitoring-benchmark/master/experiments/dataset_confs.py
- Preprocessing script for BPIC11: https://raw.githubusercontent.com/irhete/predictive-monitoring-benchmark/master/preprocessing/preprocess_logs_bpic2011.py
- Truncation rule in code: https://raw.githubusercontent.com/irhete/predictive-monitoring-benchmark/master/experiments/experiments.py and https://raw.githubusercontent.com/irhete/predictive-monitoring-benchmark/master/experiments/DatasetManager.py
- README: https://raw.githubusercontent.com/irhete/predictive-monitoring-benchmark/master/README.md
- Local CSVs: `C:/Users/Bohabara/AppData/Local/Temp/claude/C--Users-Bohabara-Desktop-process-mining/05db42ec-3109-4098-847a-cf389d8c070a/scratchpad/repos/PermutationLocationImportance/datasets/BPIC11_f{1,2,3,4}_trunc{36,40,31,40}.csv`
- Original event log (not fetched; for reference only): BPIC 2011, 4TU link given in the benchmark README: https://data.4tu.nl/repository/uuid:d9769f3d-0ab0-4fb8-803b-0d1120ffcf54

---

## 2. Concepts, ELI5 first

### 2.1 Outcome label (binary outcome of a case)
- ELI5: every patient's hospital story gets a sticker at the end — "yes, X happened" or "no, X didn't" — and the model must guess the sticker while the story is still being written.
- Precise: for each trace σ (one case = one patient's sequence of events), a labeling function assigns a class in {0,1}. In this benchmark the function is "does the trace satisfy a given LTL rule φ or not" (paper §5.1, `_tkdd_arxiv.txt`:759-763).
- Tiny example: rule "eventually a CA-125 test happens". Trace ⟨consult, blood test, CA-125 test, consult⟩ → rule satisfied. Trace ⟨consult, ultrasound⟩ → rule not satisfied.

### 2.2 LTL (Linear Temporal Logic) rule
- ELI5: a sentence about the order of events in a story, built with four words: "next", "always", "eventually", "until".
- Precise (paper Table 5, `_tkdd_arxiv.txt`:767-774): `X φ` — φ holds in the next position; `G φ` — φ holds in all subsequent positions; `F φ` — φ holds eventually (somewhere later); `φ U ψ` — φ holds at least until ψ holds, and ψ must hold now or in the future.
- Tiny example: G(CEA → F(squamous)) reads "always: if a CEA test happens, then later a squamous-cell test happens". A trace ⟨CEA, squamous, CEA⟩ violates it (the second CEA is never followed).

### 2.3 Trace cutting at the "label becomes known" point
- ELI5: if the sticker is decided the moment event X happens, we stop the story right before X; otherwise the model would just learn "I saw X, so the sticker is yes" — cheating.
- Precise: for rules of the form F(a) the class is fixed and irreversible at the first occurrence of a, so the benchmark removes event a and everything after it from the trace (paper, `_tkdd_arxiv.txt`:792-801). For G(a → F(b)) no cutting is possible because a later a can flip the result again (:801-808).
- Tiny example: trace ⟨c1, c2, CA-125, c4⟩ with rule F(CA-125) becomes ⟨c1, c2⟩ with label "satisfied"; a trace without CA-125 stays complete with label "not satisfied".

### 2.4 Prefix truncation ("trunc36")
- ELI5: we only read the first N pages of every story; N is chosen so that 9 out of 10 "positive" stories are already finished by page N, and N is never more than 40.
- Precise: see §5 (code: `min(40, ceil(90th percentile of case length among positive-class cases))`).
- Tiny example: if positive cases have lengths 5, 10, 20, 30, 36, 36, 36, 36, 36, 50 → 90th percentile ≈ 36 → trunc 36 → every case keeps at most its first 36 events.

---

## 3. (a) The exact LTL rules — verbatim from the paper (§5.1 "BPIC2011")

Context sentence (`_tkdd_arxiv.txt`:756-760): the log "contains cases from the Gynaecology department of a Dutch Academic Hospital", and "we use four different labeling functions based on LTL rules [32]".

The rules (pdftotext drops some glyphs — `g`, ligatures, ∨/→/¬ symbols — reconstructed below; the pdftotext raw is in `_tkdd_arxiv.txt`:783-790):

- bpic2011_1: φ = F("tumor marker CA-19.9") ∨ F("ca-125 using meia")
- bpic2011_2: φ = G("CEA - tumor marker using meia" → F("squamous cell carcinoma using eia"))
- bpic2011_3: φ = (¬"histological examination - biopsies nno") U ("squamous cell carcinoma using eia")
- bpic2011_4: φ = F("histological examination - big resectiep")

The paper's own gloss (:792-793): the rule for bpic2011_1 "expresses the rule that at least one of the activities "tumor marker CA-19.9" or "ca-125 using meia" must happen eventually during a case".

Note on the operator symbols: the raw extraction shows a blank between the two F(...) terms of rule 1 and between the G-antecedent and F-consequent of rule 2. ∨ and → are the standard readings and match the paper's prose ("at least one of the activities", "must always be followed by") — `[ASSUMPTION]` only on the glyphs, not on the meaning. The ¬ in rule 3 is rendered as `�` at :789; the benchmark script's cut logic (cut before "biopsies nno" AND before "squamous") and the local data (§6) are consistent with ¬.

Trace cutting per rule (paper :795-808, verbatim fragments):
- rule 1: "all the cases are cut exactly before either of these events happens"
- rule 3: "cut before the occurrence of "histological examination-biopsies nno""; rule 4: "before "histological examination-big resectiep""
- rule 2: "no cutting is performed in the bpic2011_2 dataset" because "another occurrence of "CEA-tumor marker using meia" will cause the φ to be violated again"

Discrepancy worth knowing: for bpic2011_3 the paper text mentions cutting only before "biopsies nno", but the script (below) also cuts before "squamous cell carcinoma using eia" — which is the correct thing to do for an Until rule (the label is fixed at the first of either). The local f3 file contains neither code (§6), so the script, not the sentence, describes the shipped data.

---

## 4. (b) Which class is label 1 — the paper, the benchmark code, the preprocessing swap, and the 2024 repo

### 4.1 Paper definition
`_tkdd_arxiv.txt`:759-763: "we define the class label for a case according to whether an LTL rule φ is violated or satisfied by each trace", with the displayed definition `y(σ) = 1 if φ violated in σ, 0 otherwise`. Also :893: positive ones are "(class label = 1)".
So **the paper says 1 = violated.**

### 4.2 Benchmark config and label→int mapping (verbatim)
`experiments/dataset_confs.py`:200-210:
```
dataset = "bpic2011_f%s"%formula
filename[dataset] = os.path.join(logs_dir, "BPIC11_f%s.csv"%formula)
case_id_col[dataset] = "Case ID"
activity_col[dataset] = "Activity code"
resource_col[dataset] = "Producer code"
timestamp_col[dataset] = "time:timestamp"
label_col[dataset] = "label"
pos_label[dataset] = "deviant"
neg_label[dataset] = "regular"
```
`experiments/DatasetManager.py`:143: `return [1 if label == self.pos_label else 0 for label in y]` → in the benchmark's models, **"deviant" = 1, "regular" = 0**. The processed CSVs themselves carry the strings "deviant"/"regular", not 0/1.

### 4.3 The preprocessing script SWAPS the two strings (verbatim)
`preprocessing/preprocess_logs_bpic2011.py`:14-15 and :63-65:
```
pos_label = "deviant"
neg_label = "regular"
...
data = data.set_value(col=label_col, index=(data[label_col] == pos_label), value="normal")
data = data.set_value(col=label_col, index=(data[label_col] == neg_label), value=pos_label)
data = data.set_value(col=label_col, index=(data[label_col] == "normal"), value=neg_label)
```
Net effect: rows labeled "deviant" in the input (`../labeled_logs_csv/BPIC11_f%s.csv`) become "regular" in the output (`../labeled_logs_csv_processed/`), and vice versa. The input files (on Google Drive) were not fetched, so **which semantics the input strings had is `[UNVERIFIED]`**; what matters is the output, which we can test (§6).

### 4.4 The 2024 repo
- `tools.py`:57-58: `df.loc[df[self.outcome] == "deviant", self.outcome] = 1` / `"regular" → 0` (also `dist_location_calculator.py`:48-49, `Pixel_Flipping_Process.py`:51-52). For BPIC11 this is a no-op because the shipped CSVs already contain integers 0/1 (local check: `label` values are exactly {0,1}, constant within each case for all four files). How/when the 2024 authors converted the strings to 0/1 is `[UNVERIFIED]` (no script for it in the repo).
- The 2024 paper only says: "We used public event logs widely used in the literature, such as bpic2011, bpic2012, and Sepsis event logs with the same labeling strategy as in [18]" (2024 paper §4.1, local txt :328-329).
- Column provenance: the local CSV columns (`Producer code`, `Section`, `Specialism code.1`, `group`, `Number of executions`, `timesincemidnight`, `month`, `weekday`, `hour`, `timesincelastevent`, `timesincecasestart`, `event_nr`, `open_cases`, `Age`, `Diagnosis`, `Treatment code`, `Diagnosis code`, `Specialism code`, `Diagnosis Treatment Combination ID`) are exactly the benchmark's static/dynamic columns, with `Case ID → case:concept:name` and `Activity code → concept:name`. The rare-category value `other` (script :111-114, threshold 10) appears in `concept:name` (e.g. 263 events in f1) and `missing` appears in one column. So the 2024 CSVs are the benchmark's *processed* files plus renaming and 0/1 labels — `[ASSUMPTION]`, but every observable detail matches.

### 4.5 What label 1 means in the shipped data — empirical verdict
Using f2 (uncut, only truncated at 40 events) to reconstruct the rule status of each case, and comparing with the labels in f1/f3/f4 (`_confusion.py`):

f1 — rows: "AC379414 or 378619A occurs within the first 40 events of f2"; cols: local label
```
                 label 0   label 1
occurs = False       682        41
occurs = True          0       417
```
f4 — rows: "AC356133 occurs within first 40 events of f2"
```
                 label 0   label 1
occurs = False       821       293
occurs = True          0        26
```
f3 — rows: which cut activity comes first in f2
```
                                   label 0   label 1
biopsies-nno first (φ violated)        108         0
squamous first (φ satisfied)             0       238
neither within first 40 (unknown)      754        21
```
f2 — rows: rule status using the candidate CEA code `376400` (see §7) within the 40-event window
```
                                                      label 0   label 1
376400 present, no later 376480A (φ violated)             221         9
376400 then later 376480A (φ satisfied)                     6       106
no 376400 at all (φ vacuously satisfied)                   20       778
```
Reading: wherever the rule status is observable inside the 40-event window, **label 1 ⇔ φ satisfied, label 0 ⇔ φ violated**, with zero exceptions for f1/f3/f4. The off-diagonal cells are exactly the cases where the decisive activity lies beyond event 40 of the f2 file (e.g. the 41 f1 cases labeled 1 whose marker test occurs after event 40; the 20 f2 cases with a CEA test after event 40). Additional supporting fact: in f1 all 411 cases that are visibly shorter than their f2 counterpart (i.e. were cut) carry label 1, and all 682 label-0 cases are uncut (`_cut_check.py`).

Therefore the sentence to use in the project: **in BPIC11_f1..f4, label 1 ("deviant", the positive class in the benchmark and in the 2024 code) means the LTL rule is satisfied; label 0 ("regular") means it is violated.** This is the opposite of the paper's displayed definition (1 = violated) and is explained by the swap in §4.3 under the assumption that the Drive input files followed the paper's definition `[ASSUMPTION]`. The paper's Table 6 "pos class ratio" (0.4/0.78/0.23/0.28) equals the local share of label 1, so Table 6 was computed on the processed (swapped) files `[ASSUMPTION]`.

Plain-language label-1 meaning per dataset (for the short paper / poster):
- f1: "the patient eventually receives a tumor-marker test (CA-19.9 or CA-125)" — 40% of cases.
- f2: "no CEA tumor-marker test is left without a later squamous-cell-carcinoma test" (most label-1 cases simply never have a CEA test) — 78%.
- f3: "a squamous-cell-carcinoma test is done before any histological examination of biopsies" — 23%.
- f4: "a histological examination of a large resection specimen ('big resectiep') eventually happens" — 28%.
(Medical readings of the Dutch/English activity names are informal paraphrases `[ASSUMPTION]`; the activity strings themselves are verbatim from the paper and script comments.)

---

## 5. (c) Truncation lengths 36 / 40 / 31 / 40 — two different operations

### 5.1 Operation 1: trace cutting (removes the label-revealing activity and everything after it)
Verbatim from `preprocess_logs_bpic2011.py`:
```
48 def cut_before_activity(group):
49     relevant_activity_idxs = np.where(group[activity_col] == relevant_activity)[0]
50     if len(relevant_activity_idxs) > 0:
51         cut_idx = relevant_activity_idxs[0]
52         return group[:cut_idx]
...
86         relevant_activity = "AC379414" #"tumor marker CA-19.9"
88         relevant_activity = "378619A" #"ca-125 using meia"
92         relevant_activity = "AC356134" #"histological examination - biopsies nno"
94         relevant_activity = "376480A" #"squamous cell carcinoma using eia"
98         relevant_activity = "AC356133" #"histological examination - big resectiep"
```
(lines 86-88 under `if "f1" in filename`, 92-94 under `elif "f3"`, 98 under `elif "f4"`; nothing for f2.) Each cut is applied per case after sorting by timestamp (mergesort, stable).

Side effect that explains "1121 cases" for f3: if the cut activity is the very first event, `group[:0]` is empty and the case disappears. Local check: the 19 case ids present in f2 but absent from f3 all have `AC356134` as their first event in f2 (`_code_check.py`). So 1140 − 19 = 1121 `[ASSUMPTION on mechanism, data-consistent]`. No f1/f4 case starts with its cut activity (0 cases), hence 1140 remain.

### 5.2 Operation 2: prefix-length truncation (the "trunc" number in the filename)
Verbatim from `experiments/experiments.py`:85-92:
```
# determine min and max (truncated) prefix lengths
min_prefix_length = 1
if "traffic_fines" in dataset_name:
    max_prefix_length = 10
elif "bpic2017" in dataset_name:
    max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))
else:
    max_prefix_length = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.90))
```
and `experiments/DatasetManager.py`:120-121:
```
def get_pos_case_length_quantile(self, data, quantile=0.90):
    return int(np.ceil(data[data[self.label_col]==self.pos_label].groupby(self.case_id_col).size().quantile(quantile)))
```
So trunc = min(40, ⌈90th percentile of the case length among **positive-class ("deviant")** cases⌉). For bpic2011 this yields 36 (f1), 40 (f2, capped), 31 (f3), 40 (f4, capped) — Table 6 column "trunc length" (`_tkdd_arxiv.txt`:927-936). The local files have max `event_nr` = 36 / 40 / 31 / 40 and max case length equal to those numbers, i.e. the 2024 authors shipped the logs already cut at the benchmark's max prefix length (how they produced the files is `[UNVERIFIED]`; the trunc values match the paper exactly).

Paper rationale (§5.2.4, `_tkdd_arxiv.txt`:1182-1187, verbatim fragments): "we vary the prefix length from 1 to the point where 90% of the minority class have finished (or until the end of the given trace, if it ends earlier than this point). For computational reasons, we set the upper limit of the prefix lengths to 40, except for the bpic2017 datasets where we further reduced the limit to 20." and "the aim of predictive process monitoring is to predict as early as possible". Nuance: the paper says "minority class", the code uses `pos_label` ("deviant"); for bpic2011_2 the positive class is the majority (78%), so the two descriptions differ there — the file name (40 = cap) is unaffected.

Why truncation is NOT "the point where the label becomes known": that is Operation 1 (trace cutting, §5.1). Operation 2 is a prefix-length cap for earliness and computation, computed from the positive-class length distribution.

Important for the 2024 pipeline: the shipped data are therefore prefixes of at most 36/40/31/40 events, cut *before* the label-revealing activity (f1/f3/f4). The 2024 repo additionally drops cases that contain very rare activities (`tools.py`:60-64, `frq_threshold`) — prior reports state the remaining counts as 1130/1130/1111 for f1/f2/f3 `[UNVERIFIED here — taken from the prior report, not re-checked]`.

---

## 6. (d) Counts and class ratios — paper vs local

Paper Table 6 (`_tkdd_arxiv.txt`:925-936), columns: dataset | # traces | min length | med length | max length | trunc length | # variants (after trunc) | pos class ratio | # event classes | # static attr-s | # dynamic attr-s | # static cat levels | # dynamic cat levels:
```
bpic2011_1 1140 1 25.0 1814 36 815 0.4  193 6 14 961 290
bpic2011_2 1140 1 54.5 1814 40 977 0.78 251 6 14 994 370
bpic2011_3 1121 1 21.0 1368 31 793 0.23 190 6 14 886 283
bpic2011_4 1140 1 44.0 1432 40 977 0.28 231 6 14 993 338
```
(min/med/max length are *before* prefix truncation but after trace cutting; e.g. f3 max 1368 < f1 max 1814 because cutting removed events.) Paper text :890-891: the bpic2011 datasets are "the most heterogenous in terms of case length", "the longest case consists of 1814 events".

Local CSVs (`_label_check.py`), events / cases / distinct `concept:name` values / label-1 cases / share / length min-median-max after truncation:
```
f1_trunc36: 24176 events, 1140 cases, 176 activities, 458 label-1 (0.4018), length 1 / 25.0 / 36
f2_trunc40: 31235 events, 1140 cases, 218 activities, 893 label-1 (0.7833), length 1 / 40.0 / 40
f3_trunc31: 20534 events, 1121 cases, 167 activities, 259 label-1 (0.2310), length 1 / 21.0 / 31
f4_trunc40: 30928 events, 1140 cases, 205 activities, 319 label-1 (0.2798), length 1 / 40.0 / 40
```
Case counts and positive ratios match Table 6 exactly (ratios to two decimals). The activity counts (176/218/167/205) are smaller than Table 6's "# event classes" (193/251/190/231) — expected, because the local files are prefix-truncated and contain the collapsed value `other` `[ASSUMPTION]`. Length distribution by label (local): f1 label-1 median 25 vs label-0 median 13; f4 label-1 median 40 vs label-0 median 23 — i.e. positive cases are the long ones (the decisive activity tends to happen late), which is exactly why the benchmark caps prefixes at the positive-class 90th percentile.

---

## 7. Activity codes: how to recognise the rule activities in the local CSVs

The local `concept:name` column holds the benchmark's "Activity code" values, not names. Two spellings coexist (verified): `AC<digits>` (e.g. `AC410100`, `AC370000`) and bare codes with an optional trailing letter (e.g. `378619A`, `376480A`, `376400`, `370407`), plus the collapsed value `other`. The 2024 paper prints them in lower case (`ac370000`, `ac370419`, `ac379999`; local txt :362, :380, :394).

Code → activity name, from the script comments (verified) and local presence counts (cases containing the code; `_cut_check.py`, `_code_check.py`):

| code | activity (script comment) | used by | f1 | f2 | f3 | f4 |
|---|---|---|---|---|---|---|
| `AC379414` | "tumor marker CA-19.9" | f1 rule | 0 | 18 | 12 | 18 |
| `378619A` | "ca-125 using meia" | f1 rule | 0 | 416 | 266 | 416 |
| `AC356134` | "histological examination - biopsies nno" | f3 rule | 124 | 148 | 0 | 148 |
| `376480A` | "squamous cell carcinoma using eia" | f2 and f3 rules | 267 | 270 | 0 | 270 |
| `AC356133` | "histological examination - big resectiep" | f4 rule | 17 | 26 | 11 | 0 |
| `376400` | **candidate** for "CEA - tumor marker using meia" `[ASSUMPTION]` | f2 rule | — | present in 91.9% of label-0 cases vs 12.9% of label-1 cases | — | — |

Zeros on the diagonal (f1 lacks both f1 codes, f3 lacks both f3 codes, f4 lacks its code) confirm that the local files are the cut versions. The CEA code is not named anywhere in the fetched sources (the script has no comment for it because f2 is not cut); `376400` is inferred from the f2 confusion table in §4.5 (221 of 230 "CEA not followed" cases are label 0; 778 of 798 "no CEA" cases are label 1) and from its numeric neighbourhood to `376480A`. A web search for the code mapping found nothing; the only way to verify is the original BPIC 2011 XES (4TU) — **open question**.

Note for the IMPresseD variant: because the rule activities for f1/f3/f4 were cut out, the model never sees them; any "important" activity found by Apriori or IMPresseD is a proxy that precedes the decisive test. In f2 the decisive activities (`376400`, `376480A`) remain in the data and can legitimately show up as important.

---

## 8. Open-questions register (for the orchestrator)

1. `[UNVERIFIED]` Semantics of "deviant"/"regular" in the benchmark's *input* files on Google Drive (`../labeled_logs_csv/`). Only the output semantics (label 1 = rule satisfied) are verified from data. Resolution: download the Drive folder or ask the supervisor whether "positive = rule satisfied" is the intended reading (the paper's displayed formula says the opposite).
2. `[ASSUMPTION]` `376400` = "CEA - tumor marker using meia". Resolution: check the "Activity code" attribute in the original BPIC 2011 XES (4TU) or the Drive files, which may still carry the activity name column.
3. `[UNVERIFIED]` How the 2024 authors turned the processed benchmark files into the shipped `*_trunc*.csv` (renaming `Case ID`/`Activity code`, 0/1 labels, cutting at the benchmark's max prefix length). No script in the repo; all observable details match the benchmark.
4. `[UNVERIFIED]` Rare-activity filtering counts of the 2024 pipeline (1130/1130/1111) are from a prior report and were not re-checked here.
5. Operator glyphs (∨, →, ¬) in the paper's LTL rules were reconstructed from prose because pdftotext dropped them; meaning is unambiguous from the paper's own explanations and from the script's cut logic.
6. Minor: for bpic2011_3 the paper text mentions only one cut activity; the script cuts before two. The data follow the script.
