# What We Did, How and Why: a Step-by-Step Walkthrough

*Group 11 project on explainable predictive process monitoring: the activity-location importance method of Vazifehdoostirani et al. (2024), with activity sets from Apriori and from IMPresseD. 2 October 2026.*

## How to read this

- **Who it is for.** A group mate who has not followed the work, and anyone who asks us to explain our code.
- **No prior reading needed.** Every word is explained where it first appears. The glossary at the end lists all of them.
- **How it differs from the brief.** `PROJECT_BRIEF_Group11.md` is the short version: the idea, the plan and the questions for the supervisor. This document is the detailed one: what we did, how and why, step by step, with the code. Both use the same words and the same three invented patients.
- **The five parts** follow the order of the work:
  1. Part 1: what we found in the papers and in the authors' code.
  2. Part 2: our engine. It moves activities and measures the score drop.
  3. Part 3: Apriori, the first strategy that picks the activity sets. Also the existence importance.
  4. Part 4: IMPresseD, the second strategy.
  5. Part 5: how to run everything, and what is still open.

  After Part 5 come "Results so far" and the glossary.
- **Three marks.** In the steps we mark three things, where they apply:
  - **Verified:** we checked it, and we say how.
  - **Working choice:** a setting we use for now. The supervisor may still change it.
  - **Not finished:** work that is still open.
- **Three short tags** stand right after a fact, where needed:
  - *(Not re-checked)*: we took the fact from our reference guide or from a report. We did not look at the source again.
  - *(Our reasoning)*: the reason is ours. No paper states it.
  - *(Working choice)*: as above.
- **A short path.** Some sections start with "*Detail: skip on first read.*" You can skip them and still follow the rest.
- **Code.** We point to code by file, class and function name, never by line number. The code will still change.
- **Experiments.** Our experiments have numbers: C1 to C12. (C18 belongs to C10, and there is a pilot run.) Section 5.4 lists them all. Some were still running when we wrote this. We say so where it matters, and we do not guess their results.
- **Q numbers.** Q1 to Q14 are our follow-up questions for the supervisor. Section 5.6 explains the twelve that are still open choices.

## Contents

- [The whole thing in one picture](#the-whole-thing-in-one-picture)
- [Part 1 — Where we started and what we found](#part-1--where-we-started-and-what-we-found)
  - [1.1 Read the assignment and the two papers](#11-read-the-assignment-and-the-two-papers)
  - [1.2 Get the authors' code and the data](#12-get-the-authors-code-and-the-data)
  - [1.3 Set up Python 3.12 with pinned package versions](#13-set-up-python-312-with-pinned-package-versions)
  - [1.4 Run the original 2024 code for the first time](#14-run-the-original-2024-code-for-the-first-time)
  - [1.5 Read the original code closely](#15-read-the-original-code-closely)
    - [1.5a Scores come from the training fold](#15a-scores-come-from-the-training-fold)
    - [1.5b One working copy, so moves pile up](#15b-one-working-copy-so-moves-pile-up)
    - [1.5c The position bookkeeping in shuffle_sequence](#15c-the-position-bookkeeping-in-shuffle_sequence)
    - [1.5d The folds have no seed](#15d-the-folds-have-no-seed)
    - [1.5e --top_k is read as a decimal number](#15e---top_k-is-read-as-a-decimal-number)
    - [1.5f min_support 0.5 gives only 5 sets on f3](#15f-min_support-05-gives-only-5-sets-on-f3)
  - [1.6 Find out what the labels mean](#16-find-out-what-the-labels-mean)
  - [1.7 Notice that timestamps only show the day](#17-notice-that-timestamps-only-show-the-day)
  - [1.8 The decisions that follow](#18-the-decisions-that-follow)
  - [1.9 Summary of Part 1: from finding to decision](#19-summary-of-part-1-from-finding-to-decision)
- [Part 2 — The engine: from a log to location importance](#part-2--the-engine-from-a-log-to-location-importance)
  - [2.1 EventLog: from a CSV file to traces](#21-eventlog-from-a-csv-file-to-traces)
  - [2.2 IndexEncoder: one 0/1 row per patient](#22-indexencoder-one-01-row-per-patient)
  - [2.3 Folds and the model](#23-folds-and-the-model)
    - [2.3a make_folds: five splits with a seed](#23a-make_folds-five-splits-with-a-seed)
    - [2.3b Training the model and joining the pieces](#23b-training-the-model-and-joining-the-pieces)
  - [2.4 LocationPermutationImportance: the heart](#24-locationpermutationimportance-the-heart)
    - [2.4a Finding the set in a trace](#24a-finding-the-set-in-a-trace)
    - [2.4b Drawing new positions](#24b-drawing-new-positions)
    - [2.4c How the other events slide](#24c-how-the-other-events-slide)
    - [2.4d Faithful mode: exactly the original, quirks included](#24d-faithful-mode-exactly-the-original-quirks-included)
    - [2.4e Fixed mode: what the paper describes](#24e-fixed-mode-what-the-paper-describes)
    - [2.4f The score and the result table](#24f-the-score-and-the-result-table)
  - [2.5 How we know it is right](#25-how-we-know-it-is-right)
  - [2.6 Speed](#26-speed)
- [Part 3 — Choosing sets with Apriori, and the existence importance](#part-3--choosing-sets-with-apriori-and-the-existence-importance)
  - [3.1 What Apriori does, and how the original code uses it](#31-what-apriori-does-and-how-the-original-code-uses-it)
  - [3.2 Our AprioriSelector: a top 10 for each length](#32-our-aprioriselector-a-top-10-for-each-length)
  - [3.3 Experiment C7: which min_support?](#33-experiment-c7-which-min_support)
  - [3.4 Existence importance](#34-existence-importance)
- [Part 4 — Choosing sets with IMPresseD](#part-4--choosing-sets-with-impressed)
  - [4.1 What IMPresseD does, in plain words](#41-what-impressed-does-in-plain-words)
  - [4.2 Running the original tool without its window (experiment C3)](#42-running-the-original-tool-without-its-window-experiment-c3)
  - [4.3 Case distance, and a surprise in a library (experiment C12)](#43-case-distance-and-a-surprise-in-a-library-experiment-c12)
  - [4.4 A check of the Pareto library (experiment C11)](#44-a-check-of-the-pareto-library-experiment-c11)
  - [4.5 Our small IMPresseD: the five stages, and why chains](#45-our-small-impressed-the-five-stages-and-why-chains)
  - [4.6 Stage 1: every activity becomes a pattern](#46-stage-1-every-activity-becomes-a-pattern)
  - [4.7 Stage 2: counting per patient, and the three scores](#47-stage-2-counting-per-patient-and-the-three-scores)
  - [4.8 Stages 3 and 4: the front, and how a pattern grows](#48-stages-3-and-4-the-front-and-how-a-pattern-grows)
  - [4.9 Stage 5: from patterns to activity sets, ten per length](#49-stage-5-from-patterns-to-activity-sets-ten-per-length)
  - [4.10 How we know it matches the original (experiment C4)](#410-how-we-know-it-matches-the-original-experiment-c4)
  - [4.11 What the sensitivity runs showed (experiments C5 and C6)](#411-what-the-sensitivity-runs-showed-experiments-c5-and-c6)
  - [4.12 Working choices, and what is not finished](#412-working-choices-and-what-is-not-finished)
- [Part 5 — Running it yourself, and what comes next](#part-5--running-it-yourself-and-what-comes-next)
  - [5.1 A map of the repository](#51-a-map-of-the-repository)
  - [5.2 Setup](#52-setup)
  - [5.3 How to run the tests and the experiments](#53-how-to-run-the-tests-and-the-experiments)
  - [5.4 The experiments at a glance](#54-the-experiments-at-a-glance)
  - [5.5 What is not done yet, and in which order we do it](#55-what-is-not-done-yet-and-in-which-order-we-do-it)
  - [5.6 Open choices that wait for the supervisor](#56-open-choices-that-wait-for-the-supervisor)
- [Results so far](#results-so-far)
- [Glossary](#glossary)
  - [Basic words](#basic-words)
  - [Extra words of this document](#extra-words-of-this-document)

## The whole thing in one picture

A hospital log lists, for each patient, the steps of the visit in order. This list is the **trace**. A type of step is an **activity**, and its place in the trace is its **position**. A **model** reads the trace and predicts a yes/no answer, the **label**.

The 2024 method asks: how much does the model's answer depend on where an activity is? It tests small groups of activities, the **activity sets**. Two tools can pick these sets. Apriori picks sets that are frequent. IMPresseD picks patterns that are linked to the label.

**The project's question.** When IMPresseD picks the sets instead of Apriori, does the importance of the activities and of their positions change, and how?

**The data.** We use three logs: f1, f2 and f3. They are three versions of one hospital log (BPIC11). Only the yes/no question differs. Each file keeps only the first events of every trace: at most 36, 40 and 31. These numbers are in the file names, for example `BPIC11_f1_trunc36.csv`. In the real logs the activities have code names such as `ac370442`, not words like BLOOD.

**The picture.** It shows the road from the data to the answer. Each box names the file in `project/experiments/` that does the work, and the section that explains it. Some words in the picture are new (fold, seed, XGBoost, faithful mode, fixed mode, location importance, existence importance). Parts 1 to 3 explain them where they are used.

```
event log: one CSV file per log (BPIC11 f1, f2, f3)
   |
[1] clean and sort ........................ engine.py, class EventLog           (section 2.1)
   |
   |----------------------------------------.
   |                                        |
[2] one table row per patient             [4] pick the activity sets (the "strategy")
    engine.py, class IndexEncoder             (a) Apriori:   apriori_selector.py,
    (section 2.2)                                            class AprioriSelector
[3] five folds with a seed,                                  (section 3.2)
    one XGBoost model per fold                (b) IMPresseD: impressed_chain.py,
    engine.py, function make_folds                           class ImpressedChainSelector
    (the calling script trains the model)                    (uses case_distance.py)
    (section 2.3)                                            (sections 4.5 to 4.9)
   |                                        |
   |----------------------------------------'
   |
[5] move each set to other positions ...... engine.py, class LocationPermutationImportance
[6] ask the model again, measure the drop   (same class; faithful mode or fixed mode)
   |          = location importance         (section 2.4)
   |
[7] classic shuffle test on "is the set there?" columns
   |          = existence importance ...... existence_importance.py, class ExistenceImportance
   |                                        (section 3.4)
   |
comparison of the two strategies .......... not written yet (section 5.5)
```

Only box 4 differs between the two strategies. Everything else is the same code. So a difference in the results comes from the sets, not from the code.

**What we know now, in three sentences.**

1. The authors' code runs after one small fix, but it does not do exactly what the paper says.
2. We rebuilt its core. Our version gives the same numbers as the original, about 128 times faster, and it can also follow the paper exactly.
3. The final comparison of the two strategies is not done yet.

The details are in "Results so far", after Part 5. Four experiments had no result at the time of writing: C1, C8, C9 and the pilot.

## Part 1 — Where we started and what we found

Part 1 covers what we did before we wrote our own code: seven steps (1.1 to 1.7) and the decisions that follow (1.8). Everything we built later answers something we found here. Each step says what we did, how, why, and what we found. All code shown here is the authors' original code, except our fix in section 1.4.

### 1.1 Read the assignment and the two papers

**What we did.** We read the assignment and its two papers: the 2024 paper by Vazifehdoostirani and colleagues, and their 2023 paper about IMPresseD.

**Why.** We must reproduce the 2024 method, so we must know exactly what it claims.

**What we found.** The 2024 paper is about a model that predicts a yes/no answer for a case. In our data a case is a patient. The data call this answer the **label**. The model reads the patient's **trace**: the ordered list of steps of the visit. A type of step is an **activity**. One step of one patient is an **event**.

We use three invented patients in all examples:

```
position:  1    2        3        4        5      label
Anna:      REG  BLOOD    CONSULT  SCAN     CALL   yes (1)
Bea:       REG  CONSULT  SCAN     BLOOD    CALL   no  (0)
Cleo:      REG  BLOOD    SCAN     CONSULT  CALL   yes (1)
```

REG = registration, BLOOD = blood test, CONSULT = talk with the doctor, SCAN = ultrasound scan, CALL = follow-up call.

Anna and Cleo have BLOOD at position 2 and the label yes. Bea has BLOOD at position 4 and the label no. So in this toy log the position of BLOOD decides the label. Remember this; we use it again in Parts 2, 3 and 4.

The paper asks: does it matter that BLOOD is at position 2? It moves activities to other positions and checks how much the model's score drops. (The score tells how good the model's answers are; section 1.5a.) That drop is the **location importance**. ("Location" is the paper's word for position.)

The paper tests **activity sets**: small groups of activities, such as {BLOOD, SCAN}. It picks them with Apriori, an algorithm that finds activities which often occur in the same trace. A way to pick the sets is a **strategy**.

Our variant uses a second strategy: IMPresseD. It finds small ordered patterns, such as "BLOOD then CONSULT". We keep only the activities and ignore the order. We do this for **length** 1, 2 and 3. Length is the number of different activities in a set. *(Working choice.)*

The one question the project must answer: when IMPresseD picks the sets instead of Apriori, does the importance of the activities and of their positions change, and how?

The paper leaves out details that code needs. One example: it takes the ten most frequent sets with more than one activity, but it does not say how frequent a set must be. So we turned to the authors' code.

### 1.2 Get the authors' code and the data

**What we did.** We downloaded the authors' two repositories (folders with code): `PermutationLocationImportance` (the 2024 method and the data) and `InteractivePatternDetection` (IMPresseD).

**How.** Both are in `project/external/`. We never edit a file there. When the original needs a change, we write it in a file of our own (section 1.4). We have not compared our local copy with the online repositories since the download.

**Why untouched.** Later we must show that our code gives the same numbers as the original. That only works if the original stays as it was. *(Our reasoning.)*

**What we found.**

- **The data** are in `external/PermutationLocationImportance/datasets/`: `BPIC11_f1_trunc36.csv`, `BPIC11_f2_trunc40.csv` and `BPIC11_f3_trunc31.csv`. We call these files the logs f1, f2 and f3. One row is one event: one activity of one patient.
- **Three versions of one log.** The three logs come from the same hospital log. What differs is the yes/no question, and with it the place where the traces are cut (section 1.6).
- **The number in the file name** is the largest number of events that a trace keeps: 36, 40 and 31.
- **Code names.** The activities have code names such as `ac370419`. We have no full table that turns them into medical names (Q14).
- **The core of the 2024 code** is the script `CrossValidation_ProcessPermutation.py` and the class `DataManager` in `tools.py`.
- **No package versions.** The 2024 repository has no `requirements.txt`, the file that lists package versions.
- **Two files with one name.** Both repositories contain a `tools.py`. Never mix them in one folder.

### 1.3 Set up Python 3.12 with pinned package versions

**What we did.** We made a virtual environment (a private package folder) with Python 3.12. We wrote the exact version of every package into `project/requirements.txt`. This is called pinning.

**How.** Two commands do it: `py -3.12 -m venv` makes the environment, and `pip install -r requirements.txt` installs the pinned versions. Section 5.2 has the exact commands, also for a Mac.

The main pins are pandas 2.3.3 (the library for tables), xgboost 3.4.1 (the prediction model) and mlxtend 0.25.0 (it contains Apriori).

**Why.**

- **Python 3.14 does not work.** The laptop's default Python is 3.14, with pandas 3.0.1. There the original function `index_encoding` stops. (It turns the traces into a table for the model; this is called encoding.) Its `pd.pivot_table` call fills empty positions with the number 0 in a text column, and pandas 3 refuses that. We tested this: loading works, then the encoding stops with an error (a `TypeError`).
- **There are two crashes with different causes.** With pandas 3 the encoding itself stops (the bullet above). With pandas 2 the encoding works, but a later step stops (section 1.4). Only pandas 1 would have neither crash, and pandas 1 cannot be installed on Python 3.12. The oldest pandas with a ready-made package for Python 3.12 is 2.1.1. So we take pandas 2 and repair the second crash ourselves. *(Not re-checked: the facts about pandas 1 come from our setup report, `scripts/scout/env_report.md`. We could not test pandas 1 ourselves.)*
- **Same numbers for everyone.** With pinned versions, every group member gets the same numbers. *(Our reasoning.)*

**What we found.** The install worked on our Windows laptop: a fresh install with the pinned versions, written down in the setup report. We have not repeated that install since. **Not finished:** it is not tested on a Mac.

### 1.4 Run the original 2024 code for the first time

**What we did.** We ran the original pipeline on f1, one piece at a time, to see exactly which piece fails.

**How.** Three small test scripts in `project/scripts/scout/` (`smoke_2024.py`, `smoke_2024b.py`, `smoke_2024c.py`) call `DataManager` the way the authors' main script does. Before you run them, change the absolute path in their `REPO` line.

**What runs.** Loading and cleaning, Apriori (10 sets), encoding and model training.

- Cleaning removes every patient who has an activity that occurs only once in the log. On f1 this leaves 1,130 of 1,140 patients, with 164 activities.
- The encoding makes one column for every pair of position and activity. On f1 that is 36 positions × 164 activities = 5,904 columns. 35 more columns mark positions that are empty because a trace is shorter. Together: 5,939 columns.
- Anna's row would have `e2_BLOOD = 1`. This means: event number 2 is BLOOD.

**What crashes.** The main function, `itemset_permutation_importance`. ("Itemset" is the code's word for activity set.) XGBoost stops with `DataFrame.dtypes for data must be int, float, bool or category`.

**The cause.** These lines of `DataManager.index_encoding` (original `tools.py`) build the columns of the table:

```python
        for col in indexed_col:
            one_hot = pd.get_dummies(encoded_data[col], prefix=col)
            encoded_data[one_hot.columns] = one_hot
```

and, a few lines later:

```python
        encoded_data[missing_cols] = 0
```

In the order the lines run:

1. `indexed_col` holds the position names `e1`, `e2`, and so on.
2. `pd.get_dummies` makes the columns for the pairs that occur in the data. In pandas 2 these columns hold true/false values.
3. The last line adds the pairs that never occur (`missing_cols`) as the number 0.

On f1 this gives 2,450 true/false columns next to 3,489 number columns.

Later, `itemset_permutation_importance` encodes the changed patients again and writes their rows over the old ones:

```python
                Shuffled_X = X.copy()
                Shuffled_X[Shuffled_X.index.isin(X_corrupt.index)] = X_corrupt

                predicted = model.predict(Shuffled_X)
```

`X` is the table of all patients that are scored. `X_corrupt` is the new table of only the changed patients.

Remember: a column is true/false when its position-activity pair occurs in the table, and a number when it does not. The smaller table has fewer patients and changed traces, so different pairs occur in it. The same column can therefore be true/false in `X` and a number in `X_corrupt`. The second code line copies one table into the other, and this mixes the two kinds in one column. pandas 2 then makes it a general "object" column, and XGBoost refuses that.

**The fix.** It is in our file `scripts/scout/smoke_2024c.py`. It turns every column into whole numbers:

```python
class DataManagerIntFeatures(DataManager):
    """Workaround: pandas>=2.0 pd.get_dummies returns bool; cast all feature columns to int (pandas 1.x behaviour)."""

    def index_encoding(self, data):
        enc = super().index_encoding(data)
        feat = [c for c in enc.columns if c not in (self.case_id, self.outcome)]
        enc[feat] = enc[feat].astype(int)
        return enc
```

In the order the lines run:

1. The subclass is `DataManager` with one function replaced.
2. It runs the original encoding first (`super().index_encoding`).
3. `self.case_id` and `self.outcome` are the names of the patient-id column and the label column. So `feat` lists all columns except these two.
4. The `astype(int)` line is the fix: it turns these columns into whole numbers.

Our later tests use the same subclass (in `experiments/engine_reference.py`, function `make_original_manager`).

**What we found.** With the fix, the function runs. **Verified:** by running `smoke_2024c.py`.

A **round** is: move one set once, in every trace of the fold that contains it, and score once. (A fold is a part of the patients; section 1.5a explains it.)

Our first rough timing was about 6.7 seconds per round on f1. Experiment C2 later timed 6.78 seconds on a quiet machine. We use 6.78 from here on. On a busy machine it is slower (section 2.6).

With the original settings, f1 needs 500 rounds (5 folds × 10 sets × 10 repeats). 500 × 6.78 seconds is about 57 minutes. We calculated this number; we did not run all 500 rounds. The runtime test on all three logs (experiment C1) was still running at the time of writing.

### 1.5 Read the original code closely

**What we did.** We read `tools.py` and the main script line by line and compared them with the paper.

**Why.** To reproduce means to do what the authors did. Only the code shows that fully.

**What we found.** Six behaviours differ from the paper or can block a run.

#### 1.5a Scores come from the training fold

The code splits the patients into five **folds** (parts). Five times, the model learns from four of them. The code calls these four together the training fold. The fifth is **held-out**: the model has not seen it. The score is **weighted F1**, a number from 0 to 1; higher is better.

This piece of `CrossValidation_ProcessPermutation.py` runs once per fold, right after the model is trained. It decides which patients the importance is measured on:

```python
        print("f1-score test: %.3f" % f1_score(test_y, predicted, average='weighted'))

        if Multi_activity:
            permutation_importance_train_all[i] = data_manager.itemset_permutation_importance(model, train_x, train_y,
                                                                                              train_list[i],
                                                                                              candidate_itemsets,
                                                                                              constrain=constrain)
```

In the order the lines run:

1. The first line prints the held-out score. `test_y` holds the real labels of the held-out patients, and `predicted` holds the model's answers for them. Nothing else uses this score.
2. `Multi_activity` is a switch of the script. It is on by default. It means: test sets with more than one activity. (When it is off, the script tests single activities with another routine; section 2.4f.)
3. The other lines call the importance function. They give it `train_x` and `train_y`: the table and the labels of the training patients. `train_list[i]` holds their ids for fold number `i`. `candidate_itemsets` holds the activity sets from Apriori. (`constrain` is passed on, but `itemset_permutation_importance` never reads it.)

So the importance is measured on the training patients only.

- **How we noticed.** We read which data the call passes.
- **How we measured.** In one fold of f1 the model scored 0.9967 on its training patients and 0.8888 on held-out patients. (One run of `smoke_2024b.py`. Its folds are random, see section 1.5d, so another run gives slightly different numbers.)
- **Why it matters.** The model knows its training patients almost by heart. A drop from 0.99 there does not show what the model uses for new patients.

#### 1.5b One working copy, so moves pile up

This is the start of `itemset_permutation_importance` in `tools.py`. It makes a working copy of the data and opens the loops:

```python
        sub_data = self.data[self.data[self.case_id].isin(case_list)]
        permutation_importance = []
        for itemset in frequent_itemsets:
            for r in range(n_repeats):
                shuffled_cases = []
                for case in case_list:
                    trace_case = sub_data[sub_data[self.case_id] == case].sort_values(by=['event_nr'])
                    trace = trace_case[self.activity].tolist()
```

and, further down in the same loop:

```python
                        shuffled_trace = self.shuffle_sequence(trace, set_of_items)
                        sub_data.loc[sub_data[self.case_id] == case, self.activity] = shuffled_trace
```

In the order the lines run:

1. `sub_data` is the working copy of the training patients' events. (`case_list` holds their ids.) It is made once per call, so once per fold.
2. Then come the loops over the sets (`frequent_itemsets`) and over the 10 repeats (`n_repeats`).
3. Inside, the code takes the events of one patient (`trace_case`) and reads the trace as a list of activities (`trace`).
4. `shuffle_sequence` moves the set (`set_of_items`) inside the trace.
5. The last line writes the changed trace back into `sub_data`.

Nothing resets the copy inside a fold.

Take Anna. The test of {BLOOD, SCAN} may leave her trace as REG CONSULT BLOOD CALL SCAN. The next repeat starts from this changed trace. So does the next set.

- **How we noticed.** We looked for the line that resets the copy. There is none.
- **How we measured.** We counted on the cleaned f1 log, with the ten Apriori sets in the code's order. 632 of the 652 patients who contain the second set also contain the first: 96.9%. For the third set it is 635 of 637 (99.7%), and from the fourth on 100%. These patients' traces were already moved when their set's turn comes.
- **Why it matters.** From the second set on, the drop comes from several sets together.

#### 1.5c The position bookkeeping in `shuffle_sequence`

`shuffle_sequence` moves one set inside one trace. The paper gives two rules for this:

1. **Only seen positions.** An activity may only go to a position where it appears somewhere in the log. (`Allowed_locations` holds these positions.)
2. **Keep the inner order.** The activities of the set keep their order among themselves.

For each activity of the set, the code draws a new position, `random_index`. These lines then do the move:

```python
                current_act_index = locations[IDX[subset.index(act)]]
                shuffled_sequence.insert(random_index - 1, shuffled_sequence.pop(current_act_index))
                locations[current_act_index] = min(random_index - 1, len(sequence) - 1)
                for loc in locations:
                    if current_act_index < loc < random_index - 1:
                        locations[loc] -= 1
```

The names first. `sequence` is the trace before the move, and `shuffled_sequence` is the copy that changes. `IDX` holds the places of the set's activities in the trace, and `subset` holds these activities in trace order. `locations` is a notebook: where is each element of the trace now? Python counts from 0, so position 2 is index 1.

In the order the lines run:

1. The first line looks in the notebook `locations`: where is this activity now? The answer is `current_act_index`.
2. The second line takes the activity out at that place (`pop`) and puts it back at the drawn place (`insert`). `random_index - 1` is the drawn position as a Python index.
3. The third line writes the activity's new place into the notebook.
4. The last three lines should correct the notes of the other elements. When an activity moves to the right, every element between its old and its new place slides one step to the left.
5. The error: the test `current_act_index < loc < random_index - 1` uses `<` on both sides. So the element that sat exactly on the new place is not corrected. Its note is now wrong by one.

Take Anna and {BLOOD, SCAN}. Suppose blood tests were seen at positions 2, 3, 4 and scans at 3, 4, 5.

```
start:              REG  BLOOD    CONSULT  SCAN   CALL
BLOOD drawn to 4:   REG  CONSULT  SCAN     BLOOD  CALL
SCAN drawn to 5:    REG  CONSULT  SCAN     CALL   BLOOD
```

After the first move SCAN is at index 2. The notebook still says 3, where BLOOD is now. So the second move takes BLOOD again. In the end SCAN comes before BLOOD, and BLOOD is at position 5, where no blood test was ever seen. Both rules are broken. With other draws (BLOOD to 3, SCAN to 5) the result is correct: REG CONSULT BLOOD CALL SCAN.

- **How we noticed.** We followed a small example by hand. Then we ran Anna's example through the original function with these draws forced. It returned exactly the two results above.
- **How we measured.** `scripts/scout/_fc_C.py` takes the cleaned f1 log. It keeps the 612 traces in which the set {ac370419, ac370442} occurs exactly once. It moves the set once in each trace (seed 2023). The order was reversed in 21 of them (3.4%). In 19 (3.1%) an activity landed on a never-seen position.
- **What we did about it.** Our faithful mode copies this behaviour on purpose (section 2.4d). Our fixed mode repairs it (sections 2.4b and 2.4c).

#### 1.5d The folds have no seed

A **seed** is a starting number for random steps. The same seed always gives the same random steps again. This piece of `DataManager.cross_split_test_train` makes the folds:

```python
        if regression:
            skf = KFold(n_splits=K_fold, shuffle=True)
        else:
            skf = StratifiedKFold(n_splits=K_fold, shuffle=True)
```

In the order the lines run:

1. `K_fold` is the number of folds (5).
2. `regression` is false for our yes/no labels, so the last line runs.
3. `StratifiedKFold` makes the folds, with about the same share of "yes" patients in each.
4. `shuffle=True` makes the split random. No seed is given. (The option for it is `random_state`.)

Without `random_state`, scikit-learn takes its random numbers from numpy's global generator, and the main script does not seed that generator before it makes the folds. The seed 2023 inside `itemset_permutation_importance` only makes the random moves repeatable. The main script makes the folds before that seed is set.

- **How we noticed.** `smoke_2024b.py` makes the folds without any seed and printed 0.8888 for the first fold. We also called the function twice in one run: the two splits were different. (`smoke_2024c.py` printed 0.9064, but it calls `np.random.seed(0)` just before the split. Without `random_state`, scikit-learn uses numpy's global generator, so that split is repeatable. It is the same split as our fold seed 0.)
- **Why it matters.** Nobody could repeat the numbers of a run.

#### 1.5e `--top_k` is read as a decimal number

*Detail: skip on first read.*

This option of the main script sets how many sets to keep:

```python
parser.add_argument('--top_k', default=10, type=float)
```

and this piece of `DataManager.frequent_activity_sets` uses it:

```python
        frequent_itemsets = frequent_itemsets.sort_values(['support'], ascending=False).head(
            top_k + len(self.data[self.activity].unique()))
```

In the order things happen:

1. `top_k` is the number of sets to keep. `type=float` turns a typed `--top_k 10` into the decimal number 10.0.
2. `.head(n)` keeps the first `n` rows of a table. `len(self.data[self.activity].unique())` is the number of different activities.
3. On f1, `.head(...)` then gets 174.0. That is 10.0 plus 164 activities.
4. pandas cannot keep "174.0 rows", so it stops with an error (a `TypeError`).

Without the option, the default stays the whole number 10 and this step works. (The script then still stops at the crash of section 1.4.) We tested both cases. So we never pass `--top_k` to the original. Section 3.1 walks through the whole function.

#### 1.5f `min_support` 0.5 gives only 5 sets on f3

**Support** is the share of patients whose trace contains the whole set. `min_support` is the lowest support that Apriori keeps. The default is 0.5. This piece of `frequent_activity_sets` picks the final sets:

```python
        frequent_itemsets['item_size'] = frequent_itemsets.itemsets.apply(lambda x: len(list(x)))
        frequent_itemsets = frequent_itemsets[frequent_itemsets['item_size'] > 1]

        selected_itemsets = frequent_itemsets.sort_values(['support'], ascending=False).head(top_k)
```

In the order the lines run:

1. `item_size` is the number of activities in a set.
2. The second line drops the sets with one activity (`item_size > 1`).
3. The last line keeps the `top_k` most frequent of the rest. If fewer exist, it returns fewer.

The main script only prints how many sets it found; it does not stop.

- **What we found.** At 0.5 we get 10 sets on f1 and on f2, but only 5 on f3. The paper's Figure 5 (for f3) shows 10 sets, and three of them start with `ac370442`. *(Not re-checked: we took this from our reference guide, not from the figure itself.)* On f3, every set with `ac370442` has support below 0.5. At 0.49 the function returns 10 sets, three of them with `ac370442`.
- **What it means.** The authors probably used a lower value for f3. *(Our reasoning; the paper does not say.)* Experiment C7 (section 3.3) later ran 0.49, 0.48 and 0.47 and got the same ten sets each time. From the patient counts we can calculate that every value at or below 0.494 must give the same ten sets. We ran only these three values. So the paper's figure fits many values, and it cannot tell us which one the authors used. We ask the supervisor.

### 1.6 Find out what the labels mean

**What we did.** We found out what the label means in our data.

**How.** The 2024 paper takes its labels from a benchmark paper by Teinemaa and colleagues. We read that paper and its preparation script. Then we checked the data (`scripts/scout/_confusion.py` and `_cut_check.py`; report: `scripts/scout/labels_report.md`).

**Why.** We cannot read any result if we do not know what label 1 is.

**What we found.**

- **Each label comes from a rule about medical tests.**
  - f1: at some point the patient gets a CA-19.9 or CA-125 tumour-marker test (a lab test for signs of cancer).
  - f2: every CEA tumour-marker test is later followed by a squamous-cell-carcinoma test (another lab test). Patients without a CEA test also satisfy the rule.
  - f3: a squamous-cell-carcinoma test happens, and no biopsy examination comes before it.
- **In our data, label 1 means the rule is satisfied.** The benchmark paper writes "1 = rule violated", but its script swaps the two label names. *(Not re-checked: from our label report; we did not read the benchmark paper and its script again.)* What we ran again is the data side, below.
- **There are two kinds of cutting.**
  - First, every log keeps only the first events of a trace: at most 36, 40 and 31 events. This is the number in the file name.
  - Second, in f1 and f3 each trace also stops just before the test that decides the label. Otherwise the model would simply read the answer.
  - f2 does not have this second cut. The report gives the reason: in f2 a later test can still change the answer.

**How we checked, on the raw files.** These are the files as downloaded, before any cleaning (1,140 patients in f1 and f2, 1,121 in f3).

1. f1 and f3 contain no event of the tests in their own rule. So the cut is real.
2. f2 comes from the same hospital log and still shows these tests. So we can look up the f1 tests in f2.
3. 417 patients of f1 have such a test in f2. All 417 have label 1 in f1. This confirms: label 1 means the rule is satisfied.
4. Of the other 723 patients, 682 have label 0 and 41 have label 1. For these 41 the test must come later than the 40 events that f2 keeps.

So we write "rule satisfied", never "positive". And in f1 and f3, an important activity is a hint that comes before the test, not the test itself.

### 1.7 Notice that timestamps only show the day

**What we did.** We looked at the timestamps of the three logs. A timestamp is the date and time that the file gives for an event.

**Why.** The method is about position. So we must know what the order of the events rests on.

**How.** `scripts/scout/_fc_B_data.py` counts the pairs of neighbouring events in a trace that have the same timestamp. (This script also holds an absolute path that you must change first.)

**What we found.** Every timestamp has the time 23:00, so it only shows the day. In the raw files of f1, f2 and f3, 88.8%, 87.1% and 88.5% of the pairs share a timestamp. (On the cleaned logs the shares are 88.8%, 87.1% and 88.4%; experiment C5 measured both.) So for two activities on the same day, we cannot know which came first.

The original 2024 code does not use the timestamp to order the events. The lines are in `DataManager._load_df`. The timestamp lines are commented out (switched off with `#`). The events are sorted by `event_nr`, the event number in the file:

```python
        # df[self.time_col] = df[self.time_col].str.replace("/", "-")
        # df[self.time_col] = pd.to_datetime(df[self.time_col],
        #                                    dayfirst=True).map(lambda x: x.strftime("%Y.%m.%d %H:%M:%S"))

        df.sort_values([self.case_id, 'event_nr'], ascending=[True, True], inplace=True)
```

`self.time_col` is the name of the timestamp column. The last line sorts by patient (`self.case_id`) and then by event number.

**What it means for "position".** Position is the event number that the file gives. Within one day we cannot check whether that order is the real order. We keep it, because the original code uses it, and we state it as a limit.

**What it means for IMPresseD.** The original IMPresseD code can treat events with the same timestamp as "parallel": no order between them. Experiment C5 later measured this on our logs: about 94% of all events would be marked as parallel. The patterns that the original would build from such parallel blocks are mostly too large to give sets of 1 to 3 activities. Section 4.5 has the numbers.

### 1.8 The decisions that follow

- **Our own code, not a patched script.** Most findings sit in the core of the method: which patients are scored, the working copy, the move function. A patch would be a rewrite anyway. And the original must stay untouched as our reference. Our prototype is `experiments/engine.py`; Part 2 explains it.
- **Two modes.**
  - **Faithful mode** behaves like the original, errors included. *Why:* it proves that we reproduced the original (section 2.5).
  - **Fixed mode** does what the paper describes: a clean start for every repeat, a repaired move function, and scoring on the held-out fold by default. *Why:* it is the method as the paper states it.
  - We build both, because we do not know yet which one the supervisor wants. The direct comparison (experiment C8) was still running at the time of writing.
- **Chain traces for IMPresseD.** We read each trace as a chain: one event after another in `event_nr` order, nothing parallel. This is the same order the 2024 method uses as position (section 4.5).
- **Open questions are settings, not hard-wired.** Every open question is a value that we can change without rewriting code. Example: `mode` switches between faithful and fixed. If the supervisor decides differently, we change the value and run again. Section 5.6 has the full table of settings.

### 1.9 Summary of Part 1: from finding to decision

| Finding | Decision |
|---|---|
| Python 3.14 with pandas 3 breaks the original | Python 3.12, pinned `requirements.txt` |
| True/false next to number columns crash XGBoost | Our encoding always writes whole numbers |
| Scores come from the training fold | Setting `score_on`; held-out by default |
| One working copy, moves pile up | Fixed mode: every repeat starts from the original traces |
| Order rule sometimes broken | Fixed mode: repaired move function |
| Folds have no seed | Our folds always have a seed (`make_folds`) |
| `--top_k` is read as a decimal number | We never pass it to the original |
| `min_support` 0.5 gives only 5 sets on f3 | `min_support` is a setting; we ask for the authors' value |
| Label 1 = rule satisfied; deciding tests cut out | We write "rule satisfied"; important activities are hints |
| Timestamps only show the day | Position = `event_nr`; chain traces for IMPresseD |
| The original is our reference | `external/` stays untouched |

## Part 2 — The engine: from a log to location importance

Part 1 ended with a decision: we write the core of the method ourselves (section 1.8). Part 2 is a guided tour of that code. It is one file, `project/experiments/engine.py`; we call it the engine. It turns a log into location importance numbers. We follow the data, block by block. The engine comes before the two strategies, because both strategies hand their sets to it.

**Why we wrote our own engine.** In short: the authors' code crashes with today's libraries. It needs 6.78 seconds per round (section 1.4). And it differs from the paper in four ways (sections 1.5a to 1.5d). So we rebuilt the core. The engine never imports the authors' code. Only the test and timing scripts of the engine load the authors' code. They do this through the helper file `engine_reference.py`, to compare the two.

The road through the engine, with the box numbers of the first picture:

```
CSV file
  |  EventLog                        traces, labels, allowed positions    box 1, section 2.1
  |  IndexEncoder                    one 0/1 row per patient              box 2, section 2.2
  |  make_folds                      five splits: training / held-out     box 3, section 2.3a
  |  (the calling script trains one XGBoost model per split)              box 3, section 2.3b
  |  LocationPermutationImportance   move the sets, measure the drop      boxes 5 and 6, section 2.4
  v
result table: one row per activity set, fold and repeat
```

**Words in the code.** Some words differ between this text and the code. The code says `itemset` for activity set, `case` for patient, `location` for position, `size` for length, and `test` for the held-out fold. "Permute" and "shuffle" in the code mean: move. Also: positions count from 1 (Anna's BLOOD is at position 2), but Python lists count from 0 (BLOOD has index 1).

### 2.1 `EventLog`: from a CSV file to traces

*Box 1 in the picture.*

**What it is for.** A log file is a CSV: a text table with one event per row. `EventLog` reads it and keeps three things that every later step uses: the trace of every patient, the label of every patient, and the allowed positions. The **allowed positions** of an activity are the positions where it appears somewhere in the log.

**Why it exists.** Every later step needs the traces as simple Python lists. And our numbers can only match the original's if we clean the log in exactly the same way.

This piece of `EventLog._load_frame` cleans and sorts the table:

```python
        frame[CASE_COL] = frame[CASE_COL].astype(str)
        names = frame[ACTIVITY_COL].str.lower()
        for char in (" ", "-", "_"):
            names = names.str.replace(char, "", regex=False)
        frame[ACTIVITY_COL] = names
        frame = frame.sort_values([CASE_COL, POSITION_COL])
        frame[LABEL_COL] = frame[LABEL_COL].replace({"deviant": 1, "regular": 0})

        # Drop every case that contains an activity with < frq_threshold events.
        event_counts = frame.groupby(ACTIVITY_COL)[CASE_COL].count()
        rare = event_counts[event_counts < frq_threshold].index
        bad_cases = frame.loc[frame[ACTIVITY_COL].isin(rare), CASE_COL].unique()
        frame = frame[~frame[CASE_COL].isin(bad_cases)]
```

In the order the code runs:

1. **Name clean-up.** Patient ids become text. Activity names become lower case, and spaces, `-` and `_` are removed. So `AC370000` becomes `ac370000`. *Why:* the original does this, and one spelling per activity avoids silent mismatches. Names from another tool must first pass through the helper `normalise_activity`.
2. **Sorting by event number.** `POSITION_COL` is the column `event_nr`: the position of an event in its trace. We sort by patient, then by this number. *Why:* the method is about positions, so the order must be right. The original sorts in the same way. We could not sort by timestamp instead, because the timestamps only show the day (section 1.7).
3. **Label words.** `replace` turns the words `deviant` and `regular` into 1 and 0. Our files already hold 0 and 1, so nothing changes.
4. **Removing patients with a once-only activity.** `event_counts` counts the rows per activity. `rare` holds the activities with fewer than `frq_threshold` rows; the default is 2. `bad_cases` are the patients who have such an activity. The last line drops them. *Why:* the original has this filter, so we keep it. A model cannot learn from an activity it sees once. *(Our reasoning; the paper does not mention the filter.)* On f1 it removes 10 of 1,140 patients (our own count on the raw file).

A note on step 2: in our three files the events of every patient are already in this order. The sort only changes the order of the patients: their ids are now sorted as text.

Next, the function checks that every patient's event numbers are 1, 2, 3, ... without gaps. If not, it stops with an error. *Why:* the engine treats "event number" and "place in the list" as the same thing.

Then `EventLog.__init__` stores three dictionaries. (A dictionary is a look-up table: a key goes in, a value comes out.)

- `traces` maps a patient to the list of activities.
- `labels` maps a patient to 0 or 1.
- `allowed_locations` maps an activity to the sorted list of its positions in the log.

**The Anna example.** We wrote Anna, Bea and Cleo into a small CSV file and loaded it:

```
traces             Anna: reg blood consult scan call
                   Bea:  reg consult scan blood call
                   Cleo: reg blood scan consult call
labels             Anna 1, Bea 0, Cleo 1
allowed_locations  reg [1]  blood [2, 4]  consult [2, 3, 4]  scan [3, 4]  call [5]
```

BLOOD may only go to position 2 or 4, because this log shows it nowhere else. (The loader made the names lower case. Below we keep writing BLOOD.)

**The counts we check.** After cleaning, the logs must have these sizes:

| Log | Patients | Activities | Longest trace |
|---|---|---|---|
| f1 | 1,130 | 164 | 36 |
| f2 | 1,130 | 207 | 40 |
| f3 | 1,111 | 156 | 31 |

The check `check_event_log` in `test_engine.py` stops if the patients or activities differ. *Why:* these counts are a fingerprint of the cleaning. If the filter or the file changes, every later number changes too. We want a loud stop, not quietly different results.

### 2.2 `IndexEncoder`: one 0/1 row per patient

*Box 2 in the picture.*

**What it is for.** The model needs a table with the same columns for every patient. `IndexEncoder` makes one column for every pair of position and activity. The column `e2_blood` means: event number 2 is BLOOD.

**Anna's row.** The toy log has 5 positions and 5 activities, so 25 columns:

```
e1_reg=1  e2_blood=1  e3_consult=1  e4_scan=1  e5_call=1     (the other 20 columns = 0)
```

Real traces differ in length. So there are extra columns such as `e4_0`, which means "this trace has no fourth event". These are **padding columns**. f1 gets 36 × 164 + 35 = 5,939 columns, f2 gets 8,319 and f3 gets 4,866.

**Why whole numbers instead of true/false.** The original's table mixes true/false columns with number columns, and XGBoost then stops (section 1.4). Our encoder writes small whole numbers (`dtype=np.int8`) in every column. So this crash cannot happen.

**How it works.** `transform(traces)` makes a table full of zeros and calls `write_rows`. This is the core of `IndexEncoder.write_rows`. It writes the ones into the rows:

```python
        positions = np.concatenate([self._index[:n] for n in lengths])
        matrix[rows] = 0
        matrix[np.repeat(rows, lengths), self._column[positions, codes]] = 1
        padded = (self._index >= lengths[:, None]) & (self._index < pad_until)
        padded &= self._pad_column >= 0
        row_pos, index_pos = np.nonzero(padded)
        matrix[rows[row_pos], self._pad_column[index_pos]] = 1
```

The names first. `matrix` is the table, and `rows` says which of its rows to overwrite. Before these lines, the function has turned every activity into a number (`codes`) and noted the length of every trace (`lengths`). `self._index` is simply the list 0, 1, 2, ... up to the longest trace.

In the order the lines run:

1. `positions` lists, for every event, its index in its trace.
2. `matrix[rows] = 0` clears the rows.
3. The third line sets all the ones at once. `self._column` is a look-up table, built once: position and activity code go in, the column number comes out.
4. `padded` marks the empty positions of every trace: the positions after its last event.
5. `self._pad_column` gives the column number of each padding column. The fifth line keeps only positions that have such a column.
6. The last two lines write a 1 into these padding columns.

Example for the padding: a trace with 3 events in a log whose longest trace has 5 events. Its row gets `e4_0 = 1` and `e5_0 = 1`. `pad_until` is the longest trace of the log. (Faithful mode uses a smaller value; section 2.4d says why.)

**How the model sees a move.** Suppose BLOOD moves from position 2 to position 3 in Anna's trace. Then `e2_blood` becomes 0 and `e3_blood` becomes 1 in her row. CONSULT slides to position 2, so `e3_consult` becomes 0 and `e2_consult` becomes 1. The model was trained on the old row. If its answer for the new row is different, the position mattered. This is why every column is a pair of position and activity.

**Why this makes the engine fast.** After a move, only some traces have changed. We encode only those traces, and we ask the model only about those rows. All other patients keep their old prediction. The table is filled with a few array operations, without pandas. Encoding the whole f1 log takes the original 3.80 seconds and our encoder 0.004 seconds.

One detail: the original adds the never-used columns in an order that changes each time Python starts. We sort them, so runs are repeatable. The model stays the same (check `same_model_f1`, section 2.5).

### 2.3 Folds and the model

*Box 3 in the picture.*

#### 2.3a `make_folds`: five splits with a seed

This function splits the patients into five folds. It is the body of `make_folds(labels, k=5, seed=0)`:

```python
    cases = np.array(list(labels))
    outcome = np.array([labels[case] for case in cases])
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return [
        (cases[train].tolist(), cases[test].tolist())
        for train, test in splitter.split(cases, outcome)
    ]
```

`labels` is the dictionary from section 2.1. In the order the code runs:

1. `cases` lists the patient ids. `outcome` lists their labels in the same order.
2. `StratifiedKFold` is the splitter of the scikit-learn library. "Stratified" means: every fold has about the same share of label-1 patients as the whole log. `random_state=seed` makes the split repeatable.
3. `splitter.split` gives, `k` times, the row numbers of the training patients and of the held-out patients. The last lines turn these row numbers back into patient ids.
4. The function returns `k` pairs. On f1 every pair has 904 training and 226 held-out patients.

**Why the seed matters.** The original has no seed here. Two calls of the original split function in one run gave two different splits (section 1.5d). So nobody could repeat its numbers. With `random_state=seed`, the same seed always gives the same folds, and another seed gives other folds. **Verified:** the independent verification (section 2.5) confirmed both.

This fold seed is separate from the seed for the random moves in section 2.4.

#### 2.3b Training the model and joining the pieces

**What XGBoost is.** XGBoost is the model that the paper uses. It builds many small decision trees. A decision tree is a chain of yes/no questions, such as "is BLOOD at position 2?".

**Who trains it.** The engine does not train the model. The script that calls the engine does. It creates `XGBClassifier()` with its default settings, as the original main script does. It trains the model on the table rows of the training patients. This happens once per fold and never again. All later moves only ask the model for new answers.

**The pieces joined.** The small demo at the end of `engine.py` (function `main`) shows the whole chain in run order. First the log, the encoder and one fold:

```python
    log = EventLog(DATASETS[args.dataset])
    encoder = IndexEncoder(log.traces.values(), log.activities)
    train_cases, test_cases = make_folds(log.labels, k=5, seed=args.seed)[args.fold]
```

Then the model and three example sets:

```python
    model = XGBClassifier()
    model.fit(encoder.transform(log.traces[c] for c in train_cases),
              [log.labels[c] for c in train_cases])
    top = most_frequent_activities(log, 3)
    itemsets = [top[:1], top[:2], top[:3]]
```

Then the engine:

```python
    engine = LocationPermutationImportance(
        mode=args.mode, score_on=args.score_on, n_repeats=args.repeats,
        random_state=args.seed, allowed_from=args.allowed_from, draw=args.draw)
    table = engine.compute(model, log, encoder, train_cases, test_cases,
                           itemsets, fold=args.fold)
```

In the order the lines run:

1. `EventLog` reads and cleans the log (section 2.1). `DATASETS` maps the names f1, f2 and f3 to their files. `args` holds the options typed on the command line.
2. `IndexEncoder` gets all traces and all activity names. It sets up the columns (section 2.2).
3. `make_folds` gives five splits, and `[args.fold]` takes one of them (section 2.3a).
4. `XGBClassifier()` creates the model. `model.fit` trains it on the table rows of the training patients and on their labels.
5. `most_frequent_activities` is a small helper of the demo. `itemsets` then holds three sets: the most frequent activity, the two most frequent, and the three most frequent.
6. `LocationPermutationImportance(...)` creates the engine with its settings (section 2.4).
7. `engine.compute(...)` moves every set in the fold and returns the result table (section 2.4f).

**How a selector's sets reach the engine.** In the demo the sets come from a helper. In a real run a selector delivers them. Both selectors return the same simple thing: `{1: [...], 2: [...], 3: [...]}`, a list of sets for each length (sections 3.2 and 4.9). The calling script passes one of these lists as `itemsets`. **Not finished:** so far only the timing script `c1_engine_grid.py` does this for both strategies. It takes the Apriori sets from `AprioriSelector` and the IMPresseD sets from the result file of experiment C4. The final main file is not written yet (section 5.5).

### 2.4 `LocationPermutationImportance`: the heart

*Boxes 5 and 6 in the picture.*

We create this class with a few settings. Then we call its method `compute` once per fold. It takes the trained model (section 2.3b), the log, the encoder, the patients of the fold, the activity sets and the fold number.

Two settings matter most here. `mode` is `"fixed"` or `"faithful"`; it decides how the sets are moved. `score_on` is `"test"` or `"train"`; it decides which patients we move and score: the held-out or the training patients. Section 5.6 has the full table of settings.

"Faithful mode" in our plans means `mode="faithful"` with `score_on="train"`. "Fixed mode" means `mode="fixed"` with `score_on="test"`.

#### 2.4a Finding the set in a trace

This function finds the places where a set occurs in a trace. It is the body of `find_occurrences(trace, itemset)`:

```python
    positions = [
        [index for index, act in enumerate(trace) if act == wanted]
        for wanted in set(itemset)
    ]
    return [tuple(sorted(match)) for match in zip(*positions)]
```

In the order the code runs:

1. For every activity of the set, the inner list collects the indexes where it occurs.
2. `zip` then pairs the first BLOOD with the first SCAN, the second with the second, and so on.
3. Each such match is an **occurrence**: one place where the whole set is found in the trace.

Anna with {BLOOD, SCAN} gives `[(1, 3)]`: indexes 1 and 3, so positions 2 and 4. The trace `REG BLOOD SCAN BLOOD SCAN BLOOD` gives `[(1, 2), (3, 4)]`. The third BLOOD has no partner and is not moved.

A trace is only moved if it contains the whole set and is longer than the set. *Why:* the original applies the same two tests, and we copy them. In a trace that is exactly as long as the set, nothing can move: every position is taken, and the inner order must stay. *(Our reasoning for this last sentence.)*

#### 2.4b Drawing new positions

This helper draws the new positions for one occurrence. It is `_draw_sequential`, which `permute_trace_fixed` uses:

```python
    upper_bounds = []
    bound = length + 1
    for pool in reversed(pools):
        below = [loc for loc in pool if loc < bound]
        if not below:
            return None
        bound = below[-1]
        upper_bounds.append(bound)
    upper_bounds.reverse()

    drawn, previous = [], 0
    for pool, upper in zip(pools, upper_bounds):
        window = [loc for loc in pool if previous < loc <= upper]
        previous = window[rng.integers(len(window))]
        drawn.append(previous)
    return drawn
```

The names first. `pools` holds one **pool** per activity of the occurrence, in trace order. A pool is the list of the activity's allowed positions. Positions beyond the end of the trace are already removed. So are positions taken by another moved event. `length` is the trace length, and `rng` is the random number generator.

Take {BLOOD, SCAN} in Anna's trace. Suppose BLOOD has the pool [2, 3, 4] and SCAN has [3, 4, 5]. These pools are assumed, as in section 1.5c. They are larger than the pools of the three-patient log in section 2.1 (there BLOOD has [2, 4] and SCAN has [3, 4]).

1. The first loop walks backwards. It finds the latest position each activity may take so that the later ones still fit. SCAN: 5. BLOOD: the largest position below 5, so 4.
2. The second loop draws. BLOOD is drawn from {2, 3, 4}; say 3. SCAN is drawn from the positions after 3 and up to 5: {4, 5}; say 5.

This enforces the two rules of section 1.5c. **Only seen positions:** we draw only from the pools. **Keep the inner order:** `previous < loc` puts every activity after the one before it. If no valid placement is left, the function returns `None`. The trace then stays unchanged, and the result table counts it.

**A known weak spot.** A trace can hold the set several times.

- The occurrences are placed one after the other. An early draw can take the only position that a later occurrence needs. Then the function gives up and the trace stays unchanged, although another draw would have worked.
- On a toy trace `a b a b` this happened in 1,010 of 4,000 draws (second verification script, section 2.5).
- On real f1 and f3 traces it never happened: 0 of 10,975 draws. That check covers sets of two and three activities, built from the 25 most frequent activities, with pools from the whole log.
- **Not finished:** we have not checked f2, or pools from the training patients only.

#### 2.4c How the other events slide

After the draw, the trace is built again from scratch. This is the end of `permute_trace_fixed`:

```python
    moved = set(taken.values())
    others = (act for index, act in enumerate(trace) if index not in moved)
    new_trace = [
        trace[taken[loc]] if loc in taken else next(others)
        for loc in range(1, length + 1)
    ]
    return new_trace, placements
```

In the order the code runs:

1. `taken` maps each drawn position to the old index of the event that goes there. `moved` is the set of these old indexes.
2. `others` delivers the remaining events, one by one, in their old order.
3. The code builds the new trace position by position. A drawn position gets its moved event. Every other position gets the next of the remaining events.

```
before:  REG  BLOOD    CONSULT  SCAN  CALL
after:   REG  CONSULT  BLOOD    CALL  SCAN      (BLOOD 2 -> 3, SCAN 4 -> 5)
```

**An honest limit.** The events that slide are not bound by the first rule. Here CALL lands on position 4. Suppose the log shows CALL only at position 5, as our three-patient log does. Then CALL now sits on a never-seen position.

The paper names this risk among its limitations. *(Not re-checked: from our reference guide.)* Fixed mode counts how often it happens: see the column `n_other_events_unobserved` in section 2.4f. In the full run with the original Apriori sets (held-out patients), 3.5% (f1), 4.3% (f2) and 3.7% (f3) of the events that slid landed on a never-seen position.

#### 2.4d Faithful mode: exactly the original, quirks included

This is the loop of faithful mode. It moves every set in every repeat, on one working copy. From `_run_faithful`:

```python
        rng = np.random.RandomState(self.random_state)  # = np.random.seed(...)
        working = [list(trace) for trace in traces]  # tools.py:507, never reset
        for itemset_id, items in itemsets.items():
            for repeat in range(self.n_repeats):
                rows, n_changed = [], 0
                for row, trace in enumerate(working):
                    if not _is_eligible(trace, items):
                        continue
                    working[row] = shuffle_sequence_faithful(trace, items, allowed, rng)
                    n_changed += working[row] != trace
                    rows.append(row)
```

In the order the code runs:

1. `rng` is one stream of random numbers. Its seed is `self.random_state`: 2023, as in the original. Every draw uses it. The comment says: this is the same random stream as the original's `np.random.seed`.
2. `working` is one copy of the traces, made once before both loops. Its comment names the matching place in the authors' file `tools.py`.
3. The two loops run over the sets and over the repeats. (`self.n_repeats` is the number of repeats, 10 by default.)
4. `_is_eligible` is the check of section 2.4a: whole set present, trace longer than the set.
5. Every such trace is replaced by its shuffled version. `rows` notes its row number. `n_changed` counts the traces that really look different after the shuffle.

Nothing resets `working`. So repeat 2 moves an already moved trace, and the second set starts from what the first set left behind. This is "moves pile up" (section 1.5b).

`shuffle_sequence_faithful` is a line-by-line copy of the original move. It moves the events one after the other and keeps notes on where each event now sits. These notes can go wrong. Section 1.5c shows the original lines.

**The Anna example.** Section 1.5c follows this on Anna's trace, with the same pools as in section 2.4b. In short: BLOOD is drawn to position 4. The note for SCAN then points at BLOOD, so BLOOD is moved a second time. The result is `REG CONSULT SCAN CALL BLOOD`: SCAN ends before BLOOD, and BLOOD sits on position 5, outside its pool. Our copy behaves in the same way. We ran this example with 400 seeds; 135 of them gave exactly this broken result.

Two more quirks are copied. A drawn position beyond the end of a short trace puts the event at the end. And the padding columns are only set up to the longest moved trace (the `pad_until` of section 2.2).

**Why keep the quirks?** With them we can prove that we reproduced the original, number for number (section 2.5). After that proof, any gap between faithful and fixed results comes from the repairs, not from a slip in our code.

#### 2.4e Fixed mode: what the paper describes

`_run_fixed` differs in three ways.

1. **A clean start for every repeat.** `permute_trace_fixed` never changes its input. Every repeat reads the original traces again.
2. **A repaired order rule.** All moved events are placed at once in the final trace (sections 2.4b and 2.4c). No notes can go wrong.
3. **Its own random generator for every fold, set and repeat:**

```python
                rng = np.random.default_rng(
                    [self.random_state, fold, label_hash, repeat]
                )
```

`label_hash` is a number made from the set's name. So the draws for a set do not depend on which other sets we test, or in which order. This helps our comparison: with the same folds and the same seed, a set that both strategies pick gets the same draws in both runs.

| Aspect | Faithful | Fixed |
|---|---|---|
| Data copy | one working copy, moves pile up | the original traces for every repeat |
| Order rule | the original's notes, can fail | always holds |
| New positions | may lie beyond the trace end | inside the trace, one event per position |
| Random numbers | one stream, seed 2023 | one generator per fold, set and repeat |
| Usual scoring | training patients | held-out patients (training patients also possible) |

Fixed mode has two more settings. *(Working choice for both.)*

- **`allowed_from`: where the pools come from.** The default (as in the original) uses all patients of the log. The other value uses the training patients only. *Why this could matter:* the pools from the whole log contain positions that occur only in held-out patients. So information about held-out patients enters the move step. On f1 this is true for 6 to 10% of the position-activity pairs per fold. We measured no difference in the importance.
- **`draw`: how the new positions are drawn.** The default (as in the original) is the step-by-step draw of section 2.4b. The other value makes every valid placement equally likely. It lowers the training-fold importance by 9–14% and changes the ranking on f2.

#### 2.4f The score and the result table

This piece turns the moved rows into scores. From `_score`:

```python
        baseline = weighted_f1(y_true, base_pred)
        records = []
        for itemset_id, items, repeat, rows, new_rows, counts in run:
            # Unchanged rows keep their baseline prediction, so only the
            # re-encoded rows need the model.
            pred = base_pred.copy()
            if len(rows):
                pred[rows] = model.predict(new_rows)
            permuted = weighted_f1(y_true, pred)
```

In the order the code runs:

1. `base_pred` holds the model's answers before any move, and `y_true` holds the real labels. `baseline` is the score before any move.
2. `run` delivers the rounds from section 2.4d or 2.4e, one by one.
3. For each round we copy the answers. We replace only the answers of the changed traces (`rows`) with the model's answers on their new table rows (`new_rows`).
4. Then we compute the weighted F1 again (`permuted`).

The model is never trained again.

**An example.** We made one tiny run: f1, the first fold of fold seed 0, two sets, two repeats, fixed mode on held-out patients. Some of its columns:

```
itemset             repeat  baseline  permuted  importance  n_traces_with_itemset  n_traces_changed
370407, ac370000         0    0.9064    0.8931      0.0132                    130               130
370407, ac370000         1    0.9064    0.9066     -0.0002                    130               129
ac370419, ac370443       0    0.9064    0.9155     -0.0091                    119               119
ac370419, ac370443       1    0.9064    0.9200     -0.0137                    119               119
```

(The numbers are rounded to four places after the subtraction. So the last digit can differ by one from "baseline minus permuted". The baseline 0.9064 is the same number that `smoke_2024c.py` printed in section 1.5d, because that script happened to use the same split.)

| Column | Meaning |
|---|---|
| `itemset_id`, `itemset` | key and name of the activity set |
| `fold`, `repeat` | which fold, which random move |
| `baseline`, `permuted` | weighted F1 before and after the move |
| `importance` | baseline minus permuted: the location importance |
| `n_traces_with_itemset` | scored traces that contain the set and are longer than it |
| `n_traces_changed` | traces that really changed |
| `n_traces_unchanged_infeasible` | traces left unchanged because no valid placement was found |
| `n_other_events_shifted`, `n_other_events_unobserved` | fixed mode only: events that slid, and how many landed on a never-seen position |

A negative importance means the score went up a little after the move. We read that as chance: "no measurable importance".

**From the table to one number per set.** The table has one row for every set, fold and repeat. With 5 folds and 10 repeats that is 50 importance numbers per set. The original draws them as one box per set in a box plot, and orders the boxes by their mean. (A box plot shows a line at the middle value and a box around the middle half of the numbers.) Our C2 report uses the mean of the 50 numbers, with their spread. **Not finished:** how many fold seeds the final runs average over is not decided yet (section 5.5).

**Single activities.** There is one more method, `compute_single_activities`. The original moves single activities with a different routine (`trace_permutation_importance`; the main script calls it when `Multi_activity` is off). That routine differs in three ways:

- It tests every activity of the log, not a selection.
- It moves every occurrence of the activity, one after the other, to a position drawn from the activity's allowed positions inside the trace.
- It gives importance 0 without scoring in two cases: when the activity has fewer than 2 allowed positions, and when fewer than 3 scored traces contain it.

Moves pile up here too, over the repeats and over the activities. Faithful mode copies that routine, so that the numbers match. In fixed mode a single activity is simply a set of length 1, so `compute` handles it.

### 2.5 How we know it is right

**Equal to the original.** `test_engine.py` runs the authors' code and our engine side by side. For the main check, both get the same trained model, the same fold and the same seed. On f1 (904 training patients, baseline 0.9967) and on f3 (888 training patients) we compared 9 importance values each. The largest difference is exactly 0.0; the check demands less than 0.000000000001. For example, the first f1 value is 0.041057613659320591 in both. Exact equality is possible because both make the same prediction for every row, and weighted F1 is computed from counts.

**Fixed mode has no original to compare with.** So we check properties: facts that must hold after every move. The quick run has the 18 checks that do not need the slow original (about 3 minutes). The full run has all 23 (about 38 minutes on a busy machine). Section 5.3 has the commands, and a small demo of the engine.

*Detail: skip the table on first read.* It lists the equality checks first, then the property checks. It shows the most important of the 23 checks, not all of them:

| Check in `test_engine.py` | What it proves |
|---|---|
| `event_log_f1/f2/f3` | same traces, labels and allowed positions as the original loader; the counts of section 2.1 |
| `encoder_f1/f2/f3` | same table as the original: rows, columns, values |
| `same_model_f1` | a model trained on our table equals one trained on the original table: identical probabilities for all 1,130 patients |
| `reencoding_f1` | encoding only the changed traces equals encoding everything (300 shuffled traces) |
| `find_occurrences_f1`, `shuffle_faithful_f1/f3` | our search and our copy of the original move equal the original (6,000 and 9,210 cases) |
| `faithful_equality_f1/f3` | faithful importance equals the original's: largest difference 0.0 |
| `faithful_single_equality_f3` | the same for single activities: 312 values on 200 patients. A separate slow run (`--full-single`) did it for 328 values on the whole f1 training fold |
| `fixed_properties_f1` | 3,000 random moves: inner order kept, only allowed positions inside the trace, same activities, same length |
| `determinism_f1` | same seed, same table; another seed, other draws |
| `no_accumulation_f1` | no piling up: all 1,104 moves start from the original traces; a set tested alone gives the same rows as after two other sets |
| `edge_cases_f1`, `input_validation_f1` | nothing to move gives importance 0, not a crash; wrong input gives a clear error |

`no_accumulation_f1` also shows the piling up in faithful mode. The set {ac370419, ac370443} alone gives 0.0244, 0.0244 and 0.0199. After two other sets it gives 0.0736, 0.0747 and 0.0735.

**Independent verification.** We also checked the engine with two separate scripts (`verify_engine_round1.py`, then `verify_engine.py`, both in `results/experiments/C2/`). They were written from scratch and do not import our tests. They use their own loader, their own simple encoder, other folds and other sets. They checked:

- Faithful mode against the original on f1, f2 and f3: largest difference 0.0. This includes a held-out fold, single activities and rare sets.
- The two rules of fixed mode: 51,027 moved traces in the first script, 71,458 (f1) and 60,857 (f3) in the second. No violation.
- Every score computed again by hand: difference 0.0. When we score held-out patients, the model only sees held-out traces or their moved versions.
- Two fresh Python starts give the same results.

The numbers of the second script come from its result files (`verify_r2_*.json`). **Not finished:** no written report describes that second verification yet. So we do not know whether it raised points beyond what these files show.

The first verification found 11 problems. The worst: fixed mode crashed when a set changed no trace. All 11 were handled: the code errors were repaired, and the open design points were written down and measured. After the repairs we ran the full grid again (script `engine_full_grid.py`). The **full grid** means all combinations of folds, sets and repeats. All 3,750 rows of the earlier result tables (f1, f2 and f3; faithful and fixed mode) came out exactly the same.

**What is not covered.**

- We compared with the original on some folds of every log, not on every fold.
- The single-activity routine was not compared on f2.
- The full run (23 checks) was made before a small edit of one check. Afterwards we ran only the 18 quick checks again. The engine itself did not change.

### 2.6 Speed

*Detail: skip on first read.*

Measured on f1, one fold, the 10 original Apriori sets, 2 repeats, on a quiet machine. *(Not re-checked: the numbers come from the C2 report; we have not measured them again.)*

| Variant | Seconds per round | Speed-up |
|---|---|---|
| original code, training fold | 6.78 | 1x |
| engine, faithful, training fold | 0.053 | 128x |
| engine, fixed, training fold | 0.036 | 189x |
| engine, fixed, held-out fold | 0.012 | 578x |

The held-out fold is faster because it has 226 patients instead of 904. The original's time depends on the machine load: on a busy machine it needed 13.6 to 22.4 seconds per round.

The original setting has 500 rounds for f1 and for f2: 5 folds × 10 sets × 10 repeats. (f3 has only 5 sets at this setting, so 250 rounds.) With the engine the 500 rounds take 26.3 seconds on f1 in faithful mode, and 6.0 seconds in fixed mode on held-out patients. The original would need about 57 minutes. (Calculated, not measured: see section 1.4.)

Three more experiments belong here: C1 (runtime of both codes on all three logs), C8 (faithful against fixed mode) and C9 (several seeds). C1 was still running at the time of writing, and C8 and C9 had no result file yet. So we report nothing for them. A first look at both modes and at ten fold seeds, on the original Apriori sets only, is already in `results/experiments/C2/RESULT.md` (its sections 6.1 and 6.2).

## Part 3 — Choosing sets with Apriori, and the existence importance

The engine of Part 2 needs activity sets to move. Part 3 explains the first strategy that picks them: Apriori, as the original uses it and as we use it. It also explains existence importance: the second number that the paper puts next to location importance. The code is in two files in `experiments/`: `apriori_selector.py` and `existence_importance.py`. Experiments C7 and C10 tested their settings.

### 3.1 What Apriori does, and how the original code uses it

*Box 4(a) in the picture.*

**What it is.** Apriori is an algorithm that lists the activity sets that many patients share. Before it runs, the code makes one **basket** per patient: the different activities in the trace. Order and repeats are thrown away. The key number is **support**: the share of patients whose basket contains the whole set.

```
Anna:  REG  BLOOD    CONSULT  SCAN     CALL     basket: {REG, BLOOD, CONSULT, SCAN, CALL}
Bea:   REG  CONSULT  SCAN     BLOOD    CALL     basket: the same five
Cleo:  REG  BLOOD    SCAN     CONSULT  CALL     basket: the same five
```

All three baskets contain BLOOD and SCAN. So {BLOOD, SCAN} has support 3 of 3 = 100%. If Cleo had no scan, the support would be 2 of 3 = 67%. Apriori keeps every set whose support is at least `min_support`. It never looks at the label or at positions.

**How the original code uses it.** The function is `frequent_activity_sets` in the class `DataManager` (authors' file `tools.py`). After building the baskets, it runs these lines to pick the sets. (Sections 1.5e and 1.5f showed parts of them.)

```python
        frequent_itemsets = apriori(df_2, min_support=min_support, use_colnames=True)
        frequent_itemsets = frequent_itemsets.sort_values(['support'], ascending=False).head(
            top_k + len(self.data[self.activity].unique()))

        frequent_itemsets['item_size'] = frequent_itemsets.itemsets.apply(lambda x: len(list(x)))
        frequent_itemsets = frequent_itemsets[frequent_itemsets['item_size'] > 1]

        selected_itemsets = frequent_itemsets.sort_values(['support'], ascending=False).head(top_k)
```

In the order the code runs:

1. `apriori(...)` finds all frequent sets on the whole log. The function comes from the library mlxtend. `df_2` is the basket table: one row per patient, one True/False column per activity.
2. It sorts by support and keeps the first `top_k` plus "number of activities" rows.
3. `item_size > 1` drops every set with a single activity.
4. It sorts again and keeps the `top_k` best.

Why "plus the number of activities" in step 2? Step 3 throws away the single-activity sets, and there is at most one such set per activity. With this margin, enough larger sets are left after step 3. *(Our reasoning: this is our reading of the code.)*

The main script `CrossValidation_ProcessPermutation.py` has the options `--min_support` and `--top_k`. Their defaults are 0.5 and 10. So the original selection is: one mixed top 10 of sets with two or more activities. If fewer than ten sets exist, the code returns fewer. The main script prints the number, but it does not stop (section 1.5f).

### 3.2 Our `AprioriSelector`: a top 10 for each length

**What we did.** We wrote the class `AprioriSelector`. It returns the 10 most frequent sets for length 1, for length 2 and for length 3.

**Why.** IMPresseD gives us three groups of sets, one per length. The original Apriori step gives no single activities and mixes the longer sets in one list. We must compare the two strategies length by length, so both need the same shape.

**How.** The input `traces` is a dictionary: patient id to list of activity names, for example `{"Anna": ["REG", "BLOOD", "CONSULT", "SCAN", "CALL"], ...}`. The core is the method `full_table`. It lists every frequent set with its patient count:

```python
        onehot = encode_transactions(traces)
        mined = apriori(onehot, min_support=self.min_support, use_colnames=True,
                        max_len=self.max_len)
        n_cases = len(onehot)
        rows = [
            (tuple(sorted(items)), len(items), float(support), int(round(support * n_cases)))
            for items, support in zip(mined["itemsets"], mined["support"])
        ]
        # Sorting on the integer case count makes ties exact (no float noise).
        rows.sort(key=lambda row: (row[1], -row[3], row[0]))
        return pd.DataFrame(rows, columns=TABLE_COLUMNS)
```

In the order the code runs:

1. `encode_transactions` builds the basket table, with the same statements as the original.
2. `apriori(...)` finds the frequent sets. `max_len=self.max_len` (3 by default) tells Apriori to stop at three activities.
3. Each row holds four things: the set with its names sorted, its length, its support, and `count`. `count` is the number of patients that have the set.
4. `rows.sort(...)` orders the rows: by length, then by `count` from high to low, then by name in alphabetical order.

Why the limit `max_len`? The original has no such limit. We need it for values below 0.5: on f1 at `min_support` 0.45 the search without a limit finds 262,215 sets and takes 32 seconds. With the limit, every call in C7 took at most 0.07 seconds. The limit does not change which sets of length 1 to 3 are found.

The method `select_table` then keeps the first `top_k` rows of each length and numbers them:

```python
        table = self.full_table(traces)
        selected = table.groupby("size", sort=True).head(self.top_k).reset_index(drop=True)
        selected["rank"] = selected.groupby("size").cumcount() + 1
        return selected
```

(`size` is the code's word for length.) Finally `select` turns this into the simple thing every strategy returns: `{1: [...], 2: [...], 3: [...]}`, a list of sets per length. A length with fewer than ten sets returns what exists. In the Anna example, length 1 returns only five sets, because there are only five activities.

**How ties are broken, and why it matters.** A **tie** means two sets have the same support. Our rule: the set that comes first in the alphabet wins. We compare whole patient counts, not decimal supports, so equal sets are exactly equal. The Anna example shows the rule at work. All sets there have support 100%, so the alphabet decides everything: length 1 gives BLOOD, CALL, CONSULT, REG, SCAN.

The original code has no clear tie rule. It leaves ties to the sorting routine of pandas.

- On f1, eight sets are clearly in the top 10. Four more sets tie (629 patients each) for the last two places.
- We measured which ones the original code keeps on our laptop. At `min_support` 0.5 it keeps {ac370000, ac370419, ac370442} as one of the two. At 0.55 it keeps {ac370419, ac370442} instead. So a setting that should not matter changes the selection.
- With our rule the same log always gives the same sets. *(Not tested on a second machine or with other library versions.)*
- In our per-length lists the rule decides in one place only: f2, length 2, where three pairs with 648 patients compete for two places.

**`original_top10`: a faithful copy.** The same file has the function `original_top10`. It repeats the original statements of section 3.1, with the mixed top 10 and the pandas tie order. *Why keep it?* Faithful mode must test the same sets as the authors, in the same order. The order matters there because moves pile up from set to set. The function works on plain traces, so it does not need the authors' `tools.py`. So far only the C7 script and `c1_report.py` call it. The engine tests (`test_engine.py`) still call the authors' own function directly.

**How we checked it.** The script `run_c7_apriori_sweep.py` does two checks.

- **The copy of the original** (`original_top10`): compared with the authors' function on 3 logs and 8 `min_support` values. All 24 results were identical: same sets, same order, same supports. (Function `compare_with_original`.)
- **Our selector** (`AprioriSelector`): we counted the patients again with a plain loop, without the Apriori library. All 54 counts (3 logs × 6 values × 3 lengths) were the same. So were the patient counts of all 90 selected sets. (Function `check_against_brute_force`.)
- **Not finished:** these checks are inside the experiment script. A separate test file for the selector does not exist yet.

### 3.3 Experiment C7: which `min_support`?

**What we did and why.** We need ten sets for each length on each log. So we counted how many sets exist for six `min_support` values (0.5, 0.49, 0.45, 0.4, 0.35 and 0.3, with `max_len=3`). The table shows the first four. The cells show the number of sets of length 1 / 2 / 3.

| log | 0.5 | 0.49 | 0.45 | 0.4 |
|---|---|---|---|---|
| f1 | 16 / 94 / 258 | 20 / 121 / 414 | 23 / 168 / 841 | 23 / 211 / 1330 |
| f2 | 23 / 148 / 544 | 25 / 188 / 804 | 26 / 248 / 1390 | 26 / 286 / 2036 |
| f3 | 5 / 4 / 1 | 6 / 9 / 7 | 17 / 84 / 217 | 20 / 130 / 572 |

We ran it from the project folder with `.venv/Scripts/python.exe experiments/run_c7_apriori_sweep.py`. The final run took 98.6 seconds. Nothing in it is random.

**The f3 problem.** At 0.5, f3 has only 5, 4 and 1 sets. The original code then returns 5 sets, but the paper's figure shows 10 (section 1.5f). At 0.49 the original mixed selection does give 10. But per length, no group reaches ten at 0.49. Among the values we tested, 0.45 is the first with ten sets for every length. The exact limit is 0.4788: the tenth single activity of f3 occurs in 532 of 1,111 patients.

**The rule we chose.** We take the largest tested value that gives ten sets for every length: 0.5 for f1 and f2, 0.45 for f3. *(Working choice.)*

**What we found.** Once ten sets per length exist, a lower `min_support` no longer changes the selection. For every tested value at or below the chosen one, down to 0.3, the lists were identical. The reason is simple: we take the top 10 by support, and a lower threshold only adds sets below them.

So one value of 0.45 for all logs gives the same lists. Our scripts are not uniform here. The runtime scripts use 0.45 for all logs (`c1_common.py`). The earlier script `c4_run_selector.py` uses 0.4 for f3. By the finding above, the sets are the same.

**The original selection, for faithful mode.** This is what C7 recommends for `original_top10`:

- f1 and f2: exactly 0.5, because of the ties.
- f3: 0.49. Any value at or below 0.494 gives the same ten sets (we ran 0.49, 0.48 and 0.47). So we cannot tell which value the authors used; this stays an open question. *(Not checked: the order of tied sets inside the ten may still change with the value.)*
- **Careful:** experiments C2 and C10 used 0.5 on all three logs. So they have only five sets on f3. The value 0.49 for f3 is not in a script yet.

**One more finding.** The Apriori sets overlap heavily. On f1, the ten pairs and the ten triples are built from the same five activities. This is a measured reason to try another strategy.

**Status.** **Verified:** the counts and the lists. **Working choice:** "top 10 per length" and the `min_support` values; the supervisor may change them.

### 3.4 Existence importance

*Box 7 in the picture.*

**What it is.** Existence importance asks: does it matter that a set is in the trace at all? We build a small table with one column per set. Then we run the classic shuffle test: shuffle one column across the patients and measure the score drop.

**Why the paper needs it.** A set can matter because it is there, or because of where it is. Location importance measures "where". The paper's main finding is that many activities matter through their position, while their presence alone matters little. *(Not re-checked: from our reference guide.)* That message needs both numbers.

In the Anna example the column for {BLOOD, SCAN} is 1 for all three patients. Shuffling it changes nothing, so its existence importance is zero. But the position of BLOOD decides the label (section 1.1). Only location importance can see that.

**What the authors' script really does.** The script is `Classical_Permutation.py`. Two places matter most. The first builds the column of one set for one patient:

```python
                set_of_items = set(eval(item.strip("'")))
                item_freq = min(trace.count(act) for act in set_of_items)
```

The second runs the shuffle test:

```python
        result_train = permutation_importance(
            model, train_x, train_y, n_repeats=20, random_state=42, n_jobs=2)
```

In the first excerpt:

1. The first line turns the stored text of a set back into a set of activity names (`set_of_items`). `item` is that text; the script reads it from the result file of an earlier run.
2. In the second line, `trace` is one long text: the activities joined with `->`. `trace.count(act)` counts how often the name appears in that text.
3. This is a text search, so `370407` is also found inside `370407c`.
4. The column gets the smallest count among the set's activities.

In the second excerpt, `permutation_importance` is the ready-made shuffle test of scikit-learn. It gets the training patients (`train_x`, `train_y`), and no option that names the score. `n_repeats=20` means: shuffle every column 20 times.

Example: suppose Anna had two blood tests and one scan. Then the column {BLOOD, SCAN} holds 1 for her. With two of each, it holds 2.

| Aspect | The paper says | The script does |
|---|---|---|
| Column per set | yes/no ("binary encoding") | a count: the smallest number of occurrences among the set's activities |
| Matching | not stated | text search, so `370407` is also found inside `370407c` |
| Score | decrease in F1 | accuracy (the share of correct answers) |
| Scored on | not stated | training patients |
| Patients | not stated | no rare-activity filter (f1: 1,140 patients, not 1,130) |

*(Not re-checked: the column "The paper says" follows our reference guide's reading of the paper.)*

The script also makes its own folds, without a seed (function `cross_split_test_train` in `tools.py`). And the model is a fresh XGBoost that sees only the set columns. So the two importances come from two different models on two different tables. This is why we never compare the sizes of existence and location importance, only the rankings.

**How we found the differences.**

1. We read the script.
2. We checked what "no score option" means. In our scikit-learn (1.9.1) the function then uses the model's own score. For a yes/no model that is accuracy.
3. A number check confirmed it: the result without the option equals the result with `scoring='accuracy'`.
4. Older releases, back to the first one with this function (0.22), say the same in every release we read.

Steps 2 to 4 are experiment C18. Limits: we did not read every release in full. We do not know which one the authors used. And we did not run the script itself: its file paths point to another dataset, and it needs a result file of an earlier run. We rebuilt its logic instead.

**Our `ExistenceImportance` class.** The function `encode_existence` builds the table. Its input `traces` is a pandas Series: patient id to list of activity names. `itemsets` is the list of sets.

```python
    rows = []
    for trace in traces:
        counts = [min(occurrence_count(trace, activity, matching) for activity in itemset)
                  for itemset in itemsets]
        rows.append(counts)
    encoded = pd.DataFrame(rows, index=traces.index, columns=range(len(itemsets)), dtype=int)
    if encoding == "binary":
        encoded = (encoded > 0).astype(int)
    return encoded
```

In the order the code runs:

1. For each patient and each set, it takes the smallest occurrence count among the set's activities.
2. All counts together become a table: one row per patient, one column per set.
3. If we ask for `"binary"`, every count above 0 becomes 1.

The same example again: with two blood tests and one scan, {BLOOD, SCAN} gets count 1 for Anna. With two of each, the count is 2, and the binary version is 1.

The method `run` then makes five folds with a seed, trains the model on four, and calls the shuffle test:

```python
            score_x, score_y = (train_x, train_y) if self.score_on == "train" else (test_x, test_y)
            result = permutation_importance(model, score_x, score_y, scoring=self.scoring,
                                            n_repeats=self.n_repeats,
                                            random_state=self.permutation_seed)
```

The first line picks the patients to score: training or held-out. The second line runs the shuffle test with the score we name (`scoring`). The result holds 5 folds × 20 shuffles = 100 importance numbers per set. The options:

| Option | Values | Meaning |
|---|---|---|
| `encoding` | `count` (default), `binary` | count as in the script, or yes/no as in the paper |
| `matching` | `exact` (default), `substring` | whole names, or the original text search |
| `scoring` | `f1_weighted` (default), `accuracy` | which score drop we measure |
| `score_on` | `test` (default), `train` | held-out patients, or training patients as in the original |

The fold, shuffle and model seeds are set (2023, 42, 0), so two runs give the same numbers. One note: the helper `load_log` in this file still reads the log through the authors' `DataManager`. The class itself works on plain traces.

**What experiment C10 showed.** The script is `run_c10.py`. It uses the original top-10 sets at `min_support` 0.5 (ten on f1 and f2, five on f3). It runs the eight combinations of encoding, score and scored part, all with exact names and the same folds. It adds one run with the original's settings, a repeat with five fold seeds, and a count of table values where the two matchings differ. We compared rankings with Spearman. This is a number that tells how alike two rankings are: 1 means the same order, 0 means no link.

- **f1, score:** accuracy and weighted F1 give the same ranking (Spearman 1.000).
- **f1, encoding:** count and binary differ a little (0.89 on held-out patients). The same four sets are on top, the same six near zero.
- **f1, fold seed:** another fold seed changes the ranking at least as much as the encoding does (median 0.81 to 0.82 between seeds; the median is the middle value).
- **f1, matching:** text search and exact names differ in 1 of 11,300 table values.
- **f1, original settings against ours:** Spearman 0.951; the first five places are identical.
- **f2 and f3:** the model on set columns is about as good as always answering with the bigger group. On f3 its accuracy is 0.7669, the same as always answering "no". The rankings there are noise. For one f3 set even the sign flips: +0.0305 with accuracy, -0.0569 with weighted F1.

**What we use, and why.** These are the defaults of the class.

- **Count column:** it is what the authors' script computes.
- **Exact names:** the text search is a small error.
- **Weighted F1:** location importance uses the same score, so both tests measure the same kind of drop.
- **Held-out patients:** the same reason as in section 1.5a. A score on training patients does not show what the model uses for new patients. *(Our reasoning.)*

This differs from the paper's text ("binary") and from the script (accuracy, training data). *(Working choice, until the supervisor answers.)* We plan to average over at least five fold seeds, and to show binary once as a check. No script does this as the standard yet.

**Not finished.**

- **Twin columns.** Two sets can produce exactly the same column, for example a set and a larger set that contains it. On f1, {ac370419, ac370443} and {ac370000, ac370419, ac370443} do. Then the second one always gets exactly zero. *Why:* XGBoost only uses the first of two identical columns. The experiment script lists such twin columns (function `feature_redundancy` in `run_c10.py`). The class `ExistenceImportance` does not yet detect and report them.
- **Only the original sets so far.** C10 used only the original mixed top-10 sets. The per-length Apriori lists and the IMPresseD sets have not been run through this test yet.
- **No test file.** There is no unit-test file for `existence_importance.py`.

## Part 4 — Choosing sets with IMPresseD

Part 3 picked sets by frequency alone. It also showed a weak point: on f1, the frequent pairs and triples are built from the same five activities (section 3.3). Part 4 is about the second strategy: box 4(b) in the picture. Apriori picks sets that are frequent. IMPresseD picks patterns that are also linked to the label. We turn those patterns into activity sets of length 1, 2 and 3.

The code is in `project/experiments/`: `run_original_impressed.py`, `case_distance.py`, `impressed_chain.py` and their test files. The results are in `project/results/experiments/`, folders C3, C4, C5, C11 and C12.

### 4.1 What IMPresseD does, in plain words

**Patterns grow step by step.** IMPresseD starts with single activities, such as BLOOD. Then it looks at every place where BLOOD occurs and glues a neighbour on. We write patterns as the code prints them:

```
BLOOD -> CONSULT     CONSULT comes directly after BLOOD
BLOOD ~> SCAN        SCAN comes a little later (a few events in between)
```

**Every pattern gets three scores.**

- **Frequency:** the share of patients with the pattern. Higher is better. Our code calls it `coverage`.
- **Outcome interest:** how much the pattern tells us about the label. Higher is better. Our code calls it `IG` (information gain).
- **Case distance:** how different the patients with the pattern are from those without it. Lower is better. Our code calls it `CD`.

**Then it keeps the Pareto front.** A pattern is beaten when another pattern is at least as good on all three scores and better on at least one. The **Pareto front** (short: the front) is every pattern that is not beaten. An example with invented scores:

| Pattern | Frequency | Outcome interest | Case distance | On the front? |
|---|---|---|---|---|
| `BLOOD -> CONSULT` | 0.60 | 0.30 | 0.40 | yes |
| `BLOOD ~> SCAN` | 0.50 | 0.20 | 0.50 | no: the first pattern is better on all three |
| `CONSULT -> SCAN` | 0.30 | 0.35 | 0.45 | yes: it has the highest outcome interest |

Only front patterns grow further. In the original tool a human expert chooses among them. We have no expert, so we use the automatic mode: it grows every front pattern that can still grow. (Section 4.5 names the one exception.)

### 4.2 Running the original tool without its window (experiment C3)

**What we did.** We ran the authors' automatic mode, the function `AutoStepWise_PPD`, on f1 from start to end. We did not change their code.

**How.** The tool normally starts from a window on the screen. So `run_original_impressed.py` prepares by hand what the window would prepare. The class `Settings` lists what a user would type in. Besides the column names of the log, these are its fields:

```python
    delta_time: float = -1.0  # seconds; negative => no two events are concurrent (chains)
    max_gap_between_events: int = 3
    max_extension_step: int = 1
    test_data_percentage: float = 0.2
    numerical_attributes: list = field(default_factory=lambda: ["Age"])
    categorical_attributes: list = field(default_factory=lambda: [
        "Diagnosis", "Treatment code", "Diagnosis code", "Specialism code"])
```

Field by field:

1. `delta_time = -1`: no two events count as parallel. Each trace is a simple chain (section 4.5 says why).
2. `max_gap_between_events = 3`: "a little later" means at most 3 events in between. We call this number the **gap**.
3. `max_extension_step`: how many times patterns grow. We call one such time a **growth step**.
4. `test_data_percentage = 0.2`: the tool scores patterns on 80% of the patients. The window has no default; 0.2 is our choice. *(Working choice. We searched the window's code for a default, but we did not read that whole file.)*
5. The last two fields are the **patient facts** for the case distance: age, and four columns with categories.

The script then builds by hand what the window would build: the log, a colour for each activity, the table of patient facts, the activity counts per patient, and the table of distances between patients. Five functions do this, one per input: `load_event_log`, `make_colour_dict`, `create_patient_data`, `fill_activity_counts` and `create_pairwise_distance`. Each names the lines of the window code that it copies. (The colours are only for the tool's drawings. They do not change the results.)

Then `main` calls `AutoStepWise_PPD`. The class `StageProfiler` wraps the original functions from the outside. It measures their time. It also keeps the patterns, which the original function does not return.

**Why.** Before this run we only knew that the code loads. Does it run with our library versions? How long does it take?

**What we found.** It runs without a crash. But it is slow.

| Run on f1 | Time |
|---|---|
| Original code, single activities plus one growth step | 1,632 s (27.2 min), measured |
| Original code, two growth steps | about 4 to 5.5 hours, estimated, not measured |
| A copy with three speed fixes, two growth steps | 153 s (2.6 min), measured |

Of the 27 minutes, 63% goes to comparing every new pattern with every stored pattern. Another 34% goes to the case distance, which searches a long list of patient pairs.

The copy with speed fixes is in `experiments/original_impressed_patched/`. It gives identical results wherever we ran both versions. These runs were: 120 patients and 300 patients with two growth steps, and full f1 up to the first growth step.

All times come from the C3 report. They were measured while other experiments used the same machine, so they change from run to run.

**Why this led to our own small version.** We need two growth steps on three logs, so speed matters. The run also showed things we do not want to copy:

- **The case distance measures the wrong thing.** It should compare who the patients are (age, diagnosis, ...). But the window also puts the activity counts into it. C3 measured the effect on f1: the distance mostly follows which activities a patient has (correlation 0.97) and hardly the patient facts (0.14). (A correlation of 1 means: the two move together fully. 0 means: no link.)
- **It grows on all patients but scores on 80%.** So 190 of the 2,109 patterns of the first growth step occur in no scored patient.
- **It counts double.** It counts some places where a pattern occurs twice, and a few four times (section 4.10).

So our own version is the main path. The original is our cross-check (section 4.10).

### 4.3 Case distance, and a surprise in a library (experiment C12)

**What it measures.** Take one pattern. Split the patients into those with the pattern and those without. The case distance is the average difference between a patient of the first group and a patient of the second. The difference is about patient facts: age, diagnosis, treatment code, diagnosis code, specialism code.

**Why small is good.** Suppose only old patients have `BLOOD ~> SCAN`. Then a link between this pattern and the label may really be a link between age and the label. A small case distance says: the two groups are similar people. *(Our reasoning: this example is our own illustration of the argument in the 2023 paper.)*

**The surprise.** The original code turns each category into a number code (0, 1, 2, ...). Then it calls the scipy library function `pdist(codes, 'jaccard')`. The authors pinned scipy 1.11.4 in their requirements. We use today's version, 1.18.1. The function gives different results in these two versions. We installed the old version in a throw-away folder and measured both. This is experiment C12; all scipy 1.11.4 numbers below come from that one run.

A tiny example: two patients with three facts, coded `[0, 1, 2]` and `[0, 1, 3]`. They differ on one fact of three.

| Who computes | Result | Rule behind it |
|---|---|---|
| scipy 1.11.4 | 0.5 | a fact where both have code 0 is left out: 1 of 2 |
| scipy 1.18.1 | 0.0 | it only sees "code 0 or not", so 2 and 3 look equal |
| ours | 0.3333 | 1 of 3 facts differs |

Both scipy rules depend on which category happens to get code 0. On real data the effect is large. With scipy 1.18.1, the front of single activities has 17 / 33 / 29 patterns (f1 / f2 / f3). With scipy 1.11.4 it has 7 / 12 / 7. With our distance it has 7 / 7 / 7.

**Our version.** In `case_distance.py`, the function `categorical_mismatch_share` computes the category part. For every pair of patients it gives the share of categories on which they differ:

```python
    n_cases = len(case_table)
    n_mismatches = np.zeros((n_cases, n_cases))
    for col in categorical_cols:
        codes = pd.factorize(case_table[col])[0]
        n_mismatches += codes[:, None] != codes[None, :]
    return n_mismatches / len(categorical_cols)
```

In the order the code runs:

1. `case_table` has one row per patient and one column per fact. (The code says "case" for patient.) `n_mismatches` starts as a square table of zeros, with one row and one column per patient.
2. `pd.factorize` gives every category of a column a number. Only "equal or not" is used, so the numbers themselves do not matter.
3. The line with `!=` compares every patient with every other patient: 1 where two patients differ, 0 where they agree. The result is added to `n_mismatches`.
4. The last line divides by the number of category columns.

`pairwise_case_distance` then combines this with age, as the original does:

```python
    return (n_categorical * categorical + numeric) / (1 + n_categorical)
```

In plain words: every fact counts the same. With two categories and age, the result is the average over three facts. `categorical` is the table from the function above, and `n_categorical` is the number of category columns. `numeric` is the age difference, rescaled. The smallest difference of all pairs becomes 0, and the largest becomes 1.

An example with invented facts:

```
        Age  Diagnosis  Treatment code
Anna     40      A           T1
Bea      60      A           T2
Cleo     80      B           T2
```

Anna and Bea differ on 1 of 2 categories (0.5). Their age difference, 20, is the smallest of all pairs, so it becomes 0. Combined: (2 × 0.5 + 0) / 3 = 0.3333. Anna and Cleo differ on everything: (2 × 1 + 1) / 3 = 1.

The score of a pattern comes from `pattern_case_distance`:

```python
    in_mask = np.asarray(in_mask, dtype=bool)
    if in_mask.all() or not in_mask.any():
        return undefined_value
    return float(dist_matrix[np.ix_(in_mask, ~in_mask)].mean())
```

In the order the code runs:

1. `in_mask` is True for the patients with the pattern.
2. If every patient has the pattern, or none, there is no pair to compare. Then the function returns `undefined_value`. We pass 1.0, the worst value.
3. The last line averages the distances of all pairs (patient with, patient without). `dist_matrix` is the table of distances between all patients.

**How we checked it.** **Verified** in two ways. The file `test_case_distance.py` holds 18 unit tests with hand-computed examples. (A unit test is a small test with an answer worked out by hand.) All pass. One of them renames the categories and checks that nothing changes. Experiment C12 also compared with the original code. Given the same distance table, `pattern_case_distance` and the original function agree up to rounding: the largest difference is about 0.0000000000000001. The age part and the combining formula agree with the original too (about 0.0000000000000002).

**Honest status.** Our distance is a working choice. It equals neither scipy version. It is close to what the original code gives under scipy 1.11.4: on f1, 84.5% of the patient pairs get the identical value. Under scipy 1.18.1 only 4.9% do. We do not know which scipy version produced the authors' figures.

### 4.4 A check of the Pareto library (experiment C11)

*Detail: skip on first read.*

The library `paretoset` computes the front. The authors pinned version 1.2.0; we use 1.2.5. We ran both on the same inputs. 141 of 143 recorded results are identical; the other two use features we do not need. The run also gave us two rules:

- **`distinct=False`.** The library's default, `distinct=True`, keeps only the first of several patterns with identical scores. Which one is first depends on the row order. We keep all of them.
- **Never pass NaN.** NaN means "not a number". The original returns it for an undefined case distance. In our tests a NaN row removed valid patterns from the front. So we use 1.0 instead.

### 4.5 Our small IMPresseD: the five stages, and why chains

From here on we walk through `impressed_chain.py`. The selector works in five stages:

1. Step 0: every activity becomes a pattern (section 4.6).
2. Count each pattern per patient and give it three scores (section 4.7).
3. Keep the front (section 4.8).
4. Grow the front patterns, then go back to stage 2 (section 4.8). We do this twice.
5. Turn all scored patterns into activity sets and take ten per length (section 4.9).

**How a pattern is stored.** A pattern is a small object, the class `Pattern`. It has two fields and two helpers:

```python
    labels: tuple[str, ...]
    edges: tuple[str, ...] = ()

    @property
    def extendable(self) -> bool:
        """False for patterns with an eventual edge: the original never extends them."""
        return EVENTUAL not in self.edges

    @property
    def activity_set(self) -> frozenset[str]:
        """The distinct activities of the pattern (order and repetitions dropped)."""
        return frozenset(self.labels)
```

1. `labels` are the activities in order.
2. `edges` says how neighbours are linked: `'direct'` (printed `->`) or `'eventual'` (printed `~>`). So `Pattern(("BLOOD", "CONSULT"), ("direct",))` is `BLOOD -> CONSULT`.
3. `extendable`: a pattern with a `~>` never grows further, as in the original. This is the one exception of section 4.1.
4. `activity_set` drops the order and the repeats. Stage 5 needs it.

**Why chains.** The original tool can treat events with the same timestamp as parallel, with no order between them. Section 1.7 showed that about 88% of neighbouring events share a timestamp. Experiment C5 measured what the parallel setting would do with that, on the cleaned logs f1 / f2 / f3:

- 412 / 110 / 401 traces would become one single block with no order at all.
- The "parallel" pattern around an event would hold a median (middle value) of 25 / 28 / 22 events. 87.6% / 87.7% / 86.5% of these patterns have more than 3 different activities. They cannot become a set of 1 to 3 activities.

So we read each trace as a **chain**: one event after another, in the order of the event number. The 2024 method uses the same order as "position".

On a chain every pattern is a simple line. Two patterns are then the same exactly when their `labels` and `edges` are equal. The original stores every pattern as a small network of events (a graph) and compares whole graphs. That is the slow part of section 4.2. **Verified:** experiment C4 checked that both tests agree: 21,807 pairs of patterns, no disagreement.

### 4.6 Stage 1: every activity becomes a pattern

Step 0 is in `ImpressedChainSelector.fit`. It makes one pattern for every activity:

```python
        # Step 0: single activities; an instance is one event.
        instances: dict[Pattern, set[Instance]] = {
            Pattern((activity,)): {(case, (position,)) for case, position in events}
            for activity, events in sorted(position_index.items())
        }
```

1. `position_index` lists where each activity occurs: in which trace, at which position.
2. An **instance** is one concrete place where a pattern fits: (patient number, positions in the trace).
3. `instances` maps every pattern to the set of its instances.

The code counts from 0. So Anna is patient 0, and her BLOOD (position 2 in section 1.1) is the instance `(0, (1,))`.

### 4.7 Stage 2: counting per patient, and the three scores

`count_matrix` turns the instances into a table: one row per patient, one column per pattern. A cell says how many instances that patient has.

```python
    counts = np.zeros((n_cases, len(instances)), dtype=np.int64)
    for column, found in enumerate(instances.values()):
        for case, _positions in found:
            counts[case, column] += 1
    return counts
```

The table starts with zeros. For every instance of a pattern, the cell of its patient goes up by one. Take the pattern `REG -> BLOOD` and our three patients. The column is 1, 0, 1: Anna and Cleo have it once, Bea does not. (`REG -> BLOOD` is a grown pattern; stage 4 shows how it is made.)

`score_patterns` computes the three scores from this table:

```python
    present = counts > 0
    info_gain = mutual_info_classif(counts, np.asarray(labels), discrete_features=True, random_state=0)
    if dist_matrix is None:
        case_distance = np.full(n_patterns, np.nan)
    else:
        case_distance = np.array([
            pattern_case_distance(dist_matrix, present[:, column], undefined_value=undefined_cd)
            for column in range(n_patterns)
        ])
    return pd.DataFrame({"IG": info_gain, "coverage": present.sum(axis=0) / n_cases, "CD": case_distance})
```

In the order the code runs:

1. `present` says which patients have the pattern at least once.
2. `info_gain` is the outcome interest. `mutual_info_classif` is a scikit-learn function. It measures how much the count of a pattern tells about the label; 0 means nothing. `labels` holds the yes/no label of each patient. The original makes the same call. We only add `random_state=0`, which has no effect on counts.
3. `case_distance` uses the function of section 4.3. `dist_matrix` is the table of distances between all patients.
4. `coverage`, in the last line, is the frequency: patients with the pattern, divided by all patients.

For Anna, Bea and Cleo (labels yes, no, yes; facts as in section 4.3) the code gives:

| Pattern | Frequency | Outcome interest | Case distance |
|---|---|---|---|
| `BLOOD` | 1.0 | 0.0 | undefined, so 1.0 |
| `REG -> BLOOD` | 0.667 | 0.637 | 0.333 |

This is the toy log of section 1.1 in numbers. Every patient has BLOOD, so BLOOD alone tells nothing. `REG -> BLOOD` separates yes from no perfectly.

### 4.8 Stages 3 and 4: the front, and how a pattern grows

**Stage 3: keep the front.** `pareto_front` marks the patterns that are not beaten:

```python
    values, senses = _objective_table(scores, objectives)
    return np.asarray(paretoset(values, sense=senses, distinct=distinct), dtype=bool)
```

1. `objectives` names the scores to use. The selector passes all three unless we tell it otherwise.
2. `_objective_table` looks up the direction of each: `IG` and `coverage` up, `CD` down. It stops with an error if a score is NaN.
3. `paretoset` is the library function that finds the front (section 4.4).

**Stage 4: grow the front patterns.** Growing happens in `extend_pattern`. It visits every place where the parent pattern occurs and glues one neighbour on. This is its core:

```python
    for case, start in find_instances(pattern, traces, position_index):
        trace = traces[case]
        end = start + len(labels) - 1
        inside = tuple(range(start, end + 1))
        has_predecessor, has_successor = start > 0, end < len(trace) - 1

        if has_predecessor:
            child = Pattern((trace[start - 1],) + labels, (DIRECT,) + edges)
            children[child].add((case, (start - 1,) + inside))
        if has_successor:
            child = Pattern(labels + (trace[end + 1],), edges + (DIRECT,))
            children[child].add((case, inside + (end + 1,)))
        for far in range(end + 2, min(len(trace) - 1, end + 1 + max_gap) + 1):
            child = Pattern(labels + (trace[far],), edges + (EVENTUAL,))
            children[child].add((case, inside + (far,)))
        for far in range(max(0, start - 1 - max_gap), start - 1):
            child = Pattern((trace[far],) + labels, (EVENTUAL,) + edges)
            children[child].add((case, (far,) + inside))
        if len(labels) == 1 and has_predecessor and has_successor:
            child = Pattern((trace[start - 1], labels[0], trace[start + 1]), (DIRECT, DIRECT))
            children[child].add((case, (start - 1, start, start + 1)))
```

The names first. The loop visits every instance of the parent pattern. `find_instances` returns each one as (patient number, first position). `start` and `end` are the first and last position of the instance. `inside` lists all positions it covers. `has_predecessor` and `has_successor` say whether the trace has an event before and after it.

Then five rules each make a **child** pattern, in this order:

1. **Directly before:** glue on the event just before the pattern.
2. **Directly after:** glue on the event just after it.
3. **Eventually after:** glue on a later event, with 1 to `max_gap` events in between.
4. **Eventually before:** the mirror image.
5. **Both sides:** for single activities only: the event before, the activity, the event after.

`children` maps each child to the set of its instances. It is a set, so an instance that is reached twice is stored once. Growing BLOOD in Anna's trace `REG BLOOD CONSULT SCAN CALL` with `max_gap = 3` gives:

```
REG -> BLOOD               (rule 1)
BLOOD -> CONSULT           (rule 2)
BLOOD ~> SCAN              (rule 3, one event in between)
BLOOD ~> CALL              (rule 3, two events in between)
REG -> BLOOD -> CONSULT    (rule 5)
```

Rule 4 gives nothing here: no event lies two or more places before BLOOD. Every child comes from a real place in a real trace, so no pattern is invented. The function `extend_patterns` does this for all front patterns of a step and collects the children.

**The cycle.** `fit` repeats one cycle: count, score, mark the front, store the table. A cycle ends like this:

```python
            if step == self.steps:
                break
            instances, parents_of = extend_patterns(front, trace_list, position_index, self.max_gap)
            if not instances:
                break
```

1. If this was the last step (`self.steps`), the loop stops.
2. If not, `extend_patterns` grows the front patterns. The children become the patterns of the next cycle.
3. If the front has no children, the loop stops too.

With `steps=2` we get three cycles: step 0, step 1 and step 2. Step 0 scores the single activities. Step 1 scores the children of the front activities. Step 2 scores the children of the step-1 front. The table `patterns_` keeps every scored pattern. On the real logs, with all patients (experiment C4):

| Log | Step 0: scored → front | Step 1 | Step 2 |
|---|---|---|---|
| f1 | 164 → 7 | 1,204 → 26 | 851 → 50 |
| f2 | 207 → 7 | 1,634 → 44 | 1,175 → 64 |
| f3 | 156 → 7 | 1,031 → 23 | 861 → 39 |

### 4.9 Stage 5: from patterns to activity sets, ten per length

This stage is project logic; it is not in IMPresseD. It is in the method `full_table`. First it groups the patterns by activity set and picks the best pattern of each group:

```python
        rows_of_set: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for row, pattern in zip(pool.index, pool["pattern"]):
            if len(pattern.activity_set) in self.lengths:
                rows_of_set[tuple(sorted(pattern.activity_set))].append(row)

        sets_of_size: dict[int, list[dict]] = defaultdict(list)
        for itemset, rows in rows_of_set.items():
            group = pool.loc[rows]
            non_dominated = group[pareto_front(group, self.objectives, distinct=False)]
            best = min(non_dominated.to_dict("records"), key=lambda row: tie_break_key(row, row["name"]))
```

`pool` is the table of all scored patterns. The first loop turns each pattern into its activity set and groups the patterns by set:

- **Order dropped:** `BLOOD -> CONSULT` and `CONSULT ~> BLOOD` both land in the group {BLOOD, CONSULT}.
- **Duplicates merged:** a group is one set, however many patterns it has.
- **Length = number of different activities:** `BLOOD -> BLOOD` is the set {BLOOD}, length 1.
- **Too long dropped:** `self.lengths` is (1, 2, 3). A pattern with four different activities is skipped.

The second loop gives each set one **source pattern**: the best pattern of its group. Best means: not beaten inside the group, then the highest outcome interest (`tie_break_key`). The set takes the three scores of its source pattern.

Then the sets of one length are ranked:

```python
            layers = pareto_layers(pd.DataFrame(sets), self.objectives, self.distinct)
            for row, layer in zip(sets, layers):
                row["layer"] = int(layer)
            sets.sort(key=lambda row: (row["layer"],) + tie_break_key(row, row["itemset"]))
            for rank, row in enumerate(sets, start=1):
                row["rank"] = rank
```

1. `pareto_layers` calls the front **layer** 1. It removes the front, takes the front of the rest as layer 2, and so on.
2. We sort by layer first. Inside a layer the higher outcome interest wins; then the higher frequency, the lower case distance, the name.
3. The last loop numbers the sets: rank 1, 2, 3, ...

`select` keeps ranks 1 to `k` (10) per length. It returns a list of sets per length, the same format as the Apriori selector.

A tiny example with invented scores and `k = 2`:

| Pattern | Outcome interest | Frequency | Case distance | Becomes |
|---|---|---|---|---|
| `BLOOD -> CONSULT` | 0.30 | 0.6 | 0.40 | {BLOOD, CONSULT}, source pattern |
| `CONSULT ~> BLOOD` | 0.20 | 0.5 | 0.45 | {BLOOD, CONSULT}, beaten in its group |
| `CONSULT -> SCAN` | 0.25 | 0.3 | 0.45 | {CONSULT, SCAN} |
| `BLOOD ~> SCAN` | 0.10 | 0.5 | 0.45 | {BLOOD, SCAN} |
| `BLOOD -> BLOOD` | 0.05 | 0.2 | 0.50 | {BLOOD}, length 1 |
| `REG -> BLOOD -> CONSULT -> SCAN` | 0.90 | 0.9 | 0.10 | dropped: four activities |

At length 2, {BLOOD, CONSULT} beats both other sets, so it is alone in layer 1. We need two sets, so we go to layer 2. There {CONSULT, SCAN} and {BLOOD, SCAN} do not beat each other, and the higher outcome interest wins. The code returns {BLOOD, CONSULT}, then {CONSULT, SCAN}.

### 4.10 How we know it matches the original (experiment C4)

*Detail: skip on first read.*

**What we did.** First, `test_impressed_chain.py` holds 34 unit tests on toy traces with hand-computed answers. All pass. Section 5.3 has the commands for this file and for `test_case_distance.py`.

Second, `c4_crosscheck_original.py` runs the unchanged original functions and our code on the same data. The data: 150 random f1 patients, and full f1 for the first growth step.

| Compared | Result |
|---|---|
| Child patterns | identical in all six comparisons (475, 461, 598, 1,028, 385 and 1,204 patterns) |
| Counts per patient | identical for 2,957 of 3,123 patterns; the original counts 165 exactly twice and 1 exactly four times |
| The three scores | largest difference 0.0, on all 2,087 compared patterns |

The double counting is the original's. For example, it finds `BLOOD -> BLOOD` once as "BLOOD followed by BLOOD" and once as "BLOOD preceded by BLOOD". The factor is the same for every patient of a pattern, so it changes none of the three scores.

**Speed.** The first growth step on full f1 takes the original 344.5 s and our code 0.047 s, for the same 1,204 patterns. On the full logs, our selector needs 7.1 s on f1, 3.5 s on f2 and 2.4 s on f3 to grow and score the patterns. Turning them into ranked sets adds 0.9 to 1.7 s. (All times are from the C4 report, on a busy machine.)

**One known difference.** With `distinct=False` our front is the original's front plus exact ties. In the run on the 150 patients it has 10 / 23 / 35 patterns per step; the original has 10 / 18 / 18. All 22 extra patterns tie exactly with a front pattern.

**What the independent verification added.** A second script, `verify_impressed.py` in the C4 results folder, checked our code again without reusing the first script. According to its log, it passed 53 checks and failed none. Eleven more entries are notes, not checks. (Log files are not in the shared repository, see section 5.1.) It added:

- **Other data:** 200 other f1 patients and other starting activities.
- **The full logs:** the stored runs of section 4.2 (f1 with the original code; f1, f2, f3 with the speed-fixed copy). For this check our code used the tool's own settings: its 80% of the patients and its distance. Every stored step then has the same patterns and scores, and the fronts are reproduced. The f1 run with the original code covers the single activities and the first growth step only.
- **Repeatability:** a shuffled patient order gives the same sets in the same order.
- **Weak spots** that we have not repaired yet (section 4.12).

### 4.11 What the sensitivity runs showed (experiments C5 and C6)

A sensitivity run changes one setting and shows what moves. The equal-timestamp numbers are in section 4.5. The other results, always for f1 / f2 / f3:

**The gap.** We tried `max_gap` 1, 2, 3 and 5.

- Length 1 does not depend on the gap: all ten sets stay. One exception: f2 with gap 5 keeps nine.
- Lengths 2 and 3 do. The table shows how many of the ten sets stay the same as with gap 3 (f1 / f2 / f3):

| | length-2 sets | length-3 sets |
|---|---|---|
| gap 2 | 8 / 6 / 8 | 6 / 4 / 8 |
| gap 5 | 7 / 5 / 7 | 5 / 6 / 7 |

So the gap is not a harmless detail. We must state our value and ask for the authors' value.

**Front sizes.** With three scores the step fronts are those of section 4.8. Without the case distance, the step-0 front shrinks from 7 / 7 / 7 to 3 / 4 / 1 patterns. Too few patterns grow then. So we keep all three scores.

**Are ten sets per length always available?** Yes, under two conditions.

- Sets may come from all scored patterns, not only from front patterns. In our default run every length then has at least 156 candidate sets. From front patterns only, length 1 would have 7 sets in every log.
- We go below the first front when needed. The first front has 7 sets at length 1 in every log, and 9 at length 3 in f3. Layer 2 fills the rest. Elsewhere the first front has 14 to 45 sets, and the outcome interest decides.

**The price of the case distance.** A pattern in very few patients can reach the front because its case distance is the lowest. 7 of our 90 selected sets occur in fewer than 10 patients. Example: f1, length 1, rank 7 is activity `337419c`, found in 3 patients. Such a set can hardly show any importance: almost no trace can be changed.

**Overlap with Apriori.** At length 1 the two strategies share 3 / 3 / 4 of the ten sets. At length 2 they share 0 / 1 / 0, at length 3 none. (Experiments C4 and C5 both measured this.) So we cannot compare the strategies on shared sets. We must compare two groups of sets.

### 4.12 Working choices, and what is not finished

These settings are our working choices. The supervisor may change each of them. The Q numbers are our follow-up questions (section 5.6).

- **Chain traces** in event-number order (Q11). *Why:* section 4.5.
- **Gap 3** (`max_gap=3`, Q11). *Why:* neither the paper nor the code gives a value, and the window has no default. The 3 is our own choice. Section 4.11 shows that it matters.
- **Two growth steps** (`steps=2`). *Why:* length 3 needs them. Of the ten selected length-3 sets, 6 / 8 / 6 (f1 / f2 / f3) come from the second growth step (experiment C4).
- **All three scores** (`objectives`). *Why:* without the case distance too few patterns grow (section 4.11).
- **Patient facts** (Q11): Age, Diagnosis, Treatment code, Diagnosis code, Specialism code.
- **Our own category distance** instead of scipy's (section 4.3). An undefined distance counts as 1.0. Patterns with identical scores are all kept (`distinct=False`, section 4.4).
- **Length** = number of different activities (Q1).
- **Ten sets per length:** front first, then the next layers (Q7): `k=10`.
- **Patterns first, then sets** (Q8): a set takes the scores of its best pattern. Every scored pattern may give a set (`candidate_pool="evaluated"`).
- **Search on the whole log,** with the labels of all patients (Q9). Section 5.6 says why this is a risk.

Not finished:

- **Ranks 9 and 10 can flip on rounding noise.** Two of the 90 default sets change when we round off the scores to 12 decimals. Rounding inside `score_patterns` would repair this; it is not built in yet.
- **A length-1 set can take a repeat pattern as its source.** In f2, {ac419100} takes its scores from `ac419100 ~> ac419100`. With gap 5 this pushes the most frequent activity of f2 out of the top 10. The repair, always use the plain activity, is not built in yet.
- **How the layers are built is one of two readings.** Now: we merge the patterns into sets (each set takes the scores of its best pattern) and then put the sets in layers. The other reading puts the patterns in layers first and merges them into sets afterwards. The table shows how many of the ten sets both readings share. This belongs to Q8.

| Log | length 1 | length 2 | length 3 |
|---|---|---|---|
| f1 | 9 | 10 | 9 |
| f2 | 8 | 10 | 10 |
| f3 | 5 | 8 | 7 |

- **Rare sets.** We have no minimum number of patients for a selected set.
- **Limits of the checks.** We compared with the original code for gap 3 only. For gaps 1, 2 and 5 we counted 2,700 patterns again with a second, simpler function (`count_instances`); none differed. The unpatched original was never run with two growth steps on a full log. Also, `fit` does not yet check that the distance table has the right size.

## Part 5 — Running it yourself, and what comes next

Parts 1 to 4 explained the pieces one by one. Part 5 puts them together from the practical side, so that you can repeat any of it on your own laptop. It shows where every file is, how to install and run things, what each experiment told us, and what is still open. It describes the repository as it was on 2 October 2026. Some experiments were still running then, so a few rows below have no result yet.

### 5.1 A map of the repository

**What.** Every folder and every prototype file, with one line each. **Why.** So that you can find the code behind any number in this document.

```
README.md                  start page: clone and setup commands
PROJECT_BRIEF_Group11.md   the project in plain English
GUIDE_Group11_EN.md        the long reference guide
WALKTHROUGH_Group11.md     this document
project/
  requirements.txt         the tested package versions
  experiments/             prototype modules, tests, experiment scripts
  results/experiments/     one folder per experiment, with a RESULT.md
  scripts/scout/           early checks and two reports (setup, labels)
  external/                the authors' code and data (you clone it, section 5.2)
  .venv/                   your virtual environment (you create it, section 5.2)
```

`project/experiments/` holds two kinds of files. First, the modules we will reuse, with their tests:

| File | What it is for |
|---|---|
| `engine.py` | Loader `EventLog`, encoder `IndexEncoder`, folds with a seed (`make_folds`), and the importance engine `LocationPermutationImportance`. The engine moves the sets and measures the score drop. It contains the faithful/fixed switch (the setting `mode`). |
| `engine_reference.py` | Loads the authors' 2024 `tools.py` unchanged. It only adds the small crash fix (it turns the encoded columns into whole numbers). Only `test_engine.py` and a few experiment scripts (C1, C2) use it. |
| `test_engine.py` | 23 checks of the engine. Some compare it with the original code. The others test rules that must always hold. |
| `apriori_selector.py` | `AprioriSelector`: a top 10 per length. `original_top10`: the original's mixed top 10. |
| `impressed_chain.py` | Our own small IMPresseD. `ImpressedChainSelector` grows patterns, scores them and returns sets per length. |
| `test_impressed_chain.py` | 34 tests on toy traces, answers worked out by hand. |
| `case_distance.py` | The case distance between two patients, written out step by step. |
| `test_case_distance.py` | 18 tests on small hand-made examples. |
| `existence_importance.py` | `ExistenceImportance`: the classic shuffle test behind the existence importance. |

Second, the scripts that ran the experiments. They write their output into the matching folder under `results/experiments/`.

| Experiment | Scripts |
|---|---|
| C1, runtime | `c1_common.py`, `c1_original_vs_engine.py`, `c1_engine_grid.py`, `c1_original_single.py`, `c1_threads.py`, `c1_report.py` |
| C2, the engine | `engine_benchmark.py`, `engine_full_grid.py`, `engine_stability.py`, `engine_fixed_options.py` |
| C3, original IMPresseD | `run_original_impressed.py`, `summarise_original_impressed.py`, `compare_impressed_runs.py` |
| C4, our IMPresseD | `c4_crosscheck_original.py`, `c4_run_selector.py` |
| C5 and C6, sensitivity | `c5_oracle_blocks.py`, `c5_c6_sensitivity.py` |
| C7, Apriori sweep | `run_c7_apriori_sweep.py` |
| C10 and C18, existence importance | `run_c10.py` |
| C11 and C12, library versions | `c11_paretoset_versions.py`, `c12_compare_case_distance.py`, `c12_scipy_worker.py` |

C18 is the check of what "no score option" means in scikit-learn (section 3.4). It was done together with C10.

**The state of the results folders.**

- **Folders that exist:** C1, C2, C3, C4, C5 (also covers C6), C7, C10 (also covers C18), C11, C12.
- **Incomplete:** `C1` has timing files but no `RESULT.md`. The `RESULT.md` in `C11` is short and points to the one in `C12`.
- **Missing:** `C8`, `C9`, `pilot`, and a comparison script.

**Four things are not published** in the shared repository:

- `experiments/original_impressed_patched/`. It is a copy of three files of the authors' 2023 code with three speed patches, plus the list of changes `patches.diff` (experiment C3). It contains their code, so it stays on our laptops.
- All `.log` files. Some `RESULT.md` files name a log file; you will not find it after a clone.
- All `.pkl` and `.npz` files, for example the stored distance table in the C3 run folders. The scripts build them again.
- The `tmp/` and `work/` folders inside the results. They hold throw-away files. (Python's own cache folders, `__pycache__`, are left out too.)

### 5.2 Setup

**What.** Get the authors' code and data, then install the Python packages. **Why this way.** Their code and the hospital logs are not ours, so they are not in our repository. And we pin every package version, so that everyone gets the same numbers.

From the top folder of our repository:

```bash
git clone https://github.com/MozhganVD/PermutationLocationImportance project/external/PermutationLocationImportance
git clone https://github.com/MozhganVD/InteractivePatternDetection project/external/InteractivePatternDetection
```

These two addresses are the ones in our README. We have not tried a fresh clone since. The logs f1, f2 and f3 arrive with the first clone, in its `datasets/` folder. Next, make a virtual environment (a private package folder) with Python 3.12:

```bash
# Windows (tested with Python 3.12.10 on Windows 10)
py -3.12 -m venv project/.venv
project/.venv/Scripts/python.exe -m pip install -r project/requirements.txt

# macOS / Linux (not tested yet)
python3.12 -m venv project/.venv
project/.venv/bin/python -m pip install -r project/requirements.txt
```

Three warnings:

- Do not use Python 3.14 or pandas 3: the original's encoding stops there (section 1.3). We tested only the versions in `requirements.txt`.
- Do not install the `requirements.txt` of the 2023 repository. It does not install on Python 3.12: the pandas version it asks for has no build for Python 3.12. Never import `GUI_IMPresseD_tool.py`; it opens a window at once.
- On a Mac, xgboost may also need `brew install libomp`. We have not tested this.

### 5.3 How to run the tests and the experiments

Run everything from inside `project/`. The commands are for Git Bash on Windows. On a Mac, write `.venv/bin/python` instead.

**Tests first.** *Why:* if a test fails on your laptop, you cannot trust any later number.

```bash
.venv/Scripts/python.exe -m unittest experiments/test_case_distance.py -v    # 18 tests, under 1 second
.venv/Scripts/python.exe -m unittest experiments/test_impressed_chain.py -v  # 34 tests, about 6 seconds
.venv/Scripts/python.exe experiments/test_engine.py --quick                  # 18 checks, about 3 minutes
.venv/Scripts/python.exe experiments/test_engine.py                          # all 23 checks, about 38 minutes
```

The full `test_engine.py` is slow because it also runs the authors' slow code, to compare. The 5 extra checks all do that. One caveat from section 2.5: the full run was made before a small edit of one check. Only the 18 quick checks were run again afterwards.

**A small demo, about four seconds.** It trains one model on f1. Then it takes three sets: the most frequent activity, the two most frequent, and the three most frequent. It moves each set three times in the held-out fold. It prints one row per set and repeat, with the baseline score, the new score and the importance. (Section 2.3b walks through its code.)

```bash
.venv/Scripts/python.exe experiments/engine.py --dataset f1 --mode fixed --score-on test
```

**The experiments.** *Detail: skip this block on first read.* The times are from a busy laptop. Where a command has `--dataset f1`, you can also write `f2` or `f3`. The C3 command runs f1.

```bash
PY=.venv/Scripts/python.exe
# C2: speed, full grid, stability over seeds (50 min), two open options (20 min)
$PY experiments/engine_benchmark.py --dataset f1 --repeats 2 --timing-runs 3
$PY experiments/engine_full_grid.py --dataset f1
$PY experiments/engine_stability.py --dataset f1 --seeds 10 --repeats 30
$PY experiments/engine_fixed_options.py --dataset f1 --seeds 10 --repeats 10
# C3: the original IMPresseD on f1, single activities plus one growth step (27 min)
$PY -u experiments/run_original_impressed.py --max-extension-step 1 --run-name f1_step1 --time-limit-min 45 --patterns-json results/experiments/C3/original_patterns_f1.json
# C4: cross-check with the original (15 min), then our selector (24 s)
$PY -u experiments/c4_crosscheck_original.py --full-core ac370000 --full-front
$PY -u experiments/c4_run_selector.py
# C5 and C6 (90 s, then 5 min)
$PY -u experiments/c5_oracle_blocks.py
$PY -u experiments/c5_c6_sensitivity.py
# C7 (100 s) and C10 (9 min)
$PY experiments/run_c7_apriori_sweep.py
$PY experiments/run_c10.py
# C11 (30 s) and C12 (40 s): each first needs an old package version in a throw-away folder
$PY -m pip install paretoset==1.2.0 --target results/experiments/C11/tmp/paretoset_1_2_0 --no-deps
$PY experiments/c11_paretoset_versions.py
$PY -m pip install scipy==1.11.4 numpy==1.26.4 --target results/experiments/C12/tmp/scipy_1_11_4 --no-deps --only-binary=:all:
$PY experiments/c12_compare_case_distance.py
# C1: runtime (still being measured; the last line builds the tables)
$PY experiments/c1_original_vs_engine.py --dataset f1
$PY experiments/c1_engine_grid.py --dataset f1
$PY experiments/c1_original_single.py --dataset f1
$PY experiments/c1_threads.py --dataset f1
$PY experiments/c1_report.py
```

`c1_threads.py` tests whether the engine needs all processor threads of XGBoost. It also has the options `--size` and `--timing-runs`.

We ran the two unit-test files and the demo again on the final files. For the other commands we checked that each script starts and that every option exists. We did not run them again; their times come from the `RESULT.md` files.

The two `pip install --target` lines do not touch your environment. They put an old version in a `tmp/` folder, so that the script can compare old and new. The scipy folder is large (about 216 MB). You can delete both folders afterwards.

One trap: the scripts in `scripts/scout/` were written on one laptop and still contain an absolute path. Change the `REPO` line (or the path lines) near the top before you run one.

### 5.4 The experiments at a glance

Three reminders. A round is: move one set once, in every trace of the fold that contains it, and score once. Spearman tells how alike two rankings are: 1 means the same order, 0 means no link. The gap is the largest number of events allowed between the two activities of a `~>` link.

| Experiment | Question | Answer, with its key number | Decision it supports |
|---|---|---|---|
| C1 | How long do the runs take on f1, f2, f3? | Still running at the time of writing. | Time plan; size of the demo config. |
| C2 | Can we rebuild the "move and score" loop exactly, and faster? | Yes. Faithful mode gives the original's numbers with difference 0.0. One round on f1 takes 0.053 s instead of 6.78 s (128 times faster). | We use our own engine, with both modes. |
| C3 | Does the original IMPresseD automatic mode run on f1 with our package versions? | Yes, without a crash, but slowly: 27.2 minutes for the single activities plus one growth step. Two growth steps: about 4 to 5.5 hours (estimated, not measured). | The original is a cross-check, not our main tool. |
| C4 | Does our own small IMPresseD find the same patterns? | Yes, in every comparison, for example the same 1,204 patterns on full f1 (0.047 s against 344.5 s). On a 150-patient sample the three scores (outcome interest, frequency, case distance) differ by 0.0. | We use our own version (Q10). |
| C5 and C6 | How much do the IMPresseD settings change the sets? | The gap does not change length 1 (one exception: f2 with gap 5 keeps 9 of 10). But gap 2 instead of 3 keeps only 6 to 8 of the 10 length-2 sets and 4 to 8 of the 10 length-3 sets. | Keep gap 3; name it as a setting (Q11). |
| C7 | Which `min_support` gives 10 Apriori sets per length? | 0.5 works for f1 and f2. On f3 it gives only 5, 4 and 1 sets; 0.45 gives 17, 84 and 217. | 0.5 for f1 and f2, 0.45 for f3 (Q6). |
| C8 | How much do faithful and fixed mode differ? | Still running at the time of writing. | Q4. |
| C9 | Do the rankings stay the same over seeds? | Still running at the time of writing. | How many seeds and repeats the full runs use. |
| C10 and C18 | Existence importance: count or yes/no column, accuracy or weighted F1? (C18: what does "no score option" mean?) | The original measures accuracy. On f1 the choice of score does not change the ranking (Spearman 1.000); the column type changes it a little (0.89). On f2 and f3 the existence model is no better than always guessing the bigger group. | Counts, weighted F1, held-out data (Q13). |
| C11 and C12 | Do other library versions change the Pareto front or the case distance? | paretoset: 141 of 143 results are equal; the other 2 are features we do not use. scipy: with today's version, the original case-distance code puts 17, 33 and 29 single activities on the front (f1, f2, f3); ours puts 7, 7 and 7. | Pin paretoset 1.2.5; use our own `case_distance.py`. |
| pilot | A first comparison of both strategies. (Our own wording; no file describes the pilot yet.) | Still running at the time of writing. | Comparison plan (Q12). |

For C8 and C9, the C2 report (its section 6) already has first measurements, with Apriori sets only.

### 5.5 What is not done yet, and in which order we do it

1. **The final package.** Today the code is a set of prototype scripts. The assignment asks for classes, one main file, a YAML config (a plain text file with all settings) and a README with the exact commands. We do this first, because the final numbers must come from the code we hand in. *(Our reasoning.)*
2. **The full runs.** Three logs, two strategies, three lengths, ten sets each. Main results in fixed mode, plus one faithful run per log. The C1 timing script `c1_engine_grid.py` already runs this grid for one log, to measure the time. We will still run it again with the final package. Here we also decide how many fold seeds we average over (section 2.4f).
3. **The comparison.** Six parts: overlap of the sets, importance per activity, agreement of the rankings, box plots side by side, location importance next to existence importance, and stability over seeds. There is no comparison module yet. Only pieces exist inside experiment scripts: the overlap of the sets (C4, C5), the existence importance (C10), and the stability over seeds for Apriori sets (C2).
4. **The short paper and the poster.** They need the figures of item 3.

Also open: the install on a Mac, and the experiments that are still running.

### 5.6 Open choices that wait for the supervisor

**What.** Twelve choices that the assignment and the papers leave open. **How.** Most are a setting, not a rewrite. Many are default values in a constructor (the function that creates an object). **Why.** When an answer arrives, we change one value and run again, where that is possible.

**The settings of the engine.** In `engine.py`, class `LocationPermutationImportance`:

```python
    def __init__(self, mode="fixed", score_on="test", n_repeats=10,
                 random_state=2023, allowed_from="log", draw="sequential"):
```

| Setting | Meaning | Default |
|---|---|---|
| `mode` | faithful or fixed: how the sets are moved | `"fixed"` |
| `score_on` | which fold we move and score: `"test"` = held-out, `"train"` = training | `"test"` |
| `n_repeats` | how often each set is moved in a fold | 10 |
| `random_state` | the seed of the random moves | 2023 |
| `allowed_from` | where the pools come from: `"log"` = all patients, `"train"` = training patients only (fixed mode only) | `"log"` |
| `draw` | how new positions are drawn: `"sequential"` = step by step, `"uniform"` = every valid placement equally likely (fixed mode only) | `"sequential"` |

The folds have their own seed: `make_folds(labels, k=5, seed=0)`.

**The settings of the IMPresseD selector.** In `impressed_chain.py`, class `ImpressedChainSelector`:

```python
    def __init__(self, max_gap: int = 3, steps: int = 2,
                 objectives: Sequence[str] = ("IG", "coverage", "CD"), distinct: bool = False,
                 k: int = 10, lengths: Sequence[int] = (1, 2, 3), undefined_cd: float = 1.0,
                 candidate_pool: str = "evaluated"):
```

| Setting | Meaning | Default |
|---|---|---|
| `max_gap` | the gap | 3 |
| `steps` | the number of growth steps | 2 |
| `objectives` | the scores for the front: `"IG"` = outcome interest, `"coverage"` = frequency, `"CD"` = case distance | all three |
| `distinct` | `False` keeps all patterns that have exactly the same scores | `False` |
| `k` | the number of sets per length | 10 |
| `lengths` | the lengths we keep | (1, 2, 3) |
| `undefined_cd` | the case distance of a pattern that every patient has, or none | 1.0, the worst value |
| `candidate_pool` | `"evaluated"` means: every scored pattern may give a set, not only front patterns | `"evaluated"` |

**The other settings.** `AprioriSelector` takes `min_support`, `max_len` (default 3) and `top_k` (default 10); see section 3.2. `ExistenceImportance` has the options of the table in section 3.4.

**The twelve choices, by Q number.** Each line gives the current setting, then what would change.

- **Q1, the meaning of length.** Now: the number of different activities (`lengths`). "BLOOD then BLOOD" gives {BLOOD}, length 1. Another meaning moves sets to other groups. Then we change how the selector groups the sets, and we do all runs again.
- **Q2, the fourth log, f4.** Now: `DATASETS` in `engine.py` lists f1, f2, f3. The file for f4 is in the authors' `datasets/` folder. If f4 is wanted, we add one line there, and in the few scripts that list the logs themselves. Then we run one more log.
- **Q4, faithful or fixed.** Now: main results with `mode="fixed"` and `score_on="test"` (held-out), plus one faithful run with `score_on="train"`. Both exist, so the answer only decides what we report.
- **Q6, `min_support`.** Now: 0.5 for f1 and f2, 0.45 for f3. It is the first value we give to `AprioriSelector`. A lower value gives the same top 10 per length (C7). So it does no harm that two experiment scripts use other values (0.4 for f3 in `c4_run_selector.py`, 0.45 for all logs in `c1_common.py`). The authors' value only matters for copying their mixed top 10.
- **Q7, the whole front or a set number.** Now: `k=10` sets per length, filled from the next-best layers when the front is too small. The whole front would give 7 to 45 sets per length (C4).
- **Q8, patterns first or sets first.** Now: patterns first. Each set takes the scores of its best pattern, and the sets are then ranked by these scores (method `full_table`). "Sets first" would mean: score the sets themselves, not their patterns. That needs a new IMPresseD selection (seconds) and new importance runs.
- **Q9, the whole log or training data.** Now: both selectors pick the sets from all patients, including the held-out ones. IMPresseD also uses their labels. So information about the held-out patients reaches the selection. This is called **leakage**, and we state it openly. With training data only, every fold gets its own sets, and we must rebuild how we collect results.
- **Q10, our own IMPresseD.** Now: `impressed_chain.py`. If it is not accepted, our fallback would be the patched original. *(Our reasoning: this is our own plan; no experiment has used it as a selector yet.)* It needs 2.6 minutes on f1 (C3). But it scores the patterns on 80% of the patients and with the window's case distance, so its sets would differ from ours. It is also not in the shared repository (section 5.1).
- **Q11, the IMPresseD settings.** Now: events in file order, `max_gap=3`, case distance from age and four other columns (diagnosis, treatment code, diagnosis code, specialism code). Another gap changes 2 to 5 of the 10 length-2 sets (C5).
- **Q12, the comparison plan.** Now: the six parts listed in section 5.5. The strategies share at most 1 of 10 sets at lengths 2 and 3 (C4 and C5). So we compare groups of sets and single activities.
- **Q13, existence importance.** Now: included, with `encoding="count"` and `scoring="f1_weighted"` in `ExistenceImportance`. For a yes/no column we set `encoding="binary"`.
- **Q14, label naming.** Not a code setting. Our text and plots say "rule satisfied" and "rule not satisfied". The answer only changes words.

Experiment C2 added one more question: the setting `draw`, which decides how the new positions of a set are picked. We keep `"sequential"`, as in the original code, until we have asked.

## Results so far

This is the detailed version of "What we know now" from the start of this document.

- **The original 2024 code runs, after one small fix.** It needs Python 3.12, pinned package versions, and one extra line that turns the table columns into whole numbers. One round (move one set once and score once) takes 6.78 seconds on the log f1 (section 1.4).
- **The code does not do exactly what the paper says.** It scores on training patients, it lets moves pile up, it sometimes breaks the order rule, and its folds have no seed (section 1.5).
- **Our engine gives the original's numbers, much faster.** In faithful mode, every value we compared on f1, f2 and f3 equals the original's: the largest difference is 0.0. One round takes 0.053 seconds instead of 6.78 (sections 2.5 and 2.6).
- **Fixed mode keeps the paper's two rules.** In every move that our checks looked at, each moved activity stayed on a seen position, and each set kept its inner order (section 2.5). The other events, which slide aside, are not bound by the first rule (section 2.4c).
- **Apriori needs a lower `min_support` on f3.** The value 0.5 gives ten sets per length on f1 and f2; f3 needs 0.45. Below these values, the ten sets per length did not change in our tests (section 3.3).
- **The authors' existence script differs from the paper's text.** It uses a count column, accuracy and training patients. On f2 and f3, a model that sees only set columns is about as good as always answering with the bigger group (section 3.4).
- **Timestamps only show the day.** So "position" is the event number in the file. For IMPresseD we read each trace as a simple chain (sections 1.7 and 4.5).
- **The original IMPresseD runs, but slowly; our small version matches it.** The original needs 27.2 minutes on f1 for the single activities plus one growth step. Our version finds the same patterns and the same three scores in seconds (sections 4.2 and 4.10).
- **A library change forced our own case distance.** The scipy function behind the original case distance gives different results in the version the authors pinned and in today's version (section 4.3).
- **The two strategies pick almost completely different sets.** At length 1 they share 3, 3 and 4 of the ten sets (f1, f2, f3). At lengths 2 and 3 they share at most one. So we must compare two groups of sets, not shared sets (section 4.11).

What we do not know yet is the answer to the project's question. The full runs with both strategies and the comparison itself are not done (section 5.5). Four experiments had no result at the time of writing: C1, C8, C9 and the pilot. Twelve choices wait for the supervisor (section 5.6).

## Glossary

### Basic words

- **Activity:** a type of step, such as BLOOD.
- **Event:** one row of the log: one activity of one patient.
- **Trace:** the events of one patient, in order.
- **Event log (log):** the whole data file. We use three: f1, f2 and f3.
- **Position:** the place of an event in its trace: 1, 2, 3, ... The paper's word is "location".
- **Label:** the yes/no answer of a patient; 1 means "rule satisfied".
- **Model:** it reads the table row of a patient and predicts the label. Our model is XGBoost, which builds many small decision trees.
- **Activity set:** a few activities, without order. The code says "itemset".
- **Length:** the number of different activities in a set (our working definition).
- **Strategy:** the way to pick the sets: Apriori or IMPresseD.
- **Apriori:** finds sets that occur together in many traces.
- **Support:** the share of patients whose trace contains a set. `min_support` is the lowest support Apriori keeps.
- **IMPresseD:** finds small ordered patterns and scores them, also with the label.
- **Pattern:** a few activities with an order, like "BLOOD then CONSULT".
- **Pareto front (front):** the patterns that are not beaten. A pattern is beaten when another one is at least as good on all three scores and better on at least one.
- **Fold:** the patients are split into five folds. Each fold is held out once; the model learns from the other four (the training fold).
- **Held-out:** patients the model did not learn from.
- **Weighted F1:** the model's score, from 0 to 1. Higher is better.
- **Baseline:** the score before we move anything.
- **Location importance:** the score drop after moving a set to other positions.
- **Existence importance:** the score drop after shuffling the column "is the set there".
- **Faithful mode, fixed mode:** our code behaves like the original, or with its errors repaired.
- **Leakage:** information from held-out patients slips into the method.
- **Seed:** a starting number that makes random steps repeatable.
- **Box plot:** a picture of many numbers: a line at the middle value, and a box around the middle half of the numbers.

### Extra words of this document

- **Round:** move one set once, in every trace of the fold that contains it, and score once.
- **Repeat:** one more round for the same set, with a new random move. The default is 10 repeats for every set in every fold.
- **Full grid:** all combinations of folds, sets and repeats.
- **Allowed positions (pool):** for one activity, the positions where it appears somewhere in the log. A moved activity may only go there.
- **Occurrence:** one place where a whole activity set is found in a trace. A trace can hold several.
- **Encoding:** turning the traces into a table for the model. Ours has one column per pair of position and activity.
- **Padding column:** a column such as `e4_0`. It says that a trace has no event at that position.
- **Code words:** the code says `itemset` for activity set, `case` for patient, `location` for position, `size` for length and `test` for held-out. "Permute" and "shuffle" mean: move.
- **Check:** one test inside `test_engine.py`. An equality check compares our result with the original's. A property check tests a rule that must always hold.
- **Unit test:** a small test with a hand-computed answer (the files `test_case_distance.py` and `test_impressed_chain.py`).
- **Working choice:** a setting we use until the supervisor answers. We can change it and run again.
- **Pinning:** writing the exact version of every package into `requirements.txt`.
- **Virtual environment:** a private package folder for one project (`.venv`).
- **Dictionary:** a look-up table in Python: a key goes in, a value comes out.
- **Timestamp:** the date and time that the file gives for an event.
- **Basket:** the different activities of one trace, without order and without repeats. Apriori works on baskets.
- **Tie:** two sets with the same support, or two patterns with the same scores.
- **Count encoding, binary encoding:** the existence column of a set holds a count of occurrences, or only 1 or 0.
- **Spearman:** a number that tells how alike two rankings are. 1 means the same order, 0 means no link.
- **Median:** the middle value of a list of numbers.
- **Frequency, outcome interest, case distance:** the three scores of an IMPresseD pattern. Our code calls them `coverage`, `IG` and `CD`.
- **Patient facts:** the columns used for the case distance: age, diagnosis, treatment code, diagnosis code and specialism code.
- **Instance:** one concrete place where a pattern fits: a patient and the positions in the trace.
- **Child pattern:** a pattern made by gluing a neighbouring event onto an existing pattern.
- **Growth step:** one time of growing all front patterns. The code calls it an extension step.
- **Cycle:** one pass of the IMPresseD selector: count, score, mark the front, store the table.
- **Gap (`max_gap`):** the largest number of events allowed between the two activities of an "eventually followed by" link (`~>`).
- **Chain trace:** a trace read as one event after another, with nothing parallel.
- **Parallel events:** events that the original IMPresseD treats as having no order, because their timestamps are equal.
- **Layer:** the Pareto front is layer 1. Remove it, and the front of the rest is layer 2, and so on.
- **Source pattern:** the best pattern of an activity set. The set takes its three scores from it.
- **NaN:** "not a number", the value a computer gives when a result is undefined.
