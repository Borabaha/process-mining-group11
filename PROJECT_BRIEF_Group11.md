# Does the Position of a Step Matter? Project Brief for Group 11

*Topic: Explainable Predictive Process Monitoring. This means: a model predicts how a running case will end, and we explain what the model looks at. Supervisor: Laura Genga. She co-wrote the two papers this project builds on. 2 October 2026. No prior reading needed.*

*Short on time? Read sections 1–3 (the idea) and 11 (the questions). Sections 12–14 are for looking things up. Bold terms are in the glossary (section 14).*

## 1. Start with one patient

Meet Anna, a patient we invented for this brief. Her hospital visit has five steps. Such an ordered list of steps is called a **trace**.

```
position:   1     2       3       4     5
Anna:      REG  BLOOD  CONSULT  SCAN  CALL
```

REG = registration, BLOOD = blood test, CONSULT = talk with the doctor, SCAN = ultrasound scan, CALL = follow-up call.

A model can learn from finished patient stories to guess a yes/no answer: will Anna later need a special cancer test? But a model that only says "yes" does not tell us why. We cannot look inside it. People call this a black box. The hospital wants to know what the model relies on. Is it that Anna had a blood test at all? Or that it came so early, at position 2?

Our project is about that second question: does the **position** matter?

## 2. The project in one minute

- A 2024 paper by Vazifehdoostirani and colleagues measures how much the position of activities matters to a prediction model. We must **reproduce** it.
- An **activity** is a type of step, such as BLOOD or SCAN. The paper asks about groups of activities. So it tests single activities and also small groups, called **activity sets**, for example {BLOOD, SCAN}.
- The paper picks the sets with Apriori, an algorithm that finds activities that often occur together in the same trace.
- We must also build a **variant**: the same method, but the sets come from IMPresseD, the method of a 2023 paper by the same first author.
- IMPresseD returns ordered patterns such as "BLOOD then CONSULT". The assignment tells us to ignore the order. So we keep only the set {BLOOD, CONSULT}.
- The assignment asks for sets of length 1, 2 and 3 (roughly: 1, 2 or 3 activities; see section 6). We run everything on three hospital data files, called logs: BPIC11 f1, f2 and f3. The paper's figures use these three; Q2 checks that a fourth file, f4, is out.

**The one question to answer at the end:** when IMPresseD picks the sets instead of Apriori, does the importance of the activities and of their positions change? If so, how?

## 3. The idea in plain words

Section 2 said the paper measures how much position matters. Here is how such a measurement works.

### The classic test: shuffle a column

The model reads a table: one row per patient, one column per fact. The classic test shuffles one column across the patients. After the shuffle we check the model's score (how often its answers are right). If the score drops a lot, the model relied on that column. This test is called permutation feature importance ("permutation" just means shuffling).

### Why this test cannot see position

The table in this project has one column for every pair of position and activity (step 2 of section 5 shows it). So the position of the blood test is spread over many columns: "BLOOD at position 1", "BLOOD at position 2", and so on. Shuffling one of them creates impossible patients, for example with a blood test at two positions at once. And in a real trace, moving one activity shifts the others. A column shuffle cannot do that.

### The paper's idea: move the activity inside the trace

The paper moves BLOOD to another position inside Anna's trace and asks the model again. If the score drops a lot, the position matters. The paper's word for position is **location**: the position number in the trace (1, 2, 3, ...), not a clock time or a place. We say "position"; only the score's name, location importance, keeps the paper's word.

Three patients show why this finds more than the classic test. (The paper has a similar example.)

```
Anna:  REG  BLOOD    CONSULT  SCAN     CALL   -> yes
Bea:   REG  CONSULT  SCAN     BLOOD    CALL   -> no
Cleo:  REG  BLOOD    SCAN     CONSULT  CALL   -> yes
```

Every patient has a blood test, so "has a blood test" tells the model nothing. But the position separates them perfectly: position 2 means "yes", position 4 means "no". Only a test that moves activities can see this.

## 4. The data

Before the method, a short look at the data it runs on.

One row of the data file is one **event**: which patient, which activity, when. All events of one patient, in order, form the trace. The whole file is an **event log**. (The technical word for one patient's visit is "case".)

BPIC11 is a real log from the gynaecology department of a Dutch academic hospital. What you need to know:

- f1, f2 and f3 come from the same log. Only the yes/no question (the **label**) differs.
- The files keep only the first events of each patient: at most 36 in f1, 40 in f2 and 31 in f3.
- Activities are codes such as `ac370000`, not names. So we cannot yet say what an activity means medically (Q14).
- The authors' code folder (the repository) also has a file f4. The 2024 paper does not use it, so we skip it for now (Q2).

### What the label means

Each label comes from a rule about medical tests. **Label 1 means the rule is satisfied.** We checked this in the data. Other papers name the labels differently (Q14).

- **f1:** at some point the patient gets a CA-19.9 or CA-125 tumour-marker test (a lab test for signs of cancer). 40% of the patients have label 1.
- **f2:** every CEA tumour-marker test is later followed by a squamous-cell-carcinoma test (another lab test for a type of cancer). Patients without a CEA test also get label 1. 78% have label 1.
- **f3:** a squamous-cell-carcinoma test happens, and no biopsy examination comes before it. 23% have label 1.

### Two things that change how we read results

**The deciding tests were cut out.** The people who prepared f1 and f3 cut every trace just before the test that decides the label. Otherwise the model would just read the answer. So an important activity is a hint that comes before the test, not the test itself. f2 is not cut, because there a later CEA test can still change the answer. So in f2 the deciding tests are still in the traces and can come out as important themselves.

**Timestamps only show the day.** About 87–89% of the events have the same timestamp as the event before them. So when two activities happen on the same day, we do not know which one really came first. Their position is just the order of the rows in the file. We keep this in mind when we read results, and we say it openly in our paper.

## 5. The original method, step by step

Now we know the idea and the data. Here is what the 2024 paper does, in seven steps.

```
event log
   |
[1] clean and sort
   |-----------------------------------.
[2] one table row per patient         [4] pick the activity sets
[3] train the model five times             (a) Apriori   = original
   |                                       (b) IMPresseD = our variant
   |-----------------------------------'
[5] move the set to other positions
[6] ask the model again, measure the drop, repeat 50 times
[7] compare with the classic shuffle test
```

From now on, a **strategy** is a way to pick the activity sets: Apriori or IMPresseD. Only box 4 differs between the two strategies. Everything else is the same.

### Step 1: Clean and sort the log

- **What we do:** We sort each patient's events by event number. We remove every patient who has an activity that occurs only once in the whole log. For each activity, we note all positions where it appears in the log.
- **Why:** A model cannot learn from an activity it sees once. (This is our reading; the paper does not mention this filter.) Step 5 needs the positions.
- **Out:** a clean log.
  - f1: 1,130 patients, 164 activities
  - f2: 1,130 patients, 207 activities
  - f3: 1,111 patients, 156 activities

### Step 2: Turn each trace into a fixed-width row

- **What we do:** We make one 0/1 column for every pair of position and activity. This is **index-based encoding**. The name `e2_BLOOD` means: event number 2 is BLOOD. Anna's row:

```
e1_REG=1  e2_BLOOD=1  e3_CONSULT=1  e4_SCAN=1  e5_CALL=1   (all other columns = 0)
```

- **Why:** The model needs the same columns for every patient. This encoding also keeps the position: "BLOOD at position 2" becomes a fact the model can use.
- **Out:** f1 has 5,939 columns. That is 36 positions × 164 activities = 5,904 columns, plus 35 extra columns. They mark positions that are empty because a trace has fewer than 36 events.

### Step 3: Train the model with 5-fold cross-validation

- **What we do:** We train the model XGBoost with its default settings, as the paper does. XGBoost builds many small decision trees. A decision tree is a chain of yes/no questions, such as "is BLOOD at position 2?". We split the patients into five groups, called **folds**. Five times, the model learns from four folds, and one fold is held out. This is cross-validation.
- **Why:** The held-out fold shows how good the model is on patients it has not seen.
- **Out:** five trained models, each with a **baseline score**: its score before we move anything.
  - The score is **weighted F1**, a number from 0 to 1. It is high when the model finds the real "yes" patients without many false alarms. "Weighted" means: we compute it for the "yes" and for the "no" patients, and the bigger group counts more.
  - On held-out patients of f1, the model scores about 0.89–0.91. Careful: the original code does not use this as its baseline. It uses the score on the training patients, about 0.99 (see section 7).
  - Do not mix them up: F1 is the score, f1 is the data file.

### Step 4: Pick the activity sets with Apriori

- **What we do:** Apriori lists the sets that occur together in many traces. It ignores the order and the label. Its key number is **support**: the share of patients whose trace contains the whole set. In our three-patient log, {BLOOD, SCAN} has support 100%. The original code then does three things:
  1. It keeps only sets that at least half of the patients have (the setting is `min_support` = 0.5).
  2. It throws away sets with only one activity.
  3. It sorts the rest by support and keeps the top 10.
- **Why:** We cannot test every possible group. A frequent set also gives each test plenty of patients.
- **Out:** up to ten sets per log.

### Step 5: Move the set to other positions

- **What we do:** We take every patient whose trace contains the whole set. In that trace, we move the activities of the set to new positions, chosen at random. Two rules apply:
  1. **Only seen positions.** An activity may only go to a position where it appears somewhere in the log.
  2. **Keep the inner order.** The activities of the set keep their order among themselves.

Take {BLOOD, SCAN}. Suppose the log has blood tests at positions 2, 3 and 4, and scans at 3, 4 and 5.

```
before:  REG  BLOOD    CONSULT  SCAN  CALL
after:   REG  CONSULT  BLOOD    CALL  SCAN
```

BLOOD moved from 2 to 3, SCAN from 4 to 5. BLOOD still comes first. The other activities slide to fill the gaps.

- **Why:** Rule 1 avoids traces that could never happen. Rule 2 makes sure we test position only, not the order inside the set.
- **Out:** a changed copy of the log, turned into table rows again as in step 2.

### Step 6: Measure the drop, and repeat

- **What we do:** The trained model predicts again on the changed rows. We do not train it again. The **location importance** is the baseline score minus the new score. We repeat the random move 10 times in each of the 5 folds. This gives 50 importance numbers per set. We draw them as a **box plot**: a line at the middle value, and a box around the middle half of the numbers.
- **Why:** A big drop means the model relied on where the set sits. If it learned "BLOOD at position 2 means yes", it now gets Anna wrong. We repeat because one random move can be lucky.
- **Out:** one box per set, for each log. It shows what the model uses. It is not proof of a medical fact.

### Step 7: Compare with existence importance

- **What we do:** We run the classic shuffle test on a simpler table. It has one column per set, which says whether the set is in the trace. The result is the **existence importance**.
- **Why:** A set can matter because it is in the trace at all, or because of where it is. Step 6 measured "where". This step measures "at all". With both numbers, we can tell the two reasons apart.
- **Out:** two boxes per set. The paper's main finding: many activities matter through their position, while their presence alone matters little. Careful: the two numbers come from two different tables. So we do not compare their sizes directly. We only compare which sets come out on top in each test.

## 6. Our variant: sets from IMPresseD

The paper itself admits a weak point: frequent sets are not always the sets that matter for the label. The authors suggest a next step: pick the sets with the method of their 2023 paper. That method looks for patterns that are linked to the label. Trying this is our assignment.

### What IMPresseD does

It starts with single activities and grows them into small ordered **patterns**, such as "BLOOD directly followed by CONSULT" or "BLOOD eventually followed by SCAN". Every pattern gets three scores:

- **Frequency:** the share of patients with the pattern. Higher is better.
- **Outcome interest:** how much the pattern tells us about the label. Higher is better.
- **Case distance:** how different the patients with the pattern are from the others, in things like age and diagnosis. Lower is better: a big difference suggests the pattern only reflects who the patients are.

Then it throws out every pattern that is beaten by another one. "Beaten" means: the other pattern is at least as good on all three scores and better on at least one. The patterns that are left form the **Pareto front**. In the original tool, a human expert then chooses which patterns to grow further. We have no expert. So we need the automatic mode, which chooses without a human.

### How a pattern becomes a set

We ignore the order and keep only the activities.

```
BLOOD -> CONSULT           becomes  {BLOOD, CONSULT}
CONSULT -> BLOOD           becomes  {BLOOD, CONSULT}
BLOOD ... later CONSULT    becomes  {BLOOD, CONSULT}
```

Three patterns give one set, which we keep once.

### What "length 1, 2, 3" means

Our working definition: length is the number of different activities in the set. So {BLOOD, CONSULT} has length 2, and "BLOOD then BLOOD" becomes {BLOOD}, length 1. We have not confirmed this. The 2023 paper only defines length 1, as a single activity. It is our first follow-up question (Q1).

### Apriori must get the same treatment

For IMPresseD we will have three groups of sets: length 1, length 2 and length 3. The original Apriori step gives something else: no single activities, and lengths 2 and 3 mixed in one top 10. We cannot compare three groups with one mixed list. So we also run Apriori once per length: a top 10 for length 1, for length 2 and for length 3 (Q6).

| Aspect | Apriori (original) | IMPresseD (our variant) |
|---|---|---|
| Looks at | activities that occur together in many traces | small ordered patterns with three scores |
| Uses the label? | no | yes, in the outcome interest |
| Returns | sets without order | ordered patterns; we turn them into sets |
| How many | a fixed top 10, or fewer if support is low | not fixed; the front can be small or large |

## 7. What we learned by running the original code

So far this was theory. We also ran the authors' 2024 code on f1 with Python 3.12. It did not behave exactly like the paper.

### What runs, what crashes, and the fix

- **What runs:** loading, Apriori, encoding and model training.
- **What crashes:** the main function, `itemset_permutation_importance` ("itemset" is the code's word for activity set). Cause: with today's pandas library, the table has true/false columns next to number columns, and XGBoost refuses that mix.
- **The fix:** one line that turns all encoded columns into whole numbers. We tested it; it works.
- **A trap:** do not pass `--top_k` on the command line. The script reads it as a decimal number and crashes.

### Four things the code does differently from what the paper says

1. **Scores come from training data,** the patients the model learned from. There it scores about 0.99, against 0.89–0.91 on held-out patients. Why this matters: the model knows these patients almost by heart. A drop from 0.99 on known patients does not tell us what the model would use on new patients.
2. **Moves pile up.** The code makes one copy of the data and keeps changing that same copy. So when it tests the second set, Anna's trace is still mixed up from the first set. On f1 this is true for about 97% of the patients tested for the second set. Why this matters: from the second set on, the score drop comes from several sets together, not from the one set we want to test.
3. **The order rule sometimes fails.** The code loses track of positions while it moves things, so it sometimes moves the wrong activity. Result: SCAN can end up before BLOOD, although rule 2 of step 5 forbids this. We measured one test set on f1 ({ac370419, ac370442}, 612 traces in which the set occurs exactly once). The inner order ended up reversed in about 3.4% of them.
4. **The folds have no seed.** A **seed** is a fixed starting number that makes random steps repeatable. Without it, two runs gave held-out scores of 0.889 and 0.906 for the first fold. Nobody could repeat our numbers.

### The f3 puzzle, and runtime

At `min_support` 0.5, f3 gives only 5 sets with more than one activity. The paper's figure shows 10. At 0.49 we do get 10. So the authors probably used a lower value. We have not confirmed it (Q6).

One round (move the set once, score once) takes about 7 seconds on f1. The original settings need 500 rounds per log: 10 sets × 50 repeats. That is about one hour per log for one strategy. Our full plan has two strategies and three lengths: six times as much, roughly 5.6 hours per log. These numbers are simple multiplication. We only timed f1, so f2 and f3 may be slower or faster.

### The IMPresseD code

Three problems with the original IMPresseD code:

- It is built around a window (a graphical interface). Even the automatic mode, the function `AutoStepWise_PPD`, is normally started from that window.
- Reading the code, we expect it to be slow: it compares every new pattern with every stored one. An earlier probe took minutes per activity, but we have not re-timed it (section 12).
- Its case distance uses a function of the scipy library that changed between versions. So with today's scipy the numbers may differ from the authors' numbers.

We have not yet run it from start to end on our data.

### Our answer: a switch with two modes

We build both behaviours into our code. One setting in the config file chooses: faithful or fixed.

| Aspect | Faithful mode | Fixed mode |
|---|---|---|
| Purpose | show that we reproduced the original | give results we can trust |
| Scores measured on | training data | held-out data |
| Data copy | one copy, moves pile up | a clean copy for every repeat |
| Order rule | as in the original (error kept) | repaired |
| Fold seed | fixed seed | fixed seed |

Why both? We do not know yet which one the supervisor wants (Q4). With both ready, we lose no time whatever she says. The gap between the two modes also shows how much the first three problems change the results.

## 8. How we will build it

Because of these findings we do not just reuse the authors' script. This is what we build instead.

### The building blocks

1. **Loader.** Reads and cleans a log.
2. **Encoder.** Turns traces into table rows, with the crash fix built in.
3. **Set selector.** The code for a strategy. It has two versions, Apriori and IMPresseD. Both return the same simple thing: a list of activity sets per length.
4. **Importance engine.** Makes the folds, trains the model, moves the sets and measures the score drop. It contains the faithful/fixed switch.
5. **Comparison and plots.** Takes the results of both strategies and makes the comparisons of section 9 and the box plots.
6. **One main file with a YAML config** (a plain text file with all parameters). We ship a small demo config (minutes) and a full config (hours).

### Why this shape

Only the set selector differs between the two strategies. So any difference in the results comes from the sets, not from the code. That makes the comparison fair. This shape also follows the course rules for code:

- object-oriented `.py` files (no notebooks) and one main file;
- a README with the exact commands, and a `requirements.txt`;
- all parameters in a YAML file, no absolute paths;
- results that another person can reproduce.

### Why we write our own small IMPresseD

Section 7 listed the problems of the original code, and the assignment says to focus on the essence. So we plan a small version of our own (Q10). One simplification: the original can treat two events as "parallel", meaning there is no order between them. Our version does not. It takes the events one after another, in the order of the file. This is acceptable because our timestamps only show the day: we cannot know what really happened in parallel anyway.

### First coding steps, in order

1. Loader and encoder. Check: f1 gives 1,130 patients, 164 activities and 5,939 columns.
2. Folds with a seed. Check: two runs give the same score.
3. Apriori selector. Check: with the original settings (lengths 2 and 3 mixed in one top 10), it returns the same ten sets as the authors' code on f1. One warning: four sets tie at places 9 to 12, so the last two can differ for a harmless reason.
4. Then the engine with its switch, the IMPresseD selector, and the experiment loop.

## 9. How we will compare the two strategies

When both strategies have run, we have two piles of results. This is how we compare them.

- **Overlap of the chosen sets.** We count the shared sets and divide by all sets (this number is called Jaccard). We also check the overlap among the top-ranked sets. *Why:* first we must know whether both strategies pick the same sets at all.
- **Importance per single activity.** Example: Apriori picks {BLOOD, SCAN} and IMPresseD picks {BLOOD, CONSULT}. The sets differ, so we cannot compare them one to one. But BLOOD is in both. So we give each activity the average importance of the sets that contain it, and compare activities. *Why:* the assignment asks about the importance of activities, and this works even when the two strategies pick different sets.
- **Agreement of the importance rankings.** For each strategy we rank the shared sets, and the activities, from most to least important. Then we check if the two rankings look alike. Spearman and Kendall are two standard numbers for this: 1 means the same order, 0 means no link. *Why:* the strategies may pick different sets but still point at the same important activities.
- **Importance distributions side by side.** We put the box plots of both strategies next to each other. *Why:* this shows if IMPresseD sets react more strongly to a change of position.
- **Location importance next to existence importance.** *Why:* the paper's message was "position matters, presence hardly". We check whether this still holds for sets that were picked with the label.
- **Stability over several seeds.** We repeat the runs with different seeds. *Why:* a difference smaller than the run-to-run noise is not a finding.

## 10. Deliverables and timeline

What we must hand in, and when.

We hand in the Python code, a short paper (at most 5 pages without references) and a poster with a 3-minute pitch. A technology statement is also mandatory; we write it at the end. "Peer feedback" in the table means: we review the draft paper of another group. At the poster sessions, each of us also reviews two posters.

Bold rows are fixed deadlines; the rest is our own plan.

| When | What must exist |
|---|---|
| **Fri 2 Oct, 15:00** | **Office hour 1: answers to our six questions** |
| 3–8 Oct | Setup on every laptop; original method reproduced on f1–f3 |
| 9–15 Oct | Our own code: loader, encoder, selectors, engine |
| 16–24 Oct | Full experiments, then the comparison |
| **Sun 25 Oct, 15:00** | **Questions sent for office hour 2** |
| **Mon 26 Oct, 10:30** | **Office hour 2** |
| **Wed 28 Oct** | **Short-paper draft submitted** |
| **Tue 3 Nov** | **Peer feedback submitted** |
| **Thu 5 Nov** | **Poster session of other groups; we review posters** |
| **Mon 9 Nov, 23:59** | **Final submission of everything** |
| **Thu 12 Nov** | **Our poster session, and code explanation to the lecturers** |

## 11. Questions for the supervisor

The assignment text is short and leaves choices open. The papers also skip details that the code needs. A wrong guess costs days of computing or points. So we ask, and we work with a default until we know.

We sent six questions before the deadline of 1 October. They come first below. After that we ran the code and found more open points. We call these follow-ups Q1–Q14. Each follow-up has four parts: the question in our own words, **why** we ask, what it costs **if we guess wrong**, and what we do **until then**.

When we ask:

- **(office hour 1)** Q1, Q2, Q4, Q7, Q9: on 2 October, if there is time.
- **(soon)** Q6, Q8, Q10, Q12: they also change our design. We want to e-mail them right after office hour 1. We do not know yet if she answers questions by e-mail.
- **no mark** Q3, Q5, Q11, Q13, Q14: they can wait for office hour 2 on 26 October.

When an answer arrives, add an "Answer:" line under the question. Then this brief stays useful.

### Already sent: six questions for office hour 1

1. **Where is the code of the 2024 paper?** Why: without the authors' code we would have to guess every detail. Status: solved. We found both repositories; we only listen for other code she may point to.
2. **How is Apriori used?** Why: Apriori has settings, for example how frequent a set must be. The paper does not give them. Other settings give other sets, and then our plots cannot match the paper's.
3. **Must we reproduce every run and box plot, for all lengths and logs?** Why: the full plan needs about 5.6 hours of computing per log. If a smaller part is enough, we save days. If we skip a part she expects, we lose points.
4. **May we use the authors' main script with our own set generator?** Why: this decides whether we build on their script or write our own code. Their script crashes with today's libraries, so we prefer our own. If she insists on their script, section 8 changes.
5. **Automatic IMPresseD, or expert selection?** Why: the original tool asks a human expert to choose patterns. We have no medical expert. If the automatic mode is not accepted, we have no way to choose patterns at all.
6. **Can a human expert make the result biased?** Why: an expert who chooses patterns by hand may choose what they already believe. Then the result depends on the person. If she confirms this, it is a good reason for our automatic mode.

### Follow-ups 1: what to run

**Q1. Length (office hour 1).** "Is length the number of different activities in the set? Or the number of steps in the pattern? Or the number of times IMPresseD has grown the pattern?"

- **Why:** The assignment says "pattern lengths of 1, 2 and 3" but does not define length. The 2023 paper only defines length 1. The three readings disagree: "BLOOD then BLOOD" has two steps but only one activity.
- **If we guess wrong:** Every set sits in the wrong group, and we redo all runs (about 5.6 hours per log).
- **Until then:** Length is the number of different activities, at most three.

**Q2. The fourth log (office hour 1).** "The description says 'the three BPI11 datasets', but the repository also has f4. Is f4 out?"

- **Why:** The description says three logs, the repository has four. The paper's figures use only f1–f3.
- **If we guess wrong:** We compute for hours on a log nobody wants, or we miss one she expects.
- **Until then:** We use f1–f3 only.

**Q3. Computing budget.** "(a) For length 1, must we test every activity, as in Figure 3 of your paper, or only the selected ones? (b) May we fix the random seed? (c) May we use fewer repeats or folds if time gets tight?"

- **Why:**
  - *(a)* For single activities, the original code tests all of them. f1 has 164. At 50 rounds each and about 7 seconds per round, that is about 16 hours for f1 alone. Ten selected activities take about one hour. (Our own multiplication, not a measured time.)
  - *(b)* The original code has no fold seed, so adding one is a change.
  - *(c)* If time runs out, we want to know what we may cut.
- **If we guess wrong:** We spend nights on a part nobody asked for, or we cut a part she expects.
- **Until then:** We test only the selected activities. We always set a seed. We run the full settings (10 repeats, 5 folds) overnight.

### Follow-ups 2: reproduce or repair

**Q4. As it is, or fixed (office hour 1).** "Your code scores on the training part, lets moves pile up, sometimes breaks the order rule, and has unseeded folds. Should we copy this? Or fix it, score on the held-out part, and report both?"

- **Why:** The paper mentions none of these four points; we found them in the code (section 7). We need to know what she counts as "reproduced".
- **If we guess wrong:** Both choices have a risk. If we copy the code, our numbers rest on known errors. If we fix it, our plots will not match the paper's. Then it may look as if we failed to reproduce it.
- **Until then:** We build both modes. Main results come from fixed mode, plus one faithful run per log.

**Q5. Versions and figure scripts.** "Which xgboost version did you use? Can you share the scripts behind Figures 3 and 4?"

- **Why:** The repository lists no package versions. Default settings of xgboost can change between versions, so our model may differ a little from theirs. Also, no script in the repository draws Figures 3 and 4 as they look in the paper, so we must guess the layout.
- **If we guess wrong:** Only small details change. That is why this question can wait.
- **Until then:** We use xgboost 3.4.1 with its defaults. We rebuild the figures as closely as we can.

### Follow-ups 3: the Apriori side

**Q6. Sets per length, and min_support (soon).** "(a) Your code takes one top 10 of sets with two or more activities. May we take a top 10 per length 1, 2 and 3? (b) Which min_support did you use? At 0.5 we get only 5 sets on f3, but your Figure 5 shows 10."

- **Why:**
  - *(a)* Comparing length by length is the heart of the project (section 6).
  - *(b)* f3 has too few frequent sets. At 0.5 it has only 5 sets of length 1, 4 of length 2 and 1 of length 3. So we must lower min_support, and we would like to use the authors' value.
- **If we guess wrong:** The Apriori half tests other sets than she expects, and we run it again.
- **Until then:** We take a top 10 per length, ranked by support. We use min_support 0.5 for f1 and f2. For f3 we start with 0.4. This is a guess: 0.49 gives 10 mixed sets, but we need 10 per length (section 12 tests this). We state our values in our own paper.

### Follow-ups 4: the IMPresseD side

**Q7. Whole front, or a fixed number (office hour 1).** "Should we use the whole Pareto front, or a fixed number of sets per length like Apriori? If fixed, how do we choose?"

- **Why:** Apriori always gives 10 sets per length. The front has no fixed size. An early probe gave different sizes per log; we have not re-checked it.
- **If we guess wrong:** One strategy tests many more sets than the other. Then we cannot tell if a difference comes from better sets or just from more sets.
- **Until then:** We take 10 sets per length. We take the front first. If it has fewer than 10 patterns, we remove them, compute the front of the rest, and add those. When we must choose, the higher outcome interest wins. We also report the real front size.

**Q8. Patterns first, or sets first (soon).** "Should we select patterns first and then turn them into sets? Or score the sets themselves?"

- **Why:** "BLOOD then CONSULT" and "CONSULT then BLOOD" are two patterns with different scores. But both become the same set. If we select patterns first, the set gets in when one of its patterns is good. If we score the sets, the set gets one new score and other sets may win.
- **If we guess wrong:** We test other sets than she means, and we redo the IMPresseD half.
- **Until then:** We select patterns first. We keep the best pattern per set.

**Q9. Where to find the patterns (office hour 1).** "Should we search for patterns on the whole log, as your Apriori step does? Or only on training data, as in your 2023 paper?"

- **Why:** IMPresseD uses the label. On the whole log, the labels of held-out patients influence which sets we pick. This is **leakage**: information from held-out patients slips into the method and makes results look too good. Searching inside each fold avoids it. But then every fold has its own sets, and we cannot draw one box per set over all five folds.
- **If we guess wrong:** Our results look better than they are, or we must rebuild how we collect results.
- **Until then:** We use the whole log for both strategies, and we say openly that this leaks. One extra f1 run uses training data only.

**Q10. Our own version (soon).** "May we write our own simplified IMPresseD, with automatic selection and no expert?"

- **Why:** Section 8 explains why we want our own version. But it will differ from the original in small details.
- **If we guess wrong:** Our "IMPresseD" results may not count as IMPresseD. So we ask before we build it.
- **Until then:** We write our own. The original automatic function is only a cross-check.

**Q11. Settings.** "We plan three settings. (a) We read each trace as a simple chain, in file order. (b) 'Eventually followed by' may skip at most 3 events. (c) Case distance uses age, diagnosis, treatment code, diagnosis code and specialism code. Any objection?"

- **Why:** Each setting changes which patterns reach the front.
  - *(a)* We ignore parallel events (section 8).
  - *(b)* In "BLOOD eventually followed by SCAN", at most 3 other events may lie between the two. The 3 is our own guess; neither the paper nor the code gives a value.
  - *(c)* Case distance needs facts about the patient. We chose these five columns. ("Diagnosis" and "diagnosis code" are two separate columns.)
- **If we guess wrong:** Our sets differ from what her method would give.
- **Until then:** We use the three settings as stated.

### Follow-ups 5: comparing and reporting

**Q12. Comparison plan (soon).** "We plan six comparisons: set overlap, importance per activity, ranking agreement, box plots side by side, location against existence importance, and stability over seeds. Is that what you expect?"

- **Why:** The assignment only says to check "if and how" the importance changes. It names no measure.
- **If we guess wrong:** An analysis she expects is missing, and that is hard to add in the last week.
- **Until then:** We do all six.

**Q13. Existence importance.** "Must we include existence importance? If yes, as a yes/no column or a count? Scored with F1 or accuracy?"

- **Why:** It is a whole extra part to build and run. And the paper and the code disagree. The paper describes a yes/no column (is the set in the trace?) and the F1 score. The script uses a count (how many times the set is in the trace) and passes no scoring option. With our scikit-learn version that means accuracy (the share of correct predictions). We assume the same held for the authors' version.
- **If we guess wrong:** We cannot compare our existence numbers with the paper's.
- **Until then:** We include it, with counts and weighted F1. We note that this differs from the original.

**Q14. Names.** "(a) In the data, label 1 means the rule is satisfied. Which label should we call 'positive'? (b) Is there a table from activity codes to names?"

- **Why:**
  - *(a)* The paper that created these labels calls label 1 "rule violated". One figure in the 2024 paper shows label 0 as "positive". So the sources do not agree.
  - *(b)* Without names, we can say that a code is important, but not what it means medically. Example: the code 376400 appears in 335 patients of f1, and 99.7% of them have label 1. We think it is the CEA tumour-marker test, but this is not confirmed. If it is such a test, it almost gives the label away.
- **If we guess wrong:** A sentence like "an early BLOOD leads to a positive result" says the opposite of the truth.
- **Until then:** We write "rule satisfied" and "rule not satisfied", never "positive".

## 12. What we are testing ourselves

Not every open point needs the supervisor. These we can check ourselves. The tests are running now; we will add the results to this brief.

- **Runtime on all three logs.** We only timed a few rounds on f1. *Why:* the timeline depends on it.
- **The original IMPresseD automatic mode on f1, from start to end.** So far we only know that it loads. We note the run time and the front size per length. *Why:* it feeds Q7 and Q10.
- **Several min_support values** (0.5, 0.49, 0.45, 0.4). *Why:* we want the highest one that gives 10 sets per length (Q6).
- **Faithful against fixed mode on f1.** *Why:* to see how much the problems of section 7 change the results.
- **Three seeds.** *Why:* to see whether the rankings stay similar.
- **Install on the Mac.** *Why:* the setup is tested on Windows only.

## 13. Setup and where things are

How to get the code running on your laptop.

Use Python 3.12, the version we tested. The shared `project/` folder has a `requirements.txt` with these tested versions:

```
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
```

Then make a virtual environment (a private package folder) and install:

```
# Windows (tested)
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

# macOS (not tested yet)
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On a Mac, xgboost may also need the OpenMP runtime (`brew install libomp`). We have not tested this either.

The shared project folder has:

- `external/PermutationLocationImportance/`: the 2024 code. Start with `CrossValidation_ProcessPermutation.py` and `tools.py`. Its `datasets/` folder holds our logs: `BPIC11_f1_trunc36.csv`, `BPIC11_f2_trunc40.csv`, `BPIC11_f3_trunc31.csv`.
- `external/InteractivePatternDetection/`: the 2023 code. Do not use its `requirements.txt`; it does not install on Python 3.12. Never import `GUI_IMPresseD_tool.py`; it opens a window at once.
- `scripts/scout/`: our test scripts and two reports (setup, labels). `smoke_2024c.py` is a small check of the original method on f1 with only the crash fix: one fold, 2 sets, 1 repeat, scored on the training fold.

Careful: the scout scripts were written on one laptop and still contain an absolute path. Before you run one, change the `REPO` line at the top to the shared `external/PermutationLocationImportance` folder. For `smoke_2023.py`, use `external/InteractivePatternDetection`. The `_fc_*.py` checks need the same change.

Both repositories contain a file named `tools.py`, so never mix them in one folder.

## 14. Mini glossary

- **Activity:** a type of step, such as BLOOD.
- **Event:** one row of the log: one activity of one patient.
- **Trace:** the events of one patient, in order.
- **Event log:** the whole data file.
- **Label:** the yes/no answer of a patient; 1 means "rule satisfied".
- **Activity set:** a few activities, without order. The code says "itemset".
- **Length:** the number of different activities in a set (our working definition).
- **Strategy:** the way to pick the sets: Apriori or IMPresseD.
- **Apriori:** finds sets that occur together in many traces.
- **Support:** the share of patients whose trace contains a set. `min_support` is the lowest support Apriori keeps.
- **IMPresseD:** finds small ordered patterns and scores them, also with the label.
- **Pattern:** a few activities with an order, like "BLOOD then CONSULT".
- **Pareto front:** the patterns that are not beaten. A pattern is beaten when another one is at least as good on all three scores and better on at least one.
- **Location:** the paper's word for position.
- **Index-based encoding:** one 0/1 column per pair of position and activity.
- **XGBoost:** the prediction model; it builds many small decision trees.
- **Fold, cross-validation:** the patients are split into five folds; each fold is held out once.
- **Held-out data:** patients the model did not train on.
- **Weighted F1:** the model's score, from 0 to 1.
- **Baseline score:** the score before we move anything.
- **Location importance:** the score drop after moving a set.
- **Existence importance:** the score drop after shuffling the column "is the set there".
- **Box plot:** a line at the middle value, a box around the middle half of the numbers.
- **Faithful mode, fixed mode:** our code behaves like the original, or with its errors repaired.
- **Leakage:** information from held-out patients slips into the method.
- **Seed:** a fixed number that makes random steps repeatable.
