# Explainable Predictive Process Monitoring — a from-zero guide for Group 11

**Course:** Process Mining (JM0210/JM0211), TU/e · **Group 11 topic:** Explainable Predictive Process Monitoring · **Supervisor:** Laura Genga · **Written:** 1 October 2026

## How to read this guide

This document was built from the course material in `coursematerial/`, the two papers by Vazifehdoostirani et al. (2023, 2024), the authors' public code (both repositories are copied into `project/external/`), the assignment rubric and templates, and a test installation on this laptop. Every part was drafted and then checked line by line against those sources by a second reader; the checks that involved data or code were actually executed (the scripts are in `project/scripts/scout/`).

- **Part 1** is the course itself (weeks 1–5), for the exam and for vocabulary.
- **Part 2** explains, from scratch, every concept the project needs (machine learning, encodings, XGBoost, permutation importance, Apriori, IMPresseD patterns).
- **Part 3** walks through the 2024 paper and its code step by step; **Part 4** does the same for the 2023 paper and describes the variant we must build.
- **Part 5** is the plan: deliverables, rubric points, a dated work plan, team split, environment setup and downloads.
- **Part 6** collects every open question in one place: what only you can answer, what to ask the supervisor, and what we can settle by running an experiment.

Each concept is introduced three times: an ELI5 analogy (one or two sentences a child could follow), the precise definition, and a tiny example. Skim the analogies first, then come back for the definitions when you need them.

**Tag legend.** `[UNVERIFIED]` = stated somewhere but not confirmed in a primary source; `[ASSUMPTION]` = our own choice or inference, not from the sources. Nothing else in the guide is a guess. All tagged items are listed again in Part 6.

## Table of contents

- Part 1 — The course in plain English (weeks 1–5, grading, exam)
  - 1.1 How the course works
  - 1.2 Week 1 — BPM and process mining
  - 1.3 Week 2 — Petri nets and workflow nets
  - 1.4 Week 3 — Process trees, dependency graphs, causal nets, quality
  - 1.5 Week 4 — Alpha miner and Heuristic miner
  - 1.6 Week 5 — Putting process mining into code with PM4Py
  - 1.7 Exam radar — exercise types in the instruction sheets
  - 1.8 Bridge to the Group 11 project
- Part 2 — Concepts you need for the project, explained from scratch
  - 2.1 Predictive process monitoring (PPM) vs. discovery vs. conformance
  - 2.2 Outcome-oriented prediction and binary labels
  - 2.3 What the BPIC11 labels mean
  - 2.4 Prefix, ongoing case, truncation — and `All_prefixes = False`
  - 2.5 Feature encoding: index-based vs. binary vs. frequency
  - 2.6 Supervised learning — from "five lines" to a proper picture
  - 2.7 XGBoost and why it needs fixed-length vectors
  - 2.8 Train/test split and stratified K-fold cross-validation
  - 2.9 Metrics: accuracy, precision, recall, F1, weighted F1, "baseline"
  - 2.10 Explainable AI: post-hoc, model-agnostic, global/local, factual/counterfactual
  - 2.11 Permutation feature importance (the blindfold test) and why it fails for position
  - 2.12 "Location" = position index in the trace; the Table 1 example
  - 2.13 Frequent itemsets, support, Apriori (shopping baskets)
  - 2.14 IMPresseD: process patterns, extension, interest functions, Pareto front
  - 2.15 Information gain / mutual information
  - 2.16 Rank-comparison measures for the two strategies
  - 2.17 Glossary
- Part 3 — The 2024 paper and its code, step by step
  - 3.1 The research question and the toy motivation (Table 1)
  - 3.2 Figure 2: the pipeline and the three contributions
  - 3.3 Step by step: ELI5, precise definition, code location
  - 3.4 Parameters
  - 3.5 Evaluation and results of the paper
  - 3.6 Paper vs code: discrepancy table (each entry re-verified in the code)
  - 3.7 Runtime expectations
  - 3.8 Reading map of the code
- Part 4 — The 2023 paper (IMPresseD) and how to build the variant
  - 4.1 IMPresseD in plain words
  - 4.2 The 2023 evaluation in brief
  - 4.3 The variant, concretely
  - 4.4 Design decisions
  - 4.5 Risks and performance pitfalls in the 2023 code (verified by reading)
  - 4.6 Proposed architecture (proposal, not from the sources)
  - 4.7 Experiment loop (pseudocode)
- Part 5 — Deliverables, detailed work plan, team split, environment & downloads
  - 5.1 Deliverables, one by one, with what the rubric rewards
  - 5.2 Work plan, Thu 1 Oct → Thu 12 Nov 2026
  - 5.3 Team split (4–5 members) and cross-review rule
  - 5.4 What to download / set up
  - 5.5 Risk register
  - 5.6 README skeleton and requirements.txt to start from
  - 5.7 Definition of done
- Part 6 — Open questions & uncertainties register
  - 6.0 Already sent to the supervisor (do not re-ask — collect the answers)
  - 6.1 Table A — removed
  - 6.2 Table B — Questions to ask the supervisor (Laura Genga)
  - 6.3 Table C — Things we can resolve ourselves by experiment
  - 6.4 (D) Corrections the fact-checkers made to earlier drafts
  - 6.5 Counts and how to keep this register alive

---

## Part 1 — The course in plain English (weeks 1–5, grading, exam)

Course "Process Mining – JM0211" (slides); the short-paper and poster templates print "1JM0211 Process Mining" (ShortPaper_GroupX main.tex line 12; Poster_GroupX.pdf header). TU/e, Information Systems group (IE&IS). Team (week1 overview p.3): Karolin Winter (course coordinator; organisational questions, process modelling, discovery, conformance, NLP, prescriptive monitoring), Laura Genga (process modelling, discovery, predictive process monitoring, object-centric PM; Group 11 contact, l.genga@tue.nl per Assignment_Topics p.11), Bart Verhoef (Canvas forum, PM4Py lecture). Textbook: van der Aalst, *Process Mining: Data Science in Action* (Springer 2016) plus selected chapters/papers (overview p.14). Prerequisites named on p.6: basic data mining and its evaluation, Python, teamwork/presentation skills.

### 1.1 How the course works

**Grading** (overview p.7; Rubric.xlsx). Final grade = 70 % written exam (individual, 120 min, retake possible, **minimum 5.0 in the exam to pass the course**) + 30 % group assignment (**no retake**). Every group must "explain your assignment, in particular your implementation, to the lecturers after the poster sessions". The technology statement is mandatory: not providing it = 0 points for the whole assignment. Suspected AI use beyond the allowed level, or fraud, goes to the exam committee. Assignment split (Rubric.xlsx): short paper 35 % (S1 motivation & background 4, S2 conceptual approach **10**, S3 implementation 3, S4 evaluation design & results **10**, S5 limitations & future work 4, S6 quality/structure/appearance 4); poster & pitch 25 % (P1 content 2, P2 structure 4, P3 timing 4, P4 terminology & style 3, P5 visualisations 4, P6 readability & formatting 3, P7 Q&A **5**); implementation 30 % (code structure & understandability 10, installation instructions 3, run instructions 3, **correctness & reproducibility 14**); peer reviews 10 % (short-paper review of one partner group 4, group task; poster session 1 review of two groups 3 and poster session 2 review of two groups 3, both individual). ELI5: the exam is the big boss fight you face alone; the assignment is the team quest, and the technology statement is the ticket you must hand in or the quest does not count at all.

**Schedule** (overview p.8, table rendered from the slide; lectures 13:45–15:30, instruction sessions 15:45–17:30 except week 3 instructions 16:00–17:30; room Zaal MDB 0.10; no weekday is printed on the slide):

| Wk | Date | Lecture |
|---|---|---|
| 1 | 3 Sep | Course overview; intro to BPM & process mining |
| 2 | 10 Sep | Process modelling: Petri nets & reachability analysis |
| 3 | 17 Sep | Process modelling: process trees, dependency graphs, causal nets |
| 4 | 24 Sep | Process discovery: Alpha & Heuristic miner |
| 5 | 1 Oct | Putting process mining into code with PM4Py |
| 6 | 8 Oct | Process discovery: Inductive Miner; conformance checking with footprint matrices |
| 7 | 15 Oct | Conformance checking: token-based replay, alignments |
| 8 | 22 Oct | Predictive process monitoring & explainability (Group 11's topic) |
| 9 | 29 Oct | Prescriptive process monitoring & optimisation |
| 10 | 5 Nov | Poster session 1 (presenters: groups 1–6; audience: all students; Group 11 members review two posters individually) |
| 11 | 12 Nov | Poster session 2 (presenters: groups 7–12, incl. Group 11) + assignment explanations 15:45–17:15 |
| 12 | 19 Nov | Process mining with textual sources |
| 13 | 26 Nov | Object-centric process mining |
| 14 | 3 Dec | Exam preparation (lecture + instructions 13:45–17:30) |

**Group 11 dates** (Assignment_Topics_Summary_and_Important_Dates, Group 11 block): in-person office hour 17 Sep 15:45–16:00 (past); 1st online office hour 2 Oct 15:00–15:30 (Teams), questions due 1 Oct 15:00 (already sent on 1 Oct — see §5.2 and §6.0; the sheet's rules were: every group member in CC, and the Teams link is sent only after the questions have been submitted, which they now have); 2nd office hour 26 Oct 10:30–11:00, questions due 25 Oct 15:00; short-paper draft for peer review 28 Oct 23:59; peer feedback on the partner group's draft 3 Nov 23:59; final assignment (all deliverables) 9 Nov 23:59; poster session 12 Nov 13:45–15:30; assignment explanation to the lecturers 12 Nov 16:30–16:45.

Exam date: not in the materials. [ASSUMPTION] All dates above may belong to a different edition than the one the student is in: the instruction sheets are dated "September 25, 2025" / "September 30, 2025" (Instruction3/4 headers), the template zips carry 2025-08-22 file dates, but Instruction4_solutions says "September 30, 2026". The slide-8 dates fall on Thursdays in 2026 (3 Sep 2026 is a Thursday, as is today, 1 Oct 2026) and on Wednesdays in 2025 [computed, not from the materials]; the slide prints no weekday. Confirm every date on Canvas.

**AI policy** (overview p.13). Exam: "AI - Index - Level 1" = no AI. Assignment: "AI - Index - Level 4": AI may support editing, brainstorming/inspiration, code suggestions, calculations/visualisation, transcription, but "It is not allowed to use AI to generate final products." Literature should be found via academic databases (Web of Science is named), fabricated or missing references "will not be tolerated", and no sensitive or confidential data may be entered into AI tools. Every tool use must be documented in the technology statement using the fixed template on the slide: "During the preparation of this work, I/We used [NAME TOOL / SERVICE / VERSION OF AI TOOL] in order to [REASON]. The following parts of the assignment were affected/generated by AI tool usage: [...]. After using this tool/service, [NAME STUDENT(S)] evaluated the validity of the tool's outputs, including the sources that generative AI tools have used, and edited the content as needed. As a consequence, [NAME STUDENT (S)] take(s) full responsibility for the content of their work."

**Instruction sessions** = the hands-on second block each week (exercise sheet + solutions): week 1 in a tool (Disco), weeks 2–4 pen-and-paper, week 5 in code (PM4Py, §1.6; no official solutions file exists for that sheet). [ASSUMPTION] They are the best available predictor of exam-question style (§1.7); the materials do not say so explicitly. **Tools** (lecture p.40; week-1 sheet): Disco (Fluxicon academic programme, recommended for week 1), PM4Py (Python library), Celonis academic, ProM.

### 1.2 Week 1 — BPM and process mining

**Business process.** ELI5: a recipe a company runs again and again, each run slightly different. Definition (intro p.4, from Dumas et al.): a series of events, activities and decisions that follow a particular order, can involve actors and physical or informational objects, and lead to a certain outcome.

**BPM lifecycle** (p.6, figure from Dumas et al.). ELI5: a wheel you keep turning to improve the recipe. Process identification → process discovery (as-is process model) → process analysis (insights on weaknesses and their impact) → process redesign (to-be process model) → process implementation (executable process model) → process monitoring (conformance and performance insights) → back to discovery. The slide marks discovery (solid box), analysis and monitoring (dashed boxes) as "Phases we will address and touch upon in this course". Predictive monitoring (Group 11) sits in *monitoring*.

**Three goals of process mining** (p.25). ELI5: discovery = draw the map from GPS traces; conformance = compare map and traces; enhancement = add traffic info to the map. Discover: log → model ("How does our process look like?", weeks 4 and 6). Compare: log + model → "Does the modeled behavior align with the observed behavior?" (weeks 6–7). Enhance: "How to improve the existing model?". Beyond ex-post (p.39): train a prediction model on a historical log, feed it a running case ⟨A,B,C,…⟩ from an event stream and ask: next activity? remaining time? **outcome of this case?** (Group 11's question) or which activity to execute next to optimise the outcome (prescriptive, week 9).

**Event log anatomy** (p.21). ELI5: a diary where every line says who did what, when, for which customer. The slide's table has columns Case ID, Event ID and event attributes Timestamp, Activity, Resource. **Case** (the slide labels it "case/trace"): one run of the process, e.g. one patient. **Event**: one row. Other attributes appear in real logs, e.g. `Costs` in the XES excerpt (p.23) and in running-example.csv. Case attributes are constant within a case (`case:creator` = "Fluxicon Nitro" for all rows of every case in running-example.csv; recomputed), event attributes vary per row. [ASSUMPTION] "Minimum for discovery = case ID + activity + an ordering" is a reading of p.21 ("we are often (only) interested in the control-flow perspective and therefore use a short notation ... containing only event labels"), not a sentence on the slide.

**Trace, variant, multiset.** ELI5: a trace is one patient's full story told in order; a variant is a story shape that several patients share; a multiset is the pile of all stories with a count written on each shape. For control flow the log shrinks to activity sequences, one trace per ⟨⟩ (p.21: L = [⟨prepare sample, put in centrifuge, centrifugation, prepare results⟩, …]). A **variant** is a distinct sequence shared by several cases (Disco's "variation filter" in the week-1 solutions keeps cases whose sequence is shared by at least 32 cases). Week 4 p.8: "An event log is a multiset of traces (same trace may appear multiple times). A trace is a sequence of activity names." Example (week4 p.9): L1 = [⟨a,b,c,d⟩³, ⟨a,c,b,d⟩², ⟨a,e,d⟩] = 3 + 2 + 1 = 6 cases, 3 variants.

**XES vs CSV** (p.22–24). ELI5: XES is a box with standard labels on it (an XML file) that every process-mining tool opens as-is; CSV is a plain spreadsheet where you must first tell the tool which column is the case, which is the activity and which is the time. XES = eXtensible Event Stream, an XML schema and "unified grammar for logging system behaviour in information systems" (p.22 cites Günther, XES Standard Definition, xes-standard.org, 2009; [UNVERIFIED from the course materials] that it later became IEEE standard 1849-2016). Structure in the excerpt (p.23): `<log>` ⊃ `<trace>` ⊃ `<event>`; extension keys `concept:name`, `org:resource`, `time:timestamp`, plus plain attributes `Activity`, `Resource`, `Costs`, `creator`. Standard extensions: Concept, Time, Organizational, … CSV: flat table, one event per row; on import "you have to select which column corresponds to the event label, event id, trace id, etc." (p.24). OCEL = object-centric event logs (p.24; week 13). running-example.csv already uses the PM4Py-style headers `case:concept:name`, `concept:name`, `org:resource`, `time:timestamp` (plus duplicate columns `Activity`, `Resource`, and `Costs`, `case:creator`; recomputed).

**Directly-follows graph (DFG).** ELI5: one dot per activity, an arrow a→b each time b came right after a, counts on the arrows. The term is not used in the week-1 slides. The week-1 solutions show a "Process model discovered with Disco" where "with the sliders you can show/hide less frequent paths" [ASSUMPTION: that map is a frequency-annotated directly-follows graph; the solutions do not name it so]. Instruction 3 Ex 2.1 asks for exactly this object: "a dependency graph whose edges model directly-follow relations and annotate each edge with the frequency of the relation".

**Worked example, running-example.csv** (42 events, 6 cases, 8 activities, 6 resources Ellen/Mike/Pete/Sara/Sean/Sue; recomputed with pandas). Legend (book letters): a register request, b examine thoroughly, c examine casually, d check ticket, e decide, f reinitiate request, g pay compensation, h reject request. Rows are not sorted by case (the file starts with case 3): sort by case, then timestamp.

| Case | Trace | Σ Costs |
|---|---|---|
| 1 | ⟨a,b,d,e,h⟩ | 950 |
| 2 | ⟨a,d,c,e,g⟩ | 950 |
| 3 | ⟨a,c,d,e,f,b,d,e,g⟩ | 1850 |
| 4 | ⟨a,d,b,e,h⟩ | 950 |
| 5 | ⟨a,c,d,e,f,d,c,e,f,c,d,e,h⟩ | 2750 |
| 6 | ⟨a,c,d,e,g⟩ | 950 |

Six cases, six variants (every trace unique). Activity counts: d 9, e 9, a 6, c 6, b 3, f 3, g 3, h 3. DFG (recomputed): a→b 1, a→c 3, a→d 2, b→d 2, b→e 1, c→d 4, c→e 2, d→b 1, d→c 2, d→e 6, e→f 3, e→g 3, e→h 3, f→b 1, f→c 1, f→d 1; start {a:6}, end {g:3, h:3}. Reading: b/c and d occur in both orders (concurrency), f loops back (rework), every case ends in g or h — a binary **outcome** a predictor could learn. In-lecture exercises: p.27–31 build a model for ⟨A,B,C,D,F⟩, ⟨A,C,B,E,F⟩, ⟨A,E,F⟩ (the slide-31 model: A, then XOR between "B and C in parallel" and a skip, then XOR D/E, then F) and ask "Is the provided set of traces, i.e., event log, complete for this model?" [ASSUMPTION, derived: no — the model also allows e.g. ⟨A,D,F⟩ and ⟨A,C,B,D,F⟩, which the log never shows]. P.32: A, AND(B,C), D with ⟨A,B,C,D⟩ ✓, ⟨A,C,B,D⟩ ✓, ⟨A,B,D,C⟩ ✗. P.33 (cookie order model: A, then in parallel {loop B→C→D while "quality ok?" = no} and E, then XOR F express / G standard): trace 1 ✓, traces 2–4 ✗ (marks on the slide; reasons are not written there — e.g. trace 3 ⟨A,E,B,C,D,F,G⟩ takes both XOR branches).

### 1.3 Week 2 — Petri nets and workflow nets

**Petri net.** ELI5: a board game — the circles (places) are the fields of the board, black counters (tokens) sit on the fields, and each box (transition) is one move: it takes counters off its input fields and puts counters on its output fields. Definition (week2 p.5–6): N = (P, T, F): **places** (circles; "passive elements", the slide calls them "events" — think of them as conditions or waiting states), **transitions** (squares; "active elements; tasks"), **arcs** F only place→transition or transition→place. One word of warning: the slide's Petri-net "event" (its label for a place) has nothing to do with an event-log event (one row of the diary in §1.2) — in a Petri net a place is a condition that holds while a token sits in it, and the things that happen are the transitions. Slide-5 example: P = {p1,p2,p3,p4}, T = {A,B,C,D}, F = {(p1,A),(A,p2),(p2,B),(p2,C),(B,p4),(C,p3),(p3,D),(D,p4),(p4,A)}. Preset •t = input places, postset t• = output places (p.6: •A = {p1,p4}, A• = {p2}). **Marking** (p.7–8): tokens (black dots) are the state; M counts tokens per place, written as an array M = [2,0,1,1] or as a multiset [p1², p3, p4]; (N, M) is a marked Petri net. **Place-Transition nets, enabling and firing** (p.10–11, exam-critical): every arc has a weight ≥ 1 (default 1); "A transition t is enabled, if and only if each input place p of t contains at least the number of tokens defined as the weight of the connecting arc"; firing consumes those tokens and produces the output-arc weights in each output place; firing takes no time. Output places play no role in enabling (the rule only mentions input places; no capacity limit exists in this definition). With weight 2 on p1→A and M = [2,0,1,1]: A (p1 has 2 ≥ 2, p4 has 1) and D (p3 has 1) are enabled; B and C are not (p2 empty) (p.12).

**Reachability analysis** (p.12–21). ELI5: list every board position you can reach and which move leads where. Table: one row per marking, one column per transition, "x → Mk" where enabled; write each marking once; row order and numbering may differ (p.21). Lecture example (net of p.5, weight 2 on p1→A; recomputed and identical to p.21):

```
M1 2011  A->M6  D->M2      M5 0011  D->M4
M2 2002  A->M3             M6 0110  B->M5 C->M7 D->M3
M3 0101  B->M4  C->M5      M7 0020  D->M5
M4 0002  nothing enabled = deadlock
```

The **reachability graph** has one node per marking and one labelled edge per firing (p.25–27; traffic-light example on p.25: 3 markings, "All states are reachable from all other states. The Petri net does not contain any deadlocks.").

**Properties** (p.28, verbatim definitions): **k-bounded** — "no place ever holds more than k tokens"; bounded iff some k exists; **safe** = 1-bounded; **deadlock free** — "at every reachable marking at least one transition is enabled"; **live** — a transition t is live "if from every reachable marking it is possible to enable t"; the net is live if each transition is; **reversible** — "for all reachable markings it holds that the initial marking can be reached again". Read-off from the table/graph: largest token count → k; a row with no enabled transition → deadlock; every node has a path back to M1 (strongly connected graph) → reversible. Live ⇒ deadlock-free, not conversely: week-2 solutions Ex 4c (last page, p.7 of the PDF): "False, the Petri net below is deadlock free but not live since transition A is not live" — the figure is p1(•)→A→p2 with B as a self-loop on p2: A fires once and is then dead, B can fire forever. Lecture example above (derived): 2-bounded (max entry 2), not safe, not deadlock-free (M4), not live, not reversible (M1 is never reached again).

**XOR vs AND** (p.29). ELI5: a circle with two arrows out is a fork in the road (pick one); a square with two arrows out splits you into two teams (both go). A branching place = XOR-split / merging place = XOR-join; a branching transition = AND-split / joining transition = AND-join (the slide maps these to BPMN gateways).

```
XOR:  (p1)->[A]->(p2)->[B]->(p4)      AND:  (p1)->[A]->(p2)->[B]->(p4)->[D]->(p6)
                 (p2)->[C]->(p4)                 [A]->(p3)->[C]->(p5)->[D]
```

A **silent transition τ** exists only to route (skip, loop back) and is "not directly observable in the log" (p.9, 30). Nets can be structurally different but trace-equivalent (p.30).

**Workflow net** (p.33, verbatim): a (labelled) Petri net with "a distinguished start place s (source place) that has no incoming arcs", "a distinguished end place e (sink place) that has no outgoing arcs", "every place and every transition being on a direct path between s and e". [ASSUMPTION: textbook convention, not written on p.33] the initial marking is one token in s. Slide-33 quiz (derived from the figures): net 1 is a WF-net; net 2 is not (p1 has an incoming arc from A, no sink); net 3 is not (three source places); net 4 is structurally a WF-net.

**Soundness** (p.34). ELI5: the game always finishes cleanly, every move is useful, counters never pile up. Quoted: a WF-net is sound iff
- "safeness: places cannot hold multiple tokens at the same time"
- "proper completion: if the sink place is marked, all other places are empty"
- "option to complete: it is always possible to reach the marking that marks just the sink place"
- "absence of dead parts: for any transition there is a firing sequence enabling it."

Slide-34 net (figure: p1→A, p1→C; A→p2, A→p3; C→p3; B consumes p2 and p3, produces p4; one token in p1): firing C leaves [p3] and B can never fire → option to complete violated → not sound (derived; the slide only asks the question). "Is a WF-net" ≠ "is sound".

### 1.4 Week 3 — Process trees, dependency graphs, causal nets, quality

Motivation (week3 p.5): "Petri nets (WF-nets) are not guaranteed to be sound" (livelocks, deadlocks) and "PD algorithms returning graph-based models tend to return unsound models"; common solutions: "Return models sound by construction" (trees) or "Return models with a relaxed semantics" (C-nets).

**Process tree.** ELI5: a recipe as nested boxes — "in order", "pick one", "all, any order", "repeat" — that cannot jam because every box has one way in and one way out. Def. 3.13 (p.13, book definition shown on the slide): leaves = activities a ∈ A or τ; operators ⊕ = {→, ×, ∧, ⟲}; if n ≥ 1 and ⊕ ∈ {→, ×, ∧} then ⊕(Q1,…,Qn) is a tree; ⟲(Q1,…,Qn) needs n ≥ 2. Def. 3.14 (p.15): L(a) = {⟨a⟩}, L(τ) = {⟨⟩}; → concatenates the children's languages, × takes their union, ∧ interleaves (shuffles) them; L(⟲(Q1,…,Qn)) = {σ1·σ1'·σ2·σ2'···σm | m ≥ 1, every σj ∈ L(Q1), every σj' ∈ ∪_{i≥2} L(Qi)} — starts and ends with the **do** part Q1; the redo parts Q2…Qn are "in exclusive relations" (p.12). Running example (p.8–13, book figure): →(a, ⟲(→(∧(×(b,c), d), e), f), ×(g,h)); p.11 shows trace fragments like ⟨…b,d,e,f,b,d,e…⟩ and ⟨…b,d,e,f,d,c,e…⟩.

**Sound by construction** (p.14): each operator maps to a WF-net block with one entry and one exit (→ = chain; × = all children share the entry and exit place; ∧ = τ-split, children in parallel, τ-join; ⟲ = τ in, the do child forward and the redo children backward on a cycle, τ out); nesting such blocks always yields a sound net. **Limits** (p.19–21): subprocess synchronisation (two parallel lines a,b,c and d,e,f with b before e): Q = ∧(→(a,b,c), →(d,e,f)) allows too much; an exact tree "need[s] duplicating activities, e.g." ×(→(∧(→(a,b),d), ∧(c,→(e,f))), →(a,b,c,d,e,f)) (p.20). Long-term dependencies (a…e vs b…f via shared c,d): Q = ×(→(a,c,d,e), →(b,c,d,f)) duplicates c and d; "Not discoverable from process-tree based techniques" (p.21).

**Dependency graph** (p.24–30). ELI5: a sketch of what tends to cause what, with confidence numbers but no rule about choice vs parallel. Nodes = activities, arcs = causal dependencies, "may be annotated with frequency and/or confidence/certainty" (e.g. 0.92); "no AND/XOR splits and joins!" — "each node can be viewed as a 'fuzzy' OR-join & OR-split" (p.28–29); challenge: "Interleavings should not introduce causal dependencies!" (p.28); it is the "starting (or ending) point of many discovering algorithms" (p.30). The frequency-annotated DFG of Instruction 3 Ex 2.1 is its simplest form.

**Causal net (C-net)** (p.32–65). ELI5: a dependency graph where each activity has a menu of allowed input combos and output combos, and a run counts only if every IOU it creates is later redeemed. Elements: a start activity and an end activity (p.42: "Start with the start activity ... End with the end activity ... The start and end activities cannot also happen in-between"), dependency arcs, and per activity sets of **input bindings** I(a) and **output bindings** O(a); the legend on p.32 distinguishes XOR-split/join (separate singleton bindings), AND-split/join (one binding with all arcs) and OR-split/join (several combinations). Node x on p.44–45 has 2 input bindings and 3 output bindings → "Six bindings are possible" (activity binding = input binding × output binding). **Obligations** (p.35–42): when y occurs with input binding X it removes the pending obligations (x,y), x ∈ X, and with output binding Y it creates obligations (y,z), z ∈ Y; "Obligations are like tokens (need to be there in order to be consumed)"; "At the end there should be no remaining obligations" (p.42). Walk-through p.36–41: a with output binding {b,d} creates (a,b),(a,d); b removes (a,b), creates (b,e); d removes (a,d), creates (d,e); e removes both, creates (e,g); g creates (g,z); z removes (g,z), nothing left. **Semantics are declarative** (p.47: "Only valid binding sequences are considered !!!"; p.34: "Provides replay semantics rather than execution semantics, e.g., the moment of choice is not fixed"). The WF-net translation (each binding becomes a silent routing step, p.48–61) "does not need to be sound" — its deadlocking or livelocking firing sequences are simply not valid binding sequences (p.52–53). Booking example (a start booking → b flight, c car, d hotel → e complete): "Twelve valid binding sequences are possible": abe, ace, abde, adbe, acde, adce, abcde, abdce, acbde, acdbe, adbce, adcbe (p.63). Why C-nets (p.65): output of the Heuristic miner; fit BPMN/EPC/YAWL/BPEL; "Able to model XOR, AND, and OR, but no silent steps or duplicate activities needed"; "Avoiding non-sound models".

**Four quality dimensions** (p.68, 75, 99). ELI5: a good map covers the roads you drove (fitness), invents none (precision), still works next week (generalisation), fits on a page (simplicity). Fitness = "ability to explain observed behavior" ("lift"); precision = "avoiding underfitting" ("drag"); generalization = "avoiding overfitting" ("thrust"); simplicity = "Occam's razor" ("gravity"); "several metrics are possible for all four dimensions (will see more later)" (p.99). Running log (p.76): 21 variants, 1391 cases (455 acdeh, 191 abdeg, 177 adceh, … 1 adcefdbefcdefdbeg; sum recomputed = 1391). Examples: the model on p.76–81 scores well on all four; the **non-fitting** single sequence a,c,d,e,h (p.82); the **underfitting** model on p.83–84 — one central place with every activity b–f hanging off it so anything goes between a and g/h [the name "flower model" is literature vocabulary, not on the slide]; the **overfitting** model that enumerates "all 21 variants seen in the log" as separate paths (p.85–86); simplicity: unrolled loops a,c / a,b,c / a,b,b,c … as separate paths (bad, p.94–95) vs one b self-loop (good, p.96–97); p.92–93 "risk of overfitting on 5 example traces". Caveats (p.69–74): no negative examples ("cannot see what cannot happen"), the log "contains only a fraction of possible traces", almost- vs poorly-fitting traces, loops → "often infinitely many possible traces"; the naive classification view defines recall = TP/(TP+FN), precision = TP/(TP+FP) and replay fitness = TP'/(TP'+FN') on the log (p.70–73). **Representational bias** (p.104): "Class of process models that can be discovered by a process discovery algorithm" — determines the search space and limits expressiveness; challenging constructs: concurrency, arbitrary loops, silent actions, duplicate actions, OR-splits/joins, non-free-choice behaviours, hierarchy.

### 1.5 Week 4 — Alpha miner and Heuristic miner

**Footprint relations** (week4 p.9, verbatim). ELI5: look only at what came *right after* what and sort each pair into "a then b", "either order", "never adjacent". "Direct succession: x>y iff for some case x is directly followed by y. Causality: x→y iff x>y and not y>x. Parallel: x||y iff x>y and y>x. Choice: x#y iff not x>y and not y>x." Patterns (p.10–19): sequence a→b; XOR-split a→b, a→c, b#c; XOR-join b→d, c→d, b#c; AND-split a→b, a→c, b∥c; AND-join b→d, c→d, b∥c.

**Eight steps** (p.20–27, α(L)): 1 T_L = all activities; 2 T_I = first activities of traces; 3 T_O = last activities; 4 X_L = pairs (A,B) of non-empty subsets of T_L with a →_L b for all a∈A, b∈B, a1 #_L a2 for all a1,a2∈A, b1 #_L b2 for all b1,b2∈B; 5 Y_L = maximal pairs ("Delete from set X_L all pairs (A, B) that are not maximal!"); 6 P_L = one place p(A,B) per pair in Y_L plus source i_L and sink o_L; 7 F_L = arcs a→p(A,B), p(A,B)→b, i_L→t for t∈T_I, t→o_L for t∈T_O; 8 α(L) = (P_L, T_L, F_L). Note that the # condition is also applied within A and within B for a1 = a2, so an activity with x ∥ x (a self-loop) can never sit in any pair.

**Worked example** L1 = [⟨a,b,c,d⟩³, ⟨a,c,b,d⟩², ⟨a,e,d⟩] (p.9, 18, 28). Successions: a>b, a>c, a>e, b>c, b>d, c>b, c>d, e>d. Footprint (p.18):

|   | a | b | c | d | e |
|---|---|---|---|---|---|
| a | # | → | → | # | → |
| b | ← | # | ∥ | → | # |
| c | ← | ∥ | # | → | # |
| d | # | ← | ← | # | ← |
| e | ← | # | # | → | # |

T_I = {a}, T_O = {d}. X_L = six singleton pairs ({a},{b}), ({a},{c}), ({a},{e}), ({b},{d}), ({c},{d}), ({e},{d}) plus ({a},{b,e}), ({a},{c,e}), ({b,e},{d}), ({c,e},{d}); ({a},{b,c}) is out because b∥c, not b#c. Y_L = the four non-singleton pairs. Net (p.28): start → a; a fills p1 = p({a},{b,e}) and p2 = p({a},{c,e}); b takes p1, c takes p2, e takes both; b fills p3 = p({b,e},{d}), c fills p4 = p({c,e},{d}), e fills both; d needs p3 and p4 → end. After a, either b ∥ c or e alone; language = exactly the log.

**Weaknesses** (p.29–41): (1) implicit (redundant) places — "harmless and can be solved through preprocessing" (p.29, 40); (2) length-1 loops: L7 = [⟨a,c⟩², ⟨a,b,c⟩³, ⟨a,b,b,c⟩², ⟨a,b,b,b,b,c⟩¹] (p.30) gives b>b hence b∥b, so b joins no pair and is disconnected (desired: a self-loop place around b); length-2 loops: L8 = [⟨a,b,d⟩³, ⟨a,b,c,b,d⟩², ⟨a,b,c,b,c,b,d⟩] (p.31) gives b>c and c>b hence b∥c, so c is disconnected (desired: b and c alternate); both "can be solved in multiple ways (change of algorithm or pre/post-processing)" (p.40); (3) non-local dependencies — the places a…d and b…e "are not discovered" (p.32–33), "foundational problem, not specific for Alpha algorithm" (p.40); (4) representational bias: Alpha "cannot discover transitions with duplicate or invisible labels" (p.36–37, 41); (5) the result need not be a sound WF-net: L = [⟨a,b,d,e,f⟩¹⁰, ⟨a,c,e,d,f⟩¹⁰] yields a net that "is not sound (has deadlock)" (p.38); (6) noise ("rare and infrequent behavior") and incompleteness ("too few events") — Alpha ignores frequencies, so one rare trace changes a relation (p.39, 46–47: with outliers, Alpha's net forces "b and c always need to be executed" and lets "d be executed at any time").

**Heuristic miner, two phases** (p.43): "learn a dependency graph by counting frequencies" (settings: thresholds) → "learn splits and joins" → C-net with frequencies, which can be visualised or converted to BPMN/UML/EPC/WF-nets. Counting (p.49–50, verbatim): |a >_L b| = Σ_{σ∈L} L(σ) × |{1 ≤ i < |σ| : σ(i) = a ∧ σ(i+1) = b}| — every adjacent occurrence weighted by the trace's multiplicity; "information loss when frequencies are ignored" (p.49). **Dependency measure** (p.50, verbatim, memorise):

```
|a ⇒_L b| = (|a >_L b| − |b >_L a|) / (|a >_L b| + |b >_L a| + 1)   if a ≠ b
|a ⇒_L a| = |a >_L a| / (|a >_L a| + 1)                              if a = b  (length-1 loop)
```

Range (−1, 1): ≈ 0 means both orders about equally frequent (concurrency), negative = the reverse direction dominates; the +1 makes rare evidence weak: 1 observation → 1/2 = 0.5, 45 observations → 45/46 ≈ 0.98 (p.51 shows "45(0.98)"). **Thresholds** (p.51, 65): "Both need to be above predefined thresholds! Otherwise, no causality!" — draw an arc only if |a>b| ≥ the frequency threshold *and* |a⇒b| ≥ the dependency threshold; procedure p.65: set thresholds, count direct successions, compute dependency measures, draw only arcs meeting both.

**Recomputed examples.** AND-split, 10 cases split evenly (p.52–53): |a>b| = 5, |b>a| = 0 → a⇒b = 5/6 = 0.83; |b>c| = |c>b| = 5 → **b⇒c = (5−5)/(5+5+1) = 0/11 = 0**: no arc between concurrent activities — why > alone is not enough. AND-join is symmetric (p.54–55). Self-loop (p.56–58; ac×3, abc×4, abbc×3): |b>b| = 3 → b⇒b = 3/4 = 0.75; a⇒b = 7/8, b⇒c = 7/8, a⇒c = 3/4 (the a≠b formula would give (3−3)/7 = 0 for the loop, p.57: "self loops are handled differently (otherwise 0)"). Outlier log L = [⟨a,e⟩⁵, ⟨a,b,c,e⟩¹⁰, ⟨a,c,b,e⟩¹⁰, ⟨a,b,e⟩¹, ⟨a,c,e⟩¹, ⟨a,d,e⟩¹⁰, ⟨a,d,d,e⟩², ⟨a,d,d,d,e⟩¹] (p.59–62): direct-succession table |a>b| = 11, |a>c| = 11, |a>d| = 13, |a>e| = 5, |b>c| = |c>b| = 10, |b>e| = |c>e| = 11, |d>d| = 4, |d>e| = 13 (p.59; recomputed) → a⇒b = 11/12 = 0.92, **b⇒c = 0/21 = 0**, a⇒d = 13/14 = 0.93, d⇒d = 4/5 = 0.80, a⇒e = 5/6 = 0.83 (p.61). Thresholds (≥ 2 successions, ≥ 0.7) keep a→e 5(0.83) and the d self-loop 4(0.80) (p.63); (≥ 5, ≥ 0.9) drops both (p.64).

**Length-2 loops.** The slides define only the a = b special case. The alternation measure for "a b a" patterns is attributed to Weijters, van der Aalst & Alves de Medeiros, "Process mining with the HeuristicsMiner algorithm" (2006), cited on p.80 next to "other thresholds can be defined (e.g., long-distance relations)". [UNVERIFIED] Prior analysis gives its form as (|a>>b| + |b>>a|) / (|a>>b| + |b>>a| + 1) with |a>>b| = number of a,b,a sub-sequences; the paper is not in the local materials, so the formula is unconfirmed. Low exam priority.

**Splits/joins → C-net** (p.71–85). Window heuristic (approach 1): for each occurrence of x, look at the k events before it and keep those that are dependency-graph predecessors of x → one input binding; the k events after it, keep successors → one output binding; count distinct sets. Example (p.73, k = 4, traces like …k l b g **a** d h e k…): I(a) = {b}×3, {c}×2 (XOR-join), O(a) = {d,e}×5 (AND-split). Second example (p.75): I(a) = {b}×1, {c}×2, {b,c}×2 (OR-join), O(a) = {d}×2, {d,e}×3 (OR-split). Reading: sets always seen together = AND, separate singletons = XOR, a mix = OR. Refinements out of scope (p.81): empty windows, noise filtering of rare bindings, repeated activities. Approach 2 (p.82–85): enumerate candidate binding sets — each arc must be in at least one binding, so inputs {b,c} give 5 options × outputs {d,e} 5 options = 25 candidates for a — and pick the best by replay (fitness, precision, generalisation, simplicity); randomise or use a genetic algorithm if too expensive. Output = C-net + binding-frequency table (Instruction 4 solutions Table 7: c has I = [{a}⁸³, {b}⁶⁴], O = [{d}⁸⁷, {e}⁶⁰]).

### 1.6 Week 5 — Putting process mining into code with PM4Py

Sources: the 67 lecture slides and the Week-5 instruction sheet. No official solutions file exists for the exercises; the sketches in §1.6.3 are my suggestions.

#### 1.6.1 What PM4Py is

*ELI5:* Disco is a point-and-click app: you load a CSV, press "Map", get a picture. PM4Py is the same toolbox as Python functions: you type `pm4py.discover_petri_net_alpha(log)` instead of clicking. Everything is an ordinary Python object (pandas DataFrame, Petri net, dict), so process mining can be mixed with any other Python code, including machine learning.

*Precise:* PM4Py is an open-source Python process-mining library (reference on week5 slide 2: Berti, van Zelst, Schuster, "PM4Py: A process mining library for Python", *Software Impacts* 17, 2023). Slide 3 compares it with Disco: Disco has a desktop GUI, process animation and a DFG; PM4Py has no GUI or animation but adds Petri-net modelling, reachability analysis, process trees, causal nets, Alpha/Heuristics/Inductive miners and conformance checking. Both need textual sources preprocessed into CSV/XES; predictive process monitoring is "via external ML toolkits" in PM4Py and absent in Disco (week5 slide 3).

**Install / import.** The slides show **no `pip install` line** and **state no pm4py version**. Every example uses the same imports and a main guard (week5 slides 9, 13, 14):

```python
import pandas as pd
import pm4py
if __name__ == "__main__":
    ...
```

Some examples add `import pprint` (slide 45) or sub-module imports such as `from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator` (slide 34). [UNVERIFIED] The docstrings pasted on slides 8 and 44 (`read_xes(..., variant, return_legacy_log_object, ...)`, `return_diagnostics_dataframe`) match the pm4py 2.7.x API, so `pm4py>=2.7` is a sensible pin for `requirements.txt`.

#### 1.6.2 Core workflow (function names exactly as on the slides)

**2.1 Read XES** (slides 7–9). XES is the XML standard for event logs: `<log>` → `<trace>` → `<event>`, with attributes from standard extensions (`concept:name`, `org:resource`, `time:timestamp`). One call reads it; because the default is `return_legacy_log_object=False` (slide 8), the result on slide 9 is a DataFrame with columns `concept:name, org:resource, time:timestamp, ..., case:concept:name`.

```python
log = pm4py.read_xes('<path-to-log-file.xes>')
```

**2.2 Read CSV** (slides 10–13). "In PM4Py, CSV files cannot be imported directly as event log": load with pandas, tell PM4Py which columns are the three mandatory ones (case id, activity, timestamp — the same mapping step as in Disco, slide 11), then convert.

```python
dataframe = pd.read_csv('running-example.csv', sep=',')
dataframe = pm4py.format_dataframe(dataframe, case_id='case:concept:name',
             activity_key='concept:name', timestamp_key='time:timestamp')
event_log = pm4py.convert_to_event_log(dataframe)
```

`format_dataframe` renames your columns to the pm4py defaults `case:concept:name` / `concept:name` / `time:timestamp` and parses timestamps (slide 12 also lists `start_timestamp_key` and `timest_format`). `convert_to_event_log` returns an `EventLog` object (slide 12).

**2.3 Export** (slide 14): `pm4py.write_xes(log, 'exported.xes')`; for CSV, `pm4py.convert_to_dataframe(log).to_csv('exported.csv')`.

**2.4 Basic statistics.** The slides do **not** show dedicated statistics functions (no `get_start_activities`, `get_variants`, case counts). Start/end activities appear only as by-products: `discover_dfg` returns `(dfg, start_activities, end_activities)` (slide 19) and the footprints object contains `activities`, `start_activities`, `end_activities`, `min_trace_length` (slide 51). Trace and variant counts: not shown.

**2.5 Filtering** (slides 43, 48, 56, 59–60). Slide 59 lists the documentation's filter families (timeframe, case performance, start activities, end activities, variants, attribute values, numeric attribute values, between, case size, rework, path performance), but code is shown for only three pm4py filters plus plain pandas:

```python
filtered_log = pm4py.filter_start_activities(event_log, ["A_Create Application"])  # slide 60
filtered_log = pm4py.filter_variants_top_k(event_log, 3)                           # slides 43, 60
filtered_log = pm4py.filter_variants_by_coverage_percentage(event_log, 0.1)        # slide 60
filtered_log = event_log[event_log["case:concept:name"].isin(['1', '3'])]          # slide 48, pandas
```

Slide 60 runs these on "BPI Challenge 2017 sampled.csv" and re-discovers a DFG after each filter. Time, end-activity and attribute filters are only named on slide 59; no code is shown.

**2.6 Discovery** (slides 16–38). Assumption on slide 16: traces are finished and there are no duplicate activity labels. Slide 18 lists the API: `discover_dfg`, `discover_performance_dfg`, `discover_petri_net_alpha / _inductive / _heuristics / _ilp`, `discover_process_tree_inductive`, `discover_bpmn_inductive`, `discover_heuristics_net`, `discover_footprints`, `discover_powl`. Every worked example first reads `running-example.xes`.

```python
dfg, sa, ea = pm4py.discover_directly_follows_graph(event_log)       # slide 20
pm4py.view_dfg(dfg, sa, ea, format="svg")
performance_dfg, sa, ea = pm4py.discover_performance_dfg(event_log)  # slide 23
pm4py.view_performance_dfg(performance_dfg, sa, ea, format="svg")   # aggregation: mean (default), median, min, max, sum, stdev
net, im, fm = pm4py.discover_petri_net_alpha(event_log)              # slide 28
net, im, fm = pm4py.discover_petri_net_inductive(event_log)          # slide 30 (noise_threshold=0.0)
net, im, fm = pm4py.discover_petri_net_heuristics(event_log)         # slide 31 (dependency 0.5, and 0.65, loop_two 0.5)
pm4py.view_petri_net(net, im, fm, format="svg")
tree = pm4py.discover_process_tree_inductive(event_log); pm4py.view_process_tree(tree, format="svg")  # slide 36
bpmn = pm4py.discover_bpmn_inductive(event_log); pm4py.view_bpmn(bpmn, format="svg")                  # slide 37
```

Petri-net discovery returns a triple (net, initial marking, final marking). Slide 32 shows the three nets together: Alpha gives a plain net, Inductive adds silent (black) transitions, Heuristics is the largest. **Visualisation:** only `view_*` appears; `save_vis_*` is not on the slides. Slide 38 lists converters (`convert_to_bpmn`, `convert_to_petri_net`, `convert_to_process_tree`, `convert_to_reachability_graph`, `convert_log_to_networkx`, ...).

**2.7 Evaluating a model** (slides 33–34, recap of week 3's four quality dimensions):

```python
fitness = pm4py.fitness_token_based_replay(event_log, net, im, fm)   # dict -> fitness['log_fitness']
prec    = pm4py.precision_token_based_replay(event_log, net, im, fm)
gen     = generalization_evaluator.apply(event_log, net, im, fm)
simp    = simplicity_evaluator.apply(net)
```

For the Alpha net on running-example: fitness 1.0, precision 0.753, generalization 0.526, simplicity 0.652 (slide 34).

**2.8 Conformance checking** (slides 40–58): compare model and log in both directions. The trick in every exercise: discover the model from a *filtered* log (`filter_variants_top_k(event_log, 3)` or an `.isin([...])` case subset) so the full log no longer fits.

- *Token-based replay* (slides 42–46): replay each trace, count produced / consumed / missing / remaining tokens. `replayed_traces = pm4py.conformance_diagnostics_token_based_replay(event_log, net, im, fm)` gives per trace `trace_is_fit`, `trace_fitness` (0.875 in the example), `missing_tokens`, `consumed_tokens`, `remaining_tokens`, `produced_tokens`, `activated_transitions`, `reached_marking`, `transitions_with_problems`.
- *Footprints* (slides 47–54): the alpha-miner relation matrix (→, ←, ||, #) computed from the log and from the model, then compared. `fp_log = footprints_discovery.apply(event_log, variant=footprints_discovery.Variants.ENTIRE_EVENT_LOG)`; `fp_net = footprints_discovery.apply(net, im, fm)`; visualise with `fp_visualizer.apply(fp_net, parameters={...FORMAT: "svg"})` (`Variants.SINGLE`) or `fp_visualizer.apply(fp_log, fp_net, ...)` (`Variants.COMPARISON`, mismatches in red, slide 53); `fp_conformance.apply(fp_log, fp_net, variant=...LOG_EXTENSIVE)` then `evaluation.fp_fitness(...)` / `evaluation.fp_precision(...)` (0.7969 / 0.5556, slide 54).
- *Alignments* (slides 55–58): optimal alignment per trace; `>>` marks a move on log only or on model only. Shown only by docstring: `pm4py.conformance_diagnostics_alignments(log, *args)` (slide 57); output per trace has `alignment`, `cost`, `fitness` (0.889 in the example). [UNVERIFIED] The exact call line is not on the slides; by analogy with token replay it is `pm4py.conformance_diagnostics_alignments(event_log, net, im, fm)`.

**2.9 Performance.** Only the performance DFG (2.6) is shown. No throughput-time, case-duration or waiting-time function appears on the slides.

**2.10 Beyond** (slides 61–65): building a Petri net by hand (`PetriNet`, `PetriNet.Place`, `PetriNet.Transition`, `petri_utils.add_arc_from_to`, `Marking`); the reachability graph (`reachability_graph.construct_reachability_graph(net, im)` + `ts_visualizer`, slide 63 — the coded version of the week-2 reachability table); and simulation/playout (`simulator.apply(net, im, variant=Variants.BASIC_PLAYOUT, parameters={...NO_TRACES: 5})`, slide 64), which generates traces the model allows.

#### 1.6.3 Exercise sheet (my suggested approach, not an official solution)

All three CSVs are Disco-style, so step one is always `pd.read_csv` → `format_dataframe(case_id='Case ID', activity_key='Activity', timestamp_key='Complete Timestamp', start_timestamp_key='Start Timestamp')`.

**Exercise 1 — phonerepair.csv** (columns: Case ID, Activity, Resource, Start/Complete Timestamp, countryT, err, ok, phoneT, repn; every case begins with a dummy `Start` row whose attributes are the string "Start").
1. *Which activities?* keys of `dfg` from `discover_dfg`, or `pm4py.get_event_attribute_values(log, "concept:name")` [UNVERIFIED, not on slides].
2. *Workload NL vs B (cases):* split on `countryT` into two logs with `pm4py.filter_event_attribute_values(log, "countryT", ["NL"])` [UNVERIFIED] or pandas `.isin` as on slide 48; count unique `case:concept:name`.
3. *Repair quality:* same two logs, count cases with `ok == True`.
4. *Process differences:* `discover_dfg` / `discover_petri_net_inductive` + `view_*` per country log.
5. *Business rule for a new phone:* build a "repaired" and a "new phone" log (attribute or end-activity filter), compare `repn` distributions and start/end activities from `discover_dfg`.
6. *Throughput time:* `pm4py.get_all_case_durations(log)` [UNVERIFIED] per country, then min/max/mean with pandas.
7. *Bottlenecks:* `discover_performance_dfg` + `view_performance_dfg` (slide 23); the longest edges are the waits.

**Exercise 2 — BPI Challenge 2017 sampled.csv** (already pm4py-named: `case:concept:name`, `concept:name`, `time:timestamp`, `org:resource`, `lifecycle:transition`, plus offer attributes).
1. *Column choice:* `format_dataframe` with the defaults; check with `discover_dfg` + `view_dfg` as on slide 60.
2. *Cleaning:* drop the unnamed index column, parse timestamps, filter on `lifecycle:transition` / `EventOrigin` with pandas.
3. *"Instant" activities:* each event has one timestamp; pair `schedule/start/complete` events via `lifecycle:transition` or supply `start_timestamp_key`, otherwise durations are zero.
4–5. *Understanding / short vs long:* `filter_variants_top_k` or `filter_variants_by_coverage_percentage` (slide 60) then `view_dfg`; performance DFG for long edges.
6. *Cost reduction:* logs ending in `O_Accepted` vs not (`pm4py.filter_end_activities` [UNVERIFIED]); compare variants and performance.

**Exercise 3 — PurchasingExample.csv** (Case ID, Start/Complete Timestamp, Activity, Resource, Role; note CR-only line endings).
1. *Rework:* self-loops and repeats in `discover_dfg`; `pm4py.filter_activities_rework` [UNVERIFIED]; time via performance DFG.
2. *Main flow:* `filter_variants_top_k(log, 1)` vs the rest; compare performance DFGs.
3. *Case types by throughput:* case durations [UNVERIFIED function] binned, then variants per bin.
4. *Bottlenecks:* `view_performance_dfg` with `aggregation_measure="max"` or `"mean"` (slide 22).
5. *Non-conformance Amend PR → Create RfQ:* drop cases with that directly-follows pair (`pm4py.filter_directly_follows_relation` [UNVERIFIED]) or build a Petri net without that arc (slide 61 style) and run token replay / alignments.

#### 1.6.4 Exam radar

- Why PM4Py vs Disco: the slide 3 feature table.
- The CSV import recipe, the three mandatory columns and their default names.
- Function per notation: `discover_dfg`, `discover_petri_net_alpha/inductive/heuristics`, `discover_process_tree_inductive`, `discover_bpmn_inductive`; Petri-net discovery returns (net, im, fm).
- The four quality dimensions and the matching pm4py calls (slide 34).
- The three conformance techniques; what token replay counts and how `trace_fitness` / `trace_is_fit` are read; `>>` in alignments; footprint symbols and the comparison matrix.
- How to create an unfit model for testing (filter the log first, slides 43/48/56).
- Reachability graph and playout as "beyond" topics (slides 62–65).

#### 1.6.5 Relevance to our project

*Transfers:* the CSV → DataFrame → `format_dataframe` convention. The BPIC11 CSVs already use `case:concept:name` / `concept:name` / `time:timestamp`, exactly what `format_dataframe` produces, so pm4py functions accept them directly. The IMPresseD (2023) code imports pm4py for log objects and variant filtering; `filter_variants_top_k` / `filter_variants_by_coverage_percentage` (slide 60) and `pm4py.get_variants` [UNVERIFIED, not on slides] are the relevant family. The `if __name__ == "__main__":` guard and keeping logs as DataFrames fit our OOP `.py` layout. Slide 3's note that predictive monitoring works "via external ML toolkits" describes our pandas + XGBoost pipeline exactly.

*Does not transfer:* discovery algorithms, model visualisation, conformance checking, reachability and playout. The 2024 pipeline (pandas + XGBoost + mlxtend Apriori) uses none of them, and the 2023 code uses pm4py only as a data layer.

### 1.7 Exam radar — exercise types in the instruction sheets

| Week | Typical tasks (sheets and solutions) |
|---|---|
| 1 (Disco) | Map Case/Activity/Timestamp/Resource columns; attribute filters per country (phone repair: 856 NL vs 588 BE cases; repaired 729/856 = 85 % vs 547/588 = 93 %); business rule from statistics (new_phone cases all have repn = 3; return_phone: 1276 with 1, 383 with 2, 130 with 3 repairs); throughput and waiting times; clean BPI 2017 (wrong column choice, "instant" activities without start/complete lifecycle pairs); main flow via the variation filter (sequences shared by ≥ 32 cases); filter a non-conforming path (amend purchase requisition → create RFQ). |
| 2 | (P,T,F) ↔ drawing; name XOR/AND splits and joins; •t, t•; reachability table and graph (mind weights, mark dead markings); classify bounded/safe/deadlock-free/live/reversible with justification; "every deadlock-free net is live?" (false, Ex 4c); complete traces of a net; a Petri net / safe net / WF-net replaying a log "with as little extra behavior as possible" (with/without τ); make a net reversible (loop the last transition back to the start place); turn a net into a WF-net by adding exactly one place. |
| 3 | Tree formula ↔ drawing; language of a tree (loops: list variants with a single redo); WF-net → tree; text → tree without duplicates (loan process); text → C-net; DFG with frequencies from a trace/frequency table; I/O bindings of a C-net; validity of binding sequences with reasons (s2: b repeated; s4: f and m never in one output binding of a; s5: g before its inputs; s1, s3 valid). |
| 4 | Alpha on L1–L4: footprint, eight steps, net, a trace the model allows but the log lacks (L2: ⟨h,e,f,b,c,d,e,g⟩; L3: ⟨a,e,b,f⟩; none for L1, L4); HM: artificial start/end events only if there is no unique start/end activity, direct-follows and dependency matrices (solutions use 2 decimals), both thresholds, window 4 or 2 → C-net + binding table; threshold reasoning on L3 (dropping e→f needs a dependency threshold > 0.67 and leaves two end activities = "not well-formed"; at 0.5 an AND-join {b,e}→f replaces the single input from b); on L2 at 0.2 an extra AND-split ({c},{d,e}) and AND-join ({b,c},{d}) appear. |
| 5 (PM4Py) | Code instead of pen-and-paper: the week-5 sheet re-asks week-1-style questions (phonerepair, BPI Challenge 2017 sampled, PurchasingExample) in PM4Py; no official solutions file exists — sketches in §1.6.3, exam-relevant points in §1.6.4 and the three bullets below. |

Week 5 in three lines (the full list is §1.6.4):
- the CSV import recipe — `pd.read_csv` → `pm4py.format_dataframe(case_id=…, activity_key=…, timestamp_key=…)` → `convert_to_event_log` — and the three mandatory columns with their default names;
- one discovery function per notation (`discover_dfg`, `discover_petri_net_alpha/inductive/heuristics`, `discover_process_tree_inductive`, `discover_bpmn_inductive`) and the fact that Petri-net discovery returns (net, im, fm);
- the three conformance techniques (token replay, footprints, alignments), how `trace_fitness` / `trace_is_fit` and `>>` are read, and the trick of discovering from a filtered log to get a model the full log does not fit.

Caveats in the provided solutions (checked): Instruction 4 solutions list Y_L for L4 with ({e},{f}), which is not maximal (it is contained in ({b,c,e},{f}) and ({d,e},{f}); e occurs only in ⟨e,f⟩, so e # b, c, d); Table 6 prints c⇒e = 0.99 while 60/61 = 0.98; Ex 2.4 says "from the event log L1" but means L4 (the solution uses L4). Instruction 3 Ex 3.2 s2 contains "pg", a typo for "p, g".

Traps: judging enabling by output places; forgetting arc weights; listing a marking twice; "no deadlock" ≠ live; WF-net ≠ sound; b∥c forbids ({a},{b,c}) in Alpha; using the a≠b formula on a self-loop; forgetting that *both* HM thresholds must hold.

### 1.8 Bridge to the Group 11 project

- **Event logs, cases, traces, prefixes: yes.** BPIC11 (Gynaecology department of a Dutch academic hospital): one case = one patient (labels_report); the shipped CSVs have `case:concept:name`, `concept:name` (the benchmark "Activity code", e.g. AC410100), `Resource`, `time:timestamp`, a per-case `label` ∈ {0,1} and many extra columns (Diagnosis, Treatment code, Age, event_nr, …; header of BPIC11_f1_trunc36.csv). The 2024 pipeline takes prefixes of running cases and predicts a binary **outcome** — the "What will be the outcome for this case?" box of week-1 p.39, in the monitoring phase of the BPM lifecycle.
- **Discovery algorithms: no.** Neither repository calls Alpha, Heuristic, Inductive miner or a DFG function (grep for alpha_miner/heuristics_miner/inductive_miner/discover_dfg/directly_follows over both repos: no matches). pm4py in the 2023 code only formats the dataframe, converts it to an EventLog, filters variants and converts back (IMIPD.py:3–32); in the 2024 code every pm4py line is commented out (tools.py:2–5, 99–109). Petri nets, trees and C-nets are exam material only.
- **Footprint-like relations reappear in IMPresseD.** Its pattern graphs carry a `parallel` node attribute and an `eventually` edge attribute (InteractivePatternDetection/tools.py:349, 353; Auto_IMPID.py:109) — directly- vs eventually-follows and concurrency, the same vocabulary as →, ∥ and HM's dependency measure. The assignment discards these relations and keeps activity sets of size 1–3, so week 4 tells you exactly what the variant throws away.
- **Grading hooks:** reproducibility (14 pts) and the 12 Nov oral explanation (16:30–16:45) mean every member must be able to explain the code; log every AI use for the technology statement from day one (missing statement = 0 for the whole assignment).

---

## Part 2 — Concepts you need for the project, explained from scratch

Each concept: ELI5 analogy → precise definition → tiny example → where it appears in our project. Citations: printed page numbers for the papers (2024 paper = LNBIP 503 pp. 191–203; 2023 paper = LNCS 14159 pp. 303–319), `file:line` for the two repos (2024 repo = PermutationLocationImportance, 2023 repo = InteractivePatternDetection), `s.N` for slides (page N of the week-1 lecture PDF, which equals the slide number). Everything below was re-checked against the primary sources or re-run in the Python 3.12 venv (Python 3.12.10, pandas 2.3.3, scikit-learn 1.9.1, scipy 1.18.1, xgboost 3.4.1, matplotlib 3.11.2) by the fact-checker; the scratch scripts are `scripts/scout/_fc_B_checks.py`, `_fc_B_checks2.py`, `_fc_B_checks3.py`, `_fc_B_data.py`, `_fc_B_data2.py`.

### 2.1 Predictive process monitoring (PPM) vs. discovery vs. conformance

- **ELI5:** discovery draws the road map from last year's trips; conformance checks whether a driver followed the map; predictive monitoring watches a car *still driving* and guesses where it ends up.
- **Definition:** PPM methods "aim to predict the future status of ongoing cases by analyzing historical process data" (2024 paper p.191). The week-1 slides separate Process Discovery ("Is a process model available at all?"), Conformance Checking ("Does this model (still) reflect the reality?") and Predictive & Prescriptive Process Monitoring (slides week1 s.15); s.39 lists the PPM questions — remaining time, "What will be the outcome for this case?" — as `event stream → prediction model` (both slide numbers verified by splitting the PDF text on page breaks).
- **Example:** ⟨registration, blood test, consult⟩ → model says "0.7 probability of a tumor-marker test later".
- **Project:** we do not predict live cases; we train an outcome model on a historical log and *explain* it.

### 2.2 Outcome-oriented prediction and binary labels

- **ELI5:** every finished story gets a yes/no sticker; the model learns to guess the sticker from the pages.
- **Definition:** a *labeling function* maps a trace to a class label from a finite set of categorical outcomes, "for example, for a binary outcome Y = {0, 1}" (Teinemaa et al. 2019, Def. 2.4 — ref [18] of the 2024 paper; text in the arXiv text of Teinemaa et al., ll. 165–167). Code: column `label`, strings `deviant`/`regular` → 1/0 (tools.py:57-58); the BPIC11 CSVs already hold integers 0/1, constant per case, so this mapping is a no-op for them (labels_report §4.4).
- **Example:** `label = 1` if the trace ever contains activity X.
- **Project:** XGBoost is a binary classifier of `label`; every importance score is a drop in how well it predicts that label.

### 2.3 What the BPIC11 labels mean

- **ELI5:** the stickers come from grammar rules about the order of hospital tests, and (for three of the four files) each story was cut just before the page that gives the answer away.
- **LTL in one breath (the "grammar" the rules are written in):** LTL = *Linear Temporal Logic*, a small language for rules about the *order* and *eventual occurrence* of activities in one trace, read left to right in time. The three operators used below: **F** φ = "eventually φ" (φ holds now or at some later event), **G** φ = "always φ" (φ holds at every event from here on), φ **U** ψ = "φ until ψ" (φ holds at every event until the event where ψ holds — and ψ does occur) [ASSUMPTION: textbook LTL semantics, not from the course slides; the papers only use the symbols]. Tiny example on ⟨A, B, C⟩: F(C) is true; G(A) is false (B and C are not A); (¬C) U (B) is true (no C before B, and B occurs); G(A → F(C)) is true (every A is eventually followed by a C).
- **Definition (labels_report §3, verbatim from Teinemaa et al. §5.1):** f1 = F("tumor marker CA-19.9") ∨ F("ca-125 using meia"); f2 = G("CEA - tumor marker using meia" → F("squamous cell carcinoma using eia")); f3 = (¬"histological examination - biopsies nno") U ("squamous cell carcinoma using eia"); f4 (shipped in the repo but not used in the 2024 paper's figures, check_paper-2024 A) = F("histological examination - big resectiep"). F = eventually, G = always, U = until. *Empirically, label 1 = rule satisfied*: f1 patient eventually gets a marker test (40%); f2 every CEA test is eventually followed by a squamous-cell test, incl. the vacuous "no CEA at all" (78%); f3 squamous-cell test with no biopsies-nno before it (23%) (labels_report §4.5, zero exceptions where the rule status is observable). This contradicts the paper's displayed formula (1 = violated); the benchmark preprocessing script swaps the strings `deviant`/`regular` before writing its output (labels_report §4.3, `preprocess_logs_bpic2011.py:63-65`). [ASSUMPTION] the swap fully explains the contradiction, i.e. the Drive input files followed the paper's definition; the input files' semantics are [UNVERIFIED]. In f1/f3/f4 the decisive codes were cut out of the data (f1: `AC379414`, `378619A`; f3: `AC356134`, `376480A`; f4: `AC356133` — each occurs in 0 cases of its own file and is present in the uncut f2); f2 is uncut and the candidate CEA code `376400` [ASSUMPTION: inferred from f2 label statistics only; no code-to-name table was available] remains (labels_report §7).
- **Example (f1):** ⟨c1, c2, CA-125, c4⟩ → stored as ⟨c1, c2⟩, label 1; ⟨c1, c2, c3⟩ stays whole, label 0.
- **Project:** important f1/f3 activities are *proxies* for the cut-out test. The 2024 paper's distribution plots (Fig. 4) treat label 0 as "positive"/"accepted" (check_paper-2024 B3-B4, read from the figures; not re-checked in this run); state our own convention explicitly in the paper and poster.

### 2.4 Prefix, ongoing case, truncation — and `All_prefixes = False`

- **ELI5:** a prefix is the first k pages; an ongoing case is a story still being written; truncation is "never read past page N".
- **Definition:** a prefix of length k = first k events, simulating a running case. The files are *prefix-truncated* at N = min(40, ⌈90th percentile of case length among positive-class cases⌉) = 36/40/31 for f1/f2/f3 (benchmark `experiments.py:85-92`, `DatasetManager.py:120-121`; labels_report §5.2) — hence `trunc36/40/31`. This differs from *trace cutting* (§2.3), which removes the label-revealing event and everything after it.
- **Verified in code:** `All_prefixes = False` is hard-coded (CrossValidation_ProcessPermutation.py:36), so `index_encoding(data_manager.data)` makes **one row per case = the whole truncated trace** (:65-69). `prefix_generator` (tools.py:81-95) is therefore unreachable; it uses `DataFrame.append` (tools.py:92), which was removed in pandas 2, so it would crash if enabled (env_report P5).
- **Example:** ⟨A,B,C,D⟩ with N = 3 → ⟨A,B,C⟩; with prefixes on, the full trace is kept *and* ⟨A,B⟩ … are added (prefix lengths 2..n−1: `range(2, len(...))` at tools.py:86, `iloc[:i]` at :88).
- **Project:** the model classifies complete truncated traces, not early prefixes; say so in the paper.

### 2.5 Feature encoding: index-based vs. binary vs. frequency

- **ELI5:** a model eats fixed-width spreadsheets, so a variable-length story becomes either "who sat in seat 1, seat 2, …" (index-based), "who came at all" (binary) or "how many drinks each guest had" (frequency).
- **Definition:** *Index-based*: one 0/1 column `e{i}_{a}` per (position i, activity a) plus padding `e{i}_0` for positions past the end of shorter traces (tools.py:338-364; pivot on `event_nr` with `fill_value=0`, :345-346; `pd.get_dummies` per position column, :352-354; missing (i,a) columns added as zeros, :359-362). For f1 after the rare-activity filter: 36 × 164 + 35 padding = 5,939 features (env_report; re-confirmed: `X.shape[1] == 5939`). *Binary*: one column per activity, 1 if it occurs anywhere in the case (tools.py:286-302). *Frequency*: one count column per activity (tools.py:268-284; the 2023 paper's quantitative evaluation also uses "frequency-based encoding" of patterns, p.315).
- **One-hot encoding (the name of the 0/1-column trick):** ELI5: instead of writing "seat 2 = LA" in one cell, give seat 2 one light bulb per possible activity and switch on exactly one. Precisely: a categorical variable with m possible values becomes m binary columns, of which exactly one is 1 in each row (all zeros are impossible unless the value is missing). Index-based encoding is one-hot encoding applied *per position*: `pd.get_dummies` turns the position-i column into the `e{i}_{a}` columns (tools.py:352-354), and the `fill_value=0` cells of short traces become the `e{i}_0` padding bulb. Tiny example: position 2 with possible values {A,B,C} → columns `e2_A e2_B e2_C`; a trace with B in seat 2 → `0 1 0`. Pitfall seen in env_report P1: pandas ≥ 2 makes these columns `bool`, not integers, and mixing them with the zero-filled `int64` columns crashes XGBoost under the shipped code — cast to `int`.
- **Worked example:** activities {A,B,C}; σ1 = ⟨A,B,A⟩ (label 1), σ2 = ⟨B,C⟩ (label 0); max length 3.

```
index-based:  e1_A e1_B e1_C | e2_A e2_B e2_C | e3_A e3_B e3_C e3_0 | label
      σ1        1    0    0  |  0    1    0   |  1    0    0    0   |   1
      σ2        0    1    0  |  0    0    1   |  0    0    0    1   |   0
binary:   A B C      frequency:  A B C
   σ1     1 1 0           σ1     2 1 0
   σ2     0 1 1           σ2     0 1 1
```

  `e1_C` etc. are the all-zero columns of tools.py:359-362; `e3_0` is padding produced by `get_dummies` on the `fill_value=0` cells (there is no `e1_0`: every trace has a first event).
- **Project:** location importance needs index encoding; the existence baseline uses binary encoding per the paper ("we utilized binary encoding, where each feature represented a distinct itemset", p.197). The shipped `Classical_Permutation.py` does something slightly different in multi-activity mode: each itemset feature = min over its activities of the substring count in the `'->'`-joined trace string (Classical_Permutation.py:44-58), i.e. a count, not a 0/1 flag, and an XGBoost trained on only those itemset features (:91-92). Both of our strategies feed the same index-encoded model; only the candidate sets differ.

### 2.6 Supervised learning — from "five lines" to a proper picture

- **ELI5:** a teacher gives you a pile of solved exercises (question + correct answer). You practise until you can produce the answers yourself, then the teacher checks you on *new* exercises whose answers you have never seen. Supervised learning is exactly that: learn from examples with known answers, be graded on unseen ones.
- **Definition — the four words you need:** a **row** (sample, instance) is one object we want to classify — here *one hospital case*, i.e. one truncated trace (§2.4). A **feature** is one column describing the row — here the encoded 0/1 columns `e{i}_{a}` of §2.5 (5,939 of them for f1). The **label** y is the known answer stored with the row — here `label` ∈ {0, 1} (§2.2–2.3). The **model** is a function f that maps a feature vector x to a prediction ŷ (a class, or a probability of class 1); *learning* (fitting, training) means adjusting the model's internal numbers so that f(x) ≈ y on the rows it is shown — the *training set*. A **loss** (error function) is one number that says how wrong the predictions are on a set of rows (the smaller the better); XGBoost's default `binary:logistic` minimises the *logistic loss* (log loss): predicting probability p for a row whose true label is 1 costs −ln p, so a confident wrong answer is punished hardest [ASSUMPTION: textbook definition of log loss, not from the course slides; the objective name itself was read from the fitted booster, see §2.7]. A **held-out test set** is a group of rows never used for fitting; scoring on it is like checking homework answers you have not seen — it is the only honest estimate of how the model does on new cases. **Overfitting** is memorising instead of learning: the model becomes so tailored to the training rows (their noise, their coincidences) that the training score keeps rising while the held-out score stops improving or falls; the symptom is a large train–test gap (§2.7 shows ours).
- **The five steps (the original "five lines"):**
  1. Collect rows x (features) with known answers y (labels).
  2. Choose a model family (boosted trees) and a loss (`binary:logistic`, the `XGBClassifier()` default; verified from the fitted booster's `save_config()` on xgboost 3.4.1).
  3. Fit: adjust the model so its predictions on training rows match y.
  4. Evaluate on rows the model never saw (test fold) with a metric such as F1 (§2.9).
  5. Explain: ask which inputs the fitted model relies on — where this project starts (§2.10–2.12).
- **Tiny example:** four training rows with two features and a label — (`e2_B`=1, `e3_A`=1) → 1; (`e2_B`=1, `e3_A`=0) → 1; (`e2_B`=0, `e3_A`=1) → 0; (`e2_B`=0, `e3_A`=0) → 0. A model that learns "ŷ = `e2_B`" has zero training loss. Test row (`e2_B`=1, `e3_A`=0, true label 0): the model says 1 — one error on the held-out set, which is the kind of mistake the training score alone could never reveal. A model that had instead memorised all four rows by their exact (`e2_B`, `e3_A`) pairs would be *overfitted*: perfect on training, and for a new combination it has nothing to say.
- **How this maps to our pipeline:** rows = the 1,130 f1 cases (676 label-0, 454 label-1 per env_report §3.1); features = the index-encoded columns; labels = `label`; model = one `XGBClassifier()` per fold (§2.7); training set / held-out set = the K−1 training folds / the test fold of stratified 5-fold CV (§2.8); loss = `binary:logistic` during fitting, while the *reported* number is weighted F1 (§2.9); the explanation step is the location-permutation importance of §2.11–2.12, computed — as in the shipped code — on the *training* fold (CrossValidation:95-98), so the "baseline" it starts from is a training score, not a held-out one.

### 2.7 XGBoost and why it needs fixed-length vectors

- **Decision tree (ELI5 first, because XGBoost is a pile of them):** a decision tree is a flow chart of yes/no questions on the columns that ends in a guess: "is `e2_LA` = 1?" → yes: "mostly negative"; no: "is `e8_LA` = 1?" → … Precisely: a binary tree whose internal nodes each test one feature against a threshold and whose leaves hold a prediction; every row follows exactly one root-to-leaf path. In XGBoost a leaf holds a *number* (a push up or down on the running score), not a class; the class comes from the summed pushes of all trees. Tiny example: root "`e2_LA` = 1?" → yes: leaf −0.8 (pushes toward negative); no: "`e8_LA` = 1?" → yes: leaf +0.6; no: leaf 0.
- **ELI5 (XGBoost):** a committee of many small decision trees built one after another, each new tree concentrating on the mistakes the committee has made so far (gradient boosting).
- **Definition:** gradient-boosted tree ensemble, "a top-performing ensemble model in outcome prediction according to [18]" (2024 paper p.197). Code: `XGBClassifier()` with library defaults (CrossValidation_ProcessPermutation.py:87). On the venv's xgboost 3.4.1 a fitted default booster reports 100 boosted rounds, `max_depth=6`, `eta=0.3`, `subsample=1`, `colsample_bytree=1`, `tree_method=auto`, booster `gbtree`, objective `binary:logistic` (read from `save_config()`; in xgboost 3.x `get_params()` returns `None` for unset parameters, so the booster config is the only reliable place to read them). The xgboost version the 2024 authors used is [UNVERIFIED] (the 2024 repo has no requirements file, check_paper-2024 A). With these defaults there is no row/column subsampling, and two consecutive default fits on the full f1 index-encoded data (1130 × 5939) gave bit-identical `predict_proba` outputs in this run — so with fixed folds the model is deterministic on this machine (cf. check_paper-2024 B9).
- **Why fixed length:** each tree asks "column j ≤ threshold?", so every row needs the same columns; a 12-event and a 36-event trace must become rows of equal width (§2.5).
- **Example:** tree 1: "`e2_LA`=1 → mostly negative"; tree 2 repairs tree 1's errors using `e8_LA`; the votes are summed.
- **Overfitting, measured on our data:** training weighted F1 on all 1130 f1 cases was 0.992 in the smoke test, and on fold 0 of a 5-fold split the training-fold F1 was 0.997 while the *held-out* test-fold F1 was 0.889 (another run: 0.906) (env_report §3.1–3.3). ELI5: the model scores ≈ 99% on the exercises it practised and ≈ 89% on new ones — the ≈ 0.10 gap is the signature of **overfitting** in the ML sense (§2.6): 100 trees of depth 6 on 5,939 columns and 904 training rows can memorise the training traces. Precisely: overfitting = the model fits the training rows more closely than the underlying regularity, so the training score overstates the real one; the held-out folds of §2.8 exist to measure the real one.
- **Project:** one XGBoost per fold on index-encoded traces; the baseline is close to a perfect *training* fit (0.992, env_report), and the shipped code measures importance on that training fold — so an importance value is "how much the near-memorised training score drops when the itemset is relocated", not a held-out effect; say so in the paper.

### 2.8 Train/test split and stratified K-fold cross-validation

- **ELI5:** don't grade students on the exercises they practised; K-fold examines everyone once while the others practise; "stratified" keeps the same easy/hard mix in every group.
- **Definition:** split cases into K folds; for fold k train on the other K−1, evaluate on k. Stratified folds are "made by preserving the percentage of samples for each class" (sklearn `StratifiedKFold` docstring, 1.9.1). Code: `StratifiedKFold(n_splits=K_fold, shuffle=True)` on the per-case outcome (tools.py:231-236, K_fold = 5 at CrossValidation:39), **no `random_state`** → folds change every run (env_report P3: fold-0 test F1 0.8888 vs 0.9064 in two runs).
- **Example:** 10 cases, 4 positive, K = 2 → each fold 5 cases, 2 positive.
- **Leakage (why the split must be clean):** ELI5: the student peeks at the answer key — any information from the *test* cases, or from the *label* itself, that sneaks into the training or candidate-selection step makes the held-out score look better than it will be on truly new cases. Precisely: leakage = using, at fitting or selection time, information that would not be available for a genuinely new case (test-row contents, or features that are functions of the label). Where it threatens here: (a) the 2024 code mines Apriori itemsets on the *whole* log before the CV split (§2.13; mild, because support uses no labels, but the candidate set has "seen" the test cases — the 2023 paper mines on the training set only, p.315, see §2.14); (b) BPIC11 columns such as `Variant`/`Variant index` identify the control-flow variant, i.e. they encode the trace itself (Part 4, recommended BPIC11 configuration); (c) activity `376400` in f1 is a possible near-leak of the cut-out tumor-marker test (§2.15). Tiny example: choosing the "top 10 itemsets" by looking at how well they predict the label on *all* cases, then scoring on a test fold drawn from those same cases, lets the test labels influence the choice.
- **Box plot (how the 50 values per itemset are drawn):** ELI5: a box plot is a summary picture of a list of numbers — a line at the **median** (the middle value), a **box** from the 25th to the 75th percentile (the middle 50% of the values; its width is the inter-quartile range, IQR), **whiskers** out to the furthest values that are still "not too far", and separate dots for the **outliers** beyond the whiskers. Precisely (matplotlib 3.11.2, `boxplot_stats` docstring): "the lower whisker is at the lowest datum above Q1 − whis·(Q3−Q1), and the upper whisker at the highest datum below Q3 + whis·(Q3−Q1)"; the default `whis = 1.5` "corresponds to Tukey's original definition". The 2024 code calls `plot.box(vert=False, whis=10, ax=axs)` (CrossValidation_ProcessPermutation.py:126 — verified; likewise Process_Permutation.py:78/84 and Classical_Permutation.py:140), i.e. the whiskers may stretch to 10 IQRs beyond the box, so in practice nearly every value lies inside the whiskers and almost no outlier dots appear; `showfliers=False` (used in Plotting_results.py:45/49) hides the outlier dots entirely, whatever `whis` is. Tiny example (re-run with `boxplot_stats`): values 1, 2, 3, 4, 12 → median 3, box 2–4 (IQR 2); with `whis=1.5` the upper whisker stops at 4 (limit 4 + 3 = 7) and 12 is drawn as an outlier dot; with `whis=10` (limit 4 + 20 = 24) the whisker reaches 12 and there is no dot. So the paper's Fig. 5 boxes show the median and spread of the 50 importance values, with very long whiskers.
- **Project:** 5 folds × 10 permutation repeats = 50 values per itemset, box-plotted (paper p.197; `n_repeats=10` default at tools.py:499; box plot at CrossValidation:125-126). Importance is computed on the *training* fold (`train_x, train_y, train_list[i]`, CrossValidation:95-98); the test F1 is only printed (:92). We will seed the folds (rubric: correctness & reproducibility 14 pts).

### 2.9 Metrics: accuracy, precision, recall, F1, weighted F1, "baseline"

- **ELI5:** accuracy = share of right answers; precision = of the alarms you raised, how many were real; recall = of the real fires, how many you caught; F1 is high only when both are.
- **Definition:** accuracy = (TP+TN)/all; precision = TP/(TP+FP) and recall = TP/(TP+FN) (sklearn `precision_score`/`recall_score` docstrings: "the ratio tp / (tp + fp)", "tp / (tp + fn)"); F1 = harmonic mean of precision and recall (sklearn `f1_score` docstring), i.e. F1 = 2PR/(P+R); all four were also checked numerically against sklearn on a toy confusion matrix (TP=2, FP=1, FN=1, TN=4 → P = R = F1 = 0.667, accuracy 0.75). *Weighted F1* "calculate[s] metrics for each label, and find[s] their average weighted by support" (sklearn `f1_score` docstring). Why weighted? [ASSUMPTION: textbook rationale for imbalanced classes, not stated in the paper] — the classes are imbalanced (23–78% positives); the 2024 paper only says "f1-score" (p.197) and "weighted" is visible only in the code. *Baseline* here = weighted F1 of the trained model on unpermuted data, `baseline = f1_score(y, model.predict(X), average='weighted')` (tools.py:503-504); `importance = baseline − F1(permuted)` (tools.py:539); "a decrease in the f1-score indicates the importance" (paper p.197).
- **Example:** 8 negatives, 2 positives; always-"negative" gives accuracy 0.8, F1(pos) = 0, F1(neg) = 16/18 ≈ 0.889, weighted F1 = 0.8·0.889 + 0.2·0 ≈ 0.711 (checked with sklearn: 0.7111).
- **Project:** same metric for both strategies. Caveat: `Classical_Permutation.py` calls `permutation_importance(model, train_x, train_y, n_repeats=20, random_state=42, n_jobs=2)` with no `scoring` argument (:108-109), so its PFI uses the estimator's default score — *accuracy* for a classifier — despite the axis label "Decrease in f1 score" (:143) (check_paper-2024 A).

### 2.10 Explainable AI: post-hoc, model-agnostic, global/local, factual/counterfactual

- **ELI5:** post-hoc = interview the oracle after it is built; model-agnostic = the interview uses only questions and answers, so it works on any oracle; global = "what does it care about in general", local = "why this verdict"; factual = "why yes", counterfactual = "what must change to get no".
- **Definition (2024 paper §2, pp.193-194, and §1 p.192):** *factual* explanations reveal "the reasoning behind specific predictions"; *counterfactual* ones show "what changes are necessary for an input sample to achieve a desired prediction" (examples DiCE4EL, LORELEY, CREATED). Factual methods are *intrinsically interpretable* (the paper lists rule-based classifiers, neuro-fuzzy networks and linear regression) or *post-hoc* ("explain decisions made by black-box models after they are built", p.192). Post-hoc methods are *model-specific* or *model-agnostic* ("compute explanations based on the inputs and their associated outputs", p.194). SHAP, LIME and PFI have been used "to obtain the importance of different process features at both local and global levels" (p.192).
- **One-line glosses of the three names:** **SHAP** (SHapley Additive exPlanations) splits one prediction into per-feature contributions that add up to the prediction, using game-theoretic Shapley values; **LIME** (Local Interpretable Model-agnostic Explanations) fits a simple, interpretable model around one input to see which features mattered *there*; **PFI** = permutation feature importance (§2.11), a global score. [ASSUMPTION: textbook descriptions; the 2024 paper only names the methods.] In the 2024 repo `shap` is imported only by `Classical_Permutation.py` (env_report P12).
- **Example:** one patient's SHAP values = local; mean |SHAP| over all patients = global.
- **Project:** the location method is **factual, post-hoc, model-agnostic, global** ("a novel post-hoc model-agnostic method", p.193; "measuring the global importance of the location of one or a group of activities", p.194). We never open XGBoost's internals.

### 2.11 Permutation feature importance (the blindfold test) and why it fails for position

- **ELI5:** blindfold the model on one column by shuffling that column across rows; a big score drop means the model leaned on it.
- **Definition:** Breiman 2001 (ref [2] of the 2024 paper); sklearn: "the difference between the baseline metric and metric from permutating the feature column", averaged over `n_repeats` (`permutation_importance` docstring, `importances_mean`). PFI_j = s − (1/K)·Σ_{k=1..K} s_{k,j}, s = baseline score, s_{k,j} = score after the k-th shuffle of column j (verified numerically in this run: sklearn's `importances_mean` equals baseline minus the mean permuted score).
- **Example:** shuffling `age` drops F1 0.90 → 0.70 (PFI 0.20); shuffling `noise` leaves 0.90 (PFI 0).
- **Why it fails for location (2024 paper pp.194-195):** in index encoding an activity's location "does not map to a single column; instead, it encompasses all the columns related to the same activity" (`e1_LA`, `e2_LA`, …); shuffling one column independently is "inadequate for our analysis due to intricate interdependencies among features", and "shifting the location of an activity inevitably impacts also the location of other activities in the trace". Shuffling `e2_LA` alone can yield LA at positions 2 *and* 8, or two activities in one position — impossible traces.
- **Project:** permute at *trace* level instead: move the itemset to another observed position, re-encode, re-score (§2.12). Classical PFI on an existence encoding stays as the "existence importance" baseline (paper p.197).

### 2.12 "Location" = position index in the trace; the Table 1 example

- **ELI5:** location is the seat number of an activity in the queue of events of one case — not a room, not a clock time.
- **Definition:** the ordinal index of an event in its timestamp-ordered trace; the paper speaks of LA "in position 2" and of "different moments (locations)" (p.192). Code: 1-based `event_nr` (feasibility pool `Allowed_locations[act]` = sorted unique `event_nr` per activity over the whole filtered log, tools.py:70-73; index encoding pivots on `event_nr`, :345). Itemset locations are tuples, e.g. OL(L,(ER,CR)) = {(5,4),(10,11),(2,6),(3,6),(5,7),(2,5),(4,3)} (p.196). Two constraints: *Feasibility* (move only to positions where the activity was observed anywhere in the log) and *Preserving Ordering Relation* (keep the itemset activities' relative order within each trace) (p.196). The shipped code enforces neither constraint strictly because of index bookkeeping bugs in `shuffle_sequence` (tools.py:469-474; check_paper-2024 B5). Measured: about 3.4% of the 612 single-occurrence f1 traces of the itemset {ac370419, ac370442} (seed 2023, re-verified run; an earlier check on a slightly different selection measured 3.5%) end with reversed itemset order — see §3.3.5.
- **Table 1 re-derived (2024 paper p.192; label = patient returns to the emergency room within 28 days of discharge):**

```
1 RG LA LE CR ER ST IL IA NC ER CR LE CR   Negative   LA@2
2 RG ER ST IL LE CR IA LA IC DI            Positive   LA@8
3 RG LA ER ST LE CR IA NC CR LE CR LE CR   Negative   LA@2
4 RG LE LE LA ER ST CR IA IC LE LA NC DI   Positive   LA@4, LA@11
5 RG ER ST LE CR IL IA LA NC CR LE         Positive   LA@8
6 RG LA CR ER ST LE IL IA NC DI            Negative   LA@2
```

  Every trace contains LA, so existence says nothing; LA at 2 ⇔ negative, LA at 4, 8 or 11 ⇔ positive (re-derived from the table; check_paper-2024 B1 corrected an earlier "12"). Re-counting OL(L,(ER,CR)) from the table reproduces the paper's seven tuples: trace 1 gives (5,4) and (10,11), its third CR (position 13) has no ER partner and is ignored (p.196); traces 2–6 give (2,6), (3,6), (5,7), (2,5), (4,3).
- **Project:** BPIC11 timestamps are day-level (values like `1/2/2005 23:00`) and 88.8% / 87.1% / 88.5% of consecutive event pairs within a case share a timestamp in f1/f2/f3 (re-computed in this run on the raw CSVs; matches check_paper-2023 B4), so within-day "location" is just file order (`event_nr`). List under limitations.

### 2.13 Frequent itemsets, support, Apriori (shopping baskets)

- **ELI5:** "bread & butter" is frequent if at least half the baskets contain both; Apriori's trick: if bread alone is rare, no combo with bread can be frequent, so skip them.
- **Definition:** *transaction* = a case reduced to its *set* of distinct activities (order and repeats dropped; `TransactionEncoder`, tools.py:161-166). *Support* = fraction of transactions containing all of the itemset; *frequent* = support ≥ `min_support`. Apriori builds candidates level by level using "every subset of a frequent set is frequent". Code: `mlxtend.apriori(min_support=0.5)` on the *whole* filtered log before the CV split (CrossValidation:51 vs :71), sort by support, keep the first `top_k + #activities` rows, drop size-1 itemsets, take the top 10 (tools.py:167-174; the `+ len(unique activities)` at :169 is a crude way to leave room for the singletons that are removed at :172); defaults `--top_k 10`, `--min_support 0.5` (CrossValidation:27-28, declared with `argparse` — Python's standard command-line-argument parser, which turns a terminal call such as `python script.py --top_k 10` into the variable `args.top_k`; CLI = command-line interface, i.e. the terminal — and `top_k` is declared `type=float`, which crashes `.head()` as soon as it is passed on the CLI, env_report P2); paper: "top 10 frequent itemsets of size greater than one" (p.197). Note the paper cites "an improved Apriori" [24] but the code uses plain mlxtend `apriori` (check_paper-2024 C12).
- **4-case example, min_support 0.5:**

```
c1 {A,B,C}   c2 {A,B}   c3 {A,C}   c4 {B,C,D}
size 1: A 3/4  B 3/4  C 3/4  D 1/4 -> D pruned
size 2 (from frequent singletons): {A,B} 2/4  {A,C} 2/4  {B,C} 2/4  -> kept
size 3: {A,B,C} 1/4 -> dropped; nothing containing D is ever generated
top-k of size>1: three ties at 0.5 -> tie-break decides the ranking
```

- **Project:** this is IE(L) in the original (paper §3.1, p.196: "A common and straightforward approach is measuring an itemset's interest through frequency, often accomplished using the Apriori algorithm"). Re-run in this run with the repo's `DataManager` + `frequent_activity_sets(0.5, 10)`: f1 → 10 itemsets (top: {370407, ac370000} 0.591; ranks 9–12 tie at 0.5566 (four itemsets), so which one takes rank 10 depends on pandas' sort tie-breaking (see §3.3.4), check_paper-2024 B17); f2 → 10 itemsets (top: {ac370000, ac379999} 0.690); **f3 → only 5 itemsets of size > 1 at support 0.5** (top: {370407, ac370000} 0.5275), so the paper's ten f3 itemsets in Fig. 5 require a lower support (check_paper-2024 A). The code silently uses fewer than `top_k` (C8). A per-length comparison needs top-k *per size* (mlxtend `max_len=3`, then filter by size).

### 2.14 IMPresseD: process patterns, extension, interest functions, Pareto front

- **ELI5:** Apriori asks "which activities travel together?"; IMPresseD asks "which small road-maps (A then B, A eventually C, A alongside D) are common, tell us about the outcome, *and* are not just an artefact of patient age?", growing maps one step at a time and keeping only maps nobody beats on all three questions.
- **Definition:** a process pattern "corresponds to a set of process activities (possibly annotated with additional data) with their ordering relations" (2023 paper p.303): a DAG P = (N, ↦, α, β), α = node labels (activities), β = the foundational (parent) pattern, NULL for a single node (Def. 4, p.307). **DAG = directed acyclic graph** — ELI5: dots joined by arrows, with no way to follow the arrows round in a circle, so "A before B" can never loop back to A; precisely: a directed graph (nodes N, arrows ↦) that contains no directed cycle; tiny example: A→B, A→C, B→C is a DAG, A→B→A is not. Traces are first turned into partially ordered traces (DAGs) by a conversion oracle; relations: *directly follows* (adjacency A ≠ 0), *eventually follows* (reachability R ≠ 0 for paths of length ≥ 2), *concurrent* (R = 0 in both directions) (Def. 3, pp.306-307; the literal text is slightly sloppy about directly-linked events, check_paper-2023 B14). Step 3 extracts "patterns of length-1, i.e., individual activities" (p.307). *Extension* adds nodes by six rules f ∈ {direct following, direct preceding, concurrent, eventually following, eventually preceding, direct context} (p.309); the paper's set-builder uses ∀n ∈ E′, while Fig. 2 and the code extend at the pattern's boundary (check_paper-2023 B1). Loop: convert → define interests → length-1 patterns → score → keep the Pareto front → user picks → extend → back to scoring (Steps 1–7, pp.307-308). Code: `Trace_graph_generator` flags two *consecutive* events as parallel when their timestamps differ by at most `delta_time` seconds (IMIPD.py:347; a negative `delta_time` yields a pure chain); `paretoset(objectives, sense=[...])` gives the front (Auto_IMPID.py:33-35).
- **Graph isomorphism (how the code decides two patterns are "the same"):** ELI5: two patterns are the same drawing with the nodes relabelled — if you can rename and rearrange the dots of one and get exactly the other, same arrows, same activity names on the compared dots, they count as one pattern. Precisely: a one-to-one correspondence between the two node sets that preserves the arrows (and, here, the compared node/edge attributes). Code: every new candidate is `nx.is_isomorphic`-compared with the patterns already stored at that stage (2023 repo tools.py:85-91, matching node `value` and edge `eventually`); a hit files the new instance under the existing pattern ID, a miss creates a new ID, so the same shape found in several traces is stored once (details, cost and a cheaper tuple-key equivalent for chain traces: Part 4). The paper never uses the word; it is a code-level detail. Tiny example: "A→B found in trace 7" and "A→B found in trace 12" are isomorphic → one pattern with two instances; "A→B" and "B→A" are not (the arrow direction differs) — though our variant will merge them anyway when it projects to activity sets (Project bullet below).
- **Interest functions (2023 paper §4.3, p.311):** (i) *frequency interest / case coverage* CC = share of cases with ≥ 1 instance, maximise (`Frequency_Interest` = non-zero count ÷ number of cases, IMIPD.py:92-105); (ii) *outcome interest* OI = ρ(outcome vector, per-case pattern-count vector): Spearman for continuous, information gain for categorical outcomes, maximise (code: `mutual_info_classif(..., discrete_features=True)` for a binary outcome, IMIPD.py:64-69; |Spearman| for numerical, :71-87); (iii) *case distance* CD = distance between initial case attributes of cases with vs. without the pattern, minimise, "Ideally, there must be CD(P, L, AT) = 0" (paper: 1/|L|-weighted double sum; code: `np.mean` over in–out case pairs, IMIPD.py:37-60; the published Fig. 3 values are consistent with the code's mean, check_paper-2023 B17).
- **"Jaccard" inside CD is not the Jaccard of §2.16:** the paper takes "distJac the Jaccard distance for m categorical feature" plus a normalised Euclidean distance for numerical ones, combined as (F_normal(dist_Euc) + dist_Jac)/(m+1) (p.311). The code label-encodes every categorical case attribute (`LabelEncoder`: each category → an integer code 0, 1, 2, …) and calls `pdist(X_features[cat_col].values, 'jaccard')` on those integer vectors (IMIPD.py:139-146; numerical columns: `pdist(..., 'euclid')` min-max scaled, :150-155; combination `(m·cat_dist + numeric_dist)/(1+m)`, :158). So this "Jaccard" is a *per-attribute mismatch measure between two cases' attribute vectors*, whereas §2.16 uses the *set* Jaccard similarity |A∩B|/|A∪B| between two lists of itemsets. **Version pitfall (verified in the Python 3.12 venv, scipy 1.18.1):** scipy's `jaccard` docstring carries the note "versionchanged 1.15.0 — Non-0/1 numeric input used to produce an ad hoc result. Since 1.15.0, numeric input is converted to Boolean before computation", so integer codes become 0 → False and ≥ 1 → True, and two cases with *different* non-zero codes count as *identical* on that attribute: `pdist([[0,1,2],[0,1,3]], 'jaccard')` = 0.0 and `pdist([[0,1,2],[1,1,2]], 'jaccard')` = 0.333 (re-run). On this venv the categorical part of CD therefore only registers whether a case carries the one category that `LabelEncoder` happened to code as 0. On the 2023 repo's pinned scipy 1.11.4 the same two calls give 0.5 and 0.667 (run during fact-checking, see §4.1 and Part 6 row C12): positions where both codes are 0 are ignored and every other position compares the codes, i.e. a proper mismatch count except for the first category. Which scipy version the authors actually used for the published figures is [UNVERIFIED], and what ref. [8] of the 2023 paper defines exactly was not checked [UNVERIFIED]; the group's reimplementation should compute the categorical distance itself (e.g. the share of the m attributes on which the two cases differ) instead of calling `pdist('jaccard')` on integer codes, and should say so in the paper.
- **Pareto front:** P^l dominates P^j iff "∀ I_k ∈ I, I_k(P^l) is no worse than I_k(P^j)" and "∃ I_k ∈ I, I_k(P^l) is strictly better" (p.309); the front = the undominated patterns, computed with the skyline algorithm of [5]. **Skyline algorithm** = the database community's name for exactly this computation — the "skyline" of a set of points is the subset not dominated by any other point (Borzsony, Kossmann & Stocker, "The skyline operator", ref. [5] of the 2023 paper); the `paretoset` package does the same job in the code. Example with (CC ↑, OI ↑, CD ↓): P1 (0.6, 0.10, 0.30), P2 (0.3, 0.25, 0.30), P3 (0.3, 0.05, 0.35). P2 dominates P3 (equal CC, better OI, better CD); P1 also dominates P3; P1 and P2 trade off (P1 wins CC, P2 wins OI) → front {P1, P2}, no thresholds needed. Code caveat: `paretoset`'s default `distinct=True` keeps only the first of several patterns with identical objective rows (check_paper-2023 B3, C3).
- **Non-dominated sorting (Pareto layers) — how to fill a fixed k from a front:** ELI5: peel the onion — take the front (layer 1), remove those patterns, compute the front of what is left (layer 2), and so on until you have k patterns. Precisely: layer 1 = the Pareto front of all candidates; layer i = the Pareto front of the candidates not in layers 1..i−1; a fixed top-k is filled layer by layer, with a documented tie-break (e.g. by one interest function) inside the first layer that does not fit completely [ASSUMPTION: textbook definition from multi-objective optimisation, not from the papers or the course slides]. Tiny example with the three patterns above: layer 1 = {P1, P2}, layer 2 = {P3}; k = 2 → {P1, P2}; k = 3 → all three; k = 1 → tie-break between P1 and P2. Why we need it: Apriori hands us exactly 10 itemsets, IMPresseD hands us a front of unpredictable size (Part 4 recommends k = 10 per length by non-dominated sorting and reporting the full front size).
- **Project:** the group description says "this technique will return activities with their ordering relations (e.g., A->B). You have to consider only the activity sets, thus ignoring their relations. For the experiments, consider pattern lengths of 1, 2 and 3" (description_group11), so the variant projects patterns to **activity sets** (A→B, B→A, A⇢B all become {A,B}). [ASSUMPTION] "length" = number of distinct activities in the projected set (so A→A → {A}); the 2023 paper defines only length-1 = individual activities (p.307) and never defines longer lengths — node count and extension step are equally consistent readings (check_paper-2023 B13, D1); must be confirmed with the supervisor. Prior analysis recommends chain traces (`delta_time < 0`) for BPIC11, because with `delta_time = 0` the same-day blocks (§2.12) become large concurrent blocks (78% of parallel-flagged f1 events sit in blocks of ≥ 10 nodes; 31% of directly-following instances have > 3 nodes — check_paper-2023 A/B5, probes re-run there, not in this run) [UNVERIFIED in this run]. The 2024 paper itself names this variant as future work: "we intend to construct outcome-oriented patterns introduced in [19] rather than relying on frequent itemsets" (p.201). Open choices: Pareto front vs. fixed top-10 (non-dominated sorting, above); whole-log vs. train-only mining (the 2023 paper mines on the training set only, p.315; the 2024 code mines Apriori on the whole log — the leakage question of §2.8); scoring patterns vs. sets.

### 2.15 Information gain / mutual information

- **ELI5:** how much less surprised you are about the sticker once told how often the pattern occurred in the case.
- **Definition:** entropy H(Y) = −Σ p(y) log p(y); information gain of X = H(Y) − H(Y|X); for a discrete X this equals the mutual information I(X;Y) [ASSUMPTION: textbook identity, not from the course slides; the paper says "information gain" (p.311), the code calls `mutual_info_classif` (IMIPD.py:68)]. sklearn's MI "is equal to zero if and only if two random variables are independent, and higher values mean higher dependency" (docstring, 1.9.1); `discrete_features=True` treats the counts as categorical. Units: sklearn reports MI in nats (natural log), so a perfectly informative binary feature scores ln 2 ≈ 0.693, not 1 (checked numerically).
- **Example:** labels 1,1,0,0; counts 1,1,0,0 → IG = H(Y) = 1 bit = 0.693 nats (sklearn returns 0.6931); counts 1,0,1,0 → IG = 0 (sklearn returns 0.0).
- **Project:** OI for BPIC11's binary label. On the raw f1 log (1140 cases) the top-MI activity is `376400` with MI 0.364 (re-computed in this run with `discrete_features=True`; identical for count and presence encoding), followed by `ac419100` 0.333 and `ac370000` 0.240. `376400` occurs in 335 f1 cases, of which 99.7% carry label 1, versus 15.4% label-1 among cases without it. Whether this is a near-leak (an activity routinely ordered together with the cut-out tumor-marker tests — leakage in the sense of §2.8) or a legitimate clinical proxy is [UNVERIFIED]; the code-to-name mapping is unavailable (§2.3).

### 2.16 Rank-comparison measures for the two strategies

- **Jaccard.** ELI5: of everything in either basket, the share in both. J = |A∩B|/|A∪B|. Example: {{a,b},{a,c},{b,c}} vs {{a,b},{c,d}} → 1/4. Use: do the strategies pick the same sets at all? Not to be confused with the Jaccard *distance* on label-encoded case-attribute vectors inside IMPresseD's case distance (§2.14): here A and B are the two *sets of itemsets* (one per strategy); there u and v are two *cases'* attribute vectors, and on current scipy the latter is computed on booleanised codes.
- **Overlap@k.** ELI5: compare only the two top-k shortlists. |top-k(A) ∩ top-k(B)|/k. Example: {s1,s2,s3} vs {s2,s3,s7} → 2/3. Use: agreement on the most important sets.
- **Spearman's ρ.** ELI5: rank both lists and ask whether the rankings move together. Pearson correlation of ranks, "a nonparametric measure of the monotonicity of the relationship between two datasets", varying "between -1 and +1 with 0 implying no correlation" (scipy `spearmanr` docstring, 1.18.1); without ties ρ = 1 − 6Σd²/(n(n²−1)) [ASSUMPTION: textbook formula, not from the course slides; agrees with scipy on the example below]. Example: ranks (1,2,3) vs (1,3,2) → d² = 0,1,1 → ρ = 1 − 12/24 = 0.5 (scipy: 0.5). Use: on sets common to both strategies, ranked by mean importance.
- **Kendall's τ.** ELI5: concordant pairs minus discordant pairs, over all pairs. τ_a = (C − D)/(n(n−1)/2) [ASSUMPTION: textbook formula, not from the course slides; agrees with scipy on the example below]; scipy's `kendalltau` returns τ-b by default, and "both tau-b and tau-c reduce to tau-a in the absence of ties" (docstring). Example: (1,2,3) vs (1,3,2) → pairs (1,2) and (1,3) concordant, (2,3) discordant → (2−1)/3 = 0.33 (scipy: 0.3333). Use: robust to one misplaced item; report both.

### 2.17 Glossary

| Term | Meaning |
|---|---|
| Event / trace / case / log | one activity execution / ordered events of one case / one process instance / all traces |
| PPM / outcome-oriented PPM | predicting ongoing cases / predicting a categorical (here binary) final label |
| Label 1 / 0 (BPIC11 files) | LTL rule satisfied / violated (empirical; opposite of the paper's displayed formula) |
| LTL rule | Linear Temporal Logic: a sentence about the order / eventual occurrence of activities in a trace (F eventually, G always, U until) |
| Trace cutting | drop the label-revealing event and everything after it (f1/f3/f4 only) |
| Prefix / truncation | first k events / cap every trace at N events (`truncN`, N = min(40, ⌈90th pct. positive-case length⌉)) |
| Index-based / binary / frequency encoding | 0/1 per (position, activity) / activity present / activity count |
| One-hot encoding | a categorical column with m values → m 0/1 columns, exactly one of them 1 per row (`pd.get_dummies`; index encoding = one-hot per position) |
| Supervised learning: row / feature / label | one case (truncated trace) / one encoded 0/1 column / the known 0/1 answer the model must reproduce |
| Loss / held-out test set | number measuring how wrong the predictions are (`binary:logistic` = log loss) / rows never used for fitting, the only honest score |
| Overfitting | fitting the training rows' noise instead of the regularity; symptom = train score ≫ test score (f1: ≈0.99 vs ≈0.89, env_report) |
| Decision tree | flow chart of yes/no questions on the columns ending in a guess (in XGBoost: a leaf number added to the score) |
| XGBoost | gradient-boosted tree ensemble; the black box (library defaults) |
| Stratified K-fold CV | K rotating splits keeping class proportions (unseeded in the 2024 code) |
| Leakage | information from the test cases or from the label sneaking into fitting or candidate selection (whole-log Apriori, `Variant` columns, near-leak activity `376400`) |
| Box plot (`whis`, `showfliers`) | median line, box = middle 50% (IQR), whiskers to the furthest value within `whis`·IQR of the box (default 1.5; 2024 code: 10), dots = outliers (`showfliers=False` hides them) |
| Accuracy / precision / recall / F1 / weighted F1 | (TP+TN)/all / TP/(TP+FP) / TP/(TP+FN) / harmonic mean of P and R / per-class F1 weighted by support |
| Baseline (this paper) | weighted F1 on unpermuted training-fold data |
| Post-hoc / model-agnostic / global / factual | after training / inputs-outputs only / whole model / "why this prediction" |
| SHAP / LIME | per-feature contributions that add up to one prediction (Shapley values) / a simple local model fitted around one input |
| PFI | score drop after shuffling one column, averaged over repeats |
| argparse / CLI | Python's command-line-argument parser (`--top_k 10` → `args.top_k`) / command-line interface, the terminal |
| Location | 1-based `event_nr` of an event in its trace |
| Observed locations / feasibility / order preservation | positions where it occurs anywhere in the log / move only there / keep itemset order |
| Location importance / existence importance | baseline − F1 after relocation / classical PFI on presence (existence) features |
| Itemset / transaction / support | activity set / case as set of distinct activities / share of cases containing it |
| Apriori / min_support / top-k / IE(L) | level-wise frequent-set mining / threshold / k best / pluggable itemset extractor |
| DAG | directed acyclic graph: nodes and arrows with no directed cycle (a process pattern is one) |
| Process pattern / partially ordered trace / oracle | DAG of activities with relations / trace as DAG / rule deciding concurrency |
| Graph isomorphism | two patterns are the same drawing with the nodes relabelled (`nx.is_isomorphic`, used by the 2023 code to de-duplicate patterns) |
| Pattern extension / interest function | add nodes by one of six rules / coverage, outcome interest (IG or Spearman), case distance |
| Jaccard distance (case distance, IMPresseD) | scipy `pdist('jaccard')` on label-encoded case-attribute vectors — per-attribute mismatch between two cases; booleanised since scipy 1.15 (pitfall) |
| Dominance / Pareto front / skyline | no worse everywhere, strictly better somewhere / the undominated set / the database name for computing that set |
| Non-dominated sorting / Pareto layers | peel off the front, remove it, take the next front, … — used to fill a fixed k [ASSUMPTION: textbook definition] |
| Pattern length (variant) | distinct activities in the projected set [ASSUMPTION — paper defines only length-1] |
| Mutual information (nats) | sklearn's IG surrogate; ln 2 ≈ 0.693 for a perfectly informative binary feature |
| Jaccard / overlap@k / Spearman / Kendall | set overlap of two itemset lists / top-k overlap / rank correlation / pairwise rank agreement |

---

## Part 3 — The 2024 paper and its code, step by step

Conventions: (p.N) = printed LNBIP page (191 = PDF page 1); `tools.py:N`, `CVPP:N` (`CrossValidation_ProcessPermutation.py`), `Classical:N`, `dist:N`, `PDP:N`, `Pixel:N`, `Plotting:N` = repo line numbers (commit `76899ad`, the only commit). "(verified run)" = executed in the scout's Python 3.12 venv (pandas 2.3.3, numpy 2.5.3, xgboost 3.4.1, scikit-learn 1.9.1, mlxtend 0.25.0) by the fact-checker's scripts `_fc_C.py`, `_fc_C2.py`, `_fc_C3.py`, which re-run and extend the draft author's `_verify_C.py` / `_verify_C2.py`. "(render)" = the PDF pages rendered at 400 dpi and cropped (folder `_figs/`). Folds are unseeded (see §3.3.3), so any fold-level number below is a single draw, not a result.

### 3.1 The research question and the toy motivation (Table 1)

- **ELI5.** A doctor's visit has many steps. Existing explanation tools can tell you *whether doing a step matters* for the predicted outcome. This paper asks a different question: does it matter *when in the sequence* the step happens?
- **Research question (verbatim, 2024 paper p.193):** "Given a predictive model trained on a set of process executions, how can we assess the importance of the location of a group of process activities on the classifier performance?"
- **Location** = the ordinal position (1, 2, 3, …) of an event inside its case's trace, not a clock time ("different moments (locations)", "position 2", p.192). In code it is the 1-based column `event_nr` (`tools.py:72`, `:345`).
- **Toy log (Table 1, p.192).** Six sepsis traces, label = patient returns to the emergency room within 28 days of discharge (p.192). Every trace contains LA (Lactic Acid), so *existence* of LA is useless. But LA sits at position 2 in exactly the three Negative traces (1, 3, 6), and at positions 8, 4+11, 8 in the Positive traces (2, 4, 5) (verified run, re-counted from Table 1: trace 4 = ⟨RG,LE,LE,**LA**,ER,ST,CR,IA,IC,LE,**LA**,NC,DI⟩). So *location* of LA separates the classes perfectly:

```
case 1 (Neg): RG LA LE CR ER ...            LA @2
case 2 (Pos): RG ER ST IL LE CR IA LA ...   LA @8
```

- **Why classical permutation feature importance (PFI) cannot answer it (p.194–195).** Under index-based encoding "LA" is spread over e1_LA, e2_LA, …; shuffling one column alone yields traces with LA twice or nowhere, and moving one activity shifts the others ("shifting the location of an activity inevitably impacts also the location of other activities in the trace", p.195). Hence a sequence-aware permutation.

### 3.2 Figure 2: the pipeline and the three contributions

Fig. 2 (p.195, checked on the rendered page) is a flow with two branches that share the *Event log* and the *Trained model*:

- Upper branch (standard PPM): Event log → Index-based encoding → Model training → Trained model → Performance assessment → **Baseline performance**.
- Lower branch (dashed box "For each itemset"): Event log → *Itemsets selection* (hatched) → *Location permutation* (hatched) → Index-based encoding → Performance assessment (fed by the Trained model) → *Itemsets location importance* (hatched hexagon) → Visualization.

The three hatched elements are the paper's stated contributions ("The steps depicted with hatched patterns represent this research's primary focus and contribution", p.194); the rest is standard outcome-oriented PPM.

The method is **post-hoc, model-agnostic** (p.193: "a novel post-hoc model-agnostic method"), **global** (p.194: "measuring the global importance of the location") and **factual** (p.194: "this paper's primary focus remains on factual explanations"): it only needs the trained model's predictions, and it explains the model as a whole, not one case.

### 3.3 Step by step: ELI5, precise definition, code location

#### 3.3.1 Preprocessing — `DataManager._load_df` (`tools.py:36-78`)

- **ELI5.** Clean the names, line the events up in order, throw away patients whose story contains an ultra-rare step, and write down every position at which each step has ever been seen.
- **Precise.** `pd.read_csv`; case id cast to `str` (`:38`); activity lower-cased (`:39`) and stripped of spaces, `-`, `_` (`:48-50`), so `AC370000` → `ac370000`, `W_Nabellen offertes` → `wnabellenoffertes` (verified run). If a `lifecycle:transition` column exists — only the three bpic2012 CSVs have one; the BPIC11 and sepsis CSVs do not (verified run, CSV headers) — keep COMPLETE events and renumber `event_nr` (`:41-46`). Sort by case and `event_nr` (`:55`); the timestamp column is never used for ordering (the timestamp lines are commented out, `:51-53`). Map `deviant`→1, `regular`→0 (`:57-58`; a no-op for BPIC11, whose labels are already 0/1, labels_report). **frq_threshold filter:** count *events* (rows) per activity; every *case* containing an activity with fewer than `frq_threshold=2` events in the whole log is dropped (`:60-64`). For BPIC11 this leaves 1130 cases / 164 activities (f1), 1130 / 207 (f2), 1111 / 156 (f3) (verified run). **`Allowed_locations[act]`** = sorted unique `event_nr` values at which `act` occurs anywhere in the (whole, filtered, unsplit) log (`:70-73`). This is the feasibility pool used later (verified run: `Allowed_locations['ac370000']` in f1 = [1, …, 36], every position). `L_max` = 0.8-quantile of `event_nr` (`:75`; 24.0 / 31.0 / 22.0 for f1/f2/f3, verified run) is computed but never used by the main script.
- Side note: `dist_location_calculator.py` (`dist:33-49`) and `Pixel_Flipping_Process.py` (`Pixel:34-52`) re-read the raw CSV with the same name cleaning but *without* the frequency filter, so they see all 1140 BPIC11 f1 cases, not 1130.
- **Tiny example.** Log {⟨A,B⟩, ⟨B,A,C⟩}: `Allowed_locations = {A:[1,2], B:[1,2], C:[3]}`.

#### 3.3.2 Index-based encoding — `index_encoding` (`tools.py:338-364`)

- **ELI5.** Turn every trace into one long row of 0/1 switches: "is activity X at position i?".
- **Precise.** `pivot_table(index=case, columns=event_nr, values=activity, fill_value=0)` gives columns e1..eN (N = longest trace *in the frame passed in*, `:340`); each eI is one-hot encoded with `get_dummies(prefix=eI)` (`:352-354`), so features are `e{i}_{activity}` plus `e{i}_0` for padded positions (the `fill_value=0` becomes its own dummy); missing (position, activity) combinations are added as all-zero columns from `self.data`'s activity set (`:359-362`). Feature counts (verified run): f1 36 × 164 + 35 padding = **5,939**; f2 40 × 207 + 39 = **8,319**; f3 31 × 156 + 30 = **4,866** (position 1 is never padded, hence N−1 padding columns). pandas inserts the dummy columns one by one internally and emits one `PerformanceWarning` ("highly fragmented") per insert — 5,842 warnings on f1 (env_report P4); harmless.
- **Tiny example.** Activities {A,B}, max length 3, trace ⟨B,A⟩ → `e1_A=0 e1_B=1 e2_A=1 e2_B=0 e3_0=1 e3_A=0 e3_B=0`.
- The whole (truncated) log is encoded once with `All_prefixes=False` (`CVPP:36`, `:69`): one row per case, no prefix generation. The paper never mentions prefixes (the word does not occur in the text).
- **Version trap (env_report P1).** In pandas ≥ 2 `get_dummies` returns `bool`, the zero-filled columns are `int64`; writing the permuted rows back (`tools.py:536`) up-casts to `object` and XGBoost refuses. Two fixes are now both verified: (a) cast all feature columns to `int` after encoding (scout); (b) monkey-patch/edit `pd.get_dummies(..., dtype=int)` at `:353` — with (b) all 5,939 f1 columns come out `int64` and `itemset_permutation_importance` runs without warnings (verified run: 1 itemset × 1 repeat on fold 0, 7.2 s, importance 0.0443).

#### 3.3.3 K-fold split, XGBoost, baseline (`tools.py:214-244`, `CVPP:71-92`)

- **ELI5.** Deal the patients into 5 piles; 5 times, learn from 4 piles and check on the 5th — *but* in this code the "check" on the 5th pile is only printed; the location-importance game is then played on the same 4 piles the model learned from.
- **Precise.** `StratifiedKFold(n_splits=5, shuffle=True)` over cases, stratified by label, **no `random_state`** (`tools.py:231`) → different folds on every run (env_report P3: fold-0 test F1 0.889 vs 0.906 in two runs; verified run: 0.894 in a third). Per fold: `XGBClassifier()` with library defaults (`CVPP:87`), fit on train rows; the test weighted F1 is only printed (`CVPP:92`), never saved. **Baseline** = weighted F1 of the fitted model on *its own training fold* (`tools.py:503-504`, called with `train_x, train_y` at `CVPP:95-98`). Train F1 ≈ 0.99 vs test ≈ 0.89–0.91 on f1 (env_report; verified run fold 0: train 0.9945, test 0.894). That gap is **overfitting / memorisation of the training fold**: the 100-tree, depth-6 booster (next bullet) reproduces almost every label it was fitted on but scores about 10 F1-points lower on cases it never saw — exactly why §2.6 (step 4) says to evaluate on unseen rows, and consistent with the 0.992 training F1 on all 1130 f1 cases reported in §2.7. Consequence for this paper: because the baseline is the *training* score, every location-importance value is a drop from a near-memorised ≈ 0.99, not from the ≈ 0.89 the model reaches on new cases (discrepancy #1 in §3.6).
- **Which defaults?** In the venv's xgboost 3.4.1, `get_params()` reports `None` for `n_estimators`/`max_depth`/`learning_rate` (xgboost ≥ 2 passes unspecified parameters through to the native library), and the fitted booster's config shows 100 trees, `eta` 0.3, `max_depth` 6, `subsample` 1, `colsample_bytree` 1, `min_child_weight` 1, objective `binary:logistic` (verified run). The xgboost version the authors used is not pinned anywhere in the repo (no `requirements.txt`), so the defaults behind the paper's figures are [UNVERIFIED]. With `subsample`/`colsample` = 1 the fit itself is deterministic; the only unseeded randomness is the fold split.
- **Weighted F1** (`f1_score(average='weighted')`): per-class F1 averaged with weights = class share. Tiny: classes 0/1 with F1 0.9/0.6 and shares 0.6/0.4 → 0.9·0.6 + 0.6·0.4 = 0.78.

#### 3.3.4 Itemset selection via Apriori — `frequent_activity_sets` (`tools.py:160-177`)

- **ELI5.** "Which groups of steps appear together in most patients' stories?" — like finding products often bought together (the basket intuition, the definition of support and a 4-case worked example: see §2.13; the real f1/f2/f3 output is listed only here, below).
- **Precise.** One **transaction per case** = the list of its activities (`:161-163`); `TransactionEncoder` reduces it to presence/absence (duplicates and order vanish, `:164-166`). `mlxtend.apriori(min_support, use_colnames=True)` on the *whole log* (before any split, `CVPP:51` vs `:71`). **Support** = fraction of cases containing all items. Then: sort by support, `head(top_k + #activities)` (`:168-169`), drop itemsets of size 1 (`:171-172`), keep `top_k` (`:174`). Returns `list[list[str]]`, which `CVPP:61-63` turns into `{0: [...], 1: [...]}`. Fewer than `top_k` itemsets are returned silently when support is scarce.
- **Tiny example.** Cases {A,B}, {A,B,C}, {A,C}; min_support 0.6 → {A} 1.0, {B} .67, {C} .67, {A,B} .67, {A,C} .67 ({B,C} and {A,B,C} are .33, out); after the size>1 filter: {A,B}, {A,C}.
- **Real output (verified run, min_support 0.5, top_k 10):**
  - f1: {370407, ac370000} .591, {ac370000, ac370419} .577, {ac370000, ac370443} .564, {ac370000, ac370419, ac370443} .562, {ac370419, ac370443} .562, {ac370000, ac370442} .559, {370407, ac370000, ac370419} .559, {370407, ac370419} .559, {ac370442, ac370443} .557, {ac370000, ac370419, ac370442} .557 — ten sets of size 2–3 from only **five** activities; ranks 9–12 tie at 0.5566, so which one takes rank 10 depends on pandas sort tie-breaking (`:168`, `:174`).
  - f2: {ac370000, ac379999} .690 first, then {ac370000, ac419100} .647, …; ten sets found.
  - f3: only **5** sets of size > 1 reach 0.5 ({370407, ac370000} .528, {ac370000, ac370419} .513, {370407, ac370000, ac370419} .502, {370407, ac370419} .502, {ac370000, ac370443} .501); at 0.49 there are 10, the extra five being {ac370000, ac370419, ac370443} .4995, {ac370419, ac370443} .4995, {ac370000, ac370442} .496, {ac370000, ac370442, ac370443} .494, {ac370442, ac370443} .494.
- **Single-activity vs multi-activity mode** (`--Multi_activity`, `CVPP:21-25`, `:48`, `:94-104`): multi → `itemset_permutation_importance` on the Apriori sets; single → `trace_permutation_importance` on **every** distinct activity (`tools.py:376`), not on Apriori singletons. The paper plots the top 20 single activities (p.197).
- **How the two functions differ** (both: seed 2023, n_repeats 10, train baseline, in-place `sub_data`):

| | `trace_permutation_importance` (`tools.py:366-448`) | `itemset_permutation_importance` (`:498-552`) |
|---|---|---|
| Loop | all activities (`:376`) | given itemsets (`:509`) |
| Occurrences | every occurrence, positions precomputed once (`:393`) and stale after the first move (`:404`) | greedy non-overlapping complete occurrences via `find_itemset_indexes` (`:478-496`) |
| Target pool | `constrain=True`: `Allowed_locations` clipped to trace length (`:397`); `False`: any index (`:401`) | `constrain` **never read**; always `Allowed_locations`, **not** clipped (`:461-467`) |
| Zero rules | <2 allowed locations → 0 (`:378-384`); <3 shuffled cases → 0 (`:408-413`) | none; a trace equal in length to the itemset is skipped (`:522-523`); zero shuffled cases → empty frame to `index_encoding` → `ValueError: cannot convert float NaN to integer` at `:340` (verified run on an empty frame) |
| Output | transposed frame, columns = activities (`:432-448`) | columns = itemset keys (`:546-552`) |

#### 3.3.5 Location permutation — `shuffle_sequence` (`tools.py:450-476`)

- **ELI5.** Pick the chosen steps out of a patient's story and re-insert them at other positions where such steps have been seen before, keeping their mutual order; then ask the model again.
- **Two constraints (p.196).** *Feasibility:* an activity may only land on positions in its observed-location pool ("the observed locations of the occurrences of that activity throughout the event log"). *Preserving ordering relation:* inside one itemset occurrence the activities keep their relative order (if CR came before ER, it still does).
- **Observed locations.** The paper defines joint tuples OL(L,(ER,CR)) = {(5,4),(10,11),(2,6),(3,6),(5,7),(2,5),(4,3)} for Table 1 (p.196); running `find_itemset_indexes` on the six traces reproduces exactly these seven tuples (verified run). Case 1 has two complete occurrences plus a leftover CR that has no partner and is ignored ("only the first complete occurrence is considered", p.196). The code uses *per-activity* pools `Allowed_locations[act]` (all occurrences of that activity, whole log), which is also how the paper's own Feasibility Constraint sentence is worded; the joint tuples appear only in the example.
- **Before/after (paper's example, p.197):**

```
sigma1  : RG LA LE [CR ER] ST IL IA NC [ER CR] LE CR      positions: CR@4 ER@5 | ER@10 CR@11 | CR@13
sigma1' : RG LA [CR ER] [ER] LE [CR] ST IL IA NC LE CR    positions: CR@3 ER@4 | ER@5 CR@7  | CR@13
```
Same multiset, same length (verified run). The only reading consistent with the OL set is: occurrence 1 (CR@4, ER@5) → (CR@3, ER@4) = OL tuple (ER,CR) = (4,3); occurrence 2 (ER@10, CR@11) → (ER@5, CR@7) = OL tuple (5,7); the lone CR@13 is untouched; order inside each occurrence preserved. [ASSUMPTION: σ1′ occurrence assignment — the paper does not say which occurrence went where; this reading is inferred, and the alternative assignment would give tuples (5,3) and (4,7), which are not in OL.]

- **Code walk (per occurrence `IDX`, activities in trace order):** backward pass computes `max_possibles[act]` = largest allowed position strictly below the next activity's maximum (`:459-462`); forward pass draws `random_index` uniformly from `Allowed_locations[act]` within `(previous_index, max_possibles[act]]` (`:465-467`) — this window replaces the paper's redraw loop and is never empty once the backward pass succeeded (each window contains its own `max_possibles`); then `pop` the activity at its *current* index and `insert` at `random_index-1` (`:469-470`); then update the bookkeeping dict `locations` (`:471-474`). The pool is not clipped to the trace length, so a draw beyond the end silently appends (Python list semantics): in f1, 341 of 1130 cases are shorter than 10 events (median length 25) while `ac370000` may be drawn anywhere up to 36 (verified run).
- **Backward-pass failure mode.** `max([])` at `:461` raises `ValueError` when no allowed position of an earlier activity lies below the later activity's maximum (toy: pools A:[5], B:[3], trace ⟨A,B⟩ — verified run). For the f1/f2/f3 top-10 itemsets at support 0.49, no ordering of the itemset activities triggers it (verified run, all permutations checked), so it is a theoretical risk for rare activities only.
- **The bookkeeping bug (verified on toys and on f1).** `locations` is meant to map original index → current index. After a rightward move from `p` to `q = random_index-1`, elements originally at `p+1..q` shift left by one, but `:472-474` decrements only keys with `p < key < q` (strict), so the key `q` stays stale and now points at the *moved activity itself*; `:471` additionally stores the new position under the key `current_act_index` rather than the original index. Toy (verified run): trace ⟨A,B,C,D,E⟩, itemset {A,C}, all positions allowed, draws A→3, C→4. Move 1: pop A, insert at index 2 → ⟨B,C,A,D,E⟩; `locations[2]` still = 2 but position 2 now holds **A**. Move 2 pops index 2 → A again → ⟨B,C,D,A,E⟩: C never moved and A ends *after* C. Monte-Carlo (verified run, 20,000 draws each): single-occurrence 8-event toy with uniform pools: order violated ≈14% (2-itemset {B,E}) and ≈52% (3-itemset {B,D,F}), non-itemset activities never moved; random traces with **two or three occurrences** of a 2-itemset: a non-itemset activity is displaced in ≈2% / ≈7% of traces. On real f1 traces with exactly one occurrence of {ac370419, ac370442} (612 traces, seed 2023): order reversed in 21 (3.4%) and an itemset activity lands on a position never observed for it in 19 (3.1%) — the check report measured 3.5% / 2.8% on a slightly different trace selection. Net effect: neither constraint is strictly enforced, and in multi-occurrence traces activities outside the itemset can move too.
- Permutation only touches training traces that contain the itemset (`:521`); their rows are re-encoded (`:529`), zero-padded to `X`'s columns (`:530-534`) and written over the corresponding rows of a copy of `X` (`:535-536`).

#### 3.3.6 The importance score and the box plot

- **ELI5.** Importance = how many points the model loses when you scramble the step's location.
- **Precise (from code; the paper gives no formula, only "difference between the baseline performance and the performance on the permuted event log", p.195):** for fold k, repeat r, itemset I: LI_{k,r}(I) = F1_w(y_train_k, M_k(X_train_k)) − F1_w(y_train_k, M_k(X̃_train_k^{I,r})) (`tools.py:503-504`, `:538-539`). 5 folds × 10 repeats (`n_repeats=10`, `:499`) = **50 numbers per itemset**; `CVPP:108-111` concatenates them, sorts columns by mean, saves `location_importance_<data>_Multi.csv` (`CVPP:121`) and a horizontal box plot (`whis=10`, `CVPP:126`).
- **Which script drew Fig. 3/5?** `Plotting_results.py` draws a 3×3 grid, top 20 columns, `showfliers=False`, green = location, blue = existence (`Plotting:18-51`), labels truncated to 13 characters (`Plotting:7-14`, which is why Fig. 5 shows `['370407',...`), reading hand-renamed `<data>_LI.csv` / `<data>_EI.csv` (`Plotting:27-33`). Its legend text "Itemsets location importance / Itemsets existence importance" (`Plotting:68`) is exactly Fig. 5's legend (render). Fig. 3's legend reads "activity location importance / activity existence importance" (render), which no shipped line produces, so Fig. 3 came from a modified copy of this script [ASSUMPTION: Fig. 3 code state — not in the repo].
- **Accumulation.** `sub_data` is created once per call (`:507`) and overwritten in place for every shuffled case (`:526`) inside the loops over itemsets (`:509`) and repeats (`:510`); it is never reset. Repeat r therefore permutes an already permuted log, and itemset #k is scored on traces still carrying permutations of itemsets #1..k-1. Lower bound by pure set logic (verified run, f1 top-10 in support order): 96.9% of the cases containing itemset #2 also contain itemset #1 (hence were permuted 10 times already), 99.7% for #3 and 100% from #4 on — matching the check report's 95–100%. Each fold starts fresh because `CVPP:95` calls the function anew and `np.random.seed(2023)` is reset at `:501`.

#### 3.3.7 Existence importance — `Classical_Permutation.py`

- **ELI5.** The old-fashioned question: "does it matter *whether* the step happened at all?"
- **Encoding.** Single mode: `binary_encoding` (1 if the activity occurs, `tools.py:286-302`). Multi mode (hard-coded `Multi_activity=True`, `Classical:17`): reads the itemsets back from the location CSV header with `eval` (`:29`, `:56`), joins each trace into one string with `'->'` (`:44-45`) and encodes an itemset as `min(trace.count(act) for act in itemset)` (`:57`) — a **count**, not the binary feature the paper describes (p.197: "binary encoding, where each feature represented a distinct itemset"), and `str.count('370407')` also matches `370407c`, which exists in f1 and f2 but not f3 (verified run), so that feature is inflated there.
- **Model and PFI.** A fresh `XGBClassifier()` on the existence features (in multi mode: only the 10 itemset columns), new unseeded 5-fold split, `frequency_threshold=1` (`:18`, i.e. no case is dropped: 1140 instead of 1130 f1 cases — different cases and folds from the location run); `sklearn.inspection.permutation_importance(model, train_x, train_y, n_repeats=20, random_state=42, n_jobs=2)` on the **training** data (`:108-109`). `scoring` is not passed (default `None`, verified run) → sklearn uses `estimator.score`, and `XGBClassifier.score` resolves to `ClassifierMixin.score` = **accuracy** (verified run on the venv's scikit-learn 1.9.1) [ASSUMPTION: older-sklearn scorer — the authors' unpinned scikit-learn is assumed to resolve `estimator.score` to accuracy in the same way]. The axis label "Decrease in f1 score" (`:143`, and `Plotting:66`) is therefore wrong for the blue boxes. 5 × 20 = 100 values per feature. SHAP `TreeExplainer` mean |value| is also computed (`:99-106`) but not shown in the paper. The script cannot run as-is: dataset, result folder and the location CSV it needs are hard-coded (`:19-21`).

#### 3.3.8 Location-distribution plots — `dist_location_calculator.py` (Fig. 4, Fig. 6)

- **ELI5.** A bar chart: for each (start) position, how many cases of each outcome had the step(s) there.
- **Precise.** Takes the first 10 columns of the location CSV (`dist:11`, `:52`; the CSV is already sorted by mean importance); for each itemset, keeps cases containing it (`:59-65`), takes for every itemset activity the **first** `event_nr` in Python set-iteration order (`:74-75`), reduces each case to the *first element* of that list (`'fristloc'`, `:90`) and draws stacked bars of label counts per start position (`:89-94`). The paper describes this aggregation "by starting point" (p.199). The legend is hard-coded `['positive','negative']` (`:99`) while the pivot columns are the sorted labels [0,1], so **label 0 is drawn as "positive"** (blue, matplotlib's first colour). Consistently: (a) Fig. 4a's "accepted in 61% of the cases" (p.198) equals the per-*event* share of label 0 among `wnabellenoffertes` events in bpic2012_1 (0.610 over 19,096 events; per case it is 0.519 over 4,647 cases), positions 10–28 (verified run); (b) in sepsis_1, 779 of 782 cases contain `leucocytes` and only 14.1% of them have label 1 (verified run), and Fig. 4b's dominant blue class is labelled "positive" (render). The single-activity plotting block is commented out (`:103-119`) and Fig. 4a's legend "accepted / not accepted" (render) is produced by no shipped line, so Fig. 4 came from a code state not in the repo [ASSUMPTION: Fig. 4 code state — not reproducible from the shipped script].

#### 3.3.9 `PDP_ProcessLocation.py` (not in the paper)

A partial-dependence-style curve: for every activity and fold, train XGBoost (`PDP:20-38`, i.e. one model per activity × fold), then (`tools.py:554-589`) move the activity's *first* occurrence (`trace.remove` + `insert`, `:568-569`) to each of its allowed locations in every training trace containing it, re-encode, and average `predict_proba[:,1]` (`:585`); the script plots mean ± std across folds (`PDP:65-72`). It is broken as shipped: `PDP:25`, `:30` and `tools.py:576` drop `concept:name`/`time:timestamp` columns that the encoded frame does not have → `KeyError` (read; consistent with the check report's run). Output path hard-coded to `./datasets/bpic2012_1/PDP/` (`PDP:84`).

#### 3.3.10 `Pixel_Flipping_Process.py` (not in the paper)

A faithfulness check: rank items by location importance, classical PFI and SHAP (CSV inputs `Pixel:24-26`); each item's "neutral" location = mode of its observed `event_nr` (`:60-82`). Per fold and item: (a) set the item's existence column to its mode and re-score train F1 (`:176-190`; in multi mode the write hits a boolean-indexed copy, `:181-183`, so nothing changes); (b) move the item to its neutral location in every training trace and re-score (`:193-235`). Items are neutralised one at a time from fresh copies (`:176`, `:193`), so it is *not* cumulative; the PFI and SHAP curves share identical F1 values and differ only in rank (`:189-190`). It calls `prefix_generator` (`:84`), whose `DataFrame.append` (`tools.py:92`) is gone in pandas ≥ 2, and the prefix rows would be discarded anyway because the fold lists hold original case ids while prefix ids are `<case>_<i>` (`tools.py:87`; `Pixel:136`, `:152`). `Process_Permutation.py` is legacy (private dataset `:14`; `frequency_encoding()` called without its required `data` argument `:16`); ignore it.

### 3.4 Parameters

| Parameter | Default | Where | Meaning / remark |
|---|---|---|---|
| `--address` | `./datasets/bpic2012_1_trunc40.csv` | `CVPP:11-14` | input CSV; name derived by `split('/')` (`:44`) |
| `--constrain` | True | `CVPP:16-19` | `type=bool`: `--constrain False` is still True (`bool('False')` is True, verified run); dead in multi mode |
| `--Multi_activity` | True | `CVPP:21-25` | same bool trap; single mode needs `--Multi_activity ""` or an edit |
| `--top_k` | 10 | `CVPP:27` | `type=float`, but the default is the literal `int 10`, and argparse applies `type` only to strings (values typed on the command line, or a default given as a string). Verified in the venv (Python 3.12): `parse_args([])` → `10` (`int`); `parse_args(['--top_k', '10'])` → `10.0` (`float`); `DataFrame.head(5.0)` raises `TypeError: cannot do positional indexing on RangeIndex with these indexers [5.0] of type float`. So the script **crashes only when `--top_k` is passed explicitly; the default int 10 works**: `.head(10 + #activities)` at `tools.py:168-169` is fine, `.head(10.0 + #activities)` is not. env_report P2's "crashes immediately with default CLI args" generalises its own call `frequent_activity_sets(0.5, 10.0)` (env_report [2b]), i.e. the explicit-flag case, and is wrong for the default. Fix: `type=int` |
| `--min_support` | 0.5 | `CVPP:28` | not stated in the paper |
| `candidate_selection_method` | `'apriori'` | `CVPP:37` | `'optimized'` is a stub returning None (`tools.py:97-98`); `len(None)` at `CVPP:56` raises before the None check at `:58` |
| `K_fold` | 5 | `CVPP:39` | unseeded `StratifiedKFold` (`tools.py:231`) |
| `All_prefixes` | False | `CVPP:36` | whole truncated traces |
| `frq_threshold` | 2 | `tools.py:25` | case filter; Classical uses 1 (`Classical:18`) |
| `L_max_perc` | 0.8 | `CVPP:38`, `tools.py:75` | computed, unused |
| `n_repeats`, `random_state` | 10, 2023 | `tools.py:366`, `:499`, `:501` | permutation repeats and seed |
| model | `XGBClassifier()` | `CVPP:87` | library defaults, unpinned (venv 3.4.1: 100 trees, depth 6, eta 0.3) |
| metric | weighted F1 on train fold | `tools.py:504`, `:539` | |
| classical PFI | `n_repeats=20, random_state=42, n_jobs=2`, accuracy | `Classical:108-109` | |
| plots | top 20 / `whis=10` / `showfliers=False` | `Plotting:18`, `CVPP:126`, `Plotting:45,49` | |

### 3.5 Evaluation and results of the paper

- **Datasets (p.197).** bpic2011, bpic2012 and Sepsis "with the same labeling strategy as in [18]" (Teinemaa et al. 2019). Nine panels per figure, titles read (render): bpic2011 f1 / f2 / f3, bpic2012 accepted / cancelled / declined, sepsis 1 / 2 / 3; the repo ships these ten CSVs including the unused f4. In the shipped BPIC11 CSVs label 1 = the LTL rule is *satisfied* (labels_report), e.g. f1: a CA-19.9/CA-125 tumour-marker test eventually happens (trace cut before it).
- **Fig. 3 (p.198), single activities.** Top-20 by location importance (most important at the bottom, because column 0 of the sorted CSV is drawn at y = 0); main message: "most activities demonstrate high significance based on their location, while their existence is insignificant" across the three bpic2011 logs (p.198). Numbers the paper states: `W Nabellen Offer` in bpic2012 accepted → "decrease in performance between 12% and 14%" by location, negligible by existence; `ac370000` in bpic2011 f1 has "notably high importance in relation to its existence"; in sepsis, `leucocytes` (s1), `release A` (s2), `release B` (s3) are existence-driven. By-eye values from the 400-dpi render (±0.01, boxes = interquartile, whiskers in brackets): f1 location boxes ≈ 0.055–0.09 (whiskers to ≈ 0.11), `ac370000` existence box ≈ 0.245–0.27 (0.23–0.28), all other f1 existence boxes ≈ 0; f2 location boxes ≈ 0.14–0.19, `376400` existence box ≈ 0.29–0.31 (0.26–0.33). These agree with the check report's ranges.
- **Fig. 4 (p.199).** 4a: `wnabellenoffertes` → "accepted in 61% of the cases" overall but "from 50% to 70% in different locations" (p.198; the figure annotates "50% accepted" and "70% accepted"; verified run: the band holds for the populated positions, the sparse tail positions 27–28 give 0.37 and 1.00; "accepted" = label 0, see §3.3.8). 4b: `leucocytes` is tied to the first-listed class (label 0, the 86% majority class) with little variation over location.
- **Fig. 5 (p.200), ten itemsets.** "The existence of the frequent itemsets does not exhibit much importance" — the authors' own explanation: frequent ≠ predictive (p.199). Exception named: {ac370000, ac370419} in f1 has noticeable existence importance; the rest are "primarily influential due to their location". By-eye from the render: f1 location boxes ≈ 0.06–0.10; the `['ac370000...` row's existence box ≈ 0.24–0.28; two further f1 rows have existence boxes ≈ 0.04–0.10 and ≈ 0.075–0.09, about as large as their location boxes (not discussed in the paper); f3 location boxes ≈ 0.20–0.29, and one `['ac370442...` row has an existence box spanning ≈ 0.01–0.23. The f3 panel shows ten rows, three of which start with `ac370442` (render) — itemsets that only exist below support 0.5 (§3.3.4).
- **Fig. 6 (p.201).** 6a (f2, highest location importance) {ac379999, ac370000} (support .690, the top Apriori set in f2, verified run): many start positions (x-axis 1–40) and a strongly varying class ratio; 6b (f1, highest existence importance) {ac370000, ac370419}: an almost constant ratio, peak at start positions 9–10.
- **Not reported.** The models' own test F1/AUC; any numeric table; statistical tests or confidence intervals; `min_support`, `top_k`, hyper-parameters, runtime; whether importance is on train or test data; the legend semantics; the number of itemsets actually found per log.
- **Authors' limitations (p.200).** (1) The score reflects classifier performance, not necessarily true activity–outcome relations → expert validation needed. (2) One activity's location is "inherently intertwined" with the others; repeats mitigate but do not quantify this. (3) "The risk of placing activities in prohibited locations due to dependencies persists." (4) Design choices matter, "as previous studies discussed that frequent patterns are not necessarily the most predictive ones [19]". (5) Location and existence values are not on a common scale because the two use different encodings (p.197).
- **Future work that motivates our variant (p.201, verbatim):** "we intend to construct outcome-oriented patterns introduced in [19] rather than relying on frequent itemsets" — [19] is the IMPresseD paper. Also: measure the importance of the *order* within an itemset, and better feasibility of permuted traces.

### 3.6 Paper vs code: discrepancy table (each entry re-verified in the code)

| # | Paper says | Code does | Evidence |
|---|---|---|---|
| 1 | performance "through k-fold cross-validation" (p.195) | baseline and permuted scores on the **training** fold | `CVPP:95-98` pass `train_x, train_y`; `tools.py:503-504`, `:538-539` |
| 2 | each itemset permuted independently, 10 times | permutations **accumulate** across repeats and itemsets within a fold | `sub_data` built once `:507`, mutated `:526`, loops `:509-510`; 97–100% overlap of case sets (verified run) |
| 3 | relative order preserved | bookkeeping keyed by current index with a strict, key-based window → wrong element popped, order can flip; in multi-occurrence traces non-itemset activities move | `:469-474`; toy ABCDE/{A,C} → BCDAE; 3.4% reversed on f1 single-occurrence traces (verified run) |
| 4 | feasibility = observed locations "throughout the event log" | pool from the **whole** log incl. test folds, all occurrences, not clipped to trace length (append beyond end); the bug of #3 also lands activities on unobserved positions (3.1% on f1) | `tools.py:70-73` (built in `_load_df`, before `CVPP:71`); `:461-467` |
| 5 | reproducible 5-fold CV implied | folds unseeded | `tools.py:231` |
| 6 | README documents CLI flags | `type=bool` flags cannot be switched off; `--top_k` works only when omitted (explicit value → float → `.head()` TypeError) | `CVPP:16-27`; `tools.py:168-169`; argparse check (verified run) |
| 7 | top-10 frequent itemsets per log (Fig. 5 shows 10 for f3) | default `min_support=0.5` yields only **5** size>1 itemsets on f3; 0.49 yields 10 incl. three `ac370442` sets, and the f3 panel shows three rows starting with `ac370442` | verified run of `frequent_activity_sets`; render of Fig. 5; hence the authors used a support below 0.5 (≤ ~0.494) for f3 [ASSUMPTION: f3 support < 0.5 — inferred from the render, not stated in the paper] |
| 8 | existence baseline: binary encoding, "decrease in f1-score" | min-count encoding with substring `str.count`; metric = accuracy (sklearn default scorer); different case filter and folds | `Classical:44-58`, `:108-109`, `:18`; `370407c` present in f1/f2 (verified run) |
| 9 | overlapping occurrences: only "the first complete occurrence" | greedy non-overlapping occurrences; consistent with the paper's OL example | `tools.py:478-496`; OL set reproduced (verified run) |
| 10 | IE(L) example includes singletons like {RG} | size-1 sets dropped in multi mode; singles use a different routine | `tools.py:171-172` |
| 11 | random draw, redraw on order violation | windowed draw, no rejection; `max()` of an empty list possible if pools do not chain — reproducible on a toy, not triggered by any BPIC11 top-10 itemset (verified run) | `:459-467` |
| 12 | `constrain` flag advertised in README | ignored in multi mode | `:498-552` never reads it |
| 13 | legends "accepted / not accepted", "positive / negative" | hard-coded legend over sorted labels [0,1] → label 0 drawn as positive/accepted | `dist:99`; §3.3.8 (verified run); sepsis_1 14.1% label 1 |
| 14 | cites "an improved Apriori" [24] | plain `mlxtend.apriori` | `tools.py:15`, `:167` |
| 15 | Fig. 3 legend "activity location/existence importance" | shipped `Plotting_results.py` writes "Itemsets …" (matches Fig. 5 only) | `Plotting:68`; render |

### 3.7 Runtime expectations

- Measured (env_report, f1, 904 train cases, XGBoost 3.4.1, pandas 2.3.3, int-cast workaround): `DataManager` 0.24 s; `index_encoding` of 1130 cases ≈ 3 s (3.4 s in the verified run); XGBoost fit ≈ 2.5–2.9 s per fold; **≈ 6.7–7.2 s per (itemset, repeat)**, dominated by the per-case pandas filter loop (`tools.py:512-514`, ~900 `DataFrame` filters per repeat) plus one `index_encoding` of the shuffled subset (`:529`).
- Repo defaults, 5 folds × 10 itemsets × 10 repeats = 500 iterations ≈ **56–60 min per BPIC11 log** [ASSUMPTION: linear extrapolation from 1–2 measured iterations]; f2 (trunc40, 8,319 features) and f3 (trunc31, 4,866 features) scale with cases × trace length [ASSUMPTION: scaling rule — not measured on f2/f3]. `find_itemset_indexes` enumerates C(len, k) combinations (`:481`): 9,880 for k = 3 and length 40, fine for k ≤ 3 but explosive beyond.
- Single-activity mode (the Fig. 3 reproduction): `trace_permutation_importance` loops over **all** 164 / 207 / 156 activities (f1 / f2 / f3) × 5 folds × 10 repeats → up to ≈ 8,000–10,000 shuffle-and-encode iterations per log (fewer where the zero rules of §3.3.4 skip an activity). At the ≈ 6.7–7.2 s per iteration measured for itemset mode that is roughly 15–21 h per log, i.e. **hours, not minutes** [ASSUMPTION: `trace_permutation_importance` itself was not timed; linear extrapolation from the itemset-mode rate]. Reproducing Fig. 3 over every activity is therefore **optional**, not part of the mandatory grid: Part 5 §5.5 lists it as the costliest compute risk, and Part 6 Table B row B16 proposes to run the length-1 comparison only on the length-1 sets each strategy selects and to keep the full Fig. 3 reproduction as an overnight run on f1 alone (a follow-up question for the supervisor, still open).
- The variant doubles everything (two strategies × three logs): make `n_repeats`, `K_fold`, `top_k`, `min_support` configurable and ship a demo config (env_report P9); caching traces as Python lists instead of filtering a DataFrame per case would remove most of the per-iteration time [ASSUMPTION: not measured].

### 3.8 Reading map of the code

1. `CrossValidation_ProcessPermutation.py` top to bottom (143 lines): CLI (`:9-30`), constants (`:33-46`), Apriori call (`:48-63`), encoding (`:65-69`), folds (`:71`), per-fold fit + call (`:76-104`), output (`:108-143`).
2. `tools.py: DataManager.__init__` + `_load_df` (`:24-78`) — normalisation, filter, `Allowed_locations`.
3. `frequent_activity_sets` (`:160-177`) — the only place the variant must replace.
4. `index_encoding` (`:338-364`) — understand the column naming before anything else.
5. `cross_split_test_train` (`:214-244`).
6. `find_itemset_indexes` (`:478-496`) → `shuffle_sequence` (`:450-476`) → `itemset_permutation_importance` (`:498-552`); trace the toy of §3.3.5 by hand.
7. `trace_permutation_importance` (`:366-448`) only if single-activity mode is reproduced.
8. `Classical_Permutation.py` (`:28-62` encoding, `:91-115` PFI/SHAP) and `Plotting_results.py` for Fig. 3/5.
9. `dist_location_calculator.py` for Fig. 4/6; `PDP_ProcessLocation.py`, `Pixel_Flipping_Process.py`, `Process_Permutation.py` last, and only for context.

Fact-check scripts used for this part: `_fc_C.py`, `_fc_C2.py`, `_fc_C3.py` (run in the Python 3.12 venv named at the top of this part); the figure crops used for every "(render)" statement are in the `_figs/` folder.

---

## Part 4 — The 2023 paper (IMPresseD) and how to build the variant

Sources: 2023 paper (BPM 2023, LNCS 14159, pp. 303–319; page numbers below are the printed LNCS pages), repos `InteractivePatternDetection` (`IMIPD.py`, `Auto_IMPID.py`, `tools.py`, GUI = `GUI_IMPresseD_tool.py`) and `PermutationLocationImportance` (CVPP = `CrossValidation_ProcessPermutation.py`, "2024 tools.py"); all line numbers were re-read in the cloned code during fact-checking, and every number about the BPIC11 data, `paretoset`, `mlxtend` and scipy was re-run in the the Python 3.12 venv environment (env_report) unless marked otherwise.

### 4.1 IMPresseD in plain words

**The problem.** ELI5: a hospital log hides thousands of small "recipes" (do X, then Y); you want the few that matter for the outcome without reading them all. Precisely: process pattern discovery methods (PPDMs) usually optimise one interest (mostly frequency), need cut-off thresholds or squash several metrics into one score, and suffer "pattern explosion" (§1, pp. 303–304). IMPresseD (Interactive Multi-interest Process Pattern Discovery) treats selection as a multi-objective problem, keeps only the Pareto front of several interest functions and lets a human choose which patterns to grow (§1, p. 304; Fig. 1, p. 308). Its concrete goal is *outcome-oriented* patterns (p. 305). A pattern is "a set of process activities (possibly annotated with additional data) with their ordering relations" (p. 303) — the group description asks you to keep only the *set* part.

**Event, trace, log (Def. 1–2, p. 306).** ELI5: an event is one line in a diary ("did X at time t for patient c"); a trace is one patient's diary; a log is the shelf of diaries. Precisely: event = `(activity, case, timestamp, attributes…)`; trace = non-empty event sequence whose timestamps do not decrease; log = set of traces. Example: case 7 = `⟨a, b, c, d⟩`.

**Partially ordered trace and the oracle (Def. 3, pp. 306–307).** ELI5: a flat log lists events one after another even when two happened at the same time; an "oracle" is the rule that says which were really simultaneous. Precisely: a conversion oracle φ turns a trace into a DAG φ(σ) = (E_σ, ≺_σ) with an upper-triangular adjacency matrix A and a reachability matrix R (paths of length 2 … |σ|−1): e′ *directly follows* e if A(e,e′) ≠ 0, *eventually follows* if R(e,e′) ≠ 0, and the two are *concurrent* if R is 0 both ways (the paper's wording; read literally, R ignores length-1 paths, so the text is sloppy, but the intended meaning "no path either way" is clear). In the NCR case the oracle was domain knowledge: systemic treatments starting within three days of each other, and treatments that start and end on the same day, are parallel (p. 312). In code the only oracle is a time window: `Trace_graph_generator` (IMIPD.py:331-400) makes one node per event position (:337-338); consecutive events whose start times differ by at most `delta_time` seconds (`abs(...).total_seconds() <= delta_time`, :347) are flagged `parallel=True` and wired as a block, otherwise the chain edge `i → i+1` is added (:379). A **negative `delta_time` therefore yields a pure chain** (`abs(...) <= negative` is never true). Node attributes: `value`, `parallel`, `color`; edge attribute `eventually` (bool). It needs `color_act_dict[activity]` for every activity (:338) and a datetime timestamp column (the GUI converts it with `pd.to_datetime`, GUI:236).

```
trace  a b c d        chain (delta_time<0):  0:a -> 1:b -> 2:c -> 3:d
```

**Process pattern (Def. 4, p. 307).** ELI5: a pattern is a mini-map of a few activities with arrows saying who comes before whom. Precisely: P = (N, ↦, α, β) is a DAG with nodes N, edges ↦, labelling α (node → activity) and a *foundational pattern* β (the parent it was extended from; NULL when |N| = 1). The paper's text does not type the edges; in Fig. 2 (p. 310) "eventually" edges are blue dashed arrows (P⁴: b ⇢ e), and in code every edge carries `eventually=True/False`. Patterns are `networkx.DiGraph`s; IDs are `<core>_<k>` and `<core>_<k>_<j>` (tools.py:93-97), and the core activity is recovered with `split("_")[0]` (IMIPD.py:620), which is why the GUI replaces "_" by "-" in activity names (GUI:235).

**"Pattern length" in the paper.** The only definition is Step 3: "Extracting patterns of length-1, i.e., individual activities." (p. 307). Length is never defined for larger patterns; Fig. 5 (p. 315) plots "Extension Steps" 0/1/2 on its x-axis (re-checked in the PDF), a different notion: one extension can add several nodes (P⁵ in Fig. 2 has 4 nodes: a, b, c, d — re-checked), and `A → A` has 2 nodes but one distinct activity. The group description asks for "pattern lengths of 1, 2 and 3" over "only the activity sets". [ASSUMPTION] we read "length k" as *k distinct activities in the projected set*; confirm with the supervisor (§4.4).

**Pattern instance (Def. 5, p. 307).** ELI5: one concrete place in one trace where the mini-map fits exactly. Precisely: a subset E′ of the trace's events with a bijection I: E′ → N such that adjacency, reachability and labels all match; PIS(P, L, φ) is the union over traces. Example: `a → b` has one instance in `⟨a, b, c⟩` and none in `⟨a, c, b⟩` (there `b` only *eventually* follows `a`).

**Extension rules (§4.2, p. 309; Fig. 2, p. 310).** ELI5: to grow a pattern, look at every place it occurs and glue on a neighbour. Precisely: Ext_f(P, E′) = (N ∪ V_f, ↦ ∪ ↦_f, α ∪ α′, P) adds nodes from an *actual instance* E′, so every generated pattern has at least one instance in the log (an inference from the definition, not a sentence of the paper). Six rules f: (1) direct following ↦, (2) direct preceding ↦′, (3) concurrent ‖, (4) eventually following ⇝, (5) eventually preceding ⇝′, (6) direct context dc = Ext↦ ∪ Ext↦′ ∪ Ext‖. The paper quantifies "∀n ∈ E′" (re-checked on the PDF page 309), which does not match its own Fig. 2 (P¹_ω = a→b→d adds a but not c, although c→d); the code uses *boundary* semantics: predecessors of the pattern's lowest-index node, successors of its highest (`create_embedded_pattern_in_trace`, tools.py:345-359, used at IMIPD.py:646 and :670). Code mapping:

| Rule | First extension of a single activity: `Pattern_extension` (IMIPD.py:168-328) | Later steps: `Single_Pattern_Extender` (:612-749) |
|---|---|---|
| direct preceding | core + *all* direct predecessors (:183-199) | all predecessors of the block (:645-667) |
| direct following | core + all direct successors (:201-217) | all successors (:669-691) |
| concurrent | nodes flagged parallel with identical in/out-neighbours, only if the core itself is parallel (:219-242, condition :222) | not implemented |
| eventually following | 2-node `core ⇝ x` for x in `(max(succ), max(succ)+Max_gap_between_events]` (:244-266, window :249-251) | one node per instance (:698-718) |
| eventually preceding | mirror window (:268-290, :273-275) | (:720-739) |
| direct context | compose(preceding, following, parallel); added only when ≥ 2 parts are non-empty (:292-326, condition :304-307) | not implemented |

`Max_gap_between_events` is a code-only parameter in event *positions* (it is not in the paper). With chains, extending core `b` in `⟨a, b, c, d, e⟩` with gap 2 gives `a→b`, `b→c`, `b⇝d`, `b⇝e` and the context `a→b→c` (worked through the window formulas: out = {2}, so eventually-following nodes are 3 and 4; in = {0}, so no eventually-preceding node). Eventually patterns carry `emb_trace=[]` (:261, :285), so they are dead ends: automatic mode skips them (Auto_IMPID.py:108-110), the GUI refuses them (GUI:750-751). Deduplication: each candidate is `nx.is_isomorphic`-compared with every pattern in the stage dictionary (tools.py:85-91), matching node `value` and edge `eventually` (the edge matcher is built with `iso.categorical_node_match("eventually", …)`, IMIPD.py:175/:643 — a node-match helper used as an edge match; it works because both only compare the attribute dicts); a hit appends the instance to the existing ID (:88-89), a miss creates `<core>_<next number>` (:93-101).

**Interest functions (§4.3, pp. 310–311).** The three dimensions were designed "by analyzing related literature and through discussions with domain experts" (p. 310): correlation alone yields too-rare patterns; frequent-but-uncorrelated patterns can have interesting extensions; confounders (e.g. age) can fake an effect (p. 310).

| Paper | Formula (plain words) | Direction | Code (IMIPD.py) | Paper ≠ code |
|---|---|---|---|---|
| Frequency interest CC(P, L, φ) | share of cases with ≥ 1 instance: \|{σ : \|PI\| > 0}\| / \|L\| | max | `Frequency_Interest` = non-zero count / n (:92-105) | none |
| Outcome interest OI = ρ(OV, FV) | ρ between the outcome vector and the per-case instance-count vector; Spearman for continuous, information gain for categorical outcomes | max | binary: `mutual_info_classif(x, y, discrete_features=True)` over all pattern columns at once (:63-69; sklearn reports nats); numerical: Spearman, then `.abs()` (:71-87) | code takes \|ρ\|; Fig. 3 (p. 313) shows signed values (e.g. −0.013 for capecitabine_2, re-checked in the PDF) |
| Case distance CD(P, L, AT) | Σ_{i ∈ C_P} Σ_{j ∈ C̄_P} (1/\|L\|) · dist(AT_i, AT_j), AT = user-chosen attributes of the first event; dist = (F_normal(dist_Euc) + dist_Jac)/(m+1), m categorical attributes [8] | min (ideal 0) | `Case_Distance_Interest` = **mean** over in×out pairs of a cached `pdist` (:37-60, mean at :58); `calculate_pairwise_case_distance`: label-encode categoricals → Jaccard; Euclidean on numerics, MinMax-scaled; (m·Jac + Euc)/(1+m) (:135-165, formula :158) | paper sums with 1/\|L\| (scales with \|C_P\|·\|C̄_P\|), code averages (the Fig. 3 values ≈0.57–0.68 are consistent with the mean, check_paper-2023 B17); the GUI also feeds the activity-count columns as categorical: it drops only id and outcome and passes just `numerical_attributes` as numeric (GUI:394-395), after filling per-case activity counts (GUI:419-431 / :990-1000) |

Tiny example: 4 cases, counts FV = (2, 1, 0, 0), labels OV = (1, 1, 0, 0) → CC = 0.5; the count separates the labels perfectly, so information gain is maximal: 1 bit, which sklearn reports as 0.693 nats (re-run with `mutual_info_classif`).

Verified detail for CD, and it is **scipy-version dependent** (both versions run during fact-checking): on the venv's scipy 1.18.1, `pdist(..., 'jaccard')` converts inputs to booleans, so label codes 2 and 3 count as *equal* and only "code 0 vs non-zero" is distinguished (`pdist([[0,1,2],[0,1,3]])` = 0.0, `pdist([[0,1,2],[1,1,3]])` = 0.333); on the 2023 repo's pinned scipy 1.11.4 (installed with numpy 1.26.4 in a throw-away directory) the same calls give 0.5 and 0.667, i.e. positions where both codes are 0 are ignored and all other positions compare the codes. So under the fresh environment the categorical case attributes are nearly ignored (only "alphabetically-first category or not"), while under the authors' pins they act as a proper mismatch count except for the first category. On activity counts both versions give "present vs absent" (1.18.1) or "count differs, ignoring shared zeros" (1.11.4). A reimplementation should compute the Jaccard/mismatch distance explicitly instead of relying on `pdist`.

**Pareto front (§4.1, pp. 308–309).** ELI5: keep a pattern unless some other pattern is at least as good on every score and strictly better on at least one (that other pattern 'dominates' it). Precisely: P^l dominates P^j iff P^l is no worse than P^j on all interest functions and strictly better on at least one; the non-dominated patterns (those that no other pattern dominates) form the front (skyline algorithm [5]; the same definition with a worked three-pattern example is in §2.14). Code: `paretoset(df[pareto_features], sense=pareto_sense)` (Auto_IMPID.py:33-35, :82-84, :131-133); installed `paretoset` 1.2.5 has `paretoset(costs, sense=None, distinct=True, use_numba=True)` (signature re-read), senses are case-insensitive ("Max"/"Min" from GUI:271-301 work — re-run), **`distinct=True` by default in all calls, keeping only the first of identical objective rows** (re-run: rows (1,1),(1,1),(2,0),(0,2) under max/max → `[True, False, True, True]`; with `distinct=False` all four are kept), and NaN handling is left to the user: in three small tests the result was inconsistent (in one, the NaN row was kept and two non-dominated rows were dropped; in two others the NaN row was dropped). CD is NaN when a pattern is in all or none of the training cases (mean of an empty list).

**Interactive vs automatic mode.** Interactive (GUI, user study): interests on the whole log (GUI:449, :622), the user picks foundational patterns, one front per extended core (`run_extension`, GUI:568-637), dashboards (Fig. 4, p. 314). Automatic: `Auto_IMPID.AutoStepWise_PPD` (:9-151), reachable only through the Tk window (GUI:972-1013; the module builds the root and runs `mainloop()` at import, :1022-1027, so never import it; the README names `GUI_IMPresseD_tool.py` as the only entry point). Parameters:

| Parameter | Meaning | Default |
|---|---|---|
| `Max_extension_step` | loop `for ext in range(1, Max_extension_step)` (:103): steps 0 and 1 always run; `=2` adds step 2 | none (GUI entry, :980-981) |
| `Max_gap_between_events` | eventually window in positions | none (GUI:983-984; also required on the main window :228-229) |
| `test_data_percentage` | `train_test_split(test_size=…, random_state=42, stratify=outcome if binary)` (:16-20) | none (:986-987) |
| `data`, `patient_data` | event table (`_` replaced, timestamp parsed, case id str; GUI:235-238); case table: attributes, id, outcome, one count column per activity, sorted by id, re-indexed 0..n−1 (GUI:370-387, counts :990-1000) | – |
| `pairwise_distances_array`, `pair_cases`, `start_search_points` | condensed `pdist`, list of (i,j) pairs with i<j, row offsets `k*n − k(k+1)/2` (GUI:404-411; re-derived from the loop) | cached in `<csv dir>/dist/pairwise_case_distances.pkl` (GUI:391-402) |
| `case_id`, `activity`, `outcome`, `timestamp`, `outcome_type`, `pareto_features`, `pareto_sense` | column names; `'binary'` (information gain) or `'numerical'` (\|Spearman\|); subset of `['Outcome_Interest','Frequency_Interest','Case_Distance_Interest']` with `'Max'/'Min'` (GUI:437-447; defaults Max, Max, Min :273/287/301) | – |
| `d_time`, `color_act_dict`, `save_path` | `delta_time` seconds (GUI:1002-1003); activity → colour plus `'start'`/`'end'` (GUI:239-249); JSON folder | none |

Flow: step 0 scores every activity of the event table on the train cases and takes the front (:28-36); step 1 builds trace graphs **from the whole log, test cases included** (:49-50, cached :55-61), extends every front activity into one shared dictionary (:45, :63) and takes one joint front (:78-85); steps ≥ 2 extend each front pattern separately (:106-118), one front per step (:126-134). Outputs: `train_X`/`test_X` with a count column per front pattern of every step plus `Case_ID`/`Outcome` (`training_encoded_log.csv`/`testing_encoded_log.csv`, GUI:1016-1017), and one `<patternID>.json` (networkx node-link, nodes carry `value`) per front pattern from step 1 on (:87-93, :136-142). The graph dictionary is not returned (`All_extended_patterns_dict` is local, :102).

### 4.2 The 2023 evaluation in brief

*User-based* (§5.1, pp. 312–314): NCR log of metastatic stomach/oesophageal cancer (957 cases, 32 treatment codes, 368 variants), outcome = survival time; max CC, max OI (Spearman), min CD. Step 0: 8 non-dominated treatments; experts extended capecitabine and paclitaxel: 18/206 and 14/116 extensions on the front ("a maximum of 10%" in the paper; individually 8.7% and 12.1%, pooled 32/322 = 9.9%); a ≥10-patient filter left 8 and 6. Most patterns matched known effective combinations; a few were rejected (low correlation, missing expected treatment, no radiotherapy). Threats: one case, two experts.

*Quantitative* (§5.2, pp. 314–316), "deviance-mining" style [21]: front patterns are frequency-encoded features, a decision tree predicts the outcome, 5-fold CV, "average F1-score with minimum and maximum" (p. 316; the paper does not say "weighted"). Compared: K Pareto patterns per step vs top-K by each single interest vs all patterns; all K front patterns are extended at the next step; max information gain, max frequency, min CD; discovery on the training set only; all case attributes for CD. Datasets: the preprocessed, labelled logs of Teinemaa et al. [24] — BPIC2011, BPIC2012, Production — plus NCR with survival split into three equal-frequency classes (p. 315). The number of labelings is visible only in Fig. 5 (p. 315, re-checked): panels BPIC11_1–4, BPIC12_1–3, NCR, Production = 9 panels, although the text says "10 studied event logs" (p. 316). Results: the Pareto set is "comparable or better" than single interests (abstract, §5.3) and "consistently rank[s] among the best"; only "all patterns" sometimes wins, the Pareto feature set being 47.5% of its size on average; CD alone is worst "in most" logs; outcome alone is best in 5 logs (BPIC11_2, BPIC11_4, BPIC12_1–3), frequency in 3 (BPIC11_3, NCR, Production) (p. 316). §5.3: extensions rarely help (slight gain only on NCR after step 1), attributed to parent/child overlap. The released code cannot reproduce this: `Classifiers_kFold_results` (tools.py:690-720, decision tree, weighted F1 at :709) is defined but never called anywhere in the repo (grep re-run), automatic mode does one split, and no single-interest/all-patterns baselines exist.

### 4.3 The variant, concretely

**What changes.** Only the *producer* of candidate activity sets. The 2024 pipeline keeps its encoding, XGBoost, folds, location permutation and existence importance; instead of Apriori's "top 10 frequent itemsets of size greater than one" (2024 paper p. 197; `frequent_activity_sets`, 2024 tools.py:160-177, size filter :171-172, `head(top_k)` :174) it receives sets projected from IMPresseD patterns.

**Integration point.** CVPP:37 hard-codes `candidate_selection_method = 'apriori'`; the `'optimized'` branch (:54-56) calls the stub `DataManager.optimized_activity_sets` (2024 tools.py:97-158, body commented out, returns None) and the script exits (:58-60). Contract: CVPP:61-63 builds `candidate_itemsets: dict[int, list[str]]` (`{0: ['370407','ac370000'], …}`), passed as `frequent_itemsets` to `itemset_permutation_importance` (:95-98; 2024 tools.py:498), which accepts a `str` (eval'd), `list` (→ set) or set per entry (:516-520) and shuffles only cases whose trace contains the whole set and is longer than it (:521-523). Names must be 2024-normalised: lower-case, no spaces, "-" or "_" (:39-50); the BPIC11 activity codes contain none of those characters, in the raw files or after loading (re-checked for f1/f2/f3), so only `.lower()` acts. Result columns are renamed `str(candidate_itemsets[k])` (CVPP:113-117) and written to CSV (:121); that string is `eval`'d back in Classical_Permutation.py:56, Pixel_Flipping_Process.py:64, dist_location_calculator.py:57 and 2024 tools.py:518 (grep re-run), whereas Plotting_results.py does **not** eval (column strings are reused as labels, :33-35; backslash file-name split :27-28). Keep `str(sorted(list))` naming so both strategies line up.

**Recipe for activity sets of length 1, 2, 3.**
1. Load each log through the 2024 `DataManager` (same normalisation and rare-activity filter, 2024 tools.py:36-78); re-join case attributes from the raw CSV (the loader keeps only case, activity, time, label, `event_nr`, :76).
2. Mine patterns (chain-based reimplementation or wrapped `AutoStepWise_PPD`) with `Max_extension_step=2`.
3. Project each front pattern: `acts = frozenset(d['value'] for _, d in G.nodes(data=True))`; **length = |acts|**; edges dropped. `A→A`/`A⇝A` collapse to `{A}`; a context `A→B→A` to `{A,B}`.
4. Step → length (chains): step 0 → 1; step 1 → 2-node patterns (pred/succ/eventually) → 2, context patterns (pred→core→succ) → 3 unless a repeat collapses them; step 2 → 3-node children of 2-node parents → 3 (needed for triples whose core is not in the middle, e.g. `core→B→C`). Step-2 children of 3-node context parents have 4 nodes and must be dropped unless a repeated activity collapses them to ≤ 3 distinct activities.
5. Dedupe: `A→B`, `B→A`, `A⇝B` all map to `{A,B}`; keep one row per set with the producing pattern(s) and their interest values.
6. Pick k per length (default 10, matching Apriori) by non-dominated sorting with a documented tie-break; report the full-front size; feed each length as its own `candidate_itemsets`. ELI5 of non-dominated sorting: peel the first front (the sets that no other set dominates, §2.14) off the pile, then compute the next front from the sets that are left, and repeat until k sets are collected. If the last front taken overshoots k, break the tie inside that front by a documented rule (default: higher IG first, Part 6 B7); the front of the leftover rows can be computed with the same `paretoset` call. Singletons also go through `itemset_permutation_importance` (works for k = 1: `find_itemset_indexes` then returns every occurrence, :478-496), not `trace_permutation_importance`, so all lengths share one permutation logic. Add the `< 3 shuffled cases → importance 0` guard from :408-413, which the itemset path lacks (an empty `shuffled_cases` would send an empty frame into `index_encoding`, :529).

**Recommended BPIC11 configuration.** `delta_time = -1` (chains): timestamps are day-level (every event is stamped "23:00", e.g. `1/2/2005 23:00`) and 88.8% / 87.1% / 88.5% of consecutive event pairs in f1/f2/f3 share a timestamp (re-run: 0.8877 / 0.8711 / 0.8846; no negative differences along `event_nr`), so any `delta_time ≥ 0` merges most of a trace into one concurrent block and the isomorphism tests explode; the 2024 "location" is `event_nr` anyway. Case-distance attributes: `Age` (numeric) and `Diagnosis`, `Treatment code`, `Diagnosis code`, `Specialism code` (categorical), all constant per case (re-checked; whole-log distinct values on f1: 74 / 105 / 43 / 11 / 3). Exclude `Variant`/`Variant index` (815/977/793 unique values on f1/f2/f3, re-checked: they identify the control-flow variant, i.e. leak the trace), `Diagnosis Treatment Combination ID` (id-like: 799/832/729 distinct values) and the artefacts `timesincemidnight`/`hour`, which take a single value in the whole file (re-checked). Activity-count columns: code includes them, paper does not; default: paper. `Max_gap_between_events = 3` [ASSUMPTION: no value is reported in the paper or the code], outcome = `label` (binary → information gain), IG max / CC max / CD min, `distinct=False`.

**Symmetric Apriori baseline per length.** The 2024 code drops singletons and has no size cap. Use `mlxtend.frequent_patterns.apriori(df, min_support, use_colnames=True, max_len=3)` (signature re-read on mlxtend 0.25.0: `apriori(df, min_support=0.5, use_colnames=False, max_len=None, verbose=0, low_memory=False, n_jobs=1)`), top-k by support *per size*. Re-run at `min_support=0.5` after the 2024 filter: f1 16/94/258 itemsets of size 1/2/3, f2 23/148/544, **f3 only 5/4/1** (the repo function returns 5 sets for f3, re-run; at 0.4 f3 has 20/130/572, f1 23/211/1330, f2 26/286/2036), so `min_support` must be per dataset (≈0.4 for f3) and reported.

### 4.4 Design decisions

| Decision | Options | Recommended default | Ask supervisor? |
|---|---|---|---|
| Meaning of "length" | nodes / distinct activities / extension step | distinct activities in the projected set | yes |
| Oracle | `delta_time<0` chain / `=0` same-day blocks | chain | yes (brief mention) |
| Extension rules | all six / drop eventually | all six, gap 3 | optional |
| Interest functions | IG+CC+CD / IG+CC | all three; sensitivity run with two | optional |
| Front vs fixed k | whole front / k=10 by non-dominated sorting (front by front, §4.3 step 6; dominance in §2.14) / top-k by one interest | k=10 filled front by front, tie-break by IG, + report front size | yes (Part 6 B7) |
| Score patterns or sets | select patterns then project / rescore sets | select then project, keep best pattern per set | yes |
| Mining scope | whole log (as 2024 Apriori) / train only / per fold | whole log for both strategies, state the leakage | yes |
| CD definition | code mean / paper sum; attributes only / + activity counts; `pdist` or explicit Jaccard | code mean, attributes only, explicit Jaccard | optional |
| Apriori baseline | top-10 size>1 (original) / top-k per size, `max_len=3` | per size; `min_support` 0.5 (f1, f2), 0.4 (f3) | yes |
| Implementation strategy | wrap original code / reimplement chain-only | reimplement a chain-only version of the automatic mode (recommended; "focus on the essence", description); wrap `Auto_IMPID.AutoStepWise_PPD` only as a cross-check on a small log, pending supervisor answer (Part 6 B10) | yes (Part 6 B10) |

### 4.5 Risks and performance pitfalls in the 2023 code (verified by reading)

- **Isomorphism dedupe**: every candidate is `nx.is_isomorphic`-checked against every stored pattern (tools.py:85-87) → O(instances × patterns) VF2 calls; prior probes measured on the order of minutes per core activity on f1 (about 160 s for ac370000 with 435 patterns in check_paper-2023 B16, not re-run here [UNVERIFIED in this run]; open experiment Part 6 Table C row C4). With chains a tuple key `(labels…, edge flags…)` is equivalent and O(1) (the isomorphism test ignores the `parallel` flag and node positions).
- **O(n²) pair scan**: `pair_cases.index(item, start_search_points[a])` (IMIPD.py:55) scans a Python list per in×out pair → O(|in|·|out|·n) per pattern; the direct condensed index is `start[a] + (b − a − 1)` (re-derived from the GUI:409-411 offsets), or use `squareform`.
- **Distance cache keyed per folder** (GUI:391-402): all BPIC11 CSVs sit in one folder, so f2/f3 would silently reuse f1's matrix.
- **Test cases in step 1**: graphs and extensions come from the whole log (Auto_IMPID.py:49-50), only scoring uses train rows (:75-81) → patterns with 0 train cases and NaN CD; contradicts "training set only" (p. 315) and the README's leakage note.
- **Cross-core merge and duplicate IDs**: the shared step-1 dictionary (:45) appends duplicate instances to the first core's ID (tools.py:88-89), doubling derived step-2 counts; step-≥2 dictionaries are per parent (IMIPD.py:617), so identical graphs from different parents get different IDs. Step-2 counting hard-codes `'case:concept:name'` (IMIPD.py:746) inside the instance loop (:741-746).
- **Broken requirements**: `sklearn==0.0` (requirements.txt line 10) is a deprecated meta-package; `pandas==2.0.3` has no cp312 wheel (env_report P8); `xgboost`, `PyQt5`, `seaborn`, `future`, `Pillow` are pinned but never imported (grep re-run; `pyperclip` *is* imported by the GUI, :15/:732). The `env_report` pins import `IMIPD`/`Auto_IMPID`, but an end-to-end BPIC11 run of `AutoStepWise_PPD` under them is [UNVERIFIED].
- pm4py (AGPL banner) serves only `VariantSelection` (IMIPD.py:17-34; the GUI calls it at :421 and :990), replaceable by `pd.crosstab`.

### 4.6 Proposed architecture (proposal, not from the sources)

Package `ppm_location/` with one `main.py` (argparse → YAML), PEP8, no absolute paths:

| Module / class | Responsibility |
|---|---|
| `EventLogLoader` (`data/`) | CSV → normalised event table (2024 rules), case table with CD attributes, `Allowed_locations` |
| `IndexEncoder` (`encoding/`) | pandas-2-safe index one-hot encoding (int dtype, single concat) |
| `ActivitySetSelector` (ABC) → `AprioriSelector`, `ImpressedSelector` (`selection/`) | both return `dict[int, list[list[str]]]` keyed by length; `ImpressedSelector` owns `TraceGraphBuilder`, `PatternExtender`, `InterestScorer`, `ParetoSelector`, `SetProjector` |
| `LocationPermutationImportance`, `ExistenceImportance` (`importance/`) | seeded folds, XGBoost, per-fold `sub_data` reset, `< 3` guard; binary-encoding baseline |
| `ExperimentRunner`, `Plotter` (`experiment/`, `report/`) | strategies × lengths × datasets loop, result files; paired box plots, comparison tables |

YAML keys: `datasets: [f1, f2, f3]`, `strategies: [apriori, impressed]`, `lengths: [1,2,3]`, `top_k`, `apriori.min_support: {f1: 0.5, f2: 0.5, f3: 0.4}`, `impressed.{delta_time: -1, max_gap: 3, max_extension_step: 2, interests, senses, distinct, cd_attributes, include_activity_counts}`, `cv.{k_fold: 5, seed: 42}`, `permutation.{n_repeats: 10, seed: 2023}`, `output_dir`. CLI: `python main.py --config configs/full.yaml`, plus `configs/demo.yaml` (1 fold, 2 repeats). Outputs: `results/<dataset>/<strategy>/L<k>/{sets.csv, location_importance.csv, existence_importance.csv, fold_f1.csv}` and `results/comparison/{overlap.csv, rank_changes.csv, figures/}`.

**First coding steps (proposal, in this order).** ELI5: build the plumbing that the original 2024 pipeline already has, make it reproducible, then plug in the two set producers, and only then write the experiment loop.

1. `EventLogLoader` + `IndexEncoder` reproducing the 2024 `DataManager` behaviour: activity-name normalisation (`_load_df`, 2024 tools.py:39-50), the sort on case id and `event_nr` (:45/:55), the rare-activity filter `frq_threshold` (default 2, :25/:62) and the `Allowed_locations` dictionary (:70-73) — plus the int cast of the one-hot feature columns that fixes the pandas-2 dtype crash of `itemset_permutation_importance` (env_report P1). Acceptance check: the same counts as the smoke run on f1 (1,130 cases, 164 activities, encoded frame 1,130 × 5,941; env_report §3.1/§3.3).
2. Seeded `StratifiedKFold` (the original is unseeded, 2024 tools.py:231; env_report P3).
3. `AprioriSelector` with `max_len=3` and top-k per size (§4.3, "Symmetric Apriori baseline").
4. `LocationPermutationImportance` with the `faithful`/`fixed` switch (reproduce vs repair the behaviours listed in Part 3 §3.6; Part 6 B2).
5. `ImpressedSelector` (chain trace graphs, extension rules, interest functions, Pareto front, projection to sets; §4.3 recipe).
6. Experiment runner + comparison metrics (§4.7; set overlap per length, rank correlation of importances on shared sets).

If any upstream code is vendored instead of re-written, note that **both repos ship a module named `tools.py`** with different contents (`from tools import DataManager` in the 2024 scripts vs `from tools import create_embedded_pattern_in_trace, …` in 2023 IMIPD.py:14; env_report P7), so they must be namespaced (e.g. `vendor/ppm2024_tools.py`, `vendor/impressed_tools.py`) and the 2023 import rewritten accordingly; otherwise whichever directory comes first on `sys.path` wins silently.

### 4.7 Experiment loop (pseudocode)

```
cfg = load_yaml(args.config); set_seeds(cfg)
for ds in cfg.datasets:                                   # f1, f2, f3
    log = EventLogLoader(ds).load()                       # normalised, filtered
    X, y, cases = IndexEncoder().fit_transform(log)
    folds = StratifiedKFold(cfg.cv.k_fold, shuffle=True, random_state=cfg.cv.seed)
    sets = {s: Selector(s, cfg).select(log, lengths=[1,2,3]) for s in cfg.strategies}
    for strategy in cfg.strategies:
        for k in [1, 2, 3]:
            cand = {i: lst for i, lst in enumerate(sets[strategy][k])}   # top-k sets of length k
            for f, (tr, te) in enumerate(folds.split(cases, y)):
                model = XGBClassifier(random_state=cfg.seed).fit(X[tr], y[tr])
                log_f1(ds, strategy, k, f, f1_weighted(model, X[te], y[te]))
                LI[f] = LocationPermutationImportance(log, model, cand, cfg.permutation).run(tr)
                EI[f] = ExistenceImportance(log, model, cand, cfg.permutation).run(tr)
            save(ds, strategy, k, cand, concat(LI), concat(EI))
    compare(ds): set overlap per k, rank correlation of mean importances on shared sets
```

Cost: one (itemset, repeat) took ≈6.7 s on f1 with 904 training cases (env_report, measured from 2 itemsets × 1 repeat = 13.4 s); 2 strategies × 3 lengths × 10 sets × 5 folds × 10 repeats = 3000 iterations × 6.7 s ≈ 5.6 h per dataset [ASSUMPTION: linear scaling; f2 (trunc40, 8,319 features) is larger and will be slower], hence the YAML knobs and the demo config.

---

## Part 5 — Deliverables, detailed work plan, team split, environment & downloads

**ELI5 for the whole part.** The assignment is a school science fair: you must hand in a written report (short paper), a wall display (poster) with a 3-minute talk, the working experiment itself (code), a note in your own names saying which AI tools helped you and that you checked their output (technology statement), and you must grade other teams' work (peer reviews). The jury then asks you to explain your experiment face to face (12 Nov). Everything below says exactly what each box must contain, when it is due, who does it, and how to set up the machine.

**Naming note.** The course code is written three ways in the material: "JM0211" on the slides and in the dates file, "JM0210" in the header of the Group 11 description, and "1JM0211" in the paper and poster templates (description_group11 txt header; week1 overview slide 1; main.tex `\title`; Poster_GroupX.pdf header). Use the template spelling inside the deliverables.

### 5.1 Deliverables, one by one, with what the rubric rewards

The assignment is 30 % of the course grade; the written exam is 70 % (120 min, individual, retake possible; min. 5.0 on the exam to pass; no retake for the assignment) (week1 overview slide 7; Rubric.xlsx sheet Rubric_Explanation cell A1). Weights inside the assignment: short paper 35 %, implementation 30 %, poster & pitch 25 %, peer reviews 10 %, technology statement mandatory (Rubric_Explanation rows 3–7).

#### 5.1.1 Short paper (35 %)

- **ELI5:** a 5-page lab report that a classmate could read and redo.
- **Template facts** (ShortPaper_GroupX.zip = `main.tex` + `literature.bib`; ShortPaper_GroupX.pdf is its compiled form): `article` class, A4, margins 2.5 cm left/right, 1.5 cm top/bottom, `footskip` 0.5 cm; `\pagestyle{empty}` so no page numbers; packages `biblatex` (no backend option given in the template; biblatex's default backend is biber), `mdframed`, `tikz`, `csquotes`, `hyperref`; title "1JM0211 Process Mining \\ Title of Your Topic", author line "Group Number and Group Member Names", date "Submission Date"; a framed `mybox` "Remarks (to be removed upon submission)". **Page limit: 5 pages excluding references** (main.tex remarks box). The `.bib` comment says: avoid Google Scholar for bib files, use DBLP or a tool like Zotero or Mendeley; the only sample entry is van der Aalst's 2016 book from DBLP.
- **Mandatory section order** (main.tex `\section` lines): 1 Motivation and Scientific Background; 2 Conceptual Approach ("Provide a visualization of your approach"; guiding questions: required input incl. which event-log characteristics/perspectives, parameters and their purpose, most crucial steps, methods used, output); 3 Implementation (packages and other non-conceptual details that make the approach reproducible); 4 Evaluation Design and Results (data incl. any artificial data and "the data provided through Canvas"; motivated design — quantitative metrics and/or qualitative corner cases; how the gold standard/ground truth was created; parameter settings and testing; visualised results; comparison to related methods if applicable); 5 Limitations and Future Work; then `\printbibliography`.
- **Rubric mapping and tips** (Rubric.xlsx, sheet Rubric_ShortPaper; grader text paraphrased from the sheet's "Very Good" column):

| Item | Pts | What the grader looks for (gist of the rubric text) | Concrete tip for Group 11 |
|---|---|---|---|
| S1 Motivation | 4 | relevance "using a concrete use case and/or a comprehensive example"; how the topic is embedded in the course and related to the other topics; commonalities and differences to other topics | The 2024 paper has **no hospital or re-admission example**: its motivation is that a process manager in a flexible process does not know how the moment at which a (group of) activity(ies) is executed affects the outcome, and that location insights can guide redesign heuristics (2024 paper p. 193, txt ll. 118–127). Use that, plus the BPIC11 setting (one case = one gynaecology patient; labels from LTL rules, labels_report). Course embedding: the Week 1 slide "Different Goals of Process Mining – Beyond ex-post" (outcome prediction from an event stream; week1 intro slide 39) and Week 8 (PPM and Explainability). Other groups to contrast with (verified from the dates file and week1 overview slide 10): Group 9 "PPM with Trace Clustering and LLMs" (remaining-time prediction), Group 10 "PPM Using Inter-case Features" (both PPM, both L. Genga) and Group 7 "Prescriptive Process Monitoring with Reinforcement Learning" (B. Verhoef); only their one-paragraph summaries are known locally [UNVERIFIED: their detailed descriptions sit on their Canvas group pages] |
| S2 Conceptual approach | 10 | input and its format, parameters and purpose, all crucial steps "including sound rationales", output explained and visualised | One pipeline figure with two selector boxes (Apriori vs IMPresseD → set projection); a parameter table: `min_support`, `top_k` (CVPP:27–28), `K_fold` (CVPP:39), `n_repeats`, `random_state` (tools.py:498–499), and for IMPresseD `Max_extension_step`, `Max_gap_between_events`, `test_data_percentage`, `d_time`, `pareto_features`, `pareto_sense` (Auto_IMPID.py:9–13), plus the group's own Pareto-to-top-k selection rule |
| S3 Implementation | 3 | implementation details complement §2 for reproducibility and "match the actual implementation"; the code itself is graded separately | Packages with versions from requirements.txt; module names; Python 3.12 |
| S4 Evaluation | 10 | data described in detail; design correct, described and motivated; results explained incl. failed cases; visualised; gold-standard creation explained and reasonable; parameters tested and motivated; comparison to related methods | Describe BPIC11 f1–f3, their LTL labels and what label 1 means in the shipped files (rule satisfied, labels_report); say explicitly that no ground truth for "true importance" exists and what replaces it (reproduction of the paper's Fig. 3/5 ranges + the comparison metrics of §5.2) |
| S5 Limitations | 4 | limitations derived from the design and/or the results; future work linked to them | Day-level timestamps make "location" largely recording order (check_paper-2024 §C item 1); training-fold scoring (tools.py:503–504, 538–539); accumulation of permutations (tools.py:507/526); supervised IMPresseD vs unsupervised Apriori |
| S6 Quality | 4 | page limit respected and "used adequately" (sub-section weights reflect importance); academic language; effective visualisations; statements supported by literature | Give §2 and §4 the most space (10 pts each); cite every claim; bib from DBLP |

#### 5.1.2 Poster and pitch (25 %)

- **ELI5:** one big picture-heavy page you can explain in 3 minutes to a classmate walking by.
- **Template facts** (Poster_GroupX.pdf; editable source Poster_GroupX.vsdx = Visio): header "1JM0211 Process Mining", "Title of your Topic", "Group X", "Group Member Names", JADS logo; boxes top-to-bottom: Motivation and Scientific Background; Explanation of Conceptual Approach (largest box); Evaluation Design and Results (second largest); Limitations and Future Work (smallest). Guiding questions are the same as in the paper template. "Instructions and Remarks (to be removed for printing)": design the poster only after at least a substantial draft of the short paper; avoid too much text, visualisations understandable by themselves; "the space allocation in the template gives an indication of which parts are most relevant"; print in at least A2 and upright format; be on time to hang it, bring tape; there is no fixed timing for pitches, the audience walks around freely; always one person available to pitch, the others visit other groups as audience; pitch time limit 3 minutes; every member must be able to deliver the pitch (rotation); engage in discussion when the audience asks; align the pitch with the poster content.
- **Rubric** (Rubric.xlsx, sheet Rubric_PosterandPitch): P1 Content 2 (correct, complete, matches the short paper; "content as such is graded mainly through the short paper"); P2 Structure 4 (follows the template, reasonably extended if needed; pitch follows the structure; parts logically connected); P3 Timing 4 (3 minutes "used adequately (not too short and not exceeded)"; time per part reflects importance); P4 Terminology & presentation style 3; P5 Visualisations 4; P6 Readability & formatting 3 (all elements readable, in particular visualisations); **P7 Q&A & discussion 5** (questions by peers and lecturers answered correctly and comprehensively; flexibility to answer during the pitch without losing focus). Tip: P7 is the single biggest poster item, so run a Q&A drill with the question lists of check_paper-2024 §D and check_paper-2023 §D.

#### 5.1.3 Implementation (30 %)

- **ELI5:** a kit a stranger can unpack, assemble with the enclosed instructions, and get the same numbers as your report.
- **Rules** (Instructions_Implementation.txt): Python 3.x; `.py` files, no Jupyter notebooks; `README.md` with installation and run instructions incl. "the exact bash commands in the correct order" (it links the Tilburg Science Hub README best-practices page); `requirements.txt` with all dependencies; versioning recommended (TU/e GitLab licence, https://gitlab.tue.nl/users/sign_in; https://ohmygit.org for git beginners); PEP 8; object-oriented; one main method; provide all data and code needed to reproduce the evaluation results; NEVER absolute paths, generic directory structure that needs no adjustment on another system; adjustable parameters via command-line parameters or a YAML configuration file; reasonable comments. Point deductions: too messy code (names, structure, comments), wrong/incomplete README, wrong/incomplete requirements, results not reproducible or not aligned with poster and short paper.
- **Rubric split** (Rubric_Explanation row 5): code structure & understandability 10; installation instructions 3; run instructions 3; **correctness & reproducibility 14**; "if you have developed a gold standard and your own artificial datasets you should submit them together with the implementation".
- **Team constraint (user_answers.md):** teammates work on Windows AND macOS, so the README must give both command forms and the code must be path-agnostic (`pathlib`, forward slashes never hard-coded; note `address.split('/')` in CVPP:44 is exactly the kind of thing to avoid).
- **Checklist (tick before 9 Nov):** [ ] one `main.py` + YAML config; [ ] classes, not scripts (§5.6); [ ] seeded folds and permutations — the original's `StratifiedKFold(n_splits=K_fold, shuffle=True)` has no `random_state` (tools.py:231; env_report P3); [ ] `pathlib` relative paths, output dir auto-created; [ ] a `demo` config (minutes) and a `full` config (reproduces paper/poster numbers); [ ] result CSVs/PNGs committed; [ ] fresh-clone test on a second machine (one Windows, one Mac) following the README literally; [ ] PEP 8 pass (`flake8`/`black` [ASSUMPTION: tool choice; the instructions only link PEP 8]); [ ] no notebooks; [ ] upstream `tools.py` modules renamed (env_report P7).

#### 5.1.4 Technology statement (mandatory)

- **ELI5:** the note in your own names: "these AI tools helped us with X, we checked everything, we take responsibility".
- **Verified:** mandatory; missing = 0 points for the whole assignment (Rubric_Explanation row 7; week1 overview slide 7). The assignment is AI Index Level 4 (AI as a support tool for editing, inspiration/brainstorming, code suggestions, calculations, visualisation, transcription; "It is not allowed to use AI to generate final products"); the exam is Level 1 (no AI); "Incorrect, fabricated sources or missing references will not be tolerated"; never enter sensitive or confidential data into AI platforms; literature search via academic databases such as Web of Science is strongly recommended (week1 overview slide 13). The slide says "you must add the technology statement, using the text below" — the required verbatim template: "During the preparation of this work, I/We used [NAME TOOL / SERVICE / VERSION OF AI TOOL] in order to [REASON]. The following parts of the assignment were affected/generated by AI tool usage: [INTRODUCTION / METHODS / xxx, DISCUSSION]. After using this tool/service, [NAME STUDENT(S)] evaluated the validity of the tool's outputs, including the sources that generative AI tools have used, and edited the content as needed. As a consequence, [NAME STUDENT (S)] take(s) full responsibility for the content of their work."
- [UNVERIFIED] Where and in which format it is submitted (separate file, inside the paper, Canvas form) — the group description only says "Technology Statement: see Canvas"; the slide wording "add the technology statement" suggests it is attached to the assignment submission. Writing the statement is postponed to the end by the student's choice (user_answers.md); the AI-usage log still starts now: [ASSUMPTION] keep a running log from today (tool, version, date, purpose, affected part; Part 6 Table A row A6) so the statement is complete and defensible on 12 Nov. [ASSUMPTION] pm4py's AGPL-v3 licence (banner on every import, env_report P10) belongs in the README licence note, not in the technology statement, which is about AI tools.

#### 5.1.5 Peer reviews (10 %)

- **ELI5:** you are also a judge for other teams, and judging well earns points.
- **Verified** (Rubric_Explanation row 6): three reviews — (1) short-paper draft of one other group, 4 pts, **group** task, due 3 Nov 23:59 (the partner draft is due 28 Oct 23:59; group description); (2) poster session 1 (5 Nov, Groups 1–6 present per their dates blocks), two groups, 3 pts, **individual**; (3) poster session 2 (12 Nov), two groups, 3 pts, **individual**. Assessed on "whether it is on-time, elaborate, constructive and complete w.r.t. the rubric"; for the short paper, improvement directions are expected. Use sheet Rubric_ShortPaper (S3 only partially checkable without the code) and Rubric_PosterandPitch (P1 only partially checkable without the paper) (Rubric_Explanation rows 3–4). The week1 overview (slide 8) adds: no attendance obligation, but peer feedback is asked at the poster sessions.
- [UNVERIFIED] Which group is Group 11's partner, how drafts are exchanged, and the submission deadline/mechanism of the poster-session reviews (same day? Canvas form?) — fetch from Canvas.

#### 5.1.6 Assignment explanation to lecturers (12 Nov 16:30–16:45)

Verified: "all presenting groups will have to explain their assignment, in particular their implementation, to the lecturers after the poster sessions" (Rubric_Explanation row 7; group description "Assignment Explanation: 12th November, 16:30-16:45h"); suspicion of AI use beyond the allowed level, or fraud, is reported to the exam committee (week1 overview slide 7). The slot is 15 minutes. Treat it as an oral check: every member can open the repo and walk through `main.py` → selector → permutation → plots within the slot [ASSUMPTION: format and depth of questioning are not described anywhere].

### 5.2 Work plan, Thu 1 Oct → Thu 12 Nov 2026

All weekday names were computed for 2026 (1 Oct 2026 is a Thursday; the course schedule's dates 3 Sep … 3 Dec all fall on Thursdays in 2026, whereas they were Wednesdays in 2025, so the material could be last year's edition with unchanged day-month dates [UNVERIFIED]; the lectures are stated as 13:45–15:30). Owner roles are defined in §5.3. "Done-when" is the acceptance test.

| Dates | Task | Owner | Output | Done when |
|---|---|---|---|---|
| Thu 1 Oct (today) | **Office-hour questions are already sent** (user_answers.md lists the six questions: where the PPM code is; how Apriori is used; whether all 10 repeats × 5 folds and box plots for lengths 1/2/3 on all three BPIC11 logs are required; whether `CrossValidation_ProcessPermutation.py` may be reused with an own itemset generator; automatic Pareto-front extraction vs interactive expert selection; human-in-the-loop bias; Part 6 §6.0). The Teams link is sent "once you have sent your questions", all members in CC (Assignment dates file, header). Prepare a short list of follow-ups for tomorrow from Part 6 Table B (P1 rows) that were NOT sent: f1–f3 only? (B4); reproduce bugs as-is or fix+report? (B2); meaning of "pattern length 1/2/3" (B1); Pareto front vs fixed k (B7); mine once vs per fold (B9); train-fold vs test-fold scoring (B3). Week 5 (PM4Py) slides and instruction sheet are already in `coursematerial/` (§1.6); fill in the optional Kickoff Work Plan (week1 overview slide 12: "you can use") | Writing lead + all | follow-up list | list ready before 15:00 Fri |
| Thu 1 Oct (today) | **Start the AI-usage log today** (Part 6 Table A row A6): one line per use — tool, version, date, purpose, affected part of the deliverable. The technology statement itself is postponed to the end by the student's choice (§5.1.4); the log is what makes it complete later | QA lead (everyone appends) | `docs/ai_usage_log.md` [ASSUMPTION: file name] | first entry exists; every member knows the file |
| Fri 2 Oct 15:00–15:30 | Office hour 1 (Teams). Get answers to the six sent questions; raise the follow-ups if time permits, otherwise send them for office hour 2 | all attend; QA lead takes minutes | decision log | answers written into the open-questions register (Part 6) |
| Fri 2 – Sat 3 Oct | **Decide and write down the pattern-length / selection rule** (Part 6 Table B rows B1, B7): what "length 1/2/3" means (default: distinct activities in the projected set), how many sets per length (default: k = 10 filled layer by layer from the Pareto front, tie-break by IG, front size reported) — in `decisions.md` now and later verbatim in the paper's §2 Conceptual Approach; revise if the office-hour answer differs | Variant lead + Writing lead | `decisions.md` entry | rule written with its rationale and its source (answer or default) |
| Sat 3 – Sun 4 Oct | Environment + repo: create git repo; clone both upstream repos fresh into `external/` (read-only reference; URLs in §5.4); create the venv per §5.4; re-run `smoke_2024c.py` and `smoke_2023.py` equivalents inside the project; **run Part 6 Table C experiments C1 (runtime scaling on f1/f2/f3), C3 (`Auto_IMPID` end-to-end on f1) and C7 (Apriori `min_support` sweep) — C14 is closed** | Pipeline lead | `.venv`, `requirements.txt`, smoke outputs, C1/C3/C7 result notes | `pip check` clean; smoke scripts print the known f1 numbers (1130 cases, 164 activities, 10 itemsets; env_report §3.1); C1/C3/C7 outcomes recorded in `decisions.md` |
| Mon 5 – Thu 8 Oct | **Reproduce the 2024 baseline on f1–f3 — mandatory: multi-activity mode** (Fig. 5; Apriori top-10, 5 folds × 10 repeats; CVPP:39, tools.py:499), with the int-cast workaround (env_report P1), `type=int` for `top_k` (P2) and a lower `min_support` for f3 (only 5 multi-activity itemsets at 0.5; §3.6 row 7, Part 6 Table B row B6). Compare box plots with the Fig. 5 ranges (§3.5). **Optional: single-activity mode over ALL activities** (Fig. 3; `--Multi_activity ""` because the flag is `type=bool`, CVPP:21–25) — f1 only, as an overnight run, because 164 activities × 50 (fold × repeat) re-encodings take hours (§3.7; Part 6 Table B row B16: default is to score only the selected length-1 sets). Thu 8 Oct: Week 6 lecture | Pipeline lead (+1) | `results/baseline/*`, one-page reproduction notes | three f-panels for multi-activity mode exist; deviations from Fig. 5 explained; the optional f1 single-activity panel, if run, compared with Fig. 3 |
| Fri 9 Oct (first coding steps) | **Pointer:** build in the order of §4.6 — `EventLogLoader` + `IndexEncoder` + seeded folds first (so the baseline F1 is reproducible), then `AprioriSelector` (the known top-10 f1 sets of env_report §3.1 are the acceptance test), then `LocationPermutationImportance` with the int-cast fix (env_report P1); only then the IMPresseD side | Pipeline lead | first modules + unit tests on f1 | seeded fold-0 F1 identical in two runs; f1 top-10 Apriori sets match env_report §3.1 |
| Fri 9 – Thu 15 Oct | **OOP skeleton** (`EventLogLoader`, `IndexEncoder`, `ActivitySetSelector` → `AprioriSelector`/`ImpressedSelector`, `LocationPermutationImportance`, `ExistenceImportance`, `Plotter`, `main.py` + YAML; §4.6). **IMPresseD selector**: chain-only re-implementation of IMPresseD's automatic mode per §4.3–§4.4 (`Auto_IMPID` wrapped only as a cross-check, pending Part 6 Table B row B10): chain traces (`delta_time = -1`; day-level timestamps), case-distance attributes Age (numeric), Diagnosis, Diagnosis code, Treatment code, Specialism code (categorical; all are columns of the BPIC11 CSVs), exclude Variant / Variant index / Diagnosis Treatment Combination ID; **set projection** = distinct `value` of the nodes in each pattern graph, dedupe, keep sizes 1/2/3 (§4.3 recipe steps 3–5). **Symmetric Apriori**: `mlxtend.frequent_patterns.apriori(..., max_len=3)` (parameter exists in mlxtend 0.25.0), top-k per length incl. singletons (the original drops singletons, tools.py:171–172). Thu 15 Oct: Week 7 lecture | Variant lead + Pipeline lead | both selectors return `dict[int, list[str]]` in normalised names | both selectors yield ≥1 set per length on f1; one shared encoder/permutation path |
| Fri 16 – Wed 21 Oct | **Full experiments**: 3 datasets × 2 strategies × 3 lengths, seeded folds, overnight (≈56 min per dataset-strategy for 10 sets at repo defaults, measured on f1 from 2 iterations, env_report §3.3–3.4 [ASSUMPTION: linear scaling; f2 is larger]). Paper §1–3 in parallel | Experiments lead; Writing lead | result CSVs + figures; paper §1–3 | all 18 result tables exist; §2 figure drawn |
| Thu 22 Oct | **Week 8 lecture: Predictive Process Monitoring and Explainability** — attend; align terminology (post-hoc, model-agnostic, global; permutation feature importance) and cite the slides (not yet available locally, user_answers.md) | all | notes | terminology list updated |
| Fri 23 – Sat 24 Oct | **Comparison analysis**: Jaccard and overlap@k between the two set lists per length; Spearman/Kendall between importance rankings on shared sets; importance distributions; location- vs existence-importance per set; activity-level aggregation (mean/max importance of sets containing activity a); ≥3 fold seeds for stability [ASSUMPTION: metrics are not prescribed anywhere; the assignment only says "check if and how the importance of the activities and their locations changes"; confirm on 26 Oct]. Draft paper §4 | Experiments lead; Writing lead | `results/comparison/*`, paper §4 | each metric has a table + one figure |
| Sun 25 Oct 15:00 | Questions for office hour 2 due (send Sat 24 at the latest, all members in CC) | Writing lead | e-mail | sent |
| Mon 26 Oct 10:30–11:00 | Office hour 2: validate evaluation design, metrics, limitations | all | decision log | |
| Tue 27 Oct | **Buffer** / integrate feedback; §5 written; page-limit check; bib from DBLP | Writing lead + cross-reviewers | complete draft | compiles, ≤5 pages excl. refs, remarks box removed |
| Wed 28 Oct 23:59 | **Submit short-paper draft for peer review** | Writing lead | PDF on Canvas [UNVERIFIED: exchange mechanism] | uploaded |
| Thu 29 Oct – Mon 2 Nov | **Peer review of partner draft** (group; use Rubric_ShortPaper, include improvement directions). In parallel: multi-seed runs, code cleanup, docstrings, README v1. Thu 29 Oct: Week 9 lecture | QA lead drafts, all comment | review document | every S-item scored with justification |
| Tue 3 Nov 23:59 | **Submit peer feedback** | QA lead | on Canvas | uploaded |
| Wed 4 Nov | **Poster design starts** (template rule: only after a substantial draft) — Visio template or PowerPoint rebuild, A2 portrait; pick 3–4 key figures | Writing lead + Experiments lead | poster v1 | prints legibly at A2 |
| Thu 5 Nov | **Poster session 1** (Groups 1–6 present, 13:45–15:30 per their dates blocks): attend; each member reviews two groups individually | all individually | 2 reviews per member | submitted per Canvas instructions [UNVERIFIED mechanism] |
| Fri 6 – Sun 8 Nov | **Final polish**: incorporate partner feedback; README/requirements final; fresh-clone reproduction on another laptop (Windows and Mac); numbers in paper = numbers in `results/` = numbers on poster; technology statement; poster print order (Sun 8 = buffer) | all; QA lead signs off | final paper, code, poster PDF, tech statement | fresh-clone run reproduces the demo and matches committed results |
| Mon 9 Nov 23:59 | **Final submission (all deliverables)** | QA lead | Canvas upload | confirmation screenshot |
| Tue 10 – Wed 11 Nov | Pitch rehearsals (each member, timed to 3:00); Q&A drill; code-walkthrough rehearsal within 15 minutes; print poster, buy tape | all | | every member pitched once within time |
| Thu 12 Nov 13:45–15:30 | **Poster session 2**: one member always at the poster, others review two groups each (individual) | all | | |
| Thu 12 Nov 16:30–16:45 | **Explanation to lecturers** — bring laptop with the repo and venv ready | all | | |

### 5.3 Team split (4–5 members) and cross-review rule

Groups have 4–5 members (course rule, week1 overview slide 9). The roles below are a proposal [ASSUMPTION]; the user said the split is not important yet (user_answers.md).

| Role | Owns | Deliverable lines |
|---|---|---|
| Pipeline lead | 2024 reproduction, encoder, permutation engine, seeds, bug fixes documented | code core; paper §3 |
| Variant lead | chain-only IMPresseD re-implementation (§4.3–§4.4; `Auto_IMPID` wrapper only as a cross-check), set projection, symmetric Apriori, selection rule | `ImpressedSelector`, `AprioriSelector`; paper §2 |
| Experiments lead | run matrix, compute scheduling, comparison metrics, all figures | `results/`, paper §4 figures |
| Writing lead | paper text, bibliography (DBLP), poster, pitch script, office-hour mails | paper, poster |
| QA/integration lead (5th member, or shared by the two leads with least load) | README, requirements, fresh-clone tests on Windows and Mac, AI-usage log and technology statement, peer-review coordination, minutes | README, tech statement, reviews |

**Cross-review rule:** nothing is "done" until a member who did not produce it has (a) run it (code: merge request reviewed and executed on their machine — ideally the other OS) or (b) read it against the rubric item it serves (text/poster), and recorded the check in the decision log. Rotate roles for the pitch: everyone pitches, everyone can explain every module (12 Nov).

### 5.4 What to download / set up

| Item | Status / where | Note |
|---|---|---|
| Python 3.12 | Windows laptop: installed (`py -3.12`, 3.12.10); machine default `py` is 3.14.3 — do not use (env_report §0; pandas 3 breaks `index_encoding`, relayed from the orchestrator). Mac teammate: install Python 3.12 from https://www.python.org/downloads/ (or any installer that provides a `python3.12` command) | venv per §5.6; both OS forms below |
| Git | https://git-scm.com/ (Windows: ships Git Bash; macOS: Xcode command-line tools or the same installer) | needed for the fresh-clone test on every machine |
| Remote repository | TU/e GitLab https://gitlab.tue.nl (sign-in page https://gitlab.tue.nl/users/sign_in; Instructions_Implementation: "TUe has a GitLab license"; git beginners: https://ohmygit.org/) or GitHub | one protected `main`, merge requests |
| 2024 code + datasets, PermutationLocationImportance | https://github.com/MozhganVD/PermutationLocationImportance (default branch `master`; datasets folder https://github.com/MozhganVD/PermutationLocationImportance/tree/master/datasets — the group description l. 64 prints the link without the final "s") | already copied into `project/external/PermutationLocationImportance/` on 1 Oct (re-clone from the URL if you want the git history); the data files are `datasets/BPIC11_f1_trunc36.csv`, `f2_trunc40`, `f3_trunc31` (f4 exists but is not used in the paper's figures; §3.5) |
| 2023 code, InteractivePatternDetection | https://github.com/MozhganVD/InteractivePatternDetection (default branch `main`; cited as footnote 2 of the 2023 paper) | already copied into `project/external/InteractivePatternDetection/` on 1 Oct (re-clone from the URL if you want the git history); never import `GUI_IMPresseD_tool.py` (it calls `app.master.mainloop()` at module level, line 1027, no `__main__` guard); use `Auto_IMPID.py`/`IMIPD.py` |
| Scout scripts from this session | `smoke_2024.py`, `smoke_2024b.py`, `smoke_2024c.py`, `smoke_2023.py`, the `_fc_*.py` checks (`_fc_C.py`, `_fc_C2.py`, `_fc_C3.py`, `_fc_B_data.py`, `_fc_B_data2.py`, `_fc_B_checks.py`, `_fc_B_checks2.py`, `_fc_B_checks3.py`), the 2023-paper probes `fc_data.py`, `fc_data2.py`, the label checks (`_label_check.py`, `_cut_check.py`, `_code_check.py`, `_confusion.py`), the reports `env_report.md`, `labels_report.md` and the install log `reqtest.log` | [x] copied into `project/scripts/scout/` on 1 Oct (done on 1 Oct: they are in `project/scripts/scout/`); they are the starting point of every Part 6 Table C experiment |
| Teinemaa et al. benchmark (source of the BPIC11 LTL labels) | paper https://arxiv.org/abs/1707.06766; code https://github.com/irhete/predictive-monitoring-benchmark (labels_report) | optional; only needed for the data-provenance check (Part 6 Table C row C13) and for citing the labelling in paper §4 |
| PM4Py documentation | https://pm4py.fit.fraunhofer.de/ [UNVERIFIED: the Week 5 slides cite only the PM4Py paper, doi 10.1016/j.simpa.2023.100556, and no documentation URL]; the installed pm4py 2.7.23.8 prints "Docs & Examples: https://processintelligence.solutions/pm4py" in its import banner (verified in the install log) | needed for the Week 5 exercises (§1.6) and only if IMIPD's `VariantSelection` is reused (§4.5) |
| LaTeX | Overleaf (upload `ShortPaper_GroupX.zip`) or local TeX Live/MiKTeX — tooling choice postponed to the end by the student's choice (user_answers.md) | needs biblatex + its backend (biber by default) |
| Poster template `.vsdx` | Visio needed to edit [UNVERIFIED whether TU/e/JADS licenses Visio or whether any team laptop has it]; alternative: rebuild the four-box layout in PowerPoint at A2 portrait (ISO A2 = 420 × 594 mm) — tooling choice postponed to the end by the student's choice (user_answers.md) | template says "at least A2 and upright" |
| Teams | for both online office hours; the link arrives after the questions were sent (dates file header) — the 2 Oct questions are already sent | |
| Local already | Week 1–5 slides, Week 5 instruction sheet (`week5_putting_process_mining_into_code_with_pm4py.pdf`, `week5_instructions_exercises.docx`), Rubric.xlsx, both templates, Kickoff Work Plan docx, Instructions_Implementation.pdf, dates file | |
| Canvas items to fetch | technology-statement instructions and submission place; Week 8 (PPM) slides when published (not yet available); partner group for peer review; poster-review submission mechanism; the other groups' detailed descriptions (for S1) | |
| Optional | `shap` (only for the 2024 repo's `Classical_Permutation.py`), `flake8`/`black` | |

**Environment commands.** What the scout actually ran (env_report §1.1) was: `py -3.12 -m venv <absolute path>`, `pip install --upgrade pip`, then *unpinned* `pip install numpy "pandas<3" scikit-learn xgboost mlxtend networkx paretoset scipy matplotlib pyyaml`, `pip install pm4py`, optional `pip install shap`, `pip freeze`, `pip check`; the pins in §5.6 are the versions pip resolved on 2026-10-01. The Windows lines below are env_report §1.3's generic form, verbatim (relative `.venv`, `-r requirements.txt`); the fact-check re-ran them on 2026-10-01 in a fresh `py -3.12` venv with the pinned §5.6 file (install log `reqtest.log`, copied to `project/scripts/scout/`): install succeeded, `pip check` = "No broken requirements found", `import pandas, xgboost, mlxtend, pm4py, paretoset, yaml` OK (pandas 2.3.3, xgboost 3.4.1) — Windows 10 x64 only. Calling `.venv\Scripts\python.exe -m ...` directly (instead of activating) avoids PowerShell execution-policy problems (env_report §1.2).

```bash
# Windows (PowerShell or cmd) — tested, env_report §1.3 verbatim
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip check
# macOS (Mac teammate) — [UNVERIFIED: not tested on a Mac in this session]
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
pip check
# xgboost on macOS needs the OpenMP runtime: brew install libomp   [UNVERIFIED: not tested on a Mac in this session]
```

### 5.5 Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Compute time: ≈6.7 s per (itemset, repeat) on f1 (904 training cases) → ≈56 min per dataset for 5×10×10 (env_report §3.3); single-activity mode over all 164/207/156 activities (f1/f2/f3 after the repo's rare-activity filter; §3.7) could take many hours [ASSUMPTION: linear scaling from 2 measured iterations] | high | high | YAML `demo`/`full` configs; run overnight; cache traces as lists; parallelise over itemsets with an `if __name__ == "__main__"` guard (Windows uses spawn) |
| Upstream crashes under pandas 2.x (P1 dtype upcast at tools.py:536→538; P2: float `--top_k` crashes only when the flag is passed explicitly (§3.4) — the real blocker is the dtype crash in `itemset_permutation_importance` (env_report P1), fixed by casting encoded columns to int; P5 `DataFrame.append` tools.py:92, dead because `All_prefixes = False`) | certain | medium | int cast after encoding (verified, env_report §3.3), `type=int`, avoid prefixes; reimplement the encoder with one `get_dummies(..., dtype=int)` (also verified during fact-checking, Part 6 §6.3 "Closed by experiment") |
| Non-reproducible folds (no `random_state`, tools.py:231; fold-0 test F1 0.8888 vs 0.9064 in two runs, env_report §3.2–3.3) → poster numbers ≠ rerun | certain if ignored | high (14 pts) | seed everything; commit result CSVs; fresh-clone test |
| Original's "bugs" (accumulating permutations: `sub_data` built once at tools.py:507 and mutated in place at :526; position bookkeeping tools.py:469–474; train-fold scoring tools.py:503–504/538–539 while the paper says 5-fold CV, 2024 paper §4.1) — reproduce or fix? | high | medium | ask on 2 Oct / 26 Oct; implement a switch `faithful: true/false`, report both [ASSUMPTION] |
| IMPresseD end-to-end on BPIC11 untested with the new pins (env_report §4, [UNVERIFIED] there); `similarity_measuring_patterns` (IMIPD.py:37–60) calls `pair_cases.index(item, start)` — a linear list scan — for every in×out case pair, i.e. O(n_in × n_out × n) per pattern (§4.5) | medium | high | run the `Auto_IMPID` cross-check (Part 6 Table C row C3) on the first weekend (3–4 Oct) and start the chain-only re-implementation in week 2 (9–15 Oct), not later; replace the linear scan by the direct pair index (§4.5); chain traces (`delta_time = -1`) |
| Unclear definition of "length 1/2/3" and of the Pareto-vs-top-k selection (check_paper-2023 §D 1, 3) | high | medium | office-hour question; document the chosen rule in §2 of the paper |
| Day-level timestamps make "location" largely file order (88.8 / 87.1 / 88.5 % of consecutive event pairs in f1/f2/f3 share a timestamp (re-run, §2.12); an earlier check measured 93.5–94.3 % with a different per-case measure that was not re-run [UNVERIFIED]; check_paper-2024 §C 1) | certain | medium (interpretation) | state as limitation (S5), not as a bug |
| Both repos have a `tools.py` (env_report P7) | certain | low | rename modules |
| Week 8 lecture (22 Oct) comes 6 days before the draft (28 Oct) | certain | medium | self-study PPM/XAI from the two papers now |
| Missing technology statement | low | catastrophic (0 pts) | QA lead owns it; the statement is written at the end by the student's choice, but the AI-usage log starts now (Part 6 Table A row A6) [ASSUMPTION: log format] |
| Member cannot explain code on 12 Nov | medium | high | cross-review rule; walkthrough rehearsal 10–11 Nov |
| Poster printing lead time / Visio unavailable | medium | medium | PowerPoint rebuild; order print by Fri 6 Nov [ASSUMPTION: print-service turnaround unknown] |
| Windows/Mac mix in the team (user_answers.md) | certain | medium | path-agnostic code (`pathlib`), both command forms in README, fresh-clone test on each OS |

### 5.6 README skeleton and requirements.txt to start from

`requirements.txt` (verbatim from the tested environment, env_report §6):

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

[UNVERIFIED] Whether these exact pins also resolve on macOS (Intel and Apple Silicon wheels) — the scout tested Windows 10 x64 only; the Mac teammate must run the install once early (Sat 3 – Sun 4 Oct).

README skeleton (module and config names are proposals [ASSUMPTION]; commands follow env_report §1.3):

```text
# Group 11 — Explainable PPM: activity-location importance with Apriori vs IMPresseD sets
## 1. What this does (3 sentences + pipeline figure)
## 2. Repository layout
main.py            # single entry point
config/demo.yaml   # minutes; config/full.yaml  # reproduces paper/poster numbers
src/ppm/ (loader.py, encoder.py, selectors/apriori.py, selectors/impressed.py,
          permutation.py, existence.py, comparison.py, plotting.py)
external/          # unmodified upstream repos (reference only)
data/              # BPIC11_f1_trunc36.csv, f2_trunc40, f3_trunc31
results/           # committed CSV/PNG outputs of config/full.yaml
## 3. Installation (Windows, Python 3.12)
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
## 3b. Installation (macOS/Linux, Python 3.12)   [UNVERIFIED: not tested on a Mac in this session]
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
# macOS only, before the first run: brew install libomp   (OpenMP runtime needed by xgboost)
## 4. Run
.venv\Scripts\python.exe main.py --config config/demo.yaml      # ~minutes   (macOS: .venv/bin/python ...)
.venv\Scripts\python.exe main.py --config config/full.yaml      # hours; see §6
## 5. Parameters (table: name, default, meaning, where it enters the pipeline)
## 6. Expected runtime and outputs (which files appear in results/, how they map to figures)
## 7. Reproducing the paper figures (one command per figure)
## 8. Known deviations from the original code (seeded folds, int cast, top_k int, ...)
## 9. Technology statement (or pointer to it) and licences (pm4py is AGPL-v3)
```

### 5.7 Definition of done

The project is done when all of the following hold: (1) `git clone` + the README commands on a clean Windows machine and a clean Mac reproduce `config/demo.yaml` without edits and the committed `results/` match a `config/full.yaml` rerun to the reported precision; (2) for f1, f2, f3 there are location-importance results for Apriori sets and IMPresseD sets at lengths 1, 2 and 3, plus the mandatory multi-activity reproduction of the 2024 paper's Fig. 5 (Apriori top-10) on f1–f3 with written deviations; the single-activity Fig. 3 reproduction over all activities is optional (f1 only, overnight; §3.7, Part 6 Table B row B16) and, if run, is reported with its deviations too; (3) the comparison section answers the assignment's question "if and how the importance of the activities and their locations changes" with at least one set-overlap metric, one rank-correlation metric and one activity-level view, each with stability over seeds [ASSUMPTION: metric choice]; (4) every number in paper and poster traces to a file in `results/`; (5) the paper is ≤5 pages excl. references, follows the template order, remarks box removed, references from DBLP/Zotero/Mendeley; (6) poster ≥A2 portrait printed, 3-minute pitch rehearsed by every member; (7) technology statement submitted in the required form; (8) three peer reviews delivered on time (one group, two individual); (9) every member can explain every module on 12 Nov; (10) all [UNVERIFIED]/[ASSUMPTION] items of this guide are resolved or explicitly listed as limitations.

---

## Part 6 — Open questions & uncertainties register

**ELI5.** This is the "we don't know yet" box of the whole guide. Every place where Parts 1–5 had to write [UNVERIFIED] (could not be confirmed in a source) or [ASSUMPTION] (a working guess so the plan could continue) is accounted for here: every tagged item is listed in Tables A–C or in the closed lists at the end of §6.3, sorted by *who can close it*: you (Table A), the supervisor (Table B) or a short experiment (Table C); what needs no action at all (textbook facts, readings) sits in the second closed list. Table D shows what the fact-checkers corrected in earlier drafts, so you know which parts were shaky before they were fixed.

**How this register was built.** Inputs: the residual-uncertainty lists of the five section verifiers (Parts 1–5); the "(D) questions" sections of check_paper-2024-uncovering.md (11 items) and check_paper-2023-impressed.md (8 items); the scout lists (env_report P1–P12, labels_report §8); and a grep for `[UNVERIFIED` / `[ASSUMPTION` over Parts 1–5 (Part 1 including the Week 5 add-on, §1.6; 61 tagged lines at the time of the grep — the macOS tag that Part 5 received in its final pass is listed as well). Duplicates were merged; the last column of every row says where the item came from.

**Tag legend.** [UNVERIFIED] = claimed by a prior report or by us but not confirmed in a primary source. [ASSUMPTION] = a working choice we made. "Default" = what the implementation/plan does if the row stays open. Priorities in Table B: **P1** = blocks the implementation design (ask on Fri 2 Oct if there is time, otherwise e-mail right after); **P2** = needed before the evaluation design is frozen (send by Sun 25 Oct 15:00 for office hour 2 on Mon 26 Oct 10:30–11:00); **P3** = nice to know.

**Timing facts (verified).** Office hour 1: Fri 2 Oct 15:00–15:30, Teams; the questions deadline (1 Oct 15:00) has passed and the six questions in §6.0 were sent (user answers, 1 Oct). Office hour 2: Mon 26 Oct 10:30–11:00, questions due Sun 25 Oct 15:00 (description_group11 ll. 12–15). Supervisor e-mail: l.genga@tue.nl (description). [ASSUMPTION] e-mail questions between office hours are acceptable; the description does not say.

### 6.0 Already sent to the supervisor (do not re-ask — collect the answers)

Verbatim from the user answers of 1 Oct; the arrow says what the guide already knows, and the "Follow-ups" line under each question lists the Table B rows that build on its answer, so you can collect everything on 2 Oct in one go (and see at once which B rows the answer closes).

1. "Can we find the implementation code of the PPM paper anywhere?" → already answered by the guide: github.com/MozhganVD/PermutationLocationImportance (2024 code and the BPIC11 CSVs; description_group11 l. 64) and the InteractivePatternDetection repository (2023 code; cloned locally for this guide, see §5.4). Still worth hearing whether she points to anything else.
   - Follow-ups in Table B: B17 (the scripts/CSVs behind Fig. 3 and Fig. 4 and the xgboost version — both missing from the repository). If she names other code, re-read B2 and B10 against it.
2. "The A-priori algorithm isn't really explained in the paper, how is it used or where can we find more information?" → §2.13 and §3.3.4 (`DataManager.frequent_activity_sets`, mlxtend `apriori`).
   - Follow-ups in Table B: B5 (top-k itemsets per size 1/2/3) and B6 (`min_support` per log; f3 needs a value below 0.5).
3. "Do we need to reproduce all 10 runs and the boxplots for itemset lengths 1, 2, and 3 across all three BPI11 variants?" → feeds B4, B14, B16.
   - Follow-ups in Table B: B1 (what "length 1/2/3" means for an activity set), B4 (is f4 excluded), B14 (seeded folds, reduced repeats, demo vs full configuration), B16 (single-activity Fig. 3 over all activities or only over the selected length-1 sets).
4. "Can we use the CrossValidation_ProcessPermutation.py script from the official repository and add our own itemset generator to it?" → feeds B2, B10.
   - Follow-ups in Table B: B2 (the four verified behaviours of the script: reproduce as-is or fix), B3 (train-fold vs held-out scoring), B10 (own package vs wrapping the script).
5. "Should we run [IMPresseD] automatically (Pareto front patterns of lengths 1, 2, and 3), or are we also expected to include interactive/expert selection?" → feeds B7, B10.
   - Follow-ups in Table B: B7 (whole front or fixed k per length), B8 (select-then-project vs re-scoring the sets), B10 (automatic mode only), B12 (the IMPresseD configuration for BPIC11).
6. "Is there a risk that [the expert in the loop] might introduce a bias?" → discussion question; the answer belongs in the Limitations section (S5).
   - Follow-ups in Table B: none directly; the answer supports the automatic-mode default of B10 and is material for S5, together with the leakage point of B9.

### 6.1 Table A — removed

Table A ("things only you can tell me", rows A1–A14: Canvas dates, exam date, group roles, hardware, technology-statement format, peer-review partner, tooling, code hosting) was removed on 2 Oct at the student's request: none of these items matters for the project work right now. References to rows A1–A14 elsewhere in this guide can be ignored.

### 6.2 Table B — Questions to ask the supervisor (Laura Genga)

**ELI5.** Design choices where the assignment text is silent; guessing wrong costs days of compute or rubric points, so ask, and run with the default meanwhile.

| # | Priority / when | Question (ask it like this) | Why it matters | Recommended default if unanswered | Source / related sent question |
|---|---|---|---|---|---|
| B1 | **P1** — 2 Oct if time, else e-mail now (follow-up to sent Q3) | "The description asks for pattern lengths 1, 2 and 3 over activity sets. Do you mean (a) the number of *distinct activities* in the projected set, (b) the number of *nodes* in the IMPresseD pattern, or (c) the *extension step* 0/1/2 as on the x-axis of your 2023 Fig. 5? Should a self-loop A→A count as the length-1 set {A}? And should the Apriori side also be run per size 1/2/3?" | Defines the whole experimental grid and whether the two strategies are compared like for like. The 2023 paper defines only length-1 ("individual activities", p.307); one extension can add several nodes (P⁵ in Fig. 2 has 4 nodes); step-2 children of 3-node context parents have 4 nodes. | (a) distinct activities in the projected set; A→A → {A}; drop projected sets with > 3 activities; Apriori per size 1/2/3 with `max_len=3` [ASSUMPTION]. | check_2023 D1; verifiers B, D; §2.17 (glossary row "Pattern length"); §4.1 |
| B2 | **P1** (follow-up to sent Q4) | "Your 2024 code has four behaviours we verified by running it: (1) `sub_data` is built once (tools.py:507) and permuted in place, so from the second itemset on ≈97 % and from the fourth on 100 % of the scored traces are already shuffled; (2) `shuffle_sequence` (tools.py:469–474) leaves stale indexes, so on f1 ≈3.4 % of single-occurrence traces end with reversed itemset order and ≈3.1 % land on an unobserved position; (3) the folds are unseeded (`StratifiedKFold(shuffle=True)` without `random_state`, tools.py:231); (4) `--top_k` is parsed as float (CVPP:27) and crashes `.head()` when passed explicitly. Should we reproduce the code as-is, or fix these and report both?" | Rubric "correctness & reproducibility" 14 pts; results must match the paper/poster; fixing changes the numbers; the answer turns Part 3 §3.6 rows 1–5 from "bugs to document" into "behaviour to replicate". | `faithful: true/false` switch; headline results = fixed + seeded; one faithful run per dataset as a reproduction check; deviations listed in README §8 [ASSUMPTION]. | check_2024 D2; verifier C (open design question); §5.5; env_report P2/P3 |
| B3 | **P1** (follow-up to sent Q4) | "The 2024 paper describes 5-fold cross-validation, but the script passes the *training* fold to `itemset_permutation_importance` (CVPP:95–98), so the baseline F1 and the importances are computed on training data (train weighted F1 ≈ 0.99 on f1; the held-out F1 is only printed, CVPP:92). Should we score on the held-out fold instead?" | Train-fold scores are near-saturated; held-out scoring gives smaller but more honest importances and changes every figure. | Held-out fold in the fixed mode; train fold in the faithful mode; show both for f1 [ASSUMPTION]. | check_2024 D3; verifier C |
| B4 | **P1** (follow-up to sent Q3) | "The description says 'the three BPI11 datasets' and your 2024 figures use f1, f2, f3, but the repo also ships f4. Is f4 excluded?" | Each extra dataset costs ≈1 h per strategy at repo defaults, several hours over the full length grid. | f1–f3 only; f4 at most as an appendix run [ASSUMPTION]. | check_2024 D1; description_group11 ll. 63–64; verifier C |
| B5 | **P1** (follow-up to sent Q2) | "Your 2024 code takes the top-10 frequent itemsets of size > 1 (paper p.197; tools.py:171–172 drops singletons; no size cap). For a fair baseline per length, may we take the top-k itemsets by support *per size* 1, 2, 3 (mlxtend `apriori(..., max_len=3)`), and is k = 10 per length right?" | Comparability of Apriori vs IMPresseD sets per length is the core of the assignment. | k = 10 per size, ranked by support [ASSUMPTION]; mlxtend 0.25.0 `max_len` signature verified. | check_2023 D7; §4.4 |
| B6 | **P1** (follow-up to sent Q2) | "With `min_support = 0.5` f3 yields only 5 itemsets of size > 1 (f1 and f2: 10), yet your Fig. 5 f3 panel shows 10 rows, three starting with `ac370442`, which appear together only at `min_support` ≤ ≈0.494 (their supports are .496 / .494 / .494, §3.3.4). Which `min_support` did you use per log, and which should we use?" | Fig. 5 (f3) cannot be reproduced at 0.5. Per-size counts at 0.5 after the repo's filter: f1 16/94/258, f2 23/148/544, f3 5/4/1 (sizes 1/2/3); at 0.4 f3 has 20/130/572 (re-run in Part 4's fact-check). | 0.5 for f1/f2, 0.4 for f3, reported in the paper [ASSUMPTION: the authors' value is inferred, not stated]. | check_2024 D4; verifier C; §3.6 row 7 |
| B7 | **P1** (follow-up to sent Q5) | "IMPresseD returns a Pareto front of data-driven size (our probe on BPIC11 length 1 with IG + coverage + case distance: 19/32/29 patterns for f1/f2/f3; 3/4/1 with IG + coverage only). Should we use the whole front, or a fixed k = 10 per length to match Apriori? If fixed: fill by non-dominated-sorting layers or rank by one interest function? Keep `paretoset(distinct=True)`, which drops rows with tied objectives?" | Decides how many sets per length the variant produces and the fairness of the comparison; front sizes differ per log. | k = 10 per length filled layer by layer (front, then next front), tie-break by IG; `distinct=False`; report the raw front size [ASSUMPTION]. | check_2023 D3 (probe values not re-run); verifier D; §4.4 |
| B8 | **P1** (follow-up to sent Q5) | "After projecting patterns to activity sets, A→B, B→A and A⇢B all collapse to {A,B}. Should we (a) select patterns on the Pareto front and then project (keeping the best pattern per set), or (b) use IMPresseD only to generate candidate sets and re-score the *sets*? And one joint front per extension step or one front per length?" | This is the precise definition of the variant: how the ordering information is discarded. | (a) select then project, dedupe by set, one front per length [ASSUMPTION]. | check_2024 D6; check_2023 D4; §4.4 |
| B9 | **P1** | "Where should pattern discovery run: once on the whole log (as your 2024 Apriori baseline), only on the training part (as in your 2023 paper, p.315), or inside each CV fold? IMPresseD's outcome interest uses the labels, so whole-log mining leaks label information into the held-out folds." | Leakage vs. different sets per fold (which makes the fold-wise comparison harder). | Mine once on the whole log for both strategies and state the leakage as a limitation; one sensitivity run with train-only mining on f1 [ASSUMPTION]. | check_2024 D7; check_2023 D5; §2.14; §4.4 |
| B10 | **P1** (follow-up to sent Q4/Q5) | "May we re-implement the pipeline in our own object-oriented package — the 2023 code is GUI-bound, the 2024 code crashes under pandas ≥ 2 and both ship a module named `tools.py` — with a simplified chain-only IMPresseD (`delta_time < 0`, all six extension rules) instead of wrapping the originals? Is automatic (non-interactive) extraction sufficient?" | "Focus on the essence" (description l. 60); the rubric grades code structure (10 pts) and reproducibility (14 pts). | Re-implement, cite the originals, keep the semantics, add the faithful mode; automatic mode only [ASSUMPTION]. | check_2023 D8; env_report P1/P7/P8; sent Q4/Q5 (§6.0) |
| B11 | **P1** | "The goal is to 'check if and how the importance of the activities and their locations changes'. We plan: Jaccard / overlap@k between the two set lists per length; Spearman/Kendall between the importance rankings on shared sets; importance distributions (box plots as in Fig. 5); location vs existence importance per set; activity-level aggregation (mean/max importance of the sets containing an activity); stability over ≥ 3 fold seeds. Is that what you expect, or do you want a specific analysis?" | S4 "evaluation design & results" = 10 pts; the assignment prescribes no metric. | All of the above [ASSUMPTION: metric choice]. | check_2024 D11; verifier E; §5.2; §5.7 |
| B12 | **P2** — by 25 Oct (follow-up to sent Q5) | "IMPresseD configuration for BPIC11: timestamps are day-level (every event is stamped 23:00; 88.8 / 87.1 / 88.5 % of consecutive events in f1/f2/f3 share a timestamp), so we plan chain traces (`delta_time < 0`) with `event_nr` as the location, all six extension rules, `Max_gap_between_events = 3` (no value in paper or code), interest functions IG + coverage + case distance, case distance on Age (numeric) plus Diagnosis, Treatment code, Diagnosis code, Specialism code (categorical), excluding Variant / Variant index, Diagnosis Treatment Combination ID, timesincemidnight and hour; case distance as the code's mean (not the paper's 1/\|L\| sum) and without activity-count columns. Any objection?" | With `delta_time = 0` most of a trace becomes one concurrent block (prior probe: 78 % of parallel-flagged f1 events in blocks ≥ 10 nodes [UNVERIFIED in this run]); the gap decides which eventually-follows edges exist; the CD attributes change the front. | As stated [ASSUMPTION for gap = 3 and the CD choices]. | check_2023 D2/D6; check_2024 D10; verifiers B, D; §4.3 |
| B13 | **P2** | "Is the existence-importance baseline (blue boxes in Fig. 3/5) required? Your Classical_Permutation.py uses a substring min-count feature (:44–58), trains a fresh model on only those columns and calls sklearn `permutation_importance` without `scoring`, i.e. accuracy, although the axis says 'Decrease in f1 score'. Should we use count or 0/1 encoding, and accuracy or weighted F1?" | Decides a whole extra pipeline and whether the plot labels are honest. | Include it with count encoding and `scoring='f1_weighted'`, documented as a deviation; drop it if she says it is not needed [ASSUMPTION]. | check_2024 D9; verifier C; §3.3.7 |
| B14 | **P2** (follow-up to sent Q3) | "At repo defaults (5 folds × 10 sets × 10 repeats) one dataset-strategy takes ≈56–60 min on our laptop; the full grid (2 strategies × 3 lengths × 10 sets) is ≈5.6 h per dataset [linear extrapolation]. May we fix the fold seed and, if needed, reduce repeats/folds for the graded runs, provided both a demo config and the full config are in the repository?" | Runtime risk in the plan; without a seed the original folds are not reproducible (fold-0 test F1 0.8888 vs 0.9064 in two runs). | Seeded folds always; full defaults overnight; small `demo.yaml` [ASSUMPTION: linear scaling]. | check_2024 D11; verifiers C, D, E; env_report P3/P9 |
| B15 | **P2** | "In the shipped BPIC11 CSVs label 1 = the LTL rule is *satisfied* (f1: the patient eventually gets a CA-19.9/CA-125 test), the opposite of the formula displayed in Teinemaa et al. (1 = violated); the benchmark preprocessing script swaps 'deviant'/'regular' (preprocess_logs_bpic2011.py:63–65). Which class should we call positive in plots and text? Do you have a code-to-name table for the BPIC11 activity codes — is 376400 the CEA tumour-marker test? (It has MI 0.364 with the f1 label and 99.7 % label-1 among its 335 cases.)" | Plot legends and the interpretation of "important activities" (clinical proxy vs near-leak of the cut-out tests); the 2024 Fig. 4 draws label 0 as "positive". | Use the labels as shipped; name classes by rule ("rule satisfied" / "violated"), never "positive"; treat 376400 as CEA [ASSUMPTION] and raise the leak question in Limitations. | check_2024 D8; labels_report §8; verifier B |
| B16 | **P2** (follow-up to sent Q3) | "For length 1, should we reproduce the single-activity analysis over *all* activities (Fig. 3; 164 / 207 / 156 activities after the rare-activity filter → thousands of re-encodings, hours per log), or only over the length-1 sets selected by each strategy?" | Compute budget; the Fig. 3 reproduction is the costliest part. | Only the selected length-1 sets; full Fig. 3 as an optional overnight run on f1 [ASSUMPTION]. | check_2024 D5; verifier C |
| B17 | **P3** (follow-up to sent Q1) | "Which xgboost version produced the paper's figures (the repo has no requirements file)? And could you share the scripts/CSVs behind Fig. 3 and Fig. 4, whose legends are produced by no line of the shipped code?" | Default hyper-parameters differ by version; exact figure reproduction. | `XGBClassifier()` defaults of xgboost 3.4.1 (100 rounds, max_depth 6, eta 0.3, subsample 1), stated in the README; Fig. 3/4 reproduced "in spirit" [ASSUMPTION]. | verifiers B, C; §3.3.3; §3.3.6; §3.3.8 |
| B18 | **P3** (any lecturer) | "What format does the 12 Nov 16:30–16:45 assignment explanation have (code walkthrough? questions per member?), and are the instruction-sheet exercises representative of the exam questions?" | Oral check of the implementation (15-min slot); exam preparation. | Every member can walk `main.py` → selector → permutation → plots within 15 min; instruction sheets as the exam proxy [ASSUMPTION]. | verifiers A, E; §5.1.6; §1.1 |

### 6.3 Table C — Things we can resolve ourselves by experiment

**ELI5.** Nobody needs to be asked; a script (or a short read) settles it. All experiments run in a Python 3.12 venv (never the machine default `py` = 3.14), starting from the scout scripts, copied into `<project>/scripts/scout/` (see §5.4): `smoke_2024.py`, `smoke_2024b.py`, `smoke_2024c.py`, `smoke_2023.py`, `_fc_C*.py`, `_fc_B_*.py`.

| # | Open item | Experiment (what to run) | Decision it feeds | Source |
|---|---|---|---|---|
| C1 | Runtime: is scaling linear, how long are f2/f3 and the single-activity mode? Only 1–2 iterations were timed (6.7–7.2 s per (itemset, repeat) on f1, 904 training cases). | `smoke_2024c.py`-style loop with the int-cast workaround: 1 fold × 10 sets × 2 repeats on f1, f2, f3; seconds per iteration; then 5 single activities × 1 fold × 1 repeat; extrapolate to 5 × 10 × 10 and to 164 / 207 / 156 activities × 50. | B14, B16; work-plan dates; `demo.yaml` sizes | env_report P9; §3.7; §5.5 |
| C2 | Can the permutation loop be made much faster with identical semantics? (tools.py:512–514 filters a DataFrame per case; `index_encoding` emits 5,842 PerformanceWarnings on f1; speed-up from caching is [ASSUMPTION: not measured].) | `cProfile` one iteration; cache traces as Python lists; build the one-hot frame with a single `get_dummies(..., dtype=int)`/concat; assert bit-identical importances on a seeded run before/after. | B10 (re-implementation); runtime | env_report P4; §3.7; verifier C |
| C3 | Does `Auto_IMPID.AutoStepWise_PPD` run end-to-end on BPIC11 under the pinned §5.6 versions (pm4py 2.7.23.8, networkx 3.7, numpy 2.5.3, scipy 1.18.1, paretoset 1.2.5)? Only imports and tiny pm4py/paretoset calls were tested. | Headless script on f1 with `d_time=-1`, `Max_extension_step=1`, `Max_gap_between_events=3`, `pareto_features` = IG/CC/CD (signature at Auto_IMPID.py:9–13); record runtime and front sizes. | B7, B10; C4 | env_report P12; §4.5; verifier D |
| C4 | How slow is the isomorphism deduplication (prior probe ≈ 160 s for ac370000 with 435 patterns, check_paper-2023 B16, not re-run), and is a tuple key equivalent for chains? | Time one `Pattern_extension` pass on f1 for ac370000; implement tuple-key dedupe `(labels…, edge flags…)` and assert the same pattern set (the isomorphism test ignores the `parallel` flag and node positions). | B10 (chain-only re-implementation) | §4.5; verifier D |
| C5 | Oracle and gap sensitivity on BPIC11: block sizes with `delta_time = 0` (prior probe: 78 % of parallel-flagged f1 events in blocks ≥ 10 nodes; 31 % of directly-following instances with > 3 nodes — not re-run) and the effect of `Max_gap_between_events` ∈ {1, 2, 3, 5} on pattern counts and on the length-2/3 fronts. | Re-run the check_paper-2023 probes (`fc_data.py`, `fc_data2.py`, listed with the scout scripts in §5.4) in the Python 3.12 venv; run the chain-only extractor with each gap on f1; tabulate counts and set overlap between gaps. | B12 | check_2023 D2; verifiers B, D; §2.14; §4.3 |
| C6 | Pareto-front composition per length: re-check the front sizes (19/32/29 at length 1 with three objectives, 3/4/1 with two — probe, not re-run); how many step-2 children have ≤ 3 distinct activities (3-node context parents give 4-node children); effect of `distinct=True` (drops tied rows; NaN handling inconsistent across three small tests). | Run the extractor on f1/f2/f3 for steps 0–2; count patterns per number of distinct activities; run `paretoset` with `distinct` True/False on the same frame and diff. | B1, B7, B8 | check_2023 D3 and probe list; verifier D; §4.4 |
| C7 | Apriori per-size counts vs `min_support` (0.5: f1 16/94/258, f2 23/148/544, f3 5/4/1; 0.4: f1 23/211/1330, f2 26/286/2036, f3 20/130/572 — re-run in Part 4's fact-check; f3 at 0.49 has three ac370442 sets). | Sweep `min_support` ∈ {0.5, 0.49, 0.45, 0.4} with `apriori(..., max_len=3)` per log; pick the largest support giving ≥ 10 sets per size; keep the table for the paper. | B5, B6 | §3.6 row 7; verifier D |
| C8 | How much do the original's accumulation, bookkeeping and train-fold scoring change the result? (Lower bound: 96.9 % of scored traces already permuted at itemset #2 on f1.) | Run faithful vs fixed mode on f1 with the same seed and the same 10 sets; compare mean importances, Spearman ρ of the rankings and the location box plots. | B2, B3; Limitations section | verifier C; §3.3.6 |
| C9 | Seed stability: how much do rankings move between fold seeds? (Fold-0 test F1 was 0.8888 in one run and 0.9064 in the next.) | Fixed mode with 3 seeds at demo settings on f1; Kendall τ / Spearman ρ between seeds; choose `n_repeats` / `K_fold` for `full.yaml`. | B11, B14 | env_report P3; §5.2 |
| C10 | Existence baseline: do accuracy vs `f1_weighted` scoring and count vs 0/1 encoding change the existence ranking? | Re-implement the Classical_Permutation.py flow on f1 with both scorers and both encodings; compare rankings. | B13 | verifier C; §3.3.7 |
| C11 | paretoset 1.2.0 (the 2023 pin) vs 1.2.5 (installed): case-insensitive senses, `distinct=True`, NaN behaviour. | `pip install paretoset==1.2.0 --target <tmp>` and re-run the three tiny checks from Part 4's fact-check; if equal, pin 1.2.5 and say so in the README. | B7; README | verifier D |
| C12 | Case-distance categorical part: scipy `pdist(..., 'jaccard')` compares codes under scipy 1.11.4 (0.5 / 0.667 on the test pairs) but booleanises under 1.18.1 (0.0 / 0.333). Which scipy the authors used for the published figures, and the exact definition in their ref. [8], are [UNVERIFIED] (§2.14). | Write an explicit mismatch/Jaccard function with a unit test that reproduces the 1.11.4 values; use it instead of `pdist`. | B12 | verifier D correction |
| C13 | Data provenance: (a) code-to-name mapping (is 376400 = "CEA - tumor marker using meia"?); (b) semantics of deviant/regular in the benchmark's Google Drive input files; (c) how the authors derived `*_trunc*.csv` from the benchmark output (no script in the repo; every observable detail matches). | Download the original BPIC 2011 XES from 4TU (data.4tu.nl/repository/uuid:d9769f3d-0ab0-4fb8-803b-0d1120ffcf54) and inspect the "Activity code" vs activity-name attributes; download the Teinemaa benchmark's Drive folder (drive.google.com/open?id=154hcH-HGThlcZJW5zBvCJMZvjOQDsnPR) and diff with the shipped CSVs (case ids, labels, max event_nr). | B15; paper §4 data description | labels_report §8 items 1–3; verifier B |
| C15 | Do the pinned requirements install and import on the Mac teammate's machine (Intel / Apple Silicon)? Verified on Windows 10 x64 only (fresh pinned install re-verified); Part 5 therefore tags its macOS install block [UNVERIFIED: untested on macOS] (§5.4, §5.6). CVPP:44 `address.split('/')` is a path pitfall. | Mac teammate: `python3.12 -m venv .venv`, `pip install -r requirements.txt`, `pip check`, then `smoke_2024c.py` and `smoke_2023.py`; use `pathlib` everywhere. | README install section; definition of done; closes the §5.4/§5.6 macOS tag | verifier E; §5.6; env_report §1.3 |
| C16 | pm4py API names used in the Week-5 notes (§1.6) are [UNVERIFIED]: `pm4py.get_variants`, `filter_end_activities`, `get_all_case_durations`, `filter_activities_rework`, `filter_directly_follows_relation`, `filter_event_attribute_values`, `get_event_attribute_values`, and the exact `conformance_diagnostics_alignments(log, net, im, fm)` call. | In the Python 3.12 venv: `python -c "import pm4py, inspect; print(inspect.signature(pm4py.<name>))"` for each name; fix the notes; confirm the `pm4py>=2.7` pin. | Week-5 exercises (§1.6.3); exam preparation | §1.6.1, §1.6.2, §1.6.3, §1.6.5 |
| C17 | Literature look-ups (no experiment, but self-resolvable): (a) the Weijters, van der Aalst & Alves de Medeiros (2006) length-2-loop measure, reported as (\|a>>b\| + \|b>>a\|) / (\|a>>b\| + \|b>>a\| + 1); (b) XES as IEEE 1849-2016; (c) WF-net initial marking = one token in the source place; (d) whether Disco's process map is a frequency-annotated DFG; (e) "flower model" vocabulary. | Read the paper cited on week-4 slide 80, the XES standard page, the textbook chapter on WF-nets, and Disco's documentation; write the answers into Part 1. | Exam radar (§1.7) | verifier A; §1.2 (b, d); §1.3 (c); §1.4 (e); §1.5 (a) |
| C18 | §3.3.7 assumes that sklearn's `permutation_importance` also used the estimator's default scorer (accuracy for a classifier) in the older sklearn the authors ran, i.e. that `scoring=None` has always meant `estimator.score` [ASSUMPTION]. | Read the `scoring` parameter text of `sklearn.inspection.permutation_importance` in the documentation of every release since the function was introduced (or `pip download scikit-learn==<version>` and grep `_permutation_importance.py`); note the first version, if any, in which the default differs. | B13 (the "accuracy, not F1" claim in the question); §3.6 row 8 | verifier C; §3.3.7 |

**Closed by experiment during fact-checking (do not redo):**

- **C14 (closed, 1 Oct):** post-filter counts after the rare-activity removal (tools.py:60–64), re-run by the critic with `DataManager(path, 2, None, L_max_perc=0.8)` on all three logs: f1 1130 cases / 164 activities, f2 1130 / 207, f3 1111 / 156 — matches §3.3.1; use these in the paper's data table. (Before this re-run only f1 = 1130 and the activity counts had been verified; the f2/f3 case counts came from labels_report §8 item 4 and env_report §3.1.)
- `pd.get_dummies(..., dtype=int)` at tools.py:353 also fixes the pandas-2.x crash (7.2 s per iteration, no warnings); both fixes are verified (`_fc_C.py` [10]).
- The repo main script does **not** crash with default CLI args: argparse keeps the int default 10; only an explicit `--top_k N` becomes float and crashes `.head()` (env_report P2 was wrong on this point; `_fc_C3.py`).
- An empty shuffled-case frame raises `ValueError` at tools.py:340 (verified) — hence the "< 3 cases" guard in the re-implementation.
- `max()` of an empty list at tools.py:461 reproduces on a toy but no BPIC11 top-10 itemset (support 0.49) triggers it.
- Timestamp-tie shares 0.888 / 0.871 / 0.885 re-computed; `timesincemidnight` / `hour` take one value per file.
- PFI formula = baseline − mean(permuted score) verified against sklearn 1.9.1; precision/recall/F1 and weighted-F1 examples checked numerically.
- `XGBClassifier()` default fit is deterministic (bit-identical `predict_proba` twice); defaults read from `save_config()`.
- MI of 376400 on f1 = 0.364 re-computed (count and presence identical); sklearn MI is in nats (ln 2 = 0.693 for a perfect binary feature).
- Bookkeeping-bug rates: ≈14 % / ≈52 % on the toy, 3.4 % / 3.1 % on f1; non-itemset activities moved in 0 / 2.2 / 6.9 % of traces with 1 / 2 / 3 occurrences.
- Accumulation lower bound on f1 top-10: 96.9 % (#2), 99.7 % (#3), 100 % (from #4).
- Fig. 5 legend = Plotting_results.py:68 verbatim; the Fig. 3 and Fig. 4 legends are produced by no shipped line.
- Only the bpic2012 files carry `lifecycle:transition`.
- The pinned requirements.txt installs cleanly in a fresh `py -3.12` venv on Windows (exit 0, `pip check` clean, imports OK).

**Closed without an experiment — textbook facts, readings and working placements (the tag stays in the text; the item is listed here so that every tag in Parts 1–5 is accounted for):**

- §1.2 — "minimum for discovery = case id + activity + an ordering" is a reading of week-1 slide 21, not a sentence on it [ASSUMPTION]. No action; re-read the slide if challenged (Table D, Part 1, item 11).
- §1.2 — the slide-31 completeness answer ("no: the model also allows ⟨A,D,F⟩, ⟨A,C,B,D,F⟩") is derived by replaying the slide-31 model, not printed on the slide [ASSUMPTION, derived]. Checkable in two minutes by replaying the model yourself (Table D, Part 1, item 12).
- §2.9 — "weighted F1 was chosen because of class imbalance" [ASSUMPTION]: the 2024 paper says only "f1-score" (p.197), `average='weighted'` is visible only in the code (tools.py:503–504); the rationale is ours. No experiment can settle a design motive; mention it in passing with B13 if the metric comes up, otherwise present it as a reading in the paper (Table D, Part 2, item 5).
- §2.15 — information gain of a discrete feature = mutual information I(X;Y) [ASSUMPTION: textbook identity]; the paper says "information gain" (p.311, as cited in §2.15), the code calls `mutual_info_classif` (IMIPD.py:68); the toy checks agree (1 bit = 0.693 nats). No action.
- §2.16 — Spearman ρ = 1 − 6Σd²/(n(n²−1)) (no ties) and Kendall τ_a = (C − D)/(n(n−1)/2) [ASSUMPTION: textbook formulas]; both agree with scipy 1.18.1 on the worked examples (0.5 and 0.3333). No action; the implementation calls scipy anyway.
- §3.3.5 — which σ1 occurrence of {CR, ER} went where in the paper's σ1′ example is an inference [ASSUMPTION]; it is the only assignment consistent with the OL tuples (4,3) and (5,7). Nothing in the implementation depends on it (the code's own behaviour is verified, §3.6 rows 3 and 9); a P3 curiosity at most (Table D, Part 3, item 12).
- §3.3.8 — Fig. 4 came from a code state not in the repository (its "accepted / not accepted" legend and the single-activity block are produced by no shipped line) [ASSUMPTION]; folded into B17 (ask for the scripts behind Fig. 3/4); until then Fig. 4 is reproduced "in spirit" (Table D, Part 3, item 7).
- §5.1.4 — pm4py's AGPL-v3 licence belongs in the README/licence note, not in the technology statement (which is about AI tools) [ASSUMPTION]; a placement choice, revisit once A6 (the Canvas instructions for the statement) is answered.
- §2.3 — the LTL operator semantics (F = eventually, G = always, U = until, X = next) are stated from textbooks [ASSUMPTION]; the papers and the Teinemaa benchmark only use the symbols. No action; the benchmark's cut logic and the local data agree with this reading (labels_report §3).
- §2.6 — the definition of the log-loss objective is textbook [ASSUMPTION]; the objective name itself (`binary:logistic`) was read from the fitted XGBoost booster (§2.7). No action.
- §2.10 — the one-line descriptions of SHAP and LIME are textbook [ASSUMPTION]; the 2024 paper only names the methods. No action; neither is used in our pipeline.
- §2.14 / §2.17 — non-dominated sorting ("Pareto layers") is a textbook definition from multi-objective optimisation [ASSUMPTION]; it appears in neither paper. It is our proposed way of filling a fixed top-k from a Pareto front (Part 4 §4.3 step 6) and is part of question B7 to the supervisor.
- §2.14 — which scipy version the 2023 authors used for their published case-distance values, and the exact definition in their ref. [8], are [UNVERIFIED]; both are folded into C12 (the reimplementation computes the mismatch distance explicitly, so the answer only matters for reproducing the authors' numbers).
- §5.4 / §5.6 — the pinned `requirements.txt` is untested on macOS [UNVERIFIED] (tag added in Part 5's final pass): tracked by C15 and A5; closed by the Mac teammate's install on Sat 3 – Sun 4 Oct.

### 6.4 (D) Corrections the fact-checkers made to earlier drafts

**ELI5.** The "erratum" page: what an earlier draft said, what the source actually says, and what was changed. If you read a draft before the final parts, re-read these spots.

**Part 1 (course, weeks 1–4)**

1. "XES: IEEE XML standard" → the slides cite only Günther 2009 / xes-standard.org; the IEEE 1849 remark is now [UNVERIFIED from the course materials].
2. Slide-21 event table has only Timestamp / Activity / Resource; `Costs` appears in the p.23 XES excerpt and in running-example.csv, not in that table.
3. Places are "passive elements; events" (week-2 slide 5), not "conditions"; the latter is kept only as a gloss.
4. The Exercise 4c counterexample is on p.7 of the week-2 solutions, not p.4.
5. "Instruction sessions are the best predictor of exam questions" has no source → [ASSUMPTION].
6. "Disco's process map is a DFG" → the solutions only say "process model discovered with Disco" → [ASSUMPTION].
7. The WF-net initial marking (one token in the source place) is not part of the slide-33 definition → tagged textbook convention.
8. The running-example process tree is introduced on week-3 p.8 and repeated to p.13 (not only p.13).
9. "Flower model" is literature vocabulary; the slide (p.83) says only "Underfitting model".
10. The lecture reachability example is also "not deadlock-free (M4 = [0,0,0,2] enables nothing)".
11. "Minimum columns = case id + activity + ordering" is a reading of slide 21, not a sentence on it → [ASSUMPTION].
12. The week-1 slide-31 answer ("no: the model allows ⟨A,D,F⟩, ⟨A,C,B,D,F⟩") is derived, not printed → [ASSUMPTION, derived].
13. The date caveat got its evidence (2025 vs 2026 headers; Thursdays in 2026).
14. Added "caveats in the provided solutions": Instruction4_solutions lists the non-maximal pair ({e},{f}) in Y_L for L4; Table 6 c⇒e = 0.99 while 60/61 = 0.98; Ex 2.4 says L1 but uses L4.
15. Confirmed unchanged after re-checking: course code 1JM0211; the schedule table (plus room MDB 0.10 and the week-3/11/14 time details); running-example statistics and DFG counts; the week-4 numeric examples; the alpha limitation logs L7/L8; the §1.8 (bridge) repository claims.

**Part 2 (concepts)**

1. The "both local and global" quote is on p.192 (§1), not in §2.
2. Timestamp-tie shares were re-computed (0.888 / 0.871 / 0.885) → tag removed.
3. The PFI formula was verified numerically against sklearn → no longer "docstring only".
4. Accuracy/precision/recall/F1 formulas: tag removed, docstrings and toy checks cited.
5. "Weighted F1 was chosen because of class imbalance" → [ASSUMPTION]; the paper says only "f1-score"; "weighted" is visible only in the code.
6. The IG toy example: sklearn reports 0.693 nats, not 1 bit.
7. MI 0.364 of 376400 re-computed with label statistics (335 cases, 99.7 % label 1); only the leak-vs-proxy reading stays [UNVERIFIED].
8. XGBoost determinism verified in this run; `get_params()` returns `None` in xgboost 3.x.
9. The f3 Apriori count (5 at support 0.5) re-run with supports; the `.head(top_k + #activities)` quirk and the float-`top_k` crash added.
10. The prefix generator keeps the full trace and adds prefixes of length 2..n−1.
11. Classical_Permutation.py uses substring min-count features and trains only on the itemset columns.
12. The shipped code does not strictly enforce the two permutation constraints (bookkeeping bug).
13. The Fig. 4 label reading is cited to check_paper-2024 B3–B4 and marked "not re-rendered in this run".
14. Dominance definition quoted verbatim; Def. 3 nuance and the ∀/boundary discrepancy added; `paretoset distinct=True` caveat added.
15. The variant description quoted verbatim, with the 2024 p.201 and 2023 p.315 citations.
16. The outcome definition rephrased to Teinemaa et al. Def. 2.4.
17. Week-1 slide 39 shows remaining-time and outcome questions, not "next activity".

**Part 3 (2024 paper and code)**

1. `--top_k` row rewritten: flag omitted → int 10, script works; explicit `--top_k N` → float → TypeError (env_report P2 was wrong).
2. `get_dummies(dtype=int)` fix verified (was [UNVERIFIED]).
3. Empty frame → `ValueError` at tools.py:340 verified (was "[UNVERIFIED crash]").
4. Non-itemset activities *are* moved in multi-occurrence traces: 0 / 2.2 / 6.9 % (was [UNVERIFIED]).
5. Order-violation rates: ≈14 % / ≈52 % on the toy; 3.4 % / 3.1 % on f1 (the check report's 3.5 % / 2.8 % used a different trace selection).
6. Accumulation lower bound 96.9 / 99.7 / 100 % inserted (was [UNVERIFIED]).
7. Fig. 5 attributed to the shipped Plotting_results.py (legend verbatim); Fig. 3 to a modified copy [ASSUMPTION]; discrepancy row #15 added.
8. The Fig. 5 f3 panel's ac370442 rows verified from a render; "authors used support < 0.5" stays [ASSUMPTION].
9. Fig. 3 by-eye ranges re-measured on 400-dpi crops (±0.01).
10. Panel titles now from an own render.
11. Fig. 4b: 779 of 782 sepsis_1 cases contain leucocytes, 14.1 % label 1 (verified).
12. The σ1′ example rewritten with consistent (ER,CR) tuples; the occurrence assignment tagged [ASSUMPTION].
13. Activity counts 164 / 207 / 156 (not "~160–210"); f3 feature count 4,866 added.
14. CV ELI5 corrected: the held-out pile is only printed; the importance game runs on the four training piles.
15. XGBClassifier defaults stated from the venv (100 trees, eta 0.3, depth 6); the authors' version stays [UNVERIFIED].
16. `max()` of an empty list: reproducible on a toy, never triggered by a BPIC11 top-10 itemset.
17. `lifecycle:transition` exists only in the bpic2012 files; dist/Pixel scripts re-read the raw CSV without the frequency filter.
18. The PerformanceWarnings are pandas fragmentation warnings, not one-column-at-a-time code.

**Part 4 (2023 paper and the variant)**

1. scipy `pdist(..., 'jaccard')` semantics are version-dependent (1.11.4 compares codes; 1.18.1 booleanises); explicit Jaccard recommended.
2. GUI line references for the activity-count columns corrected (:394–395; filled at :419–431 / :990–1000).
3. paretoset NaN handling is inconsistent across three small tests (a NaN row can be kept and suppress valid rows).
4. "Dimensions chosen with clinicians" → "analyzing related literature and through discussions with domain experts" (p.310).
5. Dataset labeling counts come from Fig. 5, not the text; "10 event logs" is on p.316; the paper says "average F1-score with minimum and maximum" (not "weighted").
6. Step-2 children of 3-node context parents have 4 nodes and must be dropped unless a repeated activity collapses them to ≤ 3 distinct activities.
7. IG toy = 1 bit = 0.693 nats in sklearn.
8. `timesincemidnight` / `hour` take a single value in the whole file; distinct-value counts for the retained attributes added.
9. `pyperclip` is a real (GUI-only) dependency; xgboost, PyQt5, seaborn, future, Pillow are pinned but never imported.
10. Deduplication uses `categorical_node_match` for both nodes and edges (IMIPD.py:174–175, :642–643).
11. The "< 3 shuffled cases → importance 0" guard is needed because tools.py:529 would send an empty frame into `index_encoding`.
12. Four visual paper facts re-checked (∀n∈E′; P⁵ has 4 nodes; Fig. 3 shows signed OI, e.g. capecitabine_2 = −0.013; Fig. 5 x-axis "Extension Steps" 0/1/2).
13. 2024 paper p.197 "top 10 frequent itemsets of size greater than one" quoted verbatim (txt l. 315).
14. All reused numbers re-run (timestamp ties, variant counts 815/977/793, Apriori counts, paretoset signature and `distinct` behaviour, mlxtend `max_len`, eval line numbers, Plotting lines).

**Part 5 (deliverables and plan)**

1. The "hospital re-admission example" does not exist in the 2024 paper → replaced by its real motivation (process manager in a flexible process; redesign heuristics, p.193) and the BPIC11 gynaecology setting.
2. Groups to contrast in S1 verified as 7, 9, 10; only their detailed Canvas descriptions stay [UNVERIFIED].
3. The 1 Oct row rewritten: the office-hour questions were already sent; Week 5 is local; only Week 8 remains on the Canvas list.
4. IMPresseD parameter is `Max_gap_between_events` (Auto_IMPID.py:9–13), not `Max_gap`; the other real parameter names listed.
5. Install commands: the scout ran unpinned installs; the pinned requirements.txt was re-verified in a fresh venv on Windows only; macOS [UNVERIFIED].
6. Exact GitLab URL and the ohmygit.org pointer inserted.
7. Technology-statement ELI5: "a note in your own names", not "the signed note".
8. Poster facts added: no fixed pitch timing; rubric P1 note "content as such is graded mainly through the short paper".
9. The 12 Nov explanation slot is 15 minutes; format and depth [ASSUMPTION].
10. The Windows + Mac OS mix added to checklist, risks, definition of done and both command blocks; CVPP:44 `address.split('/')` noted.
11. Course-code naming note added (JM0210 / JM0211 / 1JM0211).
12. Single-activity counts made exact per dataset (164 / 207 / 156).
13. A risk-table row with pipes inside a cell broke the markdown table → rewritten.
14. mlxtend `apriori(max_len=3)` verified; the original's singleton drop (tools.py:171–172) noted.
15. Rubric/slide citations made precise (sheet/row/slide numbers); all point values and the AI-policy wording re-verified unchanged.

### 6.5 Counts and how to keep this register alive

- Table A: 14 items. Table B: 18 questions (11 × P1, 5 × P2, 2 × P3). Table C: 17 open experiments (C1–C13 and C15–C18; the id C14 is kept for the closed row) + 14 already closed by experiment + 9 tagged items closed without an experiment. Table D: 79 correction entries (15 + 17 + 18 + 14 + 15).
- Suggested handling: copy this part into the repository as `docs/open_questions.md`; when a row is answered, move it to a dated decision log (who answered, source, decision) and keep its id, as done for C14; every row still open on 9 Nov becomes a sentence in "Limitations and Future Work" (S5, 4 pts). This closes item (10) of the Part 5 definition of done (§5.7).
- Order of attack: A3 (collect the 2 Oct answers; the per-question follow-ups in §6.0 tell you which B rows each answer closes) → B1–B11 by e-mail if not covered on 2 Oct → C1, C3, C7, C18 (cheap, this weekend) and C15 (the Mac install, Sat 3 – Sun 4 Oct) → C8/C9 once the re-implementation runs → B12–B16 for office hour 2 (send by 25 Oct 15:00).
