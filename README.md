# Process Mining, Group 11: Explainable Predictive Process Monitoring

Course project, we reproduce the activity-location importance
method of Vazifehdoostirani et al. (2024) and build a variant in which the activity sets come
from the IMPresseD pattern discovery of Vazifehdoostirani et al. (2023) instead of Apriori.
Work in progress.

## What is here

| Path | What it is |
|---|---|
| `PROJECT_BRIEF_Group11.md` | Start here. The project explained in plain English, with the plan and the questions for the supervisor. |
| `WALKTHROUGH_Group11.md` | What we did, how and why, step by step, with a guided tour of the code and the results of each experiment. |
| `GUIDE_Group11_EN.md` | Long reference guide: concepts, both papers and their code step by step, open questions. |
| `project/experiments/` | Prototype modules and experiment scripts (permutation engine, Apriori selector, chain-only IMPresseD selector, case distance, existence importance) with tests. |
| `project/results/experiments/` | Results of the experiments, one folder and one `RESULT.md` per experiment. |
| `project/scripts/scout/` | Early check scripts and two reports (environment, meaning of the BPIC11 labels). |
| `project/requirements.txt` | Tested package versions (Python 3.12). |

## What is not here, and where to get it

The two papers are published by Springer and may not be redistributed, so they are linked, not included:

- Vazifehdoostirani, M., Abbaspour Onari, M., Grau, I., Genga, L., Dijkman, R. (2024). Uncovering the Hidden Significance of Activities Location in Predictive Process Monitoring. ICPM 2023 Workshops, LNBIP 503. https://doi.org/10.1007/978-3-031-56107-8_15
- Vazifehdoostirani, M., Genga, L., Lu, X., Verhoeven, R., van Laarhoven, H., Dijkman, R. (2023). Interactive Multi-interest Process Pattern Discovery. BPM 2023, LNCS 14159. https://doi.org/10.1007/978-3-031-41620-0_18

The authors' code and the BPIC11 data are not copied into this repository either. Clone them into `project/external/`:

```bash
git clone https://github.com/MozhganVD/PermutationLocationImportance project/external/PermutationLocationImportance
```

```bash
git clone https://github.com/MozhganVD/InteractivePatternDetection project/external/InteractivePatternDetection
```

## Setup

Use Python 3.12.

```bash
# Windows
py -3.12 -m venv project/.venv
project/.venv/Scripts/python.exe -m pip install -r project/requirements.txt
```

```bash
# macOS / Linux (not tested yet)
python3.12 -m venv project/.venv
project/.venv/bin/python -m pip install -r project/requirements.txt
```
