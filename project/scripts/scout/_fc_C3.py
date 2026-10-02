import sys, warnings, argparse, random; warnings.filterwarnings("ignore")
REPO = "C:/Users/Bohabara/AppData/Local/Temp/claude/C--Users-Bohabara-Desktop-process-mining/05db42ec-3109-4098-847a-cf389d8c070a/scratchpad/repos/PermutationLocationImportance"
sys.path.insert(0, REPO)
import numpy as np, pandas as pd
from tools import DataManager
p = argparse.ArgumentParser(); p.add_argument('--top_k', default=10, type=float); p.add_argument("--constrain", type=bool, default=True)
a = p.parse_args([]); print("no flags: top_k =", repr(a.top_k), type(a.top_k).__name__)
a = p.parse_args(["--top_k", "10", "--constrain", "False"]); print("--top_k 10 --constrain False:", repr(a.top_k), "constrain =", a.constrain)
try:
    pd.DataFrame({'a': range(200)}).head(174.0); print("head(174.0) ok")
except TypeError as e: print("head(174.0) TypeError:", str(e)[:70])

print("\n=== broader MC: random traces with 1-3 occurrences of {A,C}, random pools ===")
dmx = DataManager.__new__(DataManager)
rng = random.Random(1); np.random.seed(1)
stats = {1: [0,0,0], 2: [0,0,0], 3: [0,0,0]}  # n, order_viol, nonitemset_moved
for _ in range(30000):
    L = rng.randint(6, 14)
    tr = [rng.choice("BDEFGH") for _ in range(L)]
    k = rng.randint(1, 3)
    pos = rng.sample(range(L), 2*k)
    for i, q in enumerate(pos): tr[q] = "A" if i % 2 == 0 else "C"
    dmx.Allowed_locations = {x: sorted(rng.sample(range(1, L+1), rng.randint(2, L))) for x in set(tr)}
    for x in ("A", "C"):
        dmx.Allowed_locations[x] = sorted(set(dmx.Allowed_locations[x]) | {i+1 for i, v in enumerate(tr) if v == x})
    occ = dmx.find_itemset_indexes(tr, {"A", "C"})
    n_occ = len(occ)
    try:
        out = dmx.shuffle_sequence(tr, {"A", "C"})
    except ValueError:
        continue
    s = stats[n_occ]; s[0] += 1
    others_before = [x for x in tr if x not in ("A", "C")]
    others_after = [x for x in out if x not in ("A", "C")]
    if others_before != others_after: s[2] += 1
    # order violation: for each original occurrence, first act should precede second act... approximate: compare sequence of A/C symbols
    if [x for x in out if x in ("A","C")] != [x for x in tr if x in ("A","C")]: s[1] += 1
for k, (n, v, m) in stats.items():
    print(f"{k} occurrence(s): n={n}, A/C pattern changed {v/n:.1%}, non-itemset relative order changed {m/n:.1%}")

print("\n=== sepsis_1 leucocytes ===")
dm = DataManager(REPO + "/datasets/sepsis_cases_1_Trunc30.csv", 2, None)
d = dm.data; cases = d[dm.case_id].nunique()
has = d[d[dm.activity] == "leucocytes"][dm.case_id].unique()
lab = d.drop_duplicates(dm.case_id).set_index(dm.case_id)[dm.outcome]
print("cases", cases, "| with leucocytes", len(has), "| label-1 share among them", round((lab.loc[has] == 1).mean(), 3), "| overall label-1 share", round((lab == 1).mean(), 3))
print("f2 ac379999 support:", round(dm.__class__.__name__ == "DataManager" and 0, 2))
dm2 = DataManager(REPO + "/datasets/BPIC11_f2_trunc40.csv", 2, None)
t2 = dm2.data.groupby(dm2.case_id)[dm2.activity].apply(set)
print("f2 share of cases with ac379999:", round(t2.apply(lambda s: "ac379999" in s).mean(), 3), "| {ac379999, ac370000}:", round(t2.apply(lambda s: {"ac379999","ac370000"} <= s).mean(), 3))
lst, sel = dm2.frequent_activity_sets(0.5, 10)
print("f2 top-10:", [(round(r['support'],3), sorted(r['itemsets'])) for _, r in sel.iterrows()])
