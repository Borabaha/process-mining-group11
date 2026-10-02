"""Fact-check script for final_C (read-only on sources)."""
import sys, warnings, inspect, functools, itertools, time
warnings.filterwarnings("ignore")
REPO = "C:/Users/Bohabara/AppData/Local/Temp/claude/C--Users-Bohabara-Desktop-process-mining/05db42ec-3109-4098-847a-cf389d8c070a/scratchpad/repos/PermutationLocationImportance"
sys.path.insert(0, REPO)
import numpy as np, pandas as pd
import tools
from tools import DataManager

print("pandas", pd.__version__, "numpy", np.__version__)
import xgboost, sklearn, mlxtend
print("xgboost", xgboost.__version__, "sklearn", sklearn.__version__, "mlxtend", mlxtend.__version__)

print("\n=== [0] CSV headers ===")
for f in ["BPIC11_f1_trunc36", "BPIC11_f2_trunc40", "BPIC11_f3_trunc31", "bpic2012_1_trunc40", "sepsis_cases_1_Trunc30"]:
    h = pd.read_csv(REPO + f"/datasets/{f}.csv", nrows=2)
    print(f, list(h.columns))

print("\n=== [1] DataManager stats ===")
dms = {}
for f in ["f1_trunc36", "f2_trunc40", "f3_trunc31"]:
    dm = DataManager(REPO + f"/datasets/BPIC11_{f}.csv", 2, None, L_max_perc=0.8)
    dms[f[:2]] = dm
    d = dm.data
    nact = d[dm.activity].nunique()
    maxlen = d.groupby(dm.case_id).size().max()
    print(f, "cases", d[dm.case_id].nunique(), "acts", nact, "maxlen", maxlen,
          "features", maxlen * nact + (maxlen - 1), "L_max", dm.L_max)
dm1 = dms["f1"]
al = dm1.Allowed_locations["ac370000"]
print("f1 Allowed_locations[ac370000] == list(range(1,37)):", al == list(range(1, 37)), "len", len(al))
lens = dm1.data.groupby(dm1.case_id).size()
print("f1 cases shorter than 10 events:", int((lens < 10).sum()), "| median", lens.median())

print("\n=== [2] Apriori ===")
for name, dm in dms.items():
    for ms in (0.5, 0.49):
        lst, sel = dm.frequent_activity_sets(ms, 10)
        print(f"{name} min_support={ms}: {len(lst)} itemsets")
        if name == "f3" or (name == "f1" and ms == 0.5):
            for _, r in sel.iterrows():
                print(f"    {r['support']:.4f} {sorted(r['itemsets'])}")
lst, sel = dm1.frequent_activity_sets(0.5, 14)
print("f1 ranks 8-13 supports:", [round(s, 4) for s in sel['support'].tolist()[7:13]])
acts_used = set(itertools.chain.from_iterable(dm1.frequent_activity_sets(0.5, 10)[0]))
print("f1 top-10 distinct activities:", sorted(acts_used), len(acts_used))

print("\n=== [3] substring 370407c ===")
for name, dm in dms.items():
    print(name, [a for a in dm.data[dm.activity].unique() if "370407" in a])

print("\n=== [4] sklearn default scorer for XGBClassifier ===")
from sklearn.inspection import permutation_importance
from sklearn.base import ClassifierMixin
from xgboost import XGBClassifier
print("scoring default:", inspect.signature(permutation_importance).parameters["scoring"].default)
print("XGBClassifier.score qualname:", XGBClassifier.score.__qualname__,
      "| accuracy_score in source:", "accuracy_score" in inspect.getsource(ClassifierMixin.score))

print("\n=== [5] empty frame to index_encoding ===")
try:
    dm1.index_encoding(dm1.data[dm1.data[dm1.case_id].isin([])])
    print("no crash")
except Exception as e:
    print("CRASH:", type(e).__name__, str(e)[:120])

print("\n=== [6] max() of empty list: toy + real top-10 itemsets (all activity orders) ===")
dmx = DataManager.__new__(DataManager)
dmx.Allowed_locations = {"A": [5], "B": [3]}
try:
    dmx.shuffle_sequence(["A", "B"], {"A", "B"})
    print("toy: no crash")
except Exception as e:
    print("toy CRASH:", type(e).__name__, str(e)[:80])
for name, dm in dms.items():
    lst, _ = dm.frequent_activity_sets(0.49, 10)
    bad = 0
    for its in lst:
        for order in itertools.permutations(its):
            max_location = 10000
            try:
                for act in order[::-1]:
                    m = max([a for a in dm.Allowed_locations[act] if a < max_location])
                    max_location = m
            except ValueError:
                bad += 1
    print(name, "orders that would crash the backward pass:", bad)

print("\n=== [7] bookkeeping bug toy + Monte Carlo ===")
dmx.Allowed_locations = {a: [1, 2, 3, 4, 5] for a in "ABCDE"}
draws = iter([3, 4]); orig = np.random.choice
np.random.choice = lambda arr, n: np.array([next(draws)])
try:
    out = dmx.shuffle_sequence(list("ABCDE"), {"A", "C"})
finally:
    np.random.choice = orig
print("toy ABCDE {A,C} draws 3,4 ->", "".join(out))
np.random.seed(0)
dmx.Allowed_locations = {a: list(range(1, 9)) for a in "ABCDEFGH"}
base = list("ABCDEFGH"); N = 20000
for its in ({"B", "E"}, {"B", "D", "F"}):
    v = 0; mo = 0
    for _ in range(N):
        out = dmx.shuffle_sequence(base, its)
        ordered = sorted(its, key=base.index)
        if [out.index(x) for x in ordered] != sorted(out.index(x) for x in ordered):
            v += 1
        if [x for x in out if x not in its] != [x for x in base if x not in its]:
            mo += 1
    print(f"MC {sorted(its)} single occurrence: order violated {v/N:.1%}, non-itemset moved {mo/N:.1%}")
# two occurrences toy
np.random.seed(0)
base2 = list("ABCDABCD")
dmx.Allowed_locations = {a: list(range(1, 9)) for a in "ABCD"}
v = 0; mo = 0; keep = 0
for _ in range(N):
    out = dmx.shuffle_sequence(base2, {"A", "C"})
    if [x for x in out if x not in ("A", "C")] != ["B", "D", "B", "D"]:
        mo += 1
print(f"MC two-occurrence trace ABCDABCD {{A,C}}: non-itemset relative order changed {mo/N:.1%}")

print("\n=== [8] real f1: {ac370419, ac370442} single-occurrence traces ===")
np.random.seed(2023)
its = {"ac370419", "ac370442"}
traces = dm1.data.groupby(dm1.case_id)[dm1.activity].apply(list)
n = 0; v = 0; infeasible = 0; other_moved = 0
for tr in traces:
    occ = dm1.find_itemset_indexes(tr, its)
    if len(occ) != 1 or len(tr) == 2:
        continue
    n += 1
    i0 = occ[0]
    first, second = tr[i0[0]], tr[i0[1]]
    out = dm1.shuffle_sequence(tr, its)
    if out.index(first) > out.index(second):
        v += 1
    for a in its:
        if (out.index(a) + 1) not in dm1.Allowed_locations[a]:
            infeasible += 1; break
    if [x for x in out if x not in its] != [x for x in tr if x not in its]:
        other_moved += 1
print(f"traces with exactly one occurrence: {n}; order reversed {v} ({v/n:.1%}); itemset act at unobserved position {infeasible} ({infeasible/n:.1%}); non-itemset order changed {other_moved}")

print("\n=== [9] accumulation: share of cases containing itemset k that contain an earlier itemset ===")
lst, _ = dm1.frequent_activity_sets(0.5, 10)
sets = [set(x) for x in lst]
tsets = {c: set(t) for c, t in traces.items()}
for k in range(1, len(sets)):
    cases_k = [c for c, s in tsets.items() if sets[k] <= s]
    pre = [c for c in cases_k if any(sets[j] <= tsets[c] for j in range(k))]
    print(f"itemset #{k+1} {sorted(sets[k])}: {len(pre)}/{len(cases_k)} = {len(pre)/len(cases_k):.1%}")

print("\n=== [10] get_dummies(dtype=int) alternative fix, 1 itemset x 1 repeat ===")
tools.pd.get_dummies = functools.partial(pd.get_dummies, dtype=int)
t = time.time()
enc = dm1.index_encoding(dm1.data)
print("encode", round(time.time() - t, 1), "s; dtypes:", enc.drop(columns=[dm1.case_id, dm1.outcome]).dtypes.value_counts().to_dict())
tr_list, te_list = dm1.cross_split_test_train(5)
train_x = enc[enc[dm1.case_id].isin(tr_list[0])]
train_y = train_x[dm1.outcome].tolist()
train_x = train_x.drop([dm1.case_id, dm1.outcome], axis=1)
model = XGBClassifier(); model.fit(train_x, train_y)
t = time.time()
try:
    res = dm1.itemset_permutation_importance(model, train_x, train_y, tr_list[0], {0: lst[0]}, n_repeats=1)
    print("OK, importance", res.iloc[0, 0], "in", round(time.time() - t, 1), "s")
except Exception as e:
    print("CRASH:", type(e).__name__, str(e)[:200])
