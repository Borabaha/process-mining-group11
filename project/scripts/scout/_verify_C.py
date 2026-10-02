"""Verification script for draft_C (Part 3). Read-only on sources; prints facts."""
import sys, os, warnings, inspect
warnings.filterwarnings("ignore")
REPO = "C:/Users/Bohabara/AppData/Local/Temp/claude/C--Users-Bohabara-Desktop-process-mining/05db42ec-3109-4098-847a-cf389d8c070a/scratchpad/repos/PermutationLocationImportance"
sys.path.insert(0, REPO)
import numpy as np, pandas as pd
from tools import DataManager

print("=== [A] Apriori on f3 at min_support 0.5 and 0.49 (repo's own frequent_activity_sets) ===")
dm3 = DataManager(REPO + "/datasets/BPIC11_f3_trunc31.csv", 2, None, L_max_perc=0.8)
print("f3 cases after filter:", dm3.data[dm3.case_id].nunique(), "| activities:", dm3.data[dm3.activity].nunique())
for ms in (0.5, 0.49, 0.45):
    lst, sel = dm3.frequent_activity_sets(ms, 10)
    print(f"min_support={ms}: {len(lst)} itemsets of size>1 returned")
    for _, r in sel.iterrows():
        print(f"   support={r['support']:.4f} size={r['item_size']} {sorted(r['itemsets'])}")

print("\n=== [B] f1 top-14 supports (tie check at the top-10 cutoff) ===")
dm1 = DataManager(REPO + "/datasets/BPIC11_f1_trunc36.csv", 2, None, L_max_perc=0.8)
lst, sel = dm1.frequent_activity_sets(0.5, 14)
for i, (_, r) in enumerate(sel.iterrows(), 1):
    print(f"   rank {i:2d} support={r['support']:.6f} size={r['item_size']} {sorted(r['itemsets'])}")

print("\n=== [C] substring bug: activities in f1/f2 whose name contains '370407' ===")
for name, dm in (("f1", dm1), ("f3", dm3)):
    acts = [a for a in dm.data[dm.activity].unique() if "370407" in a]
    print(f"   {name}: {acts}")
dm2 = DataManager(REPO + "/datasets/BPIC11_f2_trunc40.csv", 2, None, L_max_perc=0.8)
print("   f2:", [a for a in dm2.data[dm2.activity].unique() if "370407" in a])

print("\n=== [D] sklearn permutation_importance default scoring ===")
from sklearn.inspection import permutation_importance
from sklearn.base import ClassifierMixin
from xgboost import XGBClassifier
sig = inspect.signature(permutation_importance)
print("   permutation_importance scoring default:", sig.parameters["scoring"].default)
print("   XGBClassifier is ClassifierMixin:", issubclass(XGBClassifier, ClassifierMixin))
print("   XGBClassifier.score defined in:", XGBClassifier.score.__qualname__)
src = inspect.getsource(ClassifierMixin.score)
print("   ClassifierMixin.score uses accuracy_score:", "accuracy_score" in src)

print("\n=== [E] shuffle_sequence bookkeeping bug on a toy trace ===")
dmx = DataManager.__new__(DataManager)
# every activity may sit at any 1-based position 1..5
dmx.Allowed_locations = {a: [1, 2, 3, 4, 5] for a in "ABCDE"}
trace = list("ABCDE")
print("   find_itemset_indexes(ABCDE, {A,C}) =", dmx.find_itemset_indexes(trace, {"A", "C"}))
# force the draws: monkeypatch np.random.choice to return chosen values in order
draws = iter([3, 4])  # A -> position 3, C -> position 4 (both 1-based)
orig_choice = np.random.choice
np.random.choice = lambda arr, n: np.array([next(draws)])
try:
    out = dmx.shuffle_sequence(trace, {"A", "C"})
finally:
    np.random.choice = orig_choice
print("   toy: trace ABCDE, itemset {A,C}, draws A->3, C->4 ; result:", "".join(out),
      "| expected (paper semantics): B D A C E or similar with A before C")
print("   A index:", out.index("A"), "C index:", out.index("C"), "-> order violated:", out.index("A") > out.index("C"))

# Monte-Carlo: how often is order violated / non-itemset activity moved, uniform pools
np.random.seed(0)
n_viol = 0; n_moved_other = 0; N = 20000
dmx.Allowed_locations = {a: list(range(1, 9)) for a in "ABCDEFGH"}
base = list("ABCDEFGH")
for _ in range(N):
    out = dmx.shuffle_sequence(base, {"B", "E"})
    if out.index("B") > out.index("E"):
        n_viol += 1
    # non-itemset activities keep relative order among themselves if only B,E moved
    others = [x for x in out if x not in ("B", "E")]
    if others != [x for x in base if x not in ("B", "E")]:
        n_moved_other += 1
print(f"   MC 2-itemset {{B,E}} in ABCDEFGH, uniform pools 1..8, N={N}: order violated {n_viol/N:.3%}, "
      f"non-itemset relative order changed {n_moved_other/N:.3%}")
np.random.seed(0)
n_viol = 0; n_moved_other = 0
for _ in range(N):
    out = dmx.shuffle_sequence(base, {"B", "D", "F"})
    ok = out.index("B") < out.index("D") < out.index("F")
    if not ok:
        n_viol += 1
    others = [x for x in out if x not in ("B", "D", "F")]
    if others != [x for x in base if x not in ("B", "D", "F")]:
        n_moved_other += 1
print(f"   MC 3-itemset {{B,D,F}} in ABCDEFGH, N={N}: order violated {n_viol/N:.3%}, "
      f"non-itemset relative order changed {n_moved_other/N:.3%}")

print("\n=== [F] Allowed_locations beyond trace length: example from f1 ===")
act = "ac370000"
print("   f1 Allowed_locations[ac370000] (first 5, last 5):", dm1.Allowed_locations[act][:5], dm1.Allowed_locations[act][-5:])
lens = dm1.data.groupby(dm1.case_id).size()
print("   f1 trace length min/median/max:", lens.min(), lens.median(), lens.max())
print("   cases shorter than 10 events:", int((lens < 10).sum()))

print("\n=== [G] Multi_activity bool CLI check ===")
print("   bool('False') =", bool("False"), "| bool('') =", bool(""))

print("\n=== [H] paper OL example recomputed from Table 1 ===")
t1 = {
 1: "RG LA LE CR ER ST IL IA NC ER CR LE CR".split(),
 2: "RG ER ST IL LE CR IA LA IC DI".split(),
 3: "RG LA ER ST LE CR IA NC CR LE CR LE CR".split(),
 4: "RG LE LE LA ER ST CR IA IC LE LA NC DI".split(),
 5: "RG ER ST LE CR IL IA LA NC CR LE".split(),
 6: "RG LA CR ER ST LE IL IA NC DI".split(),
}
dmy = DataManager.__new__(DataManager)
for cid, tr in t1.items():
    occ = dmy.find_itemset_indexes(tr, {"ER", "CR"})
    print(f"   case {cid}: occurrences (0-based) {occ} -> 1-based (ER,CR) tuples:",
          [(tr.index('ER', min(o)) + 1 if False else [i+1 for i in o if tr[i]=='ER'][0], [i+1 for i in o if tr[i]=='CR'][0]) for o in occ])
    print("      LA positions (1-based):", [i+1 for i,a in enumerate(tr) if a=="LA"])
