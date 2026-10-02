"""Smoke test 2 for the 2024 repo: one fold, two itemsets, n_repeats=1 of itemset_permutation_importance."""
import os
import sys
import time
import traceback
import warnings

REPO = ("C:/Users/Bohabara/AppData/Local/Temp/claude/C--Users-Bohabara-Desktop-process-mining/"
        "05db42ec-3109-4098-847a-cf389d8c070a/scratchpad/repos/PermutationLocationImportance")
os.chdir(REPO)
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from tools import DataManager  # noqa: E402

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

dm = DataManager('datasets/BPIC11_f1_trunc36.csv', 2, None, L_max_perc=0.8)
cand, itemsets = dm.frequent_activity_sets(0.5, 10)
candidate_itemsets = {i: s for i, s in enumerate(cand)}
enc = dm.index_encoding(dm.data)

# Fold 0 exactly as CrossValidation_ProcessPermutation.py:71-88 (folds are NOT seeded in the repo)
train_list, test_list = dm.cross_split_test_train(5)
print("[5] cross_split_test_train(5): type=%s, keys=%s" % (type(train_list).__name__, list(train_list.keys())))
print("    train sizes:", [len(train_list[i]) for i in train_list], "| test sizes:", [len(test_list[i]) for i in test_list])

i = 0
train_x = enc[enc[dm.case_id].isin(train_list[i])]
train_y = train_x[dm.outcome].tolist()
train_x = train_x.drop([dm.case_id, dm.outcome], axis=1)
test_x = enc[enc[dm.case_id].isin(test_list[i])]
test_y = test_x[dm.outcome].tolist()
test_x = test_x.drop([dm.case_id, dm.outcome], axis=1)

t0 = time.perf_counter()
model = XGBClassifier()
model.fit(train_x, train_y)
t_fit = time.perf_counter() - t0
pred = model.predict(test_x)
print("[6] fold 0: fit %.2f s | test weighted F1 = %.4f | train weighted F1 = %.4f"
      % (t_fit, f1_score(test_y, pred, average='weighted'),
         f1_score(train_y, model.predict(train_x), average='weighted')))

# pick first size-2 and first size-3 itemset
sel = {}
for k, s in candidate_itemsets.items():
    if len(s) == 2 and 2 not in [len(v) for v in sel.values()]:
        sel[k] = s
    if len(s) == 3 and 3 not in [len(v) for v in sel.values()]:
        sel[k] = s
print("[7] itemsets used:", sel)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    try:
        t0 = time.perf_counter()
        res = dm.itemset_permutation_importance(model, train_x, train_y, train_list[i], sel,
                                                constrain=True, n_repeats=1)
        t_pi = time.perf_counter() - t0
        print("    itemset_permutation_importance (2 itemsets x 1 repeat) in %.1f s -> %.1f s per (itemset,repeat)"
              % (t_pi, t_pi / 2))
        print("    result:\n", res.to_string())
        print("    extrapolation: 5 folds x 10 itemsets x 10 repeats = 500 iterations -> ~%.0f min"
              % (500 * t_pi / 2 / 60))
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    kinds = {}
    for x in w:
        kinds.setdefault(type(x.message).__name__, set()).add(str(x.message)[:160])
    print("    warnings during permutation:", {k: len(v) for k, v in kinds.items()})
    for k, v in kinds.items():
        for msg in list(v)[:2]:
            print("      -", k, ":", msg)

print("\nDONE")
