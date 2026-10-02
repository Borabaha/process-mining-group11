"""Smoke test 3 for the 2024 repo: same as smoke_2024b but with features cast to int (pandas>=2 get_dummies->bool fix)."""
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
T_START = time.perf_counter()


class DataManagerIntFeatures(DataManager):
    """Workaround: pandas>=2.0 pd.get_dummies returns bool; cast all feature columns to int (pandas 1.x behaviour)."""

    def index_encoding(self, data):
        enc = super().index_encoding(data)
        feat = [c for c in enc.columns if c not in (self.case_id, self.outcome)]
        enc[feat] = enc[feat].astype(int)
        return enc


dm = DataManagerIntFeatures('datasets/BPIC11_f1_trunc36.csv', 2, None, L_max_perc=0.8)
cand, itemsets = dm.frequent_activity_sets(0.5, 10)
candidate_itemsets = {i: s for i, s in enumerate(cand)}
t0 = time.perf_counter()
enc = dm.index_encoding(dm.data)
print("[3'] index_encoding + int cast: %.2f s, shape %s, dtypes %s" % (time.perf_counter() - t0, enc.shape,
                                                                     enc.dtypes.value_counts().to_dict()))

np.random.seed(0)  # only affects nothing in the repo's split (StratifiedKFold has no random_state) - noted in report
train_list, test_list = dm.cross_split_test_train(5)
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
print("[6'] fold 0 (%d train / %d test cases): fit %.2f s | test weighted F1 = %.4f"
      % (len(train_y), len(test_y), t_fit, f1_score(test_y, model.predict(test_x), average='weighted')))

sel = {0: candidate_itemsets[0], 3: candidate_itemsets[3]}
print("[7'] itemsets used:", sel)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    try:
        t0 = time.perf_counter()
        res = dm.itemset_permutation_importance(model, train_x, train_y, train_list[i], sel,
                                                constrain=True, n_repeats=1)
        t_pi = time.perf_counter() - t0
        print("     itemset_permutation_importance (2 itemsets x 1 repeat) in %.1f s -> %.1f s per (itemset,repeat)"
              % (t_pi, t_pi / 2))
        print("     result:\n", res.to_string())
        print("     extrapolation for the repo defaults: 5 folds x 10 itemsets x 10 repeats = 500 iterations -> ~%.0f min"
              % (500 * t_pi / 2 / 60))
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    kinds = {}
    for x in w:
        kinds.setdefault(type(x.message).__name__, set()).add(str(x.message)[:160])
    print("     warnings during permutation:", {k: len(v) for k, v in kinds.items()})
    for k, v in kinds.items():
        for msg in list(v)[:2]:
            print("       -", k, ":", msg)

print("\nTOTAL script time %.1f s" % (time.perf_counter() - T_START))
print("DONE")
