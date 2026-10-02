"""Smoke test for the 2024 repo (PermutationLocationImportance) - no full pipeline."""
import os
import sys
import time
import traceback
import warnings

REPO = ("C:/Users/Bohabara/AppData/Local/Temp/claude/C--Users-Bohabara-Desktop-process-mining/"
        "05db42ec-3109-4098-847a-cf389d8c070a/scratchpad/repos/PermutationLocationImportance")
os.chdir(REPO)
sys.path.insert(0, REPO)

import numpy as np
import pandas as pd
import sklearn
import xgboost
import mlxtend

print("python", sys.version.split()[0])
print("numpy", np.__version__, "| pandas", pd.__version__, "| sklearn", sklearn.__version__,
      "| xgboost", xgboost.__version__, "| mlxtend", mlxtend.__version__)

from tools import DataManager  # noqa: E402

# --- 1. DataManager (same args as CrossValidation_ProcessPermutation.py:38-46) ---
t0 = time.perf_counter()
dm = DataManager('datasets/BPIC11_f1_trunc36.csv', 2, None, L_max_perc=0.8)
t_load = time.perf_counter() - t0
print("\n[1] DataManager loaded in %.2f s" % t_load)
print("    data shape:", dm.data.shape)
print("    #cases:", dm.data[dm.case_id].nunique())
print("    #distinct activities:", dm.data[dm.activity].nunique())
print("    max event_nr:", int(dm.data['event_nr'].max()))
print("    L_max (quantile %.2f of event_nr):" % dm.L_max_perc, dm.L_max)
print("    label dtype:", dm.data[dm.outcome].dtype, "| value counts (per event row):",
      dm.data[dm.outcome].value_counts().to_dict())
case_labels = dm.data.drop_duplicates(subset=[dm.case_id])[dm.outcome].value_counts().to_dict()
print("    label value counts (per case):", case_labels)

# --- 2. frequent_activity_sets(0.5, 10) ---
t0 = time.perf_counter()
cand, itemsets = dm.frequent_activity_sets(0.5, 10)
t_fas = time.perf_counter() - t0
print("\n[2] frequent_activity_sets(min_support=0.5, top_k=10) in %.2f s" % t_fas)
print("    %d itemsets returned" % len(cand))
for i, row in itemsets.iterrows():
    print("    support=%.3f size=%d itemset=%s" % (row['support'], row['item_size'], sorted(row['itemsets'])))

# 2b. the top_k-as-float bug (argparse default type=float -> 10.0)
print("\n[2b] frequent_activity_sets(0.5, 10.0) (float top_k as argparse would pass):")
try:
    cand_f, _ = dm.frequent_activity_sets(0.5, 10.0)
    print("    OK, %d itemsets (no error with float top_k)" % len(cand_f))
except Exception as e:  # noqa: BLE001
    print("    FAILED:", type(e).__name__, str(e)[:300])

# --- 3. index_encoding ---
t0 = time.perf_counter()
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    enc = dm.index_encoding(dm.data)
    n_warn = len(w)
    warn_types = sorted({type(x.message).__name__ for x in w})
t_enc = time.perf_counter() - t0
print("\n[3] index_encoding(dm.data) in %.2f s" % t_enc)
print("    encoded shape:", enc.shape, "| warnings:", n_warn, warn_types)
print("    first 5 columns:", enc.columns[:5].tolist())
print("    dtypes summary:", enc.dtypes.value_counts().to_dict())

# --- 4. XGBClassifier fit ---
from xgboost import XGBClassifier  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

X = enc.drop([dm.case_id, dm.outcome], axis=1)
y = enc[dm.outcome].tolist()
print("\n[4] X shape:", X.shape, "| y len:", len(y), "| y unique:", sorted(set(y)))
print("    X dtypes:", X.dtypes.value_counts().to_dict())
try:
    t0 = time.perf_counter()
    model = XGBClassifier()
    model.fit(X, y)
    t_fit = time.perf_counter() - t0
    pred = model.predict(X)
    print("    XGBClassifier() fit in %.2f s | train weighted F1 = %.4f" % (t_fit, f1_score(y, pred, average='weighted')))
except Exception:  # noqa: BLE001
    print("    FIT FAILED:")
    traceback.print_exc()
    # retry with explicit int labels / bool->int cast
    try:
        t0 = time.perf_counter()
        model = XGBClassifier()
        model.fit(X.astype(int), np.asarray(y).astype(int))
        t_fit = time.perf_counter() - t0
        print("    retry with astype(int): fit in %.2f s" % t_fit)
    except Exception:  # noqa: BLE001
        traceback.print_exc()

# --- 5. cross_split_test_train (cheap) ---
try:
    tr, te = dm.cross_split_test_train(5)
    print("\n[5] cross_split_test_train(5): folds=%d, train sizes=%s, test sizes=%s"
          % (len(tr), [len(x) for x in tr], [len(x) for x in te]))
except Exception:  # noqa: BLE001
    traceback.print_exc()

print("\nDONE")
