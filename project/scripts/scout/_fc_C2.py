import sys, warnings, json; warnings.filterwarnings("ignore")
REPO = "C:/Users/Bohabara/AppData/Local/Temp/claude/C--Users-Bohabara-Desktop-process-mining/05db42ec-3109-4098-847a-cf389d8c070a/scratchpad/repos/PermutationLocationImportance"
sys.path.insert(0, REPO)
import numpy as np, pandas as pd
from tools import DataManager
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
# OL example from Table 1
t1 = {1:"RG LA LE CR ER ST IL IA NC ER CR LE CR",2:"RG ER ST IL LE CR IA LA IC DI",3:"RG LA ER ST LE CR IA NC CR LE CR LE CR",
      4:"RG LE LE LA ER ST CR IA IC LE LA NC DI",5:"RG ER ST LE CR IL IA LA NC CR LE",6:"RG LA CR ER ST LE IL IA NC DI"}
dmy = DataManager.__new__(DataManager); ol=[]
for cid, s in t1.items():
    tr = s.split()
    for o in dmy.find_itemset_indexes(tr, {"ER","CR"}):
        ol.append(([i+1 for i in o if tr[i]=="ER"][0], [i+1 for i in o if tr[i]=="CR"][0]))
print("OL(ER,CR) recomputed:", ol)
print("LA positions:", {c:[i+1 for i,a in enumerate(s.split()) if a=="LA"] for c,s in t1.items()})
# permuted example check
s1 = "RG LA LE CR ER ST IL IA NC ER CR LE CR".split(); s1p = "RG LA CR ER ER LE CR ST IL IA NC LE CR".split()
print("sigma1' positions: CR", [i+1 for i,a in enumerate(s1p) if a=="CR"], "ER", [i+1 for i,a in enumerate(s1p) if a=="ER"], "len equal", len(s1)==len(s1p), "multiset equal", sorted(s1)==sorted(s1p))
# xgboost effective config + train-fold baseline
dm = DataManager(REPO+"/datasets/BPIC11_f1_trunc36.csv", 2, None, L_max_perc=0.8)
enc = dm.index_encoding(dm.data)
for c in enc.columns:
    if c not in (dm.case_id, dm.outcome): enc[c] = enc[c].astype(int)
tr, te = dm.cross_split_test_train(5)
X = enc[enc[dm.case_id].isin(tr[0])]; y = X[dm.outcome].tolist(); X = X.drop([dm.case_id, dm.outcome], axis=1)
Xt = enc[enc[dm.case_id].isin(te[0])]; yt = Xt[dm.outcome].tolist(); Xt = Xt.drop([dm.case_id, dm.outcome], axis=1)
m = XGBClassifier(); m.fit(X, y)
print("train-fold baseline weighted F1:", round(f1_score(y, m.predict(X), average='weighted'),4), "| test F1:", round(f1_score(yt, m.predict(Xt), average='weighted'),4))
cfg = json.loads(m.get_booster().save_config())
tp = cfg["learner"]["gradient_booster"]["tree_train_param"] if "tree_train_param" in cfg["learner"]["gradient_booster"] else cfg["learner"]["gradient_booster"]["updater"]
print("n_estimators (trees):", m.get_booster().num_boosted_rounds(), "| objective:", cfg["learner"]["objective"]["name"] if "name" in cfg["learner"]["objective"] else cfg["learner"]["learner_train_param"]["objective"])
ttp = cfg["learner"]["gradient_booster"]["tree_train_param"]
print({k: ttp[k] for k in ("eta","max_depth","subsample","colsample_bytree","min_child_weight") if k in ttp})
print("get_params n_estimators/max_depth/learning_rate:", m.get_params()["n_estimators"], m.get_params()["max_depth"], m.get_params()["learning_rate"])
