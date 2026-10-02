import warnings, inspect, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import sklearn, scipy, xgboost
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import spearmanr, kendalltau
print("versions", sklearn.__version__, scipy.__version__, xgboost.__version__, pd.__version__)
# docstring snippets
def snip(doc, key, w=250):
    i = doc.find(key); print("  ...", doc[max(0,i-60):i+w].replace("\n"," ") if i>=0 else "NOT FOUND: "+key)
print("StratifiedKFold:"); snip(StratifiedKFold.__doc__, "preserv")
print("f1_score weighted:"); snip(f1_score.__doc__, "'weighted'")
print("f1_score formula:"); snip(f1_score.__doc__, "F1 =", 200)
print("permutation_importance:"); snip(permutation_importance.__doc__, "difference between")
snip(permutation_importance.__doc__, "importances_mean")
print("mutual_info_classif:"); snip(mutual_info_classif.__doc__, "equal to zero")
print("spearmanr:"); snip(spearmanr.__doc__, "monotonic")
print("kendalltau:"); snip(kendalltau.__doc__, "tau-a"); snip(kendalltau.__doc__, "variant")
# metrics example
y=[0]*8+[1]*2; p=[0]*10
print("example acc",accuracy_score(y,p),"f1 weighted",f1_score(y,p,average='weighted'),"f1 pos",f1_score(y,p,zero_division=0))
# spearman/kendall examples
print("spearman (1,2,3)vs(1,3,2)", spearmanr([1,2,3],[1,3,2])[0], "kendall", kendalltau([1,2,3],[1,3,2])[0])
# MI example
X=np.array([[1],[1],[0],[0]]); y=np.array([1,1,0,0])
print("MI counts 1,1,0,0:", mutual_info_classif(X,y,discrete_features=True), "ln2=",np.log(2))
print("MI counts 1,0,1,0:", mutual_info_classif(np.array([[1],[0],[1],[0]]),y,discrete_features=True))
# xgboost defaults
from xgboost import XGBClassifier
m=XGBClassifier(); m.fit(np.random.RandomState(0).rand(50,4), np.random.RandomState(1).randint(0,2,50))
cfg=json.loads(m.get_booster().save_config())
tp=cfg["learner"]["gradient_booster"]["tree_train_param"] if "tree_train_param" in cfg["learner"]["gradient_booster"] else cfg["learner"]["gradient_booster"]
print("objective", cfg["learner"]["objective"]["name"], "booster", cfg["learner"]["gradient_booster"]["name"])
print("n_estimators", m.get_booster().num_boosted_rounds(), "params", m.get_params()["n_estimators"], m.get_params()["max_depth"], m.get_params()["learning_rate"], m.get_params()["subsample"], m.get_params()["tree_method"])
def find(d, keys, path=""):
    if isinstance(d, dict):
        for k,v in d.items():
            if k in keys: print("   ",path+"/"+k, "=", v)
            find(v, keys, path+"/"+k)
find(cfg, {"max_depth","eta","learning_rate","subsample","colsample_bytree","tree_method","num_parallel_tree"})
