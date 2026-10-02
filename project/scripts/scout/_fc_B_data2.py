import warnings; warnings.filterwarnings("ignore")
import sys, time
sys.path.insert(0, "C:/Users/Bohabara/AppData/Local/Temp/claude/C--Users-Bohabara-Desktop-process-mining/05db42ec-3109-4098-847a-cf389d8c070a/scratchpad/repos/PermutationLocationImportance")
import pandas as pd, numpy as np
from tools import DataManager
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier
base="C:/Users/Bohabara/AppData/Local/Temp/claude/C--Users-Bohabara-Desktop-process-mining/05db42ec-3109-4098-847a-cf389d8c070a/scratchpad/repos/PermutationLocationImportance/datasets/"
# IG of 376400 on f1 (raw file, whole log; counts per case and presence per case)
raw=pd.read_csv(base+"BPIC11_f1_trunc36.csv")
raw["act"]=raw["concept:name"].str.lower()
cnt=raw.groupby("case:concept:name")["act"].apply(lambda s:(s=="376400").sum())
lab=raw.groupby("case:concept:name")["label"].first()
print("f1 raw: cases",len(cnt),"376400 present in",(cnt>0).sum(),"cases; MI(count)",mutual_info_classif(cnt.values.reshape(-1,1),lab.values,discrete_features=True)[0],"MI(presence)",mutual_info_classif((cnt>0).astype(int).values.reshape(-1,1),lab.values,discrete_features=True)[0])
print("  label share among cases with 376400:", lab[cnt>0].mean(), "without:", lab[cnt==0].mean())
# top-5 MI activities on f1 (count encoding, whole raw log)
piv=raw.pivot_table(index="case:concept:name",columns="act",values="event_nr",aggfunc="count",fill_value=0)
mi=mutual_info_classif(piv.values,lab.loc[piv.index].values,discrete_features=True)
top=sorted(zip(mi,piv.columns),reverse=True)[:5]; print("  top-5 MI (counts):",[(a,round(m,3)) for m,a in top])
# xgboost determinism on f1 encoded data (DataManager-filtered)
dm=DataManager(base+"BPIC11_f1_trunc36.csv",2,None,L_max_perc=0.8)
enc=dm.index_encoding(dm.data)
X=enc.drop(["case:concept:name","label"],axis=1).astype(int); y=enc["label"].astype(int).tolist()
t=time.time(); m1=XGBClassifier().fit(X,y); m2=XGBClassifier().fit(X,y)
p1=m1.predict_proba(X)[:,1]; p2=m2.predict_proba(X)[:,1]
print("xgb two fits identical probs:", np.array_equal(p1,p2), "max abs diff", np.abs(p1-p2).max(), "time",round(time.time()-t,1), "n_features", X.shape[1])
