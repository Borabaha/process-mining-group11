import warnings; warnings.filterwarnings("ignore")
import sys, re, numpy as np, pandas as pd
from sklearn.metrics import f1_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
# slide pages
txt=open("C:/Users/Bohabara/AppData/Local/Temp/claude/C--Users-Bohabara-Desktop-process-mining/05db42ec-3109-4098-847a-cf389d8c070a/scratchpad/txt/week1_introduction_bpm_process_mining.txt",encoding="utf-8",errors="replace").read()
pages=txt.split("\f")
print("n pages", len(pages))
for i,p in enumerate(pages):
    for key in ["Does this model (still) reflect","Is a process model available","What will be the outcome for this case","event stream"]:
        if key in p: print("page",i+1,"contains:",key)
for i,p in enumerate(pages):
    if "What will be the outcome" in p:
        print("---- page", i+1, "----"); print(p[:1500])
    if "Is a process model available" in p:
        print("---- page", i+1, "----"); print(p[:1800])
# docstrings
d=mutual_info_classif.__doc__
for key in ["independent","dependency","natural log","nats"]:
    i=d.find(key); print("MI doc:",key,"->", d[max(0,i-200):i+120].replace("\n"," ") if i>=0 else "NOT FOUND")
d=f1_score.__doc__
for key in ["harmonic","F1 ="," F-score", "2 * TP"]:
    i=d.find(key); print("F1 doc:",key,"->", d[max(0,i-120):i+200].replace("\n"," ") if i>=0 else "NOT FOUND")
# PFI numeric check: importances_mean == baseline - mean(permuted scores)
rng=np.random.RandomState(0); X=rng.rand(200,3); y=(X[:,0]+0.1*rng.rand(200)>0.5).astype(int)
clf=LogisticRegression().fit(X,y); base=clf.score(X,y)
r=permutation_importance(clf,X,y,n_repeats=5,random_state=0)
print("PFI check: baseline",base,"importances_mean",r.importances_mean, "baseline-mean(importances)?", np.allclose(r.importances_mean, base-(base-r.importances).mean(axis=1)))
