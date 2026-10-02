import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import f1_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
d=mutual_info_classif.__doc__
for key in ["independent","dependency","natural log","nats"]:
    i=d.find(key); print("MI doc:",key,"->", d[max(0,i-200):i+120].replace("\n"," ") if i>=0 else "NOT FOUND")
d=f1_score.__doc__
for key in ["harmonic","F1 ="," F-score", "2 * TP"]:
    i=d.find(key); print("F1 doc:",key,"->", d[max(0,i-120):i+200].replace("\n"," ") if i>=0 else "NOT FOUND")
rng=np.random.RandomState(0); X=rng.rand(200,3); y=(X[:,0]+0.1*rng.rand(200)>0.5).astype(int)
clf=LogisticRegression().fit(X,y); base=clf.score(X,y)
r=permutation_importance(clf,X,y,n_repeats=5,random_state=0)
print("PFI check: baseline",base,"importances_mean",r.importances_mean, "equals baseline-mean(permuted score)?", np.allclose(r.importances_mean, base-(base-r.importances).mean(axis=1)))
