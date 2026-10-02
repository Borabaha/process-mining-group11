import pandas as pd, numpy as np
base = "repos/PermutationLocationImportance/datasets/"
c,a,l = "case:concept:name","concept:name","label"
f2 = pd.read_csv(base+"BPIC11_f2_trunc40.csv", low_memory=False).sort_values([c,"event_nr"])
def firstpos(code):
    s = f2[f2[a]==code].groupby(c)["event_nr"].min()
    return s
def lastpos(code):
    return f2[f2[a]==code].groupby(c)["event_nr"].max()
allcases = f2[c].drop_duplicates()
def label_of(fname):
    d = pd.read_csv(base+fname, low_memory=False); return d.groupby(c)[l].first()
# f1: satisfied iff AC379414 or 378619A occurs (within f2's 40-event window)
l1 = label_of("BPIC11_f1_trunc36.csv")
sat1 = allcases.isin(firstpos("AC379414").index) | allcases.isin(firstpos("378619A").index)
sat1.index = allcases.values
print("f1: rows=rule satisfied within first 40 events of f2 (True/False), cols=local label")
print(pd.crosstab(sat1.reindex(l1.index), l1))
# f4: satisfied iff AC356133 occurs
l4 = label_of("BPIC11_f4_trunc40.csv")
sat4 = allcases.isin(firstpos("AC356133").index); sat4.index = allcases.values
print("\nf4: rows=AC356133 occurs within first 40 events of f2, cols=local label")
print(pd.crosstab(sat4.reindex(l4.index), l4))
# f3: (not biopsies) U squamous: satisfied iff squamous occurs and no biopsies before it
l3 = label_of("BPIC11_f3_trunc31.csv")
fb = firstpos("AC356134"); fs = firstpos("376480A")
cat = []
for cid in l3.index:
    b = fb.get(cid, np.inf); s = fs.get(cid, np.inf)
    if s < b: cat.append("squamous first (satisfied)")
    elif b < s: cat.append("biopsies-nno first (violated)")
    else: cat.append("neither in first 40 (unknown)")
print("\nf3: rows=which cut-activity comes first in f2, cols=local label")
print(pd.crosstab(pd.Series(cat, index=l3.index), l3))
# f2: G(CEA -> F squamous); candidate CEA code 376400
l2 = label_of("BPIC11_f2_trunc40.csv")
lc = lastpos("376400"); ls = lastpos("376480A")
cat2 = []
for cid in l2.index:
    if cid not in lc.index: cat2.append("no 376400 in first 40 (vacuously satisfied)")
    elif cid in ls.index and ls[cid] > lc[cid]: cat2.append("376400 then later 376480A (satisfied)")
    else: cat2.append("376400 not followed by 376480A within 40 (violated)")
print("\nf2 (using candidate CEA code 376400): rows=rule status within first 40 events, cols=local label")
print(pd.crosstab(pd.Series(cat2, index=l2.index), l2))
