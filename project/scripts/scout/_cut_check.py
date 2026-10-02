import pandas as pd
base = "repos/PermutationLocationImportance/datasets/"
c,a,l = "case:concept:name","concept:name","label"
codes = ["AC379414","AC378619","AC356134","AC376480","AC356133"]
dfs = {k: pd.read_csv(base+f, low_memory=False) for k,f in
       [("f1","BPIC11_f1_trunc36.csv"),("f2","BPIC11_f2_trunc40.csv"),("f3","BPIC11_f3_trunc31.csv"),("f4","BPIC11_f4_trunc40.csv")]}
for k,df in dfs.items():
    print("=====",k)
    lab = df.groupby(c)[l].first()
    for code in codes:
        has = (df[df[a]==code].groupby(c).size().reindex(lab.index, fill_value=0)>0)
        print("  %s present in %d cases | among label1: %.3f | among label0: %.3f" % (code, has.sum(), has[lab==1].mean(), has[lab==0].mean()))
# compare per-case lengths f1 vs f2 (f2 is uncut, trunc40)
len2 = dfs["f2"].groupby(c).size()
for k in ["f1","f3","f4"]:
    df = dfs[k]; lab = df.groupby(c)[l].first(); lenk = df.groupby(c).size()
    common = lenk.index.intersection(len2.index)
    shorter = (lenk[common] < len2[common].clip(upper=lenk.max()))  # cut below this file's trunc length
    print("=====", k, "cases:", len(lenk), "common with f2:", len(common))
    print("  cases shorter than their f2 counterpart (i.e. likely cut):", shorter.sum())
    print("  label of likely-cut cases: 1 -> %d, 0 -> %d" % ((lab[common][shorter]==1).sum(), (lab[common][shorter]==0).sum()))
    print("  label of not-cut cases:    1 -> %d, 0 -> %d" % ((lab[common][~shorter]==1).sum(), (lab[common][~shorter]==0).sum()))
    # case ids present in f2 but missing here
    print("  cases in f2 missing here:", len(len2.index.difference(lenk.index)))
    print("  length distribution by label (min/median/max): label1", lenk[lab==1].min(), lenk[lab==1].median(), lenk[lab==1].max(), "| label0", lenk[lab==0].min(), lenk[lab==0].median(), lenk[lab==0].max())
