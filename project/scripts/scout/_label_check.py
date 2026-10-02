import pandas as pd, re, sys
base = "repos/PermutationLocationImportance/datasets/"
files = ["BPIC11_f1_trunc36.csv","BPIC11_f2_trunc40.csv","BPIC11_f3_trunc31.csv","BPIC11_f4_trunc40.csv"]
pats = {
 "ca-19.9": r"ca-19\.9|ca 19\.9|ca-199",
 "ca-125": r"ca-125|ca 125",
 "cea": r"\bcea\b",
 "squamous": r"squamous|plaveisel",
 "histolog": r"histolog",
 "tumor marker": r"tumor ?marker|tumormarker",
}
for f in files:
    df = pd.read_csv(base+f, low_memory=False)
    c, a, l = "case:concept:name", "concept:name", "label"
    print("=====", f)
    print("rows", len(df), "cases", df[c].nunique(), "activities", df[a].nunique())
    print("label values", df[l].unique())
    per_case = df.groupby(c)[l].agg(["nunique","first"])
    print("cases with >1 label value:", (per_case["nunique"]>1).sum())
    lab = per_case["first"]
    print("label counts (cases):", lab.value_counts().to_dict(), "ratio of 1 = %.4f" % (lab==1).mean())
    ln = df.groupby(c).size()
    print("prefix/trace length min/median/max:", ln.min(), ln.median(), ln.max())
    print("event_nr max:", df["event_nr"].max())
    acts = pd.Series(df[a].unique())
    for k,p in pats.items():
        hit = acts[acts.str.contains(p, case=False, regex=True)]
        print("  activities matching", k, ":", list(hit)[:10])
    # per-label: fraction of cases containing matching activities
    for k,p in pats.items():
        m = df[a].str.contains(p, case=False, regex=True)
        has = df[m].groupby(c).size().reindex(per_case.index, fill_value=0) > 0
        print("  contains[%s]: label1 %.3f (n=%d) | label0 %.3f (n=%d)" % (k, has[lab==1].mean(), (lab==1).sum(), has[lab==0].mean(), (lab==0).sum()))
    # columns
    print("columns:", list(df.columns))
