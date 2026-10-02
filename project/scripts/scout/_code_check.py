import pandas as pd, re
base = "repos/PermutationLocationImportance/datasets/"
c,a,l = "case:concept:name","concept:name","label"
dfs = {k: pd.read_csv(base+f, low_memory=False) for k,f in
       [("f1","BPIC11_f1_trunc36.csv"),("f2","BPIC11_f2_trunc40.csv"),("f3","BPIC11_f3_trunc31.csv"),("f4","BPIC11_f4_trunc40.csv")]}
for k,df in dfs.items():
    vals = pd.Series(df[a].unique())
    odd = vals[~vals.str.match(r"^AC\d+$")]
    print(k, "non-'AC<digits>' activity values:", list(odd))
    for cand in ["378619A","376480A","AC378619A","AC376480A"]:
        print("   ", cand, "in", (df[df[a]==cand][c].nunique()), "cases")
    print("   'other' in concept:name:", (df[a]=="other").sum(), "| 'missing' in any column:", (df.astype(str)=="missing").any().sum(), "cols")
# f2: find codes most prevalent in label-0 cases (candidate CEA code)
df = dfs["f2"]; lab = df.groupby(c)[l].first()
pres = df.groupby([c,a]).size().unstack(fill_value=0) > 0
p1 = pres[lab==1].mean(); p0 = pres[lab==0].mean()
tab = pd.DataFrame({"in_label0":p0, "in_label1":p1}); tab["diff"] = tab.in_label0 - tab.in_label1
print("\n=== f2: activity codes most over-represented in label-0 cases (n0=%d, n1=%d) ===" % ((lab==0).sum(), (lab==1).sum()))
print(tab.sort_values("diff", ascending=False).head(8).round(3).to_string())
print("\n=== f2: activity codes most over-represented in label-1 cases ===")
print(tab.sort_values("diff").head(5).round(3).to_string())
# f3 missing cases: first event in f2
miss = dfs["f2"][c].drop_duplicates()
miss = miss[~miss.isin(dfs["f3"][c].unique())]
first = dfs["f2"][dfs["f2"][c].isin(miss)].sort_values([c,"event_nr"]).groupby(c)[a].first()
print("\n=== 19 cases missing from f3: first activity in f2 ===", first.value_counts().to_dict())
# f1: first events of f2-cases vs cut: how many cases in f2 have first event AC379414 (would vanish in f1)?
print("f2 cases whose first event is AC379414:", (dfs["f2"].sort_values([c,"event_nr"]).groupby(c)[a].first()=="AC379414").sum())
