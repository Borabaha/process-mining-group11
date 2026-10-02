import pandas as pd, numpy as np, sys
base="repos/PermutationLocationImportance/datasets/"
for f in ["BPIC11_f1_trunc36.csv","BPIC11_f2_trunc40.csv","BPIC11_f3_trunc31.csv","BPIC11_f4_trunc40.csv"]:
    df=pd.read_csv(base+f)
    df["t"]=pd.to_datetime(df["time:timestamp"], format="mixed")
    df=df.sort_values(["case:concept:name","event_nr"])
    same=(df.groupby("case:concept:name")["t"].diff()==pd.Timedelta(0))
    cons=df.groupby("case:concept:name")["t"].diff().notna()
    raw_cases=df["case:concept:name"].nunique()
    # 2024 filter
    a=df["concept:name"].str.lower().str.replace(" ","").str.replace("-","").str.replace("_","")
    df["a"]=a
    cnt=df.groupby("a")["case:concept:name"].count()
    low=cnt[cnt<2].index
    rm=df.loc[df["a"].isin(low),"case:concept:name"].unique()
    d2=df[~df["case:concept:name"].isin(rm)]
    print(f, "cases",raw_cases,"after2024filter",d2["case:concept:name"].nunique(),"acts",df["a"].nunique(),"->",d2["a"].nunique(),
          "same-ts share %.3f"%(same.sum()/cons.sum()), "label mean %.3f"%d2.groupby("case:concept:name")["label"].first().astype(float).mean())
    if "f1" in f:
        const=[c for c in df.columns if df.groupby("case:concept:name")[c].nunique().max()==1]
        print(" constant-per-case cols:",const)
        print(" sample acts:", sorted(df["a"].unique())[:8], "...")
