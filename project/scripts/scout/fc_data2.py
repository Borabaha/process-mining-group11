import pandas as pd
base="repos/PermutationLocationImportance/datasets/"
for f in ["BPIC11_f1_trunc36.csv","BPIC11_f2_trunc40.csv","BPIC11_f3_trunc31.csv","BPIC11_f4_trunc40.csv"]:
    df=pd.read_csv(base+f)
    df["t"]=pd.to_datetime(df["time:timestamp"], format="mixed")
    df["case:concept:name"]=df["case:concept:name"].astype(str)
    for order in (["case:concept:name","event_nr"],None):
        d=df.sort_values(order) if order else df
        dif=d.groupby("case:concept:name")["t"].diff()
        print(f, "sorted" if order else "fileorder", "same/consecutive %.3f"%((dif==pd.Timedelta(0)).sum()/dif.notna().sum()),
              "same/all events %.3f"%((dif==pd.Timedelta(0)).sum()/len(d)), "neg diffs",(dif<pd.Timedelta(0)).sum())
