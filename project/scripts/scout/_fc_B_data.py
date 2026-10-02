import warnings; warnings.filterwarnings("ignore")
import sys, time
sys.path.insert(0, "C:/Users/Bohabara/AppData/Local/Temp/claude/C--Users-Bohabara-Desktop-process-mining/05db42ec-3109-4098-847a-cf389d8c070a/scratchpad/repos/PermutationLocationImportance")
import pandas as pd, numpy as np
from tools import DataManager
base="C:/Users/Bohabara/AppData/Local/Temp/claude/C--Users-Bohabara-Desktop-process-mining/05db42ec-3109-4098-847a-cf389d8c070a/scratchpad/repos/PermutationLocationImportance/datasets/"
for f in ["BPIC11_f1_trunc36.csv","BPIC11_f2_trunc40.csv","BPIC11_f3_trunc31.csv"]:
    raw=pd.read_csv(base+f)
    raw=raw.sort_values(["case:concept:name","event_nr"])
    ts=raw["time:timestamp"]
    same_prev=(raw.groupby("case:concept:name")["time:timestamp"].shift(1)==ts)
    pairs=raw.groupby("case:concept:name").size().sub(1).clip(lower=0).sum()
    print(f, "sample ts:", ts.iloc[:3].tolist(), "consecutive pairs sharing timestamp:", round(same_prev.sum()/pairs,3))
    dm=DataManager(base+f,2,None,L_max_perc=0.8)
    acts, its = dm.frequent_activity_sets(0.5,10)
    print("  kept cases", dm.data["case:concept:name"].nunique(), "activities", dm.data["concept:name"].nunique(), "itemsets(size>1,sup>=0.5):", len(acts))
    print("  ", [(sorted(list(s)), round(sup,4)) for s,sup in zip(its.itemsets, its.support)])
    print("  Allowed_locations sample ac370000:", dm.Allowed_locations.get("ac370000", [])[:5], "...")
