import sys, warnings; warnings.filterwarnings("ignore")
REPO = "C:/Users/Bohabara/AppData/Local/Temp/claude/C--Users-Bohabara-Desktop-process-mining/05db42ec-3109-4098-847a-cf389d8c070a/scratchpad/repos/PermutationLocationImportance"
sys.path.insert(0, REPO)
import pandas as pd
from tools import DataManager
dm = DataManager(REPO + "/datasets/bpic2012_1_trunc40.csv", 2, None, L_max_perc=0.8)
d = dm.data
print("bpic2012_1 cases:", d[dm.case_id].nunique(), "| label per case:", d.drop_duplicates(dm.case_id)['label'].value_counts().to_dict())
act = "wnabellenoffertes"
print("activity present:", act in set(d[dm.activity]))
ev = d[d[dm.activity] == act]
print("events of", act, ":", len(ev), "| label-0 share per EVENT:", round((ev['label']==0).mean(),3))
cases = ev[dm.case_id].unique()
cl = d[d[dm.case_id].isin(cases)].drop_duplicates(dm.case_id)
print("cases containing it:", len(cases), "| label-0 share per CASE:", round((cl['label']==0).mean(),3))
pos = ev.groupby('event_nr')['label'].apply(lambda s: (s==0).mean())
print("label-0 share by position: min %.2f at pos %d, max %.2f at pos %d; positions %d..%d" % (pos.min(), pos.idxmin(), pos.max(), pos.idxmax(), pos.index.min(), pos.index.max()))
print("sorted pivot columns would be:", sorted(d['label'].unique()))
