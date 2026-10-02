"""Smoke test for the 2023 repo (InteractivePatternDetection / IMPresseD) - imports only, no GUI."""
import os
import sys
import time
import traceback

REPO = ("C:/Users/Bohabara/AppData/Local/Temp/claude/C--Users-Bohabara-Desktop-process-mining/"
        "05db42ec-3109-4098-847a-cf389d8c070a/scratchpad/repos/InteractivePatternDetection")
os.chdir(REPO)
sys.path.insert(0, REPO)

print("python", sys.version.split()[0])

for name in ["pm4py", "paretoset", "networkx", "scipy", "sklearn", "matplotlib"]:
    t0 = time.perf_counter()
    try:
        mod = __import__(name)
        print("import %-11s OK  v=%s  (%.2f s)" % (name, getattr(mod, "__version__", "?"), time.perf_counter() - t0))
    except Exception as e:  # noqa: BLE001
        print("import %-11s FAILED: %s: %s" % (name, type(e).__name__, e))

print()
t0 = time.perf_counter()
try:
    import IMIPD  # noqa: F401
    print("import IMIPD      OK (%.2f s)" % (time.perf_counter() - t0))
    print("   IMIPD public names:", [n for n in dir(IMIPD) if not n.startswith('_') and callable(getattr(IMIPD, n))
                                     and getattr(getattr(IMIPD, n), '__module__', '') == 'IMIPD'])
except Exception:  # noqa: BLE001
    print("import IMIPD      FAILED:")
    traceback.print_exc()

t0 = time.perf_counter()
try:
    import Auto_IMPID  # noqa: F401
    print("import Auto_IMPID OK (%.2f s)" % (time.perf_counter() - t0))
    print("   has AutoStepWise_PPD:", hasattr(Auto_IMPID, "AutoStepWise_PPD"))
except Exception:  # noqa: BLE001
    print("import Auto_IMPID FAILED:")
    traceback.print_exc()

# tiny functional checks of the two "exotic" deps
try:
    import numpy as np
    import pandas as pd
    from paretoset import paretoset
    df = pd.DataFrame({"a": [1, 2, 3, 1], "b": [3, 2, 1, 1]})
    mask = paretoset(df, sense=["min", "min"])
    print("\nparetoset() tiny check -> mask:", list(mask))
except Exception:  # noqa: BLE001
    traceback.print_exc()

try:
    import pm4py
    df = pd.DataFrame({"case:concept:name": ["c1", "c1", "c2"],
                       "concept:name": ["A", "B", "A"],
                       "time:timestamp": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])})
    df = pm4py.format_dataframe(df, case_id="case:concept:name", activity_key="concept:name",
                                timestamp_key="time:timestamp")
    log = pm4py.convert_to_event_log(df)
    print("pm4py tiny check -> EventLog with %d traces" % len(log))
except Exception:  # noqa: BLE001
    traceback.print_exc()

print("\nDONE")
