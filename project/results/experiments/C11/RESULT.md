# Experiment C11 — paretoset 1.2.0 vs 1.2.5

The full write-up (what was run, tables, recommendations) is in `../C12/RESULT.md`, section 2, together with
experiment C12.

Command (from the project root, about 30 s):

```
.venv/Scripts/python.exe -m pip install paretoset==1.2.0 --target results/experiments/C11/tmp/paretoset_1_2_0 --no-deps
.venv/Scripts/python.exe experiments/c11_paretoset_versions.py
```

Result in one paragraph: 143 recorded results, 141 identical between the two versions; the 2 differences are
`sense="diff"` and `crowding_distance`, which crash in 1.2.0 under pandas 2 / numpy 2 and which we do not use.
`"max"`, `"Max"`, `"MAX"` are equivalent. `distinct=True` keeps only the first of identical objective rows
(and demotes duplicates to later layers in `paretorank`); use `distinct=False`. A NaN is treated as a tie and
can remove valid rows from the front or make the result depend on the row order; never pass NaN. Keep
`paretoset==1.2.5`. Nothing failed.

Files here: `c11_table.md` (every case, both versions side by side), `c11_results.json` (environments and the
list of differing cases), `worker_paretoset_1_2_0.json`, `worker_paretoset_1_2_5.json`, `run_log.txt`,
`tmp/` (throw-away install, can be deleted).
