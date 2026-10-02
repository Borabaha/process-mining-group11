"""Experiment C3: run the ORIGINAL IMPresseD automatic mode headlessly.

The 2023 tool (Vazifehdoostirani et al., "Interactive Multi-Interest Process
Pattern Discovery") can only be started through a Tk window
(``GUI_IMPresseD_tool.py``).  This script rebuilds, step by step, the inputs
that the GUI prepares before it calls ``Auto_IMPID.AutoStepWise_PPD`` and then
calls that function unchanged.  Every preparation function below names the GUI
lines it mirrors, so the students can check it against the original.

Nothing in ``external/`` is edited or written to.  The original functions are
only *wrapped* at run time (see :class:`StageProfiler`) to measure wall time
per stage and to keep a reference to the pattern dictionary, which the
original function does not return.

Usage (from the project root, with the project's Python 3.12 venv)::

    .venv/Scripts/python.exe -u experiments/run_original_impressed.py \
        --max-extension-step 1 --run-name step1

Deliberate differences from a GUI session (all outside the original code):

* the pairwise-distance pickle is written to the run folder instead of
  ``<csv folder>/dist`` (the GUI would write into the dataset folder and would
  silently re-use that file for every other CSV in the same folder);
* ``random.seed`` is fixed before the node colours are drawn (the GUI draws
  them unseeded; colours do not influence the discovery);
* ``test_data_percentage`` has no default in the GUI; 0.2 is used here;
* pandas ``PerformanceWarning`` ("DataFrame is highly fragmented") is silenced,
  because the original code triggers it thousands of times and floods the log.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import pickle
import random
import sys
import threading
import time
import traceback
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # IMIPD imports pyplot; never open a window

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CODE_DIR = PROJECT_ROOT / "external" / "InteractivePatternDetection"
DEFAULT_DATASET = (PROJECT_ROOT / "external" / "PermutationLocationImportance"
                   / "datasets" / "BPIC11_f1_trunc36.csv")
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "experiments" / "C3"

SEED = 2023
INTEREST_COLUMNS = ["Outcome_Interest", "Frequency_Interest", "Case_Distance_Interest"]


@dataclass
class Settings:
    """Everything a user would type or click in the GUI before the automatic mode."""

    dataset: Path = DEFAULT_DATASET
    case_id: str = "case:concept:name"
    activity: str = "concept:name"
    timestamp: str = "time:timestamp"
    outcome: str = "label"
    outcome_type: str = "binary"
    delta_time: float = -1.0  # seconds; negative => no two events are concurrent (chains)
    max_gap_between_events: int = 3
    max_extension_step: int = 1
    test_data_percentage: float = 0.2
    numerical_attributes: list = field(default_factory=lambda: ["Age"])
    categorical_attributes: list = field(default_factory=lambda: [
        "Diagnosis", "Treatment code", "Diagnosis code", "Specialism code"])
    # GUI check boxes, in GUI order, with the GUI's default directions (GUI:273/287/301).
    pareto_features: list = field(default_factory=lambda: list(INTEREST_COLUMNS))
    pareto_sense: list = field(default_factory=lambda: ["Max", "Max", "Min"])
    log_prep: str = "gui"  # "gui" = raw CSV as the GUI reads it; "dm2024" = 2024 loader filter first
    max_cases: int | None = None  # smoke tests only


def log(message: str) -> None:
    """Print a time-stamped progress line (stdout is redirected to the run log)."""
    print("[%s] %s" % (time.strftime("%H:%M:%S"), message), flush=True)


# --------------------------------------------------------------------------- #
# Input preparation: the GUI steps, without the GUI
# --------------------------------------------------------------------------- #

def apply_2024_log_filter(df: pd.DataFrame, s: Settings, frq_threshold: int = 2) -> pd.DataFrame:
    """Optional: normalise and filter the log like the 2024 ``DataManager._load_df``.

    Lower-case activity labels and drop every case that contains an activity
    with fewer than ``frq_threshold`` events (2024 tools.py:39, :60-64).  This
    is NOT part of the 2023 tool; it only makes the mined log identical to the
    one the 2024 pipeline uses (f1: 1130 cases, 164 activities).
    """
    df = df.copy()
    df[s.activity] = df[s.activity].str.lower()
    events_per_activity = df.groupby(s.activity)[s.case_id].count()
    rare = events_per_activity[events_per_activity < frq_threshold].index
    bad_cases = df.loc[df[s.activity].isin(rare), s.case_id].unique()
    return df[~df[s.case_id].isin(bad_cases)].reset_index(drop=True)


def load_event_log(s: Settings) -> pd.DataFrame:
    """Read the CSV and format the columns as ``save_setting`` does (GUI:178, :235-238)."""
    df = pd.read_csv(s.dataset)
    if s.log_prep == "dm2024":
        df = apply_2024_log_filter(df, s)
    if s.max_cases is not None:
        keep = df[s.case_id].drop_duplicates().head(s.max_cases)
        df = df[df[s.case_id].isin(keep)].reset_index(drop=True)
    # Pattern IDs are "<core activity>_<number>", so "_" must not occur in labels.
    df[s.activity] = df[s.activity].str.replace("_", "-")
    df[s.timestamp] = pd.to_datetime(df[s.timestamp])
    df[s.case_id] = df[s.case_id].astype(str)
    return df


def make_colour_dict(df: pd.DataFrame, s: Settings) -> dict:
    """One random colour per activity plus 'start'/'end' (GUI:240-249)."""
    activities = df[s.activity].unique()
    colour_codes = ["#" + "".join(random.choice("0001123456789ABCDEFABC") for _ in range(6))
                    for _ in range(len(activities))]
    colours = dict(zip(activities, colour_codes))
    colours["start"] = "k"
    colours["end"] = "k"
    return colours


def create_patient_data(df: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """Case table: attributes, id, outcome and one zero column per activity (GUI:370-387)."""
    patient_data = pd.DataFrame()
    if s.numerical_attributes:
        patient_data[s.numerical_attributes] = df[s.numerical_attributes]
    if s.categorical_attributes:
        patient_data[s.categorical_attributes] = df[s.categorical_attributes]
    patient_data[s.case_id] = df[s.case_id]
    patient_data[s.outcome] = df[s.outcome]
    patient_data = patient_data.drop_duplicates(subset=[s.case_id], keep="first")
    patient_data.sort_values(by=s.case_id, inplace=True)
    patient_data.reset_index(inplace=True, drop=True)
    patient_data[list(df[s.activity].unique())] = 0
    return patient_data


def fill_activity_counts(df: pd.DataFrame, patient_data: pd.DataFrame, s: Settings, imipd) -> pd.DataFrame:
    """Fill the per-case activity counts through the variant table (GUI:990-1000)."""
    selected_variants = imipd.VariantSelection(df, s.case_id, s.activity, s.timestamp)
    for case in selected_variants["case:concept:name"].unique():
        other_cases = selected_variants.loc[
            selected_variants["case:concept:name"] == case, "case:CaseIDs"].tolist()[0]
        trace = df.loc[df[s.case_id] == case, s.activity].tolist()
        for act in np.unique(trace):
            number_of_act = trace.count(act)
            for other_case in other_cases:
                patient_data.loc[patient_data[s.case_id] == other_case, act] = number_of_act
    return patient_data


def create_pairwise_distance(patient_data: pd.DataFrame, s: Settings, imipd, cache_dir: Path):
    """Pairwise case distances and the pair look-up lists (GUI:389-413).

    Exactly like the GUI, the feature frame is the case table minus id and
    outcome, so the activity-count columns are passed on as *categorical*
    attributes.  The pickle goes to ``cache_dir`` (see module docstring).
    """
    x_features = patient_data.drop([s.case_id, s.outcome], axis=1)
    distances = imipd.calculate_pairwise_case_distance(x_features, s.numerical_attributes)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "pairwise_case_distances.pkl", "wb") as handle:
        pickle.dump(distances, handle)

    pair_cases = [(a, b) for idx, a in enumerate(patient_data.index) for b in patient_data.index[idx + 1:]]
    case_size = len(patient_data)
    i = 0
    start_search_points = []
    for k in range(case_size):
        start_search_points.append(k * case_size - (i + k))
        i += k
    return distances, pair_cases, start_search_points


# --------------------------------------------------------------------------- #
# Instrumentation (wraps the original functions, does not change them)
# --------------------------------------------------------------------------- #

class StageProfiler:
    """Wall-clock bookkeeping for one ``AutoStepWise_PPD`` call.

    ``AutoStepWise_PPD`` is one long function, so stages are recognised by the
    calls it makes: every ``paretoset`` call closes a step (call 1 = step 0,
    call 2 = step 1, ...).  Times of wrapped functions are *inclusive*
    (``Pattern_extension`` contains ``update_pattern_dict``, and so on).
    """

    def __init__(self) -> None:
        self.stage = "step0"
        self.stage_start = time.perf_counter()
        self.stage_wall: dict[str, float] = {}
        self.seconds = defaultdict(lambda: defaultdict(float))  # stage -> function -> seconds
        self.calls = defaultdict(lambda: defaultdict(int))
        self.attribute_frames: list[pd.DataFrame] = []  # one per step, as scored by the tool
        self.front_masks: list[np.ndarray] = []
        self.pattern_dict: dict = {}  # latest dictionary of pattern graphs seen
        self._current_core = None
        self._cores_done = 0

    def timed(self, name: str, function):
        """Return ``function`` wrapped so that its wall time is added to the current stage."""
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                self.seconds[self.stage][name] += time.perf_counter() - start
                self.calls[self.stage][name] += 1
        return wrapper

    def close_stage(self, next_stage: str) -> None:
        """End the running stage and start ``next_stage``."""
        now = time.perf_counter()
        self.stage_wall[self.stage] = now - self.stage_start
        log("stage %s finished: %.1f s" % (self.stage, self.stage_wall[self.stage]))
        self.stage, self.stage_start = next_stage, now

    def wrap_paretoset(self, function):
        """Record front sizes; each call closes one step."""
        timed = self.timed("paretoset", function)

        def wrapper(costs, *args, **kwargs):
            mask = timed(costs, *args, **kwargs)
            step = len(self.front_masks)
            self.front_masks.append(np.asarray(mask))
            log("step %d: %d candidates, %d on the Pareto front (%d candidates have a NaN objective)"
                % (step, len(mask), int(np.sum(mask)), int(costs.isna().any(axis=1).sum())))
            self.close_stage("step%d" % (step + 1))
            return mask
        return wrapper

    def wrap_attributes(self, function):
        """Keep a copy of every interest-value table the tool computes."""
        timed = self.timed("create_pattern_attributes", function)

        def wrapper(*args, **kwargs):
            log("%s: scoring %d candidates ..." % (self.stage, len(args[2])))
            frame = timed(*args, **kwargs)
            self.attribute_frames.append(frame.copy())
            return frame
        return wrapper

    def wrap_pattern_extension(self, function):
        """Step-1 extension: remember the shared dictionary and log one line per core activity."""
        timed = self.timed("Pattern_extension", function)

        def wrapper(case_data, trace_graph, core_activity, *args, **kwargs):
            if core_activity != self._current_core:
                self._cores_done += 1
                log("step 1 extension: core activity #%d %s (dictionary holds %d patterns)"
                    % (self._cores_done, core_activity, len(self.pattern_dict)))
                self._current_core = core_activity
            result = timed(case_data, trace_graph, core_activity, *args, **kwargs)
            self.pattern_dict = result[0]
            return result
        return wrapper

    def wrap_single_extender(self, function):
        """Step >= 2 extension: remember the dictionary that holds all patterns so far."""
        timed = self.timed("Single_Pattern_Extender", function)

        def wrapper(*args, **kwargs):
            result = timed(*args, **kwargs)
            self.pattern_dict = result[0]
            return result
        return wrapper

    def install(self, auto_module, imipd_module) -> None:
        """Replace the names the original code looks up at call time by timed wrappers."""
        auto_module.paretoset = self.wrap_paretoset(auto_module.paretoset)
        auto_module.create_pattern_attributes = self.wrap_attributes(auto_module.create_pattern_attributes)
        auto_module.Pattern_extension = self.wrap_pattern_extension(auto_module.Pattern_extension)
        auto_module.Single_Pattern_Extender = self.wrap_single_extender(auto_module.Single_Pattern_Extender)
        auto_module.Trace_graph_generator = self.timed("Trace_graph_generator", auto_module.Trace_graph_generator)
        for name in ("update_pattern_dict", "create_embedded_pattern_in_trace",
                     "frequency_measuring_patterns", "predictive_measuring_patterns",
                     "similarity_measuring_patterns"):
            setattr(imipd_module, name, self.timed(name, getattr(imipd_module, name)))

    def summary(self) -> dict:
        """Timings as plain dictionaries (the running stage is reported up to now)."""
        stage_wall = dict(self.stage_wall)
        stage_wall[self.stage + " (open)"] = time.perf_counter() - self.stage_start
        return {
            "stage_wall_seconds": stage_wall,
            "inclusive_seconds_per_function": {k: dict(v) for k, v in self.seconds.items()},
            "calls_per_function": {k: dict(v) for k, v in self.calls.items()},
            "candidates_per_step": [int(len(m)) for m in self.front_masks],
            "front_size_per_step": [int(np.sum(m)) for m in self.front_masks],
        }


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #

def pattern_record(pattern_id: str, graph) -> dict:
    """Describe one pattern graph by activity labels and typed edges.

    Nodes are listed in the order of their position in the trace where the
    pattern was first seen; edges refer to positions in ``labels``.
    """
    nodes = sorted(graph.nodes)
    labels = [graph.nodes[n]["value"] for n in nodes]
    edges = [{"source": nodes.index(u), "target": nodes.index(v),
              "type": "eventually" if data.get("eventually") else "directly"}
             for u, v, data in graph.edges(data=True)]
    return {"id": pattern_id, "labels": labels, "edges": edges,
            "n_nodes": len(nodes), "activity_set": sorted(set(labels))}


def collect_patterns(profiler: StageProfiler) -> dict:
    """Join pattern graphs, interest values and front membership per step."""
    steps = {}
    scored_ids = set()
    pattern_items = list(profiler.pattern_dict.items())
    graphs = {pid: entry["pattern"] for pid, entry in pattern_items}
    n_instances = {pid: len(entry["Instances"]["case"]) for pid, entry in pattern_items}
    for step, frame in enumerate(profiler.attribute_frames):
        mask = profiler.front_masks[step] if step < len(profiler.front_masks) else None
        records = []
        for row_number, row in enumerate(frame.to_dict("records")):
            pid = row["patterns"]
            scored_ids.add(pid)
            if step == 0:  # single activities have no graph in the tool
                record = {"id": pid, "labels": [pid], "edges": [], "n_nodes": 1, "activity_set": [pid]}
            else:
                record = pattern_record(pid, graphs[pid])
                record["n_instances_whole_log"] = n_instances[pid]
            record["step"] = step
            record["on_front"] = bool(mask[row_number]) if mask is not None else None
            for column in ["Pattern_Frequency", "Case_Support"] + INTEREST_COLUMNS:
                value = row.get(column)
                record[column] = None if value is None or pd.isna(value) else float(value)
            records.append(record)
        steps[str(step)] = {
            "n_candidates": len(records),
            "front_size": int(np.sum(mask)) if mask is not None else None,
            "patterns": records,
        }
    unscored = [dict(pattern_record(pid, graph), n_instances_whole_log=n_instances[pid])
                for pid, graph in graphs.items() if pid not in scored_ids]
    return {"steps": steps, "unscored_patterns_from_unfinished_step": unscored}


def write_outputs(profiler: StageProfiler, run_dir: Path, patterns_json: Path, meta: dict) -> None:
    """Write timings, per-step interest tables and the pattern export."""
    meta = dict(meta, timings=profiler.summary())
    with open(run_dir / "timings.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, default=str)
    for step, frame in enumerate(profiler.attribute_frames):
        frame = frame.copy()
        if step < len(profiler.front_masks):
            frame["on_front"] = profiler.front_masks[step]
        frame.to_csv(run_dir / ("pattern_attributes_step%d.csv" % step), index=False)
    export = {"meta": meta}
    export.update(collect_patterns(profiler))
    patterns_json.parent.mkdir(parents=True, exist_ok=True)
    with open(patterns_json, "w", encoding="utf-8") as handle:
        json.dump(export, handle, indent=1, default=str)
    log("wrote %s" % patterns_json)


def start_watchdog(limit_minutes: float, on_timeout) -> None:
    """Stop the process after ``limit_minutes``, saving partial results first."""
    def fire():
        log("TIME LIMIT of %.1f min reached - writing partial results and stopping" % limit_minutes)
        try:
            on_timeout()
        except Exception:  # noqa: BLE001 - we are about to exit anyway
            traceback.print_exc()
        sys.stdout.flush()
        os._exit(124)

    timer = threading.Timer(limit_minutes * 60.0, fire)
    timer.daemon = True
    timer.start()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--code-dir", type=Path, default=DEFAULT_CODE_DIR,
                        help="folder with IMIPD.py / Auto_IMPID.py / tools.py (original or patched copy)")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--run-name", default="run", help="sub-folder of the results dir for this run")
    parser.add_argument("--patterns-json", type=Path, default=None,
                        help="where to write the pattern export (default: <run dir>/patterns.json)")
    parser.add_argument("--max-extension-step", type=int, default=1)
    parser.add_argument("--max-gap", type=int, default=3)
    parser.add_argument("--delta-time", type=float, default=-1.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--log-prep", choices=["gui", "dm2024"], default="gui")
    parser.add_argument("--max-cases", type=int, default=None, help="smoke test: keep only the first N cases")
    parser.add_argument("--time-limit-min", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = Settings(dataset=args.dataset.resolve(), delta_time=args.delta_time,
                        max_gap_between_events=args.max_gap, max_extension_step=args.max_extension_step,
                        test_data_percentage=args.test_size, log_prep=args.log_prep, max_cases=args.max_cases)
    run_dir = (args.results_dir / args.run_name).resolve()
    save_path = run_dir / "impressed_output"  # the folder a GUI user would select
    save_path.mkdir(parents=True, exist_ok=True)
    patterns_json = args.patterns_json or run_dir / "patterns.json"

    random.seed(SEED)
    np.random.seed(SEED)
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

    # The 2023 and the 2024 repo both ship a module called tools.py: put the 2023 folder first.
    sys.path.insert(0, str(args.code_dir.resolve()))
    sys.dont_write_bytecode = True  # no __pycache__ inside the read-only external/ folder
    import Auto_IMPID  # noqa: E402  (never import GUI_IMPresseD_tool: it opens a window)
    import IMIPD  # noqa: E402
    import networkx  # noqa: E402
    import paretoset  # noqa: E402
    import pm4py  # noqa: E402
    import scipy  # noqa: E402
    import sklearn  # noqa: E402

    meta = {
        "settings": asdict(settings),
        "code_dir": str(args.code_dir.resolve()),
        "python": sys.version.split()[0],
        "versions": {m.__name__: m.__version__ for m in (np, pd, scipy, sklearn, networkx, pm4py, paretoset)},
        "completed": False,
        "preparation_seconds": {},
    }
    log("settings: %s" % meta["settings"])
    log("versions: %s" % meta["versions"])
    profiler = StageProfiler()
    profiler.install(Auto_IMPID, IMIPD)
    if args.time_limit_min:
        start_watchdog(args.time_limit_min,
                       lambda: write_outputs(profiler, run_dir, patterns_json, meta))
    prep = meta["preparation_seconds"]

    try:
        start = time.perf_counter()
        df = load_event_log(settings)
        colour_dict = make_colour_dict(df, settings)
        prep["load_and_format_log"] = time.perf_counter() - start
        meta["n_events"], meta["n_cases"] = len(df), int(df[settings.case_id].nunique())
        meta["n_activities"] = int(df[settings.activity].nunique())
        log("log: %d events, %d cases, %d activities" % (meta["n_events"], meta["n_cases"], meta["n_activities"]))

        start = time.perf_counter()
        patient_data = create_patient_data(df, settings)
        patient_data = fill_activity_counts(df, patient_data, settings, IMIPD)
        prep["patient_data_and_activity_counts"] = time.perf_counter() - start
        log("case table %s built in %.1f s" % (patient_data.shape, prep["patient_data_and_activity_counts"]))

        start = time.perf_counter()
        distances, pair_cases, start_search_points = create_pairwise_distance(
            patient_data, settings, IMIPD, run_dir / "dist")
        prep["pairwise_case_distances"] = time.perf_counter() - start
        meta["n_case_pairs"] = len(pair_cases)
        log("pairwise distances: %d pairs in %.1f s (min %.4f, mean %.4f, max %.4f)"
            % (len(pair_cases), prep["pairwise_case_distances"], distances.min(), distances.mean(), distances.max()))

        profiler.stage_start = start = time.perf_counter()
        train_x, test_x = Auto_IMPID.AutoStepWise_PPD(
            settings.max_extension_step, settings.max_gap_between_events, settings.test_data_percentage,
            df, patient_data, distances, pair_cases, start_search_points,
            settings.case_id, settings.activity, settings.outcome, settings.outcome_type, settings.timestamp,
            settings.pareto_features, settings.pareto_sense, settings.delta_time, colour_dict, str(save_path))
        profiler.close_stage("after_return")
        meta["auto_stepwise_ppd_seconds"] = time.perf_counter() - start

        # GUI:1016-1017
        train_x.to_csv(save_path / "training_encoded_log.csv", index=False)
        test_x.to_csv(save_path / "testing_encoded_log.csv", index=False)
        meta["train_X_shape"], meta["test_X_shape"] = list(train_x.shape), list(test_x.shape)
        meta["completed"] = True
        log("AutoStepWise_PPD finished in %.1f s; train_X %s, test_X %s"
            % (meta["auto_stepwise_ppd_seconds"], train_x.shape, test_x.shape))
    except Exception:  # noqa: BLE001 - the experiment is about recording every crash
        meta["traceback"] = traceback.format_exc()
        log("CRASH\n" + meta["traceback"])
    write_outputs(profiler, run_dir, patterns_json, meta)
    return 0 if meta["completed"] else 1


if __name__ == "__main__":
    sys.exit(main())
