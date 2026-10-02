"""Shared helpers of experiment C1 (runtime).

* ``LoadMeter``     - wall time, CPU time of this process and the system-wide
                      CPU utilisation during a timed block (Windows only for
                      the system-wide part; no extra package needed).
* ``IterationClock`` - a file-like object that timestamps the progress lines
                      the ORIGINAL code prints, so that the time of every
                      single (itemset, repeat) can be read without touching
                      the original code.
* ``load_itemsets`` - the activity sets per size (1, 2, 3) of both strategies.
* ``summarise``     - median / mean / min / max of a list of seconds.

Nothing here imports the original repositories.
"""

from __future__ import annotations

import ctypes
import json
import os
import statistics
import sys
import time
from pathlib import Path

from apriori_selector import AprioriSelector
from engine import EventLog

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results" / "experiments" / "C1"
IMPRESSED_SETS_DIR = PROJECT_ROOT / "results" / "experiments" / "C4"

FOLD_SEED = 0          # seed of the stratified 5-fold split (as in C2)
N_FOLDS = 5
SIZES = (1, 2, 3)
# Largest support on the C7 grid at which f1, f2 and f3 all have >= 10 sets
# of every size. The top-k per size does not depend on the value as long as
# enough sets exist, because the ranking is by support.
APRIORI_MIN_SUPPORT = 0.45
TOP_K = 10


# --------------------------------------------------------------------------
# Machine load
# --------------------------------------------------------------------------
class _FileTime(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

    def ticks(self) -> int:
        """The value in 100-nanosecond units."""
        return (self.high << 32) | self.low


def system_cpu_ticks() -> tuple[int, int] | None:
    """``(idle, total)`` CPU time of the whole machine, summed over all cores.

    Uses ``GetSystemTimes`` of the Windows API (kernel time includes idle
    time there). Returns None on other operating systems.
    """
    if sys.platform != "win32":
        return None
    idle, kernel, user = _FileTime(), _FileTime(), _FileTime()
    ctypes.windll.kernel32.GetSystemTimes(
        ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
    return idle.ticks(), kernel.ticks() + user.ticks()


class LoadMeter:
    """Context manager that measures a block.

    After the block:

    seconds          wall time
    own_cores        CPU seconds of this process per wall second (1.0 = one
                     core fully used; XGBoost can use several)
    system_percent   busy share of all logical cores, whole machine, in %
                     (None when it cannot be measured)
    others_percent   ``system_percent`` minus this process's share: the load
                     caused by everything else on the machine
    """

    def __enter__(self):
        self._ticks = system_cpu_ticks()
        self._cpu = time.process_time()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        self.seconds = time.perf_counter() - self._start
        own_cpu = time.process_time() - self._cpu
        self.own_cores = own_cpu / self.seconds if self.seconds > 0 else 0.0
        self.system_percent = self.others_percent = None
        ticks = system_cpu_ticks()
        if ticks is not None and ticks[1] > self._ticks[1]:
            idle = ticks[0] - self._ticks[0]
            total = ticks[1] - self._ticks[1]
            self.system_percent = 100.0 * (1.0 - idle / total)
            own_percent = 100.0 * self.own_cores / (os.cpu_count() or 1)
            self.others_percent = max(0.0, self.system_percent - own_percent)
        return False

    def as_dict(self) -> dict:
        """The measurements as a JSON-friendly dict."""
        return {
            "seconds": self.seconds,
            "own_cores": self.own_cores,
            "system_percent": self.system_percent,
            "others_percent": self.others_percent,
        }


def sample_system_load(seconds: float = 2.0) -> float | None:
    """System-wide CPU utilisation in % while this process sleeps."""
    with LoadMeter() as meter:
        time.sleep(seconds)
    return meter.system_percent


# --------------------------------------------------------------------------
# Per-iteration timestamps of the original code
# --------------------------------------------------------------------------
class StopTiming(Exception):
    """Raised by ``IterationClock`` to end a run of the original code early."""


class IterationClock:
    """File-like object that records when the original prints a progress line.

    The original prints one line per finished (itemset, repeat)
    ("Itemset: ...", tools.py:544) and one per skipped single-activity
    iteration ("No shuffled cases. skipped!", tools.py:412). Redirect stdout
    to an instance and read ``durations()`` afterwards.

    Parameters
    ----------
    markers : texts that mark the end of one iteration when printed.
    max_events : stop the run after this many events by raising
        ``StopTiming`` (None = never).
    """

    def __init__(self, markers=("Itemset:", "No shuffled cases"), max_events=None):
        self.markers = tuple(markers)
        self.max_events = max_events
        self.events: list[tuple[str, float]] = []
        self.start = time.perf_counter()

    def restart(self) -> None:
        """Forget all events and set the start time to now."""
        self.events.clear()
        self.start = time.perf_counter()

    def mark(self, kind: str) -> None:
        """Record one finished iteration of type ``kind``."""
        self.events.append((kind, time.perf_counter()))
        if self.max_events is not None and len(self.events) >= self.max_events:
            raise StopTiming

    def write(self, text: str) -> int:
        """``print`` target: timestamp the marker lines, drop everything."""
        for marker in self.markers:
            if text.startswith(marker):
                self.mark(marker)
                break
        return len(text)

    def flush(self) -> None:
        """Nothing is buffered."""

    def durations(self) -> list[float]:
        """Seconds between consecutive events (the first counts from start)."""
        times = [self.start] + [stamp for _, stamp in self.events]
        return [later - earlier for earlier, later in zip(times, times[1:])]


# --------------------------------------------------------------------------
# Activity sets
# --------------------------------------------------------------------------
def load_itemsets(log: EventLog, dataset: str,
                  strategy: str) -> dict[int, list[list[str]]]:
    """``{size: [activity list, ...]}`` with ``TOP_K`` sets per size.

    ``strategy='apriori'``: the per-size Apriori selector of experiment C7
    (most frequent sets first). ``strategy='impressed'``: the sets selected in
    experiment C4 (``results/experiments/C4/impressed_sets_<dataset>.json``);
    they are the working selection of that experiment, used here only as a
    realistic workload.
    """
    if strategy == "apriori":
        selector = AprioriSelector(APRIORI_MIN_SUPPORT, max_len=max(SIZES), top_k=TOP_K)
        selected = selector.select(log.traces)
    elif strategy == "impressed":
        path = IMPRESSED_SETS_DIR / f"impressed_sets_{dataset}.json"
        stored = json.loads(path.read_text(encoding="utf-8"))["selection"]
        selected = {int(size): sets for size, sets in stored.items()}
    else:
        raise ValueError("strategy must be 'apriori' or 'impressed'")
    for size in SIZES:
        if len(selected.get(size, [])) < TOP_K:
            raise ValueError(
                f"{dataset}/{strategy}: fewer than {TOP_K} sets of size {size}")
        if any(len(set(items)) != size for items in selected[size]):
            raise ValueError(
                f"{dataset}/{strategy}: a set of size {size} has another size")
    return {size: [list(items) for items in selected[size][:TOP_K]] for size in SIZES}


def count_cases_with(traces, itemset) -> int:
    """Number of traces that contain every activity of ``itemset``."""
    wanted = set(itemset)
    return sum(wanted.issubset(trace) for trace in traces)


# --------------------------------------------------------------------------
# Small statistics
# --------------------------------------------------------------------------
def summarise(seconds) -> dict:
    """Median, mean, min, max and count of a list of seconds."""
    seconds = list(seconds)
    if not seconds:
        return {"n": 0, "median": None, "mean": None, "min": None, "max": None}
    return {
        "n": len(seconds),
        "median": statistics.median(seconds),
        "mean": statistics.fmean(seconds),
        "min": min(seconds),
        "max": max(seconds),
    }


def write_json(path: Path, content: dict) -> None:
    """Write ``content`` as indented JSON (creating the folder)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")
