"""Runner for experiment A (data size), B (number of buildings) and C (number of virtual nodes).

Rules from section 7 of the spec
    * guests come from 10 channels, spread as evenly as possible (differ by at most 1 guest)
    * 3 rounds per experiment with salt "0", "1", "2", and both methods use the same salt in a round
    * timing: time.perf_counter(), separately for ring build / placement / lookup / sorting /
      add-remove building. Lookups are timed in batches of 1,000 and divided, and keyboard
      input, printing and file writing are left out.
    * memory: tracemalloc, measuring what was added since the same starting point, reported as
      current and peak (MiB). Done in a separate round from the timing (tracemalloc slows
      the program down a lot).
    * report the median of the 3 rounds (plus mean and min-max to show the spread)
"""

from __future__ import annotations

import gc
import platform
import sys
import tracemalloc
from datetime import datetime
from time import perf_counter
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from hilbert_hotel import HashRing, HotelSystem
from hilbert_hotel.compare import compare_with_mod_n
from hilbert_hotel.migration import METHOD_RANGE

SALTS: Tuple[str, ...] = ("0", "1", "2")
CHANNELS = 10
LOOKUP_BATCH = 1000
RING_BUILD_REPEATS = 50  # building the ring takes a fraction of a millisecond, so average over many builds to cut noise
MIB = 1024 * 1024

Progress = Optional[Callable[[str], None]]

A_VALUE_KEYS = (
    "t_ring_build_s", "t_place_guests_s", "t_lookup_by_id_us", "t_lookup_by_room_us",
    "t_compute_building_us", "t_sort_rooms_s",
    "mem_ring_current_mib", "mem_ring_peak_mib", "mem_total_current_mib", "mem_total_peak_mib",
)
B_VALUE_KEYS = ("moved", "rate", "expected_moved", "time_s", "t_ring_update_s", "t_find_s", "t_apply_s")
C_VALUE_KEYS = (
    "max", "min", "mean", "std", "cv", "mem_ring_current_mib", "mem_ring_peak_mib",
    "mem_total_current_mib", "mem_total_peak_mib", "add_moved", "add_rate", "remove_moved", "remove_rate",
)


# ---------------------------------------------------------------------- general helpers
def median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def split_evenly(k: int, channels: int = CHANNELS) -> List[int]:
    """Split K guests over `channels` channels as evenly as possible (differ by at most 1)."""
    base, extra = divmod(k, channels)
    return [base + 1 if i < extra else base for i in range(channels)]


def build_system(n: int, v: int, k: int, salt: str, channels: int = CHANNELS,
                 method: str = METHOD_RANGE) -> HotelSystem:
    return HotelSystem.initialize(n, v, channels, split_evenly(k, channels), salt=salt,
                                  migration_method=method)


def build_ring(n: int, v: int, salt: str) -> HashRing:
    ring = HashRing(v, salt)
    ring.add_nodes(range(1, n + 1))
    return ring


def measure_memory(builder: Callable[[], object]) -> Tuple[float, float]:
    """(current, peak) in MiB of what was *added* since the same starting point, as far as
    tracemalloc can see.

    This is memory tracked by Python, not the whole process.
    """
    gc.collect()
    tracemalloc.start()
    try:
        base, _ = tracemalloc.get_traced_memory()
        if hasattr(tracemalloc, "reset_peak"):
            tracemalloc.reset_peak()
        obj = builder()  # keep obj alive until we've read the numbers, so it isn't garbage collected early
        current, peak = tracemalloc.get_traced_memory()
        del obj
    finally:
        tracemalloc.stop()
    return (current - base) / MIB, (peak - base) / MIB


def _say(progress: Progress, message: str) -> None:
    if progress:
        progress(message)


def environment_info(salts: Sequence[str] = SALTS) -> Dict[str, object]:
    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.replace("\n", " "),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "salts": list(salts),
        "channels": CHANNELS,
        "lookup_batch": LOOKUP_BATCH,
        "ring_build_repeats": RING_BUILD_REPEATS,
        "timer": "time.perf_counter",
        "memory": "tracemalloc (หน่วยความจำที่ Python ติดตาม ไม่ใช่ของทั้งโปรเซส) แยกรอบจากรอบจับเวลา",
    }


# ---------------------------------------------------------------------- experiment A
def _sample_ids(counts: Sequence[int]) -> List[Tuple[int, int]]:
    """Pick 1,000 guest ids deterministically (evenly spaced) so runs can be repeated."""
    all_ids = [(c, s) for c, count in enumerate(counts, start=1) for s in range(1, count + 1)]
    step = max(1, len(all_ids) // LOOKUP_BATCH)
    return all_ids[::step][:LOOKUP_BATCH]


def time_run_a(k: int, n: int, v: int, salt: str, channels: int = CHANNELS) -> Dict[str, float]:
    counts = split_evenly(k, channels)

    gc.collect()
    start = perf_counter()
    for _ in range(RING_BUILD_REPEATS):
        build_ring(n, v, salt)
    t_ring = (perf_counter() - start) / RING_BUILD_REPEATS

    system = HotelSystem(v, salt)
    system.ring.add_nodes(range(1, n + 1))
    gc.collect()
    start = perf_counter()
    for c, count in enumerate(counts, start=1):
        if count:
            system.add_guests(c, 1, count)
    t_place = perf_counter() - start

    sample = _sample_ids(counts)
    addresses = [system.store.get(c, s).address for c, s in sample]

    find_by_guest = system.find_by_guest
    start = perf_counter()
    for c, s in sample:
        find_by_guest(c, s)
    t_by_id = (perf_counter() - start) / len(sample)

    find_by_room = system.find_by_room
    start = perf_counter()
    for node_id, room in addresses:
        find_by_room(node_id, room)
    t_by_room = (perf_counter() - start) / len(sample)

    compute = system.compute_building
    start = perf_counter()
    for c, s in sample:
        compute(c, s)
    t_compute = (perf_counter() - start) / len(sample)

    start = perf_counter()
    system.occupied_rooms()
    t_sort = perf_counter() - start

    return {
        "t_ring_build_s": t_ring,
        "t_place_guests_s": t_place,
        "t_lookup_by_id_us": t_by_id * 1e6,
        "t_lookup_by_room_us": t_by_room * 1e6,
        "t_compute_building_us": t_compute * 1e6,
        "t_sort_rooms_s": t_sort,
    }


def memory_run(k: int, n: int, v: int, salt: str, channels: int = CHANNELS) -> Dict[str, float]:
    ring_cur, ring_peak = measure_memory(lambda: build_ring(n, v, salt))
    total_cur, total_peak = measure_memory(lambda: build_system(n, v, k, salt, channels))
    return {
        "mem_ring_current_mib": ring_cur,
        "mem_ring_peak_mib": ring_peak,
        "mem_total_current_mib": total_cur,
        "mem_total_peak_mib": total_peak,
    }


def run_experiment_a(ks: Iterable[int] = (1000, 10000, 100000), n: int = 10, v: int = 32,
                     salts: Sequence[str] = SALTS, channels: int = CHANNELS,
                     progress: Progress = None) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for k in ks:
        for round_no, salt in enumerate(salts, start=1):
            _say(progress, f"[A] K={k} N={n} V={v} salt={salt} (รอบ {round_no}/{len(salts)}) จับเวลา")
            row: Dict[str, object] = {"experiment": "A", "K": k, "N": n, "V": v, "channels": channels,
                                      "salt": salt, "round": round_no}
            row.update(time_run_a(k, n, v, salt, channels))
            _say(progress, f"[A] K={k} salt={salt} วัดหน่วยความจำ (รอบแยก)")
            row.update(memory_run(k, n, v, salt, channels))
            rows.append(row)
    return rows


# ---------------------------------------------------------------------- experiment B
def run_experiment_b(ns: Iterable[int] = (5, 10, 20), k: int = 10000, v: int = 32,
                     salts: Sequence[str] = SALTS, channels: int = CHANNELS,
                     progress: Progress = None) -> List[Dict[str, object]]:
    """Add 1 building (id N+1) and remove 1 building (id (N+1)//2, a middle one),
    comparing consistent hashing with mod N."""
    rows: List[Dict[str, object]] = []
    for n in ns:
        for round_no, salt in enumerate(salts, start=1):
            _say(progress, f"[B] N={n} K={k} V={v} salt={salt} (รอบ {round_no}/{len(salts)})")
            system = build_system(n, v, k, salt, channels)
            result = compare_with_mod_n(system, add_id=n + 1, remove_id=(n + 1) // 2)
            assert result.ok, result.message
            for cmp_row in result.data:
                rows.append({
                    "experiment": "B", "N": n, "K": k, "V": v, "channels": channels, "salt": salt,
                    "round": round_no, "operation": cmp_row.operation, "node_id": cmp_row.node_id,
                    "method": cmp_row.method, "moved": cmp_row.moved,
                    "rate": "" if cmp_row.rate is None else cmp_row.rate,
                    "expected_moved": cmp_row.expected_moved, "time_s": cmp_row.time_s,
                    "t_ring_update_s": cmp_row.timings.get("ring_update", 0.0),
                    "t_find_s": cmp_row.timings.get("find", 0.0),
                    "t_apply_s": cmp_row.timings.get("apply", 0.0),
                })
    return rows


# ---------------------------------------------------------------------- experiment C
def run_experiment_c(vs: Iterable[int] = (1, 8, 32), n: int = 10, k: int = 10000,
                     salts: Sequence[str] = SALTS, channels: int = CHANNELS,
                     progress: Progress = None) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for v in vs:
        for round_no, salt in enumerate(salts, start=1):
            _say(progress, f"[C] V={v} N={n} K={k} salt={salt} (รอบ {round_no}/{len(salts)})")
            system = build_system(n, v, k, salt, channels)
            stats = system.load_report()
            add_report = system.clone().add_building(n + 1).data
            remove_report = system.clone().remove_building((n + 1) // 2).data
            row: Dict[str, object] = {
                "experiment": "C", "V": v, "N": n, "K": k, "channels": channels, "salt": salt,
                "round": round_no, "points_M": system.ring.point_count,
                "counts": ";".join(str(stats.counts[nid]) for nid in sorted(stats.counts)),
                "max": stats.maximum, "min": stats.minimum, "mean": stats.mean, "std": stats.std,
                "cv": "" if stats.cv is None else stats.cv,
                "add_moved": add_report.moved_count, "add_rate": add_report.rate,
                "remove_moved": remove_report.moved_count, "remove_rate": remove_report.rate,
            }
            row.update(memory_run(k, n, v, salt, channels))
            rows.append(row)
    return rows


# ---------------------------------------------------------------------- summarising the rounds
def to_number(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def summarize(rows: Sequence[Dict[str, object]], group_keys: Sequence[str],
              value_keys: Sequence[str]) -> List[Dict[str, object]]:
    """Fold the rounds (different salts) into median, mean, min and max for each parameter group."""
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    order: List[Tuple[object, ...]] = []
    for row in rows:
        key = tuple(row[g] for g in group_keys)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)
    summary: List[Dict[str, object]] = []
    for key in order:
        members = groups[key]
        out: Dict[str, object] = dict(zip(group_keys, key))
        out["rounds"] = len(members)
        out["salts"] = ";".join(str(m["salt"]) for m in members)
        for value_key in value_keys:
            numbers = [x for x in (to_number(m.get(value_key)) for m in members) if x is not None]
            if numbers:
                out[f"{value_key}_median"] = median(numbers)
                out[f"{value_key}_mean"] = mean(numbers)
                out[f"{value_key}_min"] = min(numbers)
                out[f"{value_key}_max"] = max(numbers)
        summary.append(out)
    return summary
