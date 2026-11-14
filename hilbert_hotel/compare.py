"""Consistent hashing vs hash mod N on the same guests and the same building change (spec 5 and 9).

Used by the "compare" menu item and by experiment B, so both methods get the same guest ids,
the same building ids and the same guest hashes (taken from HotelSystem).
Each method runs on a copy, so the real system is untouched and every one starts from the same state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .hotel import HotelSystem
from .migration import METHOD_MODN, METHOD_RANGE, METHOD_SCAN, MigrationReport
from .modn import ModNHotel
from .result import Result


@dataclass(frozen=True)
class ComparisonRow:
    operation: str  # "add" or "remove"
    node_id: int
    method: str  # scan / range / modn
    buildings_before: int
    guests: int  # K
    moved: int
    rate: Optional[float]
    expected_moved: float  # expected value from the analysis for that method (not a measured value)
    time_s: float
    timings: Dict[str, float] = field(default_factory=dict)


def expected_moves(method: str, operation: str, n: int, k: int) -> float:
    """Expected number of movers when the hash is uniform (N = buildings before the command).

    Consistent hashing: add K/(N+1), remove K/N
    hash mod N: add K*N/(N+1), remove K*(N-1)/N  (from the share of guests whose h mod N' still
                lands on the old building, which is only 1/(N+1) and 1/N respectively; the
                derivation is in the report)
    """
    if method == METHOD_MODN:
        return k * n / (n + 1) if operation == "add" else k * (n - 1) / n
    return k / (n + 1) if operation == "add" else k / n


def build_mod_n_from(system: HotelSystem) -> ModNHotel:
    """Build a ModNHotel from the system's guests and buildings (ids sorted ascending = order added)."""
    mod = ModNHotel(system.buildings(), salt=system.ring.salt, hash_fn=system.ring.hash_fn)
    mod.add_guests_with_positions((g.c, g.s, g.pos) for g in system.store.guests())
    return mod


def _row(operation: str, report: MigrationReport) -> ComparisonRow:
    return ComparisonRow(
        operation=operation,
        node_id=report.node_id,
        method=report.method,
        buildings_before=report.buildings_before,
        guests=report.guests_before,
        moved=report.moved_count,
        rate=report.rate,
        expected_moved=expected_moves(
            report.method, operation, report.buildings_before, report.guests_before
        ),
        time_s=report.total_time,
        timings=dict(report.timings),
    )


def compare_with_mod_n(system: HotelSystem, add_id: int, remove_id: int) -> Result:
    """Run "add building add_id" and "remove building remove_id" (separately, each from the
    original state) with all three methods.

    `data` is a list of ComparisonRow (6 rows: 2 commands x 3 methods).
    Returns a failure if a building id doesn't fit, without touching the real system.
    """
    buildings = system.buildings()
    if add_id in buildings:
        return Result.failure(f"อาคาร {add_id} มีอยู่แล้ว เปรียบเทียบการเพิ่มไม่ได้")
    if remove_id not in buildings:
        return Result.failure(f"ไม่พบอาคาร {remove_id} เปรียบเทียบการลบไม่ได้")
    if len(buildings) < 2:
        return Result.failure("ต้องมีอย่างน้อย 2 อาคารจึงเปรียบเทียบการลบได้")

    rows: List[ComparisonRow] = []
    for operation, node_id in (("add", add_id), ("remove", remove_id)):
        for method in (METHOD_SCAN, METHOD_RANGE):
            copy = system.clone()
            action = copy.add_building if operation == "add" else copy.remove_building
            report = action(node_id, method=method).data
            rows.append(_row(operation, report))
        mod = build_mod_n_from(system)
        action = mod.add_building if operation == "add" else mod.remove_building
        rows.append(_row(operation, action(node_id).data))
    return Result.success("เปรียบเทียบเสร็จ", rows)
