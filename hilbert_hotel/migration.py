"""Migration records and reports (spec 4.4).

"Moved" = guest ids whose node_id is different before and after the command, one count per
guest (not per virtual node). A record has to give the guest id, old building, new building
and room number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

METHOD_SCAN = "scan"  # method A: check every guest
METHOD_RANGE = "range"  # method B: fetch only the guests in the affected ranges
METHOD_MODN = "modn"  # hash mod N baseline

METHOD_LABELS = {
    METHOD_SCAN: "วิธี ก (ตรวจแขกทุกคน)",
    METHOD_RANGE: "วิธี ข (เฉพาะช่วงที่ได้รับผลกระทบ)",
    METHOD_MODN: "hash mod N",
}


@dataclass(frozen=True, slots=True)
class MigrationRecord:
    channel_id: int  # c
    sequence_id: int  # s
    from_node: int
    to_node: int
    room_no: int  # unchanged by a move

    @property
    def guest_id(self):
        return (self.channel_id, self.sequence_id)


@dataclass
class MigrationReport:
    kind: str  # "add_building" or "remove_building"
    node_id: int
    method: str
    buildings_before: int
    buildings_after: int
    guests_before: int  # K
    moved: List[MigrationRecord] = field(default_factory=list)
    # per-phase times (seconds): ring_update = update the ring, find = find affected guests,
    # apply = update the records and indexes
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def moved_count(self) -> int:
        return len(self.moved)

    @property
    def rate(self) -> Optional[float]:
        """Migration rate = moved / K; None when K = 0."""
        if self.guests_before == 0:
            return None
        return self.moved_count / self.guests_before

    @property
    def rate_text(self) -> str:
        if self.rate is None:
            return "ไม่คำนวณอัตรา เพราะไม่มีแขก (K = 0)"
        return f"{self.rate:.6f} ({self.rate * 100:.3f}%)"

    @property
    def total_time(self) -> float:
        return sum(self.timings.values())

    def summary(self) -> str:
        action = "เพิ่ม" if self.kind == "add_building" else "ลบ"
        return (
            f"{action}อาคาร {self.node_id}: N {self.buildings_before} -> {self.buildings_after}, "
            f"K = {self.guests_before}, ย้าย {self.moved_count} คน, อัตราย้าย = {self.rate_text}"
        )
