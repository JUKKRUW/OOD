"""Comparison baseline: building = hash(guest id) mod N (spec 7.1).

    * keep building ids in the order they were added, pick the one at index H(guest) mod N
    * add building = append to the list, remove = take that id out, rest keep their order
    * moves are compared by the real building id, not by the index
    * same H as the ring (hash64 of "guest:c:s" plus salt), so the hash values are identical
"""

from __future__ import annotations

from time import perf_counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .hashing import guest_key, hash64
from .migration import METHOD_MODN, MigrationRecord, MigrationReport
from .result import Result
from .room import room_no
from .stats import LoadStats, compute_load_stats
from .validation import is_non_negative_int, is_positive_int


class ModNHotel:
    def __init__(self, node_ids: Sequence[int], salt: object = "", hash_fn=hash64) -> None:
        ids = list(node_ids)
        if not ids or len(set(ids)) != len(ids) or not all(is_non_negative_int(i) for i in ids):
            raise ValueError("ต้องมีอาคารอย่างน้อย 1 หลัง รหัสไม่ซ้ำและเป็นจำนวนเต็มไม่ติดลบ")
        self._ids: List[int] = ids
        self.salt = str(salt)
        self._hash_fn = hash_fn
        self._pos: Dict[Tuple[int, int], int] = {}  # (c, s) -> hash value H(guest)
        self._node: Dict[Tuple[int, int], int] = {}  # (c, s) -> building (real id)

    # ---------------------------------------------------------------- data
    def __len__(self) -> int:
        return len(self._node)

    @property
    def building_ids(self) -> List[int]:
        """Building ids in the order they were added (this order decides what mod N gives)."""
        return list(self._ids)

    @property
    def guest_count(self) -> int:
        return len(self._node)

    def node_of(self, c: int, s: int) -> Optional[int]:
        return self._node.get((c, s))

    def load_counts(self) -> Dict[int, int]:
        counts = {nid: 0 for nid in sorted(self._ids)}
        for nid in self._node.values():
            counts[nid] += 1
        return counts

    def load_report(self) -> LoadStats:
        return compute_load_stats(self.load_counts())

    def snapshot(self) -> Dict[Tuple[int, int], int]:
        return dict(self._node)

    # ---------------------------------------------------------------- placing guests
    def _choose(self, position: int) -> int:
        ids = self._ids
        return ids[position % len(ids)]

    def add_guests(self, c: int, s_start: int, count: int) -> Result:
        """Add a group of guests, hashing with the same function as the ring."""
        if not (is_positive_int(c) and is_positive_int(s_start)):
            return Result.failure(f"รหัสช่องทางและลำดับเริ่มต้นต้องเป็นจำนวนเต็มบวก แต่ได้ ({c!r}, {s_start!r})")
        if not is_non_negative_int(count):
            return Result.failure(f"จำนวนแขกต้องเป็นจำนวนเต็มไม่ติดลบ แต่ได้ {count!r}")
        return self.add_guests_with_positions(
            (c, s, self._hash_fn(guest_key(c, s), self.salt)) for s in range(s_start, s_start + count)
        )

    def add_guests_with_positions(self, entries: Iterable[Tuple[int, int, int]]) -> Result:
        """Add guests from (c, s, hash) entries that were already computed, so both methods can
        share the exact same hash values.

        One duplicate id rejects the whole group.
        """
        batch = list(entries)
        seen = set()
        for c, s, _pos in batch:
            key = (c, s)
            if key in self._node or key in seen:
                return Result.failure(f"รหัสแขก {key} ซ้ำ ปฏิเสธทั้งกลุ่ม ข้อมูลเดิมไม่เปลี่ยน")
            seen.add(key)
        for c, s, pos in batch:
            self._pos[(c, s)] = pos
            self._node[(c, s)] = self._choose(pos)
        return Result.success(f"เพิ่มแขก {len(batch)} คน", len(batch))

    # ---------------------------------------------------------------- add / remove building
    def add_building(self, node_id: int) -> Result:
        if not is_non_negative_int(node_id):
            return Result.failure(f"รหัสอาคารต้องเป็นจำนวนเต็มไม่ติดลบ แต่ได้ {node_id!r}")
        if node_id in self._ids:
            return Result.failure(f"อาคาร {node_id} มีอยู่แล้ว ไม่มีการเปลี่ยนแปลง")
        n_before = len(self._ids)
        k_before = len(self._node)
        t = perf_counter()
        self._ids.append(node_id)  # append to the end
        ring_update = perf_counter() - t
        return self._rebalance("add_building", node_id, n_before, n_before + 1, k_before, ring_update)

    def remove_building(self, node_id: int) -> Result:
        if not is_non_negative_int(node_id) or node_id not in self._ids:
            return Result.failure(f"ไม่พบอาคาร {node_id!r} ไม่มีการเปลี่ยนแปลง")
        if len(self._ids) == 1:
            return Result.failure(f"ลบอาคาร {node_id} ไม่ได้ เพราะเป็นอาคารสุดท้าย")
        n_before = len(self._ids)
        k_before = len(self._node)
        t = perf_counter()
        self._ids.remove(node_id)  # keeps the order of the rest
        ring_update = perf_counter() - t
        return self._rebalance("remove_building", node_id, n_before, n_before - 1, k_before, ring_update)

    def _rebalance(
        self, kind: str, node_id: int, n_before: int, n_after: int, k_before: int, ring_update: float
    ) -> Result:
        """Recompute every guest's building (mod the new N'), compare by real building id, then move. O(K)."""
        timings = {"ring_update": ring_update, "find": 0.0, "apply": 0.0}
        t = perf_counter()
        ids = self._ids
        n = len(ids)
        movers = []
        for key, pos in self._pos.items():
            new_node = ids[pos % n]
            old_node = self._node[key]
            if new_node != old_node:
                movers.append((key, old_node, new_node))
        timings["find"] = perf_counter() - t

        t = perf_counter()
        records = []
        for (c, s), old_node, new_node in movers:
            self._node[(c, s)] = new_node
            records.append(MigrationRecord(c, s, old_node, new_node, room_no(c, s)))
        records.sort(key=lambda r: (r.channel_id, r.sequence_id))
        timings["apply"] = perf_counter() - t

        report = MigrationReport(kind, node_id, METHOD_MODN, n_before, n_after, k_before, records, timings)
        return Result.success(report.summary(), report)
