"""HotelSystem: ties the HashRing and the GuestStore together.

What it does
    * takes commands from the user (menu, experiment scripts, tests) and checks the input
    * picks a building for each guest by *computing* it from the guest id and the current ring
    * adds / removes buildings, finds the movers with method A or B, and keeps the records
      and indexes in sync

Every command that can fail returns a ``Result`` (ok / message / data). It does not raise,
and when it rejects something it leaves the data alone. Bad constructor parameters (N, V, ...)
do raise ValueError, since that is a program setup mistake and not guest data.
"""

from __future__ import annotations

from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .guest import Guest
from .guest_store import DuplicateGuestError, GuestStore
from .hash_ring import HashFn, HashRing
from .hashing import hash64
from .migration import METHOD_RANGE, METHOD_SCAN, MigrationRecord, MigrationReport
from .result import Result
from .stats import LoadStats, compute_load_stats
from .validation import is_non_negative_int, is_positive_int

Mover = Tuple[Guest, int]  # (guest, new building)

# Upper limits so a typo (an extra zero or two) can't hang the machine or eat all the memory.
# The spec says the size per command depends on the machine; these keep us around 1 GiB
# (roughly 450 bytes per guest).
MAX_GUESTS_PER_COMMAND = 2_000_000
MAX_RING_POINTS = 5_000_000


class HotelSystem:
    def __init__(
        self,
        v: int = 32,
        salt: object = "",
        migration_method: str = METHOD_RANGE,
        hash_fn: HashFn = hash64,
    ) -> None:
        self._ring = HashRing(v, salt, hash_fn)
        self._store = GuestStore()
        self._check_method(migration_method)
        self.migration_method = migration_method
        self.last_migration: Optional[MigrationReport] = None

    # ---------------------------------------------------------------- setup (function 1)
    @classmethod
    def initialize(
        cls,
        n: int,
        v: int,
        channels: int,
        guests_per_channel: Union[int, Sequence[int]],
        salt: object = "",
        migration_method: str = METHOD_RANGE,
        hash_fn: HashFn = hash64,
    ) -> "HotelSystem":
        """Start a system: buildings 1..N, V points per building, `channels` channels
        (the sequence of each channel starts at 1).

        `guests_per_channel` is an int (same for every channel) or a list of length `channels`.
        Two separate steps as in spec 8 item 1: (1) build the ring, (2) place the initial guests.
        """
        if not is_positive_int(n):
            raise ValueError(f"N ต้องเป็นจำนวนเต็มบวก แต่ได้ {n!r}")
        if not is_non_negative_int(channels):
            raise ValueError(f"จำนวนช่องทางต้องเป็นจำนวนเต็มไม่ติดลบ แต่ได้ {channels!r}")
        counts = cls._expand_counts(channels, guests_per_channel)
        if sum(counts) > MAX_GUESTS_PER_COMMAND:
            raise ValueError(f"จำนวนแขกเริ่มต้นรวม {sum(counts):,} คน เกินเพดาน {MAX_GUESTS_PER_COMMAND:,} คน")
        if is_positive_int(v) and n * v > MAX_RING_POINTS:
            raise ValueError(f"จำนวนจุดบนวงแหวน N x V = {n * v:,} เกินเพดาน {MAX_RING_POINTS:,} จุด")
        system = cls(v, salt, migration_method, hash_fn)
        system._ring.add_nodes(range(1, n + 1))
        for c, count in enumerate(counts, start=1):
            if count:
                result = system.add_guests(c, 1, count)
                if not result.ok:  # never end up with a system that quietly lost guests
                    raise ValueError(result.message)
        return system

    @staticmethod
    def _expand_counts(channels: int, guests_per_channel: Union[int, Sequence[int]]) -> List[int]:
        if is_non_negative_int(guests_per_channel):
            return [guests_per_channel] * channels  # type: ignore[list-item]
        try:
            counts = list(guests_per_channel)  # type: ignore[arg-type]
        except TypeError:
            raise ValueError("จำนวนแขกต่อช่องทางต้องเป็นจำนวนเต็มหรือ list ของจำนวนเต็ม") from None
        if len(counts) != channels or not all(is_non_negative_int(x) for x in counts):
            raise ValueError("list จำนวนแขกต้องยาวเท่าจำนวนช่องทางและเป็นจำนวนเต็มไม่ติดลบทุกค่า")
        return counts

    @staticmethod
    def _check_method(method: str) -> None:
        if method not in (METHOD_SCAN, METHOD_RANGE):
            raise ValueError(f"วิธีหาผู้ย้ายต้องเป็น {METHOD_SCAN!r} หรือ {METHOD_RANGE!r} แต่ได้ {method!r}")

    # ---------------------------------------------------------------- basic info
    @property
    def ring(self) -> HashRing:
        return self._ring

    @property
    def store(self) -> GuestStore:
        return self._store

    @property
    def guest_count(self) -> int:
        """K"""
        return len(self._store)

    @property
    def building_count(self) -> int:
        """N"""
        return self._ring.node_count

    def buildings(self) -> List[int]:
        return self._ring.node_ids()

    # ---------------------------------------------------------------- add guests (function 2)
    def add_guests(self, c: int, s_start: int, count: int) -> Result:
        """Add a group of guests: channel c, sequence s_start .. s_start+count-1.

        count = 0 changes nothing. If the id range overlaps even one existing guest the whole
        group is rejected and existing guests are not touched, so their room addresses stay.
        """
        if not (is_positive_int(c) and is_positive_int(s_start)):
            return Result.failure(f"รหัสช่องทางและลำดับเริ่มต้นต้องเป็นจำนวนเต็มบวก แต่ได้ ({c!r}, {s_start!r})")
        if not is_non_negative_int(count):
            return Result.failure(f"จำนวนแขกต้องเป็นจำนวนเต็มไม่ติดลบ แต่ได้ {count!r}")
        if self._ring.node_count == 0:
            return Result.failure("ยังไม่มีอาคารในระบบ เพิ่มแขกไม่ได้")
        if count > MAX_GUESTS_PER_COMMAND:
            return Result.failure(
                f"จำนวนแขก {count:,} คนต่อคำสั่งเกินเพดาน {MAX_GUESTS_PER_COMMAND:,} คน ปฏิเสธ ข้อมูลเดิมไม่เปลี่ยน "
                f"(ตรวจว่าพิมพ์ตัวเลขถูกต้อง หรือแบ่งเพิ่มเป็นหลายครั้ง)"
            )
        if count == 0:
            return Result.success("จำนวนแขก = 0 ไม่มีการเปลี่ยนแปลงข้อมูล", 0)

        duplicates = self._store.find_duplicates(c, s_start, count)
        if duplicates:
            return Result.failure(
                f"ช่วงรหัสทับกับแขกที่มีอยู่ {len(duplicates)} คน (เช่น {duplicates[0]}) "
                f"ปฏิเสธทั้งกลุ่ม {count} คน ข้อมูลเดิมไม่เปลี่ยน",
                duplicates,
            )

        ring = self._ring
        locate = ring.locate
        position_of = ring.guest_position
        entries = []
        for s in range(s_start, s_start + count):
            pos = position_of(c, s)
            entries.append((c, s, pos, locate(pos)))
        try:
            self._store.add_many(entries)
        except DuplicateGuestError as exc:  # second line of defence, normally caught above
            return Result.failure(f"{exc} ปฏิเสธทั้งกลุ่ม ข้อมูลเดิมไม่เปลี่ยน")
        return Result.success(f"เพิ่มแขก {count} คน (ช่องทาง {c}, ลำดับ {s_start}..{s_start + count - 1})", count)

    def add_guest(self, c: int, s: int) -> Result:
        """Add one guest. `data` of the result is the Guest, with its address filled in."""
        result = self.add_guests(c, s, 1)
        if not result.ok:
            return result
        guest = self._store.get(c, s)
        return Result.success(
            f"แขก ({c}, {s}) เข้าพักอาคาร {guest.node_id} ห้อง {guest.room_no}", guest
        )

    # ---------------------------------------------------------------- remove guests (function 3)
    def remove_guest(self, c: int, s: int) -> Result:
        """Check a guest out = delete their stay record (no replacement room is made).
        If not found, say so and leave everything else alone."""
        if not (is_positive_int(c) and is_positive_int(s)):
            return Result.failure(f"รหัสแขกต้องเป็นจำนวนเต็มบวก แต่ได้ ({c!r}, {s!r})")
        guest = self._store.remove(c, s)
        if guest is None:
            return Result.failure(f"ไม่พบแขกรหัส ({c}, {s}) ไม่มีการเปลี่ยนแปลง")
        return Result.success(
            f"นำแขก ({c}, {s}) ออกจากอาคาร {guest.node_id} ห้อง {guest.room_no}", guest
        )

    # ---------------------------------------------------------------- search (function 5)
    def find_by_guest(self, c: int, s: int) -> Result:
        """Search by guest id -> building and room number."""
        guest = self._store.get(c, s) if (is_positive_int(c) and is_positive_int(s)) else None
        if guest is None:
            return Result.failure(f"ไม่พบแขกรหัส ({c}, {s})")
        return Result.success(f"แขก ({c}, {s}) อยู่อาคาร {guest.node_id} ห้อง {guest.room_no}", guest)

    def find_by_room(self, node_id: int, room: int) -> Result:
        """Search by (node_id, room_no) -> guest id."""
        if not (is_non_negative_int(node_id) and is_positive_int(room)):
            return Result.failure(f"ไม่พบผู้พักที่อาคาร {node_id!r} ห้อง {room!r} (รหัสอาคารต้องเป็นจำนวนเต็มไม่ติดลบ เลขห้องต้องเป็นจำนวนเต็มบวก)")
        guest = self._store.get_by_room(node_id, room)
        if guest is None:
            return Result.failure(f"ไม่พบผู้พักที่อาคาร {node_id} ห้อง {room}")
        return Result.success(
            f"อาคาร {node_id} ห้อง {room} คือแขก ({guest.c}, {guest.s})", guest
        )

    def compute_building(self, c: int, s: int) -> int:
        """Building of a guest worked out purely from the current ring (ignores stored data). O(log M).

        Raises ValueError for a bad id and LookupError if the ring is empty.
        """
        if not (is_positive_int(c) and is_positive_int(s)):
            raise ValueError(f"รหัสแขกต้องเป็นจำนวนเต็มบวก แต่ได้ ({c!r}, {s!r})")
        return self._ring.locate_guest(c, s)

    # ---------------------------------------------------------------- listing / load (functions 6, 7)
    def occupied_rooms(self) -> List[Guest]:
        """Occupied rooms, sorted by node_id then room_no (numeric)."""
        return self._store.sorted_rooms()

    def load_counts(self) -> Dict[int, int]:
        """Guests in every building, including the ones with 0."""
        count_in = self._store.count_in
        return {nid: count_in(nid) for nid in self._ring.node_ids()}

    def load_report(self) -> LoadStats:
        return compute_load_stats(self.load_counts())

    # ---------------------------------------------------------------- add / remove building (function 4)
    def add_building(self, node_id: int, method: Optional[str] = None) -> Result:
        """Add a building. Movers may only move into the new building. `data` is a MigrationReport."""
        method = self._pick_method(method)
        if not is_non_negative_int(node_id):
            return Result.failure(f"รหัสอาคารต้องเป็นจำนวนเต็มไม่ติดลบ แต่ได้ {node_id!r}")
        if self._ring.has_node(node_id):
            return Result.failure(f"อาคาร {node_id} มีอยู่แล้ว ไม่มีการเปลี่ยนแปลง")

        k_before = len(self._store)
        n_before = self._ring.node_count
        timings = {"ring_update": 0.0, "find": 0.0, "apply": 0.0}

        t = perf_counter()
        self._ring.add_node(node_id)  # already checked it's new, so this can't fail
        timings["ring_update"] = perf_counter() - t

        t = perf_counter()
        if method == METHOD_SCAN:
            movers = self._scan_for_movers()
        else:
            arcs = self._ring.arcs_of(node_id)
            movers = [(g, node_id) for g in self._store.guests_in_arcs(arcs) if g.node_id != node_id]
        timings["find"] = perf_counter() - t

        t = perf_counter()
        records = self._apply_moves(movers)
        timings["apply"] = perf_counter() - t

        report = MigrationReport(
            "add_building", node_id, method, n_before, n_before + 1, k_before, records, timings
        )
        self.last_migration = report
        return Result.success(report.summary(), report)

    def remove_building(self, node_id: int, method: Optional[str] = None) -> Result:
        """Remove a building. Guests in other buildings must not move. `data` is a MigrationReport."""
        method = self._pick_method(method)
        if not is_non_negative_int(node_id) or not self._ring.has_node(node_id):
            return Result.failure(f"ไม่พบอาคาร {node_id!r} ไม่มีการเปลี่ยนแปลง")
        if self._ring.node_count == 1:
            return Result.failure(
                f"ลบอาคาร {node_id} ไม่ได้ เพราะเป็นอาคารสุดท้าย (ระบบต้องมีอย่างน้อย 1 อาคาร)"
            )

        k_before = len(self._store)
        n_before = self._ring.node_count
        timings = {"ring_update": 0.0, "find": 0.0, "apply": 0.0}

        if method == METHOD_RANGE:
            # get this building's ranges *before* removing its points, afterwards nobody owns them
            t = perf_counter()
            arcs = self._ring.arcs_of(node_id)
            candidates = self._store.guests_in_arcs(arcs)
            timings["find"] += perf_counter() - t

        t = perf_counter()
        self._ring.remove_node(node_id)  # already checked it exists and isn't the last one
        timings["ring_update"] = perf_counter() - t

        t = perf_counter()
        if method == METHOD_SCAN:
            movers = self._scan_for_movers()
        else:
            locate = self._ring.locate
            movers = [(g, locate(g.pos)) for g in candidates if g.node_id == node_id]
        timings["find"] += perf_counter() - t

        t = perf_counter()
        records = self._apply_moves(movers)
        timings["apply"] = perf_counter() - t

        report = MigrationReport(
            "remove_building", node_id, method, n_before, n_before - 1, k_before, records, timings
        )
        self.last_migration = report
        return Result.success(report.summary(), report)

    def _pick_method(self, method: Optional[str]) -> str:
        method = method or self.migration_method
        self._check_method(method)
        return method

    def _scan_for_movers(self) -> List[Mover]:
        """Method A: work out a new building for every guest and compare with the old one. O(K log M)."""
        locate = self._ring.locate
        movers: List[Mover] = []
        for guest in self._store.guests():
            new_node = locate(guest.pos)
            if new_node != guest.node_id:
                movers.append((guest, new_node))
        return movers

    def _apply_moves(self, movers: List[Mover]) -> List[MigrationRecord]:
        """Move the guests in the store (room_no unchanged) and build the records, sorted by guest id."""
        move = self._store.move
        records: List[MigrationRecord] = []
        for guest, new_node in movers:
            records.append(MigrationRecord(guest.c, guest.s, guest.node_id, new_node, guest.room_no))
            move(guest.c, guest.s, new_node)
        records.sort(key=lambda r: (r.channel_id, r.sequence_id))
        return records

    # ---------------------------------------------------------------- consistency / overview
    def clone(self) -> "HotelSystem":
        """Copy the whole system (ring + guests + indexes) so we can try adding/removing
        buildings without touching the real one."""
        other = HotelSystem.__new__(HotelSystem)
        other._ring = self._ring.copy()
        other._store = self._store.copy()
        other.migration_method = self.migration_method
        other.last_migration = None
        return other

    def snapshot(self) -> Dict[Tuple[int, int], Tuple[int, int]]:
        """(c, s) -> (node_id, room_no) for every guest. Used to compare before / after in tests."""
        return {(g.c, g.s): (g.node_id, g.room_no) for g in self._store.guests()}

    def check_consistency(self, limit: int = 20) -> List[str]:
        """Check that the store, indexes and ring agree (empty = fine).

        Every guest must be in the building you get by recomputing from the current ring.
        """
        problems = list(self._store.validate())
        ring = self._ring
        for guest in self._store.guests():
            if ring.guest_position(guest.c, guest.s) != guest.pos:
                problems.append(f"ตำแหน่งแฮชของ ({guest.c}, {guest.s}) ไม่ตรงกับวงแหวน")
            if ring.locate(guest.pos) != guest.node_id:
                problems.append(f"แขก ({guest.c}, {guest.s}) อยู่อาคาร {guest.node_id} แต่วงแหวนชี้ไปอีกอาคาร")
            if len(problems) >= limit:
                problems.append("... (ตัดรายการปัญหา)")
                break
        return problems
