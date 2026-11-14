"""Guest records plus the lookup indexes.

Four structures that must always agree after any add / remove / move:
    _by_id    dict  (c, s)             -> Guest   lookup by guest id, O(1)
    _by_room  dict  (node_id, room_no) -> Guest   lookup by room address, O(1)
    _by_pos   sorted list of (pos, c, s)          pulls guests in a ring range (method B) with bisect
    _count    dict  node_id -> guests              load report without walking every guest

Every method that changes data has to update all four. `validate` checks that (used in tests).
"""

from __future__ import annotations

from bisect import bisect_left, insort
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .guest import Guest
from .room import room_no

GuestId = Tuple[int, int]
Entry = Tuple[int, int, int, int]  # (c, s, pos, node_id)


class DuplicateGuestError(ValueError):
    """Duplicate guest id (against existing guests or inside the batch). Nothing was changed."""

    def __init__(self, guest_id: GuestId) -> None:
        super().__init__(f"รหัสแขก {guest_id} ซ้ำ")
        self.guest_id = guest_id


class GuestStore:
    def __init__(self) -> None:
        self._by_id: Dict[GuestId, Guest] = {}
        self._by_room: Dict[Tuple[int, int], Guest] = {}
        self._by_pos: List[Tuple[int, int, int]] = []
        self._count: Dict[int, int] = {}

    # ---------------------------------------------------------------- read
    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, guest_id: object) -> bool:
        return guest_id in self._by_id

    def get(self, c: int, s: int) -> Optional[Guest]:
        return self._by_id.get((c, s))

    def get_by_room(self, node_id: int, room: int) -> Optional[Guest]:
        return self._by_room.get((node_id, room))

    def guests(self) -> Iterator[Guest]:
        return iter(self._by_id.values())

    def count_in(self, node_id: int) -> int:
        return self._count.get(node_id, 0)

    def counts(self) -> Dict[int, int]:
        """Guests per building (only buildings that have at least one guest)."""
        return dict(self._count)

    def find_duplicates(self, c: int, s_start: int, count: int) -> List[GuestId]:
        """Ids in (c, s_start .. s_start+count-1) that already exist. O(count)."""
        by_id = self._by_id
        return [(c, s) for s in range(s_start, s_start + count) if (c, s) in by_id]

    def copy(self) -> "GuestStore":
        """Deep copy, O(K). Every Guest is rebuilt and every index points at the new objects,
        so changing the copy never touches the original."""
        other = GuestStore()
        for guest in self._by_id.values():
            clone = Guest(guest.c, guest.s, guest.pos, guest.node_id, guest.room_no)
            other._by_id[(clone.c, clone.s)] = clone
            other._by_room[(clone.node_id, clone.room_no)] = clone
        other._by_pos = list(self._by_pos)
        other._count = dict(self._count)
        return other

    def sorted_rooms(self) -> List[Guest]:
        """Guests sorted by node_id, then by room_no (numeric). O(K log K)."""
        return sorted(self._by_id.values(), key=lambda g: (g.node_id, g.room_no))

    # ---------------------------------------------------------------- add
    def add_many(self, entries: Iterable[Entry]) -> int:
        """Add a batch of (c, s, pos, node_id), all or nothing.

        All duplicate checks happen before anything is changed; on a duplicate we raise
        DuplicateGuestError and the old data is untouched. The room number only depends on
        the guest id (`room_no`), never on "last number + 1".
        Cost: O(b) for the dicts, O(b log b) to sort the new ones, then O(K + b) to fold them
        into the position index (list.sort on two sorted runs is linear), or O(log K + K)
        when b = 1.
        """
        batch = list(entries)
        seen = set()
        rooms: List[int] = []
        for c, s, _pos, _node in batch:
            key = (c, s)
            if key in self._by_id or key in seen:
                raise DuplicateGuestError(key)
            seen.add(key)
            rooms.append(room_no(c, s))  # raises ValueError on a bad id, still before any change

        new_pos: List[Tuple[int, int, int]] = []
        for (c, s, pos, node_id), room in zip(batch, rooms):
            guest = Guest(c, s, pos, node_id, room)
            self._by_id[(c, s)] = guest
            self._by_room[(node_id, guest.room_no)] = guest
            self._count[node_id] = self._count.get(node_id, 0) + 1
            new_pos.append((pos, c, s))

        if len(new_pos) == 1:
            insort(self._by_pos, new_pos[0])
        elif new_pos:
            new_pos.sort()
            self._by_pos.extend(new_pos)
            self._by_pos.sort()
        return len(batch)

    # ---------------------------------------------------------------- remove
    def remove(self, c: int, s: int) -> Optional[Guest]:
        """Remove one guest from every index. Returns the removed Guest, or None if not found."""
        guest = self._by_id.pop((c, s), None)
        if guest is None:
            return None
        del self._by_room[(guest.node_id, guest.room_no)]
        self._dec(guest.node_id)
        i = bisect_left(self._by_pos, (guest.pos, c, s))
        if i >= len(self._by_pos) or self._by_pos[i] != (guest.pos, c, s):
            raise RuntimeError("ดัชนีตำแหน่งไม่สอดคล้องกับข้อมูลแขก")
        del self._by_pos[i]
        return guest

    # ---------------------------------------------------------------- move
    def move(self, c: int, s: int, new_node: int) -> bool:
        """Move a guest to another building. Only node_id changes, room_no stays.

        _by_room (its key has node_id) and _count need fixing. _by_id and _by_pos don't,
        because the guest id and hash position never change. Returns False if the guest
        is already in that building.
        """
        guest = self._by_id[(c, s)]
        if guest.node_id == new_node:
            return False
        del self._by_room[(guest.node_id, guest.room_no)]
        self._dec(guest.node_id)
        guest.node_id = new_node
        self._by_room[(new_node, guest.room_no)] = guest
        self._count[new_node] = self._count.get(new_node, 0) + 1
        return True

    def _dec(self, node_id: int) -> None:
        left = self._count[node_id] - 1
        if left:
            self._count[node_id] = left
        else:
            del self._count[node_id]

    # ---------------------------------------------------------------- ring ranges (method B)
    def guests_in_arcs(self, arcs: Sequence[Tuple[int, int]]) -> List[Guest]:
        """Guests whose hash position falls in any closed range [lo, hi].

        O(A log K + r) with A = number of ranges and r = guests found. Guests outside
        the ranges are never touched.
        """
        by_pos = self._by_pos
        by_id = self._by_id
        found: List[Guest] = []
        for lo, hi in arcs:
            start = bisect_left(by_pos, (lo,))
            stop = bisect_left(by_pos, (hi + 1,))
            for k in range(start, stop):
                _pos, c, s = by_pos[k]
                found.append(by_id[(c, s)])
        return found

    # ---------------------------------------------------------------- consistency check
    def validate(self) -> List[str]:
        """List of mismatches between the indexes (empty = all good)."""
        problems: List[str] = []
        n = len(self._by_id)
        if len(self._by_room) != n:
            problems.append(f"by_room มี {len(self._by_room)} รายการ แต่ by_id มี {n}")
        if len(self._by_pos) != n:
            problems.append(f"by_pos มี {len(self._by_pos)} รายการ แต่ by_id มี {n}")
        if sum(self._count.values()) != n:
            problems.append(f"count รวม {sum(self._count.values())} แต่ by_id มี {n}")
        if self._by_pos != sorted(self._by_pos):
            problems.append("by_pos ไม่เรียงตามตำแหน่ง")
        recount: Dict[int, int] = {}
        for key, guest in self._by_id.items():
            if key != (guest.c, guest.s):
                problems.append(f"คีย์ {key} ไม่ตรงกับแขก {(guest.c, guest.s)}")
            if self._by_room.get((guest.node_id, guest.room_no)) is not guest:
                problems.append(f"by_room ไม่ชี้ถึงแขก {key}")
            if guest.room_no != room_no(guest.c, guest.s):
                problems.append(f"เลขห้องของ {key} ไม่ตรงกับสูตร")
            recount[guest.node_id] = recount.get(guest.node_id, 0) + 1
        if recount != self._count:
            problems.append("count รายอาคารไม่ตรงกับข้อมูลแขกจริง")
        for pos, c, s in self._by_pos:
            guest = self._by_id.get((c, s))
            if guest is None or guest.pos != pos:
                problems.append(f"by_pos มีรายการที่ไม่ตรงกับแขก {(c, s)}")
        return problems
