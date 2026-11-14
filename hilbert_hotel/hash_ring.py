"""Consistent hashing ring with virtual nodes (spec 4.3 and 4.4).

What it keeps:
    _points  sorted list of (position, node_id, j). Sorting on the whole tuple gives the
             tie-break rule from 4.3 item 4 for free. We bisect on it, O(log M) with M = N * V.
    _nodes   set of node ids, so duplicate / missing checks are O(1)

A guest at position g goes to the first point with position >= g. If g is past the last
point we wrap around to the first one. Points with the same position are ordered by
(position, node_id, j), first one wins.
"""

from __future__ import annotations

from bisect import bisect_left
from typing import Callable, Iterable, List, Set, Tuple

from .hashing import MAX_POSITION, guest_key, hash64, vnode_key
from .result import Result
from .validation import is_non_negative_int, is_positive_int

Point = Tuple[int, int, int]  # (position, node_id, j)
Arc = Tuple[int, int]  # closed range [lo, hi], never crosses the wrap-around point
HashFn = Callable[[str, object], int]


def _merge_sorted(a: List[Point], b: List[Point]) -> List[Point]:
    """Merge two sorted lists into one sorted list, O(len(a) + len(b))."""
    merged: List[Point] = []
    i = j = 0
    la, lb = len(a), len(b)
    while i < la and j < lb:
        if a[i] <= b[j]:
            merged.append(a[i])
            i += 1
        else:
            merged.append(b[j])
            j += 1
    if i < la:
        merged.extend(a[i:])
    if j < lb:
        merged.extend(b[j:])
    return merged


class HashRing:
    def __init__(self, v: int, salt: object = "", hash_fn: HashFn = hash64) -> None:
        if not is_positive_int(v):
            raise ValueError(f"V ต้องเป็นจำนวนเต็มบวก แต่ได้ {v!r}")
        self.v = v
        self.salt = str(salt)
        self._hash_fn = hash_fn
        self._points: List[Point] = []
        self._nodes: Set[int] = set()

    # ---------------------------------------------------------------- basic info
    @property
    def node_count(self) -> int:
        """N"""
        return len(self._nodes)

    @property
    def point_count(self) -> int:
        """M = N * V"""
        return len(self._points)

    @property
    def hash_fn(self) -> HashFn:
        return self._hash_fn

    def has_node(self, node_id: int) -> bool:
        return node_id in self._nodes

    def node_ids(self) -> List[int]:
        return sorted(self._nodes)

    def points(self) -> List[Point]:
        """Copy of all points (sorted), for printing and tests."""
        return list(self._points)

    def copy(self) -> "HashRing":
        """O(M) copy. The point tuples are immutable so they can be shared."""
        other = HashRing(self.v, self.salt, self._hash_fn)
        other._points = list(self._points)
        other._nodes = set(self._nodes)
        return other

    # ---------------------------------------------------------------- positions
    def guest_position(self, c: int, s: int) -> int:
        return self._hash_fn(guest_key(c, s), self.salt)

    def _make_points(self, node_id: int) -> List[Point]:
        return sorted(
            (self._hash_fn(vnode_key(node_id, j), self.salt), node_id, j)
            for j in range(self.v)
        )

    # ---------------------------------------------------------------- add / remove
    def add_nodes(self, node_ids: Iterable[int]) -> Result:
        """Add several buildings at once, all or nothing.

        Only the new buildings' virtual nodes are added, existing points are not touched
        (4.4 item 1). Cost: build n*V new points and sort them, O(nV log nV), then merge
        with the old list in O(M + nV).
        """
        ids = list(node_ids)
        seen: Set[int] = set()
        for nid in ids:
            if not is_non_negative_int(nid):
                return Result.failure(f"รหัสอาคารต้องเป็นจำนวนเต็มไม่ติดลบ แต่ได้ {nid!r}")
            if nid in self._nodes or nid in seen:
                return Result.failure(f"อาคาร {nid} มีอยู่แล้ว ไม่มีการเปลี่ยนแปลง")
            seen.add(nid)
        new_points: List[Point] = []
        for nid in ids:
            new_points.extend(self._make_points(nid))
        new_points.sort()
        self._points = _merge_sorted(self._points, new_points)
        self._nodes.update(seen)
        return Result.success(f"เพิ่ม {len(ids)} อาคาร ({len(new_points)} จุด)", len(new_points))

    def add_node(self, node_id: int) -> Result:
        return self.add_nodes([node_id])

    def remove_node(self, node_id: int) -> Result:
        """Drop every virtual node of this building. Other points stay put. O(M)."""
        if not is_non_negative_int(node_id) or node_id not in self._nodes:
            return Result.failure(f"ไม่พบอาคาร {node_id!r} ไม่มีการเปลี่ยนแปลง")
        if len(self._nodes) == 1:
            return Result.failure(
                f"ลบอาคาร {node_id} ไม่ได้ เพราะเป็นอาคารสุดท้าย (วงแหวนต้องมีอย่างน้อย 1 อาคาร)"
            )
        before = len(self._points)
        self._points = [p for p in self._points if p[1] != node_id]
        self._nodes.discard(node_id)
        return Result.success(f"ลบอาคาร {node_id}", before - len(self._points))

    # ---------------------------------------------------------------- lookup
    def locate(self, position: int) -> int:
        """Return the node that owns `position`: first point >= position, wrapping to the start.

        O(log M) with bisect.
        """
        points = self._points
        if not points:
            raise LookupError("วงแหวนว่าง ไม่มีอาคาร")
        i = bisect_left(points, (position,))  # (p,) sorts before any (p, node, j)
        if i == len(points):
            i = 0
        return points[i][1]

    def locate_guest(self, c: int, s: int) -> int:
        """Building of a guest, worked out from the id and the current ring only (not from stored data)."""
        return self.locate(self.guest_position(c, s))

    # ---------------------------------------------------------------- ranges owned by a building
    def arcs_of(self, node_id: int, max_position: int = MAX_POSITION) -> List[Arc]:
        """Position ranges for which `locate` returns this node, as closed [lo, hi] ranges.

        The point at index i owns (points[i-1].pos, points[i].pos]. If points[i-1] has the same
        position (a collision) point i owns nothing, the first one of the group wins. The very
        first point also owns the wrap-around part (points[-1].pos, max_position].
        O(M) because we walk every point to find this building's ones.
        """
        points = self._points
        arcs: List[Arc] = []
        for i, (pos, nid, _j) in enumerate(points):
            if nid != node_id:
                continue
            if i == 0:
                arcs.append((0, pos))
                last = points[-1][0]
                if last < max_position:
                    arcs.append((last + 1, max_position))
            else:
                prev = points[i - 1][0]
                if prev == pos:
                    continue  # collision, the first point of the group already took it
                arcs.append((prev + 1, pos))
        return arcs
