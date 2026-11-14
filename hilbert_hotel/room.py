"""Room number from a guest id with the Cantor pairing function (spec 4.2).

With a = c - 1 and b = s - 1 (both >= 0):

    room_no = (a + b) * (a + b + 1) // 2 + b + 1

Integers only (// is integer division), no floats anywhere.

Why two guests never get the same room (short version, full proof is in the report)
    Let w = a + b. Then w(w+1)/2 <= room_no - 1 = w(w+1)/2 + b <= w(w+1)/2 + w < (w+1)(w+2)/2,
    so the rooms of diagonal w sit in a half-open interval that no other diagonal touches.
    Given a room_no you can get w back in only one way, and then
    b = (room_no - 1) - w(w+1)/2 is fixed too.
    -> two different ids can't share a room number (one-to-one).
    ``guest_from_room`` is just this argument run backwards.
"""

from __future__ import annotations

from math import isqrt
from typing import Tuple

from .validation import is_positive_int


def room_no(c: int, s: int) -> int:
    """Room number of guest (c, s); c and s must be positive ints."""
    if not (is_positive_int(c) and is_positive_int(s)):
        raise ValueError(f"รหัสแขกต้องเป็นจำนวนเต็มบวก แต่ได้ ({c!r}, {s!r})")
    a = c - 1
    b = s - 1
    return ((a + b) * (a + b + 1)) // 2 + b + 1


def guest_from_room(room: int) -> Tuple[int, int]:
    """Inverse: room number -> the (c, s) that maps to it (handy for checking uniqueness)."""
    if not is_positive_int(room):
        raise ValueError(f"เลขห้องต้องเป็นจำนวนเต็มบวก แต่ได้ {room!r}")
    r = room - 1
    w = (isqrt(8 * r + 1) - 1) // 2  # the diagonal w with w(w+1)/2 <= r
    b = r - (w * (w + 1)) // 2
    a = w - b
    return a + 1, b + 1
