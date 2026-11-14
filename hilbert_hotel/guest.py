"""Data held for a single guest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(slots=True)
class Guest:
    """One guest.

    c, s      guest id (channel, sequence in the channel), positive ints
    pos       64-bit hash position on the ring, from c, s and salt, so it never changes for a guest
    node_id   building the guest is in right now (worked out from the ring, kept as a cache)
    room_no   room number, from c and s only, so it doesn't change when the guest moves
    """

    c: int
    s: int
    pos: int
    node_id: int
    room_no: int

    @property
    def guest_id(self) -> Tuple[int, int]:
        return (self.c, self.s)

    @property
    def address(self) -> Tuple[int, int]:
        return (self.node_id, self.room_no)
