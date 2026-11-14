"""Hash that gives the same value on every run (spec 4.3).

* SHA-256 of the UTF-8 text, first 8 bytes (64 bits) read as a big-endian int -> 0 .. 2**64 - 1.
* No Python ``hash()``, it changes between runs (hash randomization).
* What gets hashed keeps guests and virtual nodes apart:
      guest          ->  "guest:<c>:<s>"
      virtual node   ->  "node:<node_id>:<j>"      (j = 0 .. V-1)
* The salt shifts every ring position between experiment rounds, by prefixing the text:
      empty salt      ->  text as is
      non-empty salt  ->  "<salt>|<text>"
"""

from __future__ import annotations

import hashlib

RING_BITS = 64
RING_SIZE = 1 << RING_BITS  # ring positions are 0 .. RING_SIZE - 1
MAX_POSITION = RING_SIZE - 1


def build_message(text: str, salt: object = "") -> str:
    """Text to hash, with the salt in front if there is one."""
    salt_text = str(salt)
    return f"{salt_text}|{text}" if salt_text else text


def hash64(text: str, salt: object = "") -> int:
    """64-bit position of the text (same on every run)."""
    digest = hashlib.sha256(build_message(text, salt).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def guest_key(c: int, s: int) -> str:
    return f"guest:{c}:{s}"


def vnode_key(node_id: int, j: int) -> str:
    return f"node:{node_id}:{j}"
