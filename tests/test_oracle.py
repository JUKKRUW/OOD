"""Compare against a plain reference implementation written separately.

The reference code in this file uses no bisect, no indexes and none of the project's data
structures, only the hash function. It follows the definition in the spec directly (sort all
the points, take the first one with position >= the guest's position), so it can catch logic
mistakes that both migration methods (A and B) share, which comparing A with B would never see.
"""

import random
import unittest

from hilbert_hotel import HotelSystem
from hilbert_hotel.hash_ring import HashRing
from hilbert_hotel.hashing import guest_key, hash64, vnode_key
from hilbert_hotel.room import guest_from_room, room_no

from tests.helpers import small_mod_hash


def naive_owner(node_ids, v, salt, hash_fn, position):
    points = sorted((hash_fn(vnode_key(n, j), salt), n, j) for n in node_ids for j in range(v))
    at_or_after = [p for p in points if p[0] >= position]
    return (at_or_after[0] if at_or_after else points[0])[1]  # no point >= position: wrap to the first one


def hash_function(modulus):
    return small_mod_hash(modulus) if modulus else hash64


class CantorOracleTests(unittest.TestCase):
    def test_formula_equals_walking_the_diagonals(self):
        # count rooms diagonal by diagonal: diagonal w has w+1 ids, ordered by b ascending
        room = 0
        for w in range(0, 150):
            for b in range(0, w + 1):
                room += 1
                a = w - b
                self.assertEqual(room_no(a + 1, b + 1), room)
                self.assertEqual(guest_from_room(room), (a + 1, b + 1))


class RingOracleTests(unittest.TestCase):
    def test_locate_equals_naive_definition(self):
        rng = random.Random(20260514)
        for trial in range(150):
            modulus = rng.choice([3, 7, 16, 50, 1000, None])
            fn = hash_function(modulus)
            v = rng.randint(1, 6)
            salt = str(rng.randint(0, 5))
            ids = rng.sample(range(0, 40), rng.randint(1, 9))
            ring = HashRing(v, salt, fn)
            self.assertTrue(ring.add_nodes(ids).ok)
            top = modulus or 2**64
            probes = [0, top - 1, top, top + 5] + [rng.randrange(0, top) for _ in range(40)]
            for pos, _node, _j in ring.points():  # positions exactly on a point, just before it and just after it
                probes += [pos, pos + 1, max(0, pos - 1)]
            for pos in probes:
                self.assertEqual(ring.locate(pos), naive_owner(ids, v, salt, fn, pos),
                                 f"trial={trial} modulus={modulus} v={v} ids={ids} pos={pos}")


class MigrationOracleTests(unittest.TestCase):
    def test_every_command_matches_naive_owners_and_movers(self):
        rng = random.Random(77)
        for trial in range(25):
            modulus = rng.choice([5, 11, 40, None])
            fn = hash_function(modulus)
            v = rng.randint(1, 4)
            salt = str(trial % 3)
            system = HotelSystem(v=v, salt=salt, migration_method=rng.choice(["scan", "range"]), hash_fn=fn)
            ids = set(rng.sample(range(0, 20), rng.randint(1, 5)))
            system.ring.add_nodes(sorted(ids))
            system.add_guests(1, 1, rng.randint(0, 60))
            system.add_guests(2, 1, rng.randint(0, 60))
            for step in range(30):
                before = {(g.c, g.s): g.node_id for g in system.store.guests()}
                node = rng.randint(0, 24)
                if rng.random() < 0.5:
                    result = system.add_building(node)
                    new_ids = ids | {node}
                else:
                    result = system.remove_building(node)
                    new_ids = ids - {node} if (node in ids and len(ids) > 1) else ids
                if not result.ok:
                    new_ids = ids  # rejected: the set of buildings doesn't change
                after = {(g.c, g.s): g.node_id for g in system.store.guests()}
                where = f"trial={trial} step={step}"
                for (c, s), node_after in after.items():  # the correct owner straight from the definition
                    expected = naive_owner(new_ids, v, salt, fn, fn(guest_key(c, s), salt))
                    self.assertEqual(node_after, expected, f"{where} guest=({c},{s})")
                if result.ok:
                    report = result.data
                    moved = {gid for gid in before if before[gid] != after[gid]}
                    self.assertEqual({r.guest_id for r in report.moved}, moved, where)
                    self.assertEqual(report.moved_count, len(moved), where)
                    self.assertEqual(report.guests_before, len(before), where)
                    ids = set(new_ids)
                else:
                    self.assertEqual(before, after, where)
                self.assertEqual(len(after), len(before), where)


if __name__ == "__main__":
    unittest.main()
