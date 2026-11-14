import hashlib
import os
import random
import subprocess
import sys
import unittest

from hilbert_hotel.hash_ring import HashRing
from hilbert_hotel.hashing import MAX_POSITION, RING_SIZE, guest_key, hash64, vnode_key

from tests.helpers import small_mod_hash, table_hash_fn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class HashingTests(unittest.TestCase):
    def test_matches_sha256_first_8_bytes_big_endian(self):
        digest = hashlib.sha256("guest:1:1".encode("utf-8")).digest()
        self.assertEqual(hash64("guest:1:1"), int.from_bytes(digest[:8], "big"))

    def test_golden_values_never_change(self):
        # these values must never change; if they do, the hashing method changed
        self.assertEqual(hash64("guest:1:1"), 3062031282815716176)
        self.assertEqual(hash64("guest:1:1", "0"), 3912152983899886038)
        self.assertEqual(hash64("node:1:0"), 3505680375118999729)
        self.assertEqual(hash64("node:10:31", "2"), 5399476667413166076)

    def test_position_range(self):
        for i in range(500):
            self.assertTrue(0 <= hash64(guest_key(1, i)) < RING_SIZE)
        self.assertEqual(MAX_POSITION, 2**64 - 1)

    def test_salt_prefix_rule(self):
        self.assertEqual(hash64("guest:1:1", ""), hash64("guest:1:1"))
        self.assertEqual(hash64("guest:1:1", "0"), hash64("0|guest:1:1"))
        self.assertEqual(hash64("guest:1:1", 0), hash64("guest:1:1", "0"))
        self.assertNotEqual(hash64("guest:1:1", "0"), hash64("guest:1:1", "1"))

    def test_guest_and_node_strings_are_distinct(self):
        self.assertEqual(guest_key(3, 4), "guest:3:4")
        self.assertEqual(vnode_key(3, 4), "node:3:4")
        self.assertNotEqual(hash64(guest_key(3, 4)), hash64(vnode_key(3, 4)))

    def test_same_value_in_new_processes_with_different_pythonhashseed(self):
        # after restarting the program the hash must be the same (Python's hash() would differ on every run)
        code = "from hilbert_hotel.hashing import hash64; print(hash64('guest:5:7', '1'))"
        outputs = set()
        for seed in ("0", "1", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                                  capture_output=True, text=True, check=True)
            outputs.add(proc.stdout.strip())
        self.assertEqual(outputs, {str(hash64("guest:5:7", "1"))})


def ring_with(positions, v=1):
    """A ring whose points for building nid sit where the {(nid, j): position} table says."""
    mapping = {vnode_key(nid, j): pos for (nid, j), pos in positions.items()}
    ring = HashRing(v, hash_fn=table_hash_fn(mapping))
    ring.add_nodes(sorted({nid for nid, _ in positions}))
    return ring


class LocateTests(unittest.TestCase):
    def setUp(self):
        # example 9.1: A=20, B=50, C=80
        self.ring = ring_with({(1, 0): 20, (2, 0): 50, (3, 0): 80})

    def test_exactly_on_a_point(self):
        self.assertEqual(self.ring.locate(20), 1)
        self.assertEqual(self.ring.locate(50), 2)
        self.assertEqual(self.ring.locate(80), 3)

    def test_between_points_goes_to_next_point(self):
        self.assertEqual(self.ring.locate(21), 2)
        self.assertEqual(self.ring.locate(51), 3)
        self.assertEqual(self.ring.locate(49), 2)

    def test_before_first_point(self):
        self.assertEqual(self.ring.locate(0), 1)
        self.assertEqual(self.ring.locate(10), 1)

    def test_after_last_point_wraps_to_first(self):
        self.assertEqual(self.ring.locate(81), 1)
        self.assertEqual(self.ring.locate(90), 1)
        self.assertEqual(self.ring.locate(MAX_POSITION), 1)

    def test_empty_ring_raises_lookup_error(self):
        with self.assertRaises(LookupError):
            HashRing(4).locate(5)

    def test_single_node_owns_everything(self):
        ring = ring_with({(7, 0): 40})
        for pos in (0, 39, 40, 41, MAX_POSITION):
            self.assertEqual(ring.locate(pos), 7)


class CollisionTests(unittest.TestCase):
    def test_colliding_points_are_all_kept_and_sorted_by_position_node_j(self):
        ring = ring_with({(5, 0): 40, (3, 0): 40, (9, 0): 10})
        self.assertEqual(ring.points(), [(10, 9, 0), (40, 3, 0), (40, 5, 0)])

    def test_guest_at_collided_position_uses_first_point_in_key_order(self):
        ring = ring_with({(5, 0): 40, (3, 0): 40, (9, 0): 10})
        self.assertEqual(ring.locate(40), 3)  # (40, 3, 0) comes before (40, 5, 0)
        self.assertEqual(ring.locate(11), 3)
        ring.remove_node(3)  # remove the first point of the group -> the second one takes over
        self.assertEqual(ring.locate(40), 5)
        self.assertEqual(ring.locate(11), 5)

    def test_virtual_nodes_of_same_node_colliding_ordered_by_j(self):
        mapping ={vnode_key(2, 0): 30, vnode_key(2, 1): 30, vnode_key(4, 0): 60, vnode_key(4, 1): 90}
        ring = HashRing(2, hash_fn=table_hash_fn(mapping))
        ring.add_nodes([4, 2])
        self.assertEqual(ring.points(), [(30, 2, 0), (30, 2, 1), (60, 4, 0), (90, 4, 1)])
        self.assertEqual(ring.locate(30), 2)
        self.assertEqual(ring.locate(91), 2)  # wraps around to the first point

    def test_many_guests_can_share_a_position(self):
        # several guests can hash to the same position and must go to the same building
        # (telling them apart by room number is the room formula's job, not the ring's)
        ring = ring_with({(1, 0): 20, (2, 0): 50})
        self.assertEqual({ring.locate(35) for _ in range(10)}, {2})


class MembershipTests(unittest.TestCase):
    def test_add_node_only_adds_its_own_points(self):
        ring = HashRing(8, salt="x")
        ring.add_nodes([1, 2, 3])
        before = ring.points()
        ring.add_node(4)
        after = ring.points()
        self.assertEqual(len(after), len(before) + 8)
        self.assertEqual([p for p in after if p[1] != 4], before)  # existing points don't move
        self.assertEqual(after, sorted(after))

    def test_remove_node_removes_all_its_points_and_nothing_else(self):
        ring = HashRing(8)
        ring.add_nodes([1, 2, 3])
        before = ring.points()
        self.assertTrue(ring.remove_node(2).ok)
        self.assertEqual(ring.points(), [p for p in before if p[1] != 2])
        self.assertEqual(ring.node_ids(), [1, 3])

    def test_add_then_remove_restores_ring(self):
        ring = HashRing(16, salt="s")
        ring.add_nodes([1, 2, 3, 4])
        before = ring.points()
        ring.add_node(9)
        ring.remove_node(9)
        self.assertEqual(ring.points(), before)

    def test_duplicate_add_rejected_without_change(self):
        ring = HashRing(4)
        ring.add_nodes([1, 2])
        before = ring.points()
        result = ring.add_node(2)
        self.assertFalse(result.ok)
        self.assertEqual(ring.points(), before)
        self.assertEqual(ring.point_count, 8)

    def test_batch_add_is_all_or_nothing(self):
        ring = HashRing(4)
        ring.add_nodes([1])
        before = ring.points()
        for bad in ([2, 1], [3, 3], [4, -1], [5, "x"], [6, True]):
            self.assertFalse(ring.add_nodes(bad).ok, bad)
            self.assertEqual(ring.points(), before)
            self.assertEqual(ring.node_ids(), [1])

    def test_remove_missing_node_rejected(self):
        ring = HashRing(4)
        ring.add_nodes([1, 2])
        before = ring.points()
        for bad in (99, -1, "1", None, 1.5):
            self.assertFalse(ring.remove_node(bad).ok)
        self.assertEqual(ring.points(), before)

    def test_cannot_remove_last_node(self):
        ring = HashRing(4)
        ring.add_nodes([1])
        self.assertFalse(ring.remove_node(1).ok)
        self.assertEqual(ring.node_ids(), [1])
        self.assertEqual(ring.point_count, 4)

    def test_invalid_v_rejected(self):
        for bad in (0, -3, 1.5, True, "8", None):
            with self.assertRaises(ValueError):
                HashRing(bad)

    def test_point_count_is_n_times_v(self):
        ring = HashRing(32)
        ring.add_nodes(range(1, 11))
        self.assertEqual((ring.node_count, ring.point_count), (10, 320))


class ArcTests(unittest.TestCase):
    """arcs_of must give every position to the same owner that locate returns (including collisions and wrap-around)."""

    def assert_arcs_match_locate(self, ring, max_position):
        owner_by_arc = {}
        for nid in ring.node_ids():
            for lo, hi in ring.arcs_of(nid, max_position):
                self.assertLessEqual(lo, hi)
                for pos in range(lo, hi + 1):
                    self.assertNotIn(pos, owner_by_arc, "ranges of two buildings overlap")
                    owner_by_arc[pos] = nid
        self.assertEqual(len(owner_by_arc), max_position + 1, "the ranges together don't cover the whole ring")
        for pos in range(max_position + 1):
            self.assertEqual(owner_by_arc[pos], ring.locate(pos), f"position {pos}")

    def test_simple_ring(self):
        ring = ring_with({(1, 0): 20, (2, 0): 50, (3, 0): 80})
        self.assertEqual(ring.arcs_of(1, 99), [(0, 20), (81, 99)])
        self.assertEqual(ring.arcs_of(2, 99), [(21, 50)])
        self.assertEqual(ring.arcs_of(3, 99), [(51, 80)])
        self.assert_arcs_match_locate(ring, 99)

    def test_ring_with_collisions(self):
        ring = ring_with({(5, 0): 40, (3, 0): 40, (9, 0): 10})
        self.assertEqual(ring.arcs_of(5, 99), [])  # the second point of a collision group owns nothing
        self.assert_arcs_match_locate(ring, 99)

    def test_random_rings_with_heavy_collisions(self):
        rng = random.Random(2569)
        for trial in range(40):
            modulus = rng.choice([8, 20, 50])
            v = rng.randint(1, 4)
            ring = HashRing(v, salt=str(trial), hash_fn=small_mod_hash(modulus))
            ring.add_nodes(rng.sample(range(0, 30), rng.randint(1, 8)))
            self.assert_arcs_match_locate(ring, modulus + 5)

    def test_single_point_covers_whole_ring(self):
        ring = ring_with({(1, 0): 40})
        self.assertEqual(ring.arcs_of(1, 99), [(0, 40), (41, 99)])


if __name__ == "__main__":
    unittest.main()
