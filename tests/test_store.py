import unittest

from hilbert_hotel.guest_store import DuplicateGuestError, GuestStore
from hilbert_hotel.room import room_no
from hilbert_hotel.stats import compute_load_stats


def make_store(entries):
    store = GuestStore()
    store.add_many(entries)
    return store


class GuestStoreTests(unittest.TestCase):
    def test_add_and_lookup_both_ways(self):
        store = make_store([(1, 1, 100, 7), (2, 1, 200, 8)])
        g = store.get(1, 1)
        self.assertEqual((g.node_id, g.room_no), (7, room_no(1, 1)))
        self.assertIs(store.get_by_room(7, room_no(1, 1)), g)
        self.assertIsNone(store.get_by_room(8, room_no(1, 1)))  # same room number in a different building = not found
        self.assertEqual(store.validate(), [])

    def test_room_number_is_formula_not_arrival_order(self):
        store = GuestStore()
        store.add_many([(1, 5, 50, 1)])
        store.add_many([(1, 1, 10, 1)])  # added later, but the room number still follows the formula, not "last number + 1"
        self.assertEqual(store.get(1, 5).room_no, room_no(1, 5))
        self.assertEqual(store.get(1, 1).room_no, 1)

    def test_duplicate_in_store_rejects_whole_batch(self):
        store = make_store([(1, 1, 10, 1), (1, 2, 20, 2)])
        before = (len(store), store.counts(), [g.address for g in store.sorted_rooms()])
        with self.assertRaises(DuplicateGuestError):
            store.add_many([(1, 3, 30, 1), (1, 2, 99, 3), (1, 4, 40, 1)])  # (1,2) is a duplicate, so none of the others may be added
        self.assertEqual(before, (len(store), store.counts(), [g.address for g in store.sorted_rooms()]))
        self.assertIsNone(store.get(1, 3))
        self.assertIsNone(store.get(1, 4))
        self.assertEqual(store.validate(), [])

    def test_duplicate_inside_batch_rejected(self):
        store = GuestStore()
        with self.assertRaises(DuplicateGuestError):
            store.add_many([(1, 1, 5, 1), (1, 1, 5, 1)])
        self.assertEqual(len(store), 0)

    def test_invalid_id_in_batch_leaves_store_untouched(self):
        store = make_store([(1, 1, 10, 1)])
        with self.assertRaises(ValueError):
            store.add_many([(1, 2, 20, 1), (0, 5, 30, 1)])
        self.assertEqual(len(store), 1)
        self.assertEqual(store.validate(), [])

    def test_remove_updates_every_index(self):
        store = make_store([(1, 1, 10, 1), (1, 2, 20, 1), (2, 1, 30, 2)])
        removed = store.remove(1, 2)
        self.assertEqual((removed.c, removed.s), (1, 2))
        self.assertIsNone(store.get(1, 2))
        self.assertIsNone(store.get_by_room(1, room_no(1, 2)))
        self.assertEqual(store.count_in(1), 1)
        self.assertEqual(len(store), 2)
        self.assertEqual(store.validate(), [])

    def test_remove_missing_returns_none_and_changes_nothing(self):
        store = make_store([(1, 1, 10, 1)])
        self.assertIsNone(store.remove(9, 9))
        self.assertEqual(len(store), 1)
        self.assertEqual(store.validate(), [])

    def test_removing_all_guests_of_a_node_drops_its_count(self):
        store = make_store([(1, 1, 10, 1)])
        store.remove(1, 1)
        self.assertEqual(store.counts(), {})
        self.assertEqual(store.count_in(1), 0)

    def test_move_changes_node_but_keeps_room_and_updates_indexes(self):
        store = make_store([(3, 4, 77, 1)])
        room = room_no(3, 4)
        self.assertTrue(store.move(3, 4, 2))
        g = store.get(3, 4)
        self.assertEqual((g.node_id, g.room_no), (2, room))
        self.assertIsNone(store.get_by_room(1, room))  # the room index has to follow
        self.assertIs(store.get_by_room(2, room), g)
        self.assertEqual(store.counts(), {2: 1})
        self.assertFalse(store.move(3, 4, 2))  # already there = nothing changes
        self.assertEqual(store.validate(), [])

    def test_sorted_rooms_numeric_order_not_text_order(self):
        store = make_store([(1, 1, 1, 2), (2, 1, 2, 2), (3, 1, 3, 2), (1, 4, 4, 2), (1, 5, 5, 1), (10, 1, 6, 10)])
        rooms = [(g.node_id, g.room_no) for g in store.sorted_rooms()]
        self.assertEqual(rooms, sorted(rooms))
        # node 2 comes before node 10 (numeric, not alphabetical) and room 2 before room 10
        self.assertLess(rooms.index((2, room_no(1, 1))), rooms.index((10, room_no(10, 1))))
        node2 = [r for n, r in rooms if n == 2]
        self.assertEqual(node2, sorted(node2))
        self.assertGreater(max(node2), 9)  # there are two-digit room numbers so that an alphabetical sort would really get it wrong

    def test_guests_in_arcs_returns_only_guests_in_range_inclusive(self):
        store = make_store([(1, i, pos, 1) for i, pos in enumerate([10, 20, 25, 35, 50, 70, 90], start=1)])
        got = sorted(g.pos for g in store.guests_in_arcs([(20, 35)]))
        self.assertEqual(got, [20, 25, 35])
        got = sorted(g.pos for g in store.guests_in_arcs([(0, 10), (71, 99)]))
        self.assertEqual(got, [10, 90])
        self.assertEqual(store.guests_in_arcs([(11, 19)]), [])

    def test_guests_in_arcs_with_same_position_guests(self):
        store = make_store([(1, 1, 40, 1), (1, 2, 40, 1), (1, 3, 41, 1)])
        self.assertEqual(len(store.guests_in_arcs([(40, 40)])), 2)

    def test_batch_merge_keeps_position_index_sorted(self):
        store = make_store([(1, i, (i * 37) % 101, 1) for i in range(1, 30)])
        store.add_many([(2, i, (i * 53) % 103, 2) for i in range(1, 30)])
        store.add_many([(3, 1, 55, 3)])  # adding one at a time must also insert in the right place
        self.assertEqual(store.validate(), [])

    def test_validate_detects_corruption(self):
        store = make_store([(1, 1, 10, 1), (1, 2, 20, 2)])
        store._by_room.clear()  # simulate an inconsistent index
        self.assertTrue(store.validate())


class LoadStatsTests(unittest.TestCase):
    def test_basic_values_include_zero_buildings(self):
        stats = compute_load_stats({1: 4, 2: 2, 3: 0})
        self.assertEqual((stats.maximum, stats.minimum, stats.total, stats.buildings), (4, 0, 6, 3))
        self.assertAlmostEqual(stats.mean, 2.0)
        self.assertAlmostEqual(stats.std, (8 / 3) ** 0.5)  # population version (divide by N)
        self.assertAlmostEqual(stats.cv, stats.std / stats.mean)

    def test_perfectly_balanced(self):
        stats = compute_load_stats({1: 5, 2: 5})
        self.assertEqual((stats.std, stats.cv), (0.0, 0.0))

    def test_no_guests_gives_zeros_and_no_cv(self):
        stats = compute_load_stats({1: 0, 2: 0, 3: 0})
        self.assertEqual((stats.maximum, stats.minimum, stats.mean, stats.std), (0, 0, 0.0, 0.0))
        self.assertIsNone(stats.cv)
        self.assertEqual(stats.buildings, 3)

    def test_no_buildings_does_not_divide_by_zero(self):
        stats = compute_load_stats({})
        self.assertIsNone(stats.cv)
        self.assertEqual(stats.total, 0)


if __name__ == "__main__":
    unittest.main()
