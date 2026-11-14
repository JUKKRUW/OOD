import unittest

from hilbert_hotel import HotelSystem, ModNHotel
from hilbert_hotel.hashing import guest_key, hash64
from hilbert_hotel.room import room_no


class ModNTests(unittest.TestCase):
    def test_choose_by_index_of_hash_mod_n(self):
        mod = ModNHotel([10, 20, 30], salt="s")
        mod.add_guests(1, 1, 50)
        for s in range(1, 51):
            h = hash64(guest_key(1, s), "s")
            self.assertEqual(mod.node_of(1, s), [10, 20, 30][h % 3])

    def test_add_appends_and_remove_keeps_remaining_order(self):
        mod = ModNHotel([5, 3, 9])
        mod.add_building(1)
        self.assertEqual(mod.building_ids, [5, 3, 9, 1])
        mod.remove_building(3)
        self.assertEqual(mod.building_ids, [5, 9, 1])

    def test_moves_are_counted_by_real_building_id_not_index(self):
        mod = ModNHotel([1, 2, 3])
        # hand-picked hash values: before the removal pos=1 -> ids[1]=2 ; pos=4 -> ids[1]=2
        mod.add_guests_with_positions([(1, 1, 1), (1, 2, 4)])
        report = mod.remove_building(1).data  # ids = [2, 3]
        # pos=1: ids[1%2]=3 (moves 2->3) ; pos=4: ids[0]=2, the index changes (1->0) but it is
        # still building 2, so nobody moves
        self.assertEqual([(r.guest_id, r.from_node, r.to_node) for r in report.moved], [((1, 1), 2, 3)])
        self.assertEqual(mod.node_of(1, 2), 2)

    def test_mod_n_moves_almost_everyone_when_adding_one_building(self):
        mod = ModNHotel(list(range(1, 11)), salt="0")
        mod.add_guests(1, 1, 2000)
        report = mod.add_building(11).data
        self.assertGreater(report.rate, 0.8)  # almost everyone has to move (the problem consistent hashing fixes)
        self.assertEqual(report.moved_count, len(report.moved))
        for r in report.moved:
            self.assertEqual(r.room_no, room_no(*r.guest_id))

    def test_same_guests_hashes_and_buildings_as_ring(self):
        ring_system = HotelSystem.initialize(10, 32, 5, 100, salt="2")
        entries = [(g.c, g.s, g.pos) for g in ring_system.store.guests()]
        mod = ModNHotel(ring_system.buildings(), salt="2")
        self.assertTrue(mod.add_guests_with_positions(entries).ok)
        self.assertEqual(len(mod), ring_system.guest_count)
        for c, s, pos in entries:
            self.assertEqual(pos, hash64(guest_key(c, s), "2"))  # same hash values as the ring

    def test_load_counts_include_empty_buildings(self):
        mod = ModNHotel([1, 2, 3, 4], salt="z")
        mod.add_guests_with_positions([(1, 1, 0)])
        counts = mod.load_counts()
        self.assertEqual(len(counts), 4)
        self.assertEqual(sum(counts.values()), 1)

    def test_constructor_rejects_bad_ids(self):
        for bad in ([], [1, 1], [-1], ["a"]):
            with self.assertRaises(ValueError):
                ModNHotel(bad)


if __name__ == "__main__":
    unittest.main()
