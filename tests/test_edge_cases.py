"""Every edge case from section 4.5 of the spec: report a reason, keep the old data, and never stop the program."""

import unittest

from hilbert_hotel import HotelSystem, ModNHotel
from hilbert_hotel.migration import METHOD_RANGE, METHOD_SCAN
from hilbert_hotel.stats import compute_load_stats


def state(system):
    return (system.snapshot(), system.buildings(), system.ring.points(), system.load_counts())


class Section45EdgeCases(unittest.TestCase):
    def setUp(self):
        self.system = HotelSystem.initialize(n=4, v=8, channels=3, guests_per_channel=25, salt="e")

    # 1) add a building id that already exists
    def test_add_existing_building_is_rejected(self):
        before = state(self.system)
        for method in (METHOD_SCAN, METHOD_RANGE):
            result = self.system.add_building(3, method=method)
            self.assertFalse(result.ok)
            self.assertTrue(result.message)
        self.assertEqual(state(self.system), before)  # no duplicate points on the ring
        self.assertEqual(self.system.ring.point_count, 4 * 8)
        # removing it later must remove all of its points
        self.assertTrue(self.system.remove_building(3).ok)
        self.assertEqual(self.system.ring.point_count, 3 * 8)
        self.assertFalse(any(p[1] == 3 for p in self.system.ring.points()))

    # 2) remove a building id that doesn't exist
    def test_remove_missing_building_is_rejected_and_reports_nothing(self):
        before = state(self.system)
        self.system.add_building(9)
        last = self.system.last_migration
        self.system.remove_building(9)
        self.system.last_migration = last  # reset so only the result of the command under test is left
        before = state(self.system)
        for bad in (99, 0, -5, "2", None, 2.0):
            result = self.system.remove_building(bad)
            self.assertFalse(result.ok, bad)
            self.assertIsNone(result.data)  # no migration report comes back
        self.assertIs(self.system.last_migration, last)
        self.assertEqual(state(self.system), before)

    # 3) remove the last building (even when there are no guests)
    def test_cannot_remove_the_last_building_even_when_empty(self):
        for guests in (0, 5):
            system = HotelSystem.initialize(n=1, v=8, channels=1, guests_per_channel=guests)
            before = state(system)
            result = system.remove_building(1)
            self.assertFalse(result.ok)
            self.assertEqual(state(system), before)
            self.assertEqual(system.building_count, 1)
            if guests:
                self.assertTrue(system.find_by_guest(1, 1).ok)  # the ring can still be searched, no endless loop
        system = HotelSystem.initialize(n=2, v=3, channels=1, guests_per_channel=10)
        self.assertTrue(system.remove_building(2).ok)
        self.assertFalse(system.remove_building(1).ok)  # only one building left now
        self.assertEqual(system.compute_building(1, 1), 1)

    # 4) add a duplicate guest id
    def test_add_duplicate_guest_is_rejected(self):
        before = state(self.system)
        original = self.system.find_by_guest(2, 10).data.address
        result = self.system.add_guest(2, 10)
        self.assertFalse(result.ok)
        self.assertEqual(self.system.find_by_guest(2, 10).data.address, original)  # not overwritten
        self.assertEqual(state(self.system), before)
        self.assertFalse(self.system.add_guests(2, 10, 1).ok)

    # 5) remove a guest who doesn't exist
    def test_remove_missing_guest_does_not_change_count(self):
        before = state(self.system)
        k = self.system.guest_count
        for bad in [(9, 9), (1, 26), (0, 0), ("a", 1), (None, None), (1.5, 2)]:
            result = self.system.remove_guest(*bad)
            self.assertFalse(result.ok, bad)
        self.assertEqual(self.system.guest_count, k)
        self.assertEqual(state(self.system), before)
        self.assertEqual(self.system.load_report().total, k)  # the statistics aren't thrown off

    # 6) search finds nothing
    def test_search_not_found_is_clear_message(self):
        for result in (self.system.find_by_guest(50, 50), self.system.find_by_room(1, 123456789),
                       self.system.find_by_room(77, 1), self.system.find_by_guest(0, 0)):
            self.assertFalse(result.ok)
            self.assertIsNone(result.data)
            self.assertIsInstance(result.message, str)
            self.assertTrue(result.message)

    # 7) no guests in the system yet (K = 0)
    def test_no_guests_k_zero(self):
        system = HotelSystem.initialize(n=5, v=8, channels=0, guests_per_channel=0)
        stats = system.load_report()
        self.assertEqual((stats.total, stats.maximum, stats.minimum, stats.mean, stats.std), (0, 0, 0, 0.0, 0.0))
        self.assertIsNone(stats.cv)
        self.assertEqual(len(stats.counts), 5)  # every building is listed
        add = system.add_building(6).data
        remove = system.remove_building(2).data
        for report in (add, remove):
            self.assertEqual(report.moved_count, 0)
            self.assertEqual(report.moved, [])
            self.assertIsNone(report.rate)
            self.assertIn("K = 0", report.rate_text)
        self.assertEqual(system.occupied_rooms(), [])

    def test_stats_with_no_buildings_or_guests_never_divide_by_zero(self):
        self.assertIsNone(compute_load_stats({}).cv)
        self.assertIsNone(compute_load_stats({1: 0}).cv)

    # same cases for hash mod N: it's the comparison baseline so it must not break either
    def test_mod_n_baseline_has_same_rejections(self):
        mod = ModNHotel([1, 2, 3])
        mod.add_guests(1, 1, 10)
        before = (mod.snapshot(), mod.building_ids)
        self.assertFalse(mod.add_building(2).ok)
        self.assertFalse(mod.remove_building(9).ok)
        self.assertFalse(mod.add_guests(1, 5, 3).ok)
        self.assertEqual((mod.snapshot(), mod.building_ids), before)
        mod.remove_building(1)
        mod.remove_building(2)
        self.assertFalse(mod.remove_building(3).ok)
        self.assertEqual(mod.building_ids, [3])

    def test_rejected_commands_leave_system_consistent_after_a_barrage(self):
        before = state(self.system)
        garbage = [("add_building", (3,)), ("add_building", (-1,)), ("add_building", ("x",)),
                   ("remove_building", (42,)), ("remove_guest", (7, 77)), ("add_guest", (1, 1)),
                   ("add_guests", (1, 20, 10)), ("add_guests", (1, 1, -4)), ("find_by_guest", (0, 0))]
        for name, args in garbage:
            result = getattr(self.system, name)(*args)
            self.assertFalse(result.ok, (name, args))
        self.assertEqual(state(self.system), before)
        self.assertEqual(self.system.check_consistency(), [])


class RobustnessTests(unittest.TestCase):
    """Input of the wrong type or absurdly large must be rejected with a reason, not an exception or a hang."""

    def setUp(self):
        self.system = HotelSystem.initialize(n=3, v=4, channels=2, guests_per_channel=5, salt="r")

    def test_find_by_room_with_odd_argument_types_is_a_rejection(self):
        for node, room in [([1], 2), ({}, 1), (None, None), (1.5, 2), ("1", "2"), (True, 1), (1, 0), (-1, 5)]:
            with self.subTest(node=node, room=room):
                result = self.system.find_by_room(node, room)
                self.assertFalse(result.ok)
                self.assertTrue(result.message)

    def test_huge_batch_is_rejected_immediately_without_hanging(self):
        import time
        from hilbert_hotel.hotel import MAX_GUESTS_PER_COMMAND

        before = self.system.snapshot()
        start = time.perf_counter()
        for count in (MAX_GUESTS_PER_COMMAND + 1, 10**12, 10**30):
            self.assertFalse(self.system.add_guests(9, 1, count).ok)
        self.assertLess(time.perf_counter() - start, 1.0)  # rejected before any heavy work starts
        self.assertEqual(self.system.snapshot(), before)

    def test_initialize_rejects_oversized_or_failed_setup_instead_of_silently_dropping_guests(self):
        from hilbert_hotel.hotel import MAX_GUESTS_PER_COMMAND, MAX_RING_POINTS

        with self.assertRaises(ValueError):
            HotelSystem.initialize(2, 4, 2, MAX_GUESTS_PER_COMMAND, salt="x")  # total is over the cap
        with self.assertRaises(ValueError):
            HotelSystem.initialize(MAX_RING_POINTS, 2, 1, 1)  # too many points on the ring

    def test_mod_n_baseline_validates_group_arguments_too(self):
        mod = ModNHotel([1, 2, 3])
        for args in [(0, 1, 2), ("a", 1, 2), (1, 0, 2), (1, 1, -5), (1, 1, 2.5), (True, 1, 1)]:
            with self.subTest(args=args):
                self.assertFalse(mod.add_guests(*args).ok)
        self.assertEqual(len(mod), 0)

    def test_clone_is_independent_of_the_original(self):
        original_state = state(self.system)
        copy = self.system.clone()
        self.assertEqual(copy.snapshot(), self.system.snapshot())
        copy.add_building(40)
        copy.add_guests(7, 1, 30)
        copy.remove_guest(1, 1)
        self.assertEqual(state(self.system), original_state)  # the original is completely unchanged
        self.assertEqual(copy.check_consistency(), [])
        self.assertEqual(self.system.check_consistency(), [])


if __name__ == "__main__":
    unittest.main()
