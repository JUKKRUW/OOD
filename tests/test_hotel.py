import unittest

from hilbert_hotel import HotelSystem, room_no
from hilbert_hotel.migration import METHOD_RANGE, METHOD_SCAN

from tests.helpers import A, B, C, D, build_small_hotel

METHODS = (METHOD_SCAN, METHOD_RANGE)
OWNER_LETTER = {A: "A", B: "B", C: "C", D: "D"}


def layout(system, count=7):
    """Buildings of guests (1,1)..(1,count), in order, as a string of letters."""
    return "".join(OWNER_LETTER[system.store.get(1, i).node_id] for i in range(1, count + 1))


class SmallRingExampleTests(unittest.TestCase):
    """Example 9.1: ring of size 100, V = 1, A=20 B=50 C=80, guests at 10 20 25 35 50 70 90."""

    def test_initial_state_matches_table(self):
        for method in METHODS:
            with self.subTest(method=method):
                self.assertEqual(layout(build_small_hotel(method)), "AABBBCA")  # position 90 wraps around to A

    def test_adding_D_at_35_moves_exactly_two_guests(self):
        for method in METHODS:
            with self.subTest(method=method):
                system = build_small_hotel(method)
                before = system.snapshot()
                result = system.add_building(D)
                self.assertTrue(result.ok)
                report = result.data
                self.assertEqual(layout(system), "AADDBCA")
                self.assertEqual(report.moved_count, 2)
                self.assertEqual([(r.guest_id, r.from_node, r.to_node) for r in report.moved],
                                 [((1, 3), B, D), ((1, 4), B, D)])
                for record in report.moved:  # a mover keeps the same room number
                    self.assertEqual(record.room_no, before[record.guest_id][1])
                    self.assertEqual(system.snapshot()[record.guest_id][1], record.room_no)
                self.assertEqual(report.rate, 2 / 7)
                self.assertEqual(system.check_consistency(), [])

    def test_removing_B_moves_exactly_three_guests(self):
        for method in METHODS:
            with self.subTest(method=method):
                system = build_small_hotel(method)
                before = system.snapshot()
                result = system.remove_building(B)
                self.assertTrue(result.ok)
                report = result.data
                self.assertEqual(layout(system), "AACCCCA")
                self.assertEqual(report.moved_count, 3)
                self.assertEqual([(r.guest_id, r.from_node, r.to_node) for r in report.moved],
                                 [((1, 3), B, C), ((1, 4), B, C), ((1, 5), B, C)])
                for record in report.moved:
                    self.assertEqual(record.room_no, before[record.guest_id][1])
                self.assertEqual(system.check_consistency(), [])

    def test_guests_outside_affected_range_do_not_move(self):
        system = build_small_hotel()
        before = system.snapshot()
        system.add_building(D)
        after = system.snapshot()
        moved = {gid for gid in before if before[gid] != after[gid]}
        self.assertEqual(moved, {(1, 3), (1, 4)})

    def test_wrap_around_guest_goes_to_first_point(self):
        system = build_small_hotel()
        self.assertEqual(system.store.get(1, 7).node_id, A)  # position 90 > 80, wraps to the start of the ring
        self.assertEqual(system.compute_building(1, 7), A)


class AddingGuestsTests(unittest.TestCase):
    def setUp(self):
        self.system = HotelSystem.initialize(n=6, v=16, channels=3, guests_per_channel=50, salt="t")

    def test_existing_addresses_unchanged_after_adding_old_and_new_channels(self):
        before = self.system.snapshot()
        self.assertTrue(self.system.add_guests(1, 51, 40).ok)  # existing channel, appended at the end
        self.assertTrue(self.system.add_guests(2, 51, 10).ok)
        self.assertTrue(self.system.add_guests(7, 1, 60).ok)  # a new channel
        self.assertTrue(self.system.add_guest(9, 1).ok)
        after = self.system.snapshot()
        for guest_id, address in before.items():
            self.assertEqual(after[guest_id], address, f"guest {guest_id} changed room")
        self.assertEqual(len(after), len(before) + 40 + 10 + 60 + 1)
        self.assertEqual(self.system.check_consistency(), [])

    def test_room_numbers_are_formula_values(self):
        for g in self.system.store.guests():
            self.assertEqual(g.room_no, room_no(g.c, g.s))

    def test_no_two_guests_share_an_address(self):
        self.system.add_guests(4, 1, 80)
        addresses = [g.address for g in self.system.store.guests()]
        self.assertEqual(len(addresses), len(set(addresses)))
        rooms = [g.room_no for g in self.system.store.guests()]
        self.assertEqual(len(rooms), len(set(rooms)))  # the formula gives unique room numbers even across buildings

    def test_arrival_order_does_not_change_address(self):
        other = HotelSystem.initialize(n=6, v=16, channels=0, guests_per_channel=0, salt="t")
        for s in range(30, 0, -1):  # insert in reverse order
            other.add_guest(1, s)
        for s in range(1, 31):
            self.assertEqual(other.store.get(1, s).address, self.system.store.get(1, s).address)

    def test_group_overlap_rejects_whole_group_and_keeps_data(self):
        before = self.system.snapshot()
        counts = self.system.load_counts()
        result = self.system.add_guests(2, 45, 20)  # 45..64 overlaps the existing 45..50
        self.assertFalse(result.ok)
        self.assertEqual(self.system.snapshot(), before)
        self.assertEqual(self.system.load_counts(), counts)
        self.assertIsNone(self.system.store.get(2, 51))  # the non-overlapping part must not be added either
        self.assertEqual(self.system.check_consistency(), [])

    def test_group_overlap_by_only_one_guest_is_rejected(self):
        before = self.system.snapshot()
        self.assertFalse(self.system.add_guests(3, 50, 5).ok)  # only (3,50) is a duplicate
        self.assertEqual(self.system.snapshot(), before)

    def test_zero_count_changes_nothing(self):
        before = self.system.snapshot()
        result = self.system.add_guests(1, 1, 0)
        self.assertTrue(result.ok)
        self.assertEqual(result.data, 0)
        self.assertEqual(self.system.snapshot(), before)
        self.assertTrue(self.system.add_guests(8, 1, 0).ok)  # a new channel with no guests is fine
        self.assertEqual(self.system.snapshot(), before)

    def test_invalid_input_rejected_without_exception(self):
        before = self.system.snapshot()
        for args in [(0, 1, 5), (1, 0, 5), (-1, 1, 5), (1, 1, -1), (1.5, 1, 2), (1, 1, 2.5),
                     ("1", 1, 1), (True, 1, 1), (1, 1, True), (None, 1, 1)]:
            with self.subTest(args=args):
                result = self.system.add_guests(*args)
                self.assertFalse(result.ok)
        self.assertEqual(self.system.snapshot(), before)

    def test_readding_removed_guest_is_a_fresh_stay_computed_from_current_ring(self):
        self.assertTrue(self.system.remove_guest(1, 7).ok)
        self.system.add_building(77)  # the ring changes while the guest is away
        result = self.system.add_guest(1, 7)
        self.assertTrue(result.ok)
        guest = result.data
        self.assertEqual(guest.node_id, self.system.compute_building(1, 7))
        self.assertEqual(guest.room_no, room_no(1, 7))
        self.assertEqual(self.system.check_consistency(), [])

    def test_remove_guest_updates_all_lookups(self):
        guest = self.system.find_by_guest(2, 9).data
        self.assertTrue(self.system.remove_guest(2, 9).ok)
        self.assertFalse(self.system.find_by_guest(2, 9).ok)
        self.assertFalse(self.system.find_by_room(guest.node_id, guest.room_no).ok)
        self.assertEqual(self.system.guest_count, 3 * 50 - 1)
        self.assertEqual(self.system.check_consistency(), [])


class LookupAndListingTests(unittest.TestCase):
    def setUp(self):
        self.system = HotelSystem.initialize(n=5, v=8, channels=4, guests_per_channel=30, salt="q")

    def test_two_way_search_agree(self):
        for g in list(self.system.store.guests())[:40]:
            by_id = self.system.find_by_guest(g.c, g.s).data
            by_room = self.system.find_by_room(by_id.node_id, by_id.room_no).data
            self.assertEqual((by_room.c, by_room.s), (g.c, g.s))

    def test_not_found_is_a_result_not_an_exception(self):
        self.assertFalse(self.system.find_by_guest(99, 99).ok)
        self.assertFalse(self.system.find_by_room(1, 10**9).ok)
        self.assertFalse(self.system.find_by_room(99, 1).ok)
        self.assertFalse(self.system.find_by_guest("x", None).ok)

    def test_compute_building_uses_ring_only(self):
        for g in self.system.store.guests():
            self.assertEqual(self.system.compute_building(g.c, g.s), g.node_id)
        self.assertIn(self.system.compute_building(500, 500), self.system.buildings())  # works for a guest who isn't checked in too
        with self.assertRaises(ValueError):
            self.system.compute_building(0, 1)

    def test_occupied_rooms_sorted_by_node_then_numeric_room(self):
        rows = [(g.node_id, g.room_no) for g in self.system.occupied_rooms()]
        self.assertEqual(rows, sorted(rows))
        self.assertEqual(len(rows), self.system.guest_count)

    def test_load_report_includes_empty_buildings(self):
        tiny = HotelSystem.initialize(n=8, v=1, channels=1, guests_per_channel=3)
        stats = tiny.load_report()
        self.assertEqual(len(stats.counts), 8)  # every building is listed, even empty ones
        self.assertEqual(stats.total, 3)
        self.assertEqual(stats.minimum, 0)
        self.assertEqual(sum(stats.counts.values()), 3)


class InitializationTests(unittest.TestCase):
    def test_channels_start_at_one_and_counts_per_channel(self):
        system = HotelSystem.initialize(n=3, v=4, channels=3, guests_per_channel=[2, 0, 5])
        ids = sorted(g.guest_id for g in system.store.guests())
        self.assertEqual(ids, [(1, 1), (1, 2)] + [(3, s) for s in range(1, 6)])

    def test_sizes(self):
        system = HotelSystem.initialize(n=10, v=32, channels=10, guests_per_channel=100)
        self.assertEqual((system.building_count, system.ring.point_count, system.guest_count), (10, 320, 1000))

    def test_invalid_parameters_raise_value_error(self):
        for kwargs in [dict(n=0, v=4, channels=1, guests_per_channel=1),
                       dict(n=2, v=0, channels=1, guests_per_channel=1),
                       dict(n=2, v=4, channels=-1, guests_per_channel=1),
                       dict(n=2, v=4, channels=2, guests_per_channel=[1]),
                       dict(n=2, v=4, channels=1, guests_per_channel=-3),
                       dict(n=2, v=4, channels=1, guests_per_channel="a")]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HotelSystem.initialize(**kwargs)

    def test_same_salt_same_result_different_salt_different_ring(self):
        a = HotelSystem.initialize(5, 8, 2, 20, salt="0")
        b = HotelSystem.initialize(5, 8, 2, 20, salt="0")
        c = HotelSystem.initialize(5, 8, 2, 20, salt="1")
        self.assertEqual(a.snapshot(), b.snapshot())
        self.assertNotEqual(a.ring.points(), c.ring.points())


class BuildingChangeTests(unittest.TestCase):
    def setUp(self):
        self.systems = {m: HotelSystem.initialize(10, 32, 10, 300, salt="b", migration_method=m) for m in METHODS}

    def test_add_building_movers_go_only_to_new_building(self):
        for method, system in self.systems.items():
            with self.subTest(method=method):
                before = system.snapshot()
                report = system.add_building(11).data
                after = system.snapshot()
                changed = {gid for gid in before if before[gid] != after[gid]}
                self.assertEqual(changed, {r.guest_id for r in report.moved})
                self.assertEqual(report.moved_count, len(report.moved))
                self.assertGreater(report.moved_count, 0)
                for r in report.moved:
                    self.assertEqual(r.to_node, 11)
                    self.assertNotEqual(r.from_node, 11)
                    self.assertEqual(r.from_node, before[r.guest_id][0])
                    self.assertEqual(r.room_no, before[r.guest_id][1])
                    self.assertEqual(after[r.guest_id], (11, r.room_no))
                self.assertEqual(len(after), len(before))  # no guest lost or gained
                self.assertEqual(system.check_consistency(), [])

    def test_remove_building_only_its_guests_move(self):
        for method, system in self.systems.items():
            with self.subTest(method=method):
                before = system.snapshot()
                residents = {gid for gid, (node, _room) in before.items() if node == 4}
                report = system.remove_building(4).data
                after = system.snapshot()
                self.assertEqual({r.guest_id for r in report.moved}, residents)
                for gid, address in before.items():
                    if gid not in residents:
                        self.assertEqual(after[gid], address, "guests in other buildings must not move")
                    else:
                        self.assertNotEqual(after[gid][0], 4)
                        self.assertEqual(after[gid][1], address[1])  # room number unchanged
                self.assertNotIn(4, system.buildings())
                self.assertEqual(len(after), len(before))
                self.assertEqual(system.check_consistency(), [])

    def test_report_count_equals_printed_records(self):
        for system in self.systems.values():
            report = system.add_building(12).data
            self.assertEqual(report.moved_count, len(report.moved))
            self.assertEqual(report.guests_before, 3000)
            self.assertAlmostEqual(report.rate, report.moved_count / 3000)

    def test_method_a_and_b_give_identical_reports(self):
        scan, rng = self.systems[METHOD_SCAN], self.systems[METHOD_RANGE]
        for op, arg in [("add_building", 11), ("remove_building", 3), ("add_building", 12),
                        ("remove_building", 11), ("remove_building", 1), ("add_building", 3)]:
            ra, rb = getattr(scan, op)(arg).data, getattr(rng, op)(arg).data
            self.assertEqual(ra.moved, rb.moved, f"{op}({arg})")
            self.assertEqual(scan.snapshot(), rng.snapshot())

    def test_add_then_remove_same_building_restores_everything(self):
        system = self.systems[METHOD_RANGE]
        before = system.snapshot()
        system.add_building(50)
        system.remove_building(50)
        self.assertEqual(system.snapshot(), before)

    def test_remove_then_add_same_building_restores_everything(self):
        system = self.systems[METHOD_RANGE]
        before = system.snapshot()
        system.remove_building(6)
        system.add_building(6)
        self.assertEqual(system.snapshot(), before)

    def test_method_can_be_chosen_per_call(self):
        system = HotelSystem.initialize(5, 8, 2, 50, salt="m")
        r1 = system.add_building(6, method=METHOD_SCAN).data
        r2 = system.remove_building(6, method=METHOD_RANGE).data
        self.assertEqual((r1.method, r2.method), (METHOD_SCAN, METHOD_RANGE))
        with self.assertRaises(ValueError):
            system.add_building(7, method="nonsense")

    def test_timings_recorded_per_step(self):
        report = self.systems[METHOD_RANGE].add_building(11).data
        self.assertEqual(set(report.timings), {"ring_update", "find", "apply"})
        self.assertTrue(all(t >= 0 for t in report.timings.values()))

    def test_last_migration_is_kept(self):
        system = self.systems[METHOD_RANGE]
        self.assertIsNone(system.last_migration)
        system.add_building(11)
        self.assertEqual(system.last_migration.kind, "add_building")

    def test_expected_move_rates_are_in_the_right_range(self):
        # expected K/(N+1) = 1/11 = 9.1% and K/N = 10%; we only check it's nowhere near the
        # "almost everyone" of hash mod N
        system = self.systems[METHOD_RANGE]
        add_rate = system.add_building(11).data.rate
        remove_rate = system.remove_building(2).data.rate
        self.assertTrue(0.03 < add_rate < 0.2, add_rate)
        self.assertTrue(0.03 < remove_rate < 0.2, remove_rate)


if __name__ == "__main__":
    unittest.main()
