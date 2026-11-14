import unittest

from hilbert_hotel.room import guest_from_room, room_no


class RoomNumberTests(unittest.TestCase):
    def test_table_from_assignment(self):
        # sample table from spec 4.2
        expected = {(1, 1): 1, (2, 1): 2, (1, 2): 3, (3, 1): 4, (2, 2): 5}
        for guest_id, room in expected.items():
            self.assertEqual(room_no(*guest_id), room)

    def test_returns_exact_int_not_float(self):
        self.assertIs(type(room_no(7, 9)), int)

    def test_big_integers_are_exact(self):
        c, s = 10**30 + 7, 10**25 + 3
        a, b = c - 1, s - 1
        self.assertEqual(room_no(c, s), (a + b) * (a + b + 1) // 2 + b + 1)  # no float rounding

    def test_distinct_guests_get_distinct_rooms(self):
        seen = {}
        for c in range(1, 151):
            for s in range(1, 151):
                room = room_no(c, s)
                self.assertGreaterEqual(room, 1)
                self.assertNotIn(room, seen, f"{(c, s)} collides with {seen.get(room)}")
                seen[room] = (c, s)
        self.assertEqual(len(seen), 150 * 150)

    def test_first_rooms_have_no_gaps(self):
        # the formula enumerates N x N completely: the ids in the triangle a + b < w use room
        # numbers 1 .. w(w+1)/2 with no gaps
        w = 40
        rooms = {room_no(a + 1, b + 1) for a in range(w) for b in range(w - a)}
        self.assertEqual(rooms, set(range(1, w * (w + 1) // 2 + 1)))

    def test_inverse_round_trip(self):
        for c in range(1, 60):
            for s in range(1, 60):
                self.assertEqual(guest_from_room(room_no(c, s)), (c, s))
        for room in range(1, 5000):
            self.assertEqual(room_no(*guest_from_room(room)), room)

    def test_invalid_ids_rejected(self):
        for bad in [(0, 1), (1, 0), (-1, 3), (True, 1), (1.0, 2), ("1", 2), (None, 1)]:
            with self.assertRaises(ValueError, msg=str(bad)):
                room_no(*bad)
        for bad_room in (0, -5, 2.5, True):
            with self.assertRaises(ValueError):
                guest_from_room(bad_room)


if __name__ == "__main__":
    unittest.main()
