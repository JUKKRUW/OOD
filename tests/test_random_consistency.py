"""Long runs of random commands (including rejected ones), checking that every data structure
stays consistent and that method A and method B give identical results, even on a ring full of collisions."""

import random
import unittest

from hilbert_hotel import HotelSystem
from hilbert_hotel.migration import METHOD_RANGE, METHOD_SCAN

from tests.helpers import random_operations, small_mod_hash

BUILDING_OPS = {"add_building", "remove_building"}


class RandomConsistencyTests(unittest.TestCase):
    def run_lockstep(self, seed, modulus, v):
        hash_fn = small_mod_hash(modulus) if modulus else None
        kwargs = dict(v=v, salt=str(seed))
        if hash_fn:
            kwargs["hash_fn"] = hash_fn
        scan = HotelSystem(migration_method=METHOD_SCAN, **kwargs)
        rng_sys = HotelSystem(migration_method=METHOD_RANGE, **kwargs)
        for system in (scan, rng_sys):
            system.ring.add_nodes([1, 2, 3])
            system.add_guests(1, 1, 20)
        ops_a = random_operations(scan, random.Random(seed), 120)
        ops_b = random_operations(rng_sys, random.Random(seed), 120)
        for step, (act_a, act_b) in enumerate(zip(ops_a, ops_b)):
            self.assertEqual(act_a, act_b)
            where = f"seed={seed} modulus={modulus} v={v} step={step} action={act_a}"
            self.assertEqual(scan.snapshot(), rng_sys.snapshot(), where)
            self.assertEqual(scan.buildings(), rng_sys.buildings(), where)
            if act_a in BUILDING_OPS:
                ma, mb = scan.last_migration, rng_sys.last_migration
                self.assertEqual(ma.moved if ma else None, mb.moved if mb else None, where)
            self.assertEqual(scan.check_consistency(), [], where)
            self.assertEqual(rng_sys.check_consistency(), [], where)
            self.assertGreaterEqual(scan.building_count, 1, where)

    def test_real_sha256_hashing(self):
        for seed in range(6):
            self.run_lockstep(seed, modulus=None, v=random.Random(seed).randint(1, 8))

    def test_heavy_collisions_small_position_space(self):
        for seed in range(6):
            for modulus in (7, 30):
                self.run_lockstep(seed, modulus=modulus, v=(seed % 3) + 1)

    def test_every_migration_follows_the_rules(self):
        rng = random.Random(99)
        system = HotelSystem.initialize(6, 4, 3, 60, salt="r", hash_fn=small_mod_hash(40))
        for _ in range(60):
            before = system.snapshot()
            k = len(before)
            if rng.random() < 0.5:
                node = rng.randint(0, 15)
                result = system.add_building(node)
                kind = "add"
            else:
                node = rng.choice(system.buildings() + [99])
                result = system.remove_building(node)
                kind = "remove"
            after = system.snapshot()
            self.assertEqual(len(after), k)  # no guest lost or gained
            if not result.ok:
                self.assertEqual(after, before)
                continue
            report = result.data
            changed = {gid for gid in before if before[gid] != after[gid]}
            self.assertEqual(changed, {r.guest_id for r in report.moved})
            self.assertEqual(report.moved_count, len(changed))
            for r in report.moved:
                self.assertEqual(r.room_no, before[r.guest_id][1])
                self.assertEqual(r.room_no, after[r.guest_id][1])
                if kind == "add":
                    self.assertEqual(r.to_node, node)
                else:
                    self.assertEqual(r.from_node, node)


if __name__ == "__main__":
    unittest.main()
