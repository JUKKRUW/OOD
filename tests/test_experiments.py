import os
import tempfile
import unittest

from experiments.runner import (A_VALUE_KEYS, SALTS, measure_memory, median, run_experiment_a,
                                run_experiment_b, run_experiment_c, split_evenly, summarize)
from hilbert_hotel.io_csv import read_rows
from hilbert_hotel.stats import compute_load_stats

try:
    import matplotlib  # noqa: F401

    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False


class HelperTests(unittest.TestCase):
    def test_split_evenly_spread_at_most_one(self):
        for k in (0, 1, 9, 10, 1000, 10007):
            parts = split_evenly(k, 10)
            self.assertEqual(sum(parts), k)
            self.assertLessEqual(max(parts) - min(parts), 1)
            self.assertEqual(len(parts), 10)

    def test_median(self):
        self.assertEqual(median([3, 1, 2]), 2)
        self.assertEqual(median([4, 1, 2, 3]), 2.5)

    def test_summarize_groups_and_statistics(self):
        rows = [{"salt": s, "K": 1, "x": v} for s, v in zip("012", (3.0, 1.0, 2.0))]
        rows += [{"salt": s, "K": 2, "x": v} for s, v in zip("012", (10.0, 30.0, 20.0))]
        out = summarize(rows, ["K"], ["x"])
        self.assertEqual([r["K"] for r in out], [1, 2])
        self.assertEqual((out[0]["x_median"], out[0]["x_min"], out[0]["x_max"], out[0]["rounds"]), (2.0, 1.0, 3.0, 3))
        self.assertAlmostEqual(out[1]["x_mean"], 20.0)

    def test_measure_memory_current_and_peak(self):
        current, peak = measure_memory(lambda: [bytearray(1000) for _ in range(2000)])
        self.assertGreater(current, 1.5)  # ~2 MB
        self.assertGreaterEqual(peak, current)


class ExperimentTests(unittest.TestCase):
    def test_a_rows_and_fields(self):
        rows = run_experiment_a(ks=(300,), n=4, v=4)
        self.assertEqual([r["salt"] for r in rows], list(SALTS))
        for row in rows:
            for key in A_VALUE_KEYS:
                self.assertGreater(row[key], 0, key)
            self.assertGreaterEqual(row["mem_total_peak_mib"], row["mem_total_current_mib"])
            self.assertEqual((row["K"], row["N"], row["V"]), (300, 4, 4))

    def test_b_methods_agree_and_mod_n_moves_far_more(self):
        rows = run_experiment_b(ns=(4, 6), k=1500, v=16)
        self.assertEqual(len(rows), 2 * len(SALTS) * 6)
        key = lambda r: (r["N"], r["salt"], r["operation"])
        by_case = {}
        for r in rows:
            by_case.setdefault(key(r), {})[r["method"]] = r
        for case, methods in by_case.items():
            self.assertEqual(methods["scan"]["moved"], methods["range"]["moved"], case)
            self.assertGreater(methods["modn"]["moved"], 2 * methods["scan"]["moved"], case)
            for m in methods.values():
                self.assertEqual(m["K"], 1500)
                self.assertAlmostEqual(m["rate"], m["moved"] / 1500)

    def test_b_is_reproducible_with_same_salts(self):
        first = run_experiment_b(ns=(5,), k=1000, v=8)
        second = run_experiment_b(ns=(5,), k=1000, v=8)
        self.assertEqual([r["moved"] for r in first], [r["moved"] for r in second])

    def test_c_counts_and_cv(self):
        rows = run_experiment_c(vs=(1, 8), n=5, k=1000)
        self.assertEqual(len(rows), 2 * len(SALTS))
        for row in rows:
            counts = [int(x) for x in row["counts"].split(";")]
            self.assertEqual(len(counts), 5)
            self.assertEqual(sum(counts), 1000)
            stats = compute_load_stats({i: c for i, c in enumerate(counts)})
            self.assertAlmostEqual(row["cv"], stats.cv)
            self.assertEqual((row["max"], row["min"]), (stats.maximum, stats.minimum))
            self.assertEqual(row["points_M"], 5 * row["V"])
        v1 = [r["cv"] for r in rows if r["V"] == 1]
        v8 = [r["cv"] for r in rows if r["V"] == 8]
        self.assertGreater(sum(v1) / len(v1), sum(v8) / len(v8))  # larger V = more even load

    @unittest.skipUnless(HAVE_MATPLOTLIB, "needs matplotlib to draw the graphs (the core program doesn't need it)")
    def test_run_all_quick_writes_csv_and_figures(self):
        from experiments.run_all import run_all

        with tempfile.TemporaryDirectory() as tmp:
            out = run_all(tmp, quick=True, make_plots=True, progress=None)
            for name in ("A", "B", "C"):
                raw = read_rows(os.path.join(tmp, f"experiment_{name}_raw.csv"))
                self.assertTrue(raw)
                self.assertEqual({r["salt"] for r in raw}, set(SALTS))  # the salt of every round is recorded
                self.assertTrue(read_rows(os.path.join(tmp, f"experiment_{name}_summary.csv")))
            pngs = [p for p in out["files"] if p.endswith(".png")]
            self.assertEqual(len(pngs), 6)
            for path in pngs:
                self.assertGreater(os.path.getsize(path), 5000)
            self.assertTrue(os.path.exists(os.path.join(tmp, "environment.json")))


if __name__ == "__main__":
    unittest.main()
