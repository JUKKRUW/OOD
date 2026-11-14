"""Run all of experiments A/B/C and write the CSV files (raw data of every round + summary) and the graphs.

    python -m experiments.run_all                 # full set from the spec (takes a few minutes)
    python -m experiments.run_all --quick         # small set to try things out (not what the report uses)
    python -m experiments.run_all --out results   # output folder (default: results)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable, Dict, List, Optional

from experiments.runner import (
    A_VALUE_KEYS, B_VALUE_KEYS, C_VALUE_KEYS, SALTS, environment_info,
    run_experiment_a, run_experiment_b, run_experiment_c, summarize,
)
from hilbert_hotel.io_csv import write_rows


def _fieldnames(rows: List[Dict[str, object]]) -> List[str]:
    names: List[str] = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    return names


def write_samples(out_dir: str = "results") -> List[str]:
    """Sample files for submission: sample guests (K=1,000) and the migrations from adding building 11 (N=10, V=32, salt 0)."""
    from hilbert_hotel import HotelSystem
    from hilbert_hotel.io_csv import export_guests, export_migrations

    system = HotelSystem.initialize(n=10, v=32, channels=10, guests_per_channel=100, salt="0")
    guests_path = os.path.join(out_dir, "sample_guests.csv")
    export_guests(system, guests_path)
    report = system.add_building(11).data
    migrations_path = os.path.join(out_dir, "sample_migrations.csv")
    export_migrations(report, migrations_path)
    return [guests_path, migrations_path]


def run_all(out_dir: str = "results", quick: bool = False, make_plots: bool = True,
            progress: Optional[Callable[[str], None]] = print) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    if quick:
        params = {"A": dict(ks=(500, 2000), n=10, v=32), "B": dict(ns=(5, 10), k=2000, v=32),
                  "C": dict(vs=(1, 8), n=10, k=2000)}
    else:
        params = {"A": dict(ks=(1000, 10000, 100000), n=10, v=32), "B": dict(ns=(5, 10, 20), k=10000, v=32),
                  "C": dict(vs=(1, 8, 32), n=10, k=10000)}

    outputs: Dict[str, object] = {"files": [], "params": params, "quick": quick}
    plan = [
        ("A", run_experiment_a, ["K"], A_VALUE_KEYS),
        ("B", run_experiment_b, ["N", "operation", "method"], B_VALUE_KEYS),
        ("C", run_experiment_c, ["V"], C_VALUE_KEYS),
    ]
    for name, runner, group_keys, value_keys in plan:
        rows = runner(salts=SALTS, progress=progress, **params[name])
        raw_path = os.path.join(out_dir, f"experiment_{name}_raw.csv")
        write_rows(raw_path, _fieldnames(rows), rows)
        summary = summarize(rows, group_keys, value_keys)
        summary_path = os.path.join(out_dir, f"experiment_{name}_summary.csv")
        write_rows(summary_path, _fieldnames(summary), summary)
        outputs["files"].extend([raw_path, summary_path])
        if progress:
            progress(f"เขียน {raw_path} ({len(rows)} แถว) และ {summary_path} ({len(summary)} แถว)")

    env = environment_info()
    env["quick"] = quick
    env["parameters"] = {k: {kk: list(vv) if isinstance(vv, tuple) else vv for kk, vv in v.items()}
                         for k, v in params.items()}
    env_path = os.path.join(out_dir, "environment.json")
    with open(env_path, "w", encoding="utf-8") as handle:
        json.dump(env, handle, ensure_ascii=False, indent=2)
    outputs["files"].append(env_path)
    outputs["files"].extend(write_samples(out_dir))

    if make_plots:
        from experiments import plots

        fig_dir = os.path.join(out_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        for name, plotter in (("A", plots.plot_a), ("B", plots.plot_b), ("C", plots.plot_c)):
            paths = plotter(os.path.join(out_dir, f"experiment_{name}_raw.csv"), fig_dir)
            outputs["files"].extend(paths)
            if progress:
                for path in paths:
                    progress(f"เขียนกราฟ {path}")
    return outputs


def main(argv: Optional[List[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="รันการทดลอง A/B/C ของ Distributed Hilbert's Hotel")
    parser.add_argument("--out", default="results", help="โฟลเดอร์ปลายทาง (ค่าเริ่มต้น results)")
    parser.add_argument("--quick", action="store_true", help="ชุดเล็ก ใช้ลองระบบเท่านั้น")
    parser.add_argument("--no-plots", action="store_true", help="ไม่สร้างกราฟ (ไม่ต้องใช้ matplotlib)")
    args = parser.parse_args(argv)
    run_all(args.out, quick=args.quick, make_plots=not args.no_plots)
    return 0


if __name__ == "__main__":
    sys.exit(main())
