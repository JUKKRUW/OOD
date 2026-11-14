"""Draw the graphs from the experiment CSV files (does not rerun the experiments).

Every graph has axis names with units, the parameters in the title, and shows the spread of
the 3 rounds as the median with a min-max error bar. Labels are in English so we don't depend
on a Thai font being installed.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.runner import A_VALUE_KEYS, B_VALUE_KEYS, C_VALUE_KEYS, summarize  # noqa: E402
from hilbert_hotel.io_csv import read_rows  # noqa: E402

COLORS = {"ring": "#1f77b4", "modn": "#d62728", "scan": "#2ca02c", "range": "#1f77b4", "expected": "#7f7f7f"}


def _errbar(summary: Sequence[Dict[str, object]], key: str):
    med = [float(r[f"{key}_median"]) for r in summary]
    low = [m - float(r[f"{key}_min"]) for m, r in zip(med, summary)]
    high = [float(r[f"{key}_max"]) - m for m, r in zip(med, summary)]
    return med, [low, high]


def _prep(rows: List[Dict[str, str]], numeric: Sequence[str]) -> List[Dict[str, object]]:
    out = []
    for row in rows:
        new: Dict[str, object] = dict(row)
        for key in numeric:
            if key in new and new[key] != "":
                new[key] = float(new[key])
        out.append(new)
    return out


def _caption(rows, extra: str = "") -> str:
    salts = sorted({str(r["salt"]) for r in rows})
    return f"{len(salts)} rounds (salt {', '.join(salts)}): median, bars = min-max{extra}"


def plot_a(raw_path: str, out_dir: str) -> List[str]:
    rows = _prep(read_rows(raw_path), ["K", "N", "V", *A_VALUE_KEYS])
    summary = summarize(rows, ["K"], A_VALUE_KEYS)
    ks = [float(r["K"]) for r in summary]
    n, v = int(rows[0]["N"]), int(rows[0]["V"])
    title = f"Experiment A: N={n}, V={v}, 10 channels"
    paths = []

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, key, label, name in [
        (axes[0], "t_place_guests_s", "Time to place all K guests (s)", "Initial placement"),
        (axes[2], "t_sort_rooms_s", "Time to list occupied rooms sorted (s)", "Sorted room listing"),
    ]:
        med, err = _errbar(summary, key)
        ax.errorbar(ks, med, yerr=err, marker="o", capsize=4, color=COLORS["ring"])
        ax.set_title(name)
        ax.set_ylabel(label)
    for key, name, color in [("t_lookup_by_id_us", "find by guest id", "#1f77b4"),
                             ("t_lookup_by_room_us", "find by (node_id, room_no)", "#ff7f0e"),
                             ("t_compute_building_us", "compute building from ring", "#2ca02c")]:
        med, err = _errbar(summary, key)
        axes[1].errorbar(ks, med, yerr=err, marker="o", capsize=4, label=name, color=color)
    axes[1].set_title("Search (average of 1,000 lookups)")
    axes[1].set_ylabel("Time per lookup (microseconds)")
    axes[1].legend()
    for ax in axes:
        ax.set_xscale("log")
        if ax is not axes[1]:  # a lookup takes 1-3 microseconds, a linear axis is easier to read
            ax.set_yscale("log")
        ax.set_xlabel("Number of guests K")
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"{title}\n{_caption(rows)}")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_a_time_vs_K.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for key, name, color in [("mem_total_current_mib", "ring + guests: current", "#1f77b4"),
                             ("mem_total_peak_mib", "ring + guests: peak", "#ff7f0e"),
                             ("mem_ring_current_mib", "ring only: current", "#2ca02c")]:
        med, err = _errbar(summary, key)
        ax.errorbar(ks, med, yerr=err, marker="o", capsize=4, label=name, color=color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of guests K")
    ax.set_ylabel("Python-traced memory added (MiB)")
    ax.set_title(f"{title}\n{_caption(rows)}\n(tracemalloc: Python-tracked memory, not the whole process)", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_a_memory_vs_K.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return paths


def plot_b(raw_path: str, out_dir: str) -> List[str]:
    rows = _prep(read_rows(raw_path), ["N", "K", "V", *B_VALUE_KEYS])
    k, v = int(rows[0]["K"]), int(rows[0]["V"])
    paths = []

    # ---- migration rate vs N for both methods
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, operation, title in [(axes[0], "add", "Add 1 building"), (axes[1], "remove", "Remove 1 building")]:
        ring_rows = [r for r in rows if r["operation"] == operation and r["method"] == "range"]
        modn_rows = [r for r in rows if r["operation"] == operation and r["method"] == "modn"]
        for subset, label, color, marker in [(ring_rows, "Consistent hashing (V=%d)" % v, COLORS["ring"], "o"),
                                             (modn_rows, "hash mod N", COLORS["modn"], "s")]:
            summary = summarize(subset, ["N"], ["rate"])
            med, err = _errbar(summary, "rate")
            ax.errorbar([r["N"] for r in summary], med, yerr=err, marker=marker, capsize=4,
                        label=label + " (measured)", color=color)
        ns = sorted({int(r["N"]) for r in rows})
        if operation == "add":
            ring_expected = [1 / (n + 1) for n in ns]
            modn_expected = [n / (n + 1) for n in ns]
        else:
            ring_expected = [1 / n for n in ns]
            modn_expected = [(n - 1) / n for n in ns]
        ax.plot(ns, ring_expected, "--", color=COLORS["ring"], alpha=0.6,
                label="expected: 1/(N+1)" if operation == "add" else "expected: 1/N")
        ax.plot(ns, modn_expected, "--", color=COLORS["modn"], alpha=0.6,
                label="expected: N/(N+1)" if operation == "add" else "expected: (N-1)/N")
        ax.set_title(title)
        ax.set_xlabel("Number of buildings N before the change")
        ax.set_xticks(ns)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="center right")
    axes[0].set_ylabel("Migration rate = moved guests / K (fraction)")
    fig.suptitle(f"Experiment B: K={k:,}, V={v}\n{_caption(rows)}")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_b_migration_rate_vs_N.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # ---- time of the add / remove building commands
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    labels = {"scan": "Consistent hashing, method A (scan all K)", "range": "Consistent hashing, method B (affected range)",
              "modn": "hash mod N"}
    for ax, operation, title in [(axes[0], "add", "Add 1 building"), (axes[1], "remove", "Remove 1 building")]:
        for method in ("scan", "range", "modn"):
            subset = [r for r in rows if r["operation"] == operation and r["method"] == method]
            summary = summarize(subset, ["N"], ["time_s"])
            med, err = _errbar(summary, "time_s")
            med_ms = [m * 1000 for m in med]
            err_ms = [[x * 1000 for x in err[0]], [x * 1000 for x in err[1]]]
            ax.errorbar([r["N"] for r in summary], med_ms, yerr=err_ms, marker="o", capsize=4,
                        label=labels[method], color=COLORS[method])
        ax.set_title(title)
        ax.set_xlabel("Number of buildings N before the change")
        ax.set_xticks(sorted({int(r["N"]) for r in rows}))
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Time of the whole command (ms)")
    fig.suptitle(f"Experiment B: K={k:,}, V={v}\n{_caption(rows)}")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_b_time_vs_N.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return paths


def plot_c(raw_path: str, out_dir: str) -> List[str]:
    rows = _prep(read_rows(raw_path), ["V", "N", "K", *C_VALUE_KEYS])
    summary = summarize(rows, ["V"], C_VALUE_KEYS)
    vs = [int(r["V"]) for r in summary]
    n, k = int(rows[0]["N"]), int(rows[0]["K"])
    paths = []

    # ---- effect of V on CV and memory
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    med, err = _errbar(summary, "cv")
    axes[0].errorbar(vs, med, yerr=err, marker="o", capsize=4, color=COLORS["ring"])
    axes[0].set_ylabel("Load imbalance CV = std / mean (dimensionless)")
    axes[0].set_title("Balance (lower is better)")
    for key, name, color in [("mem_total_current_mib", "ring + guests: current", "#1f77b4"),
                             ("mem_total_peak_mib", "ring + guests: peak", "#ff7f0e"),
                             ("mem_ring_current_mib", "ring only: current", "#2ca02c")]:
        med, err = _errbar(summary, key)
        axes[1].errorbar(vs, med, yerr=err, marker="o", capsize=4, label=name, color=color)
    axes[1].set_ylabel("Python-traced memory added (MiB, log scale)")
    axes[1].set_yscale("log")  # the ring is many times smaller than the guest data, so it needs a log axis to show the effect of V
    axes[1].set_title("Memory (tracemalloc)")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(vs)
        ax.set_xticklabels([str(v) for v in vs])
        ax.set_xlabel("Virtual nodes per building V")
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"Experiment C: N={n}, K={k:,}\n{_caption(rows)}")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_c_V_vs_cv_memory.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # ---- migration rate vs V
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for key, name, color in [("add_rate", "add 1 building", "#1f77b4"), ("remove_rate", "remove 1 building", "#d62728")]:
        med, err = _errbar(summary, key)
        ax.errorbar(vs, med, yerr=err, marker="o", capsize=4, label=name, color=color)
    ax.axhline(1 / (n + 1), ls="--", color="#1f77b4", alpha=0.5, label="expected add 1/(N+1)")
    ax.axhline(1 / n, ls="--", color="#d62728", alpha=0.5, label="expected remove 1/N")
    ax.set_xscale("log", base=2)
    ax.set_xticks(vs)
    ax.set_xticklabels([str(v) for v in vs])
    ax.set_xlabel("Virtual nodes per building V")
    ax.set_ylabel("Migration rate = moved guests / K (fraction)")
    ax.set_title(f"Experiment C: N={n}, K={k:,}\n{_caption(rows)}", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_c_V_vs_migration.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return paths
