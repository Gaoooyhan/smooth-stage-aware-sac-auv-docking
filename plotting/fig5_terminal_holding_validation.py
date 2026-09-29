#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot the refined Figure 5 for terminal-holding validation.

Input:
    CSV trace exported by
    evaluate_terminal_holding_validation_v1.py --trace-only

Output:
    fig5_terminal_holding_validation.(png|pdf|svg)

Design goals:
    - Keep Figure 5 focused on why consecutive terminal holding is needed.
    - Show only three subplots: position error, yaw error, and holding counter.
    - Put event labels only in panel (c) to reduce clutter.
    - Keep horizontal threshold lines, but remove text labels in panels (a) and (b).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


POSITION_TOL_M = 0.10
YAW_TOL_DEG = 5.0
HOLD_STEPS = 5


def read_trace_csv(path: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty trace CSV: {path}")
    return rows


def as_int_array(rows, key):
    return np.array([int(float(row[key])) for row in rows], dtype=int)


def as_float_array(rows, key):
    return np.array([float(row[key]) for row in rows], dtype=float)


def first_marked_step(rows, event_key):
    if event_key not in rows[0]:
        return None
    for row in rows:
        if int(float(row[event_key])) == 1:
            return int(float(row["step"]))
    return None


def rel_step(step: int | None, first_entry_step: int) -> int | None:
    if step is None:
        return None
    return int(step - first_entry_step)


def add_event_lines(ax, events):
    for x, _label, linestyle in events:
        if x is None:
            continue
        ax.axvline(x, linewidth=1.0, linestyle=linestyle, alpha=0.75)


def annotate_panel_c(ax, events):
    label_y = {
        "first entry": 4.75,
        "exit / reset": 2.45,
        "re-entry": 4.05,
        "strict success": 2.85,
    }
    x_shift = {
        "first entry": 0.18,
        "exit / reset": 0.18,
        "re-entry": 0.18,
        "strict success": 0.18,
    }

    for x, label, _linestyle in events:
        if x is None:
            continue
        ax.text(
            x + x_shift.get(label, 0.18),
            label_y.get(label, 4.4),
            label,
            rotation=90,
            va="top",
            ha="left",
            fontsize=8,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-csv", required=True)
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--prefix", default="fig5_terminal_holding_validation")
    parser.add_argument(
        "--pre-steps",
        type=int,
        default=10,
        help="Number of steps shown before the first terminal entry.",
    )
    parser.add_argument(
        "--post-steps",
        type=int,
        default=6,
        help="Number of steps shown after strict success.",
    )
    args = parser.parse_args()

    trace_path = Path(args.trace_csv)
    if not trace_path.is_file():
        raise FileNotFoundError(trace_path)

    rows = read_trace_csv(trace_path)
    available = set(rows[0].keys())
    required = {"step", "dist_xyz_m", "yaw_error_deg", "hold_count", "terminal_now"}
    missing = sorted(required.difference(available))
    if missing:
        raise ValueError(f"Missing required trace columns: {missing}")

    step = as_int_array(rows, "step")
    dist_xyz = as_float_array(rows, "dist_xyz_m")
    yaw_deg = as_float_array(rows, "yaw_error_deg")
    hold_count = as_int_array(rows, "hold_count")
    terminal_now = as_int_array(rows, "terminal_now")

    first_entry_step = first_marked_step(rows, "first_entry_event")
    if first_entry_step is None:
        idx = np.flatnonzero(terminal_now == 1)
        if idx.size == 0:
            raise RuntimeError("No terminal entry found in the trace.")
        first_entry_step = int(step[idx[0]])

    first_exit_step = first_marked_step(rows, "first_exit_event")
    first_reentry_step = first_marked_step(rows, "first_reentry_event")
    strict_step = first_marked_step(rows, "strict_success_event")

    if strict_step is None and "strict_success" in available:
        strict_success = as_int_array(rows, "strict_success")
        idx = np.flatnonzero(strict_success == 1)
        if idx.size:
            strict_step = int(step[idx[0]])

    if strict_step is None:
        raise RuntimeError("The selected trace does not reach strict success.")

    relative_step = step - int(first_entry_step)
    xmin = -int(args.pre_steps)
    xmax = int(strict_step - first_entry_step) + int(args.post_steps)
    mask = (relative_step >= xmin) & (relative_step <= xmax)
    if not np.any(mask):
        raise RuntimeError("Requested plotting window is empty.")

    x = relative_step[mask]
    y_pos = dist_xyz[mask]
    y_yaw = yaw_deg[mask]
    y_hold = hold_count[mask]

    events = [
        (rel_step(first_entry_step, first_entry_step), "first entry", "--"),
        (rel_step(first_exit_step, first_entry_step), "exit / reset", ":"),
        (rel_step(first_reentry_step, first_entry_step), "re-entry", "-."),
        (rel_step(strict_step, first_entry_step), "strict success", "--"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(7.1, 7.6), sharex=True)

    # (a) Position error
    ax = axes[0]
    ax.plot(x, y_pos, linewidth=1.8, marker="o", markersize=3.6)
    ax.axhline(POSITION_TOL_M, linewidth=1.1, linestyle="--")
    add_event_lines(ax, events)
    ax.set_ylabel("Position error (m)")
    ax.set_ylim(bottom=max(0.0, min(y_pos) - 0.004), top=max(max(y_pos) + 0.01, 0.105))
    ax.grid(True, linewidth=0.6, alpha=0.28)
    ax.text(
        0.5,
        -0.23,
        "(a) Position error",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )

    # (b) Yaw error
    ax = axes[1]
    ax.plot(x, y_yaw, linewidth=1.8, marker="o", markersize=3.6)
    ax.axhline(YAW_TOL_DEG, linewidth=1.1, linestyle="--")
    add_event_lines(ax, events)
    ax.set_ylabel("Yaw error (°)")
    ax.set_ylim(bottom=0.0, top=max(max(y_yaw) + 0.8, 5.8))
    ax.grid(True, linewidth=0.6, alpha=0.28)
    ax.text(
        0.5,
        -0.23,
        "(b) Yaw error",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )

    # (c) Consecutive holding counter
    ax = axes[2]
    ax.step(x, y_hold, where="post", linewidth=1.9)
    ax.scatter(x, y_hold, s=16)
    ax.axhline(HOLD_STEPS, linewidth=1.1, linestyle="--")
    add_event_lines(ax, events)
    annotate_panel_c(ax, events)
    ax.set_ylabel("Holding counter")
    ax.set_xlabel("Relative step from first terminal entry")
    ax.set_yticks(range(0, HOLD_STEPS + 1))
    ax.set_ylim(-0.25, HOLD_STEPS + 0.7)
    ax.grid(True, linewidth=0.6, alpha=0.28)
    ax.text(
        0.5,
        -0.27,
        "(c) Consecutive holding",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )
    ax.text(
        0.90,
        HOLD_STEPS,
        r"$N_{\mathrm{req}} = 5$",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=8,
    )

    for ax in axes:
        ax.set_xlim(xmin, xmax)

    fig.subplots_adjust(left=0.14, right=0.985, top=0.985, bottom=0.085, hspace=0.52)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for ext in ("png", "pdf", "svg"):
        out_path = output_dir / f"{args.prefix}.{ext}"
        if ext == "png":
            fig.savefig(out_path, dpi=600, bbox_inches="tight")
        else:
            fig.savefig(out_path, bbox_inches="tight")
        saved_paths.append(out_path)

    plt.close(fig)

    print("Saved:")
    for p in saved_paths:
        print(f"  {p}")

    print("Events:")
    print(f"  first entry   : step {first_entry_step}")
    print(f"  first exit    : step {first_exit_step}")
    print(f"  first re-entry: step {first_reentry_step}")
    print(f"  strict success: step {strict_step}")


if __name__ == "__main__":
    main()