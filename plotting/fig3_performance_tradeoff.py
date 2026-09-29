#!/usr/bin/env python3
"""Draw Fig. 3 as a 2-by-2 four-metric benchmark matrix.

The numerical values reproduce the enlarged-initial-state rows of Table 5.
Each method occupies one row, so reliability, terminal accuracy, and docking
efficiency can be compared without a legend or overlapping annotations.

Run
---
python fig3_performance_tradeoff.py --output-dir figures

Outputs
-------
fig3_performance_comparison.png  (600 dpi)
fig3_performance_comparison.pdf  (vector)
fig3_performance_comparison.svg  (vector)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

# Use a non-interactive backend on headless Linux servers.
mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# Enlarged-initial-state mean values from Table 5.
METHODS = {
    "Proposed": dict(success=1.000, xyz=0.0657, yaw=1.502, steps=55.31),
    "Precision SAC": dict(success=1.000, xyz=0.0648, yaw=1.630, steps=55.37),
    "ARSPPO (adapted)": dict(success=1.000, xyz=0.0489, yaw=1.602, steps=55.43),
    "Cascaded PID": dict(success=1.000, xyz=0.0678, yaw=0.417, steps=71.39),
    "Baseline SAC": dict(success=0.998, xyz=0.0672, yaw=1.747, steps=58.57),
    "Standard PPO": dict(success=0.738, xyz=0.3041, yaw=7.007, steps=256.92),
    "TD3": dict(success=0.414, xyz=0.4095, yaw=10.728, steps=559.30),
}

STYLE = {
    "Proposed": dict(color="#155F8A", marker="*"),
    "Precision SAC": dict(color="#4C96A5", marker="s"),
    "ARSPPO (adapted)": dict(color="#7667A6", marker="D"),
    "Cascaded PID": dict(color="#779554", marker="P"),
    "Baseline SAC": dict(color="#8D9AA4", marker="o"),
    "Standard PPO": dict(color="#D08055", marker="v"),
    "TD3": dict(color="#B55362", marker="X"),
}

PANELS = (
    dict(
        key="success",
        title="Strict success rate",
        xlabel="Success rate",
        xlim=(0.0, 1.08),
        ticks=(0.0, 0.25, 0.50, 0.75, 1.00),
        fmt=lambda value: f"{value:.3f}",
    ),
    dict(
        key="xyz",
        title="Final XYZ error",
        xlabel="Error (m)",
        xlim=(0.0, 0.46),
        ticks=(0.0, 0.1, 0.2, 0.3, 0.4),
        fmt=lambda value: f"{value:.4f}",
        threshold=0.10,
        threshold_label="0.10 m threshold",
    ),
    dict(
        key="yaw",
        title="Final yaw error",
        xlabel="Error (°)",
        xlim=(0.0, 12.2),
        ticks=(0.0, 3.0, 6.0, 9.0, 12.0),
        fmt=lambda value: f"{value:.3f}",
        threshold=5.0,
        threshold_label="5° threshold",
    ),
    dict(
        key="steps",
        title="Completion steps",
        xlabel="Steps",
        xlim=(0.0, 625.0),
        ticks=(0, 150, 300, 450, 600),
        fmt=lambda value: f"{value:.2f}",
    ),
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 8.8,
            "axes.linewidth": 0.75,
            "axes.edgecolor": "#35424B",
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 9.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.0,
            "ytick.major.size": 0.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_axis(ax: plt.Axes, show_method_labels: bool, y: np.ndarray) -> None:
    ax.set_ylim(len(METHODS) - 0.45, -0.55)
    ax.set_yticks(y)
    ax.tick_params(axis="y", pad=7)
    ax.grid(axis="x", color="#DCE3E8", linewidth=0.65, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    if not show_method_labels:
        ax.set_yticklabels([])

    # One subtle row highlight carries the main visual emphasis.
    ax.axhspan(-0.42, 0.42, color="#E6F1F7", alpha=0.95, zorder=0)


def add_value_label(
    ax: plt.Axes,
    value: float,
    row: int,
    text: str,
    xlim: tuple[float, float],
    color: str,
    emphasized: bool,
) -> None:
    span = xlim[1] - xlim[0]
    place_left = value > xlim[0] + 0.83 * span
    offset = -5 if place_left else 5
    ax.annotate(
        text,
        xy=(value, row),
        xytext=(offset, 0),
        textcoords="offset points",
        ha="right" if place_left else "left",
        va="center",
        fontsize=7.5,
        color=color if emphasized else "#46535C",
        fontweight="bold" if emphasized else "normal",
        clip_on=False,
        zorder=6,
    )


def draw_panel(ax: plt.Axes, panel: dict, index: int) -> None:
    names = list(METHODS)
    y = np.arange(len(names))
    show_method_labels = index % 2 == 0
    style_axis(ax, show_method_labels, y)

    ax.set_xlim(panel["xlim"])
    ax.set_xticks(panel["ticks"])
    ax.set_xlabel(panel["xlabel"], labelpad=6)

    if "threshold" in panel:
        ax.axvline(
            panel["threshold"],
            color="#8F9AA1",
            linewidth=0.9,
            linestyle=(0, (3, 2)),
            zorder=1,
        )
        ax.text(
            panel["threshold"],
            -0.54,
            panel["threshold_label"],
            ha="left",
            va="bottom",
            fontsize=6.9,
            color="#75828A",
        )

    for row, method in enumerate(names):
        value = METHODS[method][panel["key"]]
        color = STYLE[method]["color"]
        marker = STYLE[method]["marker"]
        proposed = method == "Proposed"

        # Lollipop geometry makes magnitude readable while retaining compactness.
        ax.hlines(
            row,
            panel["xlim"][0],
            value,
            color=color,
            linewidth=2.2 if proposed else 1.35,
            alpha=0.82 if proposed else 0.55,
            zorder=2,
        )
        if proposed:
            ax.scatter(
                value,
                row,
                s=180,
                color="#BBD8E7",
                alpha=0.68,
                linewidth=0,
                zorder=3,
            )
        ax.scatter(
            value,
            row,
            s=82 if proposed else 48,
            marker=marker,
            facecolor=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        add_value_label(
            ax,
            value,
            row,
            panel["fmt"](value),
            panel["xlim"],
            color,
            proposed,
        )

    if show_method_labels:
        labels = ax.set_yticklabels(names)
        for label, method in zip(labels, names):
            label.set_color(STYLE[method]["color"] if method == "Proposed" else "#29343B")
            label.set_fontweight("bold" if method == "Proposed" else "normal")

    # Journal-style subfigure caption centered below each panel.
    ax.text(
        0.5,
        -0.25,
        f"({chr(97 + index)}) {panel['title']}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.4,
        fontweight="semibold",
        color="#26323A",
    )


def make_figure(output_dir: Path, basename: str) -> None:
    configure_style()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9.4, 7.3),
        constrained_layout=False,
    )

    for index, (ax, panel) in enumerate(zip(axes.flat, PANELS)):
        draw_panel(ax, panel, index)

    fig.subplots_adjust(
        left=0.19,
        right=0.985,
        bottom=0.10,
        top=0.975,
        wspace=0.18,
        hspace=0.58,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        path = output_dir / f"{basename}.{suffix}"
        save_options = dict(bbox_inches="tight", pad_inches=0.04)
        if suffix == "png":
            save_options["dpi"] = 600
        fig.savefig(path, **save_options)
        print(f"Saved: {path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for PNG/PDF/SVG outputs (default: figures).",
    )
    parser.add_argument(
        "--basename",
        default="fig3_performance_comparison",
        help="Output filename without extension.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_figure(args.output_dir, args.basename)
