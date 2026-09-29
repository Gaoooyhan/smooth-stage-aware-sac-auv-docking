#!/usr/bin/env python3
"""Create Figure 6 from paired held-out station-wake evaluation data.

Panels (a) and (b) use a deterministically selected representative paired
episode at L5.  Panel (c) summarizes all valid held-out pairs: episode curves
are averaged within each model seed, followed by the across-seed mean and 95%
Student-t confidence interval.  The cumulative endpoint at k=5 corresponds to
the transition action-variation metric reported in Table 6.

Expected input files inside --data-dir:
  transition_terminal_per_episode.csv
  transition_terminal_per_step.csv

Example
-------
python fig6_paired_transition_behavior.py \
  --data-dir figures/station_wake_heldout_confirmation \
  --condition station_wake_L5 \
  --output-dir figures
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


HARD_METHOD = "Hard-switch stage-aware SAC"
SMOOTH_METHOD = "Smooth stage-aware SAC"
METHOD_ORDER = (HARD_METHOD, SMOOTH_METHOD)

COLORS = {
    HARD_METHOD: "#B45B57",
    SMOOTH_METHOD: "#176F95",
}
LINESTYLES = {
    HARD_METHOD: (0, (5, 2.5)),
    SMOOTH_METHOD: "-",
}
MARKERS = {
    HARD_METHOD: "s",
    SMOOTH_METHOD: "o",
}

STAGE_BOUNDARY_M = 0.50
WINDOW_RADIUS = 5
SELECTION_METRIC = "transition_sum_action_delta_norm"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 9.3,
            "axes.labelsize": 9.5,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#33424B",
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Re-run evaluate_transition_terminal_metrics_v1.py with "
            "--save-step-data to create the per-step file."
        )
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def as_float(row: dict[str, str], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid or missing field {field!r} in CSV row") from exc


def as_int(row: dict[str, str], field: str) -> int:
    return int(round(as_float(row, field)))


def pair_key(row: dict[str, str]) -> tuple[int, int, int, int]:
    return (
        as_int(row, "model_seed"),
        as_int(row, "episode"),
        as_int(row, "eval_seed"),
        as_int(row, "disturbance_seed"),
    )


def select_representative_pair(
    episode_rows: list[dict[str, str]],
    condition: str,
    hard_method: str,
    smooth_method: str,
    forced_seed: int | None,
    forced_episode: int | None,
) -> tuple[
    tuple[int, int, int, int],
    dict[str, dict[str, str]],
    float,
    float,
    list[tuple[tuple[int, int, int, int], dict[str, dict[str, str]], float]],
]:
    selected = [row for row in episode_rows if row.get("condition") == condition]
    if not selected:
        available = sorted({row.get("condition", "") for row in episode_rows})
        raise ValueError(f"Condition {condition!r} not found. Available: {available}")

    by_method: dict[str, dict[tuple[int, int, int, int], dict[str, str]]] = {
        hard_method: {},
        smooth_method: {},
    }
    for row in selected:
        method = row.get("method")
        if method in by_method:
            by_method[method][pair_key(row)] = row

    common_keys = sorted(set(by_method[hard_method]) & set(by_method[smooth_method]))
    valid: list[
        tuple[tuple[int, int, int, int], dict[str, dict[str, str]], float]
    ] = []
    for key in common_keys:
        rows = {
            hard_method: by_method[hard_method][key],
            smooth_method: by_method[smooth_method][key],
        }
        if not all(as_int(rows[m], "transition_available") == 1 for m in METHOD_ORDER):
            continue
        hard_value = as_float(rows[hard_method], SELECTION_METRIC)
        smooth_value = as_float(rows[smooth_method], SELECTION_METRIC)
        if np.isfinite(hard_value) and np.isfinite(smooth_value):
            valid.append((key, rows, smooth_value - hard_value))

    if not valid:
        raise ValueError(
            f"No valid paired transition episodes for condition {condition!r}."
        )

    differences = np.asarray([item[2] for item in valid], dtype=float)
    median_difference = float(np.median(differences))

    if (forced_seed is None) != (forced_episode is None):
        raise ValueError("Use --model-seed and --episode together, or omit both.")

    if forced_seed is not None:
        candidates = [
            item
            for item in valid
            if item[0][0] == forced_seed and item[0][1] == forced_episode
        ]
        if not candidates:
            raise ValueError(
                f"Requested seed={forced_seed}, episode={forced_episode} is not "
                "a valid paired transition episode."
            )
        chosen = candidates[0]
    else:
        chosen = min(
            valid,
            key=lambda item: (
                abs(item[2] - median_difference),
                item[0],
            ),
        )

    return chosen[0], chosen[1], chosen[2], median_difference, valid


def extract_transition_window(
    step_rows: list[dict[str, str]],
    condition: str,
    method: str,
    key: tuple[int, int, int, int],
    episode_row: dict[str, str],
    require_complete: bool = True,
) -> dict[str, np.ndarray]:
    model_seed, episode, eval_seed, disturbance_seed = key
    rows = [
        row
        for row in step_rows
        if row.get("condition") == condition
        and row.get("method") == method
        and as_int(row, "model_seed") == model_seed
        and as_int(row, "episode") == episode
        and as_int(row, "eval_seed") == eval_seed
        and as_int(row, "disturbance_seed") == disturbance_seed
    ]
    rows.sort(key=lambda row: as_int(row, "step"))
    if not rows:
        raise ValueError(f"No per-step rows found for {method}, pair={key}.")

    crossing_step = as_int(episode_row, "first_transition_step")
    expected = np.arange(-WINDOW_RADIUS, WINDOW_RADIUS + 1)
    window_by_k = {
        as_int(row, "step") - crossing_step: row
        for row in rows
        if -WINDOW_RADIUS <= as_int(row, "step") - crossing_step <= WINDOW_RADIUS
    }
    available = np.asarray(sorted(window_by_k), dtype=int)
    complete = len(window_by_k) == len(expected) and np.array_equal(
        available, expected
    )
    if require_complete and not complete:
        raise ValueError(
            f"Incomplete 11-step transition window for {method}, pair={key}: "
            f"found relative steps {available.tolist()}"
        )

    def aligned(field: str, missing_value: float) -> np.ndarray:
        return np.asarray(
            [
                as_float(window_by_k[k], field) if k in window_by_k else missing_value
                for k in expected
            ],
            dtype=float,
        )

    return {
        "k": expected,
        "relative_distance": aligned("dist_xyz_m", np.nan) - STAGE_BOUNDARY_M,
        "stage_weight": aligned("stage_weight", np.nan),
        # The evaluator clips its 11-step window at episode boundaries.  A
        # nonexistent step contributes no action increment, so zero padding
        # preserves the exact per-episode sum reported in Table 6.
        "action_change": aligned("action_delta_norm", 0.0),
    }


def t_critical_95(df: int) -> float:
    """Two-sided 95% Student-t critical value for small seed counts."""
    values = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
    }
    return values.get(df, 1.96)


def build_cumulative_summary(
    step_rows: list[dict[str, str]],
    condition: str,
    valid_pairs: list[
        tuple[tuple[int, int, int, int], dict[str, dict[str, str]], float]
    ],
) -> dict[str, dict[str, np.ndarray | int]]:
    """Average episodes within seed, then summarize the seed-level curves."""
    curves_by_seed: dict[str, dict[int, list[np.ndarray]]] = {
        method: {} for method in METHOD_ORDER
    }
    for key, paired_rows, _ in valid_pairs:
        model_seed = key[0]
        for method in METHOD_ORDER:
            trace = extract_transition_window(
                step_rows=step_rows,
                condition=condition,
                method=method,
                key=key,
                episode_row=paired_rows[method],
                require_complete=False,
            )
            cumulative = np.cumsum(trace["action_change"])
            reported_total = as_float(paired_rows[method], SELECTION_METRIC)
            if not np.isclose(cumulative[-1], reported_total, rtol=0.0, atol=1e-9):
                raise ValueError(
                    "Per-step cumulative endpoint does not match the per-episode "
                    f"metric for {method}, pair={key}: "
                    f"steps={cumulative[-1]:.12f}, reported={reported_total:.12f}"
                )
            curves_by_seed[method].setdefault(model_seed, []).append(cumulative)

    summary: dict[str, dict[str, np.ndarray | int]] = {}
    for method in METHOD_ORDER:
        seed_means = np.vstack(
            [
                np.mean(np.vstack(curves_by_seed[method][seed]), axis=0)
                for seed in sorted(curves_by_seed[method])
            ]
        )
        n_seeds = seed_means.shape[0]
        mean = np.mean(seed_means, axis=0)
        if n_seeds >= 2:
            sem = np.std(seed_means, axis=0, ddof=1) / np.sqrt(n_seeds)
            ci95 = t_critical_95(n_seeds - 1) * sem
        else:
            ci95 = np.zeros_like(mean)
        summary[method] = {
            "k": np.arange(-WINDOW_RADIUS, WINDOW_RADIUS + 1),
            "mean": mean,
            "ci95": ci95,
            "n_seeds": n_seeds,
            "n_pairs": sum(len(items) for items in curves_by_seed[method].values()),
        }
    return summary


def style_panel(ax: plt.Axes, letter: str) -> None:
    ax.axvline(0, color="#667780", linewidth=1.0, linestyle=(0, (3, 2)), zorder=1)
    ax.grid(axis="y", color="#D9E1E5", linewidth=0.65, alpha=0.80)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.012,
        0.91,
        f"({letter})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        color="#27343C",
    )


def plot_method(
    ax: plt.Axes,
    trace: dict[str, np.ndarray],
    field: str,
    method: str,
    fill: bool = False,
) -> None:
    color = COLORS[method]
    line = ax.plot(
        trace["k"],
        trace[field],
        color=color,
        linewidth=2.15 if method == SMOOTH_METHOD else 1.75,
        linestyle=LINESTYLES[method],
        marker=MARKERS[method],
        markersize=4.8,
        markerfacecolor="white" if method == HARD_METHOD else color,
        markeredgecolor=color,
        markeredgewidth=0.9,
        zorder=4,
    )[0]
    if fill:
        ax.fill_between(
            trace["k"],
            0.0,
            trace[field],
            color=color,
            alpha=0.075,
            zorder=2,
        )
    return line


def make_figure(
    traces: dict[str, dict[str, np.ndarray]],
    cumulative_summary: dict[str, dict[str, np.ndarray | int]],
    output_dir: Path,
    basename: str,
) -> None:
    configure_style()
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.35, 6.65),
        sharex=True,
        constrained_layout=False,
    )

    for letter, ax in zip("abc", axes):
        style_panel(ax, letter)
        ax.set_xlim(-5.25, 5.75)
        ax.set_xticks(np.arange(-5, 6, 1))

    for method in METHOD_ORDER:
        plot_method(axes[0], traces[method], "relative_distance", method)
        plot_method(axes[1], traces[method], "stage_weight", method)

    for method in METHOD_ORDER:
        summary = cumulative_summary[method]
        k = np.asarray(summary["k"], dtype=float)
        mean = np.asarray(summary["mean"], dtype=float)
        ci95 = np.asarray(summary["ci95"], dtype=float)
        color = COLORS[method]
        axes[2].fill_between(
            k,
            np.maximum(0.0, mean - ci95),
            mean + ci95,
            color=color,
            alpha=0.13,
            linewidth=0,
            zorder=2,
        )
        axes[2].plot(
            k,
            mean,
            color=color,
            linewidth=2.25 if method == SMOOTH_METHOD else 1.85,
            linestyle=LINESTYLES[method],
            marker=MARKERS[method],
            markersize=4.8,
            markerfacecolor="white" if method == HARD_METHOD else color,
            markeredgecolor=color,
            markeredgewidth=0.9,
            zorder=4,
        )
        axes[2].annotate(
            f"{mean[-1]:.4f}",
            xy=(k[-1], mean[-1]),
            xytext=(6, 7 if method == HARD_METHOD else -9),
            textcoords="offset points",
            ha="left",
            va="bottom" if method == HARD_METHOD else "top",
            fontsize=8.0,
            color=color,
            fontweight="semibold",
            clip_on=False,
        )

    axes[0].axhline(0.0, color="#80919A", linewidth=0.9, linestyle=(0, (4, 2)))
    axes[0].set_ylabel("Relative distance\n" r"$d_t-d_{\mathrm{sw}}$ (m)")
    axes[0].text(
        0.985,
        0.88,
        "outside",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color="#71818A",
    )
    axes[0].text(
        0.985,
        0.10,
        "inside",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#71818A",
    )

    axes[1].set_ylabel("Stage weight\n$g_t$")
    axes[1].set_ylim(-0.055, 1.055)
    axes[1].set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])

    axes[2].set_ylabel("Cumulative action variation\n$C(k)$")
    axes[2].set_xlabel("Relative step $k$", labelpad=7)
    axes[2].set_ylim(bottom=0.0)
    axes[2].text(
        0.14,
        0.91,
        "mean $\\pm$ 95% CI across seed-level means",
        transform=axes[2].transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        color="#65757E",
    )

    axes[0].annotate(
        "first inward crossing",
        xy=(0, axes[0].get_ylim()[1]),
        xytext=(7, -4),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=7.5,
        color="#61727B",
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=2.1,
            marker=MARKERS[method],
            markersize=5.2,
            markerfacecolor="white" if method == HARD_METHOD else COLORS[method],
            markeredgecolor=COLORS[method],
            label=("Hard-switch" if method == HARD_METHOD else "Smooth stage-aware"),
        )
        for method in METHOD_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.995),
        ncol=2,
        frameon=False,
        handlelength=3.0,
        columnspacing=2.0,
    )
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.095, top=0.93, hspace=0.25)

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        path = output_dir / f"{basename}.{suffix}"
        options = dict(bbox_inches="tight", pad_inches=0.04)
        if suffix == "png":
            options["dpi"] = 600
        fig.savefig(path, **options)
        print(f"Saved: {path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("figures/station_wake_heldout_confirmation"),
        help="Directory containing transition_terminal_per_episode.csv and per_step.csv.",
    )
    parser.add_argument("--condition", default="station_wake_L5")
    parser.add_argument("--hard-method", default=HARD_METHOD)
    parser.add_argument("--smooth-method", default=SMOOTH_METHOD)
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--basename", default="fig6_paired_transition_behavior")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global HARD_METHOD, SMOOTH_METHOD, METHOD_ORDER, COLORS, LINESTYLES, MARKERS

    if args.hard_method != HARD_METHOD or args.smooth_method != SMOOTH_METHOD:
        old_hard, old_smooth = HARD_METHOD, SMOOTH_METHOD
        COLORS = {
            args.hard_method: COLORS[old_hard],
            args.smooth_method: COLORS[old_smooth],
        }
        LINESTYLES = {
            args.hard_method: LINESTYLES[old_hard],
            args.smooth_method: LINESTYLES[old_smooth],
        }
        MARKERS = {
            args.hard_method: MARKERS[old_hard],
            args.smooth_method: MARKERS[old_smooth],
        }
        HARD_METHOD = args.hard_method
        SMOOTH_METHOD = args.smooth_method
        METHOD_ORDER = (HARD_METHOD, SMOOTH_METHOD)

    episode_rows = read_csv(args.data_dir / "transition_terminal_per_episode.csv")
    step_rows = read_csv(args.data_dir / "transition_terminal_per_step.csv")

    (
        key,
        selected_rows,
        chosen_difference,
        median_difference,
        valid_pairs,
    ) = select_representative_pair(
        episode_rows=episode_rows,
        condition=args.condition,
        hard_method=HARD_METHOD,
        smooth_method=SMOOTH_METHOD,
        forced_seed=args.model_seed,
        forced_episode=args.episode,
    )
    traces = {
        method: extract_transition_window(
            step_rows=step_rows,
            condition=args.condition,
            method=method,
            key=key,
            episode_row=selected_rows[method],
        )
        for method in METHOD_ORDER
    }
    cumulative_summary = build_cumulative_summary(
        step_rows=step_rows,
        condition=args.condition,
        valid_pairs=valid_pairs,
    )

    print(
        "Selected paired episode: "
        f"model_seed={key[0]}, episode={key[1]}, eval_seed={key[2]}, "
        f"disturbance_seed={key[3]}"
    )
    print(
        "Smooth - Hard transition action-variation difference: "
        f"selected={chosen_difference:.6f}, all-pair median={median_difference:.6f}"
    )
    print(
        f"Wake condition: {args.condition}; "
        f"speed={as_float(selected_rows[HARD_METHOD], 'wake_speed_m_s'):.3f} m/s"
    )
    for method in METHOD_ORDER:
        summary = cumulative_summary[method]
        print(
            f"{method}: cumulative endpoint={np.asarray(summary['mean'])[-1]:.6f}, "
            f"paired episodes={summary['n_pairs']}, seeds={summary['n_seeds']}"
        )
    make_figure(traces, cumulative_summary, args.output_dir, args.basename)


if __name__ == "__main__":
    main()