#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Harder stress-generalization evaluation for strict SAC recovery policies.

This script does NOT train. It only evaluates the already trained 15 strict models:
  - strict_baseline_v1, seeds 0-4
  - strict_precision_v2_v1, seeds 0-4
  - strict_two_stage_v3_v1, seeds 0-4

It manually sets harder initial states after env.reset(), while keeping the same
4-DOF dynamics, action scaling, and strict terminal criterion:
  XYZ error < 0.10 m
  yaw error < 5 deg
  hold steps >= 5

The purpose is to test whether smooth two-stage v3 keeps better behavior under
larger initial distance, larger initial yaw error, and stronger disturbances.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import auv_rl.envs.auv_env_strict_v1  # noqa: F401


LOG_STD_MAX = 2
LOG_STD_MIN = -5

METHODS = [
    ("Strict baseline", "strict_baseline_v1", "baseline"),
    ("Strict precision v2", "strict_precision_v2_v1", "precision_v2"),
    ("Smooth two-stage v3", "strict_two_stage_v3_v1", "two_stage_v3"),
]

# Two difficulty levels:
# stress4: harder but still reasonable.
# stress5: stronger stress test. If all methods collapse, do not use it as main evidence.
CONDITIONS = [
    {
        "condition": "stress4_nominal",
        "init_xy_range": 3.5,
        "init_z_range": 1.2,
        "init_yaw_range_deg": 120.0,
        "target_xy_range": 1.0,
        "target_z_range": 0.5,
        "min_init_dist": 2.0,
        "obs_pos_noise_std": 0.0,
        "obs_yaw_noise_std_deg": 0.0,
        "current_speed": 0.0,
        "current_random_direction": False,
    },
    {
        "condition": "stress4_combined",
        "init_xy_range": 3.5,
        "init_z_range": 1.2,
        "init_yaw_range_deg": 120.0,
        "target_xy_range": 1.0,
        "target_z_range": 0.5,
        "min_init_dist": 2.0,
        "obs_pos_noise_std": 0.03,
        "obs_yaw_noise_std_deg": 3.0,
        "current_speed": 0.03,
        "current_random_direction": True,
    },
    {
        "condition": "stress5_nominal",
        "init_xy_range": 4.5,
        "init_z_range": 1.5,
        "init_yaw_range_deg": 150.0,
        "target_xy_range": 1.0,
        "target_z_range": 0.5,
        "min_init_dist": 3.0,
        "obs_pos_noise_std": 0.0,
        "obs_yaw_noise_std_deg": 0.0,
        "current_speed": 0.0,
        "current_random_direction": False,
    },
    {
        "condition": "stress5_combined",
        "init_xy_range": 4.5,
        "init_z_range": 1.5,
        "init_yaw_range_deg": 150.0,
        "target_xy_range": 1.0,
        "target_z_range": 0.5,
        "min_init_dist": 3.0,
        "obs_pos_noise_std": 0.04,
        "obs_yaw_noise_std_deg": 4.0,
        "current_speed": 0.04,
        "current_random_direction": True,
    },
]


def wrap_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))

        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, act_dim)
        self.fc_logstd = nn.Linear(256, act_dim)

        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = (
            LOG_STD_MIN
            + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        )
        return mean, log_std

    def deterministic_action(self, obs):
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean, _ = self(x)
            action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action.cpu().numpy()[0]


def load_actor(env, checkpoint: Path):
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    actor = Actor(env)
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    actor.load_state_dict(state, strict=True)
    actor.eval()
    return actor


def write_csv(path: Path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return float(np.mean(values)) if values else float("nan")


def std_sample(values):
    return float(np.std(values, ddof=1)) if len(values) >= 2 else float("nan")


def set_harder_initial_state(env, condition, rng):
    """Overwrite initial state and target after env.reset()."""
    base = env.unwrapped

    init_xy = float(condition["init_xy_range"])
    init_z = float(condition["init_z_range"])
    yaw_range = np.deg2rad(float(condition["init_yaw_range_deg"]))
    target_xy = float(condition["target_xy_range"])
    target_z = float(condition["target_z_range"])
    min_dist = float(condition["min_init_dist"])

    for _ in range(10000):
        x = rng.uniform(-init_xy, init_xy)
        y = rng.uniform(-init_xy, init_xy)
        z = rng.uniform(-init_z, init_z)

        tx = rng.uniform(-target_xy, target_xy)
        ty = rng.uniform(-target_xy, target_xy)
        tz = rng.uniform(-target_z, target_z)

        dist = float(np.sqrt((tx - x) ** 2 + (ty - y) ** 2 + (tz - z) ** 2))
        if dist >= min_dist:
            break
    else:
        raise RuntimeError("Failed to sample a hard initial condition.")

    yaw = rng.uniform(-yaw_range, yaw_range)

    u = rng.uniform(-0.2, 0.2)
    v = rng.uniform(-0.2, 0.2)
    w = rng.uniform(-0.2, 0.2)
    r = rng.uniform(-0.1, 0.1)

    base.state = np.array([x, y, z, yaw, u, v, w, r], dtype=np.float32)
    base.target = np.array([tx, ty, tz, 0.0], dtype=np.float32)
    base.step_count = 0
    base.success_hold_count = 0
    base.prev_action = np.zeros(4, dtype=np.float32)

    dx, dy, dz, dyaw = base._goal_errors()
    dist_xy = float(np.sqrt(dx * dx + dy * dy))
    dist_xyz = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    base.prev_dist_xy = dist_xy
    base.prev_dist_xyz = dist_xyz
    base.prev_abs_dyaw = abs(float(dyaw))

    return base._get_obs(), dist_xyz, abs(float(dyaw))


def make_noisy_obs(obs, condition, rng):
    obs_policy = np.array(obs, dtype=np.float32).copy()

    pos_std = float(condition["obs_pos_noise_std"])
    yaw_std_deg = float(condition["obs_yaw_noise_std_deg"])

    if pos_std > 0.0:
        pos_noise = rng.normal(
            loc=0.0,
            scale=pos_std,
            size=3,
        ).astype(np.float32)
        obs_policy[0:3] += pos_noise
        obs_policy[8:11] -= pos_noise

    if yaw_std_deg > 0.0:
        yaw_noise = np.deg2rad(
            rng.normal(loc=0.0, scale=yaw_std_deg)
        )
        obs_policy[3] = wrap_pi(float(obs_policy[3]) + float(yaw_noise))
        obs_policy[11] = wrap_pi(float(obs[11]) - float(yaw_noise))

    return obs_policy.astype(np.float32)


def evaluate_one_model_condition(
    method,
    exp_name,
    reward_mode,
    model_seed,
    condition,
    eval_episodes,
    eval_seed_start,
    checkpoint_dir,
):
    env = gym.make(
        "AUVRecoveryStrict-v1",
        reward_mode=reward_mode,
        action_normalized=True,
        max_steps=1000,
        stage_switch_dist=0.5,
        hold_steps_required=5,
        current_speed=float(condition["current_speed"]),
        current_random_direction=bool(condition["current_random_direction"]),
        current_direction_rad=0.0,
        current_vertical_speed=0.0,
    )

    checkpoint = checkpoint_dir / (
        f"AUVRecoveryStrict-v1__{exp_name}__seed{model_seed}_actor_final.pt"
    )
    actor = load_actor(env, checkpoint)

    rows = []

    for episode in range(eval_episodes):
        eval_seed = eval_seed_start + episode

        # env.reset sets the current direction when current_random_direction=True.
        obs, reset_info = env.reset(seed=eval_seed)

        hard_seed = (
            2000000
            + 100000 * int(model_seed)
            + 1000 * int(episode)
            + sum(ord(ch) for ch in condition["condition"])
        )
        rng_hard = np.random.default_rng(hard_seed)
        obs, init_dist, init_abs_yaw = set_harder_initial_state(
            env,
            condition,
            rng_hard,
        )

        noise_seed = (
            3000000
            + 100000 * int(model_seed)
            + 1000 * int(episode)
            + sum(ord(ch) for ch in condition["condition"])
        )
        rng_noise = np.random.default_rng(noise_seed)

        terminated = False
        truncated = False
        steps = 0
        info = {}

        while not (terminated or truncated):
            obs_policy = make_noisy_obs(obs, condition, rng_noise)
            action = actor.deterministic_action(obs_policy)
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1

        row = {
            "condition": condition["condition"],
            "method": method,
            "exp_name": exp_name,
            "reward_mode": reward_mode,
            "model_seed": int(model_seed),
            "episode": int(episode),
            "eval_seed": int(eval_seed),
            "init_dist_xyz": float(init_dist),
            "init_abs_yaw_deg": float(np.degrees(init_abs_yaw)),
            "init_xy_range": float(condition["init_xy_range"]),
            "init_z_range": float(condition["init_z_range"]),
            "init_yaw_range_deg": float(condition["init_yaw_range_deg"]),
            "obs_pos_noise_std": float(condition["obs_pos_noise_std"]),
            "obs_yaw_noise_std_deg": float(condition["obs_yaw_noise_std_deg"]),
            "current_speed": float(condition["current_speed"]),
            "current_random_direction": int(bool(condition["current_random_direction"])),
            "current_vx": float(reset_info.get("current_vx", 0.0)),
            "current_vy": float(reset_info.get("current_vy", 0.0)),
            "current_vz": float(reset_info.get("current_vz", 0.0)),
            "success": int(bool(info.get("success", False))),
            "timeout": int(
                bool(truncated)
                and not bool(info.get("success", False))
            ),
            "out_of_bound": int(bool(info.get("out_of_bound", False))),
            "steps": int(steps),
            "final_dist_xy": float(info.get("dist_xy", np.nan)),
            "final_dist_xyz": float(info.get("dist_xyz", np.nan)),
            "final_yaw_error_deg": float(info.get("yaw_error_deg", np.nan)),
            "final_hold_count": int(info.get("success_hold_count", 0)),
            "final_stage_weight": float(info.get("stage_weight", np.nan)),
        }
        rows.append(row)

    env.close()
    return rows


def summarize_seed(rows):
    success_rows = [r for r in rows if r["success"] == 1]
    return {
        "condition": rows[0]["condition"],
        "method": rows[0]["method"],
        "exp_name": rows[0]["exp_name"],
        "reward_mode": rows[0]["reward_mode"],
        "model_seed": rows[0]["model_seed"],
        "eval_episodes": len(rows),
        "success_rate": mean([r["success"] for r in rows]),
        "timeout_rate": mean([r["timeout"] for r in rows]),
        "out_of_bound_rate": mean([r["out_of_bound"] for r in rows]),
        "avg_init_dist": mean([r["init_dist_xyz"] for r in rows]),
        "avg_init_abs_yaw_deg": mean([r["init_abs_yaw_deg"] for r in rows]),
        "avg_steps_all": mean([r["steps"] for r in rows]),
        "avg_steps_success": mean([r["steps"] for r in success_rows]),
        "avg_final_xyz_all": mean([r["final_dist_xyz"] for r in rows]),
        "avg_final_xyz_success": mean([r["final_dist_xyz"] for r in success_rows]),
        "avg_final_yaw_all": mean([r["final_yaw_error_deg"] for r in rows]),
        "avg_final_yaw_success": mean(
            [r["final_yaw_error_deg"] for r in success_rows]
        ),
    }


def summarize_across_seeds(per_seed_rows):
    metrics = [
        "success_rate",
        "timeout_rate",
        "out_of_bound_rate",
        "avg_init_dist",
        "avg_init_abs_yaw_deg",
        "avg_steps_all",
        "avg_steps_success",
        "avg_final_xyz_all",
        "avg_final_xyz_success",
        "avg_final_yaw_all",
        "avg_final_yaw_success",
    ]
    across = []

    for condition in [c["condition"] for c in CONDITIONS]:
        for method, exp_name, reward_mode in METHODS:
            rows = [
                r for r in per_seed_rows
                if r["condition"] == condition and r["method"] == method
            ]
            item = {
                "condition": condition,
                "method": method,
                "exp_name": exp_name,
                "reward_mode": reward_mode,
                "n_model_seeds": len(rows),
                "episodes_per_seed": rows[0]["eval_episodes"] if rows else 0,
            }
            for metric in metrics:
                values = [float(r[metric]) for r in rows]
                item[f"{metric}_mean"] = mean(values)
                item[f"{metric}_std_sample"] = std_sample(values)
            across.append(item)

    return across


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        default="/home/gyh/AUV_RL/checkpoints_strict_v1",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/gyh/AUV_RL/figures/strict_v3_stress_gap_eval20_v1",
    )
    parser.add_argument(
        "--model-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed-start", type=int, default=2000)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_episode_rows = []
    per_seed_rows = []

    for condition in CONDITIONS:
        print("\n" + "=" * 100)
        print(f"Condition: {condition['condition']}")
        print("=" * 100)

        for method, exp_name, reward_mode in METHODS:
            for model_seed in args.model_seeds:
                rows = evaluate_one_model_condition(
                    method=method,
                    exp_name=exp_name,
                    reward_mode=reward_mode,
                    model_seed=model_seed,
                    condition=condition,
                    eval_episodes=args.eval_episodes,
                    eval_seed_start=args.eval_seed_start,
                    checkpoint_dir=checkpoint_dir,
                )
                all_episode_rows.extend(rows)
                seed_summary = summarize_seed(rows)
                per_seed_rows.append(seed_summary)

                print(
                    f"{method:24s} seed{model_seed}: "
                    f"success={seed_summary['success_rate']:.3f}, "
                    f"timeout={seed_summary['timeout_rate']:.3f}, "
                    f"oob={seed_summary['out_of_bound_rate']:.3f}, "
                    f"init_dist={seed_summary['avg_init_dist']:.2f}, "
                    f"init_yaw={seed_summary['avg_init_abs_yaw_deg']:.1f}, "
                    f"xyz={seed_summary['avg_final_xyz_all']:.4f}, "
                    f"yaw={seed_summary['avg_final_yaw_all']:.3f}, "
                    f"steps={seed_summary['avg_steps_all']:.2f}"
                )

    across_rows = summarize_across_seeds(per_seed_rows)

    write_csv(output_dir / "stress_gap_per_episode.csv", all_episode_rows)
    write_csv(output_dir / "stress_gap_per_seed_summary.csv", per_seed_rows)
    write_csv(output_dir / "stress_gap_across_seed_summary.csv", across_rows)

    config = vars(args).copy()
    config["methods"] = METHODS
    config["conditions"] = CONDITIONS
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 100)
    print("STRESS GAP ACROSS-SEED SUMMARY")
    print("=" * 100)
    for row in across_rows:
        print(
            f"{row['condition']:18s} | "
            f"{row['method']:22s} | "
            f"success={row['success_rate_mean']:.3f} "
            f"± {row['success_rate_std_sample']:.3f} | "
            f"timeout={row['timeout_rate_mean']:.3f} | "
            f"oob={row['out_of_bound_rate_mean']:.3f} | "
            f"init_dist={row['avg_init_dist_mean']:.2f} | "
            f"init_yaw={row['avg_init_abs_yaw_deg_mean']:.1f} | "
            f"xyz={row['avg_final_xyz_all_mean']:.4f} | "
            f"yaw={row['avg_final_yaw_all_mean']:.3f} | "
            f"steps={row['avg_steps_all_mean']:.2f}"
        )

    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
