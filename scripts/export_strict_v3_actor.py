#!/usr/bin/env python3
"""
Export the strict v3 SAC actor checkpoint for MATLAB use.

Run this script in the Ubuntu AUV project, preferably from the project root:

    cd /path/to/AUV
    python3 export_strict_v3_actor.py \
      --ckpt checkpoints_strict_v1/AUVRecoveryStrict-v1__strict_two_stage_v3_v1__seed0_actor_final.pt \
      --outdir exported_actor_seed0 \
      --onnx

Outputs:
    actor_weights.npz       Always written. Contains all linear weights/biases.
    actor_weights.mat       Written when scipy is installed.
    actor_deterministic.onnx Written when --onnx is passed and onnx export works.
    actor_metadata.json     Observation/action conventions.
    actor_test_vector.json  A small input/output pair for MATLAB verification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Path to *_actor_final.pt checkpoint.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("exported_actor"),
        help="Output directory.",
    )
    parser.add_argument(
        "--onnx",
        action="store_true",
        help="Also export deterministic actor to ONNX.",
    )
    return parser.parse_args()


def torch_load_state_dict(ckpt_path: Path):
    import torch

    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(ckpt_path, map_location="cpu")


def build_actor_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(12, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc_mean = nn.Linear(256, 4)
            self.fc_logstd = nn.Linear(256, 4)
            self.register_buffer("action_scale", torch.ones(4, dtype=torch.float32))
            self.register_buffer("action_bias", torch.zeros(4, dtype=torch.float32))

        def forward(self, obs):
            x = F.relu(self.fc1(obs))
            x = F.relu(self.fc2(x))
            mean = self.fc_mean(x)
            return torch.tanh(mean) * self.action_scale + self.action_bias

    return Actor


def state_to_numpy(state_dict) -> dict[str, np.ndarray]:
    wanted_keys = [
        "fc1.weight",
        "fc1.bias",
        "fc2.weight",
        "fc2.bias",
        "fc_mean.weight",
        "fc_mean.bias",
        "fc_logstd.weight",
        "fc_logstd.bias",
        "action_scale",
        "action_bias",
    ]
    arrays = {}
    for key in wanted_keys:
        if key in state_dict:
            arrays[key.replace(".", "_")] = state_dict[key].detach().cpu().numpy()
    return arrays


def write_metadata(outdir: Path, ckpt_path: Path) -> None:
    metadata = {
        "source_checkpoint": str(ckpt_path),
        "env_id": "AUVRecoveryStrict-v1",
        "reward_mode": "two_stage_v3",
        "stage_switch_dist_m": 0.5,
        "dt_s": 0.05,
        "success_position_tolerance_m": 0.10,
        "success_yaw_tolerance_deg": 5.0,
        "success_hold_steps_required": 5,
        "observation_order": [
            "x",
            "y",
            "z",
            "yaw",
            "u",
            "v",
            "w",
            "r",
            "target_x_minus_x",
            "target_y_minus_y",
            "target_z_minus_z",
            "wrapToPi(target_yaw_minus_yaw)",
        ],
        "normalized_action_order": [
            "a_fx",
            "a_fy",
            "a_fz",
            "a_mz",
        ],
        "physical_action_order": [
            "Fx",
            "Fy",
            "Fz",
            "Mz",
        ],
        "physical_action_scale": [50.0, 50.0, 50.0, 20.0],
        "deterministic_actor_formula": (
            "action_norm = tanh(fc_mean(relu(fc2(relu(fc1(obs))))))"
        ),
        "matlab_row_vector_formula": [
            "h1 = max(0, obs * fc1_weight.' + fc1_bias);",
            "h2 = max(0, h1 * fc2_weight.' + fc2_bias);",
            "action_norm = tanh(h2 * fc_mean_weight.' + fc_mean_bias);",
            "tau = action_norm .* [50 50 50 20];",
        ],
    }
    (outdir / "actor_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def export_npz_and_mat(outdir: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez(outdir / "actor_weights.npz", **arrays)

    try:
        from scipy.io import savemat
    except Exception as exc:
        print(f"[WARN] scipy not available, skipped .mat export: {exc}")
        print("       If needed: python3 -m pip install scipy")
        return

    savemat(outdir / "actor_weights.mat", arrays)


def export_onnx(outdir: Path, actor) -> None:
    import torch

    dummy_obs = torch.zeros(1, 12, dtype=torch.float32)
    onnx_path = outdir / "actor_deterministic.onnx"
    torch.onnx.export(
        actor,
        dummy_obs,
        onnx_path,
        input_names=["obs"],
        output_names=["action_norm"],
        opset_version=11,
        dynamic_axes={"obs": {0: "batch"}, "action_norm": {0: "batch"}},
    )
    print(f"[OK] wrote {onnx_path}")


def write_test_vector(outdir: Path, actor) -> None:
    import torch

    # A nonzero vector makes row/order mistakes easier to catch in MATLAB.
    obs = np.array(
        [[0.8, -0.2, 0.1, 0.15, 0.05, -0.03, 0.02, 0.01, -0.8, 0.2, -0.1, -0.15]],
        dtype=np.float32,
    )
    with torch.no_grad():
        action = actor(torch.tensor(obs, dtype=torch.float32)).cpu().numpy()
    payload = {
        "obs": obs.reshape(-1).tolist(),
        "action_norm": action.reshape(-1).tolist(),
        "tau_physical": (action.reshape(-1) * np.array([50.0, 50.0, 50.0, 20.0])).tolist(),
    }
    (outdir / "actor_test_vector.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    state_dict = torch_load_state_dict(args.ckpt)
    Actor = build_actor_class()
    actor = Actor()
    actor.load_state_dict(state_dict)
    actor.eval()

    arrays = state_to_numpy(state_dict)
    export_npz_and_mat(args.outdir, arrays)
    write_metadata(args.outdir, args.ckpt)
    write_test_vector(args.outdir, actor)

    if args.onnx:
        try:
            export_onnx(args.outdir, actor)
        except Exception as exc:
            print(f"[WARN] ONNX export failed: {exc}")
            print("       If needed: python3 -m pip install onnx")

    print(f"[OK] exported actor files to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
