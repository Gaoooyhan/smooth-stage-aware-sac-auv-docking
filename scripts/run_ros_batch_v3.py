import os
import time
import pandas as pd
import numpy as np
import run_ros_batch as b

SAVE_ROOT = "/home/gyh/AUV_RL/figures/ros_batch_eval_v3"

# 使用新版严格终端标准
b.RUN_TIME = 15
b.SUCCESS_DIST_TOL = 0.10
b.POLICY_POS_TOL = 0.10
b.POLICY_YAW_TOL_DEG = 5.0
b.POLICY_HOLD_STEPS = 5

INIT_STATES = [
    [-0.8, -0.6, -0.10,  0.20, 0.0, 0.0, 0.0, 0.0],
    [-1.2,  0.8,  0.15, -0.35, 0.0, 0.0, 0.0, 0.0],
    [ 1.3, -1.0, -0.20,  0.65, 0.0, 0.0, 0.0, 0.0],
    [-1.8,  1.2,  0.20, -0.85, 0.0, 0.0, 0.0, 0.0],
    [ 2.2, -1.5, -0.25,  1.05, 0.0, 0.0, 0.0, 0.0],
    [-2.4,  1.8,  0.30, -1.20, 0.0, 0.0, 0.0, 0.0],
]

def strict_eval_from_log(logger_dir):
    csv_path = os.path.join(logger_dir, "ros_log.csv")
    df = pd.read_csv(csv_path)

    yaw_err_deg = df["dyaw"].abs() * 180.0 / np.pi
    inside = (df["dist_xyz"] < b.SUCCESS_DIST_TOL) & (yaw_err_deg < b.POLICY_YAW_TOL_DEG)

    hold = 0
    max_hold = 0
    success_step = None

    for i, ok in enumerate(inside):
        if bool(ok):
            hold += 1
        else:
            hold = 0
        max_hold = max(max_hold, hold)
        if hold >= b.POLICY_HOLD_STEPS and success_step is None:
            success_step = i

    success = success_step is not None

    if success:
        eval_step = success_step
    else:
        eval_step = len(df) - 1

    return {
        "success": int(success),
        "success_step": int(success_step) if success else -1,
        "max_hold": int(max_hold),
        "final_xyz_for_table": float(df["dist_xyz"].iloc[eval_step]),
        "final_yaw_for_table_deg": float(yaw_err_deg.iloc[eval_step]),
        "steps_for_table": int(eval_step + 1),
        "duration_s": float((eval_step + 1) * 0.05),
        "min_dist_xyz": float(df["dist_xyz"].min()),
        "end_dist_xyz": float(df["dist_xyz"].iloc[-1]),
        "end_yaw_deg": float(yaw_err_deg.iloc[-1]),
    }

def main():
    timestamp = time.strftime("ros_v3_strict_%Y%m%d_%H%M%S")
    save_dir = os.path.join(SAVE_ROOT, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    results = []

    for i, init_state in enumerate(INIT_STATES):
        raw = b.run_one_case(i, init_state)
        strict = strict_eval_from_log(raw["logger_dir"])

        row = {
            "case": i + 1,
            "init_x": init_state[0],
            "init_y": init_state[1],
            "init_z": init_state[2],
            "init_yaw_rad": init_state[3],
            "init_yaw_deg": init_state[3] * 180.0 / np.pi,
            "success": strict["success"],
            "final_xyz_m": strict["final_xyz_for_table"],
            "final_yaw_deg": strict["final_yaw_for_table_deg"],
            "steps": strict["steps_for_table"],
            "duration_s": strict["duration_s"],
            "success_step": strict["success_step"],
            "max_hold": strict["max_hold"],
            "min_dist_xyz": strict["min_dist_xyz"],
            "end_dist_xyz": strict["end_dist_xyz"],
            "end_yaw_deg": strict["end_yaw_deg"],
            "logger_dir": raw["logger_dir"],
        }

        results.append(row)

        print(
            f"[STRICT] case={row['case']}, success={row['success']}, "
            f"xyz={row['final_xyz_m']:.4f}, yaw={row['final_yaw_deg']:.4f}, "
            f"steps={row['steps']}, max_hold={row['max_hold']}"
        )

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_dir, "summary.csv"), index=False)

    with open(os.path.join(save_dir, "batch_summary.txt"), "w", encoding="utf-8") as f:
        f.write("ROS2 strict closed-loop verification for smooth two-stage SAC\n")
        f.write("=" * 70 + "\n")
        f.write(f"Num cases: {len(df)}\n")
        f.write(f"Success rate: {df['success'].mean():.3f}\n")
        f.write(f"Avg final XYZ: {df['final_xyz_m'].mean():.6f}\n")
        f.write(f"Avg final yaw deg: {df['final_yaw_deg'].mean():.6f}\n")
        f.write(f"Avg steps: {df['steps'].mean():.3f}\n")
        f.write(f"Avg duration s: {df['duration_s'].mean():.3f}\n")

    print("\nSaved strict ROS2 batch results to:")
    print(save_dir)
    print("\nSummary:")
    print(df[["case", "success", "final_xyz_m", "final_yaw_deg", "steps", "duration_s", "max_hold"]])

if __name__ == "__main__":
    main()
