import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register


try:
    register(
        id="AUVRecoveryStrict-v1",
        entry_point="auv_rl.envs.auv_env_strict_v1:AUVRecoveryStrictEnv",
    )
except Exception:
    pass


def wrap_pi(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


class AUVRecoveryStrictEnv(gym.Env):
    """
    Unified strict UUV recovery environment.

    Observation (12D):
      [x, y, z, yaw, u, v, w, r, dx, dy, dz, dyaw]

    Action (4D):
      normalized [-1, 1]^4 when action_normalized=True,
      then scaled to [Fx, Fy, Fz, Mz].

    IMPORTANT:
      The terminal success rule is identical for every reward_mode:
        dist_xyz < 0.10 m
        |yaw error| < 5 deg
        held for hold_steps_required consecutive steps

      Therefore reward_mode changes only the dense reward shaping, not the task
      definition or episode termination criterion.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        render_mode=None,
        *,
        reward_mode: str = "baseline",
        action_normalized: bool = True,
        max_steps: int = 1000,
        dt: float = 0.05,
        stage_switch_dist: float = 0.5,
        hold_steps_required: int = 5,
        current_speed: float = 0.0,
        current_random_direction: bool = False,
        current_direction_rad: float = 0.0,
        current_vertical_speed: float = 0.0,
        terminal_success_bonus: float = 100.0,
    ):
        super().__init__()
        self.render_mode = render_mode

        self.dt = float(dt)
        self.max_steps = int(max_steps)

        self.m = 50.0
        self.Iz = 10.0
        self.d_lin = np.array([20.0, 25.0, 30.0, 8.0], dtype=np.float32)

        self.precision_xyz_tol = 0.10
        self.precision_yaw_tol = 5.0 * np.pi / 180.0
        self.stage_switch_dist = float(stage_switch_dist)
        self.hold_steps_required = int(hold_steps_required)
        self.terminal_success_bonus = float(terminal_success_bonus)

        if self.stage_switch_dist <= 0.0:
            raise ValueError("stage_switch_dist must be positive")
        if self.hold_steps_required < 1:
            raise ValueError("hold_steps_required must be >= 1")

        self.current_speed = float(current_speed)
        self.current_random_direction = bool(current_random_direction)
        self.current_direction_rad = float(current_direction_rad)
        self.current_vertical_speed = float(current_vertical_speed)
        self.current_vx = 0.0
        self.current_vy = 0.0
        self.current_vz = 0.0

        valid_modes = [
            "baseline",
            "precision_v2",
            "two_stage_v1",
            "two_stage_v2",
            "two_stage_v3",
            "hard_switch_v1",
        ]
        if reward_mode not in valid_modes:
            raise ValueError(f"reward_mode must be one of {valid_modes}")
        self.reward_mode = reward_mode

        self.act_high = np.array([50.0, 50.0, 50.0, 20.0], dtype=np.float32)
        self.action_normalized = bool(action_normalized)
        if self.action_normalized:
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(4,),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(
                low=-self.act_high,
                high=self.act_high,
                dtype=np.float32,
            )

        obs_high = np.array(
            [
                5.0, 5.0, 3.0, np.pi,
                3.0, 3.0, 3.0, 2.0,
                6.0, 6.0, 4.0, np.pi,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )

        self.state = None
        self.target = None
        self.step_count = 0
        self.success_hold_count = 0
        self.prev_dist_xy = None
        self.prev_dist_xyz = None
        self.prev_abs_dyaw = None
        self.prev_action = np.zeros(4, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.success_hold_count = 0

        x = self.np_random.uniform(-1.0, 1.0)
        y = self.np_random.uniform(-1.0, 1.0)
        z = self.np_random.uniform(-0.5, 0.5)
        yaw = self.np_random.uniform(-np.pi / 4.0, np.pi / 4.0)

        u = self.np_random.uniform(-0.2, 0.2)
        v = self.np_random.uniform(-0.2, 0.2)
        w = self.np_random.uniform(-0.2, 0.2)
        r = self.np_random.uniform(-0.1, 0.1)

        self.state = np.array([x, y, z, yaw, u, v, w, r], dtype=np.float32)

        tx = self.np_random.uniform(-1.0, 1.0)
        ty = self.np_random.uniform(-1.0, 1.0)
        tz = self.np_random.uniform(-0.5, 0.5)
        tyaw = 0.0
        self.target = np.array([tx, ty, tz, tyaw], dtype=np.float32)

        if self.current_random_direction:
            current_angle = self.np_random.uniform(-np.pi, np.pi)
        else:
            current_angle = self.current_direction_rad

        self.current_vx = self.current_speed * np.cos(current_angle)
        self.current_vy = self.current_speed * np.sin(current_angle)
        self.current_vz = self.current_vertical_speed

        dx, dy, dz, dyaw = self._goal_errors()
        self.prev_dist_xy = float(np.sqrt(dx * dx + dy * dy))
        self.prev_dist_xyz = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        self.prev_abs_dyaw = abs(float(dyaw))
        self.prev_action = np.zeros(4, dtype=np.float32)

        return self._get_obs(), {
            "target": self.target.copy(),
            "reward_mode": self.reward_mode,
            "strict_success": True,
            "position_tolerance": float(self.precision_xyz_tol),
            "yaw_tolerance_deg": float(np.degrees(self.precision_yaw_tol)),
            "hold_steps_required": int(self.hold_steps_required),
        }

    def step(self, action):
        self.step_count += 1

        action = np.asarray(action, dtype=np.float32)
        if self.action_normalized:
            physical_action = np.clip(action, -1.0, 1.0) * self.act_high
        else:
            physical_action = np.clip(action, -self.act_high, self.act_high)

        x, y, z, yaw, u, v, w, r = self.state
        fx, fy, fz, mz = physical_action

        acc = np.array(
            [fx / self.m, fy / self.m, fz / self.m, mz / self.Iz],
            dtype=np.float32,
        )
        vel = np.array([u, v, w, r], dtype=np.float32)
        inertia = np.array([self.m, self.m, self.m, self.Iz], dtype=np.float32)
        vel_dot = acc - self.d_lin * vel / inertia
        u2, v2, w2, r2 = vel + vel_dot * self.dt

        cy, sy = np.cos(yaw), np.sin(yaw)
        x_dot = cy * u2 - sy * v2
        y_dot = sy * u2 + cy * v2
        z_dot = w2
        yaw_dot = r2

        x2 = x + (x_dot + self.current_vx) * self.dt
        y2 = y + (y_dot + self.current_vy) * self.dt
        z2 = z + (z_dot + self.current_vz) * self.dt
        yaw2 = wrap_pi(yaw + yaw_dot * self.dt)

        self.state = np.array([x2, y2, z2, yaw2, u2, v2, w2, r2], dtype=np.float32)

        dx, dy, dz, dyaw = self._goal_errors()
        dist_xy = float(np.sqrt(dx * dx + dy * dy))
        dist_xyz = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        abs_dyaw = abs(float(dyaw))

        delta_dist_xy = float(self.prev_dist_xy - dist_xy)
        delta_dist_xyz = float(self.prev_dist_xyz - dist_xyz)
        delta_yaw = float(self.prev_abs_dyaw - abs_dyaw)
        self.prev_dist_xy = dist_xy
        self.prev_dist_xyz = dist_xyz
        self.prev_abs_dyaw = abs_dyaw

        in_precision_stage = dist_xyz < self.stage_switch_dist

        terminal_now = (
            dist_xyz < self.precision_xyz_tol
            and abs_dyaw < self.precision_yaw_tol
        )
        if terminal_now:
            self.success_hold_count += 1
        else:
            self.success_hold_count = 0
        success = self.success_hold_count >= self.hold_steps_required

        out_of_bound = (
            abs(float(x2)) > 5.0
            or abs(float(y2)) > 5.0
            or abs(float(z2)) > 3.0
        )
        terminated = bool(success or out_of_bound)
        truncated = bool(self.step_count >= self.max_steps and not terminated)

        act_pen = float(np.sum(physical_action * physical_action))
        action_delta = physical_action - self.prev_action
        action_smooth_pen = float(np.sum(action_delta * action_delta))
        self.prev_action = physical_action.copy()

        if self.reward_mode == "baseline":
            reward = (
                -1.0 * dist_xyz
                + 2.0 * delta_dist_xyz
                - 0.2 * abs_dyaw
                - 1e-4 * act_pen
                - 0.01
            )

        elif self.reward_mode == "precision_v2":
            reward = (
                -0.8 * dist_xyz
                -0.4 * dist_xy
                + 4.5 * delta_dist_xyz
                + 1.5 * delta_dist_xy
                + 0.5 * delta_yaw
                - 0.12 * abs_dyaw
                - 1e-4 * act_pen
                - 0.01
            )

        elif self.reward_mode == "two_stage_v1":
            if not in_precision_stage:
                reward = (
                    -0.6 * dist_xyz
                    -0.2 * dist_xy
                    + 4.5 * delta_dist_xyz
                    + 1.0 * delta_dist_xy
                    - 0.06 * abs_dyaw
                    - 8e-5 * act_pen
                    - 0.01
                )
            else:
                reward = (
                    -0.4 * dist_xyz
                    -1.2 * dist_xy
                    + 2.0 * delta_dist_xyz
                    + 5.0 * delta_dist_xy
                    + 1.2 * delta_yaw
                    - 0.18 * abs_dyaw
                    - 1.2e-4 * act_pen
                    - 0.015
                )
                if dist_xy < 0.10:
                    reward += 0.8
                if dist_xyz < 0.10:
                    reward += 1.0
                if abs_dyaw < self.precision_yaw_tol:
                    reward += 0.5

        elif self.reward_mode == "two_stage_v2":
            if not in_precision_stage:
                reward = (
                    -0.55 * dist_xyz
                    -0.15 * dist_xy
                    + 4.8 * delta_dist_xyz
                    + 1.2 * delta_dist_xy
                    - 0.05 * abs_dyaw
                    - 8e-5 * act_pen
                    - 0.01
                )
            else:
                speed_norm = float(np.sqrt(u2 * u2 + v2 * v2 + w2 * w2))
                yaw_rate_pen = abs(float(r2))
                abs_dz = abs(float(dz))
                reward = (
                    -0.65 * dist_xyz
                    -1.00 * dist_xy
                    -0.60 * abs_dz
                    +2.2 * delta_dist_xyz
                    +4.2 * delta_dist_xy
                    +2.5 * delta_yaw
                    -0.45 * abs_dyaw
                    -0.08 * speed_norm
                    -0.05 * yaw_rate_pen
                    -1.2e-4 * act_pen
                    -3.0e-5 * action_smooth_pen
                    -0.015
                )
                if dist_xy < 0.10:
                    reward += 0.8
                if abs_dz < 0.05:
                    reward += 0.8
                if dist_xyz < 0.10:
                    reward += 1.2
                if abs_dyaw < 10.0 * np.pi / 180.0:
                    reward += 0.6
                if abs_dyaw < self.precision_yaw_tol:
                    reward += 1.5
                if speed_norm < 0.05:
                    reward += 0.3

        elif self.reward_mode in ("two_stage_v3", "hard_switch_v1"):
            # Controlled ablation of the stage-transition mechanism.
            #
            # Both modes use exactly the same approach reward, precision reward,
            # terminal-region damping, terminal bonus, and failure penalty.
            # The ONLY difference is the stage weight:
            #   two_stage_v3  -> smooth sigmoid weight
            #   hard_switch_v1 -> binary 0/1 switch at stage_switch_dist
            approach_reward = (
                -1.0 * dist_xyz
                + 2.0 * delta_dist_xyz
                - 0.2 * abs_dyaw
                - 1e-4 * act_pen
                - 0.01
            )

            precision_reward = (
                -0.8 * dist_xyz
                - 0.4 * dist_xy
                + 4.5 * delta_dist_xyz
                + 1.5 * delta_dist_xy
                + 0.5 * delta_yaw
                - 0.12 * abs_dyaw
                - 1e-4 * act_pen
                - 0.01
            )

            if self.reward_mode == "two_stage_v3":
                transition_width = 0.08
                gate_arg = np.clip(
                    (self.stage_switch_dist - dist_xyz) / transition_width,
                    -20.0,
                    20.0,
                )
                stage_weight = float(1.0 / (1.0 + np.exp(-gate_arg)))
            else:
                stage_weight = float(dist_xyz <= self.stage_switch_dist)

            reward = (
                (1.0 - stage_weight) * approach_reward
                + stage_weight * precision_reward
            )

            # Identical terminal-region damping for both ablation variants.
            terminal_gate_arg = np.clip(
                (0.18 - dist_xyz) / 0.04,
                -20.0,
                20.0,
            )
            terminal_weight = float(
                1.0 / (1.0 + np.exp(-terminal_gate_arg))
            )
            speed_norm = float(np.sqrt(u2 * u2 + v2 * v2 + w2 * w2))
            yaw_rate_pen = abs(float(r2))

            reward -= terminal_weight * (
                0.02 * speed_norm
                + 0.01 * yaw_rate_pen
                + 5e-6 * action_smooth_pen
            )

        else:
            raise RuntimeError(f"Unhandled reward_mode: {self.reward_mode}")

        # Common terminal terms: identical for every reward mode.
        if success:
            reward += self.terminal_success_bonus
        if out_of_bound:
            reward -= 30.0

        if self.reward_mode not in ("two_stage_v3", "hard_switch_v1"):
            stage_weight = float(in_precision_stage)

        info = {
            "dist_xy": dist_xy,
            "dist_xyz": dist_xyz,
            "dyaw": float(dyaw),
            "yaw_error_deg": float(np.degrees(abs_dyaw)),
            "delta_dist_xy": delta_dist_xy,
            "delta_dist_xyz": delta_dist_xyz,
            "delta_yaw": delta_yaw,
            "success": bool(success),
            "terminal_now": bool(terminal_now),
            "out_of_bound": bool(out_of_bound),
            "reward_mode": self.reward_mode,
            "in_precision_stage": bool(in_precision_stage),
            "stage_weight": float(stage_weight),
            "success_hold_count": int(self.success_hold_count),
            "stage_switch_dist": float(self.stage_switch_dist),
            "hold_steps_required": int(self.hold_steps_required),
            "strict_success": True,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def _goal_errors(self):
        x, y, z, yaw, *_ = self.state
        tx, ty, tz, tyaw = self.target
        dx = tx - x
        dy = ty - y
        dz = tz - z
        dyaw = wrap_pi(tyaw - yaw)
        return float(dx), float(dy), float(dz), float(dyaw)

    def _get_obs(self):
        x, y, z, yaw, u, v, w, r = self.state
        dx, dy, dz, dyaw = self._goal_errors()
        return np.array(
            [x, y, z, yaw, u, v, w, r, dx, dy, dz, dyaw],
            dtype=np.float32,
        )
