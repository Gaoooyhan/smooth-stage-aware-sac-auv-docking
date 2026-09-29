# Smooth Stage-Aware SAC for AUV Docking

This repository provides the core implementation and processed experimental data associated with the paper:

**A Smooth Stage-Aware Soft Actor-Critic Approach for Docking Control of Autonomous Underwater Vehicles**

The study investigates autonomous underwater vehicle (AUV) docking using a smooth stage-aware Soft Actor-Critic (SAC) controller. The standard SAC architecture is retained, while task-specific reward design and terminal evaluation are introduced to coordinate efficient approach and accurate terminal docking.

## Overview

The proposed controller incorporates:

- a distance-dependent smooth weighting function that continuously blends approach and precision-docking rewards;
- a terminal-region damping term that discourages residual motion and control-command variation;
- a strict docking criterion based on three-dimensional position and yaw tolerances;
- a consecutive terminal-holding requirement that distinguishes transient terminal entry from sustained docking success.

The same SAC actor is used throughout the docking process; no policy switching is introduced.

## Docking Task

The simulation uses a simplified fully actuated 4-DOF AUV model retaining:

- surge;
- sway;
- heave;
- yaw.

The continuous control input is:

```text
[Fx, Fy, Fz, Mz]
```

The policy observation contains 12 components:

```text
[x, y, z, yaw, u, v, w, r, dx, dy, dz, dyaw]
```

The control period is:

```text
0.05 s
```

Strict docking success requires:

```text
3D position error < 0.10 m
absolute yaw error < 5 deg
five consecutive control steps satisfying both tolerances
```

The five-step holding requirement corresponds to 0.25 s.

## Smooth Stage-Aware Reward

The docking task contains two continuously coordinated control objectives:

1. **Approach regulation**, emphasizing distance reduction and approach efficiency.
2. **Precision docking**, emphasizing terminal position accuracy and yaw alignment.

The proposed method uses a distance-dependent stage weight to continuously change the relative emphasis of these two reward components.

The transition midpoint is:

```text
dsw = 0.5 m
```

Unlike the proposed smooth formulation, the hard-switch ablation changes directly between the two reward components at this distance.

This controlled comparison is used to examine whether continuous reward coordination reduces local control adjustment near the stage boundary.

## Compared Controllers

The paper evaluates the following controllers:

- Baseline SAC
- Precision-reward SAC
- Hard-switch stage-aware SAC
- Smooth stage-aware SAC (proposed)
- Standard PPO
- ARSPPO (adapted)
- TD3
- Cascaded PID

The compared learning-based controllers use matched task definitions, physical action limits, terminal criteria, and training budgets where applicable.

## Evaluation Metrics

The main evaluation metrics include:

- strict docking success rate;
- final 3D position error;
- final absolute yaw error;
- episode length;
- transition action variation;
- yaw-moment variation;
- terminal mean absolute yaw rate.

### Transition Action Variation

Transition behavior is evaluated around the first inward crossing of the 0.5-m stage boundary.

An up-to-11-step transition window is used:

```text
tcross - 5, ..., tcross + 5
```

The window is truncated only when it extends beyond an episode boundary.

The cumulative variation of the normalized four-dimensional action vector is used to quantify local control adjustment around the transition.

## Experimental Conditions

### Nominal and Enlarged Initial States

The trained policies are evaluated under nominal and enlarged initial-state ranges without retraining.

The enlarged condition increases the initial position and yaw ranges to examine initial-state generalization.

### Consecutive-Holding Validation

A delay-only experiment evaluates command delays of:

```text
0, 1, 2, 3, 5, and 7 control steps
```

corresponding to:

```text
0.00, 0.05, 0.10, 0.15, 0.25, and 0.35 s
```

Other disturbance and model parameters remain nominal in this experiment.

This evaluation examines whether transient entry into the terminal tolerance region is sufficient to represent successful docking.

### Station-Wake Ablation

The hard-switch and smooth stage-aware variants are evaluated using paired held-out episodes under:

- zero current;
- a spatially varying station-wake disturbance with an outer current speed of 0.10 m/s.

The station-wake condition introduces a local change in current speed and direction near the docking region.

Transition action variation is used as the primary metric for this comparison.

### Graded Stress Evaluation

The graded stress protocol contains six levels, L0--L5, that jointly vary:

- reference current speed;
- position measurement noise;
- yaw measurement noise;
- command delay;
- mass and inertia mismatch;
- damping mismatch;
- actuator effectiveness.

These levels are controlled simulation stress indicators rather than calibrated physical sea-state categories.

## Repository Structure

```text
smooth-stage-aware-sac-auv-docking/
|
|-- auv_rl/
|   `-- envs/
|       `-- auv_env_strict_v1.py
|
|-- scripts/
|   |-- train_sac_strict_v1.py
|   |-- evaluate_strict_v3_five_seed_v1.py
|   |-- evaluate_strict_v3_hard_generalization_v1.py
|   |-- evaluate_strict_v3_stress_gap_v1.py
|   |-- export_strict_v3_actor.py
|   `-- run_ros_batch_v3.py
|
|-- plotting/
|   |-- fig3_performance_tradeoff.py
|   |-- fig5_terminal_holding_validation.py
|   `-- fig6_paired_transition_behavior.py
|
|-- data/
|   |-- table6_terminal_holding/
|   |-- table7_transition_ablation/
|   |-- fig6_transition_behavior/
|   `-- table8_graded_stress/
|
|-- LICENSE
|-- .gitignore
`-- README.md
```

## Data Description

### `data/table6_terminal_holding/`

Processed data supporting the consecutive terminal-holding validation.

The directory contains episode-level results, seed-level summaries, across-seed summaries, and the corresponding evaluation configuration.

### `data/table7_transition_ablation/`

Processed data supporting the held-out comparison between hard-switch and smooth stage-aware SAC.

The directory contains:

- episode-level transition and terminal metrics;
- across-seed summary statistics;
- evaluation configuration.

### `data/fig6_transition_behavior/`

Data used to reproduce the transition behavior shown in Fig. 6.

The directory contains:

- episode-level results;
- step-level transition trajectories;
- evaluation configuration.

### `data/table8_graded_stress/`

Processed results for the L0--L5 graded disturbance and model-uncertainty evaluation.

The directory contains:

- episode-level results;
- per-seed summaries;
- across-seed summaries;
- the complete stress-test configuration.

## Training

The proposed SAC controller is trained for:

```text
300,000 environment steps
```

using five independent training seeds:

```text
0, 1, 2, 3, 4
```

The main training implementation is provided in:

```text
scripts/train_sac_strict_v1.py
```

## Evaluation

The evaluation scripts in `scripts/` provide the main procedures used for:

- multi-seed strict docking evaluation;
- enlarged initial-state evaluation;
- hard-switch comparison;
- disturbance and model-uncertainty evaluation;
- exported-policy execution.

Evaluation uses deterministic policy inference.

## Reproducing Figures

The principal plotting scripts are provided in `plotting/`.

### Performance Comparison

```bash
python plotting/fig3_performance_tradeoff.py
```

### Consecutive Terminal-Holding Validation

```bash
python plotting/fig5_terminal_holding_validation.py
```

### Smooth versus Hard-Switch Transition Behavior

```bash
python plotting/fig6_paired_transition_behavior.py
```

Some local paths in the original research workflow may need to be adjusted to match the local directory structure.

## Key Transition-Ablation Result

Under the held-out station-wake condition with an outer current speed of 0.10 m/s, the mean transition-window action variation was:

```text
Hard-switch: 2.1106
Smooth:      1.9124
```

corresponding to a reduction of:

```text
9.39%
```

Both variants achieved the same observed strict success rate:

```text
0.998
```

The statistical comparison uses the five training-seed means as paired statistical units.

## Reproducibility Notes

This repository focuses on the core implementation and processed data directly supporting the principal experiments reported in the paper.

Temporary debugging files, smoke tests, intermediate experiments, redundant checkpoints, and development logs are not included.

Random seeds are explicitly controlled in the evaluation workflow, and paired initializations are used when methods are directly compared.

Small numerical differences may occur across hardware, operating systems, library versions, and floating-point implementations.

## Citation

If you use this repository, please cite:

```bibtex
@article{gao2026smooth,
  title   = {A Smooth Stage-Aware Soft Actor-Critic Approach for Docking Control of Autonomous Underwater Vehicles},
  author  = {Gao, Yuhan and Gao, Jian and Chen, Guofang and Li, Yufeng and Guo, Liepan},
  year    = {2026}
}
```

The bibliographic information will be updated after publication.

## License

This repository is released under the MIT License.

## Data Availability

The processed data supporting the principal quantitative results and ablation experiments reported in the paper are included in the `data/` directory.

## Contact

For questions regarding the implementation or experimental data, please contact the corresponding author listed in the paper.
