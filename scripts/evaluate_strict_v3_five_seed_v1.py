#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import auv_rl.envs.auv_env_strict_v1  # noqa: F401

LOG_STD_MAX = 2
LOG_STD_MIN = -5
METHODS = [
    ('Strict baseline', 'strict_baseline_v1', 'baseline'),
    ('Strict precision v2', 'strict_precision_v2_v1', 'precision_v2'),
    ('Smooth two-stage v3', 'strict_two_stage_v3_v1', 'two_stage_v3'),
]

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, act_dim)
        self.fc_logstd = nn.Linear(256, act_dim)
        self.register_buffer('action_scale', torch.tensor((env.action_space.high-env.action_space.low)/2.0, dtype=torch.float32))
        self.register_buffer('action_bias', torch.tensor((env.action_space.high+env.action_space.low)/2.0, dtype=torch.float32))
    def forward(self, x):
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        return mean, log_std
    def deterministic_action(self, obs):
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean, _ = self(x)
            action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action.cpu().numpy()[0]

def load_actor(env, ckpt: Path):
    if not ckpt.is_file():
        raise FileNotFoundError(f'Missing checkpoint: {ckpt}')
    actor = Actor(env)
    state = torch.load(ckpt, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    actor.load_state_dict(state, strict=True)
    actor.eval()
    return actor

def mean(v): return float(np.mean(v)) if v else float('nan')
def std(v): return float(np.std(v, ddof=1)) if len(v) >= 2 else float('nan')

def write_csv(path: Path, rows):
    if not rows: raise ValueError(f'No rows to write: {path}')
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

def eval_model(method, exp_name, reward_mode, seed, episodes, eval_seed_start, ckpt_dir):
    env = gym.make('AUVRecoveryStrict-v1', reward_mode=reward_mode, action_normalized=True, max_steps=1000, stage_switch_dist=0.5, hold_steps_required=5, current_speed=0.0)
    ckpt = ckpt_dir / f'AUVRecoveryStrict-v1__{exp_name}__seed{seed}_actor_final.pt'
    actor = load_actor(env, ckpt)
    rows = []
    for ep in range(episodes):
        eval_seed = eval_seed_start + ep
        obs, _ = env.reset(seed=eval_seed)
        terminated = truncated = False
        steps = 0; info = {}
        while not (terminated or truncated):
            action = actor.deterministic_action(obs)
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
        rows.append({
            'method': method, 'exp_name': exp_name, 'reward_mode': reward_mode, 'model_seed': seed, 'episode': ep, 'eval_seed': eval_seed,
            'success': int(bool(info.get('success', False))),
            'timeout': int(bool(truncated) and not bool(info.get('success', False))),
            'out_of_bound': int(bool(info.get('out_of_bound', False))),
            'steps': steps,
            'final_dist_xy': float(info.get('dist_xy', np.nan)),
            'final_dist_xyz': float(info.get('dist_xyz', np.nan)),
            'final_yaw_error_deg': float(info.get('yaw_error_deg', np.nan)),
            'final_hold_count': int(info.get('success_hold_count', 0)),
            'final_stage_weight': float(info.get('stage_weight', np.nan)),
        })
    env.close(); return rows

def summarize_seed(rows):
    succ = [r for r in rows if r['success'] == 1]
    return {
        'method': rows[0]['method'], 'exp_name': rows[0]['exp_name'], 'reward_mode': rows[0]['reward_mode'], 'model_seed': rows[0]['model_seed'], 'eval_episodes': len(rows),
        'success_rate': mean([r['success'] for r in rows]),
        'timeout_rate': mean([r['timeout'] for r in rows]),
        'out_of_bound_rate': mean([r['out_of_bound'] for r in rows]),
        'avg_steps_all': mean([r['steps'] for r in rows]),
        'avg_steps_success': mean([r['steps'] for r in succ]),
        'avg_final_xy_all': mean([r['final_dist_xy'] for r in rows]),
        'avg_final_xy_success': mean([r['final_dist_xy'] for r in succ]),
        'avg_final_xyz_all': mean([r['final_dist_xyz'] for r in rows]),
        'avg_final_xyz_success': mean([r['final_dist_xyz'] for r in succ]),
        'avg_final_yaw_all': mean([r['final_yaw_error_deg'] for r in rows]),
        'avg_final_yaw_success': mean([r['final_yaw_error_deg'] for r in succ]),
    }

def summarize_across(per_seed):
    metrics = ['success_rate','timeout_rate','out_of_bound_rate','avg_steps_all','avg_steps_success','avg_final_xy_all','avg_final_xy_success','avg_final_xyz_all','avg_final_xyz_success','avg_final_yaw_all','avg_final_yaw_success']
    out = []
    for method, exp_name, reward_mode in METHODS:
        rows = [r for r in per_seed if r['method'] == method]
        item = {'method': method, 'exp_name': exp_name, 'reward_mode': reward_mode, 'n_model_seeds': len(rows), 'episodes_per_seed': rows[0]['eval_episodes'] if rows else 0}
        for m in metrics:
            vals = [float(r[m]) for r in rows]
            item[m + '_mean'] = mean(vals); item[m + '_std_sample'] = std(vals)
        out.append(item)
    return out

def plot_summary(across, outdir: Path):
    labels = [r['method'] for r in across]; x = np.arange(len(labels))
    def arr(k): return [r[k] for r in across]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax=axes[0,0]; ax.bar(x, arr('success_rate_mean'), yerr=arr('success_rate_std_sample'), capsize=4); ax.set_title('(a) Strict task success'); ax.set_ylabel('Success rate'); ax.set_ylim(0,1.05); ax.grid(True,axis='y',alpha=0.35)
    ax=axes[0,1]; ax.bar(x, arr('avg_steps_all_mean'), yerr=arr('avg_steps_all_std_sample'), capsize=4); ax.set_title('(b) Average completion steps'); ax.set_ylabel('Steps'); ax.grid(True,axis='y',alpha=0.35)
    ax=axes[1,0]; ax.bar(x, arr('avg_final_xyz_all_mean'), yerr=arr('avg_final_xyz_all_std_sample'), capsize=4); ax.axhline(0.10, linestyle='--', linewidth=1.5, label='0.10 m'); ax.set_title('(c) Final XYZ error'); ax.set_ylabel('XYZ error / m'); ax.legend(); ax.grid(True,axis='y',alpha=0.35)
    ax=axes[1,1]; ax.bar(x, arr('avg_final_yaw_all_mean'), yerr=arr('avg_final_yaw_all_std_sample'), capsize=4); ax.axhline(5.0, linestyle='--', linewidth=1.5, label='5 deg'); ax.set_title('(d) Final yaw error'); ax.set_ylabel('Yaw error / deg'); ax.legend(); ax.grid(True,axis='y',alpha=0.35)
    for ax in axes.flat:
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha='right')
    fig.suptitle('Strict five-seed evaluation of SAC recovery methods')
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(outdir/'strict_five_seed_summary.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint-dir', default='/home/gyh/AUV_RL/checkpoints_strict_v1')
    p.add_argument('--output-dir', default='/home/gyh/AUV_RL/figures/strict_v3_five_seed_eval100_v1')
    p.add_argument('--model-seeds', type=int, nargs='+', default=[0,1,2,3,4])
    p.add_argument('--eval-episodes', type=int, default=100)
    p.add_argument('--eval-seed-start', type=int, default=1000)
    args = p.parse_args()
    ckpt_dir = Path(args.checkpoint_dir); outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    all_rows=[]; per_seed=[]
    for method, exp_name, reward_mode in METHODS:
        for seed in args.model_seeds:
            rows = eval_model(method, exp_name, reward_mode, seed, args.eval_episodes, args.eval_seed_start, ckpt_dir)
            all_rows.extend(rows); ss = summarize_seed(rows); per_seed.append(ss)
            print(f"{method:24s} seed{seed}: success={ss['success_rate']:.3f}, timeout={ss['timeout_rate']:.3f}, xyz={ss['avg_final_xyz_all']:.4f}, yaw={ss['avg_final_yaw_all']:.3f}, steps={ss['avg_steps_all']:.2f}")
    across = summarize_across(per_seed)
    write_csv(outdir/'strict_five_seed_per_episode.csv', all_rows)
    write_csv(outdir/'strict_five_seed_per_seed_summary.csv', per_seed)
    write_csv(outdir/'strict_five_seed_across_seed_summary.csv', across)
    with (outdir/'run_config.json').open('w', encoding='utf-8') as f: json.dump(vars(args), f, indent=2, ensure_ascii=False)
    plot_summary(across, outdir)
    print('\n' + '='*100); print('FIVE-SEED ACROSS-SEED SUMMARY'); print('='*100)
    for r in across:
        print(f"{r['method']}: success={r['success_rate_mean']:.3f} ± {r['success_rate_std_sample']:.3f}, timeout={r['timeout_rate_mean']:.3f}, xyz={r['avg_final_xyz_all_mean']:.4f} ± {r['avg_final_xyz_all_std_sample']:.4f}, yaw={r['avg_final_yaw_all_mean']:.3f} ± {r['avg_final_yaw_all_std_sample']:.3f}, steps={r['avg_steps_all_mean']:.2f} ± {r['avg_steps_all_std_sample']:.2f}")
    print(f'\nSaved outputs to: {outdir}')
if __name__ == '__main__': main()
