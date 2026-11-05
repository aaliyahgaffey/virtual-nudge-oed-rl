"""Train/evaluate a SAC policy to choose three virtual asteroid nudges.

From repository root:
  python -m scripts.train_virtual_nudge_rl --mode check
  python -m scripts.train_virtual_nudge_rl --mode baseline --episodes 10
  python -m scripts.train_virtual_nudge_rl --mode train --timesteps 20000
  python -m scripts.train_virtual_nudge_rl --mode evaluate --episodes 5
  python -m scripts.train_virtual_nudge_rl --mode evaluate --backend mujoco --episodes 1

Training defaults to the cheap SURROGATE. A MuJoCo rollout has a different
physics model and should be used as an independent validation step. Results
are not evidence of superior performance until compared with random search
and the project's existing Sobol / E-optimal nudge planner using fixed seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from src.virtual_nudge_rl_env import (VirtualNudgePlacementEnv, RLEnvConfig,
                                     plan_to_normalized_action)
from src.three_nudge_virtual_10param import Virtual10Settings, select_virtual_plan


def parse():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['check','baseline','train','evaluate'], default='check')
    ap.add_argument('--backend', choices=['surrogate','mujoco'], default='surrogate')
    ap.add_argument('--project-root', type=Path, default=Path.cwd())
    ap.add_argument('--model', type=Path, default=Path('results/rl_virtual_nudges/sac_virtual_nudge'))
    ap.add_argument('--timesteps', type=int, default=20_000)
    ap.add_argument('--episodes', type=int, default=5)
    ap.add_argument('--baseline', choices=['random','oed'], default='random')
    ap.add_argument('--observation-s', type=float, default=30.)
    ap.add_argument('--surrogate-sample-s', type=float, default=.2)
    ap.add_argument('--noise', type=float, choices=[0., 1e-4], default=0.)
    ap.add_argument('--seed', type=int, default=19)
    ap.add_argument('--no-domain-randomization', action='store_true')
    return ap.parse_args()


def make_env(args):
    cfg=Virtual10Settings(noise_std=args.noise, rng_seed=args.seed)
    rc=RLEnvConfig(backend=args.backend,
                   observation_s=args.observation_s,
                   surrogate_sample_s=args.surrogate_sample_s,
                   randomize_surrogate_truth=not args.no_domain_randomization,
                   debug_truth=(args.mode in ('baseline','evaluate')))
    return VirtualNudgePlacementEnv(args.project_root, settings=cfg, rl_config=rc)


def evaluate(env, episodes, seed, policy=None, baseline="random"):
    rewards=[]
    summaries=[]
    for k in range(episodes):
        obs,_=env.reset(seed=seed+k)
        env.action_space.seed(seed+k)  # reproducible random-policy baseline
        terminated=truncated=False
        total=0.
        actions=[]
        metrics=[]
        while not (terminated or truncated):
            if policy is not None:
                action=policy.predict(obs, deterministic=True)[0]
            elif baseline=='oed':
                plan,_=select_virtual_plan(
                    env.surface,env.identifier.last_state,env.identifier,
                    None,env.cfg,env.current_nudge+1,
                    np.random.default_rng(seed+k*101+env.current_nudge))
                action=plan_to_normalized_action(plan,env.surface,env.cfg)
            else:
                action=env.action_space.sample()
            obs,r,terminated,truncated,info=env.step(action)
            total+=float(r)
            actions.append(np.asarray(action,float).tolist())
            metrics.append({
                'reward':float(r),
                'nudge':info['nudge_number'],
                'rank':info.get('numerical_rank'),
                'impulse_Ns':info.get('impulse_Ns'),
                'point_body_m':(
                    np.asarray(info['point_body_m']).tolist()
                    if 'point_body_m' in info else None),
                'estimated_pi':(
                    np.asarray(info['estimated_pi']).tolist()
                    if 'estimated_pi' in info else None),
                'truth_diagnostics':info.get('DEV_ONLY_errors'),
                'failure_reason':info.get('failure_reason'),
            })
        rewards.append(total)
        print(f'Episode {k+1}: return={total:.4f}, nudges={len(metrics)}, '
              f'rank={metrics[-1]["rank"]}, '
              f'failure={metrics[-1]["failure_reason"]}')
        summaries.append(dict(seed=seed+k, total_reward=total,
                              actions=actions, nudges=metrics))
    print(f'Mean reward: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}')
    return summaries


def main():
    args=parse()
    if args.mode=='train' and args.backend!='surrogate':
        raise ValueError('Training is intended for the fast surrogate; '
                         'use --mode evaluate --backend mujoco to validate.')
    env=make_env(args)
    try:
        if args.mode=='check':
            from stable_baselines3.common.env_checker import check_env
            check_env(env, warn=True, skip_render_check=True)
            print('Gymnasium/SB3 environment check passed.')
            return
        if args.mode=='baseline':
            report=evaluate(env,args.episodes,args.seed,policy=None,baseline=args.baseline)
        else:
            from stable_baselines3 import SAC
            model_path=args.model if args.model.is_absolute() else args.project_root/args.model
            model_path.parent.mkdir(parents=True,exist_ok=True)
            if args.mode=='train':
                from stable_baselines3.common.monitor import Monitor
                monitored=Monitor(env)
                model=SAC('MlpPolicy',monitored, verbose=1, seed=args.seed,
                          learning_starts=1000, buffer_size=30_000,
                          batch_size=128, train_freq=1, gradient_steps=1,
                          tensorboard_log=None)
                model.learn(total_timesteps=args.timesteps, progress_bar=False)
                model.save(str(model_path))
                print(f'Saved SAC model to {model_path}.zip')
                return
            model=SAC.load(str(model_path), device='cpu')
            report=evaluate(env,args.episodes,args.seed,policy=model)
        report_path=args.project_root/'results'/'rl_virtual_nudges'/(
            f'{args.mode}_{args.baseline if args.mode == "baseline" else "rl"}_{args.backend}_rollouts.json')
        report_path.parent.mkdir(parents=True,exist_ok=True)
        report_path.write_text(json.dumps(report,indent=2))
        print(f'Rollout details: {report_path}')
    finally:
        env.close()


if __name__=='__main__':
    main()
