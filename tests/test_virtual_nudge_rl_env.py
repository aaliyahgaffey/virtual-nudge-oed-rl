"""Gymnasium integration tests (run from active_inertial_id repository root).

Requires: pip install gymnasium pytest
Optional:  pip install stable-baselines3
Does NOT require MuJoCo for these tests.
"""
from pathlib import Path
import numpy as np
import pytest

gym = pytest.importorskip('gymnasium')
from src.virtual_nudge_rl_env import (
    VirtualNudgePlacementEnv, RLEnvConfig,
    normalized_action_to_plan, plan_to_normalized_action,
)
from src.three_nudge_virtual_10param import (
    Virtual10Settings, select_virtual_plan,
)

ROOT=Path(__file__).resolve().parents[1]


def make_env(**kw):
    return VirtualNudgePlacementEnv(
        ROOT, settings=Virtual10Settings(noise_std=0.0),
        rl_config=RLEnvConfig(backend='surrogate',**kw))


def test_three_nudge_episode_updates_estimator_and_returns_valid_observations():
    env=make_env(randomize_surrogate_truth=False)
    try:
        obs,info=env.reset(seed=17)
        assert env.observation_space.contains(obs)
        assert info['backend']=='surrogate'
        num_updates=0
        for k in range(3):
            obs,reward,done,truncated,info=env.step(np.zeros(6,dtype=np.float32))
            assert env.observation_space.contains(obs)
            assert np.isfinite(reward) and reward>=0
            assert not truncated
            assert not info.get('safety_or_model_failure',False),info
            assert done==(k==2)
            assert info['estimator_updates']>num_updates
            num_updates=info['estimator_updates']
            assert info['impulse_Ns']>0
            assert info['numerical_rank']>=0
        with pytest.raises(RuntimeError):
            env.step(np.zeros(6,dtype=np.float32))
    finally:
        env.close()


def test_action_map_always_respects_existing_actuator_limits():
    env=make_env(randomize_surrogate_truth=False)
    try:
        rng=np.random.default_rng(4)
        cfg=env.cfg
        valid_count=0
        for _ in range(100):
            action=rng.uniform(-1,1,6)
            try:
                plan=normalized_action_to_plan(action,env.surface,cfg,30.)
            except ValueError:
                continue  # Some surface points could exceed torque feasibility.
            valid_count+=1
            assert cfg.min_pulse_s-1e-9<=plan.pulse_s<=cfg.max_pulse_s+1e-9
            assert cfg.impulse_min_Ns-1e-9<=plan.impulse_Ns<=cfg.max_impulse_Ns+1e-9
            assert cfg.force_min_N-1e-9<=plan.force_N<=cfg.max_force_N+1e-9
            assert np.linalg.norm(np.cross(plan.point_body,plan.force_N*plan.push_body))<=cfg.max_torque_Nm+1e-9
        assert valid_count > 50
    finally:
        env.close()


def test_reward_and_policy_observation_never_use_truth_diagnostic():
    env1=make_env(randomize_surrogate_truth=True,debug_truth=False)
    env2=make_env(randomize_surrogate_truth=True,debug_truth=True)
    try:
        o1,_=env1.reset(seed=6)
        o2,_=env2.reset(seed=6)
        np.testing.assert_allclose(o1,o2)
        a=np.array([0.1,-0.3,-0.2,0.0,0.4,-0.3],np.float32)
        for _ in range(3):
            o1,r1,d1,_,i1=env1.step(a)
            o2,r2,d2,_,i2=env2.step(a)
            np.testing.assert_allclose(o1,o2)
            assert r1==pytest.approx(r2)
            assert d1==d2
            assert 'DEV_ONLY_errors' not in i1
            assert 'DEV_ONLY_errors' in i2
    finally:
        env1.close();env2.close()


def test_existing_sobol_planner_can_use_same_step_interface():
    env=make_env(randomize_surrogate_truth=False)
    try:
        env.reset(seed=6)
        plan,_=select_virtual_plan(
            env.surface,env.identifier.last_state,env.identifier,None,env.cfg,
            1,np.random.default_rng(1))
        a=plan_to_normalized_action(plan,env.surface,env.cfg)
        recovered=normalized_action_to_plan(a,env.surface,env.cfg,30.)
        np.testing.assert_allclose(recovered.point_body,plan.point_body,atol=1e-4)
        _,reward,done,_,info=env.step(a)
        assert not done and reward>=0
        assert not info.get('safety_or_model_failure',False)
    finally:
        env.close()


def test_gymnasium_reset_step_api_passes_sb3_checker():
    sb3=pytest.importorskip('stable_baselines3')
    from stable_baselines3.common.env_checker import check_env
    env=make_env(randomize_surrogate_truth=True)
    try:
        check_env(env,warn=True,skip_render_check=True)
    finally:
        env.close()
