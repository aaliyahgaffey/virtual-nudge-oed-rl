"""Numerical/physical-regression tests for the three-nudge research prototype."""
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from src.asteroid_surface import Surface
from src.nudge_v10_math import (
    State,REFERENCE_PI,IDConfig,DesignConfig,SCALE,momentum,momentum_regressor,
    state_momentum_jacobian,propagate,action_to_nudge,com_inertia,
)
from src.nudge_v10_design import predicted_hybrid_gain,optimize_hybrid_one
from src.nudge_v10_translation import TranslationEstimator,HybridEstimator
from src.nudge_v10_rollout import rollout
ROOT=Path(__file__).resolve().parents[1]


def surface():return Surface(ROOT/'model'/'asteroid.obj')


def state():return State(np.array([3.,.5,.3]),np.array([.02,-.03,.004]),
    Rotation.from_rotvec([.2,-.3,.1]).as_matrix(),np.array([.07,.03,-.08]),0.)


def test_momentum_jacobian_matches_central_difference():
    s=state();pi=REFERENCE_PI.copy();G=state_momentum_jacobian(s,pi)
    h=1e-5
    for i in range(12):
        d=np.eye(12)[i]
        def perturb(sign):
            return State(s.r+sign*h*d[:3],s.v+sign*h*d[3:6],
                Rotation.from_rotvec(sign*h*d[6:9]).as_matrix()@s.R,
                s.w+sign*h*d[9:12],s.t)
        fd=(momentum(perturb(1),pi)-momentum(perturb(-1),pi))/(2*h)
        np.testing.assert_allclose(fd,G[:,i],rtol=3e-7,atol=1e-6)


def test_propagation_conserves_spatial_momentum_to_tight_tolerance():
    s=state();p=action_to_nudge(np.zeros(6),surface(),DesignConfig())
    after=propagate(s,REFERENCE_PI,p.point,p.push,p.impulse,[.5,2,10,30])
    J=s.R@(p.impulse*p.push)
    L=np.cross(s.r+s.R@p.point,J)
    np.testing.assert_allclose(momentum(after[0],REFERENCE_PI)-momentum(s,REFERENCE_PI),
        np.r_[J,L],atol=2e-7)
    for x in after[1:]:
        np.testing.assert_allclose(momentum(x,REFERENCE_PI),momentum(after[0],REFERENCE_PI),atol=2e-7)


def test_physical_inertia_and_surface_action_constraints():
    c,I=com_inertia(REFERENCE_PI)
    assert np.linalg.eigvalsh(I).min()>0
    surf=surface();dc=DesignConfig()
    for i in range(50):
        a=np.random.default_rng(i).uniform(-1,1,6)
        p=action_to_nudge(a,surf,dc)
        assert dc.min_impulse<=p.impulse<=dc.max_impulse+1e-9
        assert dc.min_force<=p.force<=dc.max_force+1e-9
        assert dc.min_pulse-1e-9<=p.pulse_s<=dc.max_pulse+1e-9
        assert np.linalg.norm(np.cross(p.point,p.force*p.push))<=dc.max_torque+1e-9


def test_force_is_inferred_from_pose_and_known_impulse_not_truth():
    cfg=IDConfig(sensor_sigma=0.)
    s=state();k=TranslationEstimator(cfg);k.start(s)
    p=action_to_nudge(np.zeros(6),surface(),DesignConfig())
    j=s.R@(p.impulse*p.push)
    k.add_impulse(s.t,j)
    states=propagate(s,REFERENCE_PI,p.point,p.push,p.impulse,np.arange(1.,31.,1.))
    for item in states:k.update(item)
    # A single impulse at time zero is confounded with unknown initial vC0.
    # Another impulse at a distinct time makes inverse mass identifiable.
    s2=states[-1]
    p2=action_to_nudge(np.ones(6)*.25,surface(),DesignConfig())
    k.add_impulse(s2.t,s2.R@(p2.impulse*p2.push))
    states2=propagate(s2,REFERENCE_PI,p2.point,p2.push,p2.impulse,np.arange(1.,31.,1.))
    for item in states2:k.update(item)
    assert k.reliable()
    mass,c=k.mass_com();c0,_=com_inertia(REFERENCE_PI)
    assert abs(mass-REFERENCE_PI[0])<1e-2
    assert np.linalg.norm(c-c0)<1e-4


def test_closed_loop_oed_three_nudges_noiseless():
    r=rollout(surface(),19,'oed',IDConfig(sensor_sigma=0.),DesignConfig(candidate_count=32))
    assert len(r['history'])==3
    assert [x['nudge'] for x in r['history']]==[1,2,3]
    assert r['final_errors_DEV_ONLY']['mass_error_kg']<.5
    assert r['final_errors_DEV_ONLY']['com_error_mm']<1
    assert r['final_errors_DEV_ONLY']['inertia_error_kgm2']<1
    assert all(h['completed_independent_pairs']>0 for h in r['history'])


def test_closed_loop_oed_three_nudges_noise_1e_minus_4():
    r=rollout(surface(),19,'oed',IDConfig(sensor_sigma=1e-4),DesignConfig(candidate_count=32))
    assert r['final_errors_DEV_ONLY']['mass_error_kg']<5
    assert r['final_errors_DEV_ONLY']['com_error_mm']<2
    assert r['final_errors_DEV_ONLY']['inertia_error_kgm2']<10


def test_predicted_gain_is_finite_and_does_not_need_true_pi():
    ic=IDConfig(sensor_sigma=1e-4); dc=DesignConfig(candidate_count=16)
    e=HybridEstimator(ic);s=state();e.start(s)
    a,p,g,parts=optimize_hybrid_one(surface(),s,e,dc,ic,seed=6)
    assert np.isfinite(g) and g>0
    assert len(a)==6
    assert parts['kinematic_D']>=-1e-7
    assert parts['inertia_D']>=-1e-7


def test_three_nudge_environment_if_gymnasium_installed():
    pytest.importorskip('gymnasium')
    from src.nudge_v10_env import ThreeNudgeEnv,EnvConfig
    env=ThreeNudgeEnv(ROOT,id_config=IDConfig(sensor_sigma=0.),
        design_config=DesignConfig(candidate_count=16),
        env_config=EnvConfig(randomized_truth=False,debug_truth=False))
    try:
        obs,_=env.reset(seed=20)
        assert env.observation_space.contains(obs)
        for k in range(3):
            obs,reward,term,trunc,info=env.step(np.zeros(6,np.float32))
            assert env.observation_space.contains(obs)
            assert np.isfinite(reward)
            assert 'debug_truth_errors' not in info
            assert term==(k==2)
            assert not trunc
            assert 'failure_reason' not in info,info
    finally:env.close()


def test_sb3_env_checker_if_installed():
    pytest.importorskip('gymnasium')
    pytest.importorskip('stable_baselines3')
    from stable_baselines3.common.env_checker import check_env
    from src.nudge_v10_env import ThreeNudgeEnv
    env=ThreeNudgeEnv(ROOT)
    try:check_env(env,warn=True,skip_render_check=True)
    finally:env.close()


def test_live_estimator_recomputes_without_waiting_for_segment_close():
    """The 10-vector is refreshed at every sample; force events are discrete."""
    cfg=IDConfig(sensor_sigma=0.,segment_s=2.)
    e=HybridEstimator(cfg)
    s=state(); e.start(s); e.begin_nudge(s)
    p=action_to_nudge(np.zeros(6),surface(),DesignConfig())
    J=s.R@(p.impulse*p.push)
    L=np.cross(s.r+s.R@p.point,J)
    e.add_impulse(s.t,J)
    states=propagate(s,REFERENCE_PI,p.point,p.push,p.impulse,[.25,.5,.75,1.,1.25,1.5])
    for i,ss in enumerate(states):
        e.update(ss,np.r_[J,L] if i==0 else np.zeros(6))
    assert e.events==0  # no completed 2s interval yet
    assert e.updates==6
    assert e.live_inertia_updates==6
    assert e.kin.updates==7  # includes initial state sample
    assert np.all(np.isfinite(e.pi))
    assert np.linalg.norm(e.kin.F)>0
