"""Scientific/mathematical tests; Gym-dependent checks skip if Gym unavailable."""
import numpy as np
import pytest
from pathlib import Path
from scipy.spatial.transform import Rotation
from src.inertial_dynamics import (REFERENCE_PI, PRIOR_PI, IDConfig,DesignConfig,State,
    propagate,com_inertia,action_to_nudge,skew,unit)
from src.information_gain import (D,NAMES,theta_from_pi,gaussian_gain,weak_modes,
    predicted_gain,current_cov,select_nudge)
from src.asteroid_surface_v11 import Surface

ROOT=Path(__file__).resolve().parents[1]


def test_surface_stable_and_hull_constraints():
    s=Surface(ROOT/'model/asteroid.obj')
    for lon in np.linspace(0,2*np.pi,9):
        for z in (-1,-.8,0,.7,1):
            p,n=s.point_and_normal(lon,z)
            assert np.isfinite(p).all() and np.isfinite(n).all()
            assert np.max(np.einsum('ij,j->i',s.A,p,optimize=False)+s.b)<1e-5
            assert np.isclose(np.linalg.norm(n),1)


def test_physical_central_inertia_transform():
    p=theta_from_pi(REFERENCE_PI)
    c,Ic=com_inertia(REFERENCE_PI)
    assert np.isclose(p[0],REFERENCE_PI[0])
    assert np.allclose(p[1:4],c)
    assert np.allclose(p[4:7],np.diag(Ic))
    assert not np.allclose(p[4:7],REFERENCE_PI[4:7])


def test_gaussian_gain_identity_and_weak_mode():
    p=np.diag([.1]*9+[10.]);q=p.copy()
    utility,parts=gaussian_gain(p,q)
    assert abs(utility)<1e-12
    assert parts['gaussian_EIG_nats']==pytest.approx(0,abs=1e-12)
    assert weak_modes(p)[0]['dominant'][0][0]=='Ic_yz'
    q[-1,-1]=1.
    u,parts=gaussian_gain(p,q,a_weight=0,weak_weight=0)
    assert u==pytest.approx(.5*np.log(10))
    assert parts['weak_mode_fractional_reduction']==pytest.approx(.9)


def test_bad_covariance_rejected():
    with pytest.raises(ValueError): gaussian_gain(np.diag([-1.]+[1.]*9),np.eye(10))


def test_force_surface_nudge_geometry():
    mesh=Surface(ROOT/'model/asteroid.obj')
    dc=DesignConfig(candidate_count=8)
    n=action_to_nudge(np.zeros(6),mesh,dc)
    assert dc.min_impulse<=n.impulse<=dc.max_impulse
    assert dc.min_force<=n.force<=dc.max_force
    assert n.pulse_s>=dc.min_pulse-1e-10 and n.pulse_s<=dc.max_pulse+1e-10


def test_impulse_moment_changes_angular_rate():
    mesh=Surface(ROOT/'model/asteroid.obj')
    p=action_to_nudge(np.zeros(6),mesh,DesignConfig())
    pre=State(np.array([3.,0.,0.]),np.zeros(3),np.eye(3),np.zeros(3),0)
    _,Ic=com_inertia(REFERENCE_PI)
    # Nonzero torque about CoM under force action, no external reaction modeled.
    c,_=com_inertia(REFERENCE_PI)
    angular_jump=np.linalg.solve(Ic,np.cross(p.point-c,p.push*p.impulse))
    states=propagate(pre,REFERENCE_PI,p.point,p.push,p.impulse,[1e-6])
    assert np.allclose(states[0].w,angular_jump,rtol=1e-3,atol=1e-5)


def test_continuous_estimator_and_finite_predicted_information():
    pytest.importorskip('gymnasium')
    from src.experiment_workflow import make_env,run_episode
    env=make_env(ROOT,seed=19,candidates=8,sample_s=.5,trace=True)
    row,details,trace=run_episode(env,19,'oed_v11',include_trace=True)
    assert row['success'] and row['continuous_estimator_updates']>150
    assert len(details)==3 and len(trace)>150
    assert all(np.isfinite(x['predicted_eig_nats']) for x in details)
    assert np.all(np.linalg.eigvalsh(current_cov(env.est))>0)
    assert all(x['predicted_parts']['gaussian_EIG_nats']>=-1e-6 for x in details)
    env.close()


def test_no_truth_in_information_interface():
    pytest.importorskip('gymnasium')
    from src.experiment_workflow import make_env
    env=make_env(ROOT,seed=19,candidates=8)
    a=select_nudge(env.surface,env.est,env.est.last_state,env.dcfg,env.icfg,seed=19)
    assert 'truth' not in a
    assert np.isfinite(a['utility'])
    assert len(a['parts']['weak_mode'])==10
    env.close()


def test_gymnasium_checker():
    gym=pytest.importorskip('gymnasium')
    from gymnasium.utils.env_checker import check_env
    from src.experiment_workflow import make_env
    env=make_env(ROOT,candidates=8)
    check_env(env,skip_render_check=True)
    env.close()


def test_noise_and_coverage_report():
    pytest.importorskip('gymnasium')
    from src.experiment_workflow import make_env,run_episode,offline_coverage
    import pandas as pd
    env=make_env(ROOT,seed=20,candidates=8,sample_s=1.)
    r,n,_=run_episode(env,20,'oed_v11')
    assert r['success'] and all(np.isfinite([r['z_'+name] for name in NAMES]))
    c=offline_coverage(pd.DataFrame([r]))
    assert len(c)==10
    env.close()


def test_repeated_largest_eigenvalue_scores_full_subspace():
    # With two tied worst directions, no arbitrary eigenvector should decide.
    P=np.diag([1.]*8+[10.,10.])
    Q=np.diag([1.]*8+[6.,6.])
    score,terms=gaussian_gain(P,Q,a_weight=0,weak_weight=1.)
    assert terms['weak_mode_degeneracy']==2
    assert terms['weak_mode_fractional_reduction']==pytest.approx(.4)
    # A rotation confined to the tied eigenspace leaves the objective unchanged.
    angle=.53
    G=np.eye(10);G[-2:,-2:]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
    # Small symmetric transforms without BLAS-backed matmul (macOS NumPy).
    GP=np.einsum('ik,kl,jl->ij',G,P,G,optimize=False)
    GQ=np.einsum('ik,kl,jl->ij',G,Q,G,optimize=False)
    score2,terms2=gaussian_gain(GP,GQ,a_weight=0,weak_weight=1.)
    assert score2==pytest.approx(score)


def test_nudge_incremental_information_subtracts_passive():
    from src.translation_estimation import HybridEstimator
    from src.information_gain import InformationConfig
    cfg=IDConfig(sensor_sigma=1e-4)
    est=HybridEstimator(cfg)
    pre=State(np.array([3.,0.,0.]),np.zeros(3),np.eye(3),
              np.array([.01,.02,.03]),0.)
    est.start(pre)
    surface=Surface(ROOT/'model/asteroid.obj')
    dc=DesignConfig(candidate_count=8)
    p=action_to_nudge(np.zeros(6),surface,dc)
    u,terms,p0,p1=predicted_gain(est,pre,p,dc,cfg,InformationConfig())
    assert u==pytest.approx(terms['absolute_nudge_total_utility']-terms['passive_total_utility'])
    assert terms['incremental_EIG_over_passive_nats']==pytest.approx(
        terms['gaussian_EIG_nats']-terms['passive_EIG_nats'])
    assert np.linalg.eigvalsh(p1).min()>0
    assert np.linalg.eigvalsh(p0).min()>0


def test_surface_projection_without_matmul_fpe():
    # Regression for macOS/Accelerate NumPy reporting spurious floating-point
    # errors in small BLAS matrix products with errstate(raise).
    s=Surface(ROOT/'model/asteroid.obj')
    with np.errstate(divide='raise',over='raise',invalid='raise'):
        for lon,z in [(0.,-1.),(np.pi,0.),(1.23,.75),(2*np.pi,1.)]:
            p,n=s.point_and_normal(lon,z)
            assert np.isfinite(p).all() and np.isfinite(n).all()
            violations=np.einsum('ij,j->i',s.A,p,optimize=False)+s.b
            assert violations.max()<=1e-5
