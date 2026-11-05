"""No-MuJoCo validation tests for the online full-ten-parameter estimator.
Run: python -m pytest tests/test_virtual_10param_continuous.py -q
"""
import numpy as np
from scipy.spatial.transform import Rotation
from src.physical_oed_identification import (
    OriginState, momentum_regressor, com_and_ic, SCALE)
from src.three_nudge_virtual_10param import (
    Virtual10Settings, ContinuousSpatialIdentifier,
    _candidate_from_unit, VirtualPlan)

PI_TRUE = np.array([638.027, 18.0, -12.0, 8.0,
                    57.9, 67.9, 73.3, 0.8, -0.7, 0.3])


def synthetic_state(t):
    # Vary attitude, angular rate, position and linear speed to excite all
    # ten dimensions. Not intended to be a feasible free-floating trajectory.
    return OriginState(
        r=np.array([3. + .2*np.sin(.5*t), .15*np.cos(.9*t), .1*np.sin(.8*t)]),
        v=np.array([.1*np.cos(.5*t), -.135*np.sin(.9*t), .08*np.cos(.8*t)]),
        R=Rotation.from_euler('xyz',
            [.28*np.sin(.7*t), .26*np.cos(1.1*t), .33*np.sin(1.3*t)]).as_matrix(),
        omega_body=np.array([.14*np.cos(.7*t), -.28*np.sin(1.1*t),
                              .22*np.cos(1.3*t)]), t=float(t))


def test_all_ten_parameters_streaming_recovery_without_noise():
    cfg=Virtual10Settings(noise_std=0.0, estimator_window_s=.10,
                          prior_sigma_normalized=100.0)
    est=ContinuousSpatialIdentifier(cfg)
    pre=synthetic_state(0)
    est.start(pre)
    for t in np.arange(.02, 6.001, .02):
        current=synthetic_state(t)
        dW=(momentum_regressor(current)-momentum_regressor(pre))@PI_TRUE
        est.update(current,dW)
        pre=current
    assert est.total_samples==301
    assert est.updates==300
    np.testing.assert_allclose(est.pi,PI_TRUE,rtol=1e-4,atol=1e-3)
    assert est.metrics()['rank']==10
    com_and_ic(est.pi)


def test_zero_wrench_does_not_create_fake_impulse():
    cfg=Virtual10Settings(noise_std=0.,estimator_window_s=.1)
    est=ContinuousSpatialIdentifier(cfg)
    state=OriginState(np.array([3.,0,0]),np.zeros(3),np.eye(3),np.zeros(3),0.)
    est.start(state)
    for k in range(1,251):
        new=OriginState(state.r.copy(),state.v.copy(),state.R.copy(),
                        state.omega_body.copy(),k*.002)
        est.update(new,np.zeros(6)); state=new
    assert est.total_samples==251 and est.updates==250
    np.testing.assert_allclose(est.cumulative_wrench,np.zeros(6))
    np.testing.assert_allclose(est.pi,est.prior)
    assert est.metrics()['rank']==0


def test_noise_window_measures_every_step_without_naive_differencing():
    cfg=Virtual10Settings(noise_std=1e-4,estimator_window_s=.2)
    est=ContinuousSpatialIdentifier(cfg)
    state=OriginState(np.array([3.,0,0]),np.zeros(3),np.eye(3),np.zeros(3),0.)
    est.start(state)
    for k in range(1,151):
        new=OriginState(state.r.copy(),state.v.copy(),state.R.copy(),
                        state.omega_body.copy(),k*.002)
        est.update(new,np.zeros(6));state=new
    assert est.total_samples==151
    assert est.updates == 150  # estimator refreshes each simulation step
    assert est.completed_segments == 1  # 0.3s / 0.2s, not 150 pseudo-independent rows


def test_config_rejects_invalid_window():
    try:
        Virtual10Settings(estimator_window_s=0)
    except ValueError:
        pass
    else:
        raise AssertionError('Zero window must not be accepted')


def test_impulse_integration_and_moment_about_fixed_origin():
    dt=.002;duration=.5
    F=np.array([3.,-1.,2.]); application=np.array([3.2,.1,-.2])
    integrated=np.zeros(6)
    for _ in range(int(round(duration/dt))):
        integrated+=dt*np.r_[F,np.cross(application,F)]
    np.testing.assert_allclose(integrated[:3],F*duration,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(integrated[3:],np.cross(application,F)*duration,
                               rtol=1e-12,atol=1e-12)


def test_information_only_planner_does_not_reject_large_target_motion(monkeypatch):
    """Even wildly moving target states must not trigger drift/speed screening."""
    import src.three_nudge_virtual_10param as mod
    cfg = Virtual10Settings(n_candidates=16)
    est = ContinuousSpatialIdentifier(cfg)
    # Deliberately far from the initial reference and fast-moving.
    pre = OriginState(np.array([20., 0, 0]), np.array([2., 0, 0]),
                      np.eye(3), np.zeros(3), 30.)
    initial = OriginState(np.zeros(3), np.zeros(3), np.eye(3),
                          np.zeros(3), 0.)
    class SurfaceStub:
        def point_and_normal(self, *args):
            return np.array([.4, 0, 0]), np.array([1., 0, 0])
    # Avoid dependence on the numerical OED surrogate for this policy test.
    monkeypatch.setattr(mod, '_candidate_rate_information',
                        lambda *args: np.eye(10))
    monkeypatch.setattr(mod, 'e_opt_score',
                        lambda *args: (1.0, {'E_gain':1.0,'subspace_excitation':0.0}))
    plan, scores = mod.select_virtual_plan(
        SurfaceStub(), pre, est, initial, cfg, 1, np.random.default_rng(3))
    assert plan is not None
    assert len(scores) > 0
    assert np.allclose(scores[:, 0], 1.0)
    assert scores.shape[1] == 12  # objective, J, F, T, point3, push3, E, subspace


def test_information_objective_prefers_high_gain_without_impulse_penalty(monkeypatch):
    import src.three_nudge_virtual_10param as mod
    cfg = Virtual10Settings(n_candidates=16)
    est = ContinuousSpatialIdentifier(cfg)
    pre = synthetic_state(0.0)
    low = VirtualPlan(np.array([.3,0,0]),np.array([1.,0,0]),
                      np.array([-1.,0,0]), .3, 1.5, 30.)
    high = VirtualPlan(np.array([.3,0,0]),np.array([1.,0,0]),
                       np.array([-1.,0,0]), 2.0, 4.0, 30.)
    monkeypatch.setattr(mod, '_decision_samples', lambda *args: [0, 1])
    monkeypatch.setattr(mod, '_candidate_from_unit',
                        lambda x, *args: low if x == 0 else high)
    monkeypatch.setattr(mod, '_candidate_rate_information',
                        lambda pre, pi, p, cfg: np.eye(10) * (1 if p is low else 2))
    monkeypatch.setattr(mod, 'e_opt_score',
                        lambda prior, F, projector: (float(F[0,0]),
                            {'E_gain':float(F[0,0]),'subspace_excitation':0.0}))
    plan, scores = mod.select_virtual_plan(None, pre, est, None,
                                           cfg, 1, np.random.default_rng(1))
    assert plan.impulse_Ns == high.impulse_Ns
    assert plan.utility == 2.0
    assert sorted(scores[:, 0]) == [1.0, 2.0]


def test_force_and_torque_restrictions_still_apply():
    cfg = Virtual10Settings(max_torque_Nm=0.01)
    class SurfaceStub:
        def point_and_normal(self,*args):
            return np.array([2.,0.,0.]), np.array([0.,1.,0.])
    x = np.array([.1,.1,.1,.0,.6,.6])
    assert _candidate_from_unit(x,SurfaceStub(),cfg,30.) is None


def test_online_rate_residual_is_recorded_without_statistical_interpretation():
    cfg = Virtual10Settings(noise_std=0.0, estimator_window_s=0.1)
    est = ContinuousSpatialIdentifier(cfg)
    before = synthetic_state(0.0)
    after = synthetic_state(0.02)
    est.start(before)
    wrench = (momentum_regressor(after)-momentum_regressor(before)) @ PI_TRUE
    est.update(after, wrench)
    assert est.last_window_s > 0
    assert np.isfinite(est.last_weighted_rate_residual)
    assert est.gram.shape == (10, 10)
    np.testing.assert_allclose(est.gram, est.gram.T, atol=1e-11)


def test_live_preview_is_replaced_not_added_as_independent_data():
    cfg = Virtual10Settings(estimator_window_s=0.2, prior_sigma_normalized=10.0)
    est = ContinuousSpatialIdentifier(cfg)
    prior = synthetic_state(0.0)
    est.start(prior)
    for t in [0.02, 0.04, 0.06, 0.08]:
        st = synthetic_state(t)
        dW = (momentum_regressor(st) - momentum_regressor(prior)) @ PI_TRUE
        est.update(st, dW)
        prior = st
    assert est.completed_segments == 0
    current = synthetic_state(0.08)
    dB = momentum_regressor(current) - momentum_regressor(synthetic_state(0.0))
    denom = np.sqrt(cfg.estimator_window_s)
    W = est.rate_W
    X = (W @ dB @ np.diag(SCALE)) / denom
    np.testing.assert_allclose(est.gram, X.T @ X, rtol=1e-10, atol=1e-7)


def test_completed_segments_contribute_once_and_endpoints_advance():
    cfg = Virtual10Settings(estimator_window_s=0.1, prior_sigma_normalized=10.0)
    est = ContinuousSpatialIdentifier(cfg)
    prev = synthetic_state(0.0)
    est.start(prev)
    for t in np.arange(.02, .301, .02):
        st = synthetic_state(t)
        est.update(st, (momentum_regressor(st)-momentum_regressor(prev))@PI_TRUE)
        prev = st
    assert est.total_samples == 16
    assert est.updates == 15
    assert est.completed_segments == 3
    np.testing.assert_allclose(est.gram, est.committed_gram, atol=1e-7)
    assert np.linalg.norm(est.committed_gram) > 0


def test_new_default_prior_is_less_restrictive():
    assert Virtual10Settings().prior_sigma_normalized == 10.0
