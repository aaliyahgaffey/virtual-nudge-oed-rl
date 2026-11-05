"""Analytical/unit checks for virtual 10-parameter impulse identification.

Run from repository root:
    python -m pytest tests/test_virtual_10param.py -q
No MuJoCo is needed for these mathematical tests.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from src.physical_oed_identification import (
    OriginState, com_and_ic, momentum_regressor, segment_rows, svd_observability)
from src.three_nudge_virtual_10param import (
    Virtual10Settings, VirtualPlan, _candidate_from_unit)

TRUE_PI=np.array([638.027, 18.0, -12.0, 8.0,
                  57.9, 67.9, 73.3, 0.8, -0.7, 0.3])


def instantaneous_transition(pre, pi, point_body, impulse_body):
    """Compute exact instantaneous impulse jump, with unknown COM offset."""
    mass=pi[0]
    c,Ic=com_and_ic(pi)
    R=pre.R
    wnew=pre.omega_body + np.linalg.solve(Ic,np.cross(point_body-c,impulse_body))
    vC_before=pre.v+R@np.cross(pre.omega_body,c)
    vC_after=vC_before+R@impulse_body/mass
    v_origin_after=vC_after-R@np.cross(wnew,c)
    post=OriginState(pre.r.copy(),v_origin_after,R.copy(),wnew,pre.t)
    Jw=R@impulse_body
    applied_point=pre.r+R@point_body
    six_impulse=np.r_[Jw,np.cross(applied_point,Jw)]
    return post,six_impulse


def test_full_ten_parameter_momentum_balance():
    R=Rotation.from_euler('xyz',[.4,-.25,.8]).as_matrix()
    pre=OriginState(np.array([3.1,.2,-.1]),np.array([.001,-.004,.002]),
                    R,np.array([.025,-.015,.009]))
    post,wrench=instantaneous_transition(pre,TRUE_PI,
                        np.array([.22,.34,-.28]),np.array([-.65,.12,.21]))
    A=momentum_regressor(post)-momentum_regressor(pre)
    np.testing.assert_allclose(A@TRUE_PI,wrench,atol=2e-13,rtol=1e-12)
    assert np.linalg.norm(wrench[3:])>0  # moment about WORLD origin is present


def test_three_diverse_ideal_impulses_recover_all_ten():
    rng=np.random.default_rng(72)
    rows=[];loads=[]
    # Three differently-oriented starts, each with a different off-axis
    # impulse, are enough to produce a full-rank noiseless synthetic test.
    for k in range(3):
        R=Rotation.from_rotvec(rng.normal(size=3)).as_matrix()
        pre=OriginState(rng.normal(size=3)+[3,0,0],
                        rng.normal(scale=.01,size=3), R,
                        rng.normal(scale=.025,size=3))
        p=rng.normal(scale=.3,size=3)
        J=rng.normal(scale=.75,size=3)
        post, wrench=instantaneous_transition(pre,TRUE_PI,p,J)
        A,b=segment_rows(pre,[post],wrench)
        rows.append(A);loads.append(b)
    A=np.vstack(rows);b=np.concatenate(loads)
    diag=svd_observability(A,rel_tol=1e-8)
    assert diag['rank']==10, diag['singular_values']
    estimate=np.linalg.lstsq(A,b,rcond=None)[0]
    np.testing.assert_allclose(estimate,TRUE_PI,atol=1e-9,rtol=1e-10)


def test_segment_rows_applies_wrench_once():
    R=np.eye(3)
    pre=OriginState(np.array([3.,0,0]),np.zeros(3),R,np.zeros(3))
    post,W=instantaneous_transition(pre,TRUE_PI,
                                     np.array([.3,.2,0]),np.array([-1.,.1,.2]))
    A,b=segment_rows(pre,[post,post],W)
    np.testing.assert_allclose(A[6:],np.zeros((6,10)))
    np.testing.assert_allclose(b[:6],W)
    np.testing.assert_allclose(b[6:],np.zeros(6))


def test_settings_reject_unsafe_impulse_limit():
    try:
        Virtual10Settings(impulse_max_Ns=3.5, max_impulse_Ns=3.0)
    except ValueError:
        pass
    else:
        raise AssertionError('Unsafe impulse settings must be rejected')
