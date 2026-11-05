"""v11: uncertainty-aware sequential design and interpretable diagnostics.

IMPORTANT: The v10 estimator is a composite estimator. Kinematic and angular
covariances are assembled as independent blocks here, neglecting shared attitude
measurement errors, calibration uncertainty, unmodeled forces and parameter
nonlinearity. This is a LOCAL surrogate, NOT a flight-qualified posterior.
Neither the optimizer nor the policy can inspect simulator ground truth.

Coordinates z = D^-1 [m, c_x,c_y,c_z, I_Cxx,I_Cyy,I_Czz,I_Cxy,I_Cxz,I_Cyz].
Using CENTRAL inertia prevents confusing origin inertia with central moments.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import qmc
from src.inertial_dynamics import BASIS, Nudge, propagate, action_to_nudge, com_inertia, inertia, unit
from src.translation_estimation import KIN_INVERSE_MASS_PRIOR_STD_NORMALIZED

NAMES=('m','c_x','c_y','c_z','Ic_xx','Ic_yy','Ic_zz','Ic_xy','Ic_xz','Ic_yz')
UNITS=('kg','m','m','m','kg m^2','kg m^2','kg m^2','kg m^2','kg m^2','kg m^2')
D=np.array([500., .05,.05,.05,55.,55.,55.,25.,25.,25.])
CENTRAL_SCALES=D[4:]


def _sym(m):
    m=np.asarray(m,float)
    return .5*(m+m.T)


def _inverse_spd(a):
    return cho_solve(cho_factor(_sym(a),lower=True,check_finite=True),np.eye(a.shape[0]))


def _check_cov(p):
    p=_sym(p)
    ev=np.linalg.eigvalsh(p)
    if not np.isfinite(p).all() or ev[0]<=0:
        raise ValueError(f'Invalid covariance; smallest eigenvalue {ev[0]:.4g}')
    return p


def theta_from_pi(pi):
    c,I=com_inertia(pi)
    return np.r_[float(pi[0]),c,I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]]


def raw_kin_precision(est):
    kin=est.kin
    return np.diag(1/np.array([10.]*3+[2.]*3+[2.]*3+[KIN_INVERSE_MASS_PRIOR_STD_NORMALIZED])**2) + kin.F


def kin_cov_from_precision(est,precision):
    """Marginalize unknown initial position and velocity, transform q=1/m to m."""
    kin=est.kin
    inv=_inverse_spd(precision)
    P=kin.scales[:,None]*inv*kin.scales[None,:]
    q=float(kin.x[9])
    if q<=0 or not np.isfinite(q): raise ValueError('Invalid inverse mass posterior mean')
    m=1/q
    J=np.zeros((4,10))
    J[0,9]=-m*m
    J[1:4,6:9]=np.eye(3)
    return _sym(J@P@J.T)


def angular_precision(est):
    prior_sigma=est.cfg.prior_sigma_normalized
    precision=np.eye(6)/prior_sigma**2
    # One pending pair is a preview, not a second committed observation.
    pairs=list(est.angular_pairs)
    # Mirror the minimum elapsed time required for the v11 estimator's
    # active angular-momentum preview (do not count a microscopic pair).
    anchor=est.mom.anchor
    endpoint=est.mom.last_state
    if anchor is not None and endpoint is not None:
        h=endpoint.t-anchor.t
        if h>=min(.2,est.cfg.segment_s/2)-1e-10:
            pairs.append((anchor,endpoint,(est.mom.cumulative_wrench-est.mom.anchor_wrench).copy()))
    sigma=max(.012,120*est.cfg.sensor_sigma)
    for a,b,_w in pairs:
        A=np.column_stack([b.R@E@b.w-a.R@E@a.w for E in BASIS])
        Z=A*CENTRAL_SCALES[None,:]/sigma
        precision+=Z.T@Z
    return _sym(precision)


def compose_norm_cov(Pkin,Proti):
    P=np.zeros((10,10))
    P[:4,:4]=Pkin/(D[:4,None]*D[None,:4])
    P[4:,4:]=Proti/(D[4:,None]*D[None,4:])
    return _check_cov(P)


def current_cov(est):
    pk=kin_cov_from_precision(est,raw_kin_precision(est))
    pr=CENTRAL_SCALES[:,None]*_inverse_spd(angular_precision(est))*CENTRAL_SCALES[None,:]
    return compose_norm_cov(pk,pr)


def gaussian_gain(P0,P1, a_weight=1.,weak_weight=1.):
    """Local-Gaussian EIG + normalized A-gain + weak eigenmode variance reduction.

    EIG = (logdet P0 - logdet P1)/2 [nats], exact only for linear Gaussian model.
    Weak mode = eigenvector of P0 with largest normalized variance.
    Extra terms are design preferences, not information in nats.
    """
    P0=_check_cov(P0);P1=_check_cov(P1)
    s0,l0=np.linalg.slogdet(P0);s1,l1=np.linalg.slogdet(P1)
    if s0<=0 or s1<=0: raise ValueError('Nonpositive covariance determinant')
    eig,vec=np.linalg.eigh(P0)
    v=vec[:,-1]
    # Tied eigenvalues give an arbitrary eigenvector. Score the WHOLE weak
    # eigenspace, so results are invariant to rotations within that space.
    mask=eig>=eig[-1]-1e-6*max(1.,abs(eig[-1]))
    U=vec[:,mask]
    old_weak=float(np.trace(U.T@P0@U))
    new_weak=float(np.trace(U.T@P1@U))
    reduction=(old_weak-new_weak)/max(old_weak,1e-20)
    trace_red=float((np.trace(P0)-np.trace(P1))/np.trace(P0))
    eig_gain=.5*(l0-l1)
    utility=eig_gain+a_weight*trace_red+weak_weight*reduction
    return float(utility),{
        'gaussian_EIG_nats':float(eig_gain),
        'normalized_A_reduction':trace_red,
        'weak_mode_fractional_reduction':reduction,
        'weak_mode_before_variance':float(eig[-1]),
        'weak_mode_degeneracy':int(mask.sum()),
        'weak_mode_after_variance':float(v@P1@v),
        'weak_eigenspace_after_trace':new_weak,
        'weak_mode':v.tolist(),
        'before_std_physical':(np.sqrt(np.diag(P0))*D).tolist(),
        'after_std_physical':(np.sqrt(np.diag(P1))*D).tolist(),
        'per_parameter_std_reduction_physical':((np.sqrt(np.diag(P0))-np.sqrt(np.diag(P1)))*D).tolist(),
    }


@dataclass
class InformationConfig:
    A_weight:float=1.
    weak_weight:float=1.
    # Planning uses a reduced observation set and independent endpoint approximation.
    # This is intentionally NOT the dense true simulator trajectory.
    horizon_sample_s:float=2.1
    min_gain_tol:float=-1e-6
    # Active excitation should be valued relative to PASSIVE observation.
    # Otherwise a 30-s free flight baseline is incorrectly credited to the nudge.
    subtract_passive_gain:bool=True


def _predicted_matrices(est,pre,plan,dcfg,icfg,ic:InformationConfig):
    step=max(ic.horizon_sample_s,icfg.segment_s+icfg.endpoint_gap_s)
    horizon=plan.pulse_s+plan.observe_s
    ts=np.r_[np.arange(step,horizon,step),horizon]
    states=propagate(pre,est.pi,plan.point,plan.push,plan.impulse,ts)
    # Kinematics [rC0, vC0, c, q=1/m], integrated impulse history.
    kin=est.kin
    k0=raw_kin_precision(est)
    k_add=np.zeros((10,10))
    jworld=pre.R@(plan.impulse*plan.push)
    sigma=max(icfg.sensor_sigma,1e-5)
    var=sigma*sigma*(1+float(kin.x[6:9]@kin.x[6:9]))
    for s in states:
        dt=float(s.t-kin.t0)
        K=dt*(kin.impulse+jworld)-(kin.t_impulse+float(pre.t-kin.t0)*jworld)
        X=np.zeros((3,10)); X[:,:3]=np.eye(3)
        X[:,3:6]=dt*np.eye(3);X[:,6:9]=-s.R;X[:,9]=K
        Z=X*kin.scales[np.newaxis,:]
        k_add+=Z.T@Z/var
    rot0=angular_precision(est)
    rot_add=np.zeros((6,6))
    sigma_h=max(.012,120*icfg.sensor_sigma)
    previous=pre
    for s in states:
        A=np.column_stack([s.R@E@s.w-previous.R@E@previous.w for E in BASIS])
        Z=A*CENTRAL_SCALES[None,:]/sigma_h
        rot_add+=Z.T@Z
        previous=s
    return k0,k_add,rot0,rot_add


def predicted_gain(est,pre,plan,dcfg,icfg,options=None):
    """Expected gain and nudge-INCREMENTAL gain over no-force passive observation.

    Both forecasts have EXACTLY the same total observation horizon.
    Passive forecast is computed from current estimated state/parameters and is
    never fed back as a new observation (no truth, no future-information leak).
    """
    opts=options or InformationConfig()
    k0,ka,r0,ra=_predicted_matrices(est,pre,plan,dcfg,icfg,opts)
    before=compose_norm_cov(
        kin_cov_from_precision(est,k0),
        CENTRAL_SCALES[:,None]*_inverse_spd(r0)*CENTRAL_SCALES[None,:])
    after=compose_norm_cov(
        kin_cov_from_precision(est,k0+ka),
        CENTRAL_SCALES[:,None]*_inverse_spd(r0+ra)*CENTRAL_SCALES[None,:])
    absolute_u,parts=gaussian_gain(before,after,opts.A_weight,opts.weak_weight)
    if parts['gaussian_EIG_nats']<opts.min_gain_tol:
        raise ValueError('Negative model-predicted EIG; inspect numerical stability')
    # Counterfactual free-flight observation.  Same duration, same sampling,
    # same estimator uncertainty model, but zero nudge impulse.
    passive= Nudge(plan.point,plan.normal,plan.push,0.,plan.force,
                   plan.observe_s+plan.pulse_s)
    _,pk,_,pr=_predicted_matrices(est,pre,passive,dcfg,icfg,opts)
    passive_cov=compose_norm_cov(
        kin_cov_from_precision(est,k0+pk),
        CENTRAL_SCALES[:,None]*_inverse_spd(r0+pr)*CENTRAL_SCALES[None,:])
    passive_u,passive_terms=gaussian_gain(before,passive_cov,
                                        opts.A_weight,opts.weak_weight)
    incremental_u=absolute_u-passive_u
    parts['passive_EIG_nats']=passive_terms['gaussian_EIG_nats']
    parts['incremental_EIG_over_passive_nats']=(
        parts['gaussian_EIG_nats']-passive_terms['gaussian_EIG_nats'])
    parts['passive_total_utility']=float(passive_u)
    parts['absolute_nudge_total_utility']=float(absolute_u)
    parts['incremental_utility_over_passive']=float(incremental_u)
    parts['passive_after_std_physical']=(np.sqrt(np.diag(passive_cov))*D).tolist()
    return (float(incremental_u) if opts.subtract_passive_gain else float(absolute_u)),parts,before,after


def weak_modes(P,n=3):
    """Weakest constrained physical combinations in normalized coordinates."""
    ev,V=np.linalg.eigh(_check_cov(P))
    out=[]
    for j in range(1,min(n,10)+1):
        v=V[:,-j]
        if v[np.argmax(np.abs(v))]<0:v=-v
        out.append({'rank':j,'variance':float(ev[-j]),'sd_normalized':float(np.sqrt(ev[-j])),
                    'dominant':[(NAMES[i],round(float(v[i]),4)) for i in np.argsort(-np.abs(v))[:4]],
                    'eigenvector':v.tolist()})
    return out


def select_nudge(surface,est,pre,dcfg,icfg,seed=0,options=None):
    m=int(np.ceil(np.log2(dcfg.candidate_count)))
    actions=2*qmc.Sobol(d=6,scramble=True,seed=int(seed)).random_base2(m)-1
    best=None; failures=0
    for a in actions[:dcfg.candidate_count]:
        try:
            plan=action_to_nudge(a,surface,dcfg)
            u,parts,p0,p1=predicted_gain(est,pre,plan,dcfg,icfg,options)
            if np.isfinite(u) and (best is None or u>best['utility']):
                best={'action':a.astype(np.float32),'nudge':plan,'utility':float(u),
                      'parts':parts,'before_cov':p0,'after_cov':p1}
        except (ValueError,RuntimeError,np.linalg.LinAlgError,OverflowError,FloatingPointError):failures+=1
    if best is None:raise RuntimeError(f'No valid designs among {dcfg.candidate_count} candidates')
    best['rejected_candidates']=failures
    return best


def component_summary(estimated,truth,P):
    """Ground-truth oracle: EVALUATION ONLY. Never call in policy/score."""
    delta=theta_from_pi(estimated)-theta_from_pi(truth)
    std=np.sqrt(np.diag(_check_cov(P)))*D
    z=delta/std
    return {'signed_error_physical':delta.tolist(),'abs_error_physical':np.abs(delta).tolist(),
            'reported_std_physical':std.tolist(), 'z_score':z.tolist(),
            'within_95pct_normal':(np.abs(z)<=1.96).tolist(),
            'quadratic_NES_approx':float((delta/D)@_inverse_spd(P)@(delta/D))}
