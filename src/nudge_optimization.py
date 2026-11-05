"""Analytical one-step lookahead for the HYBRID (pose + angular momentum) estimator.

The sequential decision rule is explicit: choose an admissible nudge to maximize
expected reductions in local Gaussian covariance of [c,1/m] (after eliminating
unknown initial position/velocity) and the six central-inertia coefficients.
Sums use only the CURRENT posterior, predicted states and commanded impulses.
They never use simulator ground truth or realized parameter estimation errors.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.stats import qmc
from src.inertial_dynamics import (BASIS,propagate,action_to_nudge,unit)


def _inv(P):
    return cho_solve(cho_factor(.5*(P+P.T),lower=True),np.eye(len(P)))


def _block_gain(P0,P1,indices,weight=3.):
    a=P0[np.ix_(indices,indices)];b=P1[np.ix_(indices,indices)]
    det0=np.linalg.slogdet(a);det1=np.linalg.slogdet(b)
    if min(det0[0],det1[0])<=0: raise ValueError('Invalid marginal covariance')
    d=det0[1]-det1[1]
    red=(np.trace(a)-np.trace(b))/max(np.trace(a),1e-12)
    return float(d+weight*red),float(d),float(red)


def predicted_hybrid_gain(estimator,pre,plan,icfg,dcfg):
    kin=estimator.kin
    # Include whole observation; no new constraints on asteroid translation.
    interval=max(dcfg.plan_interval_s,icfg.segment_s+icfg.endpoint_gap_s)
    horizon=plan.pulse_s+plan.observe_s
    ts=np.r_[np.arange(interval,horizon,interval),horizon]
    predicted=propagate(pre,estimator.pi,plan.point,plan.push,plan.impulse,ts)
    impulse_world=pre.R@(plan.impulse*plan.push)
    kkin=np.diag(1/np.array([10]*3+[2]*3+[2]*3+[2])**2)+kin.F
    kcand=np.zeros((10,10))
    sigma=max(icfg.sensor_sigma,1e-5)
    current_com=kin.x[6:9]
    var=sigma**2*(1+np.dot(current_com,current_com))
    for s in predicted:
        elapsed=s.t-kin.t0
        K=elapsed*(kin.impulse+impulse_world)-(
            kin.t_impulse+(pre.t-kin.t0)*impulse_world)
        X=np.zeros((3,10))
        X[:,:3]=np.eye(3);X[:,3:6]=elapsed*np.eye(3)
        X[:,6:9]=-s.R;X[:,9]=K
        Z=X*kin.scales[None,:]
        kcand+=Z.T@Z/var
    Pv0=_inv(kkin);Pv1=_inv(kkin+kcand)
    utility_v,dv,av=_block_gain(Pv0,Pv1,[6,7,8,9],dcfg.A_weight)

    # Central angular momentum: H_C=R I_C omega (6 inertia coefficients).
    scales=np.array([55.,55.,55.,25.,25.,25.])
    sigma_h=max(.012,120*icfg.sensor_sigma)
    ki=np.eye(6)/icfg.prior_sigma_normalized**2
    for a,b,w in estimator.angular_pairs:
        A=np.column_stack([b.R@E@b.w-a.R@E@a.w for E in BASIS])
        Z=A*scales[None,:]/sigma_h
        ki+=Z.T@Z
    kadd=np.zeros((6,6))
    previous=pre
    for s in predicted:
        A=np.column_stack([s.R@E@s.w-previous.R@E@previous.w for E in BASIS])
        Z=A*scales[None,:]/sigma_h
        kadd+=Z.T@Z
        previous=s
    Pi0=_inv(ki);Pi1=_inv(ki+kadd)
    utility_i,di,ai=_block_gain(Pi0,Pi1,list(range(6)),dcfg.A_weight)
    out={'kinematic_D':dv,'kinematic_A':av,'inertia_D':di,'inertia_A':ai}
    return float(utility_v+utility_i),out


def optimize_hybrid_one(surface,pre,estimator,dcfg,icfg,seed=0):
    n=int(np.ceil(np.log2(dcfg.candidate_count)))
    sample=qmc.Sobol(d=6,scramble=True,seed=seed).random_base2(n)[:dcfg.candidate_count]
    best=None;best_action=None;best_gain=-np.inf;best_parts=None
    for u in sample:
        a=(2*u-1).astype(np.float32)
        try:
            p=action_to_nudge(a,surface,dcfg)
            score,parts=predicted_hybrid_gain(estimator,pre,p,icfg,dcfg)
            if not np.isfinite(score):continue
        except (ValueError,np.linalg.LinAlgError,OverflowError,FloatingPointError,RuntimeError):
            continue
        if score>best_gain:
            best_gain=score;best=p;best_action=a;best_parts=parts
    if best is None:raise RuntimeError('No valid predicted hybrid-information design')
    return best_action,best,float(best_gain),best_parts
