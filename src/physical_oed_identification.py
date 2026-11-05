"""Ten-parameter spatial-inertia identification and sequential E-optimal OED.

No target dynamics or inertia truth is accessed by any identification routine.
A world-origin spatial momentum balance is used for actual contact impulses.
The design matrix is an *approximate* local-information surrogate, not a
statistically exact Fisher matrix when pre/post observations share errors.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

SCALE = np.array([600.,100.,100.,100.,80.,80.,80.,40.,40.,40.])
BASIS = np.array([
 [[1,0,0],[0,0,0],[0,0,0]],
 [[0,0,0],[0,1,0],[0,0,0]],
 [[0,0,0],[0,0,0],[0,0,1]],
 [[0,1,0],[1,0,0],[0,0,0]],
 [[0,0,1],[0,0,0],[1,0,0]],
 [[0,0,0],[0,0,1],[0,1,0]],
], dtype=float)


def skew(v):
    x,y,z = np.asarray(v,float)
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def normalize(v):
    v=np.asarray(v,float)
    n=np.linalg.norm(v)
    if n < 1e-12: raise ValueError('Zero direction')
    return v/n


def matrix_inertia(pi):
    return np.einsum('i,ijk->jk', np.asarray(pi,float)[4:], BASIS)


def com_and_ic(pi):
    pi=np.asarray(pi,float)
    mass=pi[0]
    if mass <= 0: raise ValueError('Non-positive mass')
    c=pi[1:4]/mass
    Io=matrix_inertia(pi)
    Ic=Io - mass*((c@c)*np.eye(3)-np.outer(c,c))
    if np.linalg.eigvalsh(Ic)[0] <= 1e-7:
        raise ValueError('Invalid center-of-mass inertia')
    return c,Ic


@dataclass
class OriginState:
    """Observed body-origin pose and velocity; never a CoM oracle."""
    r: np.ndarray
    v: np.ndarray
    R: np.ndarray
    omega_body: np.ndarray
    t: float = 0.


def momentum_regressor(s):
    """Return B(s) for [P;H_world_origin] = B(s) @ pi.

    pi=[m,h_B=m*c_B,I_O(xx,yy,zz,xy,xz,yz)], where I_O is about the
    ASTEROID BODY-FRAME ORIGIN, not about CoM.
    """
    R=np.asarray(s.R,float).reshape(3,3)
    wB=np.asarray(s.omega_body,float)
    wW=R@wB
    pA=np.zeros((3,10))
    pA[:,0]=s.v
    pA[:,1:4]=skew(wW)@R
    hA=skew(s.r)@pA
    hA[:,1:4] += -skew(s.v)@R
    for i,b in enumerate(BASIS): hA[:,4+i] = R@b@wB
    return np.vstack([pA,hA])


def wrench_world_origin(measurement):
    """Measured contact impulse; convert accumulator's fixed moment origin to world origin."""
    J=np.asarray(measurement.linear_impulse_world_Ns,float)
    L=np.asarray(measurement.moment_impulse_about_origin_world_Nms,float)
    o=np.asarray(measurement.moment_origin_world_m,float)
    return np.r_[J,L + np.cross(o,J)]


def segment_rows(pre,post_states,measured_wrench):
    """One forced transition then successive free-flight conservation rows.

    Unlike pairing one pre-state with every post-state and reusing the same
    impulse, this does not insert the impulse as independent data N times.
    Measurement noise is still correlated through adjacent state differences.
    """
    if not post_states: raise ValueError('At least one post-contact sample required')
    A=[]; b=[]
    previous=pre
    for i,post in enumerate(post_states):
        A.append(momentum_regressor(post)-momentum_regressor(previous))
        b.append(np.asarray(measured_wrench,float) if i==0 else np.zeros(6))
        previous=post
    return np.vstack(A),np.concatenate(b)


def svd_observability(A, rel_tol=1e-7):
    """SVD of dimensionless regression A diag(SCALE), plus exact rank projectors.

    The tolerance defines numerical observability, not statistical confidence.
    """
    mat=np.asarray(A,float)@np.diag(SCALE)
    U,s,Vt=np.linalg.svd(mat,full_matrices=True)
    threshold=max(rel_tol*s[0],1e-12) if len(s) else 1e-12
    rank=int(np.sum(s>threshold))
    V=Vt.T
    obs=V[:,:rank]@V[:,:rank].T
    unobs=V[:,rank:]@V[:,rank:].T
    return {'singular_values':s,'rank':rank,'P_observable':obs,
            'P_unobservable':unobs,'Vt':Vt,'condition':(
                float(s[0]/s[rank-1]) if rank else float('inf'))}


@dataclass
class EstimatorConfig:
    mass_prior_kg:float=600.
    design_momentum_sigma_Ns:float=0.08
    design_angular_sigma_Nms:float=0.025
    prior_sigma_normalized:float=1.0
    rank_tolerance:float=1e-7


def whitening(cfg):
    return np.diag([1/cfg.design_momentum_sigma_Ns]*3+
                   [1/cfg.design_angular_sigma_Nms]*3)


class SpatialIdentifier:
    """Joint constrained-prior linear estimate, singular-vector diagnostics.

    Sigma parameters are a *design/model-mismatch* weighting, NOT injected
    sensor standard deviations. Zero sensor noise is supported, without
    pretending that a zero-noise Fisher matrix is finite.
    """
    def __init__(self,cfg=None):
        self.cfg=cfg or EstimatorConfig()
        self.prior=np.array([self.cfg.mass_prior_kg,0,0,0,65,68,75,0,0,0.],float)
        self.pi=self.prior.copy()
        self.A=[]; self.b=[]
        self.P=np.diag((np.ones(10)*self.cfg.prior_sigma_normalized)**2)
        self.history=[]

    def add(self,pre,post,wrench):
        A,b=segment_rows(pre,post,wrench)
        self.A.append(A);self.b.append(b)
        M=self.regression()
        y=np.concatenate(self.b)
        S=np.diag(SCALE)
        W=whitening(self.cfg)
        Wbig=np.kron(np.eye(M.shape[0]//6),W)
        X=Wbig@M@S
        yW=Wbig@(y-M@self.prior)
        prior_info=np.eye(10)/(self.cfg.prior_sigma_normalized**2)
        precision=X.T@X+prior_info
        d=np.linalg.solve(precision,X.T@yW)
        candidate=self.prior+S@d
        # Fail closed on nonphysical models instead of silently accepting them.
        # A conservative line search retains the last physically valid estimate.
        previous=self.pi.copy()
        for alpha in (1.,.5,.25,.125,.0625,.0):
            p=previous+alpha*(candidate-previous)
            try:
                com_and_ic(p)
                self.pi=p
                break
            except ValueError: continue
        self.P=np.linalg.inv(precision)
        diag=svd_observability(X,rel_tol=self.cfg.rank_tolerance)
        self.history.append({'pi':self.pi.copy(),'rank':diag['rank'],
                             'singular_values':diag['singular_values'].copy(),
                             'condition':diag['condition'],
                             'P_unobservable':diag['P_unobservable'].copy()})
        return self.history[-1]

    def regression(self):
        return np.vstack(self.A) if self.A else np.zeros((0,10))

    def metrics(self):
        X=self.regression()@np.diag(SCALE)
        return svd_observability(X,self.cfg.rank_tolerance)


def predict_post(pre,pi,point_body,push_body,impulse_Ns,times):
    """First-order instantaneous contact impulse; then torque-free rigid rotation.

    Candidate scoring only. Physical simulation *never* applies this impulse;
    the actual event must come from MuJoCo's probe-pad/asteroid contacts.
    """
    c,Ic=com_and_ic(pi)
    R=np.asarray(pre.R,float).copy()
    w=np.asarray(pre.omega_body,float).copy()
    point=np.asarray(point_body,float)
    j=normalize(push_body)*float(impulse_Ns)
    dw=np.linalg.solve(Ic,np.cross(point-c,j))
    w=w+dw
    vC0=pre.v+R@np.cross(pre.omega_body,c)
    vC1=vC0+R@j/pi[0]
    rC0=pre.r+R@c
    t_prev=0.
    out=[]
    for t in times:
        delta=float(t)-t_prev
        n=max(1,int(np.ceil(delta/0.25)))
        h=delta/n
        for _ in range(n):
            def rate(x):return -np.linalg.solve(Ic,np.cross(x,Ic@x))
            a=rate(w); b=rate(w+h*a/2); d=rate(w+h*b/2); e=rate(w+h*d)
            wn=w+h*(a+2*b+2*d+e)/6
            wmid=.5*(w+wn)
            R=R@Rotation.from_rotvec(wmid*h).as_matrix()
            w=wn
        rC=rC0+vC1*float(t)
        out.append(OriginState(r=rC-R@c,
                    v=vC1-R@np.cross(w,c),R=R.copy(),
                    omega_body=w.copy(),t=pre.t+float(t)))
        t_prev=float(t)
    return out


def candidate_information(pre,pi,point,push,J,T,ident,period=2.):
    times=np.arange(period,T+1e-8,period)
    if len(times)==0: times=np.array([float(T)])
    states=predict_post(pre,pi,point,push,J,times)
    # ONLY the regressor is used for candidate-information scoring. No truth.
    A,_=segment_rows(pre,states,np.zeros(6))
    S=np.diag(SCALE)
    X=np.kron(np.eye(len(states)),whitening(ident.cfg))@A@S
    F=X.T@X
    return F


def e_opt_score(prior_precision,F,projector_unobs=None,d_weight=0.03):
    """E-opt worst eigenvalue improvement, plus small D-opt tie-break.

    Weak directions are already represented by lambda_min. Projection
    term additionally rewards excitation in observed regression nullspace.
    """
    A=np.asarray(prior_precision,float)
    eig0=np.linalg.eigvalsh(A)
    eig1=np.linalg.eigvalsh(A+F)
    e_gain=float(eig1[0]-eig0[0])
    if projector_unobs is None: projector_unobs=np.eye(A.shape[0])
    project_gain=float(np.trace(projector_unobs@F@projector_unobs))/10
    ld0=np.linalg.slogdet(A)[1]
    ld1=np.linalg.slogdet(A+F)[1]
    return e_gain + .01*np.log1p(project_gain) + d_weight*(ld1-ld0), {
        'E_gain':e_gain,'subspace_excitation':project_gain,'D_gain':float(ld1-ld0)}
