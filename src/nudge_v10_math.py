"""Ten-parameter sequential active identification: a self-contained mathematical core.

Physical model: free rigid body, ideal instantaneous impulse and subsequent torque-free
flight. The *real* MuJoCo backend uses finite pulses and is required for validation.

All regression is about the body-frame mesh ORIGIN (not CoM). The state is the
body-origin position/velocity, rotation body->world, angular rate in body axes.
The online estimator uses endpoint-disjoint non-overlapping segments with a
first-order errors-in-variables covariance. Consequently P is a *local Gaussian
approximation*, not a certified posterior covariance. No ground-truth parameters
are passed to the estimator or information optimizer.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import qmc
from scipy.integrate import solve_ivp

SCALE = np.array([500., 10., 10., 10., 55., 55., 55., 25., 25., 25.])
BASIS = np.array([[[1,0,0],[0,0,0],[0,0,0]], [[0,0,0],[0,1,0],[0,0,0]],
                  [[0,0,0],[0,0,0],[0,0,1]], [[0,1,0],[1,0,0],[0,0,0]],
                  [[0,0,1],[0,0,0],[1,0,0]], [[0,0,0],[0,0,1],[0,1,0]]],float)
REFERENCE_PI = np.array([500.,1.3995798628882496,-.8210759535158445,1.8001423595381043,
                         46.91371378602807,46.54148123112265,38.8822428447809,
                         -1.955842721178828,.4537741857097323,1.5778515349622746])
PRIOR_PI = np.array([600.,0.,0.,0.,65.,68.,75.,0.,0.,0.])


def skew(v):
    x,y,z=np.asarray(v,float)
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def unit(v):
    v=np.asarray(v,float); norm=np.linalg.norm(v)
    if norm<1e-12 or not np.isfinite(norm): raise ValueError('Invalid direction')
    return v/norm


def inertia(pi):
    return np.einsum('i,ijk->jk',np.asarray(pi)[4:],BASIS)


def com_inertia(pi):
    pi=np.asarray(pi,float)
    if pi[0]<=0 or not np.isfinite(pi).all(): raise ValueError('Mass/parameters invalid')
    c=pi[1:4]/pi[0]
    Ic=inertia(pi)-pi[0]*((c@c)*np.eye(3)-np.outer(c,c))
    if np.linalg.eigvalsh(Ic)[0]<=1e-7: raise ValueError('Not a physical inertia')
    # Triangle inequality is a necessary condition on central moments of inertia.
    ev=np.linalg.eigvalsh(Ic)
    if ev[2]>ev[0]+ev[1]+1e-7: raise ValueError('Central inertia violates triangle inequality')
    return c,Ic


@dataclass
class State:
    r: np.ndarray
    v: np.ndarray
    R: np.ndarray
    w: np.ndarray
    t: float=0.


def momentum_regressor(s:State):
    R=np.asarray(s.R,float); wW=R@s.w
    Bp=np.zeros((3,10)); Bp[:,0]=s.v; Bp[:,1:4]=skew(wW)@R
    Bh=skew(s.r)@Bp
    Bh[:,1:4]+=-skew(s.v)@R
    for j,I in enumerate(BASIS): Bh[:,4+j]=R@I@s.w
    return np.vstack([Bp,Bh])


def momentum(s:State,pi):
    return momentum_regressor(s)@np.asarray(pi)


def state_momentum_jacobian(s:State,pi):
    """Analytic 6x12 d(momentum)/d[r, v, world-attitude-vector, body-omega].

    Perturb the orientation by R' = exp([dtheta]x) R, with fixed world r/v.
    """
    R=s.R; m=pi[0]; q=R@pi[1:4]; wW=R@s.w; k=R@inertia(pi)@s.w
    spin=np.cross(wW,q); p=m*s.v+spin
    G=np.zeros((6,12))
    G[3:6,0:3]=-skew(p)
    G[0:3,3:6]=m*np.eye(3)
    G[3:6,3:6]=m*skew(s.r)+skew(q)
    G[0:3,6:9]=-skew(spin)
    G[3:6,6:9]=skew(s.r)@G[0:3,6:9]+skew(s.v)@skew(q)-skew(k)
    G[0:3,9:12]=-skew(q)@R
    G[3:6,9:12]=skew(s.r)@G[0:3,9:12]+R@inertia(pi)
    return G


@dataclass
class IDConfig:
    prior:np.ndarray=field(default_factory=lambda:PRIOR_PI.copy())
    prior_sigma_normalized:float=3.0
    sensor_sigma:float=1e-4
    # Small nonzero numerical/wrench calibration uncertainty, even with ideal sensors.
    impulse_sigma_Ns:float|None=None
    moment_sigma_Nms:float|None=None
    segment_s:float=2.0
    endpoint_gap_s:float=.05
    # Lower noise floor in the local Gaussian approximation (NOT artificial data).
    min_residual_sigma:float=1e-5
    # Broad geometry/mission-scale parameter bounds, not asteroid motion limits.
    # Keep nonlinear design predictions out of the nearly singular inertia regime.
    min_mass_kg:float=100.
    max_mass_kg:float=2000.
    min_central_moment_kgm2:float=5.
    max_central_moment_kgm2:float=1000.

    def __post_init__(self):
        if self.impulse_sigma_Ns is None: self.impulse_sigma_Ns=self.sensor_sigma
        if self.moment_sigma_Nms is None: self.moment_sigma_Nms=self.sensor_sigma
        if self.prior_sigma_normalized<=0 or self.segment_s<=0 or self.endpoint_gap_s<0:
            raise ValueError('Invalid prior/observation settings')
        if min(self.sensor_sigma,self.impulse_sigma_Ns,self.moment_sigma_Nms,self.min_residual_sigma)<0:
            raise ValueError('Invalid noise settings')
        com_inertia(self.prior)


def sensor_cov(cfg:IDConfig):
    return np.diag([cfg.sensor_sigma**2]*12)


def segment_covariance(a:State,b:State,pi,cfg:IDConfig,force_event=False):
    """First-order moment residual covariance, conditional on current pi estimate.

    Endpoints assumed independent: used segments have no shared endpoints.
    Regressor errors are handled to first order; nonlinear terms remain.
    """
    C=sensor_cov(cfg)
    Ga=state_momentum_jacobian(a,pi); Gb=state_momentum_jacobian(b,pi)
    V=Ga@C@Ga.T+Gb@C@Gb.T
    if force_event:
        V+=np.diag([cfg.impulse_sigma_Ns**2]*3+[cfg.moment_sigma_Nms**2]*3)
    return .5*(V+V.T)+cfg.min_residual_sigma**2*np.eye(6)


def normal_equations(a:State,b:State,wrench,pi_for_cov,cfg:IDConfig,force_event=False):
    A=momentum_regressor(b)-momentum_regressor(a)
    V=segment_covariance(a,b,pi_for_cov,cfg,force_event)
    # Make all 10 parameter dimensions comparable.
    S=A*SCALE[np.newaxis,:]
    C,lower=cho_factor(V,lower=True,check_finite=True)
    X=cho_solve((C,lower),S)
    Y=cho_solve((C,lower),np.asarray(wrench)-A@cfg.prior)
    return S.T@X, S.T@Y


class SequentialEstimator:
    """Each accepted independent endpoint-pair is committed once.

    During a pending segment, `update` refreshes the *preview estimate* at every
    measurement timestamp, but the preview never gets repeatedly accumulated.
    A short post-commit gap avoids sharing state measurements across pairs.
    """
    def __init__(self,cfg:IDConfig):
        self.cfg=cfg
        self.prior_precision=np.eye(10)/cfg.prior_sigma_normalized**2
        self.committed_F=np.zeros((10,10)); self.committed_q=np.zeros(10)
        self.precision=self.prior_precision.copy()
        self.cov=np.linalg.inv(self.precision)
        self.pi=cfg.prior.copy()
        self.last_state=None; self.anchor=None; self.last_finish=-np.inf
        self.cumulative_wrench=np.zeros(6); self.anchor_wrench=np.zeros(6)
        self.anchor_had_force=False; self.events=0; self.updates=0
        self.last_residual=float('nan')

    def start(self,state:State):
        if self.last_state is not None: raise RuntimeError('Already started')
        self.last_state=state; self.anchor=state
        self.last_finish=float(state.t)-self.cfg.endpoint_gap_s
        self.anchor_wrench=self.cumulative_wrench.copy()

    def begin_nudge(self,independent_pre_event_state:State):
        """Anchor the imminent impulse with an independent pre-event sample.

        The event must not be lost when the previous interval just committed.
        Any unfinished previous preview is dropped (it was never committed).
        This sample is independently acquired from the previous endpoint.
        """
        if self.last_state is None: raise RuntimeError('Not started')
        if abs(independent_pre_event_state.t-self.last_state.t)>1e-7:
            raise ValueError('Pre-event sample must have current timestamp')
        self.last_state=independent_pre_event_state
        self.anchor=independent_pre_event_state
        self.anchor_wrench=self.cumulative_wrench.copy()
        self._solve(self.committed_F,self.committed_q)

    def _solve(self,F,q):
        precision=self.prior_precision+.5*(F+F.T)
        cp=cho_factor(precision,lower=True)
        dx=cho_solve(cp,q)
        proposed=self.cfg.prior+SCALE*dx
        # Do not silently accept nonphysical mass/inertia. Convex blending with prior.
        previous=self.pi
        for alpha in (1.,.5,.25,.125,.0625,.0):
            candidate=previous+alpha*(proposed-previous)
            try:
                _,Ic=com_inertia(candidate)
                ev=np.linalg.eigvalsh(Ic)
                if not (self.cfg.min_mass_kg<=candidate[0]<=self.cfg.max_mass_kg and
                        ev[0]>=self.cfg.min_central_moment_kgm2 and
                        ev[-1]<=self.cfg.max_central_moment_kgm2):
                    raise ValueError('Outside broad physical prior support')
            except ValueError: continue
            self.pi=candidate
            break
        self.precision=precision
        self.cov=cho_solve(cp,np.eye(10))

    def update(self,state:State,wrench_increment):
        if self.last_state is None: raise RuntimeError('Call start() before update()')
        if state.t<=self.last_state.t: raise ValueError('Timestamps must increase')
        w=np.asarray(wrench_increment,float)
        if w.shape!=(6,) or not np.isfinite(w).all(): raise ValueError('Invalid wrench')
        self.cumulative_wrench+=w; self.last_state=state
        self.updates+=1
        if self.anchor is None:
            if state.t-self.last_finish>=self.cfg.endpoint_gap_s-1e-8:
                self.anchor=state; self.anchor_wrench=self.cumulative_wrench.copy()
            self._solve(self.committed_F,self.committed_q)
            return self.pi.copy()
        h=float(state.t-self.anchor.t)
        if h<=0: return self.pi.copy()
        dw=self.cumulative_wrench-self.anchor_wrench
        is_forced=bool(np.linalg.norm(dw)>1e-10)
        # Recompute the live pair, but never add it permanently until completion.
        f,q=normal_equations(self.anchor,state,dw,self.pi,self.cfg,force_event=is_forced)
        self._solve(self.committed_F+f,self.committed_q+q)
        self.last_residual=float(np.linalg.norm(momentum(state,self.pi)-momentum(self.anchor,self.pi)-dw))
        if h>=self.cfg.segment_s-1e-10:
            # Important: commit the pair using the SAME covariance as preview.
            self.committed_F+=f; self.committed_q+=q; self.events+=1
            self.anchor=None; self.last_finish=float(state.t)
        return self.pi.copy()

    @property
    def information(self): return self.committed_F.copy()

    @property
    def rank(self):
        sig=np.sqrt(np.maximum(np.linalg.eigvalsh(self.committed_F),0))
        return int(np.sum(sig>max(1e-10,1e-6*np.max(sig))))


def sample_state(true:State,rng:np.random.Generator,sigma:float):
    if sigma==0: return State(true.r.copy(),true.v.copy(),true.R.copy(),true.w.copy(),true.t)
    return State(true.r+rng.normal(0,sigma,3),true.v+rng.normal(0,sigma,3),
                 true.R@Rotation.from_rotvec(rng.normal(0,sigma,3)).as_matrix(),
                 true.w+rng.normal(0,sigma,3),true.t)


def propagate(pre:State,pi,point,push,impulse,times):
    """Instantaneous-impulse first-order predictor, torque-free rigid rotation.

    The applied pulse in the real model is finite-duration and must be tested
    independently. All times measured since the nudge decision, not absolute.
    """
    c,Ic=com_inertia(pi)
    R=pre.R.copy(); omega=pre.w.copy(); j=unit(push)*float(impulse)
    domega=np.linalg.solve(Ic,np.cross(np.asarray(point)-c,j))
    omega=omega+domega
    vC0=pre.v+R@np.cross(pre.w,c)
    vC1=vC0+R@j/pi[0]
    rC0=pre.r+R@c
    times=np.asarray(times,float)
    if not np.all(np.diff(times)>=0) or (len(times) and times[0]<0):
        raise ValueError('Times must be nonnegative sorted')
    if len(times)==0: return []
    # Integrate rotational dynamics and quaternion kinematics at high precision.
    # Low-accuracy midpoint stepping caused artificial momentum drift that
    # contaminated the noiseless identification experiments.
    quat0=Rotation.from_matrix(R).as_quat()  # [x,y,z,w]
    def derivative(_,y):
        q=y[:4]; w=y[4:]
        dq=np.r_[.5*(q[3]*w+np.cross(q[:3],w)), -.5*np.dot(q[:3],w)]
        dw=-np.linalg.solve(Ic,np.cross(w,Ic@w))
        return np.r_[dq,dw]
    y0=np.r_[quat0,omega]
    if float(times[-1])==0:
        samples=np.tile(y0,(len(times),1))
    else:
        sol=solve_ivp(derivative,(0.,float(times[-1])),y0,method='DOP853',
                      rtol=2e-11,atol=1e-12,dense_output=True)
        if not sol.success: raise RuntimeError(f'Rotation integration failed: {sol.message}')
        samples=sol.sol(times).T
    result=[]
    for t,y in zip(times,samples):
        quat=y[:4]/np.linalg.norm(y[:4])
        Rt=Rotation.from_quat(quat).as_matrix()
        wt=y[4:]
        rC=rC0+vC1*float(t)
        result.append(State(rC-Rt@c,vC1-Rt@np.cross(wt,c),Rt,wt,pre.t+float(t)))
    return result


@dataclass(frozen=True)
class Nudge:
    point:np.ndarray
    normal:np.ndarray
    push:np.ndarray
    impulse:float
    force:float
    observe_s:float
    @property
    def pulse_s(self): return self.impulse/self.force


@dataclass
class DesignConfig:
    # Consistent with v8 actuation limits. No asteroid translation/speed bounds.
    min_impulse:float=.2
    max_impulse:float=2.
    min_force:float=1.
    max_force:float=4.5
    max_torque:float=8.
    min_pulse:float=.12
    max_pulse:float=2.
    tangent_fraction_max:float=.2
    observe_s:float=30.
    plan_interval_s:float=2.
    A_weight:float=3.0
    candidate_count:int=64


def action_to_nudge(a,surface,cfg:DesignConfig):
    a=np.asarray(a,float).reshape(6)
    if not np.isfinite(a).all(): raise ValueError('Non-finite action')
    a=np.clip(a,-1,1)
    lon=np.pi*(a[0]+1); z=a[1]
    p,n=surface.point_and_normal(lon,z)
    axis=np.eye(3)[np.argmin(abs(n))]
    tangent=unit(np.cross(n,axis)); tangent2=np.cross(n,tangent)
    beta=cfg.tangent_fraction_max*.5*(a[3]+1)
    push=unit(-np.sqrt(1-beta**2)*n+beta*(np.cos(np.pi*(a[2]+1))*tangent+
                                               np.sin(np.pi*(a[2]+1))*tangent2))
    arm=float(np.linalg.norm(np.cross(p,push)))
    force_max=min(cfg.max_force,cfg.max_torque/max(arm,1e-9))
    Jmin=max(cfg.min_impulse,cfg.min_force*cfg.min_pulse)
    Jmax=min(cfg.max_impulse,force_max*cfg.max_pulse)
    if force_max<cfg.min_force or Jmax<Jmin: raise ValueError('No admissible actuator setting')
    J=Jmin+(Jmax-Jmin)*.5*(a[4]+1)
    fmin=max(cfg.min_force,J/cfg.max_pulse)
    fmax=min(force_max,J/cfg.min_pulse)
    if fmax+1e-12<fmin: raise ValueError('No feasible pulse')
    force=fmin+(fmax-fmin)*.5*(a[5]+1)
    return Nudge(p,n,push,float(J),float(force),cfg.observe_s)


def designed_information(pre:State,pi,nudge:Nudge,idcfg:IDConfig,design:DesignConfig):
    """Local predicted Gaussian information in normalized parameter coordinates.

    No measurements, simulator truth or noisy random regressor enter the reward.
    It is a model-based *expected* information objective, not actual posterior gain.
    """
    step=max(design.plan_interval_s,idcfg.segment_s+idcfg.endpoint_gap_s)
    horizon=nudge.pulse_s+nudge.observe_s
    times=np.arange(step,horizon,step)
    times=np.r_[times,horizon]
    # The first pair spans the commanded instantaneous impulse.
    preds=propagate(pre,pi,nudge.point,nudge.push,nudge.impulse,times)
    prev=pre; F=np.zeros((10,10))
    for j,p in enumerate(preds):
        a=momentum_regressor(p)-momentum_regressor(prev)
        cov=segment_covariance(prev,p,pi,idcfg,force_event=(j==0))
        S=a*SCALE[np.newaxis,:]
        F+=S.T@cho_solve(cho_factor(cov,lower=True),S)
        prev=p
    return .5*(F+F.T)


def gain_objective(precision,F,A_weight=3.):
    """A/D-optimal measured-uncertainty objective, returns gain and parts.

    r=log det(Λ+F)-log det(Λ) + A_weight*(tr(P)-tr(Pnew))/tr(P)
    where P=Λ^-1, Pnew=(Λ+F)^-1. Closed form conditional on F.
    """
    prec=.5*(precision+precision.T); F=.5*(F+F.T)
    vals=np.linalg.eigvalsh(F)
    if vals[0]<-1e-7: raise ValueError('Non-PSD information')
    ev0=np.linalg.slogdet(prec); ev1=np.linalg.slogdet(prec+F)
    if ev0[0]<=0 or ev1[0]<=0: raise ValueError('Non-PD precision')
    P=np.linalg.inv(prec); Pn=np.linalg.inv(prec+F)
    d=ev1[1]-ev0[1]
    a=(np.trace(P)-np.trace(Pn))/max(np.trace(P),1e-15)
    return float(d+A_weight*a), {'D_gain':float(d),'A_reduction':float(a)}


def optimize_one(surface,pre,estimator,dcfg:DesignConfig,icfg:IDConfig,seed=0):
    """Closed-loop greedy OED baseline: optimize once AFTER EACH nudge.

    Its calculation is fully explicit. It does not use hidden inertial truth.
    """
    n=int(np.ceil(np.log2(dcfg.candidate_count)))
    pts=qmc.Sobol(d=6,scramble=True,seed=seed).random_base2(n)[:dcfg.candidate_count]
    best=None; max_score=-np.inf; best_a=None
    for x in pts:
        a=2*x-1
        try:
            plan=action_to_nudge(a,surface,dcfg)
            F=designed_information(pre,estimator.pi,plan,icfg,dcfg)
            score,_=gain_objective(estimator.precision,F,dcfg.A_weight)
        except (ValueError,np.linalg.LinAlgError,FloatingPointError,OverflowError):
            continue
        if np.isfinite(score) and score>max_score:
            best_a=a.astype(np.float32); best=plan; max_score=float(score)
    if best is None: raise RuntimeError('No numerically feasible design candidates')
    return best_a,best,max_score


def error_metrics(estimated,truth):
    e=np.asarray(estimated)-np.asarray(truth)
    return {'mass_error_kg':float(abs(e[0])),
            'com_error_mm':float(1000*np.linalg.norm(
                estimated[1:4]/estimated[0]-truth[1:4]/truth[0])),
            'inertia_error_kgm2':float(np.linalg.norm(inertia(estimated)-inertia(truth),'fro'))}
