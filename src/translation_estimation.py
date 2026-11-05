"""Online linear inverse-mass and COM estimation from the full pose history.

For a free-flying rigid target, with known applied forces in WORLD axes:

  r_origin(t) = rC(0) + vC(0) t - R(t)c_body + (1/m) K(t),
  K(t) = integral_0^t (t-s) F_world(s) ds.

Linear unknowns are [rC0(3),vC0(3),c_body(3),1/m], all identified from
noisy positions and attitudes and measured virtual impulse history. This
avoids taking derivatives of noisy position to estimate the mass. Assumes
no unmodeled external translational forces and a rigid, fixed-mass body.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import cho_solve,cho_factor
from src.inertial_dynamics import State, IDConfig, SequentialEstimator, SCALE, \
    momentum_regressor,normal_equations,com_inertia,inertia,BASIS


KIN_INVERSE_MASS_PRIOR_STD_NORMALIZED = 0.30
# With q scale 0.002 1/kg and m_prior=600 kg, local sigma_m ≈216 kg.
# Must be justified/calibrated to mission precontact size-density estimates.

class TranslationEstimator:
    def __init__(self,cfg:IDConfig,sample_interval_s=.5):
        self.cfg=cfg; self.dt_min=sample_interval_s
        self.t0=None; self.last_record=-float('inf')
        self.impulse=np.zeros(3); self.t_impulse=np.zeros(3)
        self.scales=np.r_[np.ones(3)*5,np.ones(3)*.05,np.ones(3)*.05,.002]
        self.prior=None
        self.F=np.zeros((10,10));self.q=np.zeros(10)
        self.x=None;self.cov=None;self.accepted_samples=0
        self.last_sigma=float('inf')
        self.updates=0

    def start(self,state:State):
        self.t0=float(state.t)
        self.prior=np.r_[state.r,state.v,np.zeros(3),1/600.]
        self.x=self.prior.copy()
        self._update_solution(self.F,self.q)
        self.update(state)

    def add_impulse(self,time_world_s,J_world):
        if self.t0 is None: raise RuntimeError('Start before impulse')
        self.impulse+=np.asarray(J_world,float)
        self.t_impulse+=(float(time_world_s)-self.t0)*np.asarray(J_world,float)

    def _row(self,state):
        t=float(state.t-self.t0)
        K=t*self.impulse-self.t_impulse
        X=np.zeros((3,10))
        X[:,:3]=np.eye(3)
        X[:,3:6]=t*np.eye(3)
        X[:,6:9]=-state.R
        X[:,9]=K
        return X,state.r

    def _info(self,state):
        X,y=self._row(state)
        var=max(self.cfg.sensor_sigma,1e-5)**2 + (
            max(self.cfg.sensor_sigma,1e-5)*np.linalg.norm(self.x[6:9]))**2
        Z=X*self.scales[np.newaxis,:]
        r=y-X@self.prior
        return Z.T@Z/var,Z.T@r/var

    def _update_solution(self,F,q):
        prec=np.diag(1/np.array([10]*3+[2]*3+[2]*3+[KIN_INVERSE_MASS_PRIOR_STD_NORMALIZED])**2)+F
        C=cho_factor(.5*(prec+prec.T),lower=True)
        normalized=cho_solve(C,q)
        self.x=self.prior+self.scales*normalized
        self.cov=(self.scales[:,None]*cho_solve(C,np.eye(10)))*self.scales[None,:]
        self.last_sigma=float(np.sqrt(max(self.cov[9,9],0)))

    def update(self,state):
        if self.t0 is None: raise RuntimeError('Not initialized')
        self.updates+=1
        if state.t-self.last_record>=self.dt_min-1e-8:
            f,q=self._info(state)
            self.F+=f;self.q+=q
            self.last_record=float(state.t)
            self.accepted_samples+=1
            self._update_solution(self.F,self.q)
        else:
            f,q=self._info(state)
            self._update_solution(self.F+f,self.q+q)
        return self.x.copy()

    def reliable(self):
        q=self.x[9]
        return (q>0 and 100<=1/q<=2000 and
                self.last_sigma/max(q,1e-15)<.30 and
                np.sqrt(max(self.cov[6,6],self.cov[7,7],self.cov[8,8]))<.04)

    def mass_com(self):
        q=self.x[9]; return 1/q,self.x[6:9].copy()


class HybridEstimator:
    """Kinematic mass+COM plus momentum-regression six-component inertia.

    The block covariance returned here is a design approximation: translation
    and angular estimates share noisy attitude measurements. This is *not*
    a rigorously independent joint posterior, although each estimate has a
    physics-based and observable measurement equation.
    """
    def __init__(self,cfg:IDConfig):
        self.cfg=cfg
        self.mom=SequentialEstimator(cfg)
        self.kin=TranslationEstimator(cfg)
        self.pi=self.mom.pi.copy()
        self.cov=self.mom.cov.copy()
        self.precision=self.mom.precision.copy()
        self.last_state=None
        self.angular_pairs=[]
        self.completed_pairs=0
        self._previous_completed=0
        self.trace=[]
        self.fit_accepted=False
        self.live_inertia_updates=0

    def start(self,state):
        self.mom.start(state);self.kin.start(state)
        self.last_state=state
        self._combine()

    def add_impulse(self,time,linear_impulse_world):
        self.kin.add_impulse(time,linear_impulse_world)

    def begin_nudge(self,state):
        self.mom.begin_nudge(state)
        self.last_state=state
        self._combine()

    def update(self,state,wrench_increment):
        a=self.mom.anchor
        old_count=self.mom.events
        anchor_wrench=self.mom.anchor_wrench.copy()
        self.mom.update(state,wrench_increment)
        self.kin.update(state)
        if self.mom.events>old_count and a is not None:
            # We need the complete impulse accumulated over this committed pair.
            # Save measured momenta rows for the conditional inertia regression.
            # The measured wrench is retained by the base estimator's accumulator.
            self.angular_pairs.append((a,state,(self.mom.cumulative_wrench-
                                      anchor_wrench).copy()))
        self.last_state=state
        self.completed_pairs=self.mom.events
        # Refresh the rotational fit at EVERY incoming measurement. Its latest
        # unfinished independent pair is included as a live preview, never
        # permanently counted until mom.update commits its interval.
        self.live_inertia_updates+=1
        self._combine()
        self.trace.append({'time_s':float(state.t),'estimated_pi':self.pi.tolist(),
                           'mass_com_reliable':bool(self.kin.reliable()),
                           'committed_momentum_pairs':int(self.mom.events)})
        return self.pi.copy()

    def _live_angular_pairs(self):
        pairs=list(self.angular_pairs)
        anchor=self.mom.anchor
        if (anchor is not None and self.mom.last_state is not None
                and self.mom.last_state.t>anchor.t+1e-10):
            wrench=self.mom.cumulative_wrench-self.mom.anchor_wrench
            pairs.append((anchor,self.mom.last_state,wrench.copy()))
        return pairs

    def _combine(self):
        self.last_candidate=None
        self.fit_accepted=False
        self.pi=self.mom.pi.copy()
        self.cov=self.mom.cov.copy()
        if not self.kin.reliable():
            self.precision=self.mom.precision.copy()
            return
        mass,c=self.kin.mass_com()
        self.pi[:4]=np.r_[mass,mass*c]
        q=self.kin.x[9]
        # Transform kinematic covariance in [c,1/m] into [m,m*c].
        jac=np.zeros((4,4));jac[0,3]=-mass**2
        jac[1:4,:3]=mass*np.eye(3)
        jac[1:4,3]=-mass**2*c
        Pch=self.kin.cov[np.ix_([6,7,8,9],[6,7,8,9])]
        Pmh=jac@Pch@jac.T
        Pmh=(Pmh+Pmh.T)/2
        # Include exactly ONE live angular pair preview (not committed); this
        # lets inertia change after every new observation once the interval is
        # long enough for a useful signal. The preview is replaced, never
        # counted again at the next measurement or after final commit.
        pairs=list(self.angular_pairs)
        if self.mom.anchor is not None and self.mom.last_state is not None:
            h=self.mom.last_state.t-self.mom.anchor.t
            if h>=min(.2,self.cfg.segment_s/2)-1e-10:
                pairs.append((self.mom.anchor,self.mom.last_state,
                    (self.mom.cumulative_wrench-self.mom.anchor_wrench).copy()))
        # Angular momentum ABOUT CoM: its regression has no origin r×P term.
        # This avoids amplifying 1e-4 m/s velocity noise by the ~3-m world
        # position when attempting to infer inertia about the world origin.
        cIc0=com_inertia(self.cfg.prior)[1]
        prior_i=np.array([cIc0[0,0],cIc0[1,1],cIc0[2,2],cIc0[0,1],cIc0[0,2],cIc0[1,2]])
        central_scale=np.array([55.,55.,55.,25.,25.,25.])
        Pii=np.diag((central_scale*self.cfg.prior_sigma_normalized)**2)
        central_i=prior_i.copy()
        current_pairs=self._live_angular_pairs()
        if current_pairs:
            Zlist=[]; ylist=[]
            for a,b,w in current_pairs:
                # L_C = L_O - r_C × J, at the impulse instant (anchor a).
                rC=a.r+a.R@c
                torque_center=w[3:]-np.cross(rC,w[:3])
                Ai=np.column_stack([(b.R@basis@b.w-a.R@basis@a.w)
                                    for basis in BASIS])
                Zlist.append(Ai)
                ylist.append(torque_center)
            A=np.vstack(Zlist); y=np.concatenate(ylist)
            # Body angular-rate noise contributes ~sqrt(2) I sigma_omega
            # to each impulse-momentum difference; floors cover timing and
            # modelling imperfections as well as measured wrench errors.
            sigma=max(.012,120*self.cfg.sensor_sigma)
            X=A*central_scale[None,:]/sigma
            yy=(y-A@prior_i)/sigma
            pp=np.eye(6)/self.cfg.prior_sigma_normalized**2+X.T@X
            try:
                d=np.linalg.solve(pp,X.T@yy)
                proposed=prior_i+central_scale*d
                # Physically project toward the previous valid central inertia.
                for alpha in (1.,.5,.25,.125,.0625,.0):
                    arr=central_i+alpha*(proposed-central_i)
                    Ic=np.einsum('i,ijk->jk',arr,BASIS)
                    Io=Ic+mass*((c@c)*np.eye(3)-np.outer(c,c))
                    candidate=self.pi.copy()
                    candidate[4:]=[Io[0,0],Io[1,1],Io[2,2],Io[0,1],Io[0,2],Io[1,2]]
                    try: com_inertia(candidate)
                    except ValueError: continue
                    self.pi=candidate
                    self.fit_accepted=(alpha>0)
                    self.last_candidate=candidate.copy()
                    break
                Pii=(central_scale[:,None]*np.linalg.inv(pp))*central_scale[None,:]
            except np.linalg.LinAlgError:
                pass
        if self.last_candidate is None:
            Ic=np.einsum('i,ijk->jk',prior_i,BASIS)
            Io=Ic+mass*((c@c)*np.eye(3)-np.outer(c,c))
            self.pi[4:]=[Io[0,0],Io[1,1],Io[2,2],Io[0,1],Io[0,2],Io[1,2]]
        # Approximate independent blocks in normalized coordinates. The
        # transform from central to origin inertia adds uncertainty from m/c;
        # this is omitted (conservative interpretation required in README).
        self.cov=np.zeros((10,10))
        self.cov[:4,:4]=(Pmh/SCALE[:4,None])/SCALE[None,:4]
        self.cov[4:,4:]=(Pii/SCALE[4:,None])/SCALE[None,4:]
        # A small positive floor avoids reporting spurious exact certainty.
        self.cov=.5*(self.cov+self.cov.T)+np.eye(10)*1e-9
        self.precision=np.linalg.inv(self.cov)

    @property
    def events(self):return self.mom.events
    @property
    def updates(self):return self.mom.updates
    @property
    def rank(self):return self.mom.rank
