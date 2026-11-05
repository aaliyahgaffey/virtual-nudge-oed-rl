"""Information-only, CBF-free Gymnasium environment for three virtual nudges.

An ideal-impulse/torque-free rigid-body SURROGATE is used; there is no spacecraft
CBF, contact mechanics, or asteroid motion constraint. Ten parameters are
refreshed every observation sample, including live preview intervals.
True parameters are withheld from policy and design objective.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError('Install gymnasium: python -m pip install gymnasium') from exc
from src.asteroid_surface import Surface
from src.translation_estimation_base import HybridEstimator
from src.nudge_optimization import predicted_hybrid_gain
from src.inertial_dynamics import (
    SCALE, REFERENCE_PI, IDConfig, DesignConfig, State, SequentialEstimator,
    com_inertia, inertia, sample_state, propagate, action_to_nudge,
    designed_information, gain_objective, error_metrics, BASIS,
)


@dataclass
class EnvConfig:
    backend:str='surrogate' # only backend in v10 (no CBF / MuJoCo)
    randomized_truth:bool=True
    mass_std_frac:float=.03
    com_std_m:float=.002
    central_inertia_std_frac:float=.03
    surrogate_sample_s:float=1.0
    debug_truth:bool=False
    record_trace:bool=False  # off for fast RL training; enable for per-sample plots
    fail_reward:float=-10.

    def __post_init__(self):
        if self.backend!='surrogate': raise ValueError('v10 is deliberately surrogate-only; no spacecraft/asteroid CBF')
        if self.surrogate_sample_s<=0: raise ValueError('Sample period must be positive')


def random_truth(rng,rc:EnvConfig):
    """Hidden, physically consistent body inertia via randomized central inertia."""
    if not rc.randomized_truth: return REFERENCE_PI.copy()
    c0,Ic0=com_inertia(REFERENCE_PI)
    mass=REFERENCE_PI[0]*np.exp(rng.normal(0,rc.mass_std_frac))
    c=c0+rng.normal(0,rc.com_std_m,3)
    # Multiplying Ic coherently by a scalar keeps its physical inertia inequalities.
    Ic=Ic0*np.exp(rng.normal(0,rc.central_inertia_std_frac))
    Io=Ic+mass*((c@c)*np.eye(3)-np.outer(c,c))
    out=np.r_[mass,mass*c,Io[0,0],Io[1,1],Io[2,2],Io[0,1],Io[0,2],Io[1,2]]
    com_inertia(out)
    return out


class ThreeNudgeEnv(gym.Env):
    metadata={'render_modes':['ansi'],'render_fps':1}
    def __init__(self,project_root=None, id_config=None,design_config=None,
                 env_config=None,render_mode=None):
        super().__init__()
        self.root=Path(project_root or Path.cwd()).resolve()
        self.icfg=id_config or IDConfig()
        self.dcfg=design_config or DesignConfig()
        self.ecfg=env_config or EnvConfig()
        self.render_mode=render_mode
        if render_mode not in (None,'ansi'): raise ValueError('Only ansi render supported')
        self.surface=Surface(self.root/'model'/'asteroid.obj')
        self.action_space=spaces.Box(-1.,1.,shape=(6,),dtype=np.float32)
        # 10 normalized estimate + 10 log variances + 45 correlations +
        # r[3],v[3],R[9],omega[3],step[1] = 84 entries.
        self.observation_space=spaces.Box(-1.,1.,shape=(84,),dtype=np.float32)
        self.rng=np.random.default_rng(0)
        self.est=None; self.truth_pi=None; self.truth_state=None
        self.ex=None; self.n=0; self.finished=True
        self.initial_com=None
        self.estimation_trace=[]

    def _observation(self):
        e=self.est; s=e.last_state
        var=np.clip(np.diag(e.cov),1e-15,1e15)
        corr=e.cov/np.sqrt(np.outer(var,var))
        iu=np.triu_indices(10,1)
        o=np.r_[np.tanh((e.pi-self.icfg.prior)/SCALE),
                np.tanh(np.log(var)/12.),
                np.clip(corr[iu],-1,1),
                np.tanh(s.r/4.),np.tanh(s.v/.1),
                s.R.reshape(9),np.tanh(s.w/.1),self.n/3.]
        assert len(o)==84
        return np.clip(o,-1,1).astype(np.float32)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        self.rng=self.np_random
        self.est=HybridEstimator(self.icfg)
        self.n=0; self.finished=False; self.ex=None
        self.truth_pi=None; self.truth_state=None
        self.estimation_trace=[]
        self.truth_pi=random_truth(self.rng,self.ecfg)
        R=Rotation.from_rotvec(self.rng.normal(0,.1,3)).as_matrix()
        self.truth_state=State(np.array([3.,0.,0.])+self.rng.normal(0,.02,3),
                               self.rng.normal(0,.002,3), R,
                               self.rng.normal(0,.035,3),0.)
        self.est.start(sample_state(self.truth_state,self.rng,self.icfg.sensor_sigma))
        return self._observation(), {'backend':self.ecfg.backend,'nudge':0}

    def _trace_update(self, sample_time):
        """Optional per-measurement record, NEVER supplied to the RL observation."""
        if self.ecfg.record_trace:
            self.estimation_trace.append(np.r_[float(sample_time),
                float(self.n+1),self.est.pi.copy(),
                np.sqrt(np.maximum(np.diag(self.est.cov),0))].tolist())

    def save_trace_csv(self,path):
        """Save continuous estimates for plots; enable EnvConfig(record_trace=True)."""
        from pathlib import Path
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        header='time_s,nudge, ' + ','.join(f'est_{i}' for i in range(10)) + ',' + ','.join(f'norm_std_{i}' for i in range(10))
        if not self.estimation_trace: raise ValueError('No recorded samples: enable record_trace=True')
        np.savetxt(path,np.asarray(self.estimation_trace),delimiter=',',header=header.replace(' ',''),comments='')

    def _surrogate_step(self,nudge):
        start=self.truth_state
        horizon=nudge.pulse_s+nudge.observe_s
        times=np.arange(self.ecfg.surrogate_sample_s,horizon,self.ecfg.surrogate_sample_s)
        times=np.r_[times,horizon]
        times=np.unique(np.round(times,12))
        # Instantaneous surrogate impulse at the pre-nudge state timestamp.
        # Every subsequent sampled state updates the continuous online estimator.
        states=propagate(start,self.truth_pi,nudge.point,nudge.push,nudge.impulse,times)
        j_world=start.R@(nudge.impulse*nudge.push)
        p_world=start.r+start.R@nudge.point
        impulse=np.r_[j_world,np.cross(p_world,j_world)]
        impulse[:3]+=self.rng.normal(0,self.icfg.impulse_sigma_Ns,3)
        impulse[3:]+=self.rng.normal(0,self.icfg.moment_sigma_Nms,3)
        self.est.add_impulse(start.t,impulse[:3])
        for i,s in enumerate(states):
            self.est.update(sample_state(s,self.rng,self.icfg.sensor_sigma),
                            impulse if i==0 else np.zeros(6))
            self._trace_update(s.t)
        self.truth_state=states[-1]
        return {'surrogate_samples':len(states)}

    def step(self,action):
        if self.finished: raise RuntimeError('Episode complete; call reset()')
        info={'nudge':self.n+1,'backend':self.ecfg.backend}
        try:
            nudge=action_to_nudge(action,self.surface,self.dcfg)
            # Evaluate objective PRIOR to receiving new measurements.
            objective,parts=predicted_hybrid_gain(self.est,self.est.last_state,nudge,self.icfg,self.dcfg)
            # New independently sampled pre-event state prevents accidentally
            # omitting the impulse when the last segment committed at a boundary.
            pre_sample=sample_state(self.truth_state,self.rng,self.icfg.sensor_sigma)
            self.est.begin_nudge(pre_sample)
            diagnostics=self._surrogate_step(nudge)
            self.n+=1; self.finished=self.n>=3
            info.update(diagnostics)
            info.update({'reward_terms':parts,'expected_objective_raw':float(objective),
                         'impulse_Ns':float(nudge.impulse),'force_N':float(nudge.force),
                         'pulse_s':float(nudge.pulse_s),
                         'point_body_m':nudge.point.tolist(),
                         'push_body':nudge.push.tolist(),
                         'pi_est':self.est.pi.tolist(),
                         'numeric_rank':self.est.rank,
                         'estimator_committed_pairs':self.est.events,
                         'estimator_updates':self.est.updates,
                         'continuous_inertia_previews':self.est.live_inertia_updates,
                         'kinematic_measurement_updates':self.est.kin.updates})
            if self.ecfg.debug_truth:
                truth=self.truth_pi
                info['debug_truth_errors']=error_metrics(self.est.pi,truth)
            return self._observation(), float(.15*objective),self.finished,False,info
        except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError,OverflowError) as exc:
            self.finished=True
            info.update({'failure_type':type(exc).__name__,'failure_reason':str(exc)})
            return self._observation(),float(self.ecfg.fail_reward),True,False,info

    def render(self):
        return f'3-nudge env: {self.n}/3, mass_est={self.est.pi[0]:.4f} kg'

    def close(self):
        self.ex=None
