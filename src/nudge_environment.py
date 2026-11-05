"""v11 Gymnasium environment: information-only, 3 impulses, sample-wise estimation.

Inherits the v10 rigid-body surrogate and 84-observation SAC interface.
Replaces its estimator with a corrected-central-inertia-prior version and
its reward with an interpretable weak-mode-aware covariance objective.
Planning and rewards NEVER use simulator ground truth. The simulator is not
MuJoCo and does not enforce a spacecraft CBF or asteroid keep-in zone.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from src.nudge_environment_base import ThreeNudgeEnv
from src.inertial_dynamics import action_to_nudge, propagate, sample_state
from src.translation_estimation import HybridEstimator
from src.asteroid_surface_v11 import Surface
from src.information_gain import InformationConfig, predicted_gain, current_cov, weak_modes, gaussian_gain,component_summary


class InformationNudgeEnv(ThreeNudgeEnv):
    def __init__(self,*args,info_config=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.surface=Surface(self.root/'model'/'asteroid.obj')
        self.info_config=info_config or InformationConfig()
        self.info_log=[]

    def reset(self,*,seed=None,options=None):
        # Parent chooses randomized truth and state and uses sensor RNG;
        # then reuse its INITIAL noisy state in the corrected estimator.
        obs,info=super().reset(seed=seed,options=options)
        initial=self.est.last_state
        self.est=HybridEstimator(self.icfg)
        self.est.start(initial)
        self.info_log=[]
        self._episode_seed=int(seed if seed is not None else 0)
        # Separate truth, action and sensor RNG streams for paired comparisons.
        self._action_rng=np.random.default_rng(np.random.SeedSequence([self._episode_seed,3]))
        return self._observation(),info

    def step(self,action):
        if self.finished:raise RuntimeError('Episode complete')
        try:
            plan=action_to_nudge(action,self.surface,self.dcfg)
            P0=current_cov(self.est)
            predicted_utility,pred_terms,_,Ppred=predicted_gain(
                self.est,self.est.last_state,plan,self.dcfg,self.icfg,self.info_config)
        except (ValueError,RuntimeError,np.linalg.LinAlgError,OverflowError,FloatingPointError) as exc:
            self.finished=True
            return self._observation(),float(self.ecfg.fail_reward),True,False,{
                'nudge':self.n+1,'failure_type':type(exc).__name__,
                'failure_reason':str(exc),'backend':self.ecfg.backend}
        before_modes=weak_modes(P0,2)
        # One-step-ahead prediction is frozen *before* receiving the response.
        # It is not fed into the estimator or action selection.
        initial_measured=self.est.last_state
        horizon=plan.pulse_s+plan.observe_s
        try:
            forecast=propagate(initial_measured,self.est.pi,plan.point,plan.push,plan.impulse,[horizon])[-1]
        except (ValueError,np.linalg.LinAlgError,OverflowError):forecast=None
        # Fresh per-nudge sensor stream: identical seeded noise draws are available
        # to every policy, regardless of how it sampled candidate actions.
        self.rng=np.random.default_rng(np.random.SeedSequence([self._episode_seed,100,int(self.n)]))
        obs,old_reward,terminated,truncated,info=super().step(action)
        if info.get('failure_reason'):
            info['predicted_utility']=float(predicted_utility)
            return obs,self.ecfg.fail_reward,terminated,truncated,info
        P1=current_cov(self.est)
        observed_utility,observed_terms=gaussian_gain(P0,P1,
            self.info_config.A_weight,self.info_config.weak_weight)
        info['v11_predicted_eig_nats']=pred_terms['gaussian_EIG_nats']
        info['v11_passive_eig_nats']=pred_terms['passive_EIG_nats']
        info['v11_incremental_eig_nats']=pred_terms['incremental_EIG_over_passive_nats']
        info['v11_predicted_utility']=float(predicted_utility)
        info['v11_observed_covariance_eig_proxy_nats']=observed_terms['gaussian_EIG_nats']
        info['v11_observed_covariance_utility_proxy']=float(observed_utility)
        info['v11_observed_vs_predicted_eig_nats']=float(observed_terms['gaussian_EIG_nats']-pred_terms['gaussian_EIG_nats'])
        info['v11_predicted_components']=pred_terms
        info['v11_observed_components']=observed_terms
        info['v11_weak_modes_before']=before_modes
        info['v11_weak_modes_after']=weak_modes(P1,2)
        info['v11_covariance_norm_after']=P1.tolist()
        # Prequential trajectory prediction error: truth is never read here.
        if forecast is not None:
            measured=self.est.last_state
            info['prequential_prediction_residual']={
                'position_m':float(np.linalg.norm(forecast.r-measured.r)),
                'velocity_mps':float(np.linalg.norm(forecast.v-measured.v)),
                'omega_radps':float(np.linalg.norm(forecast.w-measured.w)),
                'orientation_rad':float(Rotation.from_matrix(forecast.R.T@measured.R).magnitude())}
        else:
            info['prequential_prediction_residual']=None
        if self.ecfg.debug_truth:
            info['v11_DEV_ONLY_coverage']=component_summary(self.est.pi,self.truth_pi,P1)
        self.info_log.append({'nudge':self.n,'time_s':float(self.est.last_state.t),
            'predicted_eig_nats':pred_terms['gaussian_EIG_nats'],
            'incremental_eig_over_passive_nats':pred_terms['incremental_EIG_over_passive_nats'],
            'observed_covariance_proxy_nats':observed_terms['gaussian_EIG_nats'],
            'predicted_utility':float(predicted_utility),
            'weakest_mode':before_modes[0]['dominant']})
        # Positive reward proportional to expected information, not realized truth error.
        return obs,float(.15*predicted_utility),terminated,truncated,info
