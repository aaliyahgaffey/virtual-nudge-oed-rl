"""Gymnasium RL environment for choosing three *virtual* asteroid nudges.

This is an ADDITIVE module for the existing active_inertial_id repository (v8).
The continuous ten-parameter estimator and asteroid geometry are imported from
its existing `src` modules. The policy selects a surface point, push direction,
impulse, and force, once per nudge; the estimator is updated *between* decisions.

Backends
--------
"surrogate" : inexpensive approximate impulsive rigid-body simulator built on
              `predict_post`, suitable for initial RL training. It is NOT
              MuJoCo and its pulses are instantaneous approximations. States
              are sampled at estimator intervals, not every 0.002s.
"mujoco"   : executes the actual virtual force in the user's MuJoCo experiment,
              steps the full spacecraft station-hold / CBF implementation,
              and updates the estimator EVERY physics timestep. Slow; best
              for evaluation / fine-tuning after surrogate training.

The policy NEVER receives true asteroid inertia, CoM, mass or estimation error;
reward uses changes in measured regression information ONLY. Physical truth is
used strictly by each simulator's dynamics (as in any simulation) and optional
explicit diagnostics (disabled by default).

NO asteroid drift/speed objective, constraint, or keep-in zone is introduced.
The existing servicer safety CBF and contact guard remain active in MuJoCo.

Gymnasium API: reset -> (obs, info); step -> (obs, reward, terminated,
truncated, info). One step is ONE complete nudge + observation, not a physics dt.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.spatial.transform import Rotation

from src.asteroid_surface import Surface
from src.physical_oed_identification import (
    SCALE, OriginState, com_and_ic, e_opt_score,
    predict_post, momentum_regressor,
)
from src.three_nudge_virtual_10param import (
    Virtual10Settings, VirtualPlan, ContinuousSpatialIdentifier,
    sample_origin, _virtual_step, _guard,
)

# These ten simulator parameters are copied from the repository's SAVED
# baseline validation output (500 kg mesh asteroid). They are only a DEFAULT
# for the approximate training simulator, NOT an estimator input or observation.
# A policy trained here must be validated on the real MuJoCo dynamics.
SURROGATE_REFERENCE_PI = np.array([
    500.0, 1.3995798628882496, -0.8210759535158445, 1.8001423595381043,
    46.91371378602807, 46.54148123112265, 38.8822428447809,
    -1.955842721178828, 0.4537741857097323, 1.5778515349622746,
], dtype=float)


@dataclass
class RLEnvConfig:
    backend: Literal['surrogate', 'mujoco'] = 'surrogate'
    observation_s: float = 30.0
    # Simulations begin with three decisions; one Gym step = one virtual nudge.
    total_nudges: int = 3
    # Surrogate sensor cadence; this does NOT control the MuJoCo physics dt.
    surrogate_sample_s: float = 0.2
    # Randomize the hidden surrogate inertial parameters between episodes.
    randomize_surrogate_truth: bool = True
    relative_mass_std: float = 0.03
    relative_inertia_std: float = 0.03
    com_std_m: float = 0.003
    # Terminal failure penalty ONLY for invalid dynamics / existing CBF violations.
    failure_penalty: float = -5.0
    # If True, development-only truth errors appear in info (never observation/reward).
    debug_truth: bool = False

    def __post_init__(self):
        if self.backend not in ('surrogate', 'mujoco'):
            raise ValueError('backend must be surrogate or mujoco')
        if self.total_nudges != 3:
            raise ValueError('This experiment is defined for exactly 3 nudges')
        if self.observation_s <= 0 or self.surrogate_sample_s <= 0:
            raise ValueError('Observation times must be positive')
        if min(self.relative_mass_std, self.relative_inertia_std, self.com_std_m) < 0:
            raise ValueError('Surrogate domain-randomization stds must be >= 0')


def _upper_triangle_index():
    return np.triu_indices(10)


def normalized_action_to_plan(action: np.ndarray, surface: Surface,
                              cfg: Virtual10Settings, observation_s: float) -> VirtualPlan:
    """Deterministically map [-1,1]^6 to a physically admissible nudge.

    action = [surface longitude, surface z, tangent azimuth, tangent fraction,
              impulse interpolation, force interpolation]. Force is chosen from
    the interval satisfying force, torque and pulse constraints *for the chosen
    point and impulse*. No dependence on true hidden inertial parameters.

    Impossible actuator settings raise ValueError; we do NOT silently violate
    torque or other hard limits. This is a smooth-ish, geometry-aware action map.
    """
    a = np.asarray(action, dtype=float)
    if a.shape != (6,) or not np.isfinite(a).all():
        raise ValueError('Expected six finite action values')
    a = np.clip(a, -1., 1.)
    lon = float(np.pi * (a[0] + 1.))
    z = float(a[1])
    phi = float(np.pi * (a[2] + 1.))
    point, normal = surface.point_and_normal(lon, z)
    normal = np.asarray(normal, float)
    # Robust local tangent basis, matching the repository's OED planner.
    axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
    tangent_a = np.cross(normal, axis)
    tangent_a /= np.linalg.norm(tangent_a)
    tangent_b = np.cross(normal, tangent_a)
    beta = float(cfg.tangent_fraction_max * (a[3]+1.)/2.)
    push = -np.sqrt(1. - beta*beta)*normal + beta*(
        np.cos(phi)*tangent_a + np.sin(phi)*tangent_b)
    push /= np.linalg.norm(push)

    torque_per_newton = float(np.linalg.norm(np.cross(point, push)))
    max_force_torque = (cfg.max_torque_Nm / torque_per_newton
                        if torque_per_newton > 1e-12 else float('inf'))
    force_hi = min(cfg.force_max_N, cfg.max_force_N, max_force_torque)
    force_lo = cfg.force_min_N
    if force_hi + 1e-12 < force_lo:
        raise ValueError('No feasible force at this surface point/direction')
    J_lo = max(cfg.impulse_min_Ns, force_lo * cfg.min_pulse_s)
    J_hi = min(cfg.impulse_max_Ns, cfg.max_impulse_Ns,
               force_hi * cfg.max_pulse_s)
    if J_hi + 1e-12 < J_lo:
        raise ValueError('No feasible impulse at this surface point/direction')
    J = J_lo + (a[4]+1.)/2. * max(0., J_hi-J_lo)
    feasible_F_lo = max(force_lo, J/cfg.max_pulse_s)
    feasible_F_hi = min(force_hi, J/cfg.min_pulse_s)
    if feasible_F_hi + 1e-10 < feasible_F_lo:
        raise ValueError('No feasible pulse duration at this point')
    F = feasible_F_lo + (a[5]+1.)/2. * max(0., feasible_F_hi-feasible_F_lo)
    plan = VirtualPlan(point, normal, push, float(J), float(F), float(observation_s))
    if not (cfg.min_pulse_s-1e-9 <= plan.pulse_s <= cfg.max_pulse_s+1e-9
            and plan.impulse_Ns <= cfg.max_impulse_Ns+1e-9
            and np.linalg.norm(np.cross(point,F*push)) <= cfg.max_torque_Nm+1e-9):
        raise AssertionError('Internal action-map limit error')
    return plan


def plan_to_normalized_action(plan: VirtualPlan, surface: Surface,
                              cfg: Virtual10Settings) -> np.ndarray:
    """Convert a plan from the user's Sobol/E-optimal planner to an RL action.

    Useful for evaluating both planners with the SAME Gym step, reward,
    simulator and observation pipeline. Intended for feasible planner outputs.
    """
    d=np.asarray(plan.point_body,float)
    d=d/np.linalg.norm(d)
    lon=np.arctan2(d[1],d[0]) % (2*np.pi)
    z=float(d[2])
    normal=np.asarray(plan.normal_body,float)
    push=np.asarray(plan.push_body,float)
    axis=np.eye(3)[int(np.argmin(np.abs(normal)))]
    tangent_a=np.cross(normal,axis)
    tangent_a/=np.linalg.norm(tangent_a)
    tangent_b=np.cross(normal,tangent_a)
    tvec=push-np.dot(push,normal)*normal
    beta=float(np.linalg.norm(tvec))
    phi=(np.arctan2(np.dot(tvec,tangent_b),
                    np.dot(tvec,tangent_a)) % (2*np.pi) if beta > 1e-12 else 0.)
    a3=2*beta/cfg.tangent_fraction_max-1. if cfg.tangent_fraction_max else -1.
    torque_per_newton=float(np.linalg.norm(np.cross(plan.point_body,push)))
    force_hi=min(cfg.force_max_N,cfg.max_force_N,
                 cfg.max_torque_Nm/max(1e-12,torque_per_newton))
    J_lo=max(cfg.impulse_min_Ns,cfg.force_min_N*cfg.min_pulse_s)
    J_hi=min(cfg.impulse_max_Ns,cfg.max_impulse_Ns,force_hi*cfg.max_pulse_s)
    f_lo=max(cfg.force_min_N,plan.impulse_Ns/cfg.max_pulse_s)
    f_hi=min(force_hi,plan.impulse_Ns/cfg.min_pulse_s)
    a4=2*(plan.impulse_Ns-J_lo)/max(1e-12,J_hi-J_lo)-1.
    a5=2*(plan.force_N-f_lo)/max(1e-12,f_hi-f_lo)-1.
    action=np.clip(np.array([lon/np.pi-1., z, phi/np.pi-1.,
                             a3,a4,a5],float),-1.,1.).astype(np.float32)
    # Ensure numerical loss from the inverse map does not silently change a
    # candidate by more than the tolerance relevant to this experiment.
    recovered=normalized_action_to_plan(action,surface,cfg,plan.observation_s)
    if (np.linalg.norm(recovered.point_body-plan.point_body)>1e-4 or
        np.linalg.norm(recovered.push_body-plan.push_body)>1e-4 or
        abs(recovered.impulse_Ns-plan.impulse_Ns)>1e-4 or
        abs(recovered.force_N-plan.force_N)>1e-4):
        raise ValueError('Could not invert Sobol plan into RL action')
    return action


def information_reward(precision_before: np.ndarray, precision_after: np.ndarray,
                       projector_unobservable: np.ndarray):
    """Measured information gain, NOT ground-truth estimation error.

    E-optimal improvement + mild D-optimal tie-breaking, reusing the user's
    exact `e_opt_score`. Precision matrices use the existing heuristic
    regression weighting; this is NOT a calibrated Fisher-information matrix.
    """
    delta = .5*(precision_after-precision_before+
                (precision_after-precision_before).T)
    # Preview replacement or numerical noise could make delta weakly indefinite.
    # Project the increment onto PSD; do not reward negative information gain.
    eigenvalues, V = np.linalg.eigh(delta)
    delta_psd = (V * np.maximum(eigenvalues, 0.)) @ V.T
    raw, parts = e_opt_score(precision_before, delta_psd, projector_unobservable)
    # Soft compression prevents very large OED scores from dominating training.
    reward = float(np.log1p(max(0., raw)))
    if not np.isfinite(reward):
        raise ValueError('Non-finite information reward')
    return reward, dict(raw_information_score=float(raw),
                        e_gain=float(parts['E_gain']),
                        d_gain=float(parts['D_gain']),
                        nullspace_gain=float(parts['subspace_excitation']))


class VirtualNudgePlacementEnv(gym.Env):
    """Three-step RL experiment; action optimizes *where/how* to nudge.

    Supports SAC/PPO and any other Gymnasium-compatible agent with Box actions.
    Default backend is the fast surrogate; pass backend='mujoco' for full
    simulation and CBF checks at deployment/evaluation time.
    """

    metadata = {'render_modes': ['ansi'], 'render_fps': 1}

    def __init__(self, project_root: str | Path,
                 settings: Virtual10Settings | None = None,
                 rl_config: RLEnvConfig | None = None,
                 render_mode: str | None = None):
        super().__init__()
        self.root = Path(project_root).expanduser().resolve()
        self.cfg = settings or Virtual10Settings()
        self.rl_cfg = rl_config or RLEnvConfig()
        if self.cfg.n_nudges != self.rl_cfg.total_nudges:
            raise ValueError('Number of nudges must match RL settings')
        if render_mode not in (None, 'ansi'):
            raise ValueError('Only render_mode=None or ansi is supported')
        self.render_mode = render_mode
        self.surface = Surface(self.root / 'model' / 'asteroid.obj')
        self.action_space = spaces.Box(low=-1., high=1., shape=(6,), dtype=np.float32)
        # pi[10], upper triangle of dimensionless regression Gram[55],
        # observed state: position[3], velocity[3], attitude R[9], omega[3],
        # nudge progress[1] => 84 features. No ground truth in observations.
        self.observation_space = spaces.Box(
            low=-1., high=1., shape=(84,), dtype=np.float32)
        self.identifier: ContinuousSpatialIdentifier | None = None
        self.ex = None
        self.current_nudge = 0
        self.episode_done = True
        self.initial_com = None
        self.simulator_pi = None
        self.last_info: dict[str, Any] = {}
        self._rng = np.random.default_rng(0)

    def _sample_hidden_inertia(self):
        """Surrogate physics truth (invisible to estimator / reward / policy)."""
        ref = SURROGATE_REFERENCE_PI
        if not self.rl_cfg.randomize_surrogate_truth:
            return ref.copy()
        for _ in range(100):
            mass = ref[0]*(1.+self._rng.normal(0, self.rl_cfg.relative_mass_std))
            c = ref[1:4]/ref[0] + self._rng.normal(0, self.rl_cfg.com_std_m, 3)
            # Scale all six body-origin inertia elements coherently so the
            # training distribution respects symmetry and positive inertia.
            i = ref[4:]*(
                1. + self._rng.normal(0, self.rl_cfg.relative_inertia_std))
            candidate = np.r_[mass, mass*c, i]
            try:
                com_and_ic(candidate)
                return candidate
            except ValueError:
                continue
        raise RuntimeError('Surrogate inertia randomization produced no valid body')

    def _create_mujoco_experiment(self):
        # Delayed import: training the surrogate does NOT require MuJoCo.
        from experiments.single_nudge_experiment_asteroid_cbf import (
            ExperimentConfig, SingleNudgeExperiment,
        )
        ec = ExperimentConfig(
            model_path=str(self.root/'model'/'model_ellipsoid_self_collision_asteroid_keepin.xml'),
            position_noise_std_m=self.cfg.noise_std,
            velocity_noise_std_mps=self.cfg.noise_std,
            attitude_noise_std_rad=self.cfg.noise_std,
            omega_noise_std_radps=self.cfg.noise_std,
            force_noise_std_N=self.cfg.noise_std,
            torque_noise_std_Nm=self.cfg.noise_std,
            pre_nudge_observation_s=0.2,
            post_nudge_observation_s=self.rl_cfg.observation_s,
            post_nudge_sample_period_s=self.cfg.design_prediction_period_s,
            use_ellipsoidal_cbf=True,
            use_asteroid_keepout_cbf=True,
            use_asteroid_keepin_cbf=True,  # EXISTING relative spacecraft CBF
            post_release_joint_hold=True,
            rng_seed=int(self._rng.integers(1,2**31-1)),
        )
        ex = SingleNudgeExperiment(ec)
        ex.reset_close_range()
        ex._capture_keepin_relative_reference()
        ex.move_to_observation_shell(verbose=False)
        if ex.cbf is None or not ex.cbf.shell_active:
            raise RuntimeError('Existing spacecraft observation-shell CBF is inactive')
        ex._capture_post_release_joint_reference()
        self.ex = ex
        self.initial_com = ex.target_truth().position_com_world_m.copy()  # diagnostics only
        self.identifier.start(sample_origin(ex, self.cfg.noise_std, self._rng))

    def reset(self, *, seed: int | None = None,
              options: dict | None = None):
        super().reset(seed=seed)
        self._rng = self.np_random
        self.current_nudge = 0
        self.episode_done = False
        self.ex = None
        self.initial_com = None
        self.simulator_pi = None
        self.identifier = ContinuousSpatialIdentifier(self.cfg)
        if self.rl_cfg.backend == 'surrogate':
            self.simulator_pi = self._sample_hidden_inertia()
            # A random initial pose/twist, without revealing hidden inertia.
            R = Rotation.from_rotvec(self._rng.normal(0., .05, 3)).as_matrix()
            state = OriginState(
                r=np.array([3.,0.,0.]) + self._rng.normal(0., .02, 3),
                v=self._rng.normal(0., .002, 3),
                R=R,
                omega_body=self._rng.normal(0., .005, 3),
                t=0.,
            )
            self.identifier.start(self._noisy_state(state))
            self._truth_state = state  # backend dynamics only, never observed directly
        else:
            self._create_mujoco_experiment()
        self.last_info = {'backend': self.rl_cfg.backend, 'nudge_number': 0}
        return self._observation(), self.last_info.copy()

    def _noisy_state(self, state: OriginState):
        sig=self.cfg.noise_std
        if not sig:
            return OriginState(np.asarray(state.r,float).copy(),
                               np.asarray(state.v,float).copy(),
                               np.asarray(state.R,float).copy(),
                               np.asarray(state.omega_body,float).copy(),
                               float(state.t))
        return OriginState(
            r=state.r + self._rng.normal(0.,sig,3),
            v=state.v + self._rng.normal(0.,sig,3),
            R=state.R @ Rotation.from_rotvec(self._rng.normal(0.,sig,3)).as_matrix(),
            omega_body=state.omega_body+self._rng.normal(0.,sig,3),
            t=float(state.t),
        )

    def _observation(self):
        est = self.identifier
        if est is None or est.last_state is None:
            raise RuntimeError('Environment must be reset before an observation')
        s=est.last_state
        # Normalize estimator/Gram features. No exact truth, model errors, or
        # reward decomposition are placed in the policy's observation.
        pi_norm=np.tanh((est.pi-est.prior)/SCALE)
        ix=_upper_triangle_index()
        gram=est.gram[ix]
        gram_feat=np.sign(gram)*np.log1p(np.abs(gram))/20.
        obs=np.r_[pi_norm, gram_feat,
                  np.asarray(s.r)/4., np.asarray(s.v)/.1,
                  np.asarray(s.R).reshape(-1), np.asarray(s.omega_body)/.1,
                  self.current_nudge/self.rl_cfg.total_nudges]
        return np.clip(obs, -1.,1.).astype(np.float32)

    def _surrogate_nudge(self, plan:VirtualPlan):
        """Approximate instantaneous impulse, then torque-free free flight.

        Uses hidden simulator_pi ONLY in dynamics (`predict_post`) and never
        in observations, estimation, or reward. This is a fast RL TRAINING
        surrogate, not a finite-duration MuJoCo reproduction.
        """
        pre=self._truth_state
        total_t=plan.pulse_s + plan.observation_s
        times=[plan.pulse_s]
        t=plan.pulse_s
        while t + self.rl_cfg.surrogate_sample_s < total_t - 1e-9:
            t += self.rl_cfg.surrogate_sample_s
            times.append(t)
        if total_t > times[-1] + 1e-9:
            times.append(total_t)
        post=predict_post(pre, self.simulator_pi, plan.point_body,
                          plan.push_body, plan.impulse_Ns, times)
        # Known commanded impulse measured with optional force-sensor noise.
        J_body=plan.push_body*plan.impulse_Ns
        J_world=pre.R@J_body
        point_world=pre.r+pre.R@plan.point_body
        if self.cfg.noise_std:
            J_world=J_world+self._rng.normal(0.,self.cfg.noise_std,3)*plan.pulse_s
        applied_wrench=np.r_[J_world, np.cross(point_world,J_world)]
        for i, state in enumerate(post):
            self.identifier.update(self._noisy_state(state),
                                   applied_wrench if i==0 else np.zeros(6))
        self._truth_state=post[-1]
        return {}

    def _mujoco_nudge(self, plan:VirtualPlan):
        """One full high-fidelity virtual nudge and 30-s default observation."""
        ex=self.ex
        dt=float(ex.dt)
        nsteps=max(1,int(np.ceil(plan.pulse_s/dt)))
        duration=nsteps*dt
        F_body=(plan.impulse_Ns/duration)*plan.push_body
        if (np.linalg.norm(F_body)>self.cfg.max_force_N+1e-8 or
            np.linalg.norm(np.cross(plan.point_body,F_body))>
                self.cfg.max_torque_Nm+1e-8 or
            duration>self.cfg.max_pulse_s+dt/2):
            raise ValueError('Quantized pulse would violate hard actuator bounds')
        total_CBF_before=int(ex.cbf_interventions_total)
        steps=0
        for _ in range(nsteps):
            _, measured=_virtual_step(ex,plan,F_body,dt,self._rng,self.cfg.noise_std)
            self.identifier.update(sample_origin(ex,self.cfg.noise_std,self._rng),measured)
            _guard(ex,self.cfg,self.initial_com,'RL virtual force pulse')
            steps+=1
        ex.data.qfrc_applied[:] = 0.
        for _ in range(int(round(plan.observation_s/dt))):
            ex._idle_step()
            self.identifier.update(sample_origin(ex,self.cfg.noise_std,self._rng),
                                   np.zeros(6))
            _guard(ex,self.cfg,self.initial_com,'RL observation')
            steps+=1
        return {'physics_steps':steps,
                'cbf_interventions_this_nudge':int(ex.cbf_interventions_total)-total_CBF_before}

    def _dev_metrics(self):
        """Debug only; truth is NEVER part of training observation/reward."""
        from src.physical_oed_identification import matrix_inertia
        if self.rl_cfg.backend=='surrogate':
            pi=self.simulator_pi
        else:
            from src.three_nudge_virtual_10param import true_pi_dev_only
            pi=true_pi_dev_only(self.ex)
        e=self.identifier.pi-pi
        return dict(mass_abs_error_kg=float(abs(e[0])),
                    com_error_mm=float(1000*np.linalg.norm(
                        self.identifier.pi[1:4]/self.identifier.pi[0]-pi[1:4]/pi[0])),
                    inertia_fro_error_kgm2=float(np.linalg.norm(
                        matrix_inertia(self.identifier.pi)-matrix_inertia(pi))))

    def step(self, action:np.ndarray):
        if self.episode_done:
            raise RuntimeError('Episode complete. Call reset() before step().')
        ident=self.identifier
        P0=ident.precision.copy()
        projector=ident.metrics()['P_unobservable'].copy()
        self.current_nudge += 1
        info={'backend':self.rl_cfg.backend,
              'nudge_number':int(self.current_nudge)}
        try:
            plan=normalized_action_to_plan(action,self.surface,self.cfg,
                                           self.rl_cfg.observation_s)
            data=(self._surrogate_nudge(plan) if self.rl_cfg.backend=='surrogate'
                  else self._mujoco_nudge(plan))
            gain,parts=information_reward(P0,ident.precision,projector)
            info.update(data)
            info.update(parts)
            info.update(dict(
                point_body_m=plan.point_body.copy(),
                normal_body=plan.normal_body.copy(),
                push_body=plan.push_body.copy(),
                impulse_Ns=float(plan.impulse_Ns),
                force_N=float(plan.force_N),
                pulse_s=float(plan.pulse_s),
                observation_s=float(plan.observation_s),
                estimated_pi=ident.pi.copy(),
                numerical_rank=int(ident.metrics()['rank']),
                estimator_updates=int(ident.updates),
            ))
            if self.rl_cfg.debug_truth:
                info['DEV_ONLY_errors']=self._dev_metrics()
            self.episode_done=(self.current_nudge>=self.rl_cfg.total_nudges)
            self.last_info=info
            return self._observation(), gain, self.episode_done, False, info
        except (ValueError, FloatingPointError, np.linalg.LinAlgError,
                RuntimeError) as exc:
            # Existing safety/physics failures invalidate the episode.
            # NO penalties/abort rules for asteroid drift or target speed.
            self.episode_done=True
            info.update(failure_type=type(exc).__name__,
                        failure_reason=str(exc), safety_or_model_failure=True)
            self.last_info=info
            return self._observation(), float(self.rl_cfg.failure_penalty), True, False, info

    def render(self):
        pi=self.identifier.pi if self.identifier is not None else None
        return (f'backend={self.rl_cfg.backend} nudge={self.current_nudge}/3 '
                f'mass_est={pi[0]:.3f}kg' if pi is not None else 'Environment not reset')

    def close(self):
        if self.ex is not None:
            try:
                self.ex.data.qfrc_applied[:]=0.
            except (AttributeError, ValueError):
                pass
        self.ex=None
