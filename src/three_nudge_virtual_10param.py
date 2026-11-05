"""Online continuous-timestep virtual-wrench excitation and 10-parameter ID.

Drop into active_inertial_id/src/three_nudge_virtual_10param.py.

Physical pad contact is intentionally absent. During each pulse, mj_applyFT
applies a known body-fixed force at a surface point, and the code integrates
both linear impulse and moment impulse about a fixed world origin. The
spacecraft continues to evolve under its existing CBF-filtered station hold,
but receives NO equal-and-opposite reaction from virtual contact.

Parameter order is the existing physical_oed_identification.py convention:
  [m, m*c_x, m*c_y, m*c_z, I_Oxx, I_Oyy, I_Ozz, I_Oxy, I_Oxz, I_Oyz]
with the inertia tensor ABOUT the asteroid body origin.

The estimator uses non-overlapping segments with continuously refreshed previews.
The candidate predictor is an approximate instantaneous-impulse surrogate.
Its information matrix is not a statistically exact Fisher matrix. Selection
maximizes estimated information gain only, without target drift/speed penalties
or target drift/speed limits. Asteroid travel and speed are logged as diagnostics.
The servicer's EXISTING radial CBF, contact guard, and actuator limits remain;
these runtime checks are not reachability certificates.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import csv
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from src.physical_oed_identification import (
    SCALE, OriginState, EstimatorConfig,
    com_and_ic, e_opt_score, predict_post,
    matrix_inertia, momentum_regressor, normalize,
)
from src.asteroid_surface import Surface


@dataclass
class Virtual10Settings:
    noise_std: float = 0.0        # supported: zero or 1e-4 (position, velocity, attitude, rate, force)
    n_nudges: int = 3
    n_candidates: int = 64
    impulse_min_Ns: float = 0.20
    impulse_max_Ns: float = 2.0
    force_min_N: float = 1.0
    force_max_N: float = 4.5
    max_impulse_Ns: float = 3.0
    max_force_N: float = 7.5
    max_torque_Nm: float = 8.0
    min_pulse_s: float = 0.12
    max_pulse_s: float = 2.0
    observation_choices_s: tuple = (30.0,)  # full-rate observation between excitation events
    design_prediction_period_s: float = 1.0 # optimizer surrogate ONLY; NOT measurement rate
    estimator_window_s: float = 0.20          # non-overlapping committed segments + per-step live preview
    force_rate_weight_sigma_N: float = 0.15   # objective scale, NOT sensor noise covariance
    torque_rate_weight_sigma_Nm: float = 0.05
    tangent_fraction_max: float = 0.20
    rng_seed: int = 19
    record_period_s: float = 0.10
    prior_sigma_normalized: float = 10.0  # weak initial prior; validate noise robustness separately

    def __post_init__(self):
        if self.noise_std not in (0.0, 1e-4):
            raise ValueError('noise_std must equal 0.0 or 1e-4')
        if self.n_nudges != 3:
            raise ValueError('Exactly three virtual nudges are required')
        if self.n_candidates < 16:
            raise ValueError('n_candidates must be >= 16')
        if not 0 < self.impulse_min_Ns < self.impulse_max_Ns <= self.max_impulse_Ns:
            raise ValueError('Invalid impulse bounds')
        if not 0 < self.force_min_N < self.force_max_N <= self.max_force_N:
            raise ValueError('Invalid force bounds')
        if not 0 < self.min_pulse_s < self.max_pulse_s:
            raise ValueError('Invalid pulse bounds')
        if self.prior_sigma_normalized <= 0:
            raise ValueError('Positive prior uncertainty required')
        if any(t <= 0 for t in self.observation_choices_s):
            raise ValueError('Observation times must be positive')
        if (self.estimator_window_s <= 0 or self.design_prediction_period_s <= 0 or
            self.force_rate_weight_sigma_N <= 0 or self.torque_rate_weight_sigma_Nm <= 0):
            raise ValueError('Positive estimator-window and rate-weight parameters required')


@dataclass(frozen=True)
class VirtualPlan:
    point_body: np.ndarray
    normal_body: np.ndarray
    push_body: np.ndarray
    impulse_Ns: float
    force_N: float
    observation_s: float
    utility: float = 0.0

    @property
    def pulse_s(self):
        return self.impulse_Ns / self.force_N


def sample_origin(ex, noise_std, rng):
    """Development stand-in for a body-ORIGIN pose/twist sensor; never uses CoM truth."""
    import mujoco
    mujoco.mj_forward(ex.model, ex.data)
    bid = ex.TARGET_BODY_ID
    twist = np.zeros(6)
    mujoco.mj_objectVelocity(ex.model, ex.data, mujoco.mjtObj.mjOBJ_XBODY,
                             bid, twist, 0)  # spatial/world angular then linear velocity
    R = np.asarray(ex.data.xmat[bid], float).reshape(3, 3).copy()
    r = np.asarray(ex.data.xpos[bid], float).copy()
    v = twist[3:].copy()
    w_body = R.T @ twist[:3]
    if noise_std:
        r += rng.normal(0, noise_std, 3)
        v += rng.normal(0, noise_std, 3)
        w_body += rng.normal(0, noise_std, 3)
        R = R @ Rotation.from_rotvec(rng.normal(0, noise_std, 3)).as_matrix()
    return OriginState(r=r, v=v, R=R, omega_body=w_body, t=float(ex.data.time))


def _candidate_from_unit(x, surface, cfg, T):
    # [uniform sphere lon, z, tangent azimuth, tangential fraction, J, F]
    p, n = surface.point_and_normal(2*np.pi*x[0], 2*x[1]-1)
    axis = np.eye(3)[int(np.argmin(abs(n)))]
    u = normalize(np.cross(n, axis)); v = np.cross(n, u)
    beta = cfg.tangent_fraction_max * float(x[3])
    push = normalize(-np.sqrt(1-beta*beta)*n + beta*(
        np.cos(2*np.pi*x[2])*u + np.sin(2*np.pi*x[2])*v))
    J = cfg.impulse_min_Ns + x[4]*(cfg.impulse_max_Ns-cfg.impulse_min_Ns)
    F = cfg.force_min_N + x[5]*(cfg.force_max_N-cfg.force_min_N)
    tau = J/F
    if not cfg.min_pulse_s <= tau <= cfg.max_pulse_s:
        return None
    if np.linalg.norm(np.cross(p, F*push)) > cfg.max_torque_Nm:
        return None
    return VirtualPlan(p, n, push, float(J), float(F), float(T))


def _decision_samples(cfg, rng, pre=None):
    """Sobol candidate nudges; no drift-minimizing or braking-only bias."""
    m = int(np.ceil(np.log2(cfg.n_candidates)))
    raw = qmc.Sobol(d=6, scramble=True, seed=int(rng.integers(2**31))).random_base2(m)
    return list(raw[:cfg.n_candidates])


def _candidate_rate_information(pre, pi, plan, cfg):
    """Heuristic continuous-rate information for a nominal instantaneous jump.

    This is an OED surrogate, not an exact Fisher information matrix and not
    the full finite-duration contact dynamics. It is never used as measured data.
    """
    period = float(cfg.design_prediction_period_s)
    times = [max(plan.pulse_s, 1e-6)]
    t = times[0]
    while t + period < plan.pulse_s + plan.observation_s - 1e-9:
        t += period
        times.append(t)
    if times[-1] < plan.pulse_s + plan.observation_s - 1e-9:
        times.append(plan.pulse_s + plan.observation_s)
    states = predict_post(pre, pi, plan.point_body, plan.push_body,
                          plan.impulse_Ns, times)
    rate_W = np.diag([1.0/cfg.force_rate_weight_sigma_N]*3+
                     [1.0/cfg.torque_rate_weight_sigma_Nm]*3)
    Bprev = momentum_regressor(pre)
    last_t = 0.0
    info = np.zeros((10, 10))
    for t, state in zip(times, states):
        span = t-last_t
        X = rate_W @ ((momentum_regressor(state)-Bprev)/span) @ np.diag(SCALE)
        info += span*(X.T@X)
        Bprev = momentum_regressor(state)
        last_t = t
    return 0.5*(info+info.T)


def select_virtual_plan(surface, pre, ident, initial_origin, cfg, nudge_number, rng):
    """Replan each excitation using CURRENT estimate; optimize information ONLY.

    initial_origin is retained as a compatibility argument but deliberately
    not used. No asteroid displacement/velocity constraints or costs are used.
    Pulse duration, force, impulse, and torque limits still apply. The
    spacecraft CBF/contact checks remain active when MuJoCo executes the plan.
    """
    prior_precision = ident.precision.copy()
    projector = ident.metrics()['P_unobservable']
    best = None
    scores = []
    rejected = dict(pulse_or_torque=0, information=0, accepted=0)
    for x in _decision_samples(cfg, rng, pre):
        for T in cfg.observation_choices_s:
            plan = _candidate_from_unit(x, surface, cfg, T)
            if plan is None:
                rejected['pulse_or_torque'] += 1
                continue
            try:
                F_info = _candidate_rate_information(pre, ident.pi, plan, cfg)
                gain, parts = e_opt_score(prior_precision, F_info, projector)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                rejected['information'] += 1
                continue
            if not np.isfinite(gain):
                rejected['information'] += 1
                continue
            # No displacement, target-speed, force-size, impulse-size, or time penalty.
            # Force/impulse/torque/pulse restrictions are HARD candidate constraints.
            utility = float(gain)
            rejected['accepted'] += 1
            scores.append([utility, plan.impulse_Ns, plan.force_N, T,
                           *plan.point_body, *plan.push_body,
                           parts['E_gain'], parts['subspace_excitation']])
            if best is None or utility > best.utility:
                best = replace(plan, utility=utility)
    if best is None:
        raise RuntimeError(
            f'Nudge {nudge_number}: no feasible virtual plan found. '
            f'Rejection counts: {rejected}. Check pulse/force/torque bounds, '
            'candidate sampling, and the information model. No asteroid drift '
            'or speed restrictions were applied.')
    return best, np.asarray(scores, float)


class ContinuousSpatialIdentifier:
    """Continuous estimate with sequential (non-overlapping) momentum segments.

    The sensor and estimator run EVERY physics step. An in-progress segment is
    *replaced*, not cumulatively re-added, at every step. Only a completed
    interval is committed once, eliminating the artificial repeated counting
    of almost identical 0.2-s sliding-window rows in the earlier version.

    For each completed segment [a,b] we use the integrated momentum balance
        (B_b - B_a) pi = integral_a^b W dt.
    To retain the previous rate weighting convention for intervals of varying
    durations, both sides are divided by sqrt(max(b-a, window_s)). This also
    prevents noisy VERY short preview intervals from getting excessive weight.

    Adjacent intervals share endpoint measurements; the normal matrix and P
    are sensitivity diagnostics, NOT statistically calibrated covariance.
    No ground-truth asteroid mass, CoM, or inertia enters identification.
    """

    def __init__(self, cfg: Virtual10Settings):
        self.cfg = cfg
        self.prior = np.array([600., 0., 0., 0., 65., 68., 75., 0., 0., 0.])
        self.pi = self.prior.copy()
        self.rate_W = np.diag([1.0 / cfg.force_rate_weight_sigma_N] * 3 +
                              [1.0 / cfg.torque_rate_weight_sigma_Nm] * 3)
        self.parameter_scale = np.diag(SCALE)
        self.prior_precision = np.eye(10) / cfg.prior_sigma_normalized**2
        self.committed_gram = np.zeros((10, 10))
        self.committed_rhs = np.zeros(10)
        self.gram = np.zeros((10, 10))
        self.rhs = np.zeros(10)
        self.precision = self.prior_precision.copy()
        self.P = np.linalg.inv(self.precision)
        self.cumulative_wrench = np.zeros(6)
        self.last_state = None
        self.anchor_time = None
        self.anchor_B = None
        self.anchor_wrench = None
        self.completed_segments = 0
        self.updates = 0
        self.total_samples = 0
        self.rank_tolerance = 1e-6
        self.last_window_s = 0.0
        self.last_weighted_rate_residual = float('nan')

    def start(self, initial_state):
        if self.last_state is not None:
            raise RuntimeError('Streaming estimator may be initialized only once')
        self.last_state = initial_state
        self.anchor_time = float(initial_state.t)
        self.anchor_B = momentum_regressor(initial_state)
        self.anchor_wrench = self.cumulative_wrench.copy()
        self.total_samples = 1

    def update(self, observed_state, measured_increment_world_origin):
        """Ingest one physics-step measurement and refresh all 10 estimates."""
        if self.last_state is None:
            raise RuntimeError('Call start(initial_state) first')
        dt = float(observed_state.t - self.last_state.t)
        if dt <= 0:
            raise ValueError('State timestamps must increase every timestep')
        dW = np.asarray(measured_increment_world_origin, float)
        if dW.shape != (6,) or not np.all(np.isfinite(dW)):
            raise ValueError('Six finite integrated wrench increments are required')
        self.cumulative_wrench += dW
        current_B = momentum_regressor(observed_state)
        self.total_samples += 1
        self.last_state = observed_state

        h = float(observed_state.t - self.anchor_time)
        self.last_window_s = h
        dB = current_B - self.anchor_B
        dW_segment = self.cumulative_wrench - self.anchor_wrench

        # The temporary, unfinished segment REPLACES its previous version.
        # A short preview is conservatively weighted against the full window,
        # rather than dividing noisy pose differences by a nearly-zero h.
        denom = np.sqrt(max(h, self.cfg.estimator_window_s))
        X = self.rate_W @ dB @ self.parameter_scale / denom
        y = self.rate_W @ (dW_segment - dB @ self.prior) / denom
        live_gram = X.T @ X
        live_rhs = X.T @ y
        self.gram = 0.5 * (self.committed_gram + self.committed_gram.T) + live_gram
        self.rhs = self.committed_rhs + live_rhs
        self.precision = 0.5 * (self.gram + self.gram.T) + self.prior_precision
        delta = np.linalg.solve(self.precision, self.rhs)
        candidate = self.prior + SCALE * delta

        # Preserve the physical-inertia check from the original implementation.
        previous = self.pi.copy()
        for alpha in (1., .5, .25, .125, .0625, 0.):
            p = previous + alpha * (candidate - previous)
            try:
                com_and_ic(p)
                self.pi = p
                break
            except ValueError:
                continue
        self.P = np.linalg.inv(self.precision)
        self.last_weighted_rate_residual = float(np.linalg.norm(
            self.rate_W @ (dW_segment - dB @ self.pi) / max(h, 1e-12)))
        self.updates += 1

        # Commit this interval ONCE, then begin a new interval from the
        # current state. No asteroid displacement or velocity constraint.
        if h + 0.5 * dt >= self.cfg.estimator_window_s:
            self.committed_gram += live_gram
            self.committed_rhs += live_rhs
            self.completed_segments += 1
            self.anchor_time = float(observed_state.t)
            self.anchor_B = current_B.copy()
            self.anchor_wrench = self.cumulative_wrench.copy()
        return self.pi.copy()

    def metrics(self):
        ev, V = np.linalg.eigh(0.5 * (self.gram + self.gram.T))
        ev = np.maximum(ev, 0.0)
        order = np.argsort(ev)[::-1]
        ev, V = ev[order], V[:, order]
        sig = np.sqrt(ev)
        threshold = max(1e-12, self.rank_tolerance * sig[0])
        rank = int(np.sum(sig > threshold))
        obs = V[:, :rank] @ V[:, :rank].T
        unobs = V[:, rank:] @ V[:, rank:].T
        return dict(rank=rank, singular_values=sig,
                    condition=float(sig[0] / sig[rank - 1]) if rank else float('inf'),
                    P_observable=obs, P_unobservable=unobs,
                    completed_segments=self.completed_segments)


def _virtual_step(ex, plan, F_body, dt, rng, noise_std):
    """One physical MuJoCo time-step with an externally prescribed target wrench."""
    import mujoco
    mujoco.mj_forward(ex.model, ex.data)
    bid = ex.TARGET_BODY_ID
    R = np.asarray(ex.data.xmat[bid], float).reshape(3, 3).copy()
    origin = np.asarray(ex.data.xpos[bid], float).copy()
    point_world = origin + R @ plan.point_body
    force_world = R @ F_body
    dJ = force_world*dt
    dL = np.cross(point_world, force_world)*dt  # moment about FIXED world origin
    # Sensor noise alters measured wrench, never the true force applied to MuJoCo.
    measured_force = force_world + (rng.normal(0, noise_std, 3) if noise_std else 0.)
    measured_dJ = measured_force*dt
    measured_dL = np.cross(point_world, measured_force)*dt
    ex.data.qfrc_applied[:] = 0.0
    mujoco.mj_applyFT(ex.model, ex.data, force_world, np.zeros(3),
                      point_world, bid, ex.data.qfrc_applied)
    try:
        ex.execute_step(ex._station_keep_control(), accumulate_impulse=False)
    finally:
        ex.data.qfrc_applied[:] = 0.0
    return np.r_[dJ, dL], np.r_[measured_dJ, measured_dL]


def true_pi_dev_only(ex):
    """Validation oracle. Never invoked in planner, sensor, or estimator."""
    model = ex.model; bid = ex.TARGET_BODY_ID
    m = float(model.body_mass[bid])
    c = np.asarray(model.body_ipos[bid], float)
    qi = np.asarray(model.body_iquat[bid], float)
    Ri = Rotation.from_quat(np.roll(qi, -1)).as_matrix()
    Ic = Ri @ np.diag(np.asarray(model.body_inertia[bid], float)) @ Ri.T
    Io = Ic + m*((c@c)*np.eye(3) - np.outer(c, c))
    return np.r_[m, m*c, Io[0,0], Io[1,1], Io[2,2], Io[0,1], Io[0,2], Io[1,2]]


def _guard(ex, cfg, initial_com, stage):
    """Simulation-truth watchdog, never used for optimizing or estimating."""
    import mujoco
    mujoco.mj_forward(ex.model, ex.data)
    if ex.cbf is None or not ex.cbf.shell_active:
        raise RuntimeError(f'Safety abort {stage}: radial CBF inactive')
    r = float(ex.cbf._centers(ex.data)[2])
    if (r < ex.cbf.shell_min_center_m-1e-6 or
            r > ex.cbf.shell_max_center_m+1e-6):
        raise RuntimeError(f'Safety abort {stage}: radial separation {r:.5f} m')
    for k in range(ex.data.ncon):
        c = ex.data.contact[k]
        if (ex.TARGET_GEOM_ID in (int(c.geom[0]), int(c.geom[1]))
                and float(c.dist) <= 0):
            raise RuntimeError(f'Safety abort {stage}: unintended physical contact')
    truth = ex.target_truth()
    drift = float(np.linalg.norm(truth.position_com_world_m-initial_com))
    speed = float(np.linalg.norm(truth.velocity_com_world_mps))
    # Asteroid drift/speed are OBSERVATIONS ONLY, not safety-abort conditions.
    # The existing spacecraft CBF and unintended-contact checks remain above.
    return drift, speed, r


def run_virtual_three(project_root, settings=None, output_dir=None, verbose=True):
    """Continuous-time-step sensing and online estimation, with 3 excitation events.

    MuJoCo is integrated numerically at dt. The origin pose/twist and applied
    wrench are sampled/integrated at EVERY step, and 10-parameter WLS is updated
    continuously. Only the virtual-force *schedule* consists of three events.
    No per-nudge measurement collapse, resets, or batch estimator updates.
    """
    import mujoco
    from experiments.single_nudge_experiment_asteroid_cbf import (
        ExperimentConfig, SingleNudgeExperiment)

    cfg = settings or Virtual10Settings()
    root = Path(project_root).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    output = Path(output_dir or root/'results'/'three_nudge_virtual_10param_continuous_v8')
    output.mkdir(parents=True,exist_ok=True)
    surface = Surface(root/'model'/'asteroid.obj')
    ec = ExperimentConfig(
        model_path=str(root/'model'/'model_ellipsoid_self_collision_asteroid_keepin.xml'),
        position_noise_std_m=cfg.noise_std,
        velocity_noise_std_mps=cfg.noise_std,
        attitude_noise_std_rad=cfg.noise_std,
        omega_noise_std_radps=cfg.noise_std,
        force_noise_std_N=cfg.noise_std,
        torque_noise_std_Nm=cfg.noise_std,
        pre_nudge_observation_s=0.2,
        post_nudge_observation_s=max(cfg.observation_choices_s),
        post_nudge_sample_period_s=cfg.design_prediction_period_s,
        use_ellipsoidal_cbf=True,
        use_asteroid_keepout_cbf=True,
        use_asteroid_keepin_cbf=True,
        post_release_joint_hold=True,
        rng_seed=cfg.rng_seed,
    )
    ex = SingleNudgeExperiment(ec)
    ex.reset_close_range()  # INITIAL setup only; no resets between nudges
    ex._capture_keepin_relative_reference()
    ex.move_to_observation_shell(verbose=False)
    if ex.cbf is None or not ex.cbf.shell_active:
        raise RuntimeError('Initial CBF observation shell is not active')
    ex._capture_post_release_joint_reference()
    estimator = ContinuousSpatialIdentifier(cfg)
    rng = np.random.default_rng(cfg.rng_seed+200)
    initial_com = ex.target_truth().position_com_world_m.copy()  # diagnostics only
    # The initial origin reference MUST be a real sensor sample, rather than
    # estimator.last_state (which is None until .start() has been called).
    initial_state = sample_origin(ex, cfg.noise_std, rng)
    estimator.start(initial_state)
    initial_origin = initial_state  # planner signature compatibility (not screened)
    dt = float(ex.dt)
    recorder_stride = max(1,int(round(cfg.record_period_s/dt)))
    records=[]; trace=[]; num_steps=0
    accumulated_actual_wrench = np.zeros(6)
    active_stage='initial'

    def trace_record(stage):
        m=estimator.metrics()
        truth=ex.target_truth()  # drift/speed diagnostics only; NEVER estimator/optimizer input
        trace.append({
            'time_s':float(ex.data.time),'stage':int(stage),'nudge':len(records)+1,
            'pi':estimator.pi.copy(),'rank':int(m['rank']),
            'singular_values':m['singular_values'].copy(),
            'asteroid_origin':np.asarray(ex.data.xpos[ex.TARGET_BODY_ID],float).copy(),
            'asteroid_com_truth_diagnostic':truth.position_com_world_m.copy(),
            'asteroid_com_speed_truth_diagnostic':float(np.linalg.norm(truth.velocity_com_world_mps)),
            'cumulative_force_impulse_world':accumulated_actual_wrench[:3].copy(),
            'cumulative_moment_impulse_world_origin':accumulated_actual_wrench[3:].copy(),
            'cbf_interventions':int(ex.cbf_interventions_total),
            'observed_condition':float(m['condition']),
            'last_window_s':float(estimator.last_window_s),
            'completed_segments':int(estimator.completed_segments),
            'weighted_rate_residual':float(estimator.last_weighted_rate_residual),
        })

    def persist_trace(filename):
        if not trace:
            return
        np.savez_compressed(output/filename,
            time_s=np.array([a['time_s'] for a in trace]),
            stage=np.array([a['stage'] for a in trace],dtype=int),
            nudge=np.array([a['nudge'] for a in trace],dtype=int),
            pi=np.array([a['pi'] for a in trace]),
            numerical_rank=np.array([a['rank'] for a in trace],dtype=int),
            singular_values=np.array([a['singular_values'] for a in trace]),
            asteroid_origin_m=np.array([a['asteroid_origin'] for a in trace]),
            asteroid_com_truth_diagnostic_m=np.array([a['asteroid_com_truth_diagnostic'] for a in trace]),
            asteroid_com_speed_truth_diagnostic_mps=np.array([a['asteroid_com_speed_truth_diagnostic'] for a in trace]),
            applied_linear_impulse_Ns=np.array([a['cumulative_force_impulse_world'] for a in trace]),
            applied_angular_impulse_world_origin_Nms=np.array([a['cumulative_moment_impulse_world_origin'] for a in trace]),
            cbf_interventions=np.array([a['cbf_interventions'] for a in trace],dtype=int),
            observed_condition=np.array([a['observed_condition'] for a in trace]),
            last_window_s=np.array([a['last_window_s'] for a in trace]),
            completed_segments=np.array([a['completed_segments'] for a in trace], dtype=int),
            weighted_rate_residual=np.array([a['weighted_rate_residual'] for a in trace]),
            # Save the complete online normal equations for post-run
            # regularization sensitivity checks WITHOUT rerunning MuJoCo.
            # Provisional segment is included ONLY once, not double counted.
            normal_matrix=estimator.gram.copy(),
            normal_rhs=estimator.rhs.copy(),
            prior_pi=estimator.prior.copy(),
            parameter_scale=SCALE.copy(),
            total_simulation_steps=int(num_steps),
            total_sensor_samples=int(estimator.total_samples),
            total_online_estimator_updates=int(estimator.updates),
            total_completed_estimator_segments=int(estimator.completed_segments),
            dt_s=dt)

    def advance(stage,plan=None, F_body=None):
        """One step + one six-wrench integration + one sensor + one estimator refresh."""
        nonlocal num_steps, active_stage, accumulated_actual_wrench
        active_stage=str(stage)
        if F_body is None:
            ex._idle_step()
            dW=np.zeros(6)
        else:
            actual, measured = _virtual_step(ex,plan,F_body,dt,rng,cfg.noise_std)
            dW = measured
            accumulated_actual_wrench += actual
        # One live observation and one online estimation update for EVERY step.
        measured_state=sample_origin(ex,cfg.noise_std,rng)
        estimator.update(measured_state,dW)
        _guard(ex,cfg,initial_com,active_stage)
        num_steps += 1
        if num_steps%recorder_stride==0:
            trace_record(1 if F_body is not None else 2)

    try:
        trace_record(0)
        for k in range(1,cfg.n_nudges+1):
            pre = estimator.last_state
            plan,scores=select_virtual_plan(surface,pre,estimator,initial_origin,cfg,k,rng)
            np.save(output/f'candidate_scores_{k}.npy',scores)
            nsteps=max(1,int(np.ceil(plan.pulse_s/dt)))
            duration=nsteps*dt
            F_body=(plan.impulse_Ns/duration)*plan.push_body
            if (np.linalg.norm(F_body)>cfg.max_force_N+1e-8 or
                    np.linalg.norm(np.cross(plan.point_body,F_body))>cfg.max_torque_Nm+1e-8 or
                    duration>cfg.max_pulse_s+dt/2):
                raise RuntimeError('Virtual pulse force / torque / duration violates hard bounds')
            if verbose:
                print(f'NUDGE {k} (CONTINUOUS VIRTUAL): utility={plan.utility:.3f} | '
                      f'J={plan.impulse_Ns:.4f} N s | F={np.linalg.norm(F_body):.3f} N | '
                      f'pulse={duration:.3f} s | streaming observation={plan.observation_s:g} s')
            Wpre = accumulated_actual_wrench.copy()
            updates_pre = estimator.updates
            for _ in range(nsteps):
                advance(f'virtual pulse {k}',plan,F_body)
            ex.data.qfrc_applied[:] = 0.0
            Wafter=accumulated_actual_wrench.copy()
            # Continue streaming states, zero applied wrench outside pulses.
            obs_steps=max(1,int(round(plan.observation_s/dt)))
            for _ in range(obs_steps):
                advance(f'observation {k}')
            stats=estimator.metrics()
            drift,speed,radius=_guard(ex,cfg,initial_com,f'after nudge {k}')
            row={
                'nudge':k,'point_body_m':plan.point_body.tolist(),
                'normal_body':plan.normal_body.tolist(), 'push_body':plan.push_body.tolist(),
                'impulse_commanded_Ns':float(plan.impulse_Ns),
                'impulse_applied_world_magnitude_Ns':float(np.linalg.norm((Wafter-Wpre)[:3])),
                'integrated_wrench_world_origin':(Wafter-Wpre).tolist(),
                'force_commanded_N':float(np.linalg.norm(F_body)),
                'pulse_s':duration, 'observation_s':float(plan.observation_s),
                'estimate_pi':estimator.pi.tolist(),
                'estimated_com_body_m':(estimator.pi[1:4]/estimator.pi[0]).tolist(),
                'numerical_rank':int(stats['rank']),
                'singular_values':stats['singular_values'].tolist(),
                'online_updates_during_this_nudge_and_observation':int(estimator.updates-updates_pre),
                'drift_m':drift,'speed_mps':speed,'bus_radius_m':radius,
                'cbf_interventions_total':int(ex.cbf_interventions_total),
                'end_sim_time_s':float(ex.data.time),
            }
            records.append(row)
            persist_trace('online_trace_partial.npz')
            (output/'partial_records.json').write_text(json.dumps(records,indent=2))
            if verbose:
                print(f'  Nudge {k} complete | online updates since previous nudge='
                      f'{estimator.updates-updates_pre:,} | rank={stats["rank"]}/10 | '
                      f'mass={estimator.pi[0]:.4f} kg | '
                      f'CoM={np.round(estimator.pi[1:4]/estimator.pi[0],5)} m | '
                      f'asteroid drift={drift:.4f} m')
    except Exception as err:
        persist_trace('online_trace_partial.npz')
        (output/'failure.txt').write_text(
            f'Only {len(records)} of 3 virtual nudges completed.\n'
            f'Online estimator steps={estimator.updates} / sim steps={num_steps}.\n'
            f'{type(err).__name__}: {err}\n')
        (output/'partial_records.json').write_text(json.dumps(records,indent=2))
        raise

    true_pi=true_pi_dev_only(ex)   # validation oracle, only AFTER final estimate
    error=estimator.pi-true_pi
    validation={
        'true_pi_DEV_ONLY':true_pi.tolist(), 'final_pi':estimator.pi.tolist(),
        'final_error':error.tolist(),
        'mass_relative_error':float(error[0]/true_pi[0]),
        'com_error_m':float(np.linalg.norm(
            estimator.pi[1:4]/estimator.pi[0]-true_pi[1:4]/true_pi[0])),
        'inertia_O_frobenius_error_kgm2':float(np.linalg.norm(
            matrix_inertia(estimator.pi)-matrix_inertia(true_pi))),
        'final_numerical_rank':int(estimator.metrics()['rank']),
        'completed_nudges':len(records),
        'mujoCo_steps_total':num_steps,
        'every_timestep_sensor_samples':int(estimator.total_samples),
        'online_estimator_updates':int(estimator.updates),
        'dt_s':dt,
        'estimator_window_s':float(cfg.estimator_window_s),
        'prior_sigma_normalized':float(cfg.prior_sigma_normalized),
        'completed_nonoverlapping_segments':int(estimator.completed_segments),
        'note':'Sequential time intervals with live previews; adjacent interval endpoints share noise. P is not a calibrated covariance.',
    }
    persist_trace('online_trace.npz')
    (output/'records.json').write_text(json.dumps(records,indent=2))
    (output/'validation_DEV_ONLY.json').write_text(json.dumps(validation,indent=2))
    with (output/'summary.csv').open('w',newline='') as f:
        fields=['nudge','impulse_commanded_Ns','impulse_applied_world_magnitude_Ns',
                'force_commanded_N','pulse_s','observation_s','numerical_rank',
                'online_updates_during_this_nudge_and_observation','drift_m','speed_mps']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader(); writer.writerows(records)
    return {'experiment':ex,'identifier':estimator,'records':records,
            'true_pi_DEV_ONLY':true_pi,'validation':validation,
            'output_dir':output,'trace':trace}
