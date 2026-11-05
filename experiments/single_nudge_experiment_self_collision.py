from __future__ import annotations

"""
single_nudge_experiment.py
===========================
ONE gentle nudge, followed by a LONG free-flight observation window, with NO
onboard vision/camera and an ellipsoidal SERVICER SELF-COLLISION CBF.

This module replaces the multi-file "successive nudge + camera + robust CBF"
orchestration (`successive_nudge_setup.py` + `vision.py` + `robust_cbf.py`
wired together inside the old notebooks) with a single, much shorter,
self-contained control loop, for the following reasons -- see
README_SINGLE_NUDGE.md for the full rationale:

  1. Single nudge, not successive: the mass regression is well posed after
     one impulse. The inertia regression is NOT well posed from a single
     before/after pair (3 equations, 6 unknowns) -- what actually makes it
     well posed here is sampling the post-nudge free-tumble MANY times and
     feeding each (pre, post_k) pair, all sharing the SAME measured impulse,
     into `nudge_experiment.SuccessiveNudgeSession` as if they were separate
     nudges. This is exact, not an approximation: after the nudge ends,
     angular momentum in the WORLD frame is conserved, so for any two
     post-nudge times t_j, t_k,
         R(t_k) I w_body(t_k) - R(t_j) I w_body(t_j) = 0,
     and NudgeObservation.inertia_regression() already encodes exactly this
     equation. No changes to inertial_identifier.py or nudge_experiment.py
     were needed to exploit this -- see validate_single_nudge_idea.py for a
     pure-numpy (no MuJoCo) proof using those two files unmodified.

  2. No vision: target state (position, velocity, attitude, body-frame
     angular rate) is obtained from `pose_sensor.py`, which adds injected
     Gaussian noise directly to MuJoCo ground truth. There is no camera
     anywhere in model.xml any more.

  3. Ellipsoidal self-collision CBF: fixed virtual ellipsoids are attached
     to the spacecraft bus and manipulator.  The CBF monitors selected
     bus/link and non-adjacent link/link pairs.  It does NOT impose any
     spacecraft-to-asteroid collision-avoidance barrier in this version.
     Intentional nudge-pad/asteroid contact remains governed by the independent
     force/impulse/contact-duration supervisor.

  4. Close-range start: `reset_close_range()` places the spacecraft/arm so
     that only a short (well under 1 m) final approach is needed, instead of
     the original far-field multi-meter CBF-staged approach. See
     README_SINGLE_NUDGE.md for the closed-form geometry used.

Everything else (ContactImpulseAccumulator, NudgeLimits, NudgeSupervisor,
SuccessiveNudgeSession, MultiNudgeInertialIdentifier) is the project's
existing, unmodified code.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import mujoco

from src.nudge_experiment import (
    ContactImpulseAccumulator,
    ContactImpulseMeasurement,
    NudgeLimits,
    NudgeSupervisor,
    SuccessiveNudgeSession,
    TargetMotionSnapshot,
)

from src.inertial_identifier import (
    IdentificationResult,
    MultiNudgeInertialIdentifier,
)

from src.pose_sensor import (
    GroundTruthTargetState,
    NoisyPoseSensor,
    PoseSensorConfig,
    TargetStateEstimate,
    average_estimates,
    true_target_state,
)

from src.ellipsoidal_cbf_self_collision import (
    EllipsoidalCBFSafetyFilter,
    CBFInfeasibleError,
    SafetyViolation,
)


# --------------------------------------------------------------------------- #
# Small math helpers (ported verbatim from the project's proven v18 contact
# controller; see pose_stabilization_contact_debug_v18.ipynb cell 15).
# --------------------------------------------------------------------------- #
def _normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize a near-zero vector")
    return v / n


def _clip_norm(v, max_norm):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n <= max_norm or n < 1e-12:
        return v
    return v * (max_norm / n)


def _desired_rotation_from_probe_x(push_direction_world):
    """Construct R_des with the probe's local +x aligned to the push direction."""
    x = _normalize(push_direction_world)
    up = np.array([0.0, 0.0, 1.0])
    if abs(x @ up) > 0.90:
        up = np.array([0.0, 1.0, 0.0])
    y = _normalize(np.cross(up, x))
    z = _normalize(np.cross(x, y))
    return np.column_stack((x, y, z))


def _orientation_error_world(R_current, R_desired):
    """Small-angle orientation error expressed in world coordinates."""
    return 0.5 * (
        np.cross(R_current[:, 0], R_desired[:, 0])
        + np.cross(R_current[:, 1], R_desired[:, 1])
        + np.cross(R_current[:, 2], R_desired[:, 2])
    )


def _moment_rich_push_direction(surface_normal_body, tangent_hint_body, tangential_fraction):
    """Inward push with a controlled tangential component (ported from the
    project's NUDGE_PLANS candidate-geometry helper, unchanged)."""
    n = _normalize(surface_normal_body)
    h = np.asarray(tangent_hint_body, dtype=float).reshape(3)
    tangent = h - float(h @ n) * n
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm < 1e-10:
        raise ValueError("Tangent hint is parallel to the surface normal.")
    tangent = tangent / tangent_norm
    tf = float(tangential_fraction)
    nf = float(np.sqrt(max(0.0, 1.0 - tf ** 2)))
    return _normalize(-nf * n + tf * tangent)   # -n is inward


def _quat_align_x_to(direction_world):
    """Quaternion (w,x,y,z) rotating the world +x axis onto `direction_world`."""
    d = _normalize(direction_world)
    x = np.array([1.0, 0.0, 0.0])
    c = float(x @ d)
    if c > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1.0 + 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = _normalize(np.cross(x, d))
    angle = float(np.arccos(np.clip(c, -1.0, 1.0)))
    return np.array([np.cos(angle / 2.0), *(axis * np.sin(angle / 2.0))])


# Total reach of the arm+wrist+probe chain when every hinge sits at its
# simple, well-inside-limits "straight" configuration (joint1=0,
# joint2=-90deg, joint3=+90deg; every link then points along the
# spacecraft's own local +x axis). Derived directly from model.xml's link
# lengths: 0.68 (mount) + 0.70 (link1) + 0.60 (link2) + 0.45 (link3)
# + 0.20 (probe body offset) + 0.215 (nudge_tip site offset).
ARM_STRAIGHT_REACH_M = 0.68 + 0.70 + 0.60 + 0.45 + 0.20 + 0.215
STRAIGHT_JOINTS = (0.0, -np.pi / 2.0, np.pi / 2.0)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentConfig:
    model_path: str = "model.xml"

    # --- excitation (same numbers as the project's own "v18 CONTACT-DEBUG
    # CONFIGURATION": single nudge, 5 N s, 3 N nominal, 1 s min dwell) ---
    desired_impulse_Ns: float = 5.0
    contact_force_target_N: float = 3.0
    tangential_fraction: float = 0.30
    max_impulse_Ns: float = 15.0
    max_force_N: float = 7.5
    max_torque_Nm: float = 8.0
    min_contact_time_s: float = 1.0
    max_contact_time_s: float = 6.0

    # --- assumed target envelope (perception/mission input, NOT read from
    # the mesh: a conservative pre-mission size estimate) ---
    assumed_target_radius_m: float = 0.50

    # --- contact geometry (single fixed candidate; body-frame, mirrors the
    # project's own "candidate A" style NUDGE_PLANS geometry) ---
    surface_normal_body: Tuple[float, float, float] = (-1.00, 0.55, 0.20)
    tangent_hint_body: Tuple[float, float, float] = (0.0, 0.0, 1.0)

    # --- close-range start geometry ---
    precontact_standoff_m: float = 0.20
    close_range_extra_backoff_m: float = 0.50   # -> ~0.7 m total controlled approach
    max_search_past_surface_m: float = 0.12     # no camera refinement -> generous margin
    # Whole-servicer post-contact retreat before the long observation window.

    # --- approach/contact speeds & hybrid force regulation (ported from the
    # v18 contact-force fix; see V18_CONTACT_FIX_README.md) ---
    fast_approach_speed_mps: float = 0.015
    slow_approach_speed_mps: float = 0.003
    slow_zone_m: float = 0.02
    force_admittance_mps_per_N: float = 0.0015
    force_proportional_gain: float = 0.60
    force_command_max_N: float = 7.5
    max_push_speed_mps: float = 0.008
    max_backoff_speed_mps: float = 0.010
    release_speed_mps: float = 0.010
    release_trigger_fraction: float = 0.90
    acquire_controller_force_cap_N: float = 2.0
    regulate_controller_force_cap_N: float = 9.0
    release_controller_force_cap_N: float = 2.0
    contact_base_force_reg_weight: float = 3.0
    contact_base_torque_reg_weight: float = 8.0
    required_clear_steps: int = 15

    # --- timing guards (generous fixed budgets; no camera-derived dynamic
    # acquisition-time estimate in this simplified variant). With the
    # default ~0.7 m approach distance, 0.015 m/s fast-approach speed and
    # 0.003 m/s slow-zone speed, nominal acquisition takes roughly
    # (0.68 m)/(0.015 m/s) + (~0.07 m)/(0.003 m/s) =~ 69 s; 150 s leaves
    # over 2x margin. This is a one-time cost at the start of an experiment
    # whose post-nudge window is itself >1000 s, so a generous budget here
    # costs nothing. ---
    acquisition_timeout_s: float = 150.0
    post_contact_timeout_s: float = 15.0

    # --- pose sensor (replaces vision.py) ---
    position_noise_std_m: float = 0.010
    velocity_noise_std_mps: float = 0.0008
    attitude_noise_std_rad: float = np.deg2rad(0.30)
    omega_noise_std_radps: float = np.deg2rad(0.05)

    # --- observation windows ---
    pre_nudge_observation_s: float = 60.0
    pre_nudge_sample_period_s: float = 2.0
    post_nudge_observation_s: float = 1200.0
    post_nudge_sample_period_s: float = 2.0

    # --- F/T sensor noise on the measured impulse itself (0 = ideal sensor) ---
    force_noise_std_N: float = 0.0
    torque_noise_std_Nm: float = 0.0

    # --- light rate damping applied to the spacecraft base ONLY during the
    # long free-flight observation, so ~20 minutes of zero control does not
    # let a small residual release velocity drift the servicer noticeably.
    # This is NOT target-relative collision avoidance -- just damping the
    # servicer's own inertial rate toward zero, the same way any real
    # spacecraft would fly during a long coast/observation phase. ---
    station_keep_rate_gain: float = 2.0

    # --- ellipsoidal SERVICER SELF-COLLISION CBF ---
    use_ellipsoidal_cbf: bool = True
    cbf_gamma: float = 0.20
    cbf_fd_fraction: float = 2e-3

    # The virtual ellipsoids already inflate the physical geometry.  This is
    # an additional required gap BETWEEN those ellipsoids.
    cbf_safety_buffer_m: float = 0.005

    # Only run the expensive HOCBF solve when a monitored pair is near this
    # clearance band, or is closing fast enough to reach it within the time
    # horizon below.
    cbf_activation_distance_m: float = 0.20
    cbf_activation_time_horizon_s: float = 5.0
    cbf_activation_psi_margin_mps: float = 0.05

    rng_seed: int = 0

    def __post_init__(self):
        if self.desired_impulse_Ns <= 0.0 or self.desired_impulse_Ns >= self.max_impulse_Ns:
            raise ValueError("desired_impulse_Ns must be positive and below max_impulse_Ns")
        if not (0.0 < self.contact_force_target_N < self.max_force_N):
            raise ValueError("contact_force_target_N must be positive and below max_force_N")
        if not (0.0 <= self.tangential_fraction < 1.0):
            raise ValueError("tangential_fraction must be in [0, 1)")
        if self.pre_nudge_observation_s <= 0.0 or self.post_nudge_observation_s <= 0.0:
            raise ValueError("observation durations must be positive")
        if self.pre_nudge_sample_period_s <= 0.0 or self.post_nudge_sample_period_s <= 0.0:
            raise ValueError("sample periods must be positive")
        if not (0.0 < self.cbf_gamma <= 1.0):
            raise ValueError("cbf_gamma must lie in (0, 1]")
        if self.cbf_fd_fraction <= 0.0:
            raise ValueError("cbf_fd_fraction must be positive")
        if self.cbf_safety_buffer_m < 0.0:
            raise ValueError("cbf_safety_buffer_m must be nonnegative")
        if self.cbf_activation_distance_m <= 0.0:
            raise ValueError("cbf_activation_distance_m must be positive")
        if self.cbf_activation_time_horizon_s <= 0.0:
            raise ValueError("cbf_activation_time_horizon_s must be positive")
        if self.cbf_activation_psi_margin_mps < 0.0:
            raise ValueError("cbf_activation_psi_margin_mps must be nonnegative")


@dataclass
class NudgeResult:
    measurement: ContactImpulseMeasurement
    pre_snapshot: TargetMotionSnapshot
    com_contact_estimate_m: np.ndarray
    force_time_history_s: np.ndarray
    force_push_history_N: np.ndarray
    force_command_history_N: np.ndarray
    compression_history_m: np.ndarray


@dataclass
class ObservationLog:
    times_s: np.ndarray
    estimates: List[TargetStateEstimate]
    truths: List[GroundTruthTargetState]  # DEV ONLY -- never fed to the identifier


# --------------------------------------------------------------------------- #
# Main experiment class
# --------------------------------------------------------------------------- #
class SingleNudgeExperiment:
    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        self.model = mujoco.MjModel.from_xml_path(config.model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.dt = float(self.model.opt.timestep)

        self.TARGET_BODY_ID = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "asteroid")
        self.TARGET_GEOM_ID = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "asteroid_geom")
        self.TIP_SITE_ID = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "nudge_tip")
        self.TIP_GEOM_ID = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "nudge_pad_geom")
        self.BASE_JOINT_ID = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "spacecraft_free")
        self.BASE_BODY_ID = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "spacecraft")
        self.BASE_DOF_ADR = int(self.model.jnt_dofadr[self.BASE_JOINT_ID])
        self.BASE_QPOS_ADR = int(self.model.jnt_qposadr[self.BASE_JOINT_ID])
        ids = [self.TARGET_BODY_ID, self.TARGET_GEOM_ID, self.TIP_SITE_ID, self.TIP_GEOM_ID,
               self.BASE_JOINT_ID, self.BASE_BODY_ID]
        if min(ids) < 0:
            raise RuntimeError(
                "model.xml is missing an expected name (asteroid/asteroid_geom/"
                "nudge_tip/nudge_pad_geom/spacecraft_free/spacecraft). Did you "
                "edit model.xml in a way that renamed one of these?"
            )

        self.ACTUATOR_NAMES = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]
        self.ACTUATOR_ID = {n: i for i, n in enumerate(self.ACTUATOR_NAMES) if n is not None}
        self.ACTUATOR_REG_W = self._actuator_regularization_weights()
        self.CONTACT_ACTUATOR_REG_W = self._contact_actuator_regularization_weights()

        # --------------------------------------------------------------
        # Ellipsoidal SERVICER SELF-COLLISION CBF.
        #
        # IMPORTANT:
        #   * NO asteroid geometry appears in these constraints.
        #   * Directly adjacent link/link pairs are intentionally excluded
        #     because their conservative envelopes overlap at the joints.
        #   * link1 <-> bus uses special non-overlapping fixed proxies.
        # --------------------------------------------------------------
        self.cbf = None

        self.SELF_COLLISION_PAIRS = [
            # Link 1 folding back into the spacecraft bus:
            ("cbf_bus_front", "cbf_link1_distal"),

            # Distal arm / tool versus spacecraft bus:
            ("cbf_bus", "cbf_link2"),
            ("cbf_bus", "cbf_link3"),
            ("cbf_bus", "cbf_end_effector"),
            ("cbf_bus", "cbf_probe_shaft"),

            # Non-adjacent arm self-collision:
            ("cbf_link1", "cbf_link3"),
            ("cbf_link1", "cbf_end_effector"),
            ("cbf_link1", "cbf_probe_shaft"),
            ("cbf_link2", "cbf_end_effector"),
            ("cbf_link2", "cbf_probe_shaft"),
            ("cbf_link3", "cbf_probe_shaft"),
        ]

        if config.use_ellipsoidal_cbf:
            cbf_control_weight = np.ones(
                self.model.nu,
                dtype=float,
            )

            # Base translation/rotation does not meaningfully change internal
            # servicer separation, so strongly prefer joint/wrist corrections
            # if the CBF has to intervene.
            for i, name in enumerate(
                self.ACTUATOR_NAMES
            ):
                if name is None:
                    continue
                if name.startswith("wrist_"):
                    cbf_control_weight[i] = 1.5
                elif name.startswith("base_force"):
                    cbf_control_weight[i] = 50.0
                elif name.startswith("base_torque"):
                    cbf_control_weight[i] = 100.0

            self.cbf = EllipsoidalCBFSafetyFilter(
                model=self.model,
                collision_pairs=self.SELF_COLLISION_PAIRS,
                gamma=config.cbf_gamma,
                fd_fraction=config.cbf_fd_fraction,
                control_weight=cbf_control_weight,
                safety_buffer_m=config.cbf_safety_buffer_m,
                activation_distance_m=(
                    config.cbf_activation_distance_m
                ),
                activation_time_horizon_s=(
                    config.cbf_activation_time_horizon_s
                ),
                activation_psi_margin_mps=(
                    config.cbf_activation_psi_margin_mps
                ),
            )

        self.rng = np.random.default_rng(config.rng_seed)
        self.pose_sensor = NoisyPoseSensor(
            PoseSensorConfig(
                position_noise_std_m=config.position_noise_std_m,
                velocity_noise_std_mps=config.velocity_noise_std_mps,
                attitude_noise_std_rad=config.attitude_noise_std_rad,
                omega_noise_std_radps=config.omega_noise_std_radps,
            ),
            rng=self.rng,
        )
        self.impulse_accumulator = ContactImpulseAccumulator(
            self.model,
            pad_geom_name="nudge_pad_geom",
            target_geom_name="asteroid_geom",
            force_noise_std_N=config.force_noise_std_N,
            torque_noise_std_Nm=config.torque_noise_std_Nm,
            rng_seed=config.rng_seed,
        )

        n_world = _normalize(config.surface_normal_body)
        self.push_direction_body = _moment_rich_push_direction(
            n_world, config.tangent_hint_body, config.tangential_fraction
        )
        if float(n_world @ self.push_direction_body) >= -0.15:
            raise ValueError("push direction is not sufficiently inward relative to the surface")
        self.surface_normal_body = n_world

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _actuator_regularization_weights(self):
        w = np.ones(self.model.nu)
        for i, name in enumerate(self.ACTUATOR_NAMES):
            if name is None:
                continue
            if name.startswith("base_force"):
                w[i] = 80.0
            elif name.startswith("base_torque"):
                w[i] = 20.0
            elif name.startswith("wrist_"):
                w[i] = 2.0
            elif name.startswith("motor_joint"):
                w[i] = 1.0
        return w

    def _contact_actuator_regularization_weights(self):
        w = self.ACTUATOR_REG_W.copy()
        for i, name in enumerate(self.ACTUATOR_NAMES):
            if name is None:
                continue
            if name.startswith("base_force"):
                w[i] = self.cfg.contact_base_force_reg_weight
            elif name.startswith("base_torque"):
                w[i] = self.cfg.contact_base_torque_reg_weight
        return w

    def reset_close_range(self):
        """Place the spacecraft/arm near the target, skipping the long-range
        staged approach entirely. See README_SINGLE_NUDGE.md for the
        closed-form geometry (also reproduced in fk_check.py) used here.
        """
        mujoco.mj_resetData(self.model, self.data)

        target_center = np.array([3.0, 0.0, 0.0])  # matches model.xml's asteroid pos
        contact_point = target_center + self.cfg.assumed_target_radius_m * self.surface_normal_body
        push_world = self.push_direction_body.copy()  # target R = I at reset

        standoff_back_m = self.cfg.precontact_standoff_m + self.cfg.close_range_extra_backoff_m
        desired_tip_pos = contact_point - standoff_back_m * push_world

        sc_quat = _quat_align_x_to(push_world)
        sc_pos = desired_tip_pos - ARM_STRAIGHT_REACH_M * push_world

        self.data.qpos[self.BASE_QPOS_ADR:self.BASE_QPOS_ADR + 3] = sc_pos
        self.data.qpos[self.BASE_QPOS_ADR + 3:self.BASE_QPOS_ADR + 7] = sc_quat

        j1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint1")
        j2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint2")
        j3_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint3")
        self.data.qpos[self.model.jnt_qposadr[j1_id]] = STRAIGHT_JOINTS[0]
        self.data.qpos[self.model.jnt_qposadr[j2_id]] = STRAIGHT_JOINTS[1]
        self.data.qpos[self.model.jnt_qposadr[j3_id]] = STRAIGHT_JOINTS[2]

        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        if self.cbf is not None:
            self.cbf.assert_safe(self.data)

        tip_pos, _ = self.tip_pose()
        gap = float(self.push_direction_body @ (contact_point - tip_pos))
        return {
            "target_center": target_center,
            "contact_point": contact_point,
            "precontact_point": contact_point - self.cfg.precontact_standoff_m * push_world,
            "push_direction_world": push_world,
            "tip_pos": tip_pos,
            "gap_to_assumed_contact_point_m": gap,
        }

    # ------------------------------------------------------------------
    # Geometry / control
    # ------------------------------------------------------------------
    def tip_pose(self):
        mujoco.mj_forward(self.model, self.data)
        p = self.data.site_xpos[self.TIP_SITE_ID].copy()
        R = self.data.site_xmat[self.TIP_SITE_ID].reshape(3, 3).copy()
        return p, R

    def target_truth(self) -> GroundTruthTargetState:
        return true_target_state(self.model, self.data, self.TARGET_BODY_ID)

    def target_estimate(self) -> TargetStateEstimate:
        return self.pose_sensor.sense(self.target_truth())

    def current_probe_target_wrench(self):
        F, M, p = self.impulse_accumulator.instantaneous_net_wrench_about_origin(self.data)
        return np.asarray(F, dtype=float), np.asarray(M, dtype=float), p

    def has_probe_tip_target_contact(self) -> bool:
        for cid in range(self.data.ncon):
            c = self.data.contact[cid]
            if {int(c.geom[0]), int(c.geom[1])} == {self.TIP_GEOM_ID, self.TARGET_GEOM_ID}:
                return True
        return False

    def probe_compression_m(self) -> float:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "nudge_probe_slide")
        if jid < 0:
            return float("nan")
        qadr = int(self.model.jnt_qposadr[jid])
        return max(0.0, -float(self.data.qpos[qadr]))

    def _actuator_moment_dense(self):
        raw_moment = np.asarray(self.data.actuator_moment)
        if raw_moment.ndim == 2:
            return raw_moment.copy()
        nout = int(getattr(self.model, "nout", self.model.nu))
        dense = np.zeros((nout, self.model.nv), dtype=float)
        mujoco.mju_sparse2dense(
            dense, self.data.actuator_moment, self.data.moment_rownnz,
            self.data.moment_rowadr, self.data.moment_colind,
        )
        return dense

    def operational_space_velocity_control(
        self, desired_velocity_world, target_R_world,
        kv_pos=60.0, kp_rot=14.0, kd_rot=5.0,
        max_cart_force_N=1.5, max_cart_torque_Nm=4.0,
        actuator_regularization=2e-3,
        feedforward_force_world=None,
        regularization_weights=None,
    ):
        """Low-speed Cartesian controller used near/contacting the target.
        Ported from the project's proven v18 contact controller
        (operational_space_velocity_control), unchanged in structure:
        F_cmd = kv_pos*(v_des - v) + F_ff, then mapped to actuators by
        regularized least squares.  The returned nominal command is then
        passed through the servicer self-collision HOCBF in execute_step().
        """
        mujoco.mj_forward(self.model, self.data)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.TIP_SITE_ID)

        _, R = self.tip_pose()
        v = jacp @ self.data.qvel
        omega = jacr @ self.data.qvel

        v_des = np.asarray(desired_velocity_world, dtype=float).reshape(3)
        e_R = _orientation_error_world(R, np.asarray(target_R_world))

        F_ff = np.zeros(3) if feedforward_force_world is None else np.asarray(
            feedforward_force_world, dtype=float
        ).reshape(3)

        F_vel = kv_pos * (v_des - v)
        F = _clip_norm(F_vel + F_ff, max_cart_force_N)
        M = _clip_norm(kp_rot * e_R - kd_rot * omega, max_cart_torque_Nm)
        tau_des = jacp.T @ F + jacr.T @ M

        actuator_moment_dense = self._actuator_moment_dense()
        if actuator_moment_dense.shape != (self.model.nu, self.model.nv):
            raise RuntimeError(
                f"Unexpected actuator transmission shape: {actuator_moment_dense.shape}; "
                f"expected ({self.model.nu}, {self.model.nv})."
            )
        B = actuator_moment_dense.T
        reg_w = self.ACTUATOR_REG_W if regularization_weights is None else np.asarray(
            regularization_weights, dtype=float
        ).reshape(self.model.nu)
        if np.any(reg_w <= 0.0):
            raise ValueError("regularization_weights must be strictly positive.")

        H = B.T @ B + actuator_regularization * np.diag(reg_w)
        g = B.T @ tau_des
        try:
            u = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            u = np.linalg.lstsq(H, g, rcond=None)[0]

        v_base = self.data.qvel[self.BASE_DOF_ADR:self.BASE_DOF_ADR + 3]
        w_base = self.data.qvel[self.BASE_DOF_ADR + 3:self.BASE_DOF_ADR + 6]
        for axis, suffix in enumerate(("x", "y", "z")):
            aid = self.ACTUATOR_ID.get(f"base_force_{suffix}")
            if aid is not None:
                u[aid] += -1.0 * v_base[axis]
            aid = self.ACTUATOR_ID.get(f"base_torque_{suffix}")
            if aid is not None:
                u[aid] += -0.5 * w_base[axis]

        limited = np.asarray(self.model.actuator_ctrllimited, dtype=bool)
        u[limited] = np.clip(
            u[limited], self.model.actuator_ctrlrange[limited, 0],
            self.model.actuator_ctrlrange[limited, 1],
        )
        return u

    def _station_keep_control(self):
        """Damp the free-floating base's own rate toward zero using only the
        base force/torque actuators. No target-relative term at all -- this
        is NOT collision avoidance, just basic rate damping so a long
        (>1000 s) idle observation window does not let residual momentum
        carry the servicer meaningfully across the ~3-4 m gap.
        """
        u = np.zeros(self.model.nu)
        v_base = self.data.qvel[self.BASE_DOF_ADR:self.BASE_DOF_ADR + 3]
        w_base = self.data.qvel[self.BASE_DOF_ADR + 3:self.BASE_DOF_ADR + 6]
        k = self.cfg.station_keep_rate_gain
        for axis, suffix in enumerate(("x", "y", "z")):
            aid = self.ACTUATOR_ID.get(f"base_force_{suffix}")
            if aid is not None:
                u[aid] = -k * v_base[axis]
            aid = self.ACTUATOR_ID.get(f"base_torque_{suffix}")
            if aid is not None:
                u[aid] = -k * w_base[axis]
        limited = np.asarray(self.model.actuator_ctrllimited, dtype=bool)
        u[limited] = np.clip(
            u[limited], self.model.actuator_ctrlrange[limited, 0],
            self.model.actuator_ctrlrange[limited, 1],
        )
        return u

    def execute_step(self, u_nom, accumulate_impulse=False):
        """
        Filter the nominal actuator command through the ellipsoidal
        SERVICER SELF-COLLISION CBF, then step MuJoCo once.

        The asteroid is not part of this CBF.  Intentional nudge-pad/target
        contact remains governed only by the separate contact supervisor.
        """
        u_applied = np.asarray(u_nom, dtype=float).copy()

        if self.cbf is not None:
            cbf_result = self.cbf.filter_control(
                data=self.data,
                nominal_control=u_applied,
            )
            u_applied = cbf_result.control

        self.data.ctrl[:] = u_applied
        mujoco.mj_step(self.model, self.data)

        if accumulate_impulse:
            self.impulse_accumulator.sample(self.data)

        mujoco.mj_forward(self.model, self.data)
        return u_applied

    def print_cbf_status(self):
        """Print current clearance for every monitored self-collision pair."""
        if self.cbf is None:
            print("Ellipsoidal self-collision CBF disabled.")
            return

        barriers = sorted(
            self.cbf.barriers(self.data),
            key=lambda b: b.h_m,
        )

        print("Ellipsoidal self-collision CBF clearances:")
        for barrier in barriers:
            print(
                f"  {barrier.geom_name:48s} "
                f"h = {barrier.h_m:+.6f} m"
            )

    # ------------------------------------------------------------------
    # Pre/post observation windows
    # ------------------------------------------------------------------
    def _idle_step(self, control_fn=None):
        u = self._station_keep_control() if control_fn is None else control_fn()
        self.execute_step(u, accumulate_impulse=False)

    def observe_and_average(self, duration_s: float, sample_period_s: float,
                             label: str = "") -> Tuple[TargetStateEstimate, ObservationLog]:
        """Station-keep while repeatedly sampling the noisy pose sensor, and
        return a low-noise AVERAGED estimate. A single fixed noisy sample
        would dominate downstream error no matter how many post-nudge
        samples are later collected (its noise never gets to average out),
        so this is used to build a solid pre-nudge baseline -- see
        validate_single_nudge_idea.py for the empirical before/after.
        """
        n_steps = int(round(duration_s / self.dt))
        sample_every = max(1, int(round(sample_period_s / self.dt)))
        estimates, truths, times = [], [], []
        for step in range(n_steps + 1):
            if step % sample_every == 0:
                truths.append(self.target_truth())
                estimates.append(self.target_estimate())
                times.append(float(self.data.time))
            if step == n_steps:
                break
            self._idle_step()
        log = ObservationLog(times_s=np.asarray(times), estimates=estimates, truths=truths)
        return average_estimates(estimates), log

    def observe_free_flight(
        self,
        duration_s: float,
        sample_period_s: float,
        progress_every_s: Optional[float] = 60.0,
    ) -> ObservationLog:
        """
        Long post-nudge free-flight observation.

        The target remains completely uncontrolled/free-floating.

        The servicer applies only light inertial-rate damping through
        _station_keep_control().  There is NO target-relative keep-away
        controller in this version.

        Every commanded step still passes through the internal ellipsoidal
        SELF-COLLISION HOCBF, which monitors only the configured bus/link and
        link/link pairs.
        """
        duration_s = float(duration_s)
        sample_period_s = float(
            sample_period_s
        )

        n_steps = int(
            round(duration_s / self.dt)
        )
        sample_every = max(
            1,
            int(
                round(
                    sample_period_s
                    / self.dt
                )
            ),
        )

        estimates = []
        truths = []
        times = []

        observation_start_time = float(
            self.data.time
        )
        last_progress_elapsed_s = -np.inf

        for step in range(n_steps + 1):
            t_now = float(
                self.data.time
            )
            elapsed_s = (
                t_now
                - observation_start_time
            )

            if step % sample_every == 0:
                truths.append(
                    self.target_truth()
                )
                estimates.append(
                    self.target_estimate()
                )
                times.append(t_now)

            if (
                progress_every_s is not None
                and (
                    elapsed_s
                    - last_progress_elapsed_s
                    >= float(
                        progress_every_s
                    )
                    or step == 0
                )
            ):
                if self.cbf is not None:
                    barriers = self.cbf.barriers(
                        self.data
                    )
                    worst = min(
                        barriers,
                        key=lambda b: b.h_m,
                    )
                    cbf_text = (
                        f"min_self_h="
                        f"{worst.h_m:+.3f} m "
                        f"[{worst.geom_name}]"
                    )
                else:
                    cbf_text = (
                        "self-CBF disabled"
                    )

                print(
                    f"  observing... "
                    f"elapsed={elapsed_s:8.1f} s / "
                    f"{duration_s:.1f} s "
                    f"({len(estimates)} samples, "
                    f"{cbf_text})"
                )

                last_progress_elapsed_s = (
                    elapsed_s
                )

            if step == n_steps:
                break

            try:
                self._idle_step()

            except (
                CBFInfeasibleError,
                SafetyViolation,
            ) as exc:
                if self.cbf is not None:
                    states = sorted(
                        self.cbf.barriers(
                            self.data
                        ),
                        key=lambda s: s.h_m,
                    )

                    detail = " | ".join(
                        f"{s.geom_name}: "
                        f"h={s.h_m:+.6f} m"
                        for s in states[:3]
                    )

                    exc_type = type(exc)
                    raise exc_type(
                        "Self-collision CBF failure "
                        "during free-flight observation "
                        f"at elapsed={elapsed_s:.3f} s. "
                        "Three smallest pairwise barriers: "
                        f"{detail}. Original error: {exc}"
                    ) from exc

                raise

        return ObservationLog(
            times_s=np.asarray(times),
            estimates=estimates,
            truths=truths,
        )

    # ------------------------------------------------------------------
    # The nudge itself: approach -> contact/regulate -> release
    # ------------------------------------------------------------------
    def run_nudge(self, verbose: bool = True) -> NudgeResult:
        cfg = self.cfg

        # 1. Pre-nudge observation & averaging (target is at rest, so this is
        #    a legitimate average of "the same" true state).
        if verbose:
            print(f"Pre-nudge observation: {cfg.pre_nudge_observation_s:.1f} s "
                  f"@ {cfg.pre_nudge_sample_period_s:.1f} s cadence")
        pre_avg, pre_log = self.observe_and_average(
            cfg.pre_nudge_observation_s, cfg.pre_nudge_sample_period_s, label="pre-nudge",
        )
        pre_snapshot = pre_avg.to_target_motion_snapshot()
        self.impulse_accumulator.reset(pre_avg.position_world_m)

        push_world = self.push_direction_body.copy()  # target still at rest / R~=I
        R_des = _desired_rotation_from_probe_x(push_world)
        contact_point = pre_avg.position_world_m + cfg.assumed_target_radius_m * (
            pre_avg.R_body_to_world @ self.surface_normal_body
        )
        precontact_point = contact_point - cfg.precontact_standoff_m * push_world

        supervisor = NudgeSupervisor(
            push_direction_world=push_world,
            limits=NudgeLimits(
                desired_impulse_Ns=cfg.desired_impulse_Ns,
                max_impulse_Ns=cfg.max_impulse_Ns,
                max_force_N=cfg.max_force_N,
                max_torque_Nm=cfg.max_torque_Nm,
                min_contact_time_s=cfg.min_contact_time_s,
                max_contact_time_s=cfg.max_contact_time_s,
            ),
        )
        release_trigger_Ns = cfg.release_trigger_fraction * cfg.desired_impulse_Ns

        if verbose:
            print(f"Contact point estimate: {contact_point}")
            print(f"Push direction (world): {push_world}")

        # 2. Approach -> acquire -> regulate -> release state machine.
        contact_seen = False
        releasing = False
        no_contact_count = 0
        phase_start_s = float(self.data.time)
        contact_first_seen_s = None
        last_report_s = -np.inf

        force_t, force_push, force_cmd, compression = [], [], [], []

        k = 0
        while True:
            t_now = float(self.data.time)
            tip_pos, _ = self.tip_pose()
            gap_to_contact_point = float(push_world @ (contact_point - tip_pos))

            F_now, _, _ = self.current_probe_target_wrench()
            f_push_now = max(0.0, float(push_world @ F_now))

            in_contact = self.has_probe_tip_target_contact()
            if in_contact and not contact_seen:
                contact_seen = True
                contact_first_seen_s = t_now
                if verbose:
                    print(f"  FIRST CONTACT at t={t_now - phase_start_s:.3f} s "
                          f"(gap_to_assumed_point={gap_to_contact_point:+.4f} m)")

            if (not contact_seen) and (t_now - phase_start_s) > cfg.acquisition_timeout_s:
                raise RuntimeError(
                    "Contact acquisition timed out "
                    f"(elapsed={t_now - phase_start_s:.1f} s, "
                    f"budget={cfg.acquisition_timeout_s:.1f} s, "
                    f"final gap={gap_to_contact_point:+.4f} m). "
                    "Consider increasing close_range_extra_backoff_m/standoff, "
                    "or check that model.xml's asteroid position still matches "
                    "the (3,0,0) assumption hard-coded in reset_close_range()."
                )
            if (contact_first_seen_s is not None
                    and t_now - contact_first_seen_s > cfg.post_contact_timeout_s):
                raise RuntimeError(
                    f"Post-contact regulation/release exceeded "
                    f"{cfg.post_contact_timeout_s:.1f} s wall time."
                )

            measurement = self.impulse_accumulator.measurement()
            projected_impulse = float(push_world @ measurement.linear_impulse_world_Ns)
            if (contact_seen
                    and measurement.contact_duration_s >= cfg.min_contact_time_s
                    and projected_impulse >= release_trigger_Ns):
                releasing = True

            feedforward_force_world = np.zeros(3)
            allocation_weights = self.ACTUATOR_REG_W
            last_force_command_N = 0.0

            if releasing:
                v_des = -cfg.release_speed_mps * push_world
                controller_force_cap = cfg.release_controller_force_cap_N
            elif not contact_seen:
                if gap_to_contact_point < -cfg.max_search_past_surface_m:
                    raise RuntimeError(
                        "No pad-target contact detected after the full permitted "
                        f"{cfg.max_search_past_surface_m:.3f} m planned-surface search "
                        f"(gap={gap_to_contact_point:+.4f} m). The assumed target "
                        "radius/contact-point may be too far from the true surface "
                        "in this direction; increase max_search_past_surface_m or "
                        "check the mesh."
                    )
                speed = (cfg.fast_approach_speed_mps if gap_to_contact_point > cfg.slow_zone_m
                         else cfg.slow_approach_speed_mps)
                v_des = speed * push_world
                controller_force_cap = cfg.acquire_controller_force_cap_N
            else:
                force_error = cfg.contact_force_target_N - f_push_now
                speed_cmd = float(np.clip(
                    cfg.force_admittance_mps_per_N * force_error,
                    -cfg.max_backoff_speed_mps, cfg.max_push_speed_mps,
                ))
                v_des = speed_cmd * push_world
                force_command_N = float(np.clip(
                    cfg.contact_force_target_N + cfg.force_proportional_gain * force_error,
                    0.0, cfg.force_command_max_N,
                ))
                feedforward_force_world = force_command_N * push_world
                last_force_command_N = force_command_N
                controller_force_cap = cfg.regulate_controller_force_cap_N
                allocation_weights = self.CONTACT_ACTUATOR_REG_W

            u_nom = self.operational_space_velocity_control(
                v_des, R_des, kv_pos=60.0, kp_rot=14.0, kd_rot=5.0,
                max_cart_force_N=controller_force_cap, max_cart_torque_Nm=4.0,
                feedforward_force_world=feedforward_force_world,
                regularization_weights=allocation_weights,
            )
            self.execute_step(u_nom, accumulate_impulse=True)

            measurement = self.impulse_accumulator.measurement()
            status = supervisor.evaluate(measurement)

            if verbose and measurement.contact_duration_s > 0.0 and (
                    measurement.contact_duration_s - last_report_s >= 0.25):
                mode = "release" if releasing else ("regulate" if contact_seen else "acquire")
                print(
                    f"  contact t={measurement.contact_duration_s:.3f} s, mode={mode}, "
                    f"F_push={f_push_now:+.3f} N, F_cmd={last_force_command_N:.3f} N, "
                    f"compression={1000.0 * self.probe_compression_m():.2f} mm, "
                    f"peak={measurement.peak_force_N:.3f} N, "
                    f"J_dir={status.projected_impulse_Ns:+.4f} N s"
                )
                last_report_s = measurement.contact_duration_s

            force_t.append(t_now)
            force_push.append(f_push_now)
            force_cmd.append(last_force_command_N)
            compression.append(self.probe_compression_m())

            if status.emergency_abort:
                self.data.ctrl[:] = 0.0
                mujoco.mj_forward(self.model, self.data)
                raise RuntimeError("Nudge supervisor emergency abort: " + status.reason)

            if self.has_probe_tip_target_contact():
                no_contact_count = 0
            elif releasing:
                no_contact_count += 1
                if no_contact_count >= cfg.required_clear_steps:
                    if verbose:
                        print(f"Contact released cleanly: peak={measurement.peak_force_N:.3f} N, "
                              f"|J|={measurement.impulse_magnitude_Ns:.4f} N s")
                    break

            if status.stop:
                releasing = True
            k += 1

        # Freeze the completed contact measurement immediately after clean
        # physical release.  No post-nudge target-separation retreat is
        # performed in this self-collision-only experiment.
        measurement = self.impulse_accumulator.measurement()

        # Estimate the contact-time target CoM immediately after release.
        com_draws = [
            self.target_estimate().position_world_m
            for _ in range(5)
        ]
        com_contact_estimate = np.mean(
            com_draws,
            axis=0,
        )

        # ------------------------------------------------------------------
        # No post-nudge whole-servicer retreat in the self-collision-only
        # experiment.
        #
        # A base translation retreat was previously added only to increase
        # spacecraft-to-asteroid separation during the long observation.
        # Target collision avoidance is deliberately OUT OF SCOPE here.
        #
        # Applying a base-only retreat can excite relative arm motion in the
        # free-flying servicer and is therefore counterproductive for this
        # self-collision study.  Once the nudge pad has physically released,
        # proceed directly to the free-flight observation.  The internal
        # self-collision HOCBF remains active on every subsequent servicer
        # control step.
        # ------------------------------------------------------------------

        if verbose:
            print(
                f"Measured total impulse: "
                f"{measurement.linear_impulse_world_Ns} N s "
                f"(|J|={measurement.impulse_magnitude_Ns:.4f} N s)"
            )
            print(
                f"Peak force = {measurement.peak_force_N:.3f} N "
                f"(hard cap {cfg.max_force_N:.1f} N)"
            )
            print(
                f"Contact duration = "
                f"{measurement.contact_duration_s:.3f} s"
            )

        return NudgeResult(
            measurement=measurement,
            pre_snapshot=pre_snapshot,
            com_contact_estimate_m=com_contact_estimate,
            force_time_history_s=np.asarray(force_t),
            force_push_history_N=np.asarray(force_push),
            force_command_history_N=np.asarray(force_cmd),
            compression_history_m=np.asarray(compression),
        )

    # ------------------------------------------------------------------
    # Identification
    # ------------------------------------------------------------------
    def identify(
        self, nudge_result: NudgeResult, post_log: ObservationLog,
    ) -> IdentificationResult:
        """Turn ONE physical nudge + MANY post-nudge free-flight samples into
        a full-rank mass + inertia estimate, by pairing the single pre-nudge
        snapshot with every post-nudge sample and feeding each pair to
        `SuccessiveNudgeSession.add_completed_nudge` as if it were its own
        nudge. All of them share the exact same measured impulse -- what
        varies is R(t_k)/omega(t_k) from the real free tumble.
        """
        session = SuccessiveNudgeSession(MultiNudgeInertialIdentifier())
        for est in post_log.estimates:
            post_snapshot = est.to_target_motion_snapshot()
            session.add_completed_nudge(
                nudge_result.measurement,
                nudge_result.pre_snapshot,
                post_snapshot,
                com_world_during_contact_m=nudge_result.com_contact_estimate_m,
                label=f"t={est.t_s:.1f}s",
            )
        return session.latest()
