from __future__ import annotations

"""
single_nudge_experiment.py
===========================
ONE gentle nudge, followed by a LONG free-flight observation window, with NO
onboard vision/camera and coupled ellipsoidal SERVICER SELF-COLLISION and ASTEROID KEEP-OUT CBFs.

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

  3. Coupled self- and asteroid-collision CBF: virtual ellipsoids enforce
     the 11 original bus/arm self-collision pairs, the bus–asteroid keep-out,
     and six arm-envelope-to-physical-asteroid-mesh barriers. Only the nudge
     contact pad is exempt, with separate force/impulse/duration supervision.

  4. Close-range start: `reset_close_range()` places the spacecraft/arm so
     that only a short (well under 1 m) final approach is needed, instead of
     the original far-field multi-meter CBF-staged approach. See
     README_SINGLE_NUDGE.md for the closed-form geometry used.

Everything else (ContactImpulseAccumulator, NudgeLimits, NudgeSupervisor,
SuccessiveNudgeSession, MultiNudgeInertialIdentifier) is the project's
existing, unmodified code.
"""

from dataclasses import dataclass, field
from pathlib import Path
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

from src.asteroid_keepout_cbf import (
    ASTEROID_KEEP_OUT_PAIR, planned_mesh_contact_point_body,
)
from src.asteroid_keepin_cbf import AsteroidKeepInCBFFilter

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
    # SIMULATION PLANNING ONLY. A known OBJ hull supplies the intended contact
    # point for reproducibility, but neither the estimator nor the HOCBF reads
    # asteroid truth inertia/mass. Replace this with relative-nav geometry for flight.
    use_mesh_contact_planning: bool = True
    approach_cross_track_gain_per_s: float = 0.60
    max_approach_lateral_speed_mps: float = 0.008

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
    position_noise_std_m: float = 1e-4
    velocity_noise_std_mps: float = 1e-4
    attitude_noise_std_rad: float = np.deg2rad(1e-4)
    omega_noise_std_radps: float = np.deg2rad(1e-4)

    # --- observation windows ---
    pre_nudge_observation_s: float = 60.0
    pre_nudge_sample_period_s: float = 2.0
    post_nudge_observation_s: float = 1200.0
    post_nudge_sample_period_s: float = 2.0

    # --- F/T sensor noise on the measured impulse itself (0 = ideal sensor) ---
    force_noise_std_N: float = 1e-4
    torque_noise_std_Nm: float = 1e-4

    # --- light rate damping applied to the spacecraft base ONLY during the
    # long free-flight observation, so ~20 minutes of zero control does not
    # let a small residual release velocity drift the servicer noticeably.
    # This is NOT target-relative collision avoidance -- just damping the
    # servicer's own inertial rate toward zero, the same way any real
    # spacecraft would fly during a long coast/observation phase. ---
    station_keep_rate_gain: float = 2.0  # fallback rate damping if pose hold disabled

    # Keep the free-floating spacecraft bus at a commanded inertial pose whenever
    # no maneuver is requested. The reference is captured at close-range reset
    # and RE-CAPTURED after the pad clears the asteroid. Thus the approach and
    # force-controlled nudge are unchanged; the post-nudge phase holds the
    # actual release pose, not the pre-contact position.
    # The XML's free-joint motor transmission uses WORLD linear force axes and
    # BODY-LOCAL torque axes. Angular qvel is also local to the spacecraft.
    base_pose_hold: bool = True
    base_position_kp_N_per_m: float = 0.20
    base_position_kd_Ns_per_m: float = 16.0
    base_attitude_kp_Nm_per_rad: float = 0.90
    base_attitude_kd_Nms_per_rad: float = 14.0

    # Post-nudge annular zone: 3.25 <= bus–asteroid <= 4.00 m.
    # Maximum separation is active at the close-range reset and remains active.
    # Minimum separation activates after controlled contact-free departure.
    # The transition buffer is NOT a preferred observation radius.
    use_asteroid_keepin_cbf: bool = True
    keepin_radius_m: float = 5.15    # XML sphere, yields 4.00 m maximum bus-center distance
    keepin_margin_m: float = 0.05
    inner_shell_radius_m: Optional[float] = 3.25
    shell_activation_buffer_m: float = 0.08
    min_pad_clearance_m: float = 0.02  # physical contact check, not a CBF
    shell_departure_max_speed_mps: float = 0.07
    shell_departure_velocity_gain: float = 100.0
    shell_departure_timeout_s: float = 240.0

    # Translational control during POST-NUDGE observation only. Track the
    # asteroid's translation and preserve the bus/asteroid offset obtained at
    # shell entry. This is NOT a selected preferred radius and does not alter
    # any CBF limit. The radial buffer starts nominal corrective thrust before
    # the hard actuator-limited CBF must intervene.
    relative_hold_kp_N_per_m: float = 8.0
    relative_hold_kd_Ns_per_m: float = 80.0
    outer_soft_buffer_m: float = 0.25
    inner_soft_buffer_m: float = 0.10
    radial_guard_kp_N_per_m: float = 24.0
    radial_guard_kd_Ns_per_m: float = 85.0

    # Optional active arm joint / ball-wrist hold ONLY after clean release.
    # Default is OFF for backwards compatibility. The contact controller
    # and nudge impulse logic are completely unchanged.
    post_release_joint_hold: bool = False
    post_release_joint_kp: Tuple[float, float, float] = (6.0, 6.0, 5.0)
    post_release_joint_kd: Tuple[float, float, float] = (5.0, 4.0, 3.0)
    post_release_wrist_kp: float = 1.5
    post_release_wrist_kd: float = 1.0

    # --- ellipsoidal SERVICER SELF-COLLISION CBF ---
    use_ellipsoidal_cbf: bool = True
    use_asteroid_keepout_cbf: bool = True  # combined with self-collision in one filter
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
        if self.use_asteroid_keepin_cbf and not self.use_asteroid_keepout_cbf:
            raise ValueError("Minimum separation requires the bus–asteroid keep-out CBF")
        if self.use_asteroid_keepin_cbf:
            if self.inner_shell_radius_m is None or self.inner_shell_radius_m <= 0.0:
                raise ValueError("inner_shell_radius_m must be positive")
            if self.keepin_radius_m <= 0 or self.keepin_margin_m < 0:
                raise ValueError("keep-in radius/margin must be valid")
            if self.min_pad_clearance_m < 0.0 or self.shell_activation_buffer_m <= 0.02:
                raise ValueError("Invalid physical clearance or activation buffer")
            if self.shell_departure_max_speed_mps <= 0.0 or self.shell_departure_timeout_s <= 0.0:
                raise ValueError("Departure speed/time must be positive")
            for name in ("relative_hold_kp_N_per_m", "relative_hold_kd_Ns_per_m",
                         "radial_guard_kp_N_per_m", "radial_guard_kd_Ns_per_m"):
                value = getattr(self, name)
                if not np.isfinite(value) or value <= 0.0:
                    raise ValueError(f"{name} must be finite and positive")
            if not (0.0 < self.outer_soft_buffer_m < 0.5 * (4.0 - self.inner_shell_radius_m)):
                raise ValueError("outer_soft_buffer_m must fit inside the radial shell")
            if not (0.0 < self.inner_soft_buffer_m < 0.5 * (4.0 - self.inner_shell_radius_m)):
                raise ValueError("inner_soft_buffer_m must fit inside the radial shell")
        if self.use_asteroid_keepout_cbf and not self.use_ellipsoidal_cbf:
            raise ValueError("Asteroid keep-out requires the shared ellipsoidal CBF filter")
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
        if self.approach_cross_track_gain_per_s < 0.0:
            raise ValueError("approach_cross_track_gain_per_s must be nonnegative")
        if self.max_approach_lateral_speed_mps < 0.0:
            raise ValueError("max_approach_lateral_speed_mps must be nonnegative")
        if self.cbf_activation_psi_margin_mps < 0.0:
            raise ValueError("cbf_activation_psi_margin_mps must be nonnegative")
        for name in (
            "base_position_kp_N_per_m", "base_position_kd_Ns_per_m",
            "base_attitude_kp_Nm_per_rad", "base_attitude_kd_Nms_per_rad",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


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

        # One coupled actuator-level HOCBF/QP. The bus uses the conservative
        # asteroid ellipsoid; all NON-CONTACT arm components use their existing
        # enclosing ellipsoids against the actual asteroid collision mesh.
        # Do not pair the pad with the asteroid: pad contact is intentional.
        self.ARM_ASTEROID_COLLISION_PAIRS = (
            [(name, "asteroid_geom") for name in (
                "cbf_arm_mount", "cbf_link1", "cbf_link2", "cbf_link3",
                "cbf_end_effector", "cbf_probe_shaft",
            )] if config.use_asteroid_keepout_cbf else []
        )
        self.ASTEROID_COLLISION_PAIRS = (
            [ASTEROID_KEEP_OUT_PAIR] if config.use_asteroid_keepout_cbf else []
        )
        self.ALL_CBF_PAIRS = (
            self.SELF_COLLISION_PAIRS
            + self.ASTEROID_COLLISION_PAIRS
            + self.ARM_ASTEROID_COLLISION_PAIRS
        )
        self.cbf_last_result = None
        self.cbf_interventions_total = 0
        self._post_release_joint_reference = None
        self._base_pose_reference = None
        self._keepin_relative_reference = None
        self._departure_direction = None
        self._shell_departure_needed = False
        if config.use_asteroid_keepout_cbf:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                    ASTEROID_KEEP_OUT_PAIR[1])
            if gid < 0:
                raise ValueError("Model is missing cbf_asteroid_keepout geom")
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
                    # Unlike arm self-collision, bus/target collision MUST be
                    # corrected primarily by whole-servicer translation.
                    cbf_control_weight[i] = (
                        8.0 if config.use_asteroid_keepout_cbf else 50.0
                    )
                elif name.startswith("base_torque"):
                    cbf_control_weight[i] = 100.0

            filter_class = (AsteroidKeepInCBFFilter
                            if config.use_asteroid_keepin_cbf
                            else EllipsoidalCBFSafetyFilter)
            keepin_kwargs = (
                dict(inner_shell_radius_m=config.inner_shell_radius_m,
                     min_pad_clearance_m=config.min_pad_clearance_m,
                     keepin_radius_m=config.keepin_radius_m,
                     keepin_margin_m=config.keepin_margin_m)
                if config.use_asteroid_keepin_cbf else {}
            )
            self.cbf = filter_class(
                **keepin_kwargs,
                model=self.model,
                collision_pairs=self.ALL_CBF_PAIRS,
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
        self.planned_contact_geometry = None
        self.planned_contact_body_m = (
            float(self.cfg.assumed_target_radius_m) * self.surface_normal_body
        )
        if self.cfg.use_mesh_contact_planning:
            obj_path = Path(self.cfg.model_path).resolve().parent / "asteroid.obj"
            self.planned_contact_geometry = planned_mesh_contact_point_body(
                obj_path,
                surface_normal_body=self.surface_normal_body,
                push_direction_body=self.push_direction_body,
                assumed_radius_m=self.cfg.assumed_target_radius_m,
            )
            self.planned_contact_body_m = np.asarray(
                self.planned_contact_geometry["contact_point_body_m"], dtype=float
            )

    def contact_plan_world(self, target_origin_world, target_R_body_to_world):
        """Preflight planned contact coordinate using the known target mesh.

        The target pose may come from the relative navigation estimate. The
        planning point remains fixed in the target body frame.
        """
        return (np.asarray(target_origin_world, dtype=float).reshape(3)
                + np.asarray(target_R_body_to_world, dtype=float).reshape(3,3)
                @ self.planned_contact_body_m)

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
        self._post_release_joint_reference = None
        self._base_pose_reference = None
        self._keepin_relative_reference = None
        self._departure_direction = None
        self._shell_departure_needed = False
        if self.cbf is not None and self.cfg.use_asteroid_keepin_cbf:
            self.cbf.keepin_active = False
            self.cbf.shell_active = False

        target_center = np.array([3.0, 0.0, 0.0])  # matches model.xml's asteroid pos
        contact_point = self.contact_plan_world(target_center, np.eye(3))
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
        if self.cfg.base_pose_hold:
            self._capture_base_pose_reference()

        if self.cbf is not None:
            if self.cfg.use_asteroid_keepin_cbf:
                self.cbf.activate(self.data)  # enforce maximum r <= 4.00 m from t=0
            self.cbf.assert_safe(self.data)

        tip_pos, _ = self.tip_pose()
        gap = float(self.push_direction_body @ (contact_point - tip_pos))
        return {
            "target_center": target_center,
            "contact_point": contact_point,
            "contact_point_body": self.planned_contact_body_m.copy(),
            "mesh_contact_plan": self.planned_contact_geometry,
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

    @staticmethod
    def _shortest_quaternion_error_rotvec(current_wxyz, target_wxyz):
        """Local-frame rotation vector taking current wrist quaternion to target.

        Quaternions are MuJoCo [w,x,y,z]. This avoids Euler-angle singularities
        and deals with q and -q representing the same attitude.
        """
        q = np.asarray(current_wxyz, dtype=float).reshape(4)
        d = np.asarray(target_wxyz, dtype=float).reshape(4)
        q = q / np.linalg.norm(q)
        d = d / np.linalg.norm(d)
        # q_error = conjugate(q_current) * q_target
        scalar = float(q[0] * d[0] + q[1:] @ d[1:])
        vec = q[0] * d[1:] - d[0] * q[1:] - np.cross(q[1:], d[1:])
        if scalar < 0.0:
            scalar, vec = -scalar, -vec
        vnorm = float(np.linalg.norm(vec))
        if vnorm < 1e-12:
            return 2.0 * vec
        angle = 2.0 * np.arctan2(vnorm, max(scalar, 0.0))
        return (angle / vnorm) * vec

    def _capture_post_release_joint_reference(self):
        """Freeze the *measured* release pose, after the pad has cleared."""
        ref = {}
        for name in ("joint1", "joint2", "joint3", "spherical_joint"):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"Joint hold is missing {name}")
            qadr = int(self.model.jnt_qposadr[jid])
            vadr = int(self.model.jnt_dofadr[jid])
            if name == "spherical_joint":
                ref[name] = (qadr, vadr, self.data.qpos[qadr:qadr + 4].copy())
            else:
                ref[name] = (qadr, vadr, float(self.data.qpos[qadr]))
        self._post_release_joint_reference = ref

    def _add_post_release_joint_hold(self, u):
        """Add actuator-limited PD arm torques to base rate damping.

        Called only during free-flight observation, never approach/contact.
        The returned nominal command is still filtered by execute_step's
        existing *combined* self-collision and asteroid CBF solver.
        """
        ref = self._post_release_joint_reference
        if ref is None or not self.cfg.post_release_joint_hold:
            return u
        u = np.asarray(u, dtype=float).copy()
        for i, name in enumerate(("joint1", "joint2", "joint3")):
            qadr, vadr, q_target = ref[name]
            error = q_target - float(self.data.qpos[qadr])
            velocity = float(self.data.qvel[vadr])
            torque = (self.cfg.post_release_joint_kp[i] * error
                      - self.cfg.post_release_joint_kd[i] * velocity)
            actuator_id = self.ACTUATOR_ID[f"motor_{name}"]
            u[actuator_id] = torque

        qadr, vadr, target_quat = ref["spherical_joint"]
        error_local = self._shortest_quaternion_error_rotvec(
            self.data.qpos[qadr:qadr + 4], target_quat
        )
        angular_rate_local = np.asarray(self.data.qvel[vadr:vadr + 3], dtype=float)
        wrist_torque = (self.cfg.post_release_wrist_kp * error_local
                        - self.cfg.post_release_wrist_kd * angular_rate_local)
        for axis, suffix in enumerate(("x", "y", "z")):
            u[self.ACTUATOR_ID[f"wrist_{suffix}"]] = wrist_torque[axis]
        return u

    def _capture_base_pose_reference(self):
        """Hold the spacecraft's current *inertial* position and attitude.

        Invoked once at reset and again after clean nudge release. Holding the
        RELEASE pose avoids commanding a large and unsafe return trajectory to
        the original approach position during the post-nudge observation.
        """
        adr = self.BASE_QPOS_ADR
        position = np.asarray(self.data.qpos[adr:adr + 3], dtype=float).copy()
        quaternion = np.asarray(self.data.qpos[adr + 3:adr + 7], dtype=float).copy()
        quaternion /= np.linalg.norm(quaternion)
        self._base_pose_reference = (position, quaternion)

    def _capture_keepin_relative_reference(self):
        """Record a departure DIRECTION, not an observation setpoint.

        The minimum radial CBF remains off during intentional contact and
        controlled withdrawal. The asteroid collision keep-out remains on.
        """
        if not self.cfg.use_asteroid_keepin_cbf:
            return
        mujoco.mj_forward(self.model, self.data)
        target = np.asarray(self.data.geom_xpos[self.cbf.target_gid]).copy()
        bus = np.asarray(self.data.geom_xpos[self.cbf.bus_gid]).copy()
        delta = bus - target
        sep = float(np.linalg.norm(delta))
        if sep <= 1e-10:
            raise RuntimeError("Cannot depart at zero bus–asteroid separation")
        self._departure_direction = delta / sep
        if self.cbf.shell_min_center_m + self.cfg.shell_activation_buffer_m >= self.cbf.shell_max_center_m - 0.05:
            raise ValueError("Not enough room for inner CBF activation inside outer CBF")
        if not self.cbf.keepin_active:
            self.cbf.activate(self.data)
        self._keepin_relative_reference = None
        self._shell_departure_needed = True

    def move_to_observation_shell(self, verbose=True):
        """Depart just beyond the minimum bus-center radius, then enable it.

        The activation buffer is solely a mode-transition guard. Upon
        activation, the observer holds the ACTUAL bus-to-asteroid displacement
        achieved at shell entry, not a preselected preferred radius.
        The inner radial CBF, outer maximum-distance CBF and asteroid
        keep-out protect the annular zone, subject to actuation feasibility.
        """
        if not self.cfg.use_asteroid_keepin_cbf or not self._shell_departure_needed:
            return 0.0
        if not self.cfg.base_pose_hold:
            raise RuntimeError("Controlled departure requires base_pose_hold=True")
        start = float(self.data.time)
        threshold = (self.cbf.shell_min_center_m
                     + self.cfg.shell_activation_buffer_m)
        n_steps = int(np.ceil(self.cfg.shell_departure_timeout_s / self.dt))
        for step in range(n_steps):
            mujoco.mj_forward(self.model, self.data)
            target = np.asarray(self.data.geom_xpos[self.cbf.target_gid], dtype=float)
            bus = np.asarray(self.data.geom_xpos[self.cbf.bus_gid], dtype=float)
            sep = float(np.linalg.norm(bus - target))
            pad_gap = self.cbf.pad_surface_clearance(self.data)
            if pad_gap < 0.0:
                raise RuntimeError("Physical probe–asteroid overlap during departure")
            if sep >= threshold and pad_gap >= self.cfg.min_pad_clearance_m + 0.005:
                self.cbf.activate_shell(self.data)
                # Preserve the ALREADY ATTAINED relative bus/asteroid offset.
                # The target will drift in inertial coordinates after contact;
                # inertial bus hold would reach the outer 4 m wall unnecessarily.
                # No preselected observation radius is introduced.
                self._keepin_relative_reference = (bus - target).copy()
                self._capture_base_pose_reference()  # attitude hold reference
                self._shell_departure_needed = False
                if verbose:
                    print(f"Minimum-separation CBF active after "
                          f"{self.data.time-start:.2f} s: "
                          f"bus separation={sep:.3f} m, "
                          f"probe gap={pad_gap:.3f} m, "
                          f"minimum CBF h={self.cbf.inner_clearance(self.data):.3f} m")
                return float(self.data.time - start)

            # Short, bounded departure maneuver to the activation threshold.
            # Threshold is NOT used as the observation-phase reference.
            desired_bus = target + self._departure_direction * threshold
            e_world = desired_bus - bus
            desired_relative_v = _clip_norm(0.14 * e_world,
                                           self.cfg.shell_departure_max_speed_mps)
            v_target = self._keepin_target_velocity()
            v_base = np.asarray(self.data.qvel[self.BASE_DOF_ADR:self.BASE_DOF_ADR + 3])
            u_nom = self._station_keep_control()
            f_world = (self.cfg.shell_departure_velocity_gain
                       * (v_target + desired_relative_v - v_base))
            for axis, suffix in enumerate(("x", "y", "z")):
                aid = self.ACTUATOR_ID.get(f"base_force_{suffix}")
                if aid is not None:
                    u_nom[aid] = f_world[axis]
            limited = np.asarray(self.model.actuator_ctrllimited, dtype=bool)
            u_nom[limited] = np.clip(u_nom[limited],
                                     self.model.actuator_ctrlrange[limited, 0],
                                     self.model.actuator_ctrlrange[limited, 1])
            self.execute_step(u_nom, accumulate_impulse=False)
            if self.cbf.outer_clearance(self.data) < -1e-6:
                raise SafetyViolation(
                    "Hard 4.00 m maximum separation exceeded during departure: "
                    f"r={self.cbf._centers(self.data)[2]:.6f} m")
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                if self.TARGET_GEOM_ID in (int(contact.geom[0]), int(contact.geom[1])):
                    raise RuntimeError(
                        f"Unexpected asteroid recontact during departure at "
                        f"t={self.data.time-start:.3f} s; identification aborted"
                    )
        raise RuntimeError(
            f"Departure timeout after {self.cfg.shell_departure_timeout_s:.1f} s: "
            f"center separation={sep:.3f} m, "
            f"probe gap={pad_gap:.3f} m; minimum CBF NOT activated"
        )

    def _keepin_target_velocity(self):
        """Return the asteroid reference-center velocity in world axes."""
        center = np.asarray(self.data.geom_xpos[self.cbf.target_gid], dtype=float)
        return self.cbf._point_velocity_world(self.data, self.TARGET_BODY_ID, center)

    def _station_keep_control(self):
        """Inertial bus pose hold + optional manipulator joint hold.

        This is called only when the servicer should be stationary (pre-nudge
        baseline and post-nudge observation). run_nudge() keeps using its
        original task-space control for approach, contact and release.

        MuJoCo free-joint linear qvel and gear axes are WORLD-frame quantities;
        rotational qvel and rotational free-joint gear axes are BODY-frame.
        Consequently, use world position/velocity and local quaternion error/
        angular rate, with no arbitrary mixing of frames.
        """
        u = np.zeros(self.model.nu)
        v_world = np.asarray(
            self.data.qvel[self.BASE_DOF_ADR:self.BASE_DOF_ADR + 3], dtype=float
        )
        omega_body = np.asarray(
            self.data.qvel[self.BASE_DOF_ADR + 3:self.BASE_DOF_ADR + 6], dtype=float
        )

        if self.cfg.base_pose_hold:
            if self._base_pose_reference is None:
                self._capture_base_pose_reference()
            desired_pos, desired_quat = self._base_pose_reference
            desired_vel_world = np.zeros(3)
            current_pos = np.asarray(
                self.data.qpos[self.BASE_QPOS_ADR:self.BASE_QPOS_ADR + 3],
                dtype=float,
            )
            current_quat = np.asarray(
                self.data.qpos[self.BASE_QPOS_ADR + 3:self.BASE_QPOS_ADR + 7],
                dtype=float,
            )
            error_pos_world = desired_pos - current_pos
            error_angle_body = self._shortest_quaternion_error_rotvec(
                current_quat, desired_quat
            )
            force_world = (
                self.cfg.base_position_kp_N_per_m * error_pos_world
                - self.cfg.base_position_kd_Ns_per_m * (v_world - desired_vel_world)
            )

            # After controlled departure, HOLD RELATIVE TRANSLATION rather
            # than a fixed inertial point. The asteroid receives an impulse
            # and drifts; holding an inertial point otherwise drives the
            # spacecraft toward the hard 4.00 m maximum-separation CBF.
            if (self.cfg.use_asteroid_keepin_cbf and self.cbf is not None
                    and self.cbf.shell_active
                    and self._keepin_relative_reference is not None):
                mujoco.mj_forward(self.model, self.data)
                bus = np.asarray(self.data.geom_xpos[self.cbf.bus_gid], dtype=float)
                asteroid = np.asarray(self.data.geom_xpos[self.cbf.target_gid], dtype=float)
                rel = bus - asteroid
                r = float(np.linalg.norm(rel))
                if r <= 1e-9:
                    raise SafetyViolation("Undefined asteroid-relative direction")
                n = rel / r
                bus_velocity = self.cbf._point_velocity_world(
                    self.data, self.cbf.bus_bid, bus)
                target_velocity = self._keepin_target_velocity()
                v_rel = bus_velocity - target_velocity
                radial_rate = float(n @ v_rel)

                # No new CBF or preferred radius: the reference is captured
                # from the actual safe departure state. Feed forward target
                # motion using relative velocity damping, and bias commands
                # inward BEFORE reaching the outer CBF boundary.
                rel_error = self._keepin_relative_reference - rel
                force_world = (
                    self.cfg.relative_hold_kp_N_per_m * rel_error
                    - self.cfg.relative_hold_kd_Ns_per_m * v_rel
                )
                outer_start = (self.cbf.shell_max_center_m
                               - self.cfg.outer_soft_buffer_m)
                inner_start = (self.cbf.shell_min_center_m
                               + self.cfg.inner_soft_buffer_m)
                if r > outer_start:
                    correction = (self.cfg.radial_guard_kp_N_per_m
                                  * (r - outer_start)
                                  + self.cfg.radial_guard_kd_Ns_per_m
                                  * max(radial_rate, 0.0))
                    force_world -= correction * n
                if r < inner_start:
                    correction = (self.cfg.radial_guard_kp_N_per_m
                                  * (inner_start - r)
                                  + self.cfg.radial_guard_kd_Ns_per_m
                                  * max(-radial_rate, 0.0))
                    force_world += correction * n

            torque_body = (
                self.cfg.base_attitude_kp_Nm_per_rad * error_angle_body
                - self.cfg.base_attitude_kd_Nms_per_rad * omega_body
            )
        else:
            k = self.cfg.station_keep_rate_gain
            force_world = -k * v_world
            torque_body = -k * omega_body

        for axis, suffix in enumerate(("x", "y", "z")):
            aid = self.ACTUATOR_ID.get(f"base_force_{suffix}")
            if aid is not None:
                u[aid] = force_world[axis]
            aid = self.ACTUATOR_ID.get(f"base_torque_{suffix}")
            if aid is not None:
                u[aid] = torque_body[axis]

        # Preserve the previously implemented post-release arm/wrist PD hold.
        u = self._add_post_release_joint_hold(u)
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

        Bus–asteroid keep-out and six arm–asteroid mesh barriers share the
        same CBF optimization. ONLY nudge-pad contact is intentional and
        supervised separately by force, impulse, and contact-duration limits.
        """
        u_applied = np.asarray(u_nom, dtype=float).copy()

        if self.cbf is not None:
            cbf_result = self.cbf.filter_control(
                data=self.data,
                nominal_control=u_applied,
            )
            u_applied = cbf_result.control
            self.cbf_last_result = cbf_result
            self.cbf_interventions_total += int(cbf_result.modified)

        self.data.ctrl[:] = u_applied
        mujoco.mj_step(self.model, self.data)

        if accumulate_impulse:
            self.impulse_accumulator.sample(self.data)

        mujoco.mj_forward(self.model, self.data)
        # Runtime fail-safe: only the pad is authorized to touch the target.
        # In case of CBF infeasibility / imperfect geometry, never silently
        # accept unintended physical arm/servicer contact as a valid nudge.
        if self.cfg.use_asteroid_keepout_cbf:
            for contact_index in range(self.data.ncon):
                contact = self.data.contact[contact_index]
                g1, g2 = int(contact.geom[0]), int(contact.geom[1])
                if self.TARGET_GEOM_ID not in (g1, g2):
                    continue
                other = g2 if g1 == self.TARGET_GEOM_ID else g1
                if other == self.TIP_GEOM_ID:
                    continue  # explicitly authorized contact point
                # MuJoCo can report proximity contacts at positive distances.
                # Abort for actual touching/penetration, not mere proximity.
                if float(contact.dist) <= 0.0:
                    other_name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, other
                    )
                    raise SafetyViolation(
                        f"Unintended asteroid contact: geom={other_name}, "
                        f"t={self.data.time:.3f} s, signed contact gap="
                        f"{float(contact.dist):+.6f} m. Only nudge_pad_geom "
                        "is allowed to contact the asteroid."
                    )
        return u_applied

    def print_cbf_status(self):
        """Print current clearance for all monitored self- and target pairs."""
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

        The servicer applies inertial base position/attitude holding through
        _station_keep_control() and optional post-release arm/wrist PD hold.
        The outer maximum-distance CBF is active from reset. After withdrawal,
        the inner minimum-distance CBF is also enforced. The nominal controller
        follows the asteroid's translation and preserves the attained relative
        offset (no preselected observation radius). Near the radial limits it
        adds bounded early corrective thrust; the hard 3.25/4.00 m CBFs remain
        the safety constraints and the actuator limits remain unchanged.

        Every commanded step still passes through the internal ellipsoidal
        SELF-COLLISION HOCBF, which monitors only the configured bus/link and
        link/link pairs.
        """
        if (self.cfg.use_asteroid_keepin_cbf
                and self._shell_departure_needed):
            raise RuntimeError(
                "Call move_to_observation_shell() after the nudge and "
                "before observe_free_flight()."
            )
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
                        f"min_h={worst.h_m:+.3f} m [{worst.geom_name}]"
                    )
                    if self.cfg.use_asteroid_keepin_cbf:
                        cbf_text += (
                            f", r={self.cbf._centers(self.data)[2]:.3f} m"
                            f", h_outer={self.cbf.outer_clearance(self.data):+.3f} m"
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
                if self.cfg.use_asteroid_keepin_cbf:
                    # Independent hard-bound checks after every integration
                    # step: never silently accept a violated radial CBF.
                    r_now = self.cbf._centers(self.data)[2]
                    if r_now > self.cbf.shell_max_center_m + 1e-6:
                        raise SafetyViolation(
                            "Hard maximum-separation violation: "
                            f"r={r_now:.6f} m > "
                            f"{self.cbf.shell_max_center_m:.6f} m")
                    if (self.cbf.shell_active and
                            r_now < self.cbf.shell_min_center_m - 1e-6):
                        raise SafetyViolation(
                            "Hard minimum-separation violation: "
                            f"r={r_now:.6f} m < "
                            f"{self.cbf.shell_min_center_m:.6f} m")
                    # Diagnostic abort for physical asteroid recontact. This is
                    # NOT a new CBF for the arm or probe.
                    gap = self.cbf.pad_surface_clearance(self.data)
                    if gap < 0.0:
                        raise SafetyViolation(
                            f"Probe intersected asteroid during observation "
                            f"(signed gap={gap:.4f} m)")
                    for i in range(self.data.ncon):
                        pair = self.data.contact[i].geom
                        if self.TARGET_GEOM_ID in (int(pair[0]), int(pair[1])):
                            raise SafetyViolation(
                                "Unexpected servicer–asteroid contact during observation")

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
                        "Coupled safety-filter failure "
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
        contact_point = self.contact_plan_world(
            pre_avg.position_world_m, pre_avg.R_body_to_world
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
            print(f"Planned contact point (world): {contact_point}")
            print(f"Planned contact point (asteroid body): {self.planned_contact_body_m}")
            print(f"Push direction (world): {push_world}")
            if self.planned_contact_geometry is not None:
                print("  convex-hull surface correction along push: "
                      f"{self.planned_contact_geometry['shift_from_assumed_along_push_m']:+.4f} m")

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
                # The old controller applied forward speed but no lateral
                # tracking, allowing the tip to drift off the 3-D contact ray.
                # Restore the desired ray without changing the push direction.
                cross_track = contact_point - tip_pos - gap_to_contact_point * push_world
                cross_track_error = float(np.linalg.norm(cross_track))
                if gap_to_contact_point < -cfg.max_search_past_surface_m:
                    try:
                        pair_clearance = next(
                            b.h_m for b in self.cbf.barriers(self.data)
                            if (b.geom_a_name, b.geom_b_name) == ASTEROID_KEEP_OUT_PAIR
                        ) if self.cbf is not None else float("nan")
                    except (ValueError, StopIteration):
                        pair_clearance = float("nan")
                    raise RuntimeError(
                        "No pad-target contact detected past the planned collision-hull "
                        f"surface (search limit {cfg.max_search_past_surface_m:.3f} m; "
                        f"axial gap={gap_to_contact_point:+.4f} m, "
                        f"cross-track error={cross_track_error:.4f} m, "
                        f"bus-target CBF clearance={pair_clearance:.4f} m, "
                        f"MuJoCo contacts={self.data.ncon}). "
                        f"Tip={tip_pos}, desired surface point={contact_point}. "
                        "Inspect geometry and collision pairs; do not simply expand the "
                        "search window without checking the actual pad/mesh alignment."
                    )
                speed = (cfg.fast_approach_speed_mps if gap_to_contact_point > cfg.slow_zone_m
                         else cfg.slow_approach_speed_mps)
                lateral_velocity = _clip_norm(
                    cfg.approach_cross_track_gain_per_s * cross_track,
                    cfg.max_approach_lateral_speed_mps,
                )
                v_des = speed * push_world + lateral_velocity
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

        # Switch from the nudge's deliberately moving task-space controller
        # to passive-phase station keeping. Hold the current safe release pose;
        # do not command the spacecraft back toward its pre-contact location.
        if self.cfg.base_pose_hold:
            self._capture_base_pose_reference()

        # The arm/wrist reference is also captured only after clean release.
        # No joint-position hold is added during the force-controlled nudge.
        if self.cfg.post_release_joint_hold:
            self._capture_post_release_joint_reference()

        # Plan controlled withdrawal; the MINIMUM radial CBF is enabled
        # only after the probe has cleared and the bus is outside its radius.
        self._capture_keepin_relative_reference()

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
