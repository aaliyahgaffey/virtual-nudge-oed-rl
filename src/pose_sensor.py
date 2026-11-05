from __future__ import annotations
import numpy as np
import mujoco

from .nudge_experiment import TargetMotionSnapshot

"""
pose_sensor.py
==============
Replaces vision.py for the single-nudge, no-vision experiment.

There is no camera anywhere in model.xml any more (see the v19 comments at
the top of that file), and no depth/segmentation rendering happens here at
all. Instead we take MuJoCo's own ground truth for the target body and add
injected Gaussian sensor noise on top of it, standing in for whatever
onboard relative-navigation estimator (radar, LIDAR, a beacon-based filter,
...) the mission would actually fly. This is deliberately simpler than a
real filter: no dynamics model, no EKF gain, no time correlation -- just
i.i.d. noise on each fresh "measurement" of the state. That is a reasonable
stand-in for a first pass, but note the (documented) limitation this implies
for the mass estimate in README_SINGLE_NUDGE.md.

Ground-truth extraction below mirrors, field for field, the DEV-ONLY
`perceive_target_dev_truth()` helper used throughout the project's earlier
notebooks (same `mj_objectVelocity`/`xipos`/`xmat` calls, same frame
conventions), so the noisy outputs remain drop-in compatible with
`nudge_experiment.TargetMotionSnapshot` and `inertial_identifier.py`'s
regression equations without any convention translation.

Frame conventions (unchanged from the rest of the project):
  * `R_body_to_world` maps a body-frame vector into the world/inertial frame
    (this is exactly `data.xmat[body_id]`, reshaped to 3x3).
  * `omega_body_radps` is the target's angular velocity expressed in its own
    body-fixed axes (`R_body_to_world.T @ omega_world`), matching what
    `inertial_identifier.NudgeObservation` expects.
  * `velocity_com_world_mps` is the true CENTER-OF-MASS translational
    velocity in world coordinates (not the velocity of the body-frame
    origin, which can differ by `omega x r_com_offset`).
"""


from dataclasses import dataclass


@dataclass
class GroundTruthTargetState:
    """MuJoCo-privileged ground truth. NEVER feed this directly into the
    identifier -- it exists only (a) to be noised by NoisyPoseSensor, and
    (b) for the notebook's clearly-marked DEV-ONLY validation plots/cells."""

    t_s: float
    position_com_world_m: np.ndarray
    R_body_to_world: np.ndarray
    velocity_com_world_mps: np.ndarray
    omega_body_radps: np.ndarray
    omega_world_radps: np.ndarray


@dataclass
class TargetStateEstimate:
    """Noisy onboard estimate. This is what the experiment is actually
    allowed to use."""

    t_s: float
    position_world_m: np.ndarray
    R_body_to_world: np.ndarray
    velocity_world_mps: np.ndarray
    omega_body_radps: np.ndarray
    omega_world_radps: np.ndarray

    def to_target_motion_snapshot(self) -> TargetMotionSnapshot:
        return TargetMotionSnapshot(
            v_com_world_mps=self.velocity_world_mps,
            R_body_to_world=self.R_body_to_world,
            omega_body_radps=self.omega_body_radps,
        )


@dataclass(frozen=True)
class PoseSensorConfig:
    """1-sigma noise levels for the simulated onboard relative-nav estimate.

    Defaults are a plausible close-range LIDAR/laser-ranging relative-nav
    budget (centimeter-level position, sub-degree attitude). They are NOT
    taken from Meng et al. (that figure was not available to us -- see
    README_SINGLE_NUDGE.md) and should be replaced with your own sensor's
    numbers whenever you have them.
    """

    position_noise_std_m: float = 0.010
    velocity_noise_std_mps: float = 0.0008
    attitude_noise_std_rad: float = np.deg2rad(0.30)
    omega_noise_std_radps: float = np.deg2rad(0.05)

    def __post_init__(self):
        for name in (
            "position_noise_std_m",
            "velocity_noise_std_mps",
            "attitude_noise_std_rad",
            "omega_noise_std_radps",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")


def _body_spatial_velocity_world(model, data, body_id: int):
    """MuJoCo spatial velocity in world coordinates: [omega_world; v_origin_world].

    Identical call/convention to the project's earlier
    `_body_spatial_velocity_world` dev-truth helper.
    """
    vel6 = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, vel6, 0)
    return vel6[:3].copy(), vel6[3:].copy()


def true_target_state(model, data, target_body_id: int) -> GroundTruthTargetState:
    """MuJoCo ground truth for the target body's rigid-body motion state.

    Uses only `mj_objectVelocity`, `data.xipos`, and `data.xmat` -- no mesh,
    mass, or inertia is read here (mirrors the "no privileged shape/inertia"
    rule already used throughout the rest of this project).
    """
    mujoco.mj_forward(model, data)
    R = data.xmat[target_body_id].reshape(3, 3).copy()
    body_origin = data.xpos[target_body_id].copy()
    com = data.xipos[target_body_id].copy()

    omega_world, v_origin_world = _body_spatial_velocity_world(model, data, target_body_id)
    v_com = v_origin_world + np.cross(omega_world, com - body_origin)

    return GroundTruthTargetState(
        t_s=float(data.time),
        position_com_world_m=com,
        R_body_to_world=R,
        velocity_com_world_mps=v_com,
        omega_body_radps=R.T @ omega_world,
        omega_world_radps=omega_world,
    )


class NoisyPoseSensor:
    """Injects independent Gaussian noise onto ground truth, each call.

    No time correlation and no dynamics model: every `.sense()` call is an
    independent noisy draw of the CURRENT true state.  This is intentionally
    simple.  Call it repeatedly and average (see
    `single_nudge_experiment.py`'s pre-nudge averaging step and the
    many-post-sample identification strategy) rather than trusting any one
    draw -- that averaging is what actually gives you a usable SNR, not a
    single sample.
    """

    def __init__(self, config: PoseSensorConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng

    def sense(self, truth: GroundTruthTargetState) -> TargetStateEstimate:
        cfg = self.config
        position = truth.position_com_world_m + self.rng.normal(
            scale=cfg.position_noise_std_m, size=3
        )
        velocity = truth.velocity_com_world_mps + self.rng.normal(
            scale=cfg.velocity_noise_std_mps, size=3
        )
        omega_world = truth.omega_world_radps + self.rng.normal(
            scale=cfg.omega_noise_std_radps, size=3
        )

        # Small-angle attitude perturbation, applied in the WORLD frame:
        # R_noisy = exp([dtheta]_x) @ R_true. This models a bounded
        # angular-knowledge error of the estimated body frame relative to
        # world, independent of the target's own current attitude.
        dtheta = self.rng.normal(scale=cfg.attitude_noise_std_rad, size=3)
        angle = float(np.linalg.norm(dtheta))
        if angle < 1e-12:
            dR = np.eye(3)
        else:
            axis = dtheta / angle
            K = np.array([
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ])
            dR = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
        R_noisy = dR @ truth.R_body_to_world
        # Defensive re-orthonormalization (numerical safety only).
        U, _, Vt = np.linalg.svd(R_noisy)
        R_noisy = U @ Vt
        if np.linalg.det(R_noisy) < 0.0:
            U[:, -1] *= -1.0
            R_noisy = U @ Vt

        omega_body = R_noisy.T @ omega_world

        return TargetStateEstimate(
            t_s=truth.t_s,
            position_world_m=position,
            R_body_to_world=R_noisy,
            velocity_world_mps=velocity,
            omega_body_radps=omega_body,
            omega_world_radps=omega_world,
        )


def average_estimates(samples: list[TargetStateEstimate]) -> TargetStateEstimate:
    """Chordal-mean average of several independent noisy samples of what is
    assumed to be (approximately) the SAME true state -- i.e. for averaging
    down noise over a short window where the target is not accelerating
    (e.g. the pre-nudge quiescent observation). Do NOT use this across a long
    free-flight window where the true state is actually changing (tumbling);
    for that, keep every sample separate (see single_nudge_experiment.py).
    """
    if not samples:
        raise ValueError("samples must be non-empty")
    pos = np.mean([s.position_world_m for s in samples], axis=0)
    vel = np.mean([s.velocity_world_mps for s in samples], axis=0)
    omega_world = np.mean([s.omega_world_radps for s in samples], axis=0)
    R_mean = np.mean([s.R_body_to_world for s in samples], axis=0)
    U, _, Vt = np.linalg.svd(R_mean)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0.0:
        U[:, -1] *= -1.0
        R_avg = U @ Vt
    omega_body = R_avg.T @ omega_world
    return TargetStateEstimate(
        t_s=float(np.mean([s.t_s for s in samples])),
        position_world_m=pos,
        R_body_to_world=R_avg,
        velocity_world_mps=vel,
        omega_body_radps=omega_body,
        omega_world_radps=omega_world,
    )
