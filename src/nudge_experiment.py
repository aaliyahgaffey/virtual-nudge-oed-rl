from __future__ import annotations

"""
nudge_experiment.py
===================
Utilities for running SUCCESSIVE gentle-contact experiments in MuJoCo.

This module does three things:

1) Integrates the contact wrench between the dedicated `nudge_pad_geom` and the
   asteroid to produce measured linear and angular impulses.
2) Enforces simple experiment-level abort limits (peak force, duration, impulse).
3) Feeds completed pre/post motion estimates into `MultiNudgeInertialIdentifier`
   and returns an updated mass / inertia estimate after every nudge.

It deliberately leaves trajectory generation and any optional safety filtering
to the caller. The current single-nudge no-CBF experiment applies its nominal
control directly.

Simulation-vs-flight note
-------------------------
`mj_contactForce` is used here as the simulated 6-axis force/moment sensor.  It
is NOT an inertial-parameter oracle: no target mass or inertia is read.  In a
hardware experiment, replace this accumulator with the actual wrist/probe F/T
sensor stream; the rest of the identification API stays the same.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import mujoco

from .inertial_identifier import (
    IdentificationResult,
    MultiNudgeInertialIdentifier,
    NudgeObservation,
    angular_impulse_about_com,
)


def _vec3(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float).reshape(3)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a


@dataclass
class ContactImpulseMeasurement:
    linear_impulse_world_Ns: np.ndarray
    moment_impulse_about_origin_world_Nms: np.ndarray
    moment_origin_world_m: np.ndarray

    contact_duration_s: float
    peak_force_N: float
    peak_torque_Nm: float
    n_contact_samples: int

    force_time_history: List[np.ndarray] = field(default_factory=list)
    torque_time_history: List[np.ndarray] = field(default_factory=list)
    contact_point_history: List[np.ndarray] = field(default_factory=list)

    @property
    def impulse_magnitude_Ns(self) -> float:
        return float(np.linalg.norm(self.linear_impulse_world_Ns))


class ContactImpulseAccumulator:
    """Integrate pad->asteroid contact wrench without using target inertia."""

    def __init__(
        self,
        model: mujoco.MjModel,
        pad_geom_name: str = "nudge_pad_geom",
        target_geom_name: str = "asteroid_geom",
        force_noise_std_N: float = 0.0,
        torque_noise_std_Nm: float = 0.0,
        rng_seed: int = 0,
        keep_history: bool = True,
    ):
        self.model = model
        self.pad_gid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, pad_geom_name
        )
        self.target_gid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, target_geom_name
        )
        if self.pad_gid < 0:
            raise ValueError(f"Unknown pad geom '{pad_geom_name}'")
        if self.target_gid < 0:
            raise ValueError(f"Unknown target geom '{target_geom_name}'")

        self.force_noise_std_N = float(force_noise_std_N)
        self.torque_noise_std_Nm = float(torque_noise_std_Nm)
        if self.force_noise_std_N < 0.0 or self.torque_noise_std_Nm < 0.0:
            raise ValueError("Noise standard deviations must be nonnegative")
        self.rng = np.random.default_rng(rng_seed)
        self.keep_history = bool(keep_history)

        self.reset((0.0, 0.0, 0.0))

    def reset(self, moment_origin_world_m: Sequence[float]) -> None:
        self.origin = _vec3(moment_origin_world_m, "moment_origin_world_m")
        self.linear_impulse = np.zeros(3)
        self.moment_impulse = np.zeros(3)
        self.contact_duration = 0.0
        self.peak_force = 0.0
        self.peak_torque = 0.0
        self.n_contact_samples = 0
        self.force_history: List[np.ndarray] = []
        self.torque_history: List[np.ndarray] = []
        self.point_history: List[np.ndarray] = []

    def _matching_contact_wrench_world(
        self, data: mujoco.MjData, contact_id: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        contact = data.contact[contact_id]
        g0 = int(contact.geom[0])
        g1 = int(contact.geom[1])
        if {g0, g1} != {int(self.pad_gid), int(self.target_gid)}:
            return None

        wrench_contact = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(self.model, data, contact_id, wrench_contact)

        # MuJoCo stores contact-frame axes as rows; column-vector transform is
        # therefore world = frame.T @ local.  Its force convention points from
        # contact geom[0] toward geom[1].
        R_cw = contact.frame.reshape(3, 3)
        force_world_0_to_1 = R_cw.T @ wrench_contact[:3]
        torque_world_0_to_1 = R_cw.T @ wrench_contact[3:6]

        # We want the wrench ON THE TARGET.  If target is geom[1], convention is
        # already pad/geom0 -> target/geom1.  If target is geom[0], flip sign.
        if g1 == int(self.target_gid):
            sign = 1.0
        else:
            sign = -1.0

        force_target = sign * force_world_0_to_1
        direct_torque_target = sign * torque_world_0_to_1
        contact_pos = np.asarray(contact.pos, dtype=float).copy()
        return force_target, direct_torque_target, contact_pos

    def instantaneous_net_wrench_about_origin(
        self, data: mujoco.MjData
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Return net target contact force and moment about configured origin."""
        F = np.zeros(3)
        M = np.zeros(3)
        weighted_point = np.zeros(3)
        force_weight_sum = 0.0

        for cid in range(data.ncon):
            result = self._matching_contact_wrench_world(data, cid)
            if result is None:
                continue
            f, tau_direct, p = result
            F += f
            M += tau_direct + np.cross(p - self.origin, f)
            w = float(np.linalg.norm(f))
            weighted_point += w * p
            force_weight_sum += w

        if self.force_noise_std_N > 0.0:
            F = F + self.rng.normal(scale=self.force_noise_std_N, size=3)
        if self.torque_noise_std_Nm > 0.0:
            M = M + self.rng.normal(scale=self.torque_noise_std_Nm, size=3)

        if force_weight_sum > 0.0:
            p_net = weighted_point / force_weight_sum
        else:
            p_net = None
        return F, M, p_net

    def sample(self, data: mujoco.MjData, dt_s: Optional[float] = None) -> None:
        """Integrate one force/moment sample after MuJoCo has computed contacts."""
        dt = float(self.model.opt.timestep if dt_s is None else dt_s)
        if dt <= 0.0:
            raise ValueError("dt_s must be positive")

        F, M, p = self.instantaneous_net_wrench_about_origin(data)
        fmag = float(np.linalg.norm(F))
        tmag = float(np.linalg.norm(M))

        # Do not integrate pure synthetic sensor noise when there is no actual
        # pad-target contact.  Check contact matching separately.
        has_contact = False
        for cid in range(data.ncon):
            c = data.contact[cid]
            if {int(c.geom[0]), int(c.geom[1])} == {
                int(self.pad_gid),
                int(self.target_gid),
            }:
                has_contact = True
                break

        if not has_contact:
            return

        self.linear_impulse += F * dt
        self.moment_impulse += M * dt
        self.contact_duration += dt
        self.peak_force = max(self.peak_force, fmag)
        self.peak_torque = max(self.peak_torque, tmag)
        self.n_contact_samples += 1

        if self.keep_history:
            self.force_history.append(F.copy())
            self.torque_history.append(M.copy())
            if p is not None:
                self.point_history.append(p.copy())

    def measurement(self) -> ContactImpulseMeasurement:
        return ContactImpulseMeasurement(
            linear_impulse_world_Ns=self.linear_impulse.copy(),
            moment_impulse_about_origin_world_Nms=self.moment_impulse.copy(),
            moment_origin_world_m=self.origin.copy(),
            contact_duration_s=float(self.contact_duration),
            peak_force_N=float(self.peak_force),
            peak_torque_Nm=float(self.peak_torque),
            n_contact_samples=int(self.n_contact_samples),
            force_time_history=[x.copy() for x in self.force_history],
            torque_time_history=[x.copy() for x in self.torque_history],
            contact_point_history=[x.copy() for x in self.point_history],
        )


@dataclass(frozen=True)
class NudgeLimits:
    desired_impulse_Ns: float = 0.5
    # With ||F(t)|| <= 10 N and at most 1.0 s of accumulated contact,
    # the vector impulse magnitude satisfies ||J|| <= ∫||F||dt <= 10 N s.
    # Keep this consistent with max_force_N * max_contact_time_s.
    max_impulse_Ns: float = 15.0
    max_force_N: float = 5.0
    max_torque_Nm: float = 3.0
    min_contact_time_s: float = 0.0
    max_contact_time_s: float = 1.0

    def __post_init__(self):
        if self.desired_impulse_Ns <= 0.0:
            raise ValueError("desired_impulse_Ns must be positive")
        if self.max_impulse_Ns < self.desired_impulse_Ns:
            raise ValueError("max_impulse_Ns must be >= desired_impulse_Ns")
        if self.max_force_N <= 0.0 or self.max_torque_Nm <= 0.0:
            raise ValueError("force/torque limits must be positive")
        if self.min_contact_time_s < 0.0:
            raise ValueError("min_contact_time_s must be nonnegative")
        if self.max_contact_time_s <= 0.0:
            raise ValueError("max_contact_time_s must be positive")
        if self.min_contact_time_s >= self.max_contact_time_s:
            raise ValueError("min_contact_time_s must be < max_contact_time_s")


@dataclass
class NudgeSupervisorStatus:
    stop: bool
    emergency_abort: bool
    reason: str
    projected_impulse_Ns: float


class NudgeSupervisor:
    """Experiment-level stop/abort logic for a gentle commanded nudge."""

    def __init__(self, push_direction_world: Sequence[float], limits: NudgeLimits):
        d = _vec3(push_direction_world, "push_direction_world")
        n = float(np.linalg.norm(d))
        if n < 1e-12:
            raise ValueError("push_direction_world must be nonzero")
        self.direction = d / n
        self.limits = limits

    def evaluate(self, measurement: ContactImpulseMeasurement) -> NudgeSupervisorStatus:
        projected = float(self.direction @ measurement.linear_impulse_world_Ns)

        if measurement.peak_force_N > self.limits.max_force_N:
            return NudgeSupervisorStatus(
                True, True,
                f"peak force {measurement.peak_force_N:.3f} N exceeds limit",
                projected,
            )
        if measurement.peak_torque_Nm > self.limits.max_torque_Nm:
            return NudgeSupervisorStatus(
                True, True,
                f"peak torque {measurement.peak_torque_Nm:.3f} N m exceeds limit",
                projected,
            )
        if measurement.impulse_magnitude_Ns > self.limits.max_impulse_Ns:
            return NudgeSupervisorStatus(
                True, True,
                f"impulse {measurement.impulse_magnitude_Ns:.3f} N s exceeds hard limit",
                projected,
            )
        if measurement.contact_duration_s > self.limits.max_contact_time_s:
            return NudgeSupervisorStatus(
                True, True,
                f"contact duration {measurement.contact_duration_s:.3f} s exceeds limit",
                projected,
            )
        if (
            projected >= self.limits.desired_impulse_Ns
            and measurement.contact_duration_s >= self.limits.min_contact_time_s
        ):
            return NudgeSupervisorStatus(
                True, False,
                (
                    f"desired directional impulse reached ({projected:.3f} N s) "
                    f"after minimum dwell {self.limits.min_contact_time_s:.3f} s"
                ),
                projected,
            )
        return NudgeSupervisorStatus(False, False, "continue", projected)


@dataclass
class TargetMotionSnapshot:
    """Perception/filter output immediately before or after a nudge."""

    v_com_world_mps: np.ndarray
    R_body_to_world: np.ndarray
    omega_body_radps: np.ndarray

    def __post_init__(self):
        self.v_com_world_mps = _vec3(self.v_com_world_mps, "v_com_world_mps")
        self.R_body_to_world = np.asarray(self.R_body_to_world, dtype=float).reshape(3, 3)
        self.omega_body_radps = _vec3(self.omega_body_radps, "omega_body_radps")


class SuccessiveNudgeSession:
    """Store completed nudges and update the inertial estimate after each one."""

    def __init__(self, identifier: Optional[MultiNudgeInertialIdentifier] = None):
        self.identifier = identifier or MultiNudgeInertialIdentifier()
        self.results: List[IdentificationResult] = []

    def add_completed_nudge(
        self,
        measurement: ContactImpulseMeasurement,
        pre: TargetMotionSnapshot,
        post: TargetMotionSnapshot,
        com_world_during_contact_m: Sequence[float],
        label: str = "",
        mass_weight: float = 1.0,
        inertia_weight: float = 1.0,
    ) -> IdentificationResult:
        """Convert one measured impulse + pre/post motion into an observation.

        The moment accumulator uses a fixed world origin.  We shift that measured
        angular impulse to the target CoM before performing inertia regression.
        """
        com = _vec3(com_world_during_contact_m, "com_world_during_contact_m")
        r_origin_to_com = com - measurement.moment_origin_world_m
        L_com = angular_impulse_about_com(
            measurement.linear_impulse_world_Ns,
            measurement.moment_impulse_about_origin_world_Nms,
            r_origin_to_com,
        )

        obs = NudgeObservation(
            linear_impulse_world=measurement.linear_impulse_world_Ns,
            v_com_before_world=pre.v_com_world_mps,
            v_com_after_world=post.v_com_world_mps,
            angular_impulse_com_world=L_com,
            R_body_to_world_before=pre.R_body_to_world,
            R_body_to_world_after=post.R_body_to_world,
            omega_body_before=pre.omega_body_radps,
            omega_body_after=post.omega_body_radps,
            mass_weight=mass_weight,
            inertia_weight=inertia_weight,
            label=label,
        )
        self.identifier.add_nudge(obs)
        result = self.identifier.estimate()
        self.results.append(result)
        return result

    def latest(self) -> Optional[IdentificationResult]:
        return None if not self.results else self.results[-1]

    def ready_to_stop(
        self,
        minimum_nudges: int = 3,
        max_mass_relative_std: float = 0.03,
        max_inertia_condition_number: float = 1e5,
    ) -> bool:
        """A conservative, transparent stopping rule for the experiment loop."""
        result = self.latest()
        if result is None or result.n_nudges < minimum_nudges:
            return False
        if result.mass_kg <= 0.0:
            return False
        mass_rel_std = result.mass_std_kg / max(abs(result.mass_kg), 1e-12)
        return bool(
            mass_rel_std <= max_mass_relative_std
            and result.inertia_rank == 6
            and result.inertia_condition_number <= max_inertia_condition_number
            and not result.inertia_projection_applied
        )


def print_identification_summary(result: IdentificationResult) -> None:
    print(f"nudges used: {result.n_nudges}")
    print(
        f"mass: {result.mass_kg:.6f} +/- {result.mass_std_kg:.6f} kg "
        f"(residual RMS {result.mass_residual_rms_Ns:.3e} N s)"
    )
    print("body-frame inertia [kg m^2]:")
    print(result.inertia_body_kgm2)
    print("principal moments [kg m^2]:", result.principal_moments_kgm2)
    print(
        f"inertia rank={result.inertia_rank}/6, "
        f"condition={result.inertia_condition_number:.3e}, "
        f"residual RMS={result.inertia_residual_rms_Nms:.3e} N m s"
    )
    for warning in result.warnings:
        print("WARNING:", warning)
