from __future__ import annotations

"""
inertial_identifier.py
======================
Batch / successive inertial identification for a gently nudged, non-cooperative
rigid target.

This file intentionally does NOT read MuJoCo body mass or inertia.  It consumes
only quantities that an onboard system could estimate or measure:

  * linear impulse J from a force / moment sensor,
  * angular impulse L about the target CoM (or data that can be converted to it),
  * pre/post target translational velocity,
  * pre/post target attitude and angular velocity.

The paper by Meng, Liang, and Ma (2019) uses the measured impulse to recover the
mass scale and the inertia scale after a visual EKF has estimated CoM offset,
principal-axis orientation, and normalized principal moments.  Here we preserve
that impulse-based physics, but extend it to SUCCESSIVE nudges by stacking all
of the impulse / momentum equations in weighted least squares.

The inertia estimate returned here is the full symmetric 3x3 tensor in whatever
body-fixed observation frame your pose tracker uses.  Its eigenvalues are the
principal moments and its eigenvectors are the principal axes.

Important assumption
--------------------
The input angular impulse must be about the target CENTER OF MASS.  If your
force/moment measurement is referenced to another point, use
`angular_impulse_about_com(...)` after obtaining a CoM estimate.  The helper
`fit_free_flight_com_offset(...)` estimates the fixed body-frame offset between
CoM and geometric center from a free-flight vision sequence.
"""

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


_EPS = 1e-12


def _vec3(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float).reshape(3)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a


def _mat33(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a


def skew(v: Sequence[float]) -> np.ndarray:
    x, y, z = _vec3(v, "v")
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float
    )


def inertia_design_matrix(omega_body: Sequence[float]) -> np.ndarray:
    """Return M(omega) such that I(6) @ omega == M(omega) @ theta.

    theta = [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
    for a symmetric body-frame inertia tensor.
    """
    wx, wy, wz = _vec3(omega_body, "omega_body")
    return np.array(
        [
            [wx, 0.0, 0.0, wy, wz, 0.0],
            [0.0, wy, 0.0, wx, 0.0, wz],
            [0.0, 0.0, wz, 0.0, wx, wy],
        ],
        dtype=float,
    )


def theta_to_inertia(theta: Sequence[float]) -> np.ndarray:
    Ixx, Iyy, Izz, Ixy, Ixz, Iyz = np.asarray(theta, dtype=float).reshape(6)
    return np.array(
        [
            [Ixx, Ixy, Ixz],
            [Ixy, Iyy, Iyz],
            [Ixz, Iyz, Izz],
        ],
        dtype=float,
    )


def inertia_to_theta(I: Sequence[Sequence[float]]) -> np.ndarray:
    I = _mat33(I, "I")
    return np.array([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]])


def _rotation_quality(R: np.ndarray) -> Tuple[float, float]:
    ortho = float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro"))
    det_err = float(abs(np.linalg.det(R) - 1.0))
    return ortho, det_err


def _check_rotation(R: np.ndarray, name: str, tol: float = 1e-3) -> None:
    ortho, det_err = _rotation_quality(R)
    if ortho > tol or det_err > tol:
        raise ValueError(
            f"{name} is not a valid rotation matrix: "
            f"||R^T R-I||_F={ortho:.3e}, |det(R)-1|={det_err:.3e}"
        )


def _principal_decomposition(I: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(0.5 * (I + I.T))
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    # Keep the principal-axis basis right-handed.
    if np.linalg.det(vecs) < 0.0:
        vecs[:, -1] *= -1.0
    return vals, vecs


def _project_physical_inertia(
    I_raw: np.ndarray,
    min_principal_moment: float = 1e-6,
    triangle_margin: float = 1e-6,
) -> Tuple[np.ndarray, bool]:
    """Project an estimated symmetric inertia to a basic physically valid set.

    Conditions enforced:
      1) positive principal moments,
      2) rigid-body triangle inequalities Imax < Iother1 + Iother2.

    This is deliberately a transparent projection, not a hidden replacement for
    the raw WLS result.  Both raw and projected tensors are returned to callers.
    """
    vals, Q = _principal_decomposition(I_raw)
    original = vals.copy()
    vals = np.maximum(vals, float(min_principal_moment))

    # Sorted ascending.  Only the largest can violate the nontrivial inequality.
    max_allowed = max(
        float(min_principal_moment),
        (vals[0] + vals[1]) * (1.0 - float(triangle_margin)),
    )
    if vals[2] >= vals[0] + vals[1]:
        vals[2] = max_allowed

    changed = not np.allclose(vals, original, rtol=1e-10, atol=1e-12)
    I_proj = Q @ np.diag(vals) @ Q.T
    return 0.5 * (I_proj + I_proj.T), changed


@dataclass
class NudgeObservation:
    """One measured short-duration target excitation.

    All vectors are expressed in SI units.

    `R_body_to_world_*` maps a body-frame vector into the inertial/world frame.
    `omega_body_*` is expressed in that same tracked body-fixed observation frame.

    `angular_impulse_com_world` MUST be the rotational impulse about the target
    CoM.  If you measured a wrench about another origin, convert it first.
    """

    linear_impulse_world: np.ndarray
    v_com_before_world: np.ndarray
    v_com_after_world: np.ndarray

    angular_impulse_com_world: np.ndarray
    R_body_to_world_before: np.ndarray
    R_body_to_world_after: np.ndarray
    omega_body_before: np.ndarray
    omega_body_after: np.ndarray

    mass_weight: float = 1.0
    inertia_weight: float = 1.0
    label: str = ""

    def __post_init__(self):
        self.linear_impulse_world = _vec3(
            self.linear_impulse_world, "linear_impulse_world"
        )
        self.v_com_before_world = _vec3(
            self.v_com_before_world, "v_com_before_world"
        )
        self.v_com_after_world = _vec3(
            self.v_com_after_world, "v_com_after_world"
        )
        self.angular_impulse_com_world = _vec3(
            self.angular_impulse_com_world, "angular_impulse_com_world"
        )
        self.R_body_to_world_before = _mat33(
            self.R_body_to_world_before, "R_body_to_world_before"
        )
        self.R_body_to_world_after = _mat33(
            self.R_body_to_world_after, "R_body_to_world_after"
        )
        _check_rotation(self.R_body_to_world_before, "R_body_to_world_before")
        _check_rotation(self.R_body_to_world_after, "R_body_to_world_after")
        self.omega_body_before = _vec3(
            self.omega_body_before, "omega_body_before"
        )
        self.omega_body_after = _vec3(
            self.omega_body_after, "omega_body_after"
        )
        self.mass_weight = float(self.mass_weight)
        self.inertia_weight = float(self.inertia_weight)
        if self.mass_weight <= 0.0 or self.inertia_weight <= 0.0:
            raise ValueError("Observation weights must be strictly positive")

    @property
    def delta_v_com_world(self) -> np.ndarray:
        return self.v_com_after_world - self.v_com_before_world

    def inertia_regression(self) -> Tuple[np.ndarray, np.ndarray]:
        """A_i, b_i for A_i theta_I = angular_impulse_about_CoM."""
        A = (
            self.R_body_to_world_after @ inertia_design_matrix(self.omega_body_after)
            - self.R_body_to_world_before
            @ inertia_design_matrix(self.omega_body_before)
        )
        b = self.angular_impulse_com_world.copy()
        return A, b


@dataclass
class FreeFlightCOMResult:
    """Result of fitting r_gc(t)=r_com0 + v_com*t + R(t)*rho."""

    r_com_at_t0_world: np.ndarray
    v_com_world: np.ndarray
    rho_com_to_geometric_center_body: np.ndarray
    covariance: np.ndarray
    residual_rms_m: float
    rank: int
    condition_number: float

    def com_position_from_geometric_center(
        self, geometric_center_world: Sequence[float], R_body_to_world: np.ndarray
    ) -> np.ndarray:
        r_gc = _vec3(geometric_center_world, "geometric_center_world")
        R = _mat33(R_body_to_world, "R_body_to_world")
        return r_gc - R @ self.rho_com_to_geometric_center_body


@dataclass
class IdentificationResult:
    n_nudges: int

    mass_kg: float
    mass_std_kg: float
    mass_residual_rms_Ns: float
    mass_information: float

    inertia_raw_body_kgm2: np.ndarray
    inertia_body_kgm2: np.ndarray
    inertia_theta_covariance: np.ndarray
    inertia_residual_rms_Nms: float
    inertia_rank: int
    inertia_condition_number: float
    inertia_projection_applied: bool

    principal_moments_kgm2: np.ndarray
    principal_axes_body: np.ndarray

    warnings: List[str] = field(default_factory=list)

    @property
    def sufficient_inertia_excitation(self) -> bool:
        return self.inertia_rank == 6 and np.isfinite(self.inertia_condition_number)

    def as_dict(self) -> dict:
        return {
            "n_nudges": self.n_nudges,
            "mass_kg": self.mass_kg,
            "mass_std_kg": self.mass_std_kg,
            "mass_residual_rms_Ns": self.mass_residual_rms_Ns,
            "mass_information": self.mass_information,
            "inertia_raw_body_kgm2": self.inertia_raw_body_kgm2.copy(),
            "inertia_body_kgm2": self.inertia_body_kgm2.copy(),
            "inertia_theta_covariance": self.inertia_theta_covariance.copy(),
            "inertia_residual_rms_Nms": self.inertia_residual_rms_Nms,
            "inertia_rank": self.inertia_rank,
            "inertia_condition_number": self.inertia_condition_number,
            "inertia_projection_applied": self.inertia_projection_applied,
            "principal_moments_kgm2": self.principal_moments_kgm2.copy(),
            "principal_axes_body": self.principal_axes_body.copy(),
            "warnings": list(self.warnings),
        }


@dataclass
class NudgeCandidate:
    """Candidate impulse used only for experiment-design scoring."""

    linear_impulse_world: np.ndarray
    angular_impulse_com_world: np.ndarray
    R_body_to_world: np.ndarray
    omega_body_before: np.ndarray
    label: str = ""

    def __post_init__(self):
        self.linear_impulse_world = _vec3(
            self.linear_impulse_world, "linear_impulse_world"
        )
        self.angular_impulse_com_world = _vec3(
            self.angular_impulse_com_world, "angular_impulse_com_world"
        )
        self.R_body_to_world = _mat33(self.R_body_to_world, "R_body_to_world")
        _check_rotation(self.R_body_to_world, "R_body_to_world")
        self.omega_body_before = _vec3(self.omega_body_before, "omega_body_before")


class MultiNudgeInertialIdentifier:
    """Accumulate nudges and re-estimate mass + full body-frame inertia.

    This is intentionally batch-on-history rather than a hidden recursive filter:
    after each nudge, call `estimate()` and the new solution uses every accepted
    excitation.  This makes numerical rank and conditioning directly inspectable.
    """

    def __init__(self, ridge: float = 1e-10):
        self.observations: List[NudgeObservation] = []
        self.ridge = float(ridge)
        if self.ridge < 0.0:
            raise ValueError("ridge must be nonnegative")

    def clear(self) -> None:
        self.observations.clear()

    def add_nudge(self, obs: NudgeObservation) -> None:
        if not isinstance(obs, NudgeObservation):
            raise TypeError("obs must be a NudgeObservation")
        self.observations.append(obs)

    def extend(self, observations: Iterable[NudgeObservation]) -> None:
        for obs in observations:
            self.add_nudge(obs)

    # ------------------------------------------------------------------
    # Paper-consistent mass equation, generalized to many nudges.
    # ------------------------------------------------------------------
    @staticmethod
    def mass_from_single_nudge(
        delta_v_com_world: Sequence[float], linear_impulse_world: Sequence[float]
    ) -> float:
        """Meng et al. Eq. (21): m = dv^T J / dv^T dv."""
        dv = _vec3(delta_v_com_world, "delta_v_com_world")
        J = _vec3(linear_impulse_world, "linear_impulse_world")
        denom = float(dv @ dv)
        if denom < _EPS:
            raise ValueError("Velocity change is too small to identify mass")
        return float(dv @ J / denom)

    def _mass_system(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.observations:
            raise ValueError("No nudge observations have been added")
        A_rows, b_rows, w_rows = [], [], []
        for obs in self.observations:
            dv = obs.delta_v_com_world
            for j in range(3):
                A_rows.append([dv[j]])
                b_rows.append(obs.linear_impulse_world[j])
                w_rows.append(obs.mass_weight)
        return (
            np.asarray(A_rows, dtype=float),
            np.asarray(b_rows, dtype=float),
            np.asarray(w_rows, dtype=float),
        )

    def estimate_mass(self) -> Tuple[float, float, float, float]:
        A, b, w = self._mass_system()
        sqrt_w = np.sqrt(w)
        Aw = A * sqrt_w[:, None]
        bw = b * sqrt_w
        information = float((Aw.T @ Aw).item())
        if information < _EPS:
            raise ValueError(
                "Mass is unobservable: all measured center-of-mass velocity changes are near zero"
            )
        m = float((Aw.T @ bw).item() / information)
        residual = A[:, 0] * m - b
        weighted_sse = float(np.sum(w * residual**2))
        dof = max(len(b) - 1, 1)
        sigma2 = weighted_sse / dof
        var_m = sigma2 / information
        rms = float(np.sqrt(np.mean(residual**2)))
        return m, float(np.sqrt(max(var_m, 0.0))), rms, information

    # ------------------------------------------------------------------
    # Full symmetric inertia from angular impulse / angular momentum.
    # ------------------------------------------------------------------
    def _inertia_system(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.observations:
            raise ValueError("No nudge observations have been added")
        A_blocks, b_blocks, w_rows = [], [], []
        for obs in self.observations:
            A_i, b_i = obs.inertia_regression()
            A_blocks.append(A_i)
            b_blocks.append(b_i)
            w_rows.extend([obs.inertia_weight] * 3)
        return (
            np.vstack(A_blocks),
            np.concatenate(b_blocks),
            np.asarray(w_rows, dtype=float),
        )

    def inertia_information_matrix(self) -> np.ndarray:
        A, _, w = self._inertia_system()
        Aw = A * np.sqrt(w)[:, None]
        return Aw.T @ Aw

    def estimate_inertia(self):
        A, b, w = self._inertia_system()
        sqrt_w = np.sqrt(w)
        Aw = A * sqrt_w[:, None]
        bw = b * sqrt_w

        # Rank/conditioning are based on the data, not on ridge regularization.
        rank = int(np.linalg.matrix_rank(Aw, tol=1e-10))
        singular_values = np.linalg.svd(Aw, compute_uv=False)
        if len(singular_values) == 0 or singular_values[-1] < 1e-14:
            cond = np.inf
        else:
            cond = float(singular_values[0] / singular_values[-1])

        H = Aw.T @ Aw + self.ridge * np.eye(6)
        g = Aw.T @ bw
        try:
            theta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            theta = np.linalg.lstsq(Aw, bw, rcond=None)[0]

        residual = A @ theta - b
        weighted_sse = float(np.sum(w * residual**2))
        dof = max(len(b) - rank, 1)
        sigma2 = weighted_sse / dof
        covariance = sigma2 * np.linalg.pinv(H)
        rms = float(np.sqrt(np.mean(residual**2)))

        I_raw = theta_to_inertia(theta)
        I_phys, projected = _project_physical_inertia(I_raw)
        principal_moments, principal_axes = _principal_decomposition(I_phys)
        return (
            I_raw,
            I_phys,
            covariance,
            rms,
            rank,
            cond,
            projected,
            principal_moments,
            principal_axes,
        )

    def estimate(self) -> IdentificationResult:
        m, m_std, m_rms, m_info = self.estimate_mass()
        (
            I_raw,
            I_phys,
            cov_I,
            I_rms,
            rank,
            cond,
            projected,
            principal_moments,
            principal_axes,
        ) = self.estimate_inertia()

        warnings: List[str] = []
        if m <= 0.0:
            warnings.append(
                "Estimated mass is non-positive; check impulse sign conventions and velocity estimates."
            )
        if rank < 6:
            warnings.append(
                f"Inertia regression rank is {rank}/6.  Add nudges that generate angular impulse "
                "about different axes and/or begin from different angular states."
            )
        if cond > 1e6:
            warnings.append(
                f"Inertia regression is poorly conditioned (condition number {cond:.3e}). "
                "Use more directionally diverse nudges."
            )
        if projected:
            warnings.append(
                "Raw WLS inertia was not physically admissible and was projected to positive, "
                "triangle-inequality-satisfying principal moments. Inspect raw tensor and data quality."
            )

        return IdentificationResult(
            n_nudges=len(self.observations),
            mass_kg=m,
            mass_std_kg=m_std,
            mass_residual_rms_Ns=m_rms,
            mass_information=m_info,
            inertia_raw_body_kgm2=I_raw,
            inertia_body_kgm2=I_phys,
            inertia_theta_covariance=cov_I,
            inertia_residual_rms_Nms=I_rms,
            inertia_rank=rank,
            inertia_condition_number=cond,
            inertia_projection_applied=projected,
            principal_moments_kgm2=principal_moments,
            principal_axes_body=principal_axes,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Successive experiment design.
    # ------------------------------------------------------------------
    def score_nudge_candidates(
        self,
        candidates: Sequence[NudgeCandidate],
        inertia_prior_body: np.ndarray,
        angular_impulse_noise_std: float = 1.0,
        ridge_information: float = 1e-8,
    ) -> List[Tuple[NudgeCandidate, float]]:
        """Rank candidate nudges by predicted D-optimal inertia information.

        For each candidate we predict the instantaneous angular-velocity jump
        using the current/prior inertia and score the log-determinant of the
        resulting 6-parameter information matrix.  Larger is better.

        This does NOT perform path/contact safety checking.  Filter candidate
        contact points/directions through your planner + CBF first, then score
        the safe subset here.
        """
        if not candidates:
            return []
        I = _mat33(inertia_prior_body, "inertia_prior_body")
        vals = np.linalg.eigvalsh(0.5 * (I + I.T))
        if np.min(vals) <= 0.0:
            raise ValueError("inertia_prior_body must be positive definite")

        if self.observations:
            F0 = self.inertia_information_matrix()
        else:
            F0 = np.zeros((6, 6), dtype=float)

        sigma2 = max(float(angular_impulse_noise_std) ** 2, 1e-16)
        base = F0 + float(ridge_information) * np.eye(6)
        scored: List[Tuple[NudgeCandidate, float]] = []

        for cand in candidates:
            R = cand.R_body_to_world
            H_before_body = I @ cand.omega_body_before
            L_body = R.T @ cand.angular_impulse_com_world
            omega_after = np.linalg.solve(I, H_before_body + L_body)

            # Instantaneous impulse: orientation change during the impact itself
            # is negligible, so R_after ~= R_before for experiment design.
            A_c = R @ (
                inertia_design_matrix(omega_after)
                - inertia_design_matrix(cand.omega_body_before)
            )
            F = base + (A_c.T @ A_c) / sigma2
            sign, logdet = np.linalg.slogdet(F)
            score = -np.inf if sign <= 0 else float(logdet)
            scored.append((cand, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored


# ---------------------------------------------------------------------------
# CoM / impulse-reference helpers
# ---------------------------------------------------------------------------
def fit_free_flight_com_offset(
    times_s: Sequence[float],
    geometric_center_world: Sequence[Sequence[float]],
    R_body_to_world: Sequence[Sequence[Sequence[float]]],
    sample_weights: Optional[Sequence[float]] = None,
) -> FreeFlightCOMResult:
    """Estimate CoM motion and body-fixed geometric-center offset in free flight.

    Model (same kinematic idea used by the paper's translational EKF):

        r_gc(t) = r_com(t0) + v_com (t-t0) + R_BI(t) rho_B

    where rho_B points from CoM to geometric center in the tracked body frame.
    This is linear in the 9 unknowns [r_com0, v_com, rho_B], so a batch least
    squares fit is simple and auditable.
    """
    t = np.asarray(times_s, dtype=float).reshape(-1)
    rgc = np.asarray(geometric_center_world, dtype=float).reshape(-1, 3)
    Rs = np.asarray(R_body_to_world, dtype=float).reshape(-1, 3, 3)
    if len(t) != len(rgc) or len(t) != len(Rs):
        raise ValueError("times, positions, and rotations must have equal length")
    if len(t) < 3:
        raise ValueError("At least 3 free-flight pose samples are required")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(rgc)) and np.all(np.isfinite(Rs))):
        raise ValueError("Free-flight data contain non-finite values")

    for i, R in enumerate(Rs):
        _check_rotation(R, f"R_body_to_world[{i}]", tol=3e-3)

    tau = t - t[0]
    A = np.zeros((3 * len(t), 9), dtype=float)
    b = rgc.reshape(-1)
    for k, (tk, R) in enumerate(zip(tau, Rs)):
        row = slice(3 * k, 3 * k + 3)
        A[row, 0:3] = np.eye(3)
        A[row, 3:6] = tk * np.eye(3)
        A[row, 6:9] = R

    if sample_weights is None:
        w = np.ones(3 * len(t), dtype=float)
    else:
        sw = np.asarray(sample_weights, dtype=float).reshape(-1)
        if len(sw) != len(t) or np.any(sw <= 0.0):
            raise ValueError("sample_weights must be positive with one weight per sample")
        w = np.repeat(sw, 3)

    Aw = A * np.sqrt(w)[:, None]
    bw = b * np.sqrt(w)
    x, *_ = np.linalg.lstsq(Aw, bw, rcond=None)

    residual = A @ x - b
    rank = int(np.linalg.matrix_rank(Aw))
    svals = np.linalg.svd(Aw, compute_uv=False)
    cond = np.inf if svals[-1] < 1e-14 else float(svals[0] / svals[-1])
    dof = max(len(b) - rank, 1)
    sigma2 = float(np.sum(w * residual**2) / dof)
    covariance = sigma2 * np.linalg.pinv(Aw.T @ Aw)

    return FreeFlightCOMResult(
        r_com_at_t0_world=x[0:3].copy(),
        v_com_world=x[3:6].copy(),
        rho_com_to_geometric_center_body=x[6:9].copy(),
        covariance=covariance,
        residual_rms_m=float(np.sqrt(np.mean(residual**2))),
        rank=rank,
        condition_number=cond,
    )


def angular_impulse_about_com(
    linear_impulse_world: Sequence[float],
    moment_impulse_about_origin_world: Sequence[float],
    r_origin_to_com_world: Sequence[float],
) -> np.ndarray:
    """Shift an angular impulse from a fixed origin O to the target CoM.

        L_C = L_O - r_{O->C} x J
    """
    J = _vec3(linear_impulse_world, "linear_impulse_world")
    L_O = _vec3(moment_impulse_about_origin_world, "moment_impulse_about_origin_world")
    r_OC = _vec3(r_origin_to_com_world, "r_origin_to_com_world")
    return L_O - np.cross(r_OC, J)


def angular_impulse_from_contact(
    contact_point_world: Sequence[float],
    com_world: Sequence[float],
    linear_impulse_world: Sequence[float],
    direct_torque_impulse_world: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Angular impulse about CoM from a known contact-point impulse."""
    r = _vec3(contact_point_world, "contact_point_world") - _vec3(com_world, "com_world")
    J = _vec3(linear_impulse_world, "linear_impulse_world")
    tau_imp = _vec3(direct_torque_impulse_world, "direct_torque_impulse_world")
    return np.cross(r, J) + tau_imp


def paper_inertia_scale_factor(
    h_world_per_unit_scale: Sequence[float],
    measured_rotational_angular_impulse_world: Sequence[float],
) -> float:
    """Meng et al. Eq. (25): scalar inertia scale K=(h^T L)/(h^T h)."""
    h = _vec3(h_world_per_unit_scale, "h_world_per_unit_scale")
    L = _vec3(
        measured_rotational_angular_impulse_world,
        "measured_rotational_angular_impulse_world",
    )
    denom = float(h @ h)
    if denom < _EPS:
        raise ValueError("h is too small to identify the inertia scale")
    return float(h @ L / denom)


# ---------------------------------------------------------------------------
# Deterministic synthetic self-test.  No MuJoCo dependency.
# ---------------------------------------------------------------------------
def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _self_test() -> None:
    rng = np.random.default_rng(12)
    m_true = 638.027
    I_true = np.array(
        [
            [70.4168, -2.9396, 0.6888],
            [-2.9396, 69.8541, 2.3643],
            [0.6888, 2.3643, 58.3635],
        ]
    )

    estimator = MultiNudgeInertialIdentifier()
    v = np.array([0.01, -0.02, 0.005])

    for k in range(10):
        R = _random_rotation(rng)
        omega0 = rng.normal(scale=0.025, size=3)
        J = rng.normal(scale=1.5, size=3)
        L = rng.normal(scale=0.8, size=3)

        dv = J / m_true
        v1 = v + dv
        H0_body = I_true @ omega0
        H1_body = H0_body + R.T @ L
        omega1 = np.linalg.solve(I_true, H1_body)

        estimator.add_nudge(
            NudgeObservation(
                linear_impulse_world=J,
                v_com_before_world=v,
                v_com_after_world=v1,
                angular_impulse_com_world=L,
                R_body_to_world_before=R,
                R_body_to_world_after=R,
                omega_body_before=omega0,
                omega_body_after=omega1,
                label=f"synthetic_{k}",
            )
        )
        v = v1

    result = estimator.estimate()
    assert abs(result.mass_kg - m_true) < 1e-8, (result.mass_kg, m_true)
    assert np.allclose(result.inertia_raw_body_kgm2, I_true, atol=1e-8), (
        result.inertia_raw_body_kgm2,
        I_true,
    )
    assert result.inertia_rank == 6

    # Free-flight CoM offset fit.
    rho_true = np.array([0.12, -0.04, 0.07])
    r0_true = np.array([2.0, -1.0, 0.5])
    vc_true = np.array([0.03, -0.01, 0.02])
    times = np.linspace(0.0, 8.0, 20)
    Rs = np.stack([_random_rotation(rng) for _ in times])
    rgc = np.stack([r0_true + vc_true * t + R @ rho_true for t, R in zip(times, Rs)])
    ff = fit_free_flight_com_offset(times, rgc, Rs)
    assert np.allclose(ff.r_com_at_t0_world, r0_true, atol=1e-9)
    assert np.allclose(ff.v_com_world, vc_true, atol=1e-9)
    assert np.allclose(ff.rho_com_to_geometric_center_body, rho_true, atol=1e-9)

    print("inertial_identifier.py self-test passed")
    print(f"  mass = {result.mass_kg:.6f} kg")
    print("  inertia =")
    print(result.inertia_raw_body_kgm2)
    print("  principal moments =", result.principal_moments_kgm2)


if __name__ == "__main__":
    _self_test()
