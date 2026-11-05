from __future__ import annotations

"""
ellipsoidal_cbf.py
==================

Velocity-aware ellipsoidal SELF-COLLISION safety filter for the servicing
spacecraft/manipulator.

Scope of this version
---------------------
This file enforces geometric collision constraints for the servicer itself
and (when configured) existing arm-envelope ellipsoids versus the physical
asteroid mesh. The intentional contact pad is explicitly excluded.

Instead, fixed virtual ellipsoids attached to the spacecraft bus and robotic
arm are used only to prevent SERVICER SELF-COLLISION:

    * arm/link geometry versus the spacecraft bus
    * non-adjacent arm/link geometry versus other arm/link geometry

For each monitored pair (A, B),

    h(q) = distance(A, B) - safety_buffer

and

    psi_1 = h_dot + alpha_1 h.

The controls are actuator forces/torques, so the geometric distance has
relative degree two.  The filter therefore uses a velocity-aware / high-order
barrier condition on psi_1 rather than trying to control one-step position
distance directly.

Important geometry note
-----------------------
Directly connected components are NOT blindly paired if their conservative
ellipsoidal envelopes overlap at the mechanical joint.  For the special
link1-to-bus case, the XML supplies an additional fixed bus-front ellipsoid and
a distal-link1 ellipsoid so that a meaningful non-overlapping self-collision
barrier can still be enforced.

When configured by the experiment, arm ellipsoids are checked against the
physical asteroid mesh. The intentionally contacting pad is excluded;
its force, impulse and duration are supervised independently.

Distance robustness
-------------------
The virtual safety geoms are ellipsoids.  Before relying on MuJoCo's signed
geom-distance query, this implementation computes conservative analytic
separation certificates from ellipsoid support functions.  A positive
projection gap along any tested axis proves separation and prevents a false
zero-distance report from being interpreted as self-collision.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import mujoco
from scipy.optimize import minimize, linprog


CBF_IMPLEMENTATION_VERSION = "self-collision-and-arm-mesh-cbf-v4"

class CBFInfeasibleError(RuntimeError):
    """Raised when no validated control satisfying the CBF constraints is found."""


class SafetyViolation(RuntimeError):
    """Raised when the current self-collision state is materially unsafe."""


@dataclass
class BarrierState:
    # Kept as geom_name for compatibility with the existing notebook.
    # It now contains a PAIR label, e.g. "cbf_bus <-> cbf_link3".
    geom_name: str
    geom_a_name: str
    geom_b_name: str
    h_m: float
    nearest_points_world: np.ndarray
    normal_world: np.ndarray


@dataclass
class CBFFilterResult:
    control: np.ndarray
    nominal_control: np.ndarray
    modified: bool
    barriers_before: List[BarrierState]
    barriers_nominal_next: List[BarrierState]
    barriers_filtered_next: List[BarrierState]
    constraint_residuals_m: np.ndarray
    rate_relaxed: bool
    solver_message: str


class EllipsoidalCBFSafetyFilter:
    """
    Pairwise self-collision HOCBF for fixed virtual ellipsoids.

    Parameters
    ----------
    model
        MuJoCo model.
    collision_pairs
        Sequence of (geom_a_name, geom_b_name) pairs.  Both geoms must be
        virtual ellipsoids.
    gamma
        Discrete decay parameter for psi_1, in (0, 1].
    fd_fraction
        Fraction of actuator control range used for finite differences.
    distmax_m
        Maximum distance passed to mj_geomDistance.
    control_weight
        Positive actuator weights for staying close to nominal control.
    safety_buffer_m
        Required separation BETWEEN the virtual ellipsoids at h=0.
        The ellipsoids themselves already provide geometric inflation.
    velocity_alpha_per_s
        alpha_1 in psi_1 = h_dot + alpha_1 h.
    activation_distance_m
        Full HOCBF optimization is activated when any h is within this band.
    activation_time_horizon_s
        It is also activated early when a closing pair would reach the band
        within this time.
    activation_psi_margin_mps
        The filter also activates when any pair's current high-order barrier
        psi_1 approaches this margin.  This prevents waiting until psi_1 is
        already nearly infeasible before the QP is turned on.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        collision_pairs: Sequence[Tuple[str, str]],
        gamma: float = 0.20,
        fd_fraction: float = 2e-3,
        distmax_m: float = 5.0,
        control_weight: Optional[Sequence[float]] = None,
        safety_buffer_m: float = 0.01,
        velocity_alpha_per_s: float = 2.0,
        violation_tolerance_m: float = 5e-5,
        recovery_tolerance_m: float = 5e-4,
        validation_tolerance_m: float = 2e-5,
        psi_tolerance_mps: float = 2e-4,
        allow_rate_relaxation: bool = True,
        activation_distance_m: float = 0.10,
        activation_time_horizon_s: float = 1.0,
        activation_psi_margin_mps: float = 0.05,
    ):
        self.model = model
        self.collision_pairs = [
            (str(a), str(b))
            for a, b in collision_pairs
        ]

        if not self.collision_pairs:
            raise ValueError(
                "At least one self-collision pair is required."
            )

        self.pair_geom_ids = []
        self.pair_body_ids = []
        self.barrier_names = []

        for geom_a_name, geom_b_name in self.collision_pairs:
            if geom_a_name == geom_b_name:
                raise ValueError(
                    f"Self-collision pair repeats the same geom: "
                    f"{geom_a_name!r}."
                )

            gid_a = int(
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    geom_a_name,
                )
            )
            gid_b = int(
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    geom_b_name,
                )
            )

            if gid_a < 0:
                raise ValueError(
                    f"Unknown self-collision geom '{geom_a_name}'."
                )
            if gid_b < 0:
                raise ValueError(
                    f"Unknown self-collision geom '{geom_b_name}'."
                )

            # The original self-collision and bus-asteroid pairs use ellipsoid
            # proxies on BOTH sides. Arm-to-target pairs instead use the arm's
            # existing enclosing ellipsoid and the ACTUAL asteroid mesh. This
            # avoids the inflated asteroid keep-out proxy blocking the only
            # intentionally allowed contact: the separate nudge pad.
            for name, gid in ((geom_a_name, gid_a), (geom_b_name, gid_b)):
                kind = int(model.geom_type[gid])
                is_ellipsoid = kind == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)
                is_target_mesh = (
                    name == "asteroid_geom"
                    and geom_b_name == "asteroid_geom"
                    and kind == int(mujoco.mjtGeom.mjGEOM_MESH)
                )
                if not (is_ellipsoid or is_target_mesh):
                    raise ValueError(
                        f"CBF pair geom '{name}' must be an ellipsoid, "
                        "except for the asteroid_geom target mesh."
                    )
            if geom_b_name == "asteroid_geom" and geom_a_name == "nudge_pad_geom":
                raise ValueError("Intentional contact pad must not be included in asteroid avoidance CBF")

            bid_a = int(model.geom_bodyid[gid_a])
            bid_b = int(model.geom_bodyid[gid_b])

            if bid_a == bid_b:
                raise ValueError(
                    f"Pair '{geom_a_name}' / '{geom_b_name}' is attached "
                    "to the same rigid body; it cannot create a meaningful "
                    "articulated self-collision barrier."
                )

            self.pair_geom_ids.append(
                (gid_a, gid_b)
            )
            self.pair_body_ids.append(
                (bid_a, bid_b)
            )
            self.barrier_names.append(
                f"{geom_a_name} <-> {geom_b_name}"
            )

        self.gamma = float(gamma)
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError(
                "gamma must be in (0, 1]."
            )

        self.fd_fraction = float(fd_fraction)
        if self.fd_fraction <= 0.0:
            raise ValueError(
                "fd_fraction must be positive."
            )

        self.distmax_m = float(distmax_m)
        if self.distmax_m <= 0.0:
            raise ValueError(
                "distmax_m must be positive."
            )

        self.safety_buffer_m = float(safety_buffer_m)
        self.velocity_alpha_per_s = float(
            velocity_alpha_per_s
        )
        self.violation_tolerance_m = float(
            violation_tolerance_m
        )
        self.recovery_tolerance_m = float(
            recovery_tolerance_m
        )
        self.validation_tolerance_m = float(
            validation_tolerance_m
        )
        self.psi_tolerance_mps = float(
            psi_tolerance_mps
        )
        self.allow_rate_relaxation = bool(
            allow_rate_relaxation
        )
        self.activation_distance_m = float(
            activation_distance_m
        )
        self.activation_time_horizon_s = float(
            activation_time_horizon_s
        )
        self.activation_psi_margin_mps = float(
            activation_psi_margin_mps
        )

        if self.safety_buffer_m < 0.0:
            raise ValueError(
                "safety_buffer_m must be nonnegative."
            )
        if self.velocity_alpha_per_s <= 0.0:
            raise ValueError(
                "velocity_alpha_per_s must be positive."
            )
        if self.violation_tolerance_m < 0.0:
            raise ValueError(
                "violation_tolerance_m must be nonnegative."
            )
        if (
            self.recovery_tolerance_m
            < self.violation_tolerance_m
        ):
            raise ValueError(
                "recovery_tolerance_m must be >= "
                "violation_tolerance_m."
            )
        if self.validation_tolerance_m < 0.0:
            raise ValueError(
                "validation_tolerance_m must be nonnegative."
            )
        if self.psi_tolerance_mps < 0.0:
            raise ValueError(
                "psi_tolerance_mps must be nonnegative."
            )
        if self.activation_distance_m <= 0.0:
            raise ValueError(
                "activation_distance_m must be positive."
            )
        if self.activation_time_horizon_s <= 0.0:
            raise ValueError(
                "activation_time_horizon_s must be positive."
            )
        if self.activation_psi_margin_mps < 0.0:
            raise ValueError(
                "activation_psi_margin_mps must be nonnegative."
            )

        if control_weight is None:
            self.W = np.eye(model.nu)
        else:
            cw = np.asarray(
                control_weight,
                dtype=float,
            )
            if cw.ndim == 1:
                if cw.shape != (model.nu,):
                    raise ValueError(
                        "control_weight must have length model.nu."
                    )
                if np.any(cw <= 0.0):
                    raise ValueError(
                        "All control weights must be positive."
                    )
                self.W = np.diag(cw)

            elif cw.shape == (
                model.nu,
                model.nu,
            ):
                self.W = 0.5 * (
                    cw + cw.T
                )
                if (
                    np.min(
                        np.linalg.eigvalsh(self.W)
                    )
                    <= 0.0
                ):
                    raise ValueError(
                        "control_weight matrix must be "
                        "positive definite."
                    )
            else:
                raise ValueError(
                    "control_weight must be a length-nu "
                    "vector or nu-by-nu matrix."
                )

        self._scratch = mujoco.MjData(model)

    # ------------------------------------------------------------------
    # Pairwise ellipsoid geometry
    # ------------------------------------------------------------------

    def _ellipsoid_support_point(
        self,
        data: mujoco.MjData,
        geom_id: int,
        direction_world: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Support radius and support point of a MuJoCo ellipsoid.

        For an ellipsoid centered at c with world rotation R and semi-axes
        a = [a1, a2, a3], the support function in unit direction n is

            r(n) = sqrt(n^T R diag(a^2) R^T n),

        and the corresponding support point is

            p(n) = c + R diag(a^2) R^T n / r(n).

        This is analytic and does not depend on MuJoCo's contact-distance
        solver.
        """
        gid = int(geom_id)

        n = np.asarray(
            direction_world,
            dtype=float,
        ).reshape(3)

        n_norm = float(
            np.linalg.norm(n)
        )
        if n_norm < 1e-12:
            raise ValueError(
                "Ellipsoid support direction is near zero."
            )
        n = n / n_norm

        center = np.asarray(
            data.geom_xpos[gid],
            dtype=float,
        ).reshape(3)

        R = np.asarray(
            data.geom_xmat[gid],
            dtype=float,
        ).reshape(3, 3)

        semi_axes = np.asarray(
            self.model.geom_size[gid],
            dtype=float,
        ).reshape(3)

        # Shape matrix in world coordinates.
        Q = (
            R
            @ np.diag(
                semi_axes * semi_axes
            )
            @ R.T
        )

        Qn = Q @ n

        radius_sq = float(
            n @ Qn
        )
        radius = float(
            np.sqrt(
                max(
                    radius_sq,
                    0.0,
                )
            )
        )

        if radius < 1e-12:
            point = center.copy()
        else:
            point = (
                center
                + Qn / radius
            )

        return radius, point

    def _support_axis_certificate(
        self,
        data: mujoco.MjData,
        gid_a: int,
        gid_b: int,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Conservative ellipsoid separation certificate.

        We test several physically meaningful axes.  For any unit axis n,

            gap(n)
              = |n^T(c_b-c_a)|
                - r_a(n)
                - r_b(n)

        is a certified lower bound on Euclidean separation whenever gap(n)>0.
        Therefore a positive maximum over the tested axes proves that the two
        ellipsoids are separated, even if mj_geomDistance returns an erroneous
        zero.

        Returns
        -------
        gap_m
            Largest certified projection gap. May be <= 0 if the tested axes
            cannot certify separation.
        fromto
            Support points associated with the best axis.
        normal_world
            Best separating-axis normal oriented from geom A toward geom B.
        """
        gid_a = int(gid_a)
        gid_b = int(gid_b)

        c_a = np.asarray(
            data.geom_xpos[gid_a],
            dtype=float,
        ).reshape(3)
        c_b = np.asarray(
            data.geom_xpos[gid_b],
            dtype=float,
        ).reshape(3)

        delta = c_b - c_a
        center_distance = float(
            np.linalg.norm(delta)
        )

        R_a = np.asarray(
            data.geom_xmat[gid_a],
            dtype=float,
        ).reshape(3, 3)
        R_b = np.asarray(
            data.geom_xmat[gid_b],
            dtype=float,
        ).reshape(3, 3)

        candidate_axes = []

        # Center-to-center direction.
        if center_distance > 1e-12:
            candidate_axes.append(
                delta / center_distance
            )

        # Principal axes of both ellipsoids.
        for j in range(3):
            candidate_axes.append(
                R_a[:, j].copy()
            )
            candidate_axes.append(
                R_b[:, j].copy()
            )

        # Cross products of principal axes.  These are not required for a
        # proof of ellipsoid intersection, but each one is a perfectly valid
        # separating-axis certificate if it yields a positive projection gap.
        for i in range(3):
            for j in range(3):
                axis = np.cross(
                    R_a[:, i],
                    R_b[:, j],
                )
                if (
                    float(
                        np.linalg.norm(axis)
                    )
                    > 1e-10
                ):
                    candidate_axes.append(
                        axis
                    )

        best_gap = -np.inf
        best_fromto = np.zeros(
            6,
            dtype=float,
        )
        best_normal = (
            delta / center_distance
            if center_distance > 1e-12
            else np.array(
                [1.0, 0.0, 0.0],
                dtype=float,
            )
        )

        for axis in candidate_axes:
            u = np.asarray(
                axis,
                dtype=float,
            ).reshape(3)

            u_norm = float(
                np.linalg.norm(u)
            )
            if u_norm < 1e-12:
                continue
            u = u / u_norm

            signed_center_projection = float(
                u @ delta
            )

            # Orient n from A toward B along this tested axis.
            n = (
                u
                if signed_center_projection >= 0.0
                else -u
            )

            r_a, p_a = (
                self._ellipsoid_support_point(
                    data,
                    gid_a,
                    n,
                )
            )

            r_b, p_b_negative = (
                self._ellipsoid_support_point(
                    data,
                    gid_b,
                    -n,
                )
            )

            projection_gap = (
                abs(
                    signed_center_projection
                )
                - r_a
                - r_b
            )

            if projection_gap > best_gap:
                best_gap = float(
                    projection_gap
                )
                best_fromto = np.concatenate(
                    (
                        p_a,
                        p_b_negative,
                    )
                )
                best_normal = n.copy()

        return (
            float(best_gap),
            best_fromto,
            best_normal,
        )

    def _geom_distance(
        self,
        data: mujoco.MjData,
        gid_a: int,
        gid_b: int,
    ) -> Tuple[
        float,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Conservative pairwise distance query for fixed ellipsoid proxies.

        The query deliberately does NOT trust mj_geomDistance blindly.

        1. If enclosing bounding spheres are separated, their gap is a valid
           conservative lower bound.
        2. Otherwise, analytic ellipsoid support functions are used to search
           for a separating-axis certificate.  Any positive projection gap is
           also a mathematically valid lower bound.
        3. Only when neither conservative certificate proves separation do we
           call mj_geomDistance for the near/contact case.

        This avoids false zero-distance reports for clearly separated virtual
        ellipsoids while remaining conservative.
        """
        gid_a = int(gid_a)
        gid_b = int(gid_b)

        c_a = np.asarray(
            data.geom_xpos[gid_a],
            dtype=float,
        ).reshape(3)
        c_b = np.asarray(
            data.geom_xpos[gid_b],
            dtype=float,
        ).reshape(3)

        delta = c_b - c_a
        center_distance = float(
            np.linalg.norm(delta)
        )

        if center_distance > 1e-12:
            center_normal = (
                delta / center_distance
            )
        else:
            center_normal = np.array(
                [1.0, 0.0, 0.0],
                dtype=float,
            )

        r_bound_a = float(
            self.model.geom_rbound[gid_a]
        )
        r_bound_b = float(
            self.model.geom_rbound[gid_b]
        )

        sphere_lower_bound = (
            center_distance
            - r_bound_a
            - r_bound_b
        )

        sphere_fromto = np.concatenate(
            (
                c_a
                + r_bound_a
                * center_normal,
                c_b
                - r_bound_b
                * center_normal,
            )
        )

        # Fast far-field certificate.
        # For asteroid mesh pairs, use MuJoCo's geometry distance
        # rather than the conservative bounding-sphere approximation.

        is_mesh_pair = (
            int(self.model.geom_type[gid_a])
            == int(mujoco.mjtGeom.mjGEOM_MESH)
            or
            int(self.model.geom_type[gid_b])
            == int(mujoco.mjtGeom.mjGEOM_MESH)
        )

        if sphere_lower_bound > 0.0 and not is_mesh_pair:
            return (
                float(sphere_lower_bound),
                sphere_fromto,
                center_normal,
            )

        # Ellipsoid vs *physical asteroid mesh*:
        #   sphere-lower-bound above is valid for arbitrary enclosed geoms;
        #   near the mesh, query MuJoCo directly for the witness points.
        #   Do not use the asteroid's conservative 0.60/0.60/0.64 keep-out
        #   envelope here: that would prematurely forbid planned pad contact.
        #   The pad itself has NO CBF row.
        if int(self.model.geom_type[gid_b]) == int(mujoco.mjtGeom.mjGEOM_MESH):
            fromto = np.zeros(6, dtype=float)
            dist = float(mujoco.mj_geomDistance(
                self.model, data, gid_a, gid_b, self.distmax_m, fromto
            ))
            if not np.isfinite(dist):
                raise SafetyViolation("Nonfinite arm-to-asteroid mesh distance")
            witness_delta = fromto[3:] - fromto[:3]
            witness_norm = float(np.linalg.norm(witness_delta))
            normal = (witness_delta / witness_norm
                      if witness_norm > 1e-12 else center_normal)
            return dist, fromto, normal

        # Analytic ellipsoid support-axis certificate.  This is especially
        # important for long, thin ellipsoids whose bounding spheres overlap.
        (
            support_gap,
            support_fromto,
            support_normal,
        ) = self._support_axis_certificate(
            data,
            gid_a,
            gid_b,
        )

        if support_gap > 0.0:
            return (
                float(
                    support_gap
                ),
                support_fromto,
                support_normal,
            )

        # No tested axis can certify separation.  Use MuJoCo's signed geom
        # distance for the genuinely near/contact regime.
        fromto = np.zeros(
            6,
            dtype=float,
        )

        d_exact = float(
            mujoco.mj_geomDistance(
                self.model,
                data,
                gid_a,
                gid_b,
                self.distmax_m,
                fromto,
            )
        )

        if not np.isfinite(d_exact):
            raise SafetyViolation(
                "mj_geomDistance returned a non-finite "
                "ellipsoid separation."
            )

        exact_delta = (
            fromto[3:]
            - fromto[:3]
        )
        exact_norm = float(
            np.linalg.norm(
                exact_delta
            )
        )

        exact_normal = (
            exact_delta / exact_norm
            if exact_norm > 1e-12
            else support_normal
        )

        return (
            d_exact,
            fromto.copy(),
            exact_normal,
        )


    def barriers(
        self,
        data: mujoco.MjData,
    ) -> List[BarrierState]:
        mujoco.mj_forward(
            self.model,
            data,
        )

        states = []

        for pair_index, (
            (geom_a_name, geom_b_name),
            (gid_a, gid_b),
        ) in enumerate(
            zip(
                self.collision_pairs,
                self.pair_geom_ids,
            )
        ):
            (
                raw_distance,
                fromto,
                normal_world,
            ) = self._geom_distance(
                data,
                gid_a,
                gid_b,
            )

            h = (
                raw_distance
                - self.safety_buffer_m
            )

            states.append(
                BarrierState(
                    geom_name=self.barrier_names[
                        pair_index
                    ],
                    geom_a_name=geom_a_name,
                    geom_b_name=geom_b_name,
                    h_m=float(h),
                    nearest_points_world=fromto,
                    normal_world=np.asarray(
                        normal_world,
                        dtype=float,
                    ).copy(),
                )
            )

        return states

    def minimum_barrier(
        self,
        data: mujoco.MjData,
    ) -> BarrierState:
        return min(
            self.barriers(data),
            key=lambda s: s.h_m,
        )

    def _point_velocity_world(
        self,
        data: mujoco.MjData,
        body_id: int,
        point_world: np.ndarray,
    ) -> np.ndarray:
        jacp = np.zeros(
            (3, self.model.nv),
            dtype=float,
        )
        jacr = np.zeros(
            (3, self.model.nv),
            dtype=float,
        )

        mujoco.mj_jac(
            self.model,
            data,
            jacp,
            jacr,
            np.asarray(
                point_world,
                dtype=float,
            ),
            int(body_id),
        )

        return (
            jacp
            @ np.asarray(
                data.qvel,
                dtype=float,
            )
        )

    def barrier_rates(
        self,
        data: mujoco.MjData,
        states: Optional[
            List[BarrierState]
        ] = None,
    ) -> np.ndarray:
        """
        Pairwise signed-distance rate.

        h_dot > 0 : pair separating
        h_dot < 0 : pair closing
        """
        if states is None:
            states = self.barriers(
                data
            )

        rates = np.zeros(
            len(states),
            dtype=float,
        )

        for i, (
            state,
            (gid_a, gid_b),
            (body_a, body_b),
        ) in enumerate(
            zip(
                states,
                self.pair_geom_ids,
                self.pair_body_ids,
            )
        ):
            p_a = np.asarray(
                state.nearest_points_world[:3],
                dtype=float,
            )
            p_b = np.asarray(
                state.nearest_points_world[3:],
                dtype=float,
            )

            normal_a_to_b = np.asarray(
                state.normal_world,
                dtype=float,
            ).reshape(3)

            normal_norm = float(
                np.linalg.norm(
                    normal_a_to_b
                )
            )

            if normal_norm < 1e-12:
                delta = (
                    np.asarray(
                        data.geom_xpos[gid_b],
                        dtype=float,
                    )
                    - np.asarray(
                        data.geom_xpos[gid_a],
                        dtype=float,
                    )
                )
                normal_norm = float(
                    np.linalg.norm(
                        delta
                    )
                )

                if normal_norm < 1e-12:
                    raise SafetyViolation(
                        "Cannot determine distance normal "
                        f"for self-collision pair "
                        f"'{state.geom_name}'."
                    )

                normal_a_to_b = (
                    delta / normal_norm
                )
            else:
                normal_a_to_b = (
                    normal_a_to_b
                    / normal_norm
                )

            v_a = self._point_velocity_world(
                data,
                body_a,
                p_a,
            )
            v_b = self._point_velocity_world(
                data,
                body_b,
                p_b,
            )

            rates[i] = float(
                normal_a_to_b
                @ (v_b - v_a)
            )

        return rates

    def _psi1(
        self,
        data: mujoco.MjData,
    ):
        states = self.barriers(
            data
        )
        h = np.asarray(
            [s.h_m for s in states],
            dtype=float,
        )
        hdot = self.barrier_rates(
            data,
            states,
        )
        psi = (
            hdot
            + self.velocity_alpha_per_s
            * h
        )
        return states, hdot, psi

    def assert_safe(
        self,
        data: mujoco.MjData,
    ) -> None:
        states, hdot, psi = (
            self._psi1(data)
        )

        bad_h = [
            (s, rate)
            for s, rate in zip(
                states,
                hdot,
            )
            if (
                s.h_m
                < -self.violation_tolerance_m
            )
        ]

        if bad_h:
            detail = ", ".join(
                f"{s.geom_name}: "
                f"h={s.h_m:.6f} m, "
                f"hdot={rate:+.6f} m/s"
                for s, rate in bad_h
            )
            raise SafetyViolation(
                "Self-collision CBF position safe set "
                "already violated: "
                + detail
            )

        bad_psi = [
            (s, rate, psi_i)
            for s, rate, psi_i in zip(
                states,
                hdot,
                psi,
            )
            if (
                psi_i
                < -self.psi_tolerance_mps
            )
        ]

        if bad_psi:
            detail = ", ".join(
                f"{s.geom_name}: "
                f"h={s.h_m:.6f} m, "
                f"hdot={rate:+.6f} m/s, "
                f"psi1={psi_i:+.6f} m/s"
                for s, rate, psi_i
                in bad_psi
            )
            raise SafetyViolation(
                "Self-collision HOCBF velocity safe set "
                "already violated: "
                + detail
            )

    # ------------------------------------------------------------------
    # Scratch rollout
    # ------------------------------------------------------------------

    def _copy_dynamic_state(
        self,
        src: mujoco.MjData,
        dst: mujoco.MjData,
    ) -> None:
        # IMPORTANT: this scratch MjData object is reused across many finite-
        # difference and nonlinear-optimization rollouts.  Clear all derived
        # contact/solver/integration state first so a previous hypothetical
        # rollout cannot contaminate the next one.
        mujoco.mj_resetData(self.model, dst)

        dst.qpos[:] = src.qpos
        dst.qvel[:] = src.qvel

        if self.model.na:
            dst.act[:] = src.act
        if self.model.nmocap:
            dst.mocap_pos[:] = src.mocap_pos
            dst.mocap_quat[:] = src.mocap_quat
        if self.model.nuserdata:
            dst.userdata[:] = src.userdata

        # These externally applied generalized/body forces are part of the
        # current dynamics state and should be copied when present.
        dst.qfrc_applied[:] = src.qfrc_applied
        dst.xfrc_applied[:] = src.xfrc_applied

        # Warm-start values can matter during contact solves.
        dst.qacc_warmstart[:] = src.qacc_warmstart

        dst.time = src.time

    def _next_state_quantities(
        self,
        data: mujoco.MjData,
        control: np.ndarray,
    ):
        self._copy_dynamic_state(data, self._scratch)
        self._scratch.ctrl[:] = control

        mujoco.mj_forward(self.model, self._scratch)
        mujoco.mj_step(self.model, self._scratch)
        mujoco.mj_forward(self.model, self._scratch)

        states, hdot, psi = self._psi1(self._scratch)
        h = np.asarray([s.h_m for s in states], dtype=float)

        return states, h, hdot, psi

    # ------------------------------------------------------------------
    # Bounds and finite-difference linearization
    # ------------------------------------------------------------------

    def _default_control_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lo = np.full(self.model.nu, -np.inf, dtype=float)
        hi = np.full(self.model.nu, np.inf, dtype=float)

        limited = np.asarray(
            self.model.actuator_ctrllimited,
            dtype=bool,
        )

        lo[limited] = self.model.actuator_ctrlrange[limited, 0]
        hi[limited] = self.model.actuator_ctrlrange[limited, 1]

        return lo, hi

    def _linearize_next_psi(
        self,
        data: mujoco.MjData,
        u_nom: np.ndarray,
        psi_nom_next: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
    ) -> np.ndarray:
        """
        G[i,j] = d psi1_i(x_{k+1}) / d u_j around u_nom.

        Unlike d h(q_{k+1}) / d u, this derivative contains velocity response
        and therefore has first-order control authority at a force/torque input.
        """
        nbar = len(self.collision_pairs)
        G = np.zeros((nbar, self.model.nu), dtype=float)

        for j in range(self.model.nu):
            if np.isfinite(lo[j]) and np.isfinite(hi[j]):
                scale = max(
                    1.0,
                    float(hi[j] - lo[j]),
                )
            else:
                scale = max(
                    1.0,
                    abs(float(u_nom[j])),
                )

            du = self.fd_fraction * scale

            up = u_nom.copy()
            um = u_nom.copy()

            up[j] = min(hi[j], u_nom[j] + du)
            um[j] = max(lo[j], u_nom[j] - du)

            moved_up = abs(up[j] - u_nom[j]) > 1e-14
            moved_dn = abs(um[j] - u_nom[j]) > 1e-14

            if moved_up and moved_dn:
                _, _, _, psi_p = self._next_state_quantities(
                    data,
                    up,
                )
                _, _, _, psi_m = self._next_state_quantities(
                    data,
                    um,
                )
                G[:, j] = (
                    (psi_p - psi_m)
                    / (up[j] - um[j])
                )

            elif moved_up:
                _, _, _, psi_p = self._next_state_quantities(
                    data,
                    up,
                )
                G[:, j] = (
                    (psi_p - psi_nom_next)
                    / (up[j] - u_nom[j])
                )

            elif moved_dn:
                _, _, _, psi_m = self._next_state_quantities(
                    data,
                    um,
                )
                G[:, j] = (
                    (psi_nom_next - psi_m)
                    / (u_nom[j] - um[j])
                )

        return G

    def _solve_qp(
        self,
        u_nom: np.ndarray,
        G: np.ndarray,
        rhs: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        label: str,
        row_names: Optional[Sequence[str]] = None,
    ) -> Tuple[np.ndarray, str]:
        """
        Solve

            min 0.5 (u-u_nom)^T W (u-u_nom)
            s.t. G u >= rhs
                 lo <= u <= hi

        Rows are normalized before SLSQP so a small physical sensitivity does
        not become a numerical conditioning problem.
        """
        if G.shape[0] == 0:
            return u_nom.copy(), "No active HOCBF constraints."

        if row_names is None:
            row_names = self.barrier_names
        else:
            row_names = list(row_names)

        if len(row_names) != G.shape[0]:
            raise ValueError(
                "row_names length must match the number of QP rows."
            )

        row_scale = np.max(np.abs(G), axis=1)

        # Detect constraints with effectively no instantaneous control authority.
        no_authority = row_scale < 1e-10
        if np.any(no_authority):
            # If G≈0, feasibility reduces to 0 >= rhs.
            impossible = no_authority & (rhs > self.psi_tolerance_mps)
            if np.any(impossible):
                idx = int(np.where(impossible)[0][0])
                raise CBFInfeasibleError(
                    f"{label}: '{row_names[idx]}' has effectively "
                    "zero control authority in the linearized HOCBF constraint "
                    f"but requires rhs={rhs[idx]:+.3e}."
                )

        safe_scale = np.maximum(row_scale, 1e-8)
        Gs = G / safe_scale[:, None]
        rhss = rhs / safe_scale

        def objective(u):
            du = u - u_nom
            return 0.5 * float(du @ self.W @ du)

        def objective_jac(u):
            return self.W @ (u - u_nom)

        constraints = {
            "type": "ineq",
            "fun": lambda u: Gs @ u - rhss,
            "jac": lambda u: Gs,
        }

        bounds = [
            (
                None if not np.isfinite(lo_i) else float(lo_i),
                None if not np.isfinite(hi_i) else float(hi_i),
            )
            for lo_i, hi_i in zip(lo, hi)
        ]

        sol = minimize(
            objective,
            np.clip(u_nom, lo, hi),
            jac=objective_jac,
            method="SLSQP",
            bounds=bounds,
            constraints=[constraints],
            options={
                "ftol": 1e-10,
                "maxiter": 120,
                "disp": False,
            },
        )

        if sol.x is None or not np.all(np.isfinite(sol.x)):
            raise CBFInfeasibleError(
                f"{label}: QP returned no finite control."
            )

        physical_residual = G @ sol.x - rhs

        if (
            (not sol.success)
            or np.min(physical_residual) < -self.psi_tolerance_mps
        ):
            # SLSQP may report "Positive directional derivative for linesearch"
            # even if the hard linear inequalities have a feasible solution.
            # Before declaring the CBF infeasible, solve a HARD feasibility LP.
            # No slack, margin reduction, or actuator-limit increase is used.
            feasibility = linprog(
                c=np.zeros_like(u_nom),
                A_ub=-Gs,
                b_ub=-rhss,
                bounds=bounds,
                method="highs",
            )
            if feasibility.success and feasibility.x is not None:
                feasible_u = np.asarray(feasibility.x, dtype=float)
                lp_residual = G @ feasible_u - rhs
                if np.min(lp_residual) >= -self.psi_tolerance_mps:
                    # Try to recover a near-nominal command from a FEASIBLE
                    # initial point, instead of accepting a vertex solution.
                    retry = minimize(
                        objective,
                        feasible_u,
                        jac=objective_jac,
                        method="SLSQP",
                        bounds=bounds,
                        constraints=[constraints],
                        options={"ftol": 1e-10, "maxiter": 250, "disp": False},
                    )
                    if (retry.x is not None
                            and np.all(np.isfinite(retry.x))
                            and np.min(G @ retry.x - rhs)
                            >= -self.psi_tolerance_mps):
                        return (np.asarray(retry.x, dtype=float),
                                "LP-feasible SLSQP retry: " + str(retry.message))
                    # The outer filter still validates the exact nonlinear
                    # MuJoCo next step after this linearized feasibility check.
                    return feasible_u, "Hard CBF LP feasibility recovery"

            idx = int(np.argmin(physical_residual))
            extra = (f"; hard-feasibility LP: {feasibility.message}"
                     if feasibility is not None else "")
            raise CBFInfeasibleError(
                f"{label}: QP failed for "
                f"'{row_names[idx]}': {sol.message}; "
                f"minimum HOCBF residual="
                f"{physical_residual[idx]:+.3e} m/s" + extra + "."
            )

        return np.asarray(sol.x, dtype=float), str(sol.message)


    def _solve_nonlinear_fallback(
        self,
        data: mujoco.MjData,
        u_nom: np.ndarray,
        u_seed: np.ndarray,
        required_psi: np.ndarray,
        h_floor: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        label: str,
        active_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, str]:
        """
        Direct nonlinear MuJoCo fallback.

        This is used only if the fast linearized HOCBF QP produces a command
        that fails the exact one-step MuJoCo validation.  It optimizes the same
        quadratic deviation from u_nom, but evaluates the HOCBF and geometric
        constraints for the ACTIVE collision pairs through the actual MuJoCo
        rollout rather than through the local linear model.

        The fallback is deliberately not the primary solver because it is much
        more expensive.  It is a recovery mechanism for local linearization error in the pairwise
        ellipsoid-distance dynamics.
        """
        if active_indices is None:
            active_indices = np.arange(
                len(self.collision_pairs),
                dtype=int,
            )
        else:
            active_indices = np.asarray(
                active_indices,
                dtype=int,
            )

        if required_psi.shape != (len(active_indices),):
            raise ValueError(
                "required_psi must contain one value per active pair."
            )

        active_names = [
            self.barrier_names[int(i)]
            for i in active_indices
        ]

        bounds = [
            (
                None if not np.isfinite(lo_i) else float(lo_i),
                None if not np.isfinite(hi_i) else float(hi_i),
            )
            for lo_i, hi_i in zip(lo, hi)
        ]

        # Physical scaling for the nonlinear constraints.  SLSQP behaves much
        # better if meter and meter/second residuals are O(1) numerically.
        psi_scale = max(
            1e-3,
            float(np.max(np.abs(required_psi))) if required_psi.size else 1e-3,
        )
        h_scale = max(
            1e-3,
            float(self.safety_buffer_m),
        )

        cache_u = None
        cache_h = None
        cache_psi = None

        def rollout(u):
            nonlocal cache_u, cache_h, cache_psi

            u = np.asarray(u, dtype=float)

            if (
                cache_u is not None
                and np.array_equal(u, cache_u)
            ):
                return cache_h, cache_psi

            _, h, _, psi = self._next_state_quantities(
                data,
                u,
            )

            cache_u = u.copy()
            cache_h = h.copy()
            cache_psi = psi.copy()

            return cache_h, cache_psi

        def objective(u):
            du = np.asarray(u, dtype=float) - u_nom
            return 0.5 * float(du @ self.W @ du)

        def objective_jac(u):
            return self.W @ (
                np.asarray(u, dtype=float) - u_nom
            )

        constraints = [
            {
                "type": "ineq",
                "fun": lambda u: (
                    rollout(u)[1][active_indices]
                    - required_psi
                ) / psi_scale,
            },
            {
                "type": "ineq",
                "fun": lambda u: (
                    rollout(u)[0][active_indices]
                    - h_floor[active_indices]
                ) / h_scale,
            },
        ]

        # Try the linear-HOCBF result first, then the nominal command, then
        # zero actuation.  This gives the nonlinear solver multiple sensible
        # deterministic starting points without changing the safety criterion.
        starts = [
            np.clip(
                np.asarray(u_seed, dtype=float),
                lo,
                hi,
            ),
            np.clip(
                np.asarray(u_nom, dtype=float),
                lo,
                hi,
            ),
            np.clip(
                np.zeros(self.model.nu, dtype=float),
                lo,
                hi,
            ),
        ]

        best_u = None
        best_obj = np.inf
        best_message = None
        failure_messages = []

        for start_index, start in enumerate(starts):
            sol = minimize(
                objective,
                start,
                jac=objective_jac,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={
                    "ftol": 1e-10,
                    "maxiter": 120,
                    "disp": False,
                },
            )

            if sol.x is None or not np.all(np.isfinite(sol.x)):
                failure_messages.append(
                    f"start {start_index}: no finite control"
                )
                continue

            candidate = np.asarray(
                sol.x,
                dtype=float,
            )

            h_candidate, psi_candidate = rollout(
                candidate
            )

            psi_residual = (
                psi_candidate[active_indices]
                - required_psi
            )
            h_residual = (
                h_candidate[active_indices]
                - h_floor[active_indices]
            )

            feasible = (
                np.min(psi_residual)
                >= -self.psi_tolerance_mps
                and np.min(h_residual)
                >= -self.validation_tolerance_m
            )

            if feasible:
                value = objective(candidate)
                if value < best_obj:
                    best_u = candidate.copy()
                    best_obj = value
                    best_message = (
                        f"{label}: nonlinear MuJoCo fallback "
                        f"accepted from start {start_index}; "
                        f"SLSQP='{sol.message}'."
                    )
            else:
                psi_idx = int(
                    np.argmin(psi_residual)
                )
                h_idx = int(
                    np.argmin(h_residual)
                )
                h_global_idx = int(
                    active_indices[h_idx]
                )
                failure_messages.append(
                    f"start {start_index}: "
                    f"psi_residual[{active_names[psi_idx]}]="
                    f"{psi_residual[psi_idx]:+.3e} m/s, "
                    f"h_candidate[{active_names[h_idx]}]="
                    f"{h_candidate[h_global_idx]:+.6f} m, "
                    f"h_residual={h_residual[h_idx]:+.3e} m; "
                    f"SLSQP='{sol.message}'"
                )

        if best_u is None:
            raise CBFInfeasibleError(
                f"{label}: direct nonlinear MuJoCo fallback "
                "could not find a validated command. "
                + " | ".join(failure_messages)
            )

        return best_u, best_message

    # ------------------------------------------------------------------
    # Main filter
    # ------------------------------------------------------------------

    def filter_control(
        self,
        data: mujoco.MjData,
        nominal_control: Sequence[float],
        u_lo: Optional[Sequence[float]] = None,
        u_hi: Optional[Sequence[float]] = None,
    ) -> CBFFilterResult:

        u_nom = np.asarray(
            nominal_control,
            dtype=float,
        ).reshape(self.model.nu)

        if not np.all(np.isfinite(u_nom)):
            raise ValueError(
                "nominal_control contains non-finite values."
            )

        default_lo, default_hi = self._default_control_bounds()

        lo = (
            default_lo
            if u_lo is None
            else np.asarray(
                u_lo,
                dtype=float,
            ).reshape(self.model.nu)
        )
        hi = (
            default_hi
            if u_hi is None
            else np.asarray(
                u_hi,
                dtype=float,
            ).reshape(self.model.nu)
        )

        if np.any(lo > hi):
            raise ValueError("Invalid actuator bounds.")

        u_nom = np.clip(u_nom, lo, hi)

        before, hdot_now, psi_now = self._psi1(data)
        h_now = np.asarray(
            [b.h_m for b in before],
            dtype=float,
        )

        worst_h_idx = int(np.argmin(h_now))
        if h_now[worst_h_idx] < -self.recovery_tolerance_m:
            b = before[worst_h_idx]
            gid_a, gid_b = self.pair_geom_ids[worst_h_idx]

            c_a = np.asarray(
                data.geom_xpos[gid_a],
                dtype=float,
            )
            c_b = np.asarray(
                data.geom_xpos[gid_b],
                dtype=float,
            )

            center_distance = float(
                np.linalg.norm(c_b - c_a)
            )
            sphere_lower_bound = (
                center_distance
                - float(self.model.geom_rbound[gid_a])
                - float(self.model.geom_rbound[gid_b])
            )

            raise SafetyViolation(
                f"Current state materially violates self-collision pair "
                f"'{b.geom_name}': h={b.h_m:.6f} m; recovery limit is "
                f"{-self.recovery_tolerance_m:.6f} m; "
                f"center distance={center_distance:.6f} m; "
                f"bounding-sphere lower bound="
                f"{sphere_lower_bound:.6f} m."
            )

        # --------------------------------------------------------------
        # Near-field self-collision CBF activation band
        # --------------------------------------------------------------
        #
        # Do not run the expensive one-step HOCBF optimization when every
        # monitored self-collision pair has comfortable separation.  Activate
        # the filter early if a pair is already near the configured band or is
        # closing quickly enough to reach it within the look-ahead horizon.
        #
        # EACH PAIR is activated independently if:
        #   1) its h is already inside activation_distance_m, OR
        #   2) it is predicted to reach that activation band within
        #      activation_time_horizon_s, OR
        #   3) its psi1 is close to the configured activation margin.
        #
        # Critically, one close pair no longer forces unrelated, comfortably
        # separated pairs into the same QP.
        distance_to_activation = (
            h_now - self.activation_distance_m
        )

        time_to_activation = np.full_like(
            h_now,
            np.inf,
            dtype=float,
        )

        closing = hdot_now < 0.0
        time_to_activation[closing] = (
            np.maximum(
                distance_to_activation[closing],
                0.0,
            )
            / (-hdot_now[closing])
        )

        active_mask = (
            (h_now <= self.activation_distance_m)
            | (
                time_to_activation
                <= self.activation_time_horizon_s
            )
            | (
                psi_now
                <= self.activation_psi_margin_mps
            )
        )

        active_indices = np.flatnonzero(
            active_mask
        )
        activation_required = (
            active_indices.size > 0
        )

        if not activation_required:
            return CBFFilterResult(
                control=u_nom.copy(),
                nominal_control=u_nom.copy(),
                modified=False,
                barriers_before=before,
                # No hypothetical rollout is needed in the far field.
                # Reuse the current barriers for diagnostics.
                barriers_nominal_next=before,
                barriers_filtered_next=before,
                constraint_residuals_m=np.zeros_like(
                    h_now,
                    dtype=float,
                ),
                rate_relaxed=False,
                solver_message=(
                    "Self-collision CBF inactive outside activation band: "
                    f"minimum h={np.min(h_now):.6f} m, "
                    f"activation distance="
                    f"{self.activation_distance_m:.3f} m; "
                    f"minimum psi1={np.min(psi_now):+.6f} m/s."
                ),
            )

        (
            nominal_next_states,
            h_nom_next,
            hdot_nom_next,
            psi_nom_next,
        ) = self._next_state_quantities(
            data,
            u_nom,
        )

        # High-order discrete condition, enforced ONLY for the
        # currently active self-collision pairs:
        #
        #   psi1_{k+1} >= (1-gamma) psi1_k
        #
        # Other pairs continue to be monitored every step, but they do not
        # become QP rows merely because a different pair is close.
        required_psi_full = (
            (1.0 - self.gamma)
            * psi_now
        )

        required_psi = (
            required_psi_full[
                active_indices
            ]
        )

        active_names = [
            self.barrier_names[int(i)]
            for i in active_indices
        ]

        # If the nominal command is already safe for every ACTIVE rate
        # constraint, and every ACTIVE pair remains geometrically outside the
        # small validation tolerance, do not modify it.
        #
        # Inactive pairs are still checked at the START of every real control
        # step and can become active through h, time-to-activation, or psi1.
        # They are intentionally not imposed as one-step nonlinear constraints
        # merely because another pair activated the filter.
        nominal_psi_ok = np.all(
            psi_nom_next[active_indices]
            >= required_psi
            - self.psi_tolerance_mps
        )
        nominal_h_ok = np.all(
            h_nom_next[active_indices]
            >= -self.validation_tolerance_m
        )

        if nominal_psi_ok and nominal_h_ok:
            residual = np.zeros_like(
                psi_now,
                dtype=float,
            )
            residual[active_indices] = (
                psi_nom_next[active_indices]
                - required_psi
            )

            return CBFFilterResult(
                control=u_nom.copy(),
                nominal_control=u_nom.copy(),
                modified=False,
                barriers_before=before,
                barriers_nominal_next=nominal_next_states,
                barriers_filtered_next=nominal_next_states,
                constraint_residuals_m=residual,
                rate_relaxed=False,
                solver_message=(
                    "Nominal control already satisfies "
                    "all active self-collision HOCBF rows. "
                    "Active pairs: "
                    + ", ".join(active_names)
                ),
            )

        G_full = self._linearize_next_psi(
            data=data,
            u_nom=u_nom,
            psi_nom_next=psi_nom_next,
            lo=lo,
            hi=hi,
        )

        # Keep only the rows belonging to pairs that actually triggered
        # activation.
        G = G_full[
            active_indices,
            :
        ]

        psi_nom_active = psi_nom_next[
            active_indices
        ]

        # Linearization:
        #
        # psi_next(u)
        #   ~= psi_nom_next + G (u-u_nom)
        #
        # therefore:
        #
        # G u >= required_psi - psi_nom_next + G u_nom
        rhs_rate = (
            required_psi
            - psi_nom_active
            + G @ u_nom
        )

        rate_relaxed = False

        try:
            u_safe, message = self._solve_qp(
                u_nom,
                G,
                rhs_rate,
                lo,
                hi,
                label="exponential-HOCBF",
                row_names=active_names,
            )
            nonlinear_required = (
                required_psi.copy()
            )

        except CBFInfeasibleError as first_error:
            if not self.allow_rate_relaxation:
                raise

            # If psi_now >= 0, preserve psi_next >= 0.
            # If psi_now is slightly negative, require monotonic recovery.
            hard_required = np.where(
                psi_now[active_indices]
                >= 0.0,
                0.0,
                required_psi,
            )

            rhs_hard = (
                hard_required
                - psi_nom_active
                + G @ u_nom
            )

            try:
                u_safe, message2 = self._solve_qp(
                    u_nom,
                    G,
                    rhs_hard,
                    lo,
                    hi,
                    label=(
                        "hard-safe/recovery HOCBF"
                    ),
                    row_names=active_names,
                )

            except CBFInfeasibleError as second_error:
                local_idx = int(
                    np.argmin(
                        psi_now[
                            active_indices
                        ]
                    )
                )
                global_idx = int(
                    active_indices[
                        local_idx
                    ]
                )
                worst_rate_state = before[
                    global_idx
                ]

                raise CBFInfeasibleError(
                    f"Exponential HOCBF infeasible "
                    f"({first_error}); "
                    f"hard-safe/recovery HOCBF also "
                    f"infeasible ({second_error}). "
                    f"Current limiting ACTIVE pair "
                    f"'{worst_rate_state.geom_name}': "
                    f"h="
                    f"{worst_rate_state.h_m:+.6f} m, "
                    f"hdot="
                    f"{hdot_now[global_idx]:+.6f} m/s, "
                    f"psi1="
                    f"{psi_now[global_idx]:+.6f} m/s. "
                    f"Active pairs: "
                    + ", ".join(active_names)
                ) from second_error

            message = (
                f"Rate relaxed. {message2}. "
                f"Original solve: {first_error}"
            )
            nonlinear_required = (
                hard_required.copy()
            )
            rate_relaxed = True

        (
            filtered_next_states,
            h_filtered_next,
            hdot_filtered_next,
            psi_filtered_next,
        ) = self._next_state_quantities(
            data,
            u_safe,
        )

        psi_residual_active = (
            psi_filtered_next[
                active_indices
            ]
            - nonlinear_required
        )

        psi_residual = np.zeros_like(
            psi_now,
            dtype=float,
        )
        psi_residual[
            active_indices
        ] = psi_residual_active

        # The fast QP uses a local linearization of psi1_{k+1}(u).  For articulated self-collision geometry, the nearest-point pair can change
        # as the arm moves and make that local model inaccurate.  If exact MuJoCo validation fails, do NOT
        # simply enlarge the tolerance: switch to a direct nonlinear fallback
        # that evaluates the actual one-step rollout inside the optimization.
        linear_validation_failed = (
            np.min(psi_residual_active)
            < -self.psi_tolerance_mps
        )

        geometric_validation_failed = (
            np.min(
                h_filtered_next[
                    active_indices
                ]
            )
            < -self.recovery_tolerance_m
        )

        if (
            linear_validation_failed
            or geometric_validation_failed
        ):
            # Maintain the same buffered geometric recovery band.  A state
            # currently outside the mathematical h>=0 boundary by only a tiny
            # numerical amount is allowed to recover, but the nonlinear solver
            # may never leave the configured recovery band.
            h_floor = np.full_like(
                h_now,
                -self.recovery_tolerance_m,
                dtype=float,
            )

            linear_failure_detail = ""
            if linear_validation_failed:
                local_idx = int(
                    np.argmin(
                        psi_residual_active
                    )
                )
                global_idx = int(
                    active_indices[
                        local_idx
                    ]
                )
                linear_failure_detail = (
                    "Linearized validation miss for "
                    f"'{self.barrier_names[global_idx]}': "
                    f"psi1_next="
                    f"{psi_filtered_next[global_idx]:+.6e} m/s, "
                    f"required="
                    f"{nonlinear_required[local_idx]:+.6e} m/s, "
                    f"residual="
                    f"{psi_residual_active[local_idx]:+.3e} m/s. "
                )

            try:
                u_safe, nonlinear_message = self._solve_nonlinear_fallback(
                    data=data,
                    u_nom=u_nom,
                    u_seed=u_safe,
                    required_psi=nonlinear_required,
                    h_floor=h_floor,
                    lo=lo,
                    hi=hi,
                    label="validated HOCBF",
                    active_indices=active_indices,
                )

            except CBFInfeasibleError as strict_nonlinear_error:
                if not self.allow_rate_relaxation:
                    raise

                # The fast linearized solve may have succeeded for the
                # exponential-rate requirement while the exact nonlinear
                # MuJoCo rollout reveals that requirement is too aggressive.
                # Apply the SAME rate-relaxation policy here that we already
                # use for the linear QP:
                #
                #   psi_now >= 0  -> require psi_next >= 0
                #   psi_now < 0   -> require monotonic recovery
                #
                # This preserves the HOCBF safe set without pretending that
                # the stricter exponential decay rate is physically achievable
                # in this step.
                relaxed_required = np.where(
                    psi_now[active_indices]
                    >= 0.0,
                    0.0,
                    required_psi,
                )

                u_safe, nonlinear_message2 = self._solve_nonlinear_fallback(
                    data=data,
                    u_nom=u_nom,
                    u_seed=u_safe,
                    required_psi=relaxed_required,
                    h_floor=h_floor,
                    lo=lo,
                    hi=hi,
                    label="validated hard-safe/recovery HOCBF",
                    active_indices=active_indices,
                )

                nonlinear_message = (
                    "Strict nonlinear HOCBF rate was infeasible and was "
                    "relaxed to the hard-safe/recovery condition. "
                    f"Strict solve: {strict_nonlinear_error}. "
                    f"{nonlinear_message2}"
                )
                nonlinear_required = relaxed_required
                rate_relaxed = True

            (
                filtered_next_states,
                h_filtered_next,
                hdot_filtered_next,
                psi_filtered_next,
            ) = self._next_state_quantities(
                data,
                u_safe,
            )

            psi_residual_active = (
                psi_filtered_next[
                    active_indices
                ]
                - nonlinear_required
            )

            psi_residual = np.zeros_like(
                psi_now,
                dtype=float,
            )
            psi_residual[
                active_indices
            ] = psi_residual_active

            if (
                np.min(
                    psi_residual_active
                )
                < -self.psi_tolerance_mps
            ):
                local_idx = int(
                    np.argmin(
                        psi_residual_active
                    )
                )
                global_idx = int(
                    active_indices[
                        local_idx
                    ]
                )
                raise CBFInfeasibleError(
                    linear_failure_detail
                    + "Nonlinear fallback returned an "
                    "unvalidated command for "
                    f"'{self.barrier_names[global_idx]}': "
                    f"residual="
                    f"{psi_residual_active[local_idx]:+.3e} m/s."
                )

            h_active_next = (
                h_filtered_next[
                    active_indices
                ]
            )

            if (
                np.min(h_active_next)
                < -self.recovery_tolerance_m
            ):
                local_idx = int(
                    np.argmin(
                        h_active_next
                    )
                )
                global_idx = int(
                    active_indices[
                        local_idx
                    ]
                )
                raise SafetyViolation(
                    "Nonlinear HOCBF fallback would leave "
                    "the permitted geometric recovery band "
                    f"for ACTIVE pair "
                    f"'{self.barrier_names[global_idx]}': "
                    f"h_next="
                    f"{h_filtered_next[global_idx]:+.6f} m."
                )

            message = (
                f"{message} {linear_failure_detail}"
                f"{nonlinear_message}"
            )

        return CBFFilterResult(
            control=u_safe,
            nominal_control=u_nom.copy(),
            modified=bool(
                np.linalg.norm(u_safe - u_nom) > 1e-9
            ),
            barriers_before=before,
            barriers_nominal_next=nominal_next_states,
            barriers_filtered_next=filtered_next_states,
            constraint_residuals_m=psi_residual,
            rate_relaxed=rate_relaxed,
            solver_message=message,
        )
