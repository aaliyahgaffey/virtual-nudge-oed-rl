"""Two asteroid-centered radial CBFs and the existing asteroid keep-out.

The outer bus-center maximum separation r <= 4.00 m is active from the
close-range reset onward. The inner r >= 3.25 m becomes active only after
contact-free departure. No preferred observation radius is imposed.
The signed pad–asteroid gap is also checked; it is NOT an independent CBF.
"""
from __future__ import annotations

import numpy as np
import mujoco

from src.ellipsoidal_cbf_self_collision import (
    BarrierState, EllipsoidalCBFSafetyFilter, SafetyViolation,
)

SHELL_LABEL = "MINIMUM-SEPARATION: bus-center radius"
OUTER_LABEL = "MAXIMUM-SEPARATION: bus-center radius"
KEEPIN_GEOM = "cbf_asteroid_keepin"


class AsteroidKeepInCBFFilter(EllipsoidalCBFSafetyFilter):
    """Legacy class name retained for experiment import compatibility."""

    def __init__(self, *, inner_shell_radius_m: float = 3.25,
                 keepin_radius_m: float = 5.15,
                 keepin_margin_m: float = 0.05,
                 min_pad_clearance_m: float = 0.02, **kwargs):
        super().__init__(**kwargs)
        self.shell_min_center_m = float(inner_shell_radius_m)
        self.min_pad_clearance_m = float(min_pad_clearance_m)
        self.keepin_radius_m = float(keepin_radius_m)
        self.keepin_margin_m = float(keepin_margin_m)
        self.keepin_active = False
        if not np.isfinite(self.shell_min_center_m) or self.shell_min_center_m <= 0:
            raise ValueError("Minimum bus-center radius must be finite and positive")
        if not np.isfinite(self.min_pad_clearance_m) or self.min_pad_clearance_m < 0:
            raise ValueError("Minimum physical pad gap must be nonnegative")
        self.shell_active = False
        self.bus_gid = self._geom_id("cbf_bus")
        self.target_gid = self._geom_id(KEEPIN_GEOM)
        self.asteroid_gid = self._geom_id("asteroid_geom")
        self.keepin_gid = self.target_gid
        self.pad_gid = self._geom_id("nudge_pad_geom")
        self.bus_bid = int(self.model.geom_bodyid[self.bus_gid])
        self.target_bid = int(self.model.geom_bodyid[self.target_gid])
        self.bus_bound_m = float(self.model.geom_rbound[self.bus_gid])
        self.shell_max_center_m = (self.keepin_radius_m - self.keepin_margin_m
                                   - self.bus_bound_m)
        if not (0 < self.shell_min_center_m < self.shell_max_center_m):
            raise ValueError("The minimum radius must be positive and below the maximum")
        if abs(self.shell_max_center_m - 4.0) > 1e-6:
            raise ValueError(f"Expected 4.000 m outer bound, got {self.shell_max_center_m:.3f} m")
        xml_radius = float(self.model.geom_size[self.keepin_gid, 0])
        if int(self.model.geom_type[self.keepin_gid]) != int(mujoco.mjtGeom.mjGEOM_SPHERE):
            raise ValueError("cbf_asteroid_keepin must be an XML sphere")
        if not np.isclose(xml_radius, self.keepin_radius_m, atol=1e-8):
            raise ValueError(f"Keep-in radius mismatch: XML={xml_radius:.3f} m, config={self.keepin_radius_m:.3f} m")
        self.collision_pairs.append(("cbf_bus", "virtual_minimum_radial_boundary"))
        self.pair_geom_ids.append((self.bus_gid, self.target_gid))
        self.pair_body_ids.append((self.bus_bid, self.target_bid))
        self.barrier_names.append(SHELL_LABEL)
        # Second virtual radial row: outer keep-in, same coupled optimization.
        self.collision_pairs.append(("cbf_bus", KEEPIN_GEOM))
        self.pair_geom_ids.append((self.bus_gid, self.target_gid))
        self.pair_body_ids.append((self.bus_bid, self.target_bid))
        self.barrier_names.append(OUTER_LABEL)

    def activate(self, data):
        """Activate the 4 m maximum-distance CBF at close-range reset."""
        self.keepin_active = True
        try:
            _, _, psi1 = self._psi1(data)
            if self.outer_clearance(data) < 0 or psi1[-1] < -self.psi_tolerance_mps:
                raise SafetyViolation("Initial state is not viable for the 4 m maximum CBF")
        except Exception:
            self.keepin_active = False
            raise

    def outer_clearance(self, data):
        return float(self.shell_max_center_m - self._centers(data)[2])

    def _geom_id(self, name):
        gid = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name))
        if gid < 0:
            raise ValueError(f"Missing MuJoCo geom: {name}")
        return gid

    def _centers(self, data):
        bus = np.asarray(data.geom_xpos[self.bus_gid], dtype=float).copy()
        asteroid = np.asarray(data.geom_xpos[self.target_gid], dtype=float).copy()
        return bus, asteroid, float(np.linalg.norm(bus - asteroid))

    def inner_clearance(self, data):
        return float(self._centers(data)[2] - self.shell_min_center_m)

    def pad_surface_clearance(self, data):
        fromto = np.zeros(6)
        return float(mujoco.mj_geomDistance(
            self.model, data, self.pad_gid, self.asteroid_gid, 30.0, fromto
        ))

    def activate_shell(self, data):
        """Enable minimum radius after departure, once HOCBF is viable."""
        if not self.keepin_active:
            raise SafetyViolation("The outer CBF must be active before the inner CBF")
        gap = self.pad_surface_clearance(data)
        # Evaluate the *actual* radial HOCBF viability, not the large
        # inactive placeholder barrier used during deliberate contact.
        self.shell_active = True
        try:
            _, _, psi1 = self._psi1(data)
            inner_h = float(self.inner_clearance(data))
            if (inner_h < 0.02 or psi1[-2] < -self.psi_tolerance_mps
                    or gap < self.min_pad_clearance_m):
                raise SafetyViolation(
                    f"Minimum-radius CBF cannot be activated: "
                    f"h={inner_h:.4f} m, psi1={psi1[-2]:+.4f} m/s, "
                    f"physical pad gap={gap:.4f} m"
                )
        except Exception:
            self.shell_active = False
            raise

    def barriers(self, data):
        mujoco.mj_forward(self.model, data)
        states = []
        for i, ((a, b), (ga, gb)) in enumerate(
                zip(self.collision_pairs[:-2], self.pair_geom_ids[:-2])):
            dist, fromto, normal = self._geom_distance(data, ga, gb)
            states.append(BarrierState(
                geom_name=self.barrier_names[i], geom_a_name=a, geom_b_name=b,
                h_m=float(dist - self.safety_buffer_m),
                nearest_points_world=np.asarray(fromto).copy(),
                normal_world=np.asarray(normal).copy()))
        bus, asteroid, sep = self._centers(data)
        if sep <= 1e-10:
            raise SafetyViolation("Bus and asteroid center coincide")
        outward = (bus - asteroid) / sep
        states.append(BarrierState(
            geom_name=SHELL_LABEL,
            geom_a_name="cbf_bus", geom_b_name="virtual_minimum_radial_boundary",
            h_m=self.inner_clearance(data) if self.shell_active else 1e6,
            nearest_points_world=np.r_[bus, asteroid], normal_world=outward))
        states.append(BarrierState(
            geom_name=OUTER_LABEL,
            geom_a_name="cbf_bus", geom_b_name=KEEPIN_GEOM,
            h_m=self.outer_clearance(data) if self.keepin_active else 1e6,
            nearest_points_world=np.r_[bus, asteroid], normal_world=-outward))
        return states

    def barrier_rates(self, data, states=None):
        if states is None:
            states = self.barriers(data)
        original = super().barrier_rates(data, states[:-2])
        bus, asteroid, sep = self._centers(data)
        n = (bus - asteroid) / max(sep, 1e-10)
        v_bus = self._point_velocity_world(data, self.bus_bid, bus)
        v_asteroid = self._point_velocity_world(data, self.target_bid, asteroid)
        radial_rate = float(n @ (v_bus - v_asteroid))
        return np.r_[original, radial_rate if self.shell_active else 0.0,
                     -radial_rate if self.keepin_active else 0.0]
