from __future__ import annotations
"""Fixed asteroid keep-out geometry used in the combined ellipsoidal HOCBF.

The actual control filter is the project's EllipsoidalCBFSafetyFilter:
this module simply defines the target pair and checks the mesh enclosure.
A single filter jointly enforces self-collision and the bus/target barrier.
"""
from pathlib import Path
import numpy as np

ASTEROID_KEEP_OUT_PAIR = ("cbf_bus", "cbf_asteroid_keepout")
KEEP_OUT_SEMIAXES_M = np.array([0.60, 0.60, 0.64])
ASTEROID_MESH_SCALE = 0.921956130974350


def verify_obj_enclosure(obj_path: str | Path, semiaxes=KEEP_OUT_SEMIAXES_M,
                         mesh_scale=ASTEROID_MESH_SCALE) -> dict:
    """Development-time check only; do NOT use mesh truth in an onboard filter.

    Tests whether each scaled OBJ vertex is inside the known fixed envelope.
    The ellipsoid and mesh share the asteroid body's nominal mesh frame here.
    """
    vertices=[]
    with Path(obj_path).open() as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
    if not vertices:
        raise ValueError(f"No vertices in {obj_path}")
    v=np.asarray(vertices,dtype=float)*float(mesh_scale)
    a=np.asarray(semiaxes,dtype=float).reshape(3)
    if np.any(a<=0): raise ValueError("Semiaxes must be positive")
    max_normalized_radius=float(np.linalg.norm(v/a,axis=1).max())
    return dict(n_vertices=len(v), maximum_ellipsoid_radius=max_normalized_radius,
                encloses_all_vertices=bool(max_normalized_radius<=1.0),
                maximum_vertex_radius_m=float(np.linalg.norm(v,axis=1).max()))


def asteroid_barrier_state(experiment):
    """Return the live bus/asteroid barrier from the combined CBF (no new QP)."""
    if experiment.cbf is None:
        raise RuntimeError("CBF is disabled")
    states=experiment.cbf.barriers(experiment.data)
    for state in states:
        if (state.geom_a_name,state.geom_b_name)==ASTEROID_KEEP_OUT_PAIR:
            return state
    raise RuntimeError("Bus/asteroid pair is missing from the active CBF")


def planned_mesh_contact_point_body(
    obj_path: str | Path,
    surface_normal_body,
    push_direction_body,
    assumed_radius_m: float = 0.50,
    mesh_scale: float = ASTEROID_MESH_SCALE,
):
    """Development-only: find the first convex-mesh surface along the intended push.

    MuJoCo contacts with a single mesh geom use its convex collision hull.
    The planned approach line passes through assumed_radius_m * surface_normal.
    Intersect that same line with the preloaded OBJ's *convex hull* to locate
    the nominal first-contact surface. The mesh is a known planning model;
    NO asteroid mass, inertia, or estimator truth is accessed.

    For a flight system with genuinely unknown geometry, replace this with a
    navigation-derived point and its uncertainty, rather than using mesh truth.
    """
    from scipy.spatial import ConvexHull

    verts = []
    with Path(obj_path).open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(t) for t in line.split()[1:4]])
    if len(verts) < 4:
        raise ValueError(f"Not enough vertices for a mesh hull: {obj_path}")
    vertices = np.asarray(verts, dtype=float) * float(mesh_scale)
    n = np.asarray(surface_normal_body, dtype=float).reshape(3)
    d = np.asarray(push_direction_body, dtype=float).reshape(3)
    n /= np.linalg.norm(n)
    d /= np.linalg.norm(d)
    if np.dot(n, d) >= -0.1:
        raise ValueError("Push must point into the target from this surface normal")

    nominal = float(assumed_radius_m) * n
    # Begin well outside the collision hull, moving in the commanded direction.
    max_r = float(np.max(np.linalg.norm(vertices, axis=1)))
    start = nominal - (2.0 * max_r + 0.5) * d
    hull = ConvexHull(vertices)
    A, b = hull.equations[:, :3], hull.equations[:, 3]
    ad = A @ d
    intercept = A @ start + b
    entering = ad < -1e-12
    leaving = ad > 1e-12
    if not (np.any(entering) and np.any(leaving)):
        raise RuntimeError("Cannot intersect planned ray with convex asteroid mesh")
    t_enter = float(np.max(-intercept[entering] / ad[entering]))
    t_exit = float(np.min(-intercept[leaving] / ad[leaving]))
    if t_exit <= max(t_enter, 0.0) or t_enter < 0.0:
        raise RuntimeError(
            "Planned push line misses the asteroid collision hull. "
            "Choose another surface normal or tangent direction."
        )
    hit = start + t_enter * d
    along_push_shift = float(np.dot(hit - nominal, d))
    return {
        "contact_point_body_m": hit,
        "assumed_surface_point_body_m": nominal,
        "shift_from_assumed_along_push_m": along_push_shift,
        "collision_geometry": "scaled OBJ convex hull (simulation planning only)",
    }
