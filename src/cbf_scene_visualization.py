"""Visualize the *actual* single-nudge geometry and all virtual CBF ellipsoids.

This reads poses and geometry from the loaded MuJoCo model and MjData. It does
not change dynamics, collision filtering, actuator controls or CBF constraints.
The asteroid surface comes from the exact OBJ referenced by this project's XML.

Two visualization modes:
  1. Matplotlib 3-D solid geometry (works without OpenGL).
  2. Optional MuJoCo scene snapshot (requires working OpenGL/EGL context).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Circle, Wedge
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


BLUE = (0.18, 0.43, 0.98)
ORANGE = (1.0, 0.43, 0.08)
CYAN = (0.03, 0.62, 0.72)  # visible keep-in geometry (XML)
RED = (0.93, 0.28, 0.27)   # bus-center minimum separation, CBF
VIOLET = (0.49, 0.27, 0.78)  # desired observation radius (not a CBF)
TEAL = (0.04, 0.52, 0.48)  # bus-center maximum separation, CBF


def load_obj_triangles(obj_path: str | Path, scale: float = 1.0):
    """Read triangular OBJ faces (triangulate polygons with a fan if needed)."""
    vertices = []
    faces = []
    with Path(obj_path).open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                pieces = line.split()[1:]
                if len(pieces) < 3:
                    continue
                ids = []
                for item in pieces:
                    val = int(item.split("/")[0])
                    ids.append(val - 1 if val > 0 else len(vertices) + val)
                for k in range(1, len(ids) - 1):
                    faces.append((ids[0], ids[k], ids[k + 1]))
    vertices = np.asarray(vertices, dtype=float) * float(scale)
    faces = np.asarray(faces, dtype=int)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(faces) == 0:
        raise ValueError("OBJ is missing 3-D vertices or polygon faces")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("OBJ face index outside vertex array")
    return vertices, faces


def _ellipsoid_xyz(center, rotation, radii, n_lat=23, n_lon=39):
    u, v = np.meshgrid(
        np.linspace(0, 2 * np.pi, n_lon),
        np.linspace(0, np.pi, n_lat),
    )
    xyz = np.stack(
        (radii[0] * np.sin(v) * np.cos(u),
         radii[1] * np.sin(v) * np.sin(u),
         radii[2] * np.cos(v)),
        axis=0,
    )
    world = rotation @ xyz.reshape(3, -1) + center.reshape(3, 1)
    return world.reshape((3,) + u.shape)


def _uv_primitive(radius, half_length=0.0, capsule=False, n_theta=14, n_phi=22):
    """Unit sphere or z-aligned MuJoCo capsule surface as triangles."""
    t = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    v = []
    for theta in t:
        s = np.sin(theta)
        z = radius * np.cos(theta)
        if capsule:
            z += half_length if theta <= 0.5 * np.pi else -half_length
        v.extend([[radius*s*np.cos(p), radius*s*np.sin(p), z] for p in phi])
    faces = []
    for i in range(n_theta - 1):
        for j in range(n_phi):
            a = i*n_phi + j
            b = i*n_phi + (j + 1) % n_phi
            c = (i+1)*n_phi + j
            d = (i+1)*n_phi + (j + 1) % n_phi
            faces.extend([(a, b, c), (b, d, c)])
    return np.asarray(v), np.asarray(faces)


def _box_primitive(sizes):
    x, y, z = np.asarray(sizes[:3], dtype=float)
    v = np.array([[sx*x, sy*y, sz*z]
                  for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)], dtype=float)
    face_quads = [(0,1,3,2), (4,6,7,5), (0,4,5,1),
                  (2,3,7,6), (0,2,6,4), (1,5,7,3)]
    faces = [(a,b,c) for a,b,c,d in face_quads] + [(a,c,d) for a,b,c,d in face_quads]
    return v, np.asarray(faces, dtype=int)


def _physical_color(model, geom_id):
    # Material colors are not always propagated to geom_rgba, so check the mat ID.
    matid = int(model.geom_matid[geom_id])
    if matid >= 0:
        return np.asarray(model.mat_rgba[matid, :3], dtype=float)
    return np.asarray(model.geom_rgba[geom_id, :3], dtype=float)


def _add_solid_mesh(ax, vertices_world, faces, color, alpha=1.0):
    triangles = vertices_world[faces]
    a = triangles[:,1] - triangles[:,0]
    b = triangles[:,2] - triangles[:,0]
    n = np.cross(a, b)
    lengths = np.linalg.norm(n, axis=1)
    n /= np.maximum(lengths[:,None], 1e-12)
    light = np.array([0.34, -0.42, 0.84])
    light /= np.linalg.norm(light)
    # Two-sided lighting so winding in an OBJ does not produce a black mesh.
    brightness = 0.58 + 0.42 * np.abs(n @ light)
    rgba = np.column_stack([
        np.clip(brightness[:,None] * np.asarray(color)[None,:], 0, 1),
        np.full(len(faces), alpha),
    ])
    collection = Poly3DCollection(triangles, facecolors=rgba,
                                  edgecolors="none", linewidths=0.0)
    ax.add_collection3d(collection)


def _get_geom_name(model, gid, mujoco):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"


def _add_real_geoms(ax, model, data, mujoco, *, draw_spacecraft=True,
                    draw_asteroid=True, asteroid_obj=None, mesh_scale=1.0):
    names_drawn = []
    types = mujoco.mjtGeom
    for gid in range(int(model.ngeom)):
        name = _get_geom_name(model, gid, mujoco)
        # All CBF geometry is virtual, irrespective of its MuJoCo group.
        # In particular, cbf_asteroid_keepin is group 4 and can be 9.5 m
        # across in radius. Never render it as an opaque physical sphere.
        if int(model.geom_group[gid]) == 5 or name.startswith("cbf_"):
            continue
        is_asteroid = name == "asteroid_geom"
        if is_asteroid and not draw_asteroid:
            continue
        if not is_asteroid and not draw_spacecraft:
            continue
        gtype = int(model.geom_type[gid])
        if is_asteroid:
            if asteroid_obj is None:
                raise ValueError("An asteroid_obj path is needed to draw the actual asteroid mesh")
            vertices, faces = load_obj_triangles(asteroid_obj, mesh_scale)
            # The XML uses the same asteroid-local origin for this OBJ and the CBF.
            bid = int(model.geom_bodyid[gid])
            R = np.asarray(data.xmat[bid]).reshape(3,3)
            p = np.asarray(data.xpos[bid], dtype=float)
            vertices_world = vertices @ R.T + p
            rgb = _physical_color(model, gid)
        elif gtype == int(types.mjGEOM_BOX):
            vertices, faces = _box_primitive(model.geom_size[gid])
            vertices_world = vertices @ np.asarray(data.geom_xmat[gid]).reshape(3,3).T + data.geom_xpos[gid]
            rgb = _physical_color(model, gid)
        elif gtype == int(types.mjGEOM_CAPSULE):
            radius = float(model.geom_size[gid,0])
            half_length = float(model.geom_size[gid,1])
            vertices, faces = _uv_primitive(radius, half_length, capsule=True)
            vertices_world = vertices @ np.asarray(data.geom_xmat[gid]).reshape(3,3).T + data.geom_xpos[gid]
            rgb = _physical_color(model, gid)
        elif gtype == int(types.mjGEOM_SPHERE):
            vertices, faces = _uv_primitive(float(model.geom_size[gid,0]))
            vertices_world = vertices @ np.asarray(data.geom_xmat[gid]).reshape(3,3).T + data.geom_xpos[gid]
            rgb = _physical_color(model, gid)
        else:
            # The supplied physical model uses boxes, capsules, a pad sphere and mesh.
            # Any other primitive is skipped rather than drawing fake geometry.
            continue
        _add_solid_mesh(ax, vertices_world, faces, rgb)
        names_drawn.append(name)
    return names_drawn


def _add_virtual_proxies(ax, model, data, mujoco, *, draw_spacecraft=True,
                         draw_asteroid=True, alpha=0.19):
    names = []
    for gid in range(int(model.ngeom)):
        if int(model.geom_group[gid]) != 5:
            continue
        name = _get_geom_name(model, gid, mujoco)
        is_asteroid = name == "cbf_asteroid_keepout"
        if is_asteroid and not draw_asteroid:
            continue
        if not is_asteroid and not draw_spacecraft:
            continue
        if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            continue
        xyz = _ellipsoid_xyz(
            np.asarray(data.geom_xpos[gid], dtype=float),
            np.asarray(data.geom_xmat[gid], dtype=float).reshape(3,3),
            np.asarray(model.geom_size[gid,:3], dtype=float),
        )
        color = ORANGE if is_asteroid else BLUE
        ax.plot_surface(*xyz, color=color,
                        alpha=(0.23 if is_asteroid else alpha),
                        linewidth=0, shade=True, antialiased=False)
        names.append(name)
    return names


def _set_view(ax, center, extents, *, azim=-65, elev=21):
    center = np.asarray(center, dtype=float)
    extents = np.asarray(extents, dtype=float)
    ax.set_xlim(center[0] - extents[0], center[0] + extents[0])
    ax.set_ylim(center[1] - extents[1], center[1] + extents[1])
    ax.set_zlim(center[2] - extents[2], center[2] + extents[2])
    ax.set_box_aspect(extents)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x [m]',labelpad=3)
    ax.set_ylabel('y [m]',labelpad=3)
    ax.set_zlabel('z [m]',labelpad=3)
    ax.tick_params(labelsize=8, pad=0)


def _draw_shell_wireframe(ax, center, radius, color, *, alpha=0.12,
                          line_width=0.65):
    """Render a translucent spherical *surface* as sparse latitude/longitude lines.

    This creates no solid faces; the transparent safety boundaries cannot
    cover or recolor the physical spacecraft or asteroid mesh.
    """
    xyz = _ellipsoid_xyz(
        np.asarray(center, dtype=float), np.eye(3),
        np.repeat(float(radius), 3), n_lat=25, n_lon=49,
    )
    ax.plot_wireframe(
        *xyz, rstride=4, cstride=6,
        color=color, linewidth=line_width, alpha=float(alpha),
    )


def _draw_shell_transparent_surface(ax, center, radius, color, *, alpha=0.04):
    """Add a faint 3-D skin, not a filled/opaque physical sphere."""
    xyz = _ellipsoid_xyz(
        np.asarray(center, dtype=float), np.eye(3),
        np.repeat(float(radius), 3), n_lat=17, n_lon=33,
    )
    ax.plot_surface(*xyz, color=color, alpha=float(alpha),
                    linewidth=0.0, shade=False, antialiased=False,
                    rstride=1, cstride=1)


def _plot_annular_cross_section(ax, shell, spacecraft_world, asteroid_world,
                                 model=None, data=None, mujoco=None):
    """Schematic cross-section: orange keep-out, red forbidden band, teal shell."""
    inner, outer = shell['inner'], shell['outer']
    if inner is None or outer is None:
        raise ValueError('Pass cbf_filter=exp.cbf for both true radial CBFs')
    limit = outer * 1.22
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_facecolor('white')
    ax.add_patch(Circle((0, 0), outer, facecolor='#d7f0ec',
                        edgecolor=TEAL, linewidth=2.4, zorder=1))
    ax.add_patch(Circle((0, 0), inner, facecolor='#fff0f0',
                        edgecolor=RED, linewidth=2.3, zorder=2))
    ko_radius = 0.0
    if model is not None and mujoco is not None:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                'cbf_asteroid_keepout')
        if gid >= 0:
            # Ellipsoid's largest semiaxis shown as a schematic circular
            # envelope; the actual 3D collision CBF uses the true ellipsoid.
            ko_radius = float(np.max(np.asarray(model.geom_size[gid, :3])))
    if ko_radius > 0:
        ax.add_patch(Circle((0, 0), ko_radius, facecolor=ORANGE,
                            alpha=0.27, edgecolor=ORANGE, linewidth=2, zorder=4))
    symbol_r = max(ko_radius * .70, inner * .025)
    ax.add_patch(Circle((0, 0), symbol_r, facecolor='#806b59',
                        edgecolor='#654f40', linewidth=.8, zorder=5))
    sep = float(np.linalg.norm(np.asarray(spacecraft_world) - np.asarray(asteroid_world)))
    ax.scatter([sep], [0], color='#2366ba', edgecolor='white', lw=1.3, s=95, zorder=8)
    ax.plot([0, sep], [0, 0], linestyle='--', linewidth=1.1,
            color='#98a6b6', zorder=3)
    if data is not None:
        ax.text(.5, -.06,
                f'At t = {float(data.time):.1f} s, the spacecraft bus center is '
                f'{sep:.2f} m from the asteroid center.',
                transform=ax.transAxes, ha='center', va='top', fontsize=8,
                color='#445566', clip_on=False)
    ax.set_title('Asteroid-centered operating shell', fontsize=13, weight='semibold', pad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)


def _draw_shell_key(ax, shell):
    """Three CBFs only: collision keep-out, radial minimum, radial maximum."""
    ax.set_axis_off()
    inner, outer = shell['inner'], shell['outer']
    if inner is None or outer is None:
        ax.text(.02, .9, 'Pass cbf_filter=exp.cbf', transform=ax.transAxes)
        return
    labels = [
        (ORANGE, 'ASTEROID KEEP-OUT CBF',
         'Bus may not intersect the\nasteroid collision envelope.'),
        (RED, f'MINIMUM-SEPARATION CBF · {inner:.2f} m',
         'After departure, bus center must\nremain outside the red boundary.'),
        (TEAL, f'MAXIMUM-SEPARATION CBF · {outer:.2f} m',
         'From reset, bus center must\nremain inside the teal boundary.'),
    ]
    for y,(color,heading,detail) in zip((.87,.59,.31),labels):
        ax.plot([.02,.14],[y-.012,y-.012],color=color,lw=3,
                solid_capstyle='round',transform=ax.transAxes)
        ax.text(.19,y,heading,fontsize=8.5,fontweight='bold',
                color=color,ha='left',va='top',transform=ax.transAxes)
        ax.text(.19,y-.072,detail,fontsize=8.4,
                color='#43505c',ha='left',va='top',transform=ax.transAxes)
    ax.text(.02,.08,
        ('Inner: ACTIVE' if shell.get('shell_active') else 'Inner: pending departure')
        + '  |  ' + ('Outer: ACTIVE' if shell.get('keepin_active') else 'Outer: inactive'),
        fontsize=8.2,color='#43505c',transform=ax.transAxes)


def _read_shell_radii(model, data, mujoco, cbf_filter,
                      preferred_radius_m, shell_inner_radius_m,
                      shell_outer_radius_m, setpoint_buffer_m):
    """Read *enforced* 3.25 m and 4.00 m radii; no preferred radius."""
    if cbf_filter is None and shell_inner_radius_m is None:
        return None
    gid = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'asteroid_geom')
    if gid < 0: raise ValueError('Expected asteroid_geom')
    center = np.asarray(data.geom_xpos[gid],dtype=float).copy()
    inner = shell_inner_radius_m if shell_inner_radius_m is not None else getattr(cbf_filter,'shell_min_center_m',None)
    outer = shell_outer_radius_m if shell_outer_radius_m is not None else getattr(cbf_filter,'shell_max_center_m',None)
    return dict(center=center,inner=None if inner is None else float(inner),
                outer=None if outer is None else float(outer), preferred=None,
                shell_active=bool(getattr(cbf_filter,'shell_active',False)),
                keepin_active=bool(getattr(cbf_filter,'keepin_active',False)))


def plot_actual_cbf_geometry(model, data, asteroid_obj, *,
                             mesh_scale=0.921956130974350,
                             save_path=None,
                             title='MuJoCo geometry with all CBF proxies',
                             show_keepin=True,
                             keepin_wire_alpha=0.10,
                             cbf_filter=None,
                             preferred_radius_m=None,
                             setpoint_buffer_m=None,
                             shell_inner_radius_m=None,
                             shell_outer_radius_m=None):
    """Render actual geometry and color-coded transparent nested safe-set bounds.

    Color convention:
      blue:     original arm/bus self-collision ellipsoids
      orange:   asteroid keep-out ellipsoid (real XML proxy)
      coral:    radial minimum *bus-center* clearance CBF (after activation)
      teal:    maximum bus-center distance CBF (active from reset)
      no preferred radius

    Large 3-D boundaries are faint translucent surfaces plus sparse wireframes.
    The fifth panel displays a strongly shaded true-radii 2-D cross-section.
    This is a visualization-only operation.

    ``cbf_filter=exp.cbf`` provides the active minimum radial CBF value. The original model-only call still works,
    but requires the filter to display the nested radial CBFs.
    """
    import mujoco
    mujoco.mj_forward(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'spacecraft')
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'asteroid')
    if sid < 0 or aid < 0:
        raise ValueError("Expected bodies 'spacecraft' and 'asteroid'")
    s = np.asarray(data.xpos[sid], dtype=float).copy()
    a = np.asarray(data.xpos[aid], dtype=float).copy()
    shell = _read_shell_radii(
        model, data, mujoco, cbf_filter, preferred_radius_m,
        shell_inner_radius_m, shell_outer_radius_m, setpoint_buffer_m,
    ) if show_keepin else None
    with_shell = shell is not None

    if with_shell:
        # Top: actual physics model, arm and asteroid.
        # Bottom: 3-D transparent CBF shells, and a strongly shaded 2-D section.
        fig = plt.figure(figsize=(18, 12.5), facecolor='white')
        grid = fig.add_gridspec(2, 6, height_ratios=(0.95, 1.12),
                               left=0.035, right=0.97, bottom=0.105,
                               top=0.925, hspace=0.31, wspace=0.38)
        ax_all = fig.add_subplot(grid[0, 0:2], projection='3d')
        ax_sc = fig.add_subplot(grid[0, 2:4], projection='3d')
        ax_ast = fig.add_subplot(grid[0, 4:6], projection='3d')
        ax_shell = fig.add_subplot(grid[1, 0:3], projection='3d')
        right_grid = grid[1, 3:6].subgridspec(1, 5, wspace=0.02)
        ax_section = fig.add_subplot(right_grid[0, 0:3])
        ax_key = fig.add_subplot(right_grid[0, 3:5])
        axes = [ax_all, ax_sc, ax_ast, ax_shell, ax_section, ax_key]
    else:
        fig = plt.figure(figsize=(18, 6.2), facecolor='white')
        axes = [fig.add_subplot(1, 3, k, projection='3d') for k in range(1, 4)]
        ax_all, ax_sc, ax_ast = axes
        ax_shell = None
        ax_section = None
        ax_key = None

    drawn = {}
    for ax, physical_sc, physical_ast, label in [
        (ax_all, True, True, 'overview'),
        (ax_sc, True, False, 'spacecraft'),
        (ax_ast, False, True, 'asteroid'),
    ]:
        drawn[label] = {}
        drawn[label]['physical'] = _add_real_geoms(
            ax, model, data, mujoco,
            draw_spacecraft=physical_sc, draw_asteroid=physical_ast,
            asteroid_obj=asteroid_obj, mesh_scale=mesh_scale,
        )
        drawn[label]['cbf'] = _add_virtual_proxies(
            ax, model, data, mujoco,
            draw_spacecraft=physical_sc, draw_asteroid=physical_ast,
            alpha=0.13,
        )

    sep = float(np.linalg.norm(a - s))
    mid = (s + a) / 2.0
    _set_view(ax_all, mid, (max(2.2, sep / 2 + 1.9), 2.0, 1.9),
              azim=-64, elev=20)
    # Offset the arm close-up toward the actual arm's approximate direction.
    sc_extents = (3.0, 2.0, 1.8)
    _set_view(ax_sc, s + np.array([0.65, 0.0, 0.0]), sc_extents,
              azim=-64, elev=22)
    _set_view(ax_ast, a, (0.85, 0.85, 0.85), azim=-65, elev=22)
    ax_all.set_title('Actual spacecraft + asteroid', fontsize=12, pad=12)
    ax_sc.set_title('Physical arm + all blue self-collision proxies', fontsize=12, pad=12)
    ax_ast.set_title('Real asteroid OBJ + orange keep-out', fontsize=12, pad=12)

    if with_shell:
        drawn['keepin_shell'] = {'physical': [], 'cbf': [], 'radii_m': shell}
        center = shell['center']
        alpha = float(keepin_wire_alpha)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError('keepin_wire_alpha must be between 0 and 1')

        # Low-opacity 3-D surfaces illustrate containment without hiding
        # physical bodies; the true filled annulus is shown clearly at right.
        if shell['outer'] is not None:
            _draw_shell_transparent_surface(ax_shell, center, shell['outer'], TEAL,
                                            alpha=0.032)
        if shell['inner'] is not None:
            _draw_shell_transparent_surface(ax_shell, center, shell['inner'], RED,
                                            alpha=0.047)

        # Draw the largest boundary first; sparse lines avoid occlusion.
        if shell['outer'] is not None:
            _draw_shell_wireframe(ax_shell, center, shell['outer'], TEAL,
                                  alpha=min(1.0, alpha * 1.5), line_width=0.85)
            drawn['keepin_shell']['cbf'].append('outer_bus_center_cbf')
        if shell['inner'] is not None:
            _draw_shell_wireframe(ax_shell, center, shell['inner'], RED,
                                  alpha=min(1.0, alpha * 2.0), line_width=0.90)
            drawn['keepin_shell']['cbf'].append('inner_bus_center_cbf')
        if shell['preferred'] is not None:
            _draw_shell_wireframe(ax_shell, center, shell['preferred'], VIOLET,
                                  alpha=min(1.0, alpha * 1.6), line_width=0.7)
            drawn['keepin_shell']['cbf'].append('preferred_reference_not_cbf')

        ax_shell.scatter(*a, color='#6d5a4a', s=65, depthshade=False)
        ax_shell.scatter(*s, color='#2465a8', s=70, depthshade=False)
        ax_shell.plot([a[0], s[0]], [a[1], s[1]], [a[2], s[2]],
                      color='#626e7d', linestyle='--', alpha=0.65, lw=1.2)
        _set_view(ax_shell, center, (shell['outer'] * 1.25,) * 3,
                  azim=-65, elev=20)
        ax_shell.set_title('Bus-center minimum and maximum CBFs', fontsize=12, pad=12)
        _plot_annular_cross_section(
            ax_section, shell, s, a, model=model, data=data, mujoco=mujoco,
        )
        _draw_shell_key(ax_key, shell)

        lines = [f'Bus–asteroid separation: {sep:.2f} m']
        if shell['inner'] is not None:
            lines.append(f'Red: minimum bus-center r = {shell["inner"]:.2f} m')
        if shell['preferred'] is not None:
            lines.append(f'Violet: preferred r = {shell["preferred"]:.2f} m')
        if shell['outer'] is not None:
            lines.append(f'Teal: maximum bus-center r = {shell["outer"]:.2f} m')

        if cbf_filter is not None and not getattr(cbf_filter, 'shell_active', True):
            lines.append('Minimum CBF shown but not yet active')
        ax_shell.text2D(0.02, 0.97, '\n'.join(lines), transform=ax_shell.transAxes,
                        va='top', fontsize=8.5,
                        bbox=dict(facecolor='white', alpha=0.78, edgecolor='none'))

    for ax in (ax_all, ax_sc, ax_ast, ax_shell) if with_shell else axes:
        ax.grid(True, alpha=0.12)
        ax.set_facecolor('white')

    fig.suptitle(f'{title}   |   t = {float(data.time):.2f} s',
                 fontsize=15, y=0.975)
    handles = [
        Patch(facecolor=(0.7, 0.72, 0.75), label='Physical spacecraft/arm/asteroid'),
        Patch(facecolor=BLUE, alpha=0.20, label='Arm/bus self-collision CBFs'),
        Patch(facecolor=ORANGE, alpha=0.25, label='Asteroid keep-out CBF'),
    ]
    if with_shell:
        if shell['inner'] is not None:
            handles.append(Patch(facecolor=RED, alpha=0.35, label='Minimum separation CBF'))
        if shell['outer'] is not None:
            handles.append(Patch(facecolor=TEAL, alpha=0.35, label='Maximum separation CBF'))

    fig.legend(handles=handles, loc='lower center', ncol=3 if with_shell else 3,
               frameon=False, bbox_to_anchor=(0.5, 0.004), fontsize=9)
    if not with_shell:
        fig.subplots_adjust(left=0.01, right=0.99, top=0.85,
                            bottom=0.17, wspace=0.04)

    all_cbf = drawn['overview']['cbf']
    if 'cbf_asteroid_keepout' not in all_cbf or len(all_cbf) <= 2:
        raise RuntimeError(f'Expected arm CBF proxies and asteroid keep-out: {all_cbf}')
    if 'asteroid_geom' not in drawn['overview']['physical']:
        raise RuntimeError('Actual asteroid mesh was not drawn')
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='white')
    return fig, drawn


def render_mujoco_cbf_snapshot(model, data, *, save_path=None,
                                width=1100,height=680, azimuth=105,
                                elevation=-17, camera_distance=7.5):
    """Optional real MuJoCo pixel render, with all virtual group-5 ellipsoids."""
    import mujoco
    from PIL import Image

    mujoco.mj_forward(model, data)
    sid = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'spacecraft')
    aid = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'asteroid')
    lookat = 0.5*(np.asarray(data.xpos[sid]) + np.asarray(data.xpos[aid]))
    opts = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opts)
    opts.geomgroup[:] = 1
    # The retired outer shell visualization group remains disabled.
    opts.geomgroup[4] = 0
    opts.geomgroup[5] = 1
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = lookat
    cam.distance = max(camera_distance, np.linalg.norm(data.xpos[sid]-data.xpos[aid])*1.8)
    cam.azimuth = float(azimuth)
    cam.elevation = float(elevation)
    with mujoco.Renderer(model,width=int(width),height=int(height)) as renderer:
        renderer.update_scene(data,camera=cam,scene_option=opts)
        rgb = renderer.render().copy()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(save_path)
    return rgb
