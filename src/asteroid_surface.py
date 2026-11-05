"""Convex-hull asteroid surface geometry shared by virtual nudge planning.

This is the original Surface implementation moved out of the legacy
three_nudge_oed_v2.py module without modifying its intersection logic.
"""
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull


class Surface:
    """Convex collision hull, ray-parameterized from geometric origin.

    Every ray direction on S^2 intersects the convex mesh hull boundary,
    covering accessible convex-hull surfaces (not concave crevices).
    """
    def __init__(self, obj_path, scale=0.921956130974350):
        verts=[]
        for line in Path(obj_path).read_text().splitlines():
            if line.startswith('v '): verts.append([float(z) for z in line.split()[1:4]])
        self.vertices=np.asarray(verts, float)*scale
        if len(self.vertices)<4: raise ValueError('No asteroid mesh vertices')
        hull=ConvexHull(self.vertices)
        self.A=hull.equations[:,:3].copy()
        self.b=hull.equations[:,3].copy()
        if np.max(self.b)>=0:
            raise ValueError('Mesh-frame origin must be inside convex hull')

    def point_and_normal(self, lon, z):
        z=float(np.clip(z,-1,1))
        rho=np.sqrt(max(0,1-z*z))
        direction=np.array([rho*np.cos(lon),rho*np.sin(lon),z])
        denom=self.A@direction
        valid=denom>1e-10
        if not np.any(valid): raise ValueError('Cannot intersect hull')
        ts=np.full(len(denom),np.inf)
        ts[valid]=-self.b[valid]/denom[valid]
        j=int(np.argmin(ts))
        pt=float(ts[j])*direction
        normal=self.A[j].copy()
        normal/=np.linalg.norm(normal)
        if not np.all(self.A@pt+self.b<=1e-6): raise ValueError('Hull intersection invalid')
        return pt, normal
