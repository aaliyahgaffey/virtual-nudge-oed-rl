"""Numerically guarded convex-hull ray intersect; geometric origin must lie inside."""
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
class Surface:
    def __init__(self, obj_path, scale=0.921956130974350):
        vertices=[]
        for line in Path(obj_path).read_text().splitlines():
            if line.startswith('v '):vertices.append([float(x) for x in line.split()[1:4]])
        self.vertices=np.asarray(vertices,float)*scale
        if self.vertices.ndim!=2 or self.vertices.shape[1]!=3 or len(self.vertices)<4 or not np.isfinite(self.vertices).all():
            raise ValueError('Invalid asteroid vertices')
        hull=ConvexHull(self.vertices)
        self.A=np.ascontiguousarray(hull.equations[:,:3],dtype=np.float64)
        self.b=np.ascontiguousarray(hull.equations[:,3],dtype=np.float64)
        if not np.isfinite(self.A).all() or not np.isfinite(self.b).all() or self.b.max()>=-1e-10:
            raise ValueError('Convex hull must contain body origin strictly')
    def point_and_normal(self,lon,z):
        if not np.isfinite([lon,z]).all():raise ValueError('Nonfinite spherical coordinate')
        z=float(np.clip(z,-1.,1.));rho=float(np.sqrt(max(0.,1-z*z)))
        d=np.array([rho*np.cos(lon),rho*np.sin(lon),z],dtype=float)
        with np.errstate(over='raise',divide='raise',invalid='raise'):
            # For tiny 3D dot products, avoid platform-specific BLAS matmul
            # floating-point flags (observed with macOS/NumPy under errstate).
            denom=np.einsum('ij,j->i', self.A, d, optimize=False)
            mask=denom>1e-10
            if not mask.any():raise ValueError('Ray misses asteroid hull')
            idx=np.flatnonzero(mask)
            distances=-self.b[idx]/denom[idx]
            k=int(idx[np.argmin(distances)])
            length=float(-self.b[k]/denom[k])
            point=length*d
            if not np.isfinite(point).all() or np.max(np.einsum('ij,j->i', self.A, point, optimize=False)+self.b)>1e-5:
                raise ValueError('Bad ray/hull intersection')
        normal=self.A[k]/np.linalg.norm(self.A[k])
        return point,normal
