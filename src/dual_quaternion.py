from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------- #
# Quaternion
# --------------------------------------------------------------------------- #
class Quaternion:
    __slots__ = ("w", "v")

    def __init__(self, w: float = 0.0, v=(0.0, 0.0, 0.0)):
        self.w = float(w)
        self.v = np.asarray(v, dtype=float).reshape(3)

    # ---- constructors -----------------------------------------------------
    @staticmethod
    def identity() -> "Quaternion":
        return Quaternion(1.0, (0.0, 0.0, 0.0))

    @staticmethod
    def zero() -> "Quaternion":
        return Quaternion(0.0, (0.0, 0.0, 0.0))

    @staticmethod
    def from_vec3(v) -> "Quaternion":
        """Pure/vector quaternion representation of a 3-vector (Hv)."""
        return Quaternion(0.0, v)

    @staticmethod
    def from_wxyz(arr) -> "Quaternion":
        arr = np.asarray(arr, dtype=float).reshape(4)
        return Quaternion(arr[0], arr[1:4])

    @staticmethod
    def from_mujoco_xquat(xquat) -> "Quaternion":
        """MuJoCo stores quats as (w, x, y, z) already, so this is an alias."""
        return Quaternion.from_wxyz(xquat)

    # ---- representation -----------------------------------------------------
    def as_wxyz(self) -> np.ndarray:
        return np.concatenate(([self.w], self.v))

    def copy(self) -> "Quaternion":
        return Quaternion(self.w, self.v.copy())

    def __repr__(self):
        return f"Quaternion(w={self.w:.6f}, v={self.v})"

    # ---- Table I operations -------------------------------------------------
    def __add__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(self.w + other.w, self.v + other.v)

    def __sub__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(self.w - other.w, self.v - other.v)

    def __neg__(self) -> "Quaternion":
        return Quaternion(-self.w, -self.v)

    def __rmul__(self, scalar: float) -> "Quaternion":
        # scalar * quaternion
        if isinstance(scalar, (int, float, np.floating)):
            return Quaternion(scalar * self.w, scalar * self.v)
        return NotImplemented

    def __mul__(self, other):
        # quaternion * quaternion (Hamilton product) or quaternion * scalar
        if isinstance(other, (int, float, np.floating)):
            return Quaternion(self.w * other, self.v * other)
        if isinstance(other, Quaternion):
            a0, a = self.w, self.v
            b0, b = other.w, other.v
            w = a0 * b0 - np.dot(a, b)
            v = a0 * b + b0 * a + np.cross(a, b)
            return Quaternion(w, v)
        return NotImplemented

    def conjugate(self) -> "Quaternion":
        return Quaternion(self.w, -self.v)

    def dot(self, other: "Quaternion") -> "Quaternion":
        """Table I dot product: scalar-valued quaternion (0 vector part)."""
        return Quaternion(self.w * other.w + np.dot(self.v, other.v), (0.0, 0.0, 0.0))

    def cross(self, other: "Quaternion") -> "Quaternion":
        """Table I cross product."""
        w = 0.0
        v = self.w * other.v + other.w * self.v + np.cross(self.v, other.v)
        return Quaternion(w, v)

    def norm(self) -> float:
        d = self.dot(self)
        return float(np.sqrt(max(d.w, 0.0)))

    def normalize(self) -> "Quaternion":
        n = self.norm()
        if n < 1e-14:
            raise ValueError("Cannot normalize a near-zero quaternion")
        return Quaternion(self.w / n, self.v / n)

    def inverse(self) -> "Quaternion":
        """Unit-quaternion inverse == conjugate. General inverse divides by |q|^2."""
        n2 = self.w ** 2 + np.dot(self.v, self.v)
        if n2 < 1e-14:
            raise ValueError("Cannot invert a near-zero quaternion")
        c = self.conjugate()
        return Quaternion(c.w / n2, c.v / n2)

    def skew4(self) -> np.ndarray:
        """4x4 [.]x operator such that skew4(s) @ a_wxyz == (s x a) as a 4-vector,
        matching Eq. (1) for the quaternion case (s is a pure/vector quaternion)."""
        sx, sy, sz = self.v
        S = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -sz, sy],
            [0.0, sz, 0.0, -sx],
            [0.0, -sy, sx, 0.0],
        ])
        return S


# --------------------------------------------------------------------------- #
# Dual Quaternion
# --------------------------------------------------------------------------- #
class DualQuaternion:
    __slots__ = ("qr", "qd")

    def __init__(self, qr: Quaternion = None, qd: Quaternion = None):
        self.qr = qr.copy() if qr is not None else Quaternion.identity()
        self.qd = qd.copy() if qd is not None else Quaternion.zero()

    # ---- constructors -------------------------------------------------------
    @staticmethod
    def identity() -> "DualQuaternion":
        return DualQuaternion(Quaternion.identity(), Quaternion.zero())

    @staticmethod
    def zero() -> "DualQuaternion":
        return DualQuaternion(Quaternion.zero(), Quaternion.zero())

    @staticmethod
    def from_pos_quat(pos, quat_wxyz) -> "DualQuaternion":
        """Build the *pose* dual quaternion q_B/I from a position (3,) and an
        orientation quaternion (w,x,y,z), following

            q_r  = quat_wxyz            (orientation of B rel. I)
            q_d  = 1/2 * q_r * r        (r = position expressed in B-frame)

        `pos` here is expressed in the SAME frame as MuJoCo reports it
        (world/inertial frame). We rotate it into body coordinates first,
        since the paper defines r^B_{B/I} in B-frame coordinates.
        """
        qr = Quaternion.from_wxyz(quat_wxyz).normalize()
        pos_world = Quaternion.from_vec3(pos)
        # rotate world->body: v_B = q* v_I q   (adjoint, since q = q_B/I)
        pos_body = qr.conjugate() * pos_world * qr
        qd = 0.5 * (qr * pos_body)
        return DualQuaternion(qr, qd)

    def to_pos_quat(self):
        """Inverse of from_pos_quat: returns (pos_world (3,), quat_wxyz (4,))."""
        qr = self.qr.normalize()
        # r^B = 2 * qr^-1 * qd  (pure quaternion, position in body frame)
        pos_body_q = 2.0 * (qr.inverse() * self.qd)
        pos_world_q = qr * pos_body_q * qr.conjugate()
        return pos_world_q.v.copy(), qr.as_wxyz()

    def copy(self) -> "DualQuaternion":
        return DualQuaternion(self.qr.copy(), self.qd.copy())

    def __repr__(self):
        return f"DualQuaternion(qr={self.qr}, qd={self.qd})"

    def as_vec8(self) -> np.ndarray:
        """[qr(4); qd(4)] as used for E158/E145 style row-selection matrices."""
        return np.concatenate([self.qr.as_wxyz(), self.qd.as_wxyz()])

    # ---- Table II operations -------------------------------------------------
    def __add__(self, other: "DualQuaternion") -> "DualQuaternion":
        return DualQuaternion(self.qr + other.qr, self.qd + other.qd)

    def __sub__(self, other: "DualQuaternion") -> "DualQuaternion":
        return DualQuaternion(self.qr - other.qr, self.qd - other.qd)

    def __neg__(self) -> "DualQuaternion":
        return DualQuaternion(-self.qr, -self.qd)

    def __rmul__(self, scalar: float) -> "DualQuaternion":
        if isinstance(scalar, (int, float, np.floating)):
            return DualQuaternion(scalar * self.qr, scalar * self.qd)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float, np.floating)):
            return DualQuaternion(self.qr * other, self.qd * other)
        if isinstance(other, DualQuaternion):
            qr = self.qr * other.qr
            qd = self.qd * other.qr + self.qr * other.qd
            return DualQuaternion(qr, qd)
        return NotImplemented

    def conjugate(self) -> "DualQuaternion":
        """a* = (ar*) + (ad*)  -- used for frame/adjoint transforms."""
        return DualQuaternion(self.qr.conjugate(), self.qd.conjugate())

    def swap(self) -> "DualQuaternion":
        """a^s = ad + ar (Table II)."""
        return DualQuaternion(self.qd, self.qr)

    def dot(self, other: "DualQuaternion") -> "DualQuaternion":
        return DualQuaternion(self.qr.dot(other.qr), self.qd.dot(other.qr) + self.qr.dot(other.qd))

    def cross(self, other: "DualQuaternion") -> "DualQuaternion":
        return DualQuaternion(self.qr.cross(other.qr), self.qd.cross(other.qr) + self.qr.cross(other.qd))

    def circle(self, other: "DualQuaternion") -> float:
        """a o b = (ar.br + ad.bd) + 0  -> returns the real scalar (dual part is 0).
        Used for the vector dual-quaternion identities (3)-(5), not for testing
        unit-pose-ness (use `.dot()` / `.norm()` for that, per the paper's text
        'q.q = q*q = 1')."""
        real = self.qr.dot(other.qr) + self.qd.dot(other.qd)
        return float(real.w)

    def norm(self) -> float:
        """||a|| = sqrt(a . a), consistent with the paper's statement that a unit
        pose dual quaternion satisfies q.q = 1 (dot product, Table II). Returns
        the real part only (dual part is 0 for a valid/unit pose)."""
        d = self.dot(self)
        return float(np.sqrt(max(d.qr.w, 0.0)))

    def normalize(self) -> "DualQuaternion":
        """Project onto the unit dual-quaternion manifold (valid pose)."""
        qr = self.qr.normalize()
        # enforce qr . qd = 0 (orthogonality constraint) by projecting qd
        d = qr.dot(self.qd).w
        qd = self.qd - d * qr
        return DualQuaternion(qr, qd)

    def vec(self) -> "DualQuaternion":
        """vec{a} = (0, ar) + eps(0, ad) -- strips scalar parts."""
        return DualQuaternion(Quaternion(0.0, self.qr.v.copy()), Quaternion(0.0, self.qd.v.copy()))

    # ---- adjoint / kinematics -------------------------------------------------
    def adjoint(self, s: "DualQuaternion") -> "DualQuaternion":
        """Transform a dual vector s (expressed in frame X) into frame Y, given
        self = q_Y/X:   s^Y = q*_{Y/X} s^X q_{Y/X}."""
        return self.conjugate() * s * self


# --------------------------------------------------------------------------- #
# Self-test (run with `python dual_quaternion.py`)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    def random_unit_quat():
        v = rng.normal(size=4)
        v /= np.linalg.norm(v)
        return v

    # 1) round-trip pos/quat -> DQ -> pos/quat
    for _ in range(200):
        pos = rng.normal(size=3) * 3.0
        quat = random_unit_quat()
        dq = DualQuaternion.from_pos_quat(pos, quat)
        pos2, quat2 = dq.to_pos_quat()
        assert np.allclose(pos, pos2, atol=1e-9), (pos, pos2)
        # quaternion sign ambiguity: q and -q are the same rotation
        assert np.allclose(quat, quat2, atol=1e-9) or np.allclose(quat, -quat2, atol=1e-9)

    # 2) unit dual quaternion constraints: qr.qr = 1, qr.qd = 0
    for _ in range(200):
        pos = rng.normal(size=3)
        quat = random_unit_quat()
        dq = DualQuaternion.from_pos_quat(pos, quat)
        assert abs(dq.qr.dot(dq.qr).w - 1.0) < 1e-9
        assert abs(dq.qr.dot(dq.qd).w) < 1e-9
        assert abs(dq.norm() - 1.0) < 1e-9

    # 3) composition of poses matches composing homogeneous transforms
    def quat_to_R(qwxyz):
        w, x, y, z = qwxyz
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    for _ in range(200):
        pos_a = rng.normal(size=3)
        quat_a = random_unit_quat()
        pos_b = rng.normal(size=3)
        quat_b = random_unit_quat()

        # ground truth: frame A rel I, frame B rel A -> B rel I
        Ra = quat_to_R(quat_a)
        Rb = quat_to_R(quat_b)
        R_BI = Ra @ Rb
        p_BI = pos_a + Ra @ pos_b

        dq_a = DualQuaternion.from_pos_quat(pos_a, quat_a)  # q_A/I
        dq_b = DualQuaternion.from_pos_quat(pos_b, quat_b)  # q_B/A
        dq_c = dq_a * dq_b  # should be q_B/I

        p_test, q_test = dq_c.to_pos_quat()
        assert np.allclose(p_test, p_BI, atol=1e-8), (p_test, p_BI)
        R_test = quat_to_R(q_test if np.dot(q_test, [1, 0, 0, 0]) >= -2 else -q_test)
        assert np.allclose(R_test, R_BI, atol=1e-8)

    print("All dual_quaternion.py self-tests passed.")
