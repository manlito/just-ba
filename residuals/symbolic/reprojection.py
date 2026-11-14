"""Reprojection residual: symbolic definition.

    r = project(K, R(q) X + t) - x_obs                       (2 values, pixels)

Conventions (document 02, and decision log D45 to D47):

- Pose maps world to camera. Stored as a unit quaternion q = (w, x, y, z) and a
  translation t. Camera axes: x right, y down, z forward.
- Tangent of the pose is 6 values, rotation first: (dtheta, dt). The update is
  R <- Exp(dtheta) * R and t <- t + dt (rotation and translation updated
  separately, rotation perturbed on the left, dtheta a rotation vector in radians).
- Camera models and parameter order follow COLMAP.
- Jacobians are row-major: J_pose is 2 x 6, J_point is 2 x 3, J_intrinsics is
  2 x (number of model parameters).
"""

from dataclasses import dataclass
from typing import Callable, List

from wrenfold import code_generation as cg, sym, type_annotations as ta


def quat_to_rotation(q) -> sym.MatrixExpr:
    """Rotation matrix of a unit quaternion (w, x, y, z)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return sym.matrix(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )




# Camera models: name, COLMAP id, parameter names, distortion function.
# Each takes normalized coordinates (u, v) and the parameter vector, and returns
# pixel coordinates.


def _simple_pinhole(u, v, k):
    f, cx, cy = k[0], k[1], k[2]
    return f * u + cx, f * v + cy


def _pinhole(u, v, k):
    fx, fy, cx, cy = k[0], k[1], k[2], k[3]
    return fx * u + cx, fy * v + cy


def _simple_radial(u, v, k):
    f, cx, cy, k1 = k[0], k[1], k[2], k[3]
    radial = k1 * (u * u + v * v)
    return f * (u + u * radial) + cx, f * (v + v * radial) + cy


def _radial(u, v, k):
    f, cx, cy, k1, k2 = k[0], k[1], k[2], k[3], k[4]
    r2 = u * u + v * v
    radial = k1 * r2 + k2 * r2 * r2
    return f * (u + u * radial) + cx, f * (v + v * radial) + cy


def _opencv(u, v, k):
    fx, fy, cx, cy, k1, k2, p1, p2 = (k[i] for i in range(8))
    uu, vv, uv = u * u, v * v, u * v
    r2 = uu + vv
    radial = k1 * r2 + k2 * r2 * r2
    du = u * radial + 2 * p1 * uv + p2 * (r2 + 2 * uu)
    dv = v * radial + 2 * p2 * uv + p1 * (r2 + 2 * vv)
    return fx * (u + du) + cx, fy * (v + dv) + cy


# Below this radius in normalized coordinates the fisheye model is the identity,
# as in COLMAP: theta_d / r -> 1 there, and the formula divides by r.
FISHEYE_EPSILON = 1e-12


def _opencv_fisheye(u, v, k):
    fx, fy, cx, cy, k1, k2, k3, k4 = (k[i] for i in range(8))
    r = sym.sqrt(u * u + v * v)
    theta = sym.atan(r)
    t2 = theta * theta
    theta_d = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    factor = sym.where(r > FISHEYE_EPSILON, theta_d / r, 1)
    return fx * (u * factor) + cx, fy * (v * factor) + cy


@dataclass(frozen=True)
class CameraModel:
    name: str
    colmap_id: int
    params: List[str]
    project: Callable


CAMERA_MODELS = [
    CameraModel("simple_pinhole", 0, ["f", "cx", "cy"], _simple_pinhole),
    CameraModel("pinhole", 1, ["fx", "fy", "cx", "cy"], _pinhole),
    CameraModel("simple_radial", 2, ["f", "cx", "cy", "k"], _simple_radial),
    CameraModel("radial", 3, ["f", "cx", "cy", "k1", "k2"], _radial),
    CameraModel("opencv", 4, ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"], _opencv),
    CameraModel("opencv_fisheye", 5, ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"], _opencv_fisheye),
]


def _vector_annotation(n: int):
    return getattr(ta, f"Vector{n}")


def _matrix_annotation(rows: int, cols: int):
    return getattr(ta, f"Matrix{rows}x{cols}")


def make_reprojection(model: CameraModel):
    """Builds the wrenfold function for one camera model."""
    n = len(model.params)
    Kvec = _vector_annotation(n)

    def reprojection(q: ta.Vector4, t: ta.Vector3, point: ta.Vector3, intrinsics: Kvec,
                     observed: ta.Vector2):
        # Left perturbation of the rotation: Exp(dtheta) * R * X, to first order
        # (I + [dtheta]_x) * R * X. First order is enough because only the
        # derivative at dtheta = 0 is used. Applying it to the rotated point, rather
        # than to the quaternion, keeps the unit-norm assumption exact.
        d = sym.symbols("d0, d1, d2")
        skew = sym.matrix([[0, -d[2], d[1]], [d[2], 0, -d[0]], [-d[1], d[0], 0]])
        rotated = quat_to_rotation(q) * point
        p_cam = rotated + skew * rotated + t

        u = p_cam[0] / p_cam[2]
        v = p_cam[1] / p_cam[2]
        x, y = model.project(u, v, intrinsics)
        residual_perturbed = sym.vector(x - observed[0], y - observed[1])

        at_zero = [(d[0], 0), (d[1], 0), (d[2], 0)]
        residual = residual_perturbed.subs(at_zero)
        j_rot = residual_perturbed.jacobian(list(d)).subs(at_zero)
        j_trans = residual.jacobian(t)
        j_pose = sym.hstack([j_rot, j_trans])
        j_point = residual.jacobian(point)
        j_intr = residual.jacobian(intrinsics)

        return [
            cg.OutputArg(residual, "residual"),
            cg.OutputArg(j_pose, "J_pose", is_optional=True),
            cg.OutputArg(j_point, "J_point", is_optional=True),
            cg.OutputArg(j_intr, "J_intrinsics", is_optional=True),
        ]

    reprojection.__name__ = f"reprojection_{model.name}"
    return reprojection
