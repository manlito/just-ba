"""Direction residuals: gravity prior and Manhattan line (document 02, sections 2
and 5). Both constrain a camera's rotation only.

Gravity prior on image i, with g_w the direction of gravity in the working frame
and g_c the direction measured in camera i's axes (both unit):

    r = R_i g_w - g_c                                          (3 values)

The chordal difference: |r| = 2 sin(angle / 2), so it has a single minimum (a cross
product would also vanish for gravity measured upside down).

Manhattan line in image i, assigned to axis e (a unit vector in the building's
axes), with R_M the rotation from the building's axes to the working frame, and
d = R_i R_M e the axis in camera i's axes. Lines along d project to the great
circles through d. The segment's endpoints, at pixels x_0 and x_1, have normalized
coordinates u_j = undistort(K, x_j) and unit bearings b_j. The residual is the
pixel distance of the two endpoints from the best of those circles, the circle
(its one free angle) eliminated in closed form (D163):

    r = s_0 s_1 (d . (b_0 x b_1)) / sqrt(s_0^2 |d x b_0|^2 + s_1^2 |d x b_1|^2)
                                                                (1 value, pixels)

s_j, in pixels per radian, scales an angular offset of b_j across the circle
through b_j and d into a pixel distance across that circle's image (a curve under
distortion): with J_j the 2 x 3 map from the sphere's tangent plane at b_j to
pixels, tau_j the circle's unit tangent at b_j and nu_j = b_j x tau_j,

    s_j = |det [J_j tau_j, J_j nu_j]| / |J_j tau_j|.

The minimum of s_0^2 (m . b_0)^2 + s_1^2 (m . b_1)^2 over unit normals m of
circles through d is the smaller eigenvalue of a 2 x 2 matrix with determinant
s_0^2 s_1^2 (d . (b_0 x b_1))^2 and trace the denominator's square; r^2 is that
determinant over the trace, which is the eigenvalue to first order in the misfit
(the eigenvalue lies between r^2 and 2 r^2). An endpoint at the vanishing point
itself (b_j = d) is 0 / 0 here; the residual tends to zero there, and the C++
wrapper writes it as zero (D163). For a pinhole camera and a short segment r = L sin(alpha) /
sqrt(2), alpha the image angle between the segment and the direction towards the
vanishing point, L its length in pixels. d is never projected, so a vanishing point
at infinity or behind the camera needs no special case.

The intrinsics enter twice: through J_j, and through u_j, which moves with K for
fixed pixels. The second is the implicit function's derivative, du_j/dK =
-A_j^-1 dpi/dK, with A_j = dpi/du at u_j (pi the camera model, normalized to
pixels), so J_intrinsics is exact without differentiating the iterative inverse.

Tangents: the pose as in reprojection.py (rotation on the left, in camera axes);
R_M <- Exp(dphi) R_M with dphi in the working frame's axes. Jacobians are
row-major: J_pose is rows x 6 (translation columns zero), J_frame 1 x 3,
J_intrinsics 1 x (number of model parameters). A frame with one degree of freedom
(a heading about the vertical u) uses J_frame u.
"""

from wrenfold import code_generation as cg, sym, type_annotations as ta

from reprojection import CAMERA_MODELS, quat_to_rotation


def _skew(d):
    return sym.matrix([[0, -d[2], d[1]], [d[2], 0, -d[0]], [-d[1], d[0], 0]])


def _at_zero(expr, symbols):
    return expr.subs([(s, 0) for s in symbols])


def gravity_prior(q: ta.Vector4, g_world: ta.Vector3, g_camera: ta.Vector3):
    d = sym.symbols("d0, d1, d2")
    e = sym.symbols("e0, e1, e2")
    all_d = list(d) + list(e)
    R = (sym.eye(3) + _skew(d)) * quat_to_rotation(q)
    r = R * g_world - g_camera
    residual = _at_zero(r, all_d)
    J_pose = _at_zero(r.jacobian(all_d), all_d)
    return [cg.OutputArg(residual, "residual"), cg.OutputArg(J_pose, "J_pose", is_optional=True)]


def _inverse2(A):
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    return sym.matrix([[A[1, 1] / det, -A[0, 1] / det], [-A[1, 0] / det, A[0, 0] / det]])


def _cross(a, b):
    return sym.vector(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _endpoint(model, d, uv, K):
    """Bearing, s (pixels per radian across the circle through it and d) and
    |d x b|^2 at one endpoint; also the camera model's pixel at uv."""
    u, v = uv[0], uv[1]
    norm = sym.sqrt(u * u + v * v + 1)
    b = sym.vector(u / norm, v / norm, 1 / norm)
    x, y = model.project(u, v, K)
    A = sym.vector(x, y).jacobian([u, v])
    # Bearing displacement db (tangent at b) to normalized coordinates: u = b_x / b_z.
    P = norm * sym.matrix([[1, 0, -u], [0, 1, -v]])
    J = A * P
    db = _dot(d, b)
    tau = sym.vector(d[0] - db * b[0], d[1] - db * b[1], d[2] - db * b[2])
    t2 = _dot(tau, tau)  # |d x b|^2, d unit
    tn = sym.sqrt(t2)
    tau = sym.vector(tau[0] / tn, tau[1] / tn, tau[2] / tn)
    nu = _cross(b, tau)
    Jt = J * tau
    Jn = J * nu
    s = sym.abs(Jt[0] * Jn[1] - Jt[1] * Jn[0]) / sym.sqrt(Jt[0] * Jt[0] + Jt[1] * Jt[1])
    return b, s, t2, sym.vector(x, y), A


def make_manhattan_line(model):
    """Manhattan line residual for one camera model (module docstring)."""
    n = len(model.params)
    Kvec = getattr(ta, f"Vector{n}")

    def manhattan_line(q: ta.Vector4, q_frame: ta.Vector4, axis: ta.Vector3, uv0: ta.Vector2, uv1: ta.Vector2,
                       intrinsics: Kvec):
        dr = sym.symbols("d0, d1, d2")
        f = sym.symbols("f0, f1, f2")
        all_d = list(dr) + list(f)
        R = (sym.eye(3) + _skew(dr)) * quat_to_rotation(q)
        RM = (sym.eye(3) + _skew(f)) * quat_to_rotation(q_frame)
        d = R * (RM * axis)
        b0, s0, t0, x0, A0 = _endpoint(model, d, uv0, intrinsics)
        b1, s1, t1, x1, A1 = _endpoint(model, d, uv1, intrinsics)
        r = sym.matrix([[s0 * s1 * _dot(d, _cross(b0, b1)) / sym.sqrt(s0 * s0 * t0 + s1 * s1 * t1)]])
        residual = _at_zero(r, all_d)
        J_rot = _at_zero(r.jacobian(list(dr)), all_d)
        J_pose = sym.hstack([J_rot, sym.zeros(1, 3)])
        J_frame = _at_zero(r.jacobian(list(f)), all_d)
        # Total derivative in K, the endpoints' pixels held: du_j/dK = -A_j^-1 dpi/dK.
        J_K = residual.jacobian(intrinsics)
        for uv, x, A in ((uv0, _at_zero(x0, all_d), A0), (uv1, _at_zero(x1, all_d), A1)):
            J_K = J_K - residual.jacobian(uv) * (_inverse2(A) * x.jacobian(intrinsics))
        return [
            cg.OutputArg(residual, "residual"),
            cg.OutputArg(J_pose, "J_pose", is_optional=True),
            cg.OutputArg(J_frame, "J_frame", is_optional=True),
            cg.OutputArg(J_K, "J_intrinsics", is_optional=True),
        ]

    manhattan_line.__name__ = f"manhattan_line_{model.name}"
    return manhattan_line


FUNCTIONS = [gravity_prior] + [make_manhattan_line(m) for m in CAMERA_MODELS]
