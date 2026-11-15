"""Depth residual at a keypoint (document 02, section 4a).

For observation (i, j) with a depth reading d and the per-image correction
(a_i, b_i) of image i:

    p   = R_i X_j + t_i                       point in camera i
    z   = p_z            (depth along the optical axis), or |p| (range along the ray)
    r   = (z - b) / a - d                     linear form, in the reading's units
    r   = a / (z - b) - 1 / d                 inverse form

The solved depth is mapped into the reading's units rather than the reading into
the scene's (D87): scaling the scene by s with (a, b) -> (s a, s b) leaves r
unchanged, so a free scale cannot shrink the scene to shrink the residuals. With
a = 1, b = 0 both say z - d and 1 / z - 1 / d.

Unwhitened; the backend divides by sigma. Tangents: the pose as in
reprojection.py; the point and (a, b) additively. Jacobians are row-major 1 x 6,
1 x 3 and 1 x 2.
"""

from wrenfold import code_generation as cg, sym, type_annotations as ta

from reprojection import quat_to_rotation


def _skew(d):
    return sym.matrix([[0, -d[2], d[1]], [d[2], 0, -d[0]], [-d[1], d[0], 0]])


def _at_zero(expr, symbols):
    return expr.subs([(s, 0) for s in symbols])


def _make(range_depth: bool, inverse: bool):
    def depth(q: ta.Vector4, t: ta.Vector3, point: ta.Vector3, depth_reading: ta.FloatScalar,
              correction: ta.Vector2):
        d = sym.symbols("d0, d1, d2")
        e = sym.symbols("e0, e1, e2")
        x = sym.symbols("x0, x1, x2")
        c = sym.symbols("c0, c1")
        all_d = list(d) + list(e) + list(x) + list(c)
        R = (sym.eye(3) + _skew(d)) * quat_to_rotation(q)
        p = R * (point + sym.vector(*x)) + t + sym.vector(*e)
        z = sym.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) if range_depth else p[2]
        a = correction[0] + c[0]
        b = correction[1] + c[1]
        value = a / (z - b) - 1 / depth_reading if inverse else (z - b) / a - depth_reading
        r = sym.matrix([[value]])
        residual = _at_zero(r, all_d)
        J_pose = _at_zero(r.jacobian(list(d) + list(e)), all_d)
        J_point = _at_zero(r.jacobian(list(x)), all_d)
        J_corr = _at_zero(r.jacobian(list(c)), all_d)
        return [
            cg.OutputArg(residual, "residual"),
            cg.OutputArg(J_pose, "J_pose", is_optional=True),
            cg.OutputArg(J_point, "J_point", is_optional=True),
            cg.OutputArg(J_corr, "J_correction", is_optional=True),
        ]

    name = "depth_" + ("range" if range_depth else "z") + ("_inverse" if inverse else "")
    depth.__name__ = name
    return depth


FUNCTIONS = [_make(False, False), _make(True, False), _make(False, True), _make(True, True)]
