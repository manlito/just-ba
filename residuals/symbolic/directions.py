"""Direction residuals: gravity prior and Manhattan line (document 02, sections 2
and 5). Both constrain a camera's rotation only.

Gravity prior on image i, with g_w the direction of gravity in the working frame
and g_c the direction measured in camera i's axes (both unit):

    r = R_i g_w - g_c                                          (3 values)

The chordal difference: |r| = 2 sin(angle / 2), so it has a single minimum (a cross
product would also vanish for gravity measured upside down).

Manhattan line in image i, assigned to axis e (a unit vector in the building's
axes), with n the unit normal of the plane through the camera centre and the
segment, in camera i's axes, and R_M the rotation from the building's axes to the
working frame:

    r = n . (R_i R_M e)                                         (1 value)

the sine of the angle by which the axis misses the plane.

Tangents: the pose as in reprojection.py (rotation on the left, in camera axes);
R_M <- Exp(dphi) R_M with dphi in the working frame's axes. Jacobians are
row-major: J_pose is rows x 6 (translation columns zero), J_frame 1 x 3. A frame
with one degree of freedom (a heading about the vertical u) uses J_frame u.
"""

from wrenfold import code_generation as cg, sym, type_annotations as ta

from reprojection import quat_to_rotation


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


def manhattan_line(q: ta.Vector4, normal: ta.Vector3, q_frame: ta.Vector4, axis: ta.Vector3):
    d = sym.symbols("d0, d1, d2")
    e = sym.symbols("e0, e1, e2")
    f = sym.symbols("f0, f1, f2")
    all_d = list(d) + list(e) + list(f)
    R = (sym.eye(3) + _skew(d)) * quat_to_rotation(q)
    RM = (sym.eye(3) + _skew(f)) * quat_to_rotation(q_frame)
    r = sym.matrix([[(normal.T * (R * (RM * axis)))[0]]])
    residual = _at_zero(r, all_d)
    J_pose = _at_zero(r.jacobian(list(d) + list(e)), all_d)
    J_frame = _at_zero(r.jacobian(list(f)), all_d)
    return [
        cg.OutputArg(residual, "residual"),
        cg.OutputArg(J_pose, "J_pose", is_optional=True),
        cg.OutputArg(J_frame, "J_frame", is_optional=True),
    ]


FUNCTIONS = [gravity_prior, manhattan_line]
