"""Pose prior residuals: symbolic definitions (document 02, section 2).

Relative prior between images i and j, with the prior T_p = (R_p, t_p) for the
motion T_j * T_i^-1 (camera i coordinates to camera j coordinates):

    T_ji = T_j * T_i^-1 = (R_j R_i^T, t_j - R_j R_i^T t_i)
    r    = [ Log(R_p^T R_ji) ;  R_p^T (t_ji - t_p) ]                  (6 values)

This is document 02's Log(T_p^-1 * T_ji) with rotation and translation taken
separately, as for the pose tangent (D46): radians, then metres in camera i's axes.

Global prior on image i, from a prior set whose frame relates to the working frame
by a similarity S = (R_s, t_s, s), X_working = s R_s X_set + t_s:

    C   = -R^T t                                  camera centre, working frame
    r   = [ Log(R R_s R_p^T) ;  R_s^T (C - t_s) / s - C_p ]             (6 values)

R_p and C_p are the prior's rotation and camera centre in the set's frame. The
position residual is in the set's own units, the units its sigma is given in. For
a set that is the working frame, S is the identity.

Tangents (D46): poses as in reprojection.py; the set's rigid part the same way
(R_s <- Exp(dtheta) R_s, t_s <- t_s + dt); its scale s <- s * exp(ds).
Jacobians are row-major: 6 x 6 per pose, 6 x 6 for the rigid part, 6 x 1 for scale.
"""

from wrenfold import code_generation as cg, sym, type_annotations as ta
from wrenfold.geometry import Quaternion

from reprojection import quat_to_rotation

# Below this norm of the quaternion's vector part, Log uses its small-angle form.
LOG_EPSILON = 1e-9


def _skew(d):
    return sym.matrix([[0, -d[2], d[1]], [d[2], 0, -d[0]], [-d[1], d[0], 0]])


def _quat(v) -> Quaternion:
    return Quaternion.from_wxyz(v)


def _perturbed(q: Quaternion, d) -> Quaternion:
    # Exp(d) * q to first order: (1, d / 2) * q. Only the derivative at d = 0 is
    # used, so the missing normalization (second order) does not matter.
    return Quaternion(1, d[0] / 2, d[1] / 2, d[2] / 2) * q


def _log(q: Quaternion):
    return q.to_rotation_vector(epsilon=LOG_EPSILON, use_atan2=True)


def _at_zero(expr, symbols):
    return expr.subs([(s, 0) for s in symbols])


def relative_pose_prior(qi: ta.Vector4, ti: ta.Vector3, qj: ta.Vector4, tj: ta.Vector3, q_prior: ta.Vector4,
                        t_prior: ta.Vector3):
    di = sym.symbols("di0, di1, di2")
    dj = sym.symbols("dj0, dj1, dj2")
    ei = sym.symbols("ei0, ei1, ei2")
    ej = sym.symbols("ej0, ej1, ej2")
    all_d = list(di) + list(dj) + list(ei) + list(ej)

    Ri = (sym.eye(3) + _skew(di)) * quat_to_rotation(qi)
    Rj = (sym.eye(3) + _skew(dj)) * quat_to_rotation(qj)
    ti_p = ti + sym.vector(*ei)
    tj_p = tj + sym.vector(*ej)
    Rp = quat_to_rotation(q_prior)

    qe = _quat(q_prior).conjugate() * _perturbed(_quat(qj), dj) * _perturbed(_quat(qi), di).conjugate()
    r_rot = _log(qe)
    t_ji = tj_p - Rj * Ri.T * ti_p
    r_trans = Rp.T * (t_ji - t_prior)
    r = sym.vstack([r_rot, r_trans])

    residual = _at_zero(r, all_d)
    J_i = _at_zero(r.jacobian(list(di) + list(ei)), all_d)
    J_j = _at_zero(r.jacobian(list(dj) + list(ej)), all_d)
    return [
        cg.OutputArg(residual, "residual"),
        cg.OutputArg(J_i, "J_i", is_optional=True),
        cg.OutputArg(J_j, "J_j", is_optional=True),
    ]


def global_pose_prior(q: ta.Vector4, t: ta.Vector3, q_set: ta.Vector4, t_set: ta.Vector3, scale: ta.FloatScalar,
                      q_prior: ta.Vector4, c_prior: ta.Vector3):
    d = sym.symbols("d0, d1, d2")
    e = sym.symbols("e0, e1, e2")
    ds_rot = sym.symbols("f0, f1, f2")
    ds_t = sym.symbols("g0, g1, g2")
    ds_s = sym.symbols("h0")
    all_d = list(d) + list(e) + list(ds_rot) + list(ds_t) + [ds_s]

    R = (sym.eye(3) + _skew(d)) * quat_to_rotation(q)
    t_p = t + sym.vector(*e)
    Rs = (sym.eye(3) + _skew(ds_rot)) * quat_to_rotation(q_set)
    ts = t_set + sym.vector(*ds_t)
    s = scale * (1 + ds_s)  # s * exp(ds) to first order

    qe = _perturbed(_quat(q), d) * _perturbed(_quat(q_set), ds_rot) * _quat(q_prior).conjugate()
    r_rot = _log(qe)
    C = -R.T * t_p
    r_pos = Rs.T * (C - ts) / s - c_prior
    r = sym.vstack([r_rot, r_pos])

    residual = _at_zero(r, all_d)
    J_pose = _at_zero(r.jacobian(list(d) + list(e)), all_d)
    J_rigid = _at_zero(r.jacobian(list(ds_rot) + list(ds_t)), all_d)
    J_scale = _at_zero(r.jacobian([ds_s]), all_d)
    return [
        cg.OutputArg(residual, "residual"),
        cg.OutputArg(J_pose, "J_pose", is_optional=True),
        cg.OutputArg(J_rigid, "J_rigid", is_optional=True),
        cg.OutputArg(J_scale, "J_scale", is_optional=True),
    ]


FUNCTIONS = [relative_pose_prior, global_pose_prior]
