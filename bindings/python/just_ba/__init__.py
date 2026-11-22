"""just-ba: minimal, fast bundle adjustment.

The compiled core is in ``just_ba._core``; everything in it is re-exported here.
This module adds what is easier in Python: building a Problem from keyword
arguments, and converting poses between camera conventions.

Conventions of the library: poses map world to camera and are stored as
``qw qx qy qz tx ty tz``; camera axes are x right, y down, z forward (OpenCV,
COLMAP). Quaternions are Hamilton, (w, x, y, z).
"""

from __future__ import annotations

import numpy as np

from ._build_info import CUDA_VERSION, VARIANT, __version__

if VARIANT.startswith("cu"):
    from . import _nvidia

    _nvidia.preload(VARIANT)
    try:
        from . import _core
    except ImportError as e:
        raise ImportError(f"{e}; {_nvidia.import_error_hint(VARIANT)}") from e

from ._core import *  # noqa: F401,F403
from ._core import Problem, num_params

__all__ = [name for name in dir() if not name.startswith("_")]

# Camera axes of each convention, as columns in OpenCV camera axes:
# OpenGL (ARKit, ARCore, Unity-style): x right, y up, z backward.
_FLIP = {
    "opencv": np.eye(3),
    "colmap": np.eye(3),
    "opengl": np.diag([1.0, -1.0, -1.0]),
    "arkit": np.diag([1.0, -1.0, -1.0]),
    "arcore": np.diag([1.0, -1.0, -1.0]),
}


def _flip(convention: str) -> np.ndarray:
    try:
        return _FLIP[convention.lower()]
    except KeyError:
        raise ValueError(f"unknown camera convention {convention!r}; one of {sorted(_FLIP)}") from None


def quat_from_matrix(R) -> np.ndarray:
    """Unit quaternions (w, x, y, z), w >= 0, of rotation matrices (..., 3, 3).

    >>> jba.quat_from_matrix(np.eye(3))
    array([1., 0., 0., 0.])
    """
    R = np.asarray(R, dtype=np.float64)
    flat = R.reshape(-1, 3, 3)
    q = np.empty((flat.shape[0], 4))
    for k, M in enumerate(flat):
        tr = M[0, 0] + M[1, 1] + M[2, 2]
        if tr > 0:
            s = 2 * np.sqrt(tr + 1)
            q[k] = [s / 4, (M[2, 1] - M[1, 2]) / s, (M[0, 2] - M[2, 0]) / s, (M[1, 0] - M[0, 1]) / s]
        elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
            s = 2 * np.sqrt(1 + M[0, 0] - M[1, 1] - M[2, 2])
            q[k] = [(M[2, 1] - M[1, 2]) / s, s / 4, (M[0, 1] + M[1, 0]) / s, (M[0, 2] + M[2, 0]) / s]
        elif M[1, 1] > M[2, 2]:
            s = 2 * np.sqrt(1 + M[1, 1] - M[0, 0] - M[2, 2])
            q[k] = [(M[0, 2] - M[2, 0]) / s, (M[0, 1] + M[1, 0]) / s, s / 4, (M[1, 2] + M[2, 1]) / s]
        else:
            s = 2 * np.sqrt(1 + M[2, 2] - M[0, 0] - M[1, 1])
            q[k] = [(M[1, 0] - M[0, 1]) / s, (M[0, 2] + M[2, 0]) / s, (M[1, 2] + M[2, 1]) / s, s / 4]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1
    return q.reshape(R.shape[:-2] + (4,))


def matrix_from_quat(q) -> np.ndarray:
    """Rotation matrices (..., 3, 3) of unit quaternions (..., 4), (w, x, y, z).

    >>> jba.matrix_from_quat([0.0, 1.0, 0.0, 0.0])  # half a turn about x
    array([[ 1.,  0.,  0.],
           [ 0., -1.,  0.],
           [ 0.,  0., -1.]])
    """
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = np.moveaxis(q, -1, 0)
    R = np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    )
    return R.reshape(q.shape[:-1] + (3, 3))


def poses_from_camera_to_world(T, convention: str = "opencv") -> np.ndarray:
    """Library poses (N, 7) from camera-to-world matrices (N, 4, 4) or (4, 4).

    ``convention`` names the camera axes of T: "opencv" (x right, y down, z
    forward), or "opengl" / "arkit" / "arcore" (x right, y up, z backward, the
    camera looking down -z). The world frame is kept as it is.

    ARKit's ``ARCamera.transform`` and ARCore's ``Camera.getPose()`` follow the
    OpenGL camera axes according to their documentation; check this on a real
    session before relying on it.

    An ARKit camera at (1, 2, 3) with the world's axes: in the library's axes it
    looks down the world's -z, so its rotation is half a turn about x.

    >>> T = np.eye(4); T[:3, 3] = [1.0, 2.0, 3.0]
    >>> jba.poses_from_camera_to_world(T, "arkit")
    array([ 0.,  1.,  0.,  0., -1.,  2.,  3.])
    """
    T = np.asarray(T, dtype=np.float64)
    single = T.ndim == 2
    T = T.reshape(-1, 4, 4)
    R_cw = T[:, :3, :3] @ _flip(convention)  # camera-to-world, OpenCV camera axes
    C = T[:, :3, 3]
    R = np.transpose(R_cw, (0, 2, 1))  # world to camera
    t = -np.einsum("nij,nj->ni", R, C)
    out = np.concatenate([quat_from_matrix(R), t], axis=1)
    return out[0] if single else out


def camera_to_world_from_poses(poses, convention: str = "opencv") -> np.ndarray:
    """Camera-to-world matrices (N, 4, 4) from library poses (N, 7); inverse of
    poses_from_camera_to_world.

    >>> jba.camera_to_world_from_poses([0.0, 1.0, 0.0, 0.0, -1.0, 2.0, 3.0], "arkit")
    array([[1., 0., 0., 1.],
           [0., 1., 0., 2.],
           [0., 0., 1., 3.],
           [0., 0., 0., 1.]])
    """
    poses = np.asarray(poses, dtype=np.float64)
    single = poses.ndim == 1
    poses = poses.reshape(-1, 7)
    R = matrix_from_quat(poses[:, :4])
    R_cw = np.transpose(R, (0, 2, 1))
    C = -np.einsum("nij,nj->ni", R_cw, poses[:, 4:])
    T = np.zeros((poses.shape[0], 4, 4))
    T[:, :3, :3] = R_cw @ _flip(convention)
    T[:, :3, 3] = C
    T[:, 3, 3] = 1
    return T[0] if single else T


def camera_centres(poses) -> np.ndarray:
    """Camera centres (N, 3) of library poses (N, 7): C = -R^T t.

    >>> jba.camera_centres([0.0, 1.0, 0.0, 0.0, -1.0, 2.0, 3.0])
    array([[1., 2., 3.]])
    """
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 7)
    R = matrix_from_quat(poses[:, :4])
    return -np.einsum("nji,nj->ni", R, poses[:, 4:])


def make_problem(model, intrinsics, **fields) -> Problem:
    """A Problem from keyword arguments: ``model`` (name or CameraModel),
    ``intrinsics``, then any Problem attribute (poses, points, obs_image, ...,
    rel_images, rel_pose, glob_image, glob_pose, set_relation, set_transform).

    >>> s = jba.make_problem("PINHOLE", [500.0, 500.0, 320.0, 240.0],
    ...                      poses=np.tile([1.0, 0, 0, 0, 0, 0, 0], (2, 1)))
    >>> s.num_images, s.num_points
    (2, 0)
    """
    p = Problem()
    p.model = model
    p.intrinsics = np.asarray(intrinsics, dtype=np.float64)
    for name, value in fields.items():
        if not hasattr(p, name):
            raise TypeError(f"Problem has no attribute {name!r}")
        if name == "set_relation":
            p.set_relation = list(value)
        else:
            setattr(p, name, np.asarray(value))
    if "poses" not in fields:
        raise TypeError("make_problem needs poses (N, 7): one per image, identity is fine for refine")
    err = p.validate()
    if err:
        raise ValueError(f"invalid problem: {err}")
    return p
