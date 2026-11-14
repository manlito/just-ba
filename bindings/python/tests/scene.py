"""Small synthetic scenes in NumPy for the Python tests: a room seen by a camera
walking a loop, keypoints with noise in shuffled order, matches between images,
and a drifting tracker."""

import numpy as np

import just_ba as jba

WIDTH, HEIGHT = 640, 480
K = np.array([280.0, 280.0, 320.0, 240.0])  # PINHOLE, about 97 degrees across: strong geometry


def look_pose(C, yaw):
    """World-to-camera pose of a camera at C looking horizontally at `yaw`
    (world y is down, as the camera's)."""
    z = np.array([np.cos(yaw), 0.0, np.sin(yaw)])
    y = np.array([0.0, 1.0, 0.0])
    x = np.cross(y, z)
    R = np.stack([x, y, z])  # rows: camera axes in world coordinates
    q = jba.quat_from_matrix(R)
    return np.concatenate([q, -R @ C])


def room(n_images=40, n_points=6000, noise=0.5, seed=0, drift=0.002):
    rng = np.random.default_rng(seed)
    # Points on the walls, floor and ceiling of a 6 x 2.7 x 5 m room.
    lo, hi = np.array([-3.0, -1.2, -2.5]), np.array([3.0, 1.5, 2.5])
    pts = rng.uniform(lo, hi, size=(n_points, 3))
    face = rng.integers(0, 6, size=n_points)
    axis, side = face // 2, face % 2
    pts[np.arange(n_points), axis] = np.where(side == 0, lo[axis], hi[axis])

    poses = []
    for s in range(n_images):
        a = 2 * np.pi * s / n_images
        C = np.array([1.2 * np.cos(a), 0.05 * np.sin(5 * a), 1.0 * np.sin(a)])
        poses.append(look_pose(C, a + 0.5 * np.sin(3 * a)))
    poses = np.array(poses)

    # Keypoints: visible points with noise, shuffled per image.
    keypoints, point_of = [], []
    R = jba.matrix_from_quat(poses[:, :4])
    for i in range(n_images):
        pc = pts @ R[i].T + poses[i, 4:]
        ok = pc[:, 2] > 0.3
        uv = np.full((n_points, 2), -1.0)
        uv[ok] = K[:2] * pc[ok, :2] / pc[ok, 2:] + K[2:]
        ok &= (uv[:, 0] >= 0) & (uv[:, 0] < WIDTH) & (uv[:, 1] >= 0) & (uv[:, 1] < HEIGHT)
        idx = rng.permutation(np.flatnonzero(ok))
        keypoints.append(uv[idx] + rng.normal(0, noise, size=(idx.size, 2)))
        point_of.append(idx)

    # Matches between images up to 4 apart in capture order (and around the loop).
    matches = []
    where = [dict(zip(p.tolist(), range(p.size))) for p in point_of]
    for i in range(n_images):
        for d in range(1, 5):
            j = (i + d) % n_images
            common = set(where[i]) & set(where[j])
            for pt in common:
                matches.append((min(i, j), where[min(i, j)][pt], max(i, j), where[max(i, j)][pt]))
    matches = np.array(sorted(matches), dtype=np.int32)

    # A tracker that drifts in heading and position along the loop.
    tracked = poses.copy()
    yaw, offset = 0.0, np.zeros(3)
    for s in range(1, n_images):
        yaw += drift * rng.normal()
        offset += drift * rng.normal(size=3)
        c, sn = np.cos(yaw), np.sin(yaw)
        D = np.array([[c, 0, -sn], [0, 1, 0], [sn, 0, c]])  # rotation about the vertical
        Rw = jba.matrix_from_quat(poses[s, :4])
        C = jba.camera_centres(poses[s])[0]
        Rt = Rw @ D.T
        tracked[s, :4] = jba.quat_from_matrix(Rt)
        tracked[s, 4:] = -Rt @ (D @ C + offset)
    return dict(poses=poses, points=pts, keypoints=keypoints, point_of=point_of, matches=matches, tracked=tracked)


def aligned_errors(estimate, truth):
    """Median camera centre (m) and rotation (rad) errors after the rigid
    alignment of the estimate onto the truth (rotation from the camera rotations)."""
    Re, Rt = jba.matrix_from_quat(estimate[:, :4]), jba.matrix_from_quat(truth[:, :4])
    # R_t ~ R_e G^T  =>  G ~ R_t^T R_e; average by SVD of the sum.
    U, _, Vt = np.linalg.svd(np.einsum("nji,njk->ik", Rt, Re))
    G = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    Ce, Ct = jba.camera_centres(estimate), jba.camera_centres(truth)
    t = Ct.mean(0) - G @ Ce.mean(0)
    dc = np.linalg.norm(Ce @ G.T + t - Ct, axis=1)
    Ra = Re @ G.T
    cos = np.clip((np.einsum("nij,nij->n", Ra, Rt) - 1) / 2, -1, 1)
    return float(np.median(dc)), float(np.median(np.arccos(cos)))
