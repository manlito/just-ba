"""The live demo's reconstruction in Python: a small incremental mapper on the Python
packages alone (poselib_gpu and just_ba), for the end-to-end test next to it.

It follows the rules of the demo's mapper (demo/src/mapper.cpp) with the thresholds the
live demo gives it (demo::colmap_thresholds), shortened where the test does not need
the rest: one model, no focal length search, no position priors. Given the same
keypoints and matches it registers the same images as the C++ mapper, with as many
points to within a few percent.

  1. Verify every pair: poselib-gpu's relative pose (LO-RANSAC, 4 px). A pair's inliers
     go on when it has 15 or more of them and a quarter of its matches.
  2. Tracks: the connected components of the verified matches. One with two keypoints
     in an image is kept, and a point takes at most one keypoint of each image (D140);
     just_ba.build_tracks would drop it (ConflictPolicy.DROP) or needs poses to split
     it (REPAIR).
  3. First pair: the most inliers (100 or more) with a median triangulation angle of
     16 degrees, else 8, 4, 2; its points triangulated.
  4. Register: the unregistered images that see the most points, four at a time, by
     poselib-gpu's PnP (12 px); kept with 30 inliers and a quarter of the points. After
     each, new points are triangulated, existing ones continued, everything adjusted by
     just_ba (Cauchy at 4 px, focal length and distortion refined), filtered
     (reprojection over 4 px, behind the camera, under 1.5 degrees of triangulation),
     and tracks whose point leaves keypoints of three images unused split.
  5. A final adjustment, filter and split, twice.

Points are triangulated robustly from pairs of rays and take keypoints within 2 degrees
of their ray (COLMAP's create and continue max_angle_error); a keypoint the filter
dropped comes back only within 4 px. Tracks seen in two images only are not
triangulated (COLMAP's tri_ignore_two_view_tracks, as the live demo).

Pixel coordinates are just-keypoints': x right, y down, (0, 0) the centre of the
top-left pixel. Cameras are SIMPLE_RADIAL (f, cx, cy, k) for both libraries.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

import just_ba as jba
import poselib_gpu as pg

# The backends by the name poselib-gpu, just-keypoints and just-matching give them.
BACKENDS = {"cpu": jba.Backend.CPU, "metal": jba.Backend.METAL, "cuda": jba.Backend.CUDA}


@dataclass
class Options:
    pair_max_error_px: float = 4.0
    min_pair_inliers: int = 15
    min_pair_inlier_ratio: float = 0.25
    min_first_pair_inliers: int = 100
    pnp_max_error_px: float = 12.0
    min_pnp_inliers: int = 30
    min_pnp_inlier_ratio: float = 0.25
    pnp_batch: int = 4
    max_reproj_px: float = 4.0
    min_tri_angle_deg: float = 1.5
    tri_max_angle_error_deg: float = 2.0  # COLMAP's create_max_angle_error and continue_max_angle_error
    tri_hypotheses: int = 20              # pairs of rays tried per track
    ignore_two_view_tracks: bool = True
    local_iterations: int = 25
    final_iterations: int = 50
    refine_intrinsics: int = 1 | 4  # focal length and distortion; the principal point fixed
    seed: int = 7


@dataclass
class Result:
    registered: np.ndarray            # (n,) bool
    poses: np.ndarray                 # (n, 7) world to camera; meaningful where registered
    intrinsics: np.ndarray            # (4,) f, cx, cy, k, refined
    points: np.ndarray                # (P, 3)
    problem: jba.Problem              # the last adjustment's problem, solved
    reprojection_px: np.ndarray       # (K,) per observation of the final model
    pairs_kept: int = 0
    tracks: int = 0                   # built from the matches
    splits: int = 0                   # tracks split off them
    first_pair: tuple = ()
    adjustments: int = 0
    seconds: dict = field(default_factory=dict)


def _groups(owner):
    """Rows of `owner` (sorted) grouped by run length: yields (rows (G, L), first row's
    owner per group (G,)), one per distinct length L."""
    if len(owner) == 0:
        return
    starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
    counts = np.diff(np.r_[starts, len(owner)])
    for L in np.unique(counts):
        s = starts[counts == L]
        yield s[:, None] + np.arange(L), owner[s]


class Mapper:
    def __init__(self, keypoints, sizes, focal, pairs, backend="cpu", options=None):
        """keypoints: (n_i, 2) per image; sizes: (n, 2) width, height (one camera, so
        all the same); focal: pixels; pairs: (a, b, idx (m, 2)) per matched pair."""
        self.o = options or Options()
        self.kp = [np.asarray(k, dtype=np.float64) for k in keypoints]
        self.n = len(self.kp)
        w, h = sizes[0]
        self.K = np.array([focal, (w - 1) / 2.0, (h - 1) / 2.0, 0.0])
        self.pairs = pairs
        self.backend = backend
        self.ctx = pg.Context(backend, seed=self.o.seed)
        self.ba_backend = BACKENDS[backend]
        self.adjustments = 0
        self.splits = 0
        self.seconds = {"verify": 0.0, "tracks": 0.0, "triangulate": 0.0, "split": 0.0, "pnp": 0.0, "adjust": 0.0}

    # -- cameras

    def _camera(self):
        return pg.Camera("SIMPLE_RADIAL", list(self.K))

    def _sync(self):
        self.R = jba.matrix_from_quat(self.poses[:, :4])
        self.t = self.poses[:, 4:].copy()
        self.C = -np.einsum("nji,nj->ni", self.R, self.t)

    def _rays(self, k):
        """Undistorted normalised image coordinates of observations k."""
        f, cx, cy, d = self.K
        u = (self.o_xy[k] - [cx, cy]) / f
        x = u.copy()
        for _ in range(10):
            x = u / (1.0 + d * (x * x).sum(axis=1, keepdims=True))
        return x

    def _project(self, k, X):
        """Reprojection error (pixels) and depth of observations k of points X (len(k), 3)."""
        img = self.o_image[k]
        Xc = np.einsum("kij,kj->ki", self.R[img], X) + self.t[img]
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            x = Xc[:, :2] / z[:, None]
            f, cx, cy, d = self.K
            uv = f * x * (1.0 + d * (x * x).sum(axis=1, keepdims=True)) + [cx, cy]
            err = np.linalg.norm(uv - self.o_xy[k], axis=1)
        err[~np.isfinite(err)] = np.inf
        return err, z

    def _max_angle_deg(self, rows, X):
        """Largest angle between the rays from points X (G, 3) to the cameras of their
        observations rows (G, L)."""
        v = self.C[self.o_image[rows]] - X[:, None, :]
        v /= np.linalg.norm(v, axis=2, keepdims=True)
        cos = np.einsum("gai,gbi->gab", v, v).min(axis=(1, 2))
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    # -- points

    def _one_per_image(self, k, score=None):
        """Of observations k, one per track and image: the lowest score's (else the first)."""
        if score is not None:
            order = np.argsort(score, kind="stable")
            k = k[order]
        key = self.o_track[k] * self.n + self.o_image[k]
        _, first = np.unique(key, return_index=True)
        return np.sort(k[first])

    def _world_rays(self, k):
        """Unit rays of observations k in the world frame."""
        d = np.einsum("kji,kj->ki", self.R[self.o_image[k]], np.c_[self._rays(k), np.ones(len(k))])
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def _angle_error(self, k, X):
        """Angle (radians) between the rays of observations k and their points X
        (len(k), 3); over 90 degrees for a point behind the camera."""
        v = X - self.C[self.o_image[k]]
        with np.errstate(divide="ignore", invalid="ignore"):
            cos = (self._world_rays(k) * v).sum(axis=1) / np.linalg.norm(v, axis=1)
        return np.arccos(np.clip(np.nan_to_num(cos, nan=-1.0), -1.0, 1.0))

    def _triangulate(self, idx):
        """Points for the tracks of observations idx (sorted by track), robustly, as the
        library's triangulate_robust: the pair of rays (from different images) whose
        midpoint has the most rays within tri_max_angle_error_deg, then a DLT on those,
        at most one per image (the closest). A point needs two such rays and
        min_tri_angle_deg between them. Returns (tracks, X, inlier observations)."""
        max_err = np.radians(self.o.tri_max_angle_error_deg)
        cos_min_tri = np.cos(np.radians(self.o.min_tri_angle_deg))
        rng = np.random.default_rng(self.o.seed)
        out_t, out_X, out_obs = [], [], []
        for rows, t in _groups(self.o_track[idx]):
            k = idx[rows]
            G, L = k.shape
            img = self.o_image[k]
            d = self._world_rays(k.ravel()).reshape(G, L, 3)
            C = self.C[img]

            def errors(X):  # (G, ..., 3) points against every ray of their track: (G, ..., L)
                v = X[..., None, :] - C.reshape((G,) + (1,) * (X.ndim - 2) + (L, 3))
                cos = (v * d.reshape(v.shape[:1] + (1,) * (X.ndim - 2) + (L, 3))).sum(-1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    cos = np.nan_to_num(cos / np.linalg.norm(v, axis=-1), nan=-1.0)
                return np.arccos(np.clip(cos, -1.0, 1.0))

            def one_per_image(inl, err):  # the closest inlier of each image
                same = img[:, :, None] == img[:, None, :]
                order = np.arange(L)
                closer = (err[:, None, :] < err[:, :, None]) | \
                         ((err[:, None, :] == err[:, :, None]) & (order[None, :] < order[:, None]))
                return inl & ~(same & inl[:, None, :] & closer).any(axis=2)

            # Hypotheses: midpoints of pairs of rays.
            pa, pb = np.triu_indices(L, 1)
            if len(pa) > self.o.tri_hypotheses:
                pick = np.sort(rng.choice(len(pa), self.o.tri_hypotheses, replace=False))
                pa, pb = pa[pick], pb[pick]
            da, db, w = d[:, pa], d[:, pb], C[:, pa] - C[:, pb]
            b = (da * db).sum(-1)
            e, f = (da * w).sum(-1), (db * w).sum(-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                s, u = (b * f - e) / (1 - b * b), (f - b * e) / (1 - b * b)
            Xh = 0.5 * (C[:, pa] + s[..., None] * da + C[:, pb] + u[..., None] * db)
            valid = (img[:, pa] != img[:, pb]) & (b < cos_min_tri) & (s > 0) & (u > 0)
            inl = errors(Xh) <= max_err
            score = np.where(valid, inl.sum(-1), -1)
            best = score.argmax(axis=1)
            g = np.arange(G)
            ok = score[g, best] >= 2
            err = errors(Xh[g, best][:, None, :])[:, 0]
            inl = one_per_image(err <= max_err, err)

            # Refined from the inliers (DLT; other rows weighted by zero), inliers again.
            x = self._rays(k.ravel()).reshape(G, L, 2)
            P = np.concatenate([self.R[img], self.t[img][..., None]], axis=3)
            A = np.concatenate([x[..., 0:1] * P[:, :, 2] - P[:, :, 0], x[..., 1:2] * P[:, :, 2] - P[:, :, 1]], axis=1)
            A *= np.tile(inl, 2)[..., None]
            V = np.linalg.svd(A)[2][:, -1]
            with np.errstate(divide="ignore", invalid="ignore"):
                X = V[:, :3] / V[:, 3:4]
            err = errors(X[:, None, :])[:, 0]
            inl = one_per_image(err <= max_err, err)
            # Two inliers or more, a wide enough angle between them.
            v = X[:, None, :] - C
            with np.errstate(divide="ignore", invalid="ignore"):
                v /= np.linalg.norm(v, axis=2, keepdims=True)
            cos = np.nan_to_num(np.einsum("gai,gbi->gab", v, v), nan=1.0)
            cos[~(inl[:, :, None] & inl[:, None, :])] = 1.0
            ok &= (inl.sum(1) >= 2) & (cos.min(axis=(1, 2)) < cos_min_tri) & np.isfinite(X).all(1)
            out_t.append(t[ok])
            out_X.append(X[ok])
            out_obs.append(k[ok][inl[ok]])
        if not out_t:
            return np.zeros(0, int), np.zeros((0, 3)), np.zeros(0, int)
        return np.concatenate(out_t), np.concatenate(out_X), np.concatenate(out_obs)

    def _update_points(self):
        t0 = time.perf_counter()
        reg = self.registered[self.o_image]
        # Continue the points: each registered image without an inlier gets its closest
        # keypoint within tri_max_angle_error_deg; one the filter dropped only within
        # max_reproj_px (COLMAP's complete_max_reproj_error).
        k = np.flatnonzero(reg & self.has_point[self.o_track] & ~self.obs_in)
        if len(k):
            X = self.X[self.o_track[k]]
            err, z = self._project(k, X)
            angle = self._angle_error(k, X)
            ok = np.where(self.obs_filtered[k], err <= self.o.max_reproj_px,
                          angle <= np.radians(self.o.tri_max_angle_error_deg)) & (z > 0)
            k, err = k[ok], angle[ok]
            seen = np.flatnonzero(self.obs_in)
            taken = np.isin(self.o_track[k] * self.n + self.o_image[k], self.o_track[seen] * self.n + self.o_image[seen])
            self.obs_in[self._one_per_image(k[~taken], err[~taken])] = True
        # New points for tracks without one.
        eligible = ~self.has_point & (self.track_views >= (3 if self.o.ignore_two_view_tracks else 2))
        use = reg & eligible[self.o_track]
        n_reg = np.bincount(self.o_track[use], minlength=len(self.track_views))
        use &= n_reg[self.o_track] >= 2
        tracks, X, obs = self._triangulate(np.flatnonzero(use))
        self.X[tracks] = X
        self.has_point[tracks] = True
        self.obs_in[obs] = True
        self.seconds["triangulate"] += time.perf_counter() - t0

    def _split(self):
        """A point that leaves keypoints of three registered images or more unused (in
        images it has no keypoint of) may be two points joined by a wrong match, or a
        repeated pattern: those keypoints are triangulated on their own and, if they make
        a point, move to a track of their own (the demo's split_leftovers, D140). After
        adjustments only, as a split is never undone."""
        t0 = time.perf_counter()
        try:
            self._split_leftovers()
        finally:
            self.seconds["split"] += time.perf_counter() - t0

    def _split_leftovers(self):
        k = np.flatnonzero(self.registered[self.o_image] & self.has_point[self.o_track] & ~self.obs_in)
        if not len(k):
            return
        pairs = np.unique(self.o_track[k] * self.n + self.o_image[k])
        views = np.bincount(pairs // self.n, minlength=len(self.track_views))
        split = np.flatnonzero(views >= 3)
        if not len(split):
            return
        k = k[np.isin(self.o_track[k], split)]
        old = self.o_track[k].copy()
        new_id = np.full(len(self.track_views), -1)
        new_id[split] = len(self.track_views) + np.arange(len(split))
        self.o_track[k] = new_id[old]
        order = np.argsort(self.o_track[k], kind="stable")
        k, old = k[order], old[order]
        tracks, X, obs = self._triangulate(k)
        made = np.zeros(len(self.track_views) + len(split), bool)
        made[tracks] = True
        back = ~made[self.o_track[k]]
        self.o_track[k[back]] = old[back]
        kept = split[made[new_id[split]]]
        if not len(kept):
            return
        # Renumber the tracks made so they follow the existing ones without gaps.
        final = np.full(len(made), -1)
        final[new_id[kept]] = len(self.track_views) + np.arange(len(kept))
        moved = k[~back]
        self.o_track[moved] = final[self.o_track[moved]]
        self.track_views = np.r_[self.track_views, views[kept]]
        self.has_point = np.r_[self.has_point, np.ones(len(kept), bool)]
        self.X = np.r_[self.X, X[np.argsort(final[tracks])]]
        self.obs_in[obs] = True
        split = kept
        # Observations stay sorted by track for _groups.
        order = np.argsort(self.o_track, kind="stable")
        for name in ("o_track", "o_image", "o_xy", "obs_in", "obs_filtered"):
            setattr(self, name, getattr(self, name)[order])
        self.splits += len(split)

    def _filter(self):
        k = np.flatnonzero(self.obs_in)
        err, z = self._project(k, self.X[self.o_track[k]])
        out = k[(err > self.o.max_reproj_px) | (z <= 0)]
        self.obs_in[out] = False
        self.obs_filtered[out] = True
        k = np.flatnonzero(self.obs_in)
        n_in = np.bincount(self.o_track[k], minlength=len(self.track_views))
        drop = self.has_point & (n_in < 2)
        for rows, t in _groups(self.o_track[k]):
            if rows.shape[1] >= 2:
                narrow = self._max_angle_deg(k[rows], self.X[t]) < self.o.min_tri_angle_deg
                drop[t[narrow]] = True
        self.has_point[drop] = False
        self.obs_in[drop[self.o_track]] = False

    # -- adjustment

    def _problem(self):
        images = np.flatnonzero(self.registered)
        dense = np.full(self.n, -1)
        dense[images] = np.arange(len(images))
        k = np.flatnonzero(self.obs_in & self.registered[self.o_image])
        n_in = np.bincount(self.o_track[k], minlength=len(self.track_views))
        k = k[n_in[self.o_track[k]] >= 2]
        tracks, obs_point = np.unique(self.o_track[k], return_inverse=True)
        p = jba.make_problem(
            "SIMPLE_RADIAL", self.K, poses=self.poses[images], points=self.X[tracks],
            obs_image=dense[self.o_image[k]].astype(np.int32), obs_point=obs_point.astype(np.int32),
            obs_xy=self.o_xy[k], intrinsics_free=np.array([self.o.refine_intrinsics], dtype=np.uint8))
        return p, images, tracks

    def _adjust(self, iterations):
        t0 = time.perf_counter()
        p, images, tracks = self._problem()
        o = jba.SolverOptions()
        o.max_iterations = iterations
        o.loss = jba.Loss(jba.LossType.CAUCHY, self.o.max_reproj_px)
        o.backend = self.ba_backend
        s = jba.solve(p, o)
        if s.termination == jba.Termination.FAILURE:
            raise RuntimeError(f"adjustment failed: {s.message}")
        self.poses[images] = p.poses
        self.X[tracks] = p.points
        self.K = np.asarray(p.intrinsics, dtype=np.float64).ravel().copy()
        self._sync()
        self.adjustments += 1
        self.seconds["adjust"] += time.perf_counter() - t0
        return p

    # -- the run

    def _verify(self):
        t0 = time.perf_counter()
        xa = [self.kp[a][idx[:, 0]] for a, b, idx in self.pairs]
        xb = [self.kp[b][idx[:, 1]] for a, b, idx in self.pairs]
        r = self.ctx.relative_pose(xa, xb, self._camera(), max_error_px=self.o.pair_max_error_px)
        self.pair_result = r
        m = np.array([len(idx) for _, _, idx in self.pairs])
        kept = ((r.status == 0) | (r.status == 3)) & (r.n_inliers >= self.o.min_pair_inliers) & \
               (r.n_inliers >= self.o.min_pair_inlier_ratio * m)
        self.pair_kept = kept
        verified = [np.column_stack([np.full(len(idx), a), idx[:, 0], np.full(len(idx), b), idx[:, 1]])[r.inliers[i]]
                    for i, (a, b, idx) in enumerate(self.pairs) if kept[i]]
        self.seconds["verify"] = time.perf_counter() - t0
        return np.concatenate(verified).astype(np.int32)

    def _tracks(self, verified):
        """The connected components of the verified matches, as the demo's mapper
        builds them (D140): a component with two keypoints in one image is kept, and
        triangulation takes at most one keypoint of an image for a point."""
        t0 = time.perf_counter()
        base = np.r_[0, np.cumsum([len(k) for k in self.kp])]
        u = base[verified[:, 0]] + verified[:, 1]
        v = base[verified[:, 2]] + verified[:, 3]
        label = np.arange(base[-1])
        while True:  # label propagation with pointer jumping: each component's smallest id
            m = np.minimum(label[u], label[v])
            np.minimum.at(label, u, m)
            np.minimum.at(label, v, m)
            while True:
                jumped = label[label]
                if np.array_equal(jumped, label):
                    break
                label = jumped
            if np.array_equal(label[u], label[v]):
                break
        used = np.zeros(base[-1], bool)
        used[u] = used[v] = True
        kps = np.flatnonzero(used)
        kps = kps[np.argsort(label[kps], kind="stable")]
        _, self.o_track = np.unique(label[kps], return_inverse=True)
        self.o_image = np.searchsorted(base, kps, side="right") - 1
        self.o_xy = np.concatenate(self.kp)[kps]
        self.n_tracks = n_tracks = int(self.o_track.max()) + 1
        self.track_views = np.bincount(np.unique(self.o_track * self.n + self.o_image) // self.n, minlength=n_tracks)
        self.has_point = np.zeros(n_tracks, bool)
        self.X = np.zeros((n_tracks, 3))
        self.obs_in = np.zeros(len(self.o_track), bool)
        self.obs_filtered = np.zeros(len(self.o_track), bool)  # dropped by a filter once
        self.seconds["tracks"] = time.perf_counter() - t0

    def _first_pair(self):
        r = self.pair_result
        for floor in (16.0, 8.0, 4.0, 2.0):
            ok = self.pair_kept & (r.status == 0) & (r.n_inliers >= self.o.min_first_pair_inliers) & \
                 (r.median_tri_angle_deg >= floor)
            if ok.any():
                return int(np.flatnonzero(ok)[np.argmax(r.n_inliers[ok])])
        raise RuntimeError("no pair to start from")

    def _register(self, failed_at):
        """One round of PnP; the images registered (may be empty: then those tried are
        marked in failed_at with the points they saw)."""
        k = np.flatnonzero(self.has_point[self.o_track] & ~self.registered[self.o_image])
        seen = np.bincount(self.o_image[k], minlength=self.n)
        order = [int(i) for i in np.argsort(-seen, kind="stable")
                 if not self.registered[i] and seen[i] >= self.o.min_pnp_inliers and seen[i] > 1.2 * failed_at[i]]
        cands = order[:self.o.pnp_batch]
        if not cands:
            return None
        t0 = time.perf_counter()
        xy, xyz = [], []
        for i in cands:
            ki = k[self.o_image[k] == i]
            ki = ki[np.unique(self.o_track[ki], return_index=True)[1]]  # one keypoint per point
            xy.append(self.o_xy[ki])
            xyz.append(self.X[self.o_track[ki]])
        r = self.ctx.absolute_pose(xy, xyz, self._camera(), max_error_px=self.o.pnp_max_error_px,
                                   min_inlier_ratio=self.o.min_pnp_inlier_ratio)
        self.seconds["pnp"] += time.perf_counter() - t0
        added = []
        for c, i in enumerate(cands):
            m = len(xy[c])
            if r.status[c] == 0 and r.n_inliers[c] >= self.o.min_pnp_inliers and \
                    r.n_inliers[c] >= self.o.min_pnp_inlier_ratio * m:
                self.poses[i] = r.pose[c]
                self.registered[i] = True
                added.append(i)
            else:
                failed_at[i] = seen[i]
        return added

    def run(self) -> Result:
        t_all = time.perf_counter()
        verified = self._verify()
        self._tracks(verified)
        self.poses = np.tile([1.0, 0, 0, 0, 0, 0, 0], (self.n, 1))
        self.registered = np.zeros(self.n, bool)

        first = self._first_pair()
        a, b, _ = self.pairs[first]
        self.poses[b] = self.pair_result.pose[first]
        self.registered[[a, b]] = True
        self._sync()
        self._update_points()
        self._adjust(self.o.local_iterations)
        self._filter()

        failed_at = np.zeros(self.n)
        while True:
            added = self._register(failed_at)
            if added is None:
                break
            if not added:
                continue
            self._sync()
            self._update_points()
            self._adjust(self.o.local_iterations)
            self._filter()
            self._split()

        # The finish: once more with the points the last adjustment and splits allow.
        for _ in range(2):
            self._update_points()
            self._adjust(self.o.final_iterations)
            self._filter()
            self._split()
        p, _, _ = self._problem()
        k = np.flatnonzero(self.obs_in & self.registered[self.o_image])
        err, _ = self._project(k, self.X[self.o_track[k]])
        self.seconds["total"] = time.perf_counter() - t_all
        return Result(registered=self.registered.copy(), poses=self.poses.copy(), intrinsics=self.K.copy(),
                      points=np.asarray(p.points), problem=p, reprojection_px=err,
                      pairs_kept=int(self.pair_kept.sum()), tracks=self.n_tracks, splits=self.splits,
                      first_pair=(int(a), int(b)), adjustments=self.adjustments, seconds=self.seconds)
