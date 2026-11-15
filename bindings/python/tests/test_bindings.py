import math
import os
import sys

import numpy as np
import pytest

import just_ba as jba

sys.path.insert(0, os.path.dirname(__file__))
import scene  # noqa: E402


def small_problem(seed=1, noise=0.3):
    rng = np.random.default_rng(seed)
    poses = []
    for i in range(8):
        a = 0.15 * (i - 3.5)
        poses.append(scene.look_pose(np.array([2 * np.sin(a), 0.0, -4 + 0.1 * i]), np.pi / 2 - a))
    poses = np.array(poses)
    pts = rng.uniform([-1, -1, -1], [1, 1, 1], size=(150, 3))
    R = jba.matrix_from_quat(poses[:, :4])
    obs_i, obs_j, xy = [], [], []
    for i in range(8):
        pc = pts @ R[i].T + poses[i, 4:]
        uv = scene.K[:2] * pc[:, :2] / pc[:, 2:] + scene.K[2:]
        for j in range(150):
            if pc[j, 2] > 0.5:
                obs_i.append(i)
                obs_j.append(j)
                xy.append(uv[j] + rng.normal(0, noise, 2))
    return jba.make_problem("PINHOLE", scene.K, poses=poses, points=pts, obs_image=obs_i, obs_point=obs_j,
                            obs_xy=np.array(xy)), poses, pts


def perturb(p, seed=2):
    rng = np.random.default_rng(seed)
    p.points = p.points + rng.normal(0, 0.02, p.points.shape)
    poses = p.poses
    poses[:, 4:] += rng.normal(0, 0.02, (poses.shape[0], 3))
    p.poses = poses


def test_arrays_round_trip_and_shapes():
    p, poses, pts = small_problem()
    assert p.model == jba.CameraModel.PINHOLE
    assert p.num_images == 8 and p.num_points == 150
    assert p.poses.shape == (8, 7) and p.points.shape == (150, 3)
    assert p.obs_image.dtype == np.int32 and p.obs_xy.dtype == np.float64
    np.testing.assert_array_equal(p.points, pts)
    # Returned arrays are copies.
    a = p.points
    a[0, 0] = 99
    assert p.points[0, 0] != 99
    with pytest.raises(ValueError):
        p.poses = np.zeros((3, 6))
    # Quaternions from single-precision sources are normalised on the way in.
    q = p.poses
    q[:, :4] = q[:, :4].astype(np.float32).astype(np.float64) * (1 + 3e-8)
    assert np.abs(np.linalg.norm(q[:, :4], axis=1) - 1).max() > 1e-8
    p.poses = q
    np.testing.assert_allclose(np.linalg.norm(p.poses[:, :4], axis=1), 1, atol=1e-15)
    assert p.validate() == ""
    p.model = jba.CameraModel.RADIAL
    assert p.model == jba.CameraModel.RADIAL and jba.num_params("RADIAL") == 5
    with pytest.raises(ValueError):
        p.model = "NO_SUCH_MODEL"


def test_file_matches_the_documented_reader(tmp_path):
    p, _, _ = small_problem()
    jba.add_relative_priors_from_poses(p, p.poses, np.array([[0, 1], [1, 2]]))
    p.glob_image = [3]
    p.glob_pose = p.poses[3:4]
    p.set_relation = [jba.FrameRelation.SIMILARITY]
    p.set_transform = [[1, 0, 0, 0, 0, 0, 0, 2.0]]
    path = str(tmp_path / "p.jba")
    jba.write_problem(path, p)

    # The reader from document 11, verbatim.
    def read_jba(path):
        data = open(path, "rb").read()
        assert data[:8] == b"\x89JBA\r\n\x1a\n"
        version, count = np.frombuffer(data, "<u4", 2, 8)
        assert version == 1
        dtypes = {1: "<u1", 2: "<i4", 3: "<i8", 4: "<f4", 5: "<f8"}
        pos, out = 16, {}
        for _ in range(count):
            n = int(np.frombuffer(data, "<u4", 1, pos)[0]); pos += 4
            name = data[pos:pos + n].decode(); pos += n
            dtype, ndim = data[pos], data[pos + 1]; pos += 4
            shape = tuple(int(d) for d in np.frombuffer(data, "<u8", ndim, pos)); pos += 8 * ndim
            pos = (pos + 7) // 8 * 8
            a = np.frombuffer(data, dtypes[dtype], int(np.prod(shape)), pos).reshape(shape)
            pos = (pos + a.nbytes + 7) // 8 * 8
            out[name] = a
        return out

    d = read_jba(path)
    np.testing.assert_array_equal(d["image.pose"], p.poses)
    np.testing.assert_array_equal(d["prior.relative.pose"], p.rel_pose)
    np.testing.assert_array_equal(d["prior.set.transform"], p.set_transform)
    q = jba.read_problem(path)
    np.testing.assert_array_equal(q.rel_images, [[0, 1], [1, 2]])
    np.testing.assert_array_equal(q.set_relation, [2])
    np.testing.assert_array_equal(q.obs_xy, p.obs_xy)


def test_solve_callback_and_cancellation():
    p, poses, pts = small_problem(noise=0.0)
    perturb(p)
    seen = []
    o = jba.SolverOptions()
    o.function_tolerance = 1e-12
    o.max_iterations = 100
    s = jba.solve(p, o, callback=lambda it: seen.append(it.cost))
    assert s.termination == jba.Termination.CONVERGENCE
    assert s.final_cost < 1e-12 * s.initial_cost
    assert len(seen) == s.iterations + 1 and seen == [h.cost for h in s.history]
    assert len(s.gauge.fixed_images) == 1

    p2, _, _ = small_problem()
    perturb(p2)
    s2 = jba.solve(p2, callback=lambda it: it.iteration < 2)
    assert s2.termination == jba.Termination.CANCELLED and s2.iterations == 2

    def boom(it):
        raise KeyError("stop here")

    p3, _, _ = small_problem()
    with pytest.raises(KeyError, match="stop here"):
        jba.solve(p3, callback=boom)

    bad, _, _ = small_problem()
    bad.obs_point = np.full(bad.num_observations, 999)
    with pytest.raises(ValueError):
        jba.solve(bad)


def test_evaluate_reports_costs():
    p, _, _ = small_problem()
    o = jba.SolverOptions()
    o.loss = jba.Loss("huber", 2.0)
    r = jba.evaluate(p, o)
    assert r.reprojection_error.shape == (p.num_observations,)
    assert r.in_front.all()
    assert 0 < r.reprojection_cost < 0.5 * np.sum((r.reprojection_error) ** 2) + 1e-9


def test_conventions():
    # An ARKit camera at the origin with the identity transform looks down -z with
    # y up: in library axes that is a rotation by pi about x.
    pose = jba.poses_from_camera_to_world(np.eye(4), "arkit")
    np.testing.assert_allclose(np.abs(pose[:4]), [0, 1, 0, 0], atol=1e-15)
    np.testing.assert_allclose(pose[4:], 0, atol=1e-15)
    # A world point in front of that camera (z = -2) is at camera depth +2.
    R = jba.matrix_from_quat(pose[:4])
    np.testing.assert_allclose(R @ [0, 0, -2.0] + pose[4:], [0, 0, 2], atol=1e-15)
    # Round trips, for random transforms and both conventions.
    rng = np.random.default_rng(4)
    for conv in ("opencv", "arkit", "arcore", "opengl"):
        Q = rng.normal(size=(5, 4))
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        T = np.zeros((5, 4, 4))
        T[:, :3, :3] = jba.matrix_from_quat(Q)
        T[:, :3, 3] = rng.normal(size=(5, 3))
        T[:, 3, 3] = 1
        back = jba.camera_to_world_from_poses(jba.poses_from_camera_to_world(T, conv), conv)
        np.testing.assert_allclose(back, T, atol=1e-12)
    with pytest.raises(ValueError):
        jba.poses_from_camera_to_world(np.eye(4), "unity")


def test_build_tracks():
    sc = scene.room(n_images=12, n_points=1500, noise=0.0)
    cams = jba.make_problem("PINHOLE", scene.K, poses=sc["poses"])
    tracks, stats = jba.build_tracks(sc["keypoints"], sc["matches"], jba.TrackOptions(), cams)
    assert len(tracks) == stats.tracks > 0
    assert stats.rejected_matches == 0
    # Every track holds keypoints of a single point.
    begin, img, kp = tracks.begin, tracks.image, tracks.keypoint
    for t in range(len(tracks)):
        ids = {int(sc["point_of"][img[o]][kp[o]]) for o in range(begin[t], begin[t + 1])}
        assert len(ids) == 1


def test_refine_end_to_end(tmp_path):
    sc = scene.room()
    truth, tracked = sc["poses"], sc["tracked"]
    n = truth.shape[0]
    pairs = np.array([(i, (i + d) % n) for i in range(n) for d in (1, 2, 3)], dtype=np.int32)
    s = jba.make_problem("PINHOLE", scene.K, poses=tracked)
    jba.add_relative_priors_from_poses(s, tracked, pairs)
    before = scene.aligned_errors(jba.initialize_poses(s).poses, truth)
    r = jba.refine(s, sc["keypoints"], matches=sc["matches"])
    after = scene.aligned_errors(r.problem.poses, truth)
    print(f"refine from Python: before {before[0]:.4f} m {math.degrees(before[1]):.3f} deg, "
          f"after {after[0]:.4f} m {math.degrees(after[1]):.4f} deg, points {r.problem.num_points}")
    assert r.num_groups == 1 and r.refined.all()
    # Scale comes from the drifting tracker's priors alone. With a narrow lens a small
    # loop trades rotation for translation cheaply; this scene's lens is wide.
    assert after[0] < 0.005 and after[1] < math.radians(0.1)
    assert after[0] < 0.5 * before[0] and after[1] < 0.5 * before[1]
    assert r.problem.num_points > 1000
    assert r.obs_keypoint.shape == (r.problem.num_observations,)
    np.testing.assert_array_equal(r.num_observations, np.bincount(r.problem.obs_image, minlength=n))
    assert jba.RefineOptions().min_observations == 30 and (r.num_observations >= 30).all()
    # Observations point at the keypoints they came from.
    for k in range(0, r.problem.num_observations, 97):
        i = r.problem.obs_image[k]
        np.testing.assert_array_equal(r.problem.obs_xy[k], sc["keypoints"][i][r.obs_keypoint[k]])
    assert len(r.report.downweighted_relative) == 0
    assert all(sm.termination == jba.Termination.CONVERGENCE for sm in r.summaries)
    # The solved problem carries the losses refine used (loss.* sections, D80).
    # One per residual type: reprojection, relative, global, gravity prior, ...
    o = jba.RefineOptions().solver
    expected = [o.loss, o.relative_prior_loss, o.global_prior_loss, o.gravity_loss]
    np.testing.assert_array_equal(r.problem.loss_type[:4], [l.type.value for l in expected])
    np.testing.assert_array_equal(r.problem.loss_scale[:4], [l.scale for l in expected])
    assert r.problem.loss_scale[0] == 2.0
    path = str(tmp_path / "solved.jba")
    jba.write_problem(path, r.problem)
    o = jba.with_problem_losses(jba.SolverOptions(), jba.read_problem(path))
    assert o.loss.type == jba.LossType.CAUCHY and o.loss.scale == 2.0

    with pytest.raises(ValueError):
        jba.refine(s, sc["keypoints"])  # neither matches nor tracks


def test_manhattan_lines():
    # Room edges and frames along the world axes, seen from the scene's cameras.
    sc = scene.room(n_images=20, n_points=500)
    rng = np.random.default_rng(3)
    lo, hi = np.array([-3.0, -1.2, -2.5]), np.array([3.0, 1.5, 2.5])
    segs = []
    for _ in range(300):
        axis = rng.integers(0, 3)
        a = rng.uniform(lo, hi)
        wall = rng.choice([k for k in range(3) if k != axis])
        a[wall] = lo[wall] if rng.random() < 0.5 else hi[wall]
        b = a.copy()
        b[axis] = np.clip(a[axis] + rng.uniform(0.5, 2.0), lo[axis], hi[axis])
        segs.append((a, b))
    R = jba.matrix_from_quat(sc["poses"][:, :4])
    line_image, line_xy = [], []
    for i in range(len(sc["poses"])):
        for a, b in segs:
            ends = []
            for X in (a, b):
                pc = R[i] @ X + sc["poses"][i, 4:]
                if pc[2] < 0.3:
                    break
                uv = scene.K[:2] * pc[:2] / pc[2] + scene.K[2:]
                if not (0 <= uv[0] < scene.WIDTH and 0 <= uv[1] < scene.HEIGHT):
                    break
                ends.append(uv)
            if len(ends) == 2 and np.linalg.norm(ends[1] - ends[0]) > 40:
                line_image.append(i)
                line_xy.append(np.concatenate(ends) + rng.normal(0, 0.3, 4))
    p = jba.make_problem("PINHOLE", scene.K, poses=sc["poses"], gravity_world=[0, 1, 0],
                         line_image=line_image, line_xy=np.array(line_xy))
    assert p.num_lines > 300 and not p.has_manhattan()
    a = jba.assign_lines(p)
    assert a.accepted, a.message
    assert a.support > 0.9 and sum(a.assigned) > 0.8 * p.num_lines
    jba.apply_assignment(p, a)
    assert p.has_manhattan() and p.manhattan_dof == 1
    np.testing.assert_array_equal(p.line_axis, a.axis)
    rep = jba.evaluate(p)
    assigned = p.line_axis >= 0
    assert np.nanmax(rep.line_angle[assigned]) < np.radians(3)
    assert np.isnan(rep.line_angle[~assigned]).all()
    # Random directions: refused.
    q = p.copy()
    q.line_xy = rng.uniform([0, 0, 0, 0], [640, 480, 640, 480], size=(p.num_lines, 4))
    r = jba.assign_lines(q)
    assert not r.accepted and "Manhattan" in r.message


def test_depth_in_refine():
    sc = scene.room(n_images=24, n_points=3000)
    n = len(sc["poses"])
    R = jba.matrix_from_quat(sc["poses"][:, :4])
    rng = np.random.default_rng(5)
    depth = []
    for i in range(n):
        pc = sc["points"][sc["point_of"][i]] @ R[i].T + sc["poses"][i, 4:]
        d = pc[:, 2] + rng.normal(0, 0.01, pc.shape[0])
        d[rng.random(d.size) < 0.5] = np.nan
        depth.append(d)
    s = jba.make_problem("PINHOLE", scene.K, poses=sc["tracked"])
    pairs = np.array([(i, (i + 1) % n) for i in range(n)], dtype=np.int32)
    jba.add_relative_priors_from_poses(s, sc["tracked"], pairs)
    r = jba.refine(s, sc["keypoints"], matches=sc["matches"], keypoint_depth=depth)
    assert r.problem.num_depth_observations > 0.3 * r.problem.num_observations
    rep = r.report
    has = ~np.isnan(r.problem.obs_depth)
    assert rep.depth_cost > 0
    np.testing.assert_allclose(rep.depth_refined[has], rep.depth_corrected[has], atol=0.1)
    assert np.isnan(rep.depth_refined[~has]).all()
    p = r.problem
    assert p.depth_kind == jba.DepthKind.Z and p.depth_form == jba.DepthForm.LINEAR
