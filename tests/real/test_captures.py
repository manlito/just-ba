"""Real sessions through the Python API (milestone 2's exit, document 07).

The captures are not in the repository: they are private and large. Each is a .jba
file as exported by the owner's matching tool (just-ba-match), in tests/data or
tests/data/real (both git-ignored), or in the folder named by JBA_REAL_CAPTURES.
Besides the Problem (tracker poses, one camera, relative priors) it carries
keypoints and matches as extra sections, read here with document 11's reader:

  match.keypoint_begin  i64 [N+1]   keypoints of image i: rows begin[i]..begin[i+1]-1
  match.keypoints       f64 [K, 2]  pixels, the intrinsics' convention
  match.matches         i32 [M, 4]  image_a, keypoint_a, image_b, keypoint_b

Without captures every test is skipped, so CI runs none of this.

The checks are loose on purpose: real sessions have no ground truth, and some are
poor (tracking lost and regained, frames with nothing matched). They catch a
pipeline that breaks, not one that is a little worse; the printed numbers are for
reading.
"""

import glob
import json
import math
import os

import numpy as np
import pytest

import just_ba as jba

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def capture_files():
    if os.environ.get("JBA_REAL_CAPTURES"):
        dirs = [os.environ["JBA_REAL_CAPTURES"]]
    else:
        dirs = [DATA, os.path.join(DATA, "real")]
    files = sorted({os.path.realpath(f) for d in dirs for f in glob.glob(os.path.join(d, "*.jba"))})
    return [f for f in files if "match.matches" in read_sections(f)]


def read_sections(path):
    """Document 11's reference reader."""
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


def relative_change(before, after, group, refined):
    """Median change of the relative pose between consecutive refined images of a
    group, from the tracker's poses to the refined ones: degrees and metres."""
    Ra, Rb = jba.matrix_from_quat(before[:, :4]), jba.matrix_from_quat(after[:, :4])
    Ca, Cb = jba.camera_centres(before), jba.camera_centres(after)
    rot, pos = [], []
    for i in range(len(before) - 1):
        j = i + 1
        if not (refined[i] and refined[j] and group[i] == group[j]):
            continue
        D = (Ra[j] @ Ra[i].T).T @ (Rb[j] @ Rb[i].T)
        rot.append(math.degrees(math.acos(np.clip((np.trace(D) - 1) / 2, -1, 1))))
        pos.append(np.linalg.norm(Ra[i] @ (Ca[j] - Ca[i]) - Rb[i] @ (Cb[j] - Cb[i])))
    return float(np.median(rot)), float(np.median(pos))


@pytest.mark.parametrize("path", capture_files(), ids=os.path.basename)
def test_capture_refines(path):
    sec = read_sections(path)
    begin, kp = sec["match.keypoint_begin"], sec["match.keypoints"]
    keypoints = [kp[begin[i]:begin[i + 1]] for i in range(len(begin) - 1)]
    scene = jba.read_problem(path)
    assert scene.validate() == ""
    assert len(keypoints) == scene.num_images

    r = jba.refine(scene, keypoints, matches=np.asarray(sec["match.matches"]))
    P, rep = r.problem, r.report
    err = np.asarray(rep.reprojection_error)
    refined = r.refined.astype(bool)
    rot, pos = relative_change(scene.poses, P.poses, np.asarray(r.group), refined)
    info = json.loads(bytes(sec["capture.info"]).decode()) if "capture.info" in sec else {}
    print(f"\n{os.path.basename(path)} ({info.get('device', {}).get('model', '?')}): "
          f"{scene.num_images} images, {r.num_groups} groups, {int(refined.sum())} refined, "
          f"{P.num_points} points, reprojection median {np.median(err):.2f} px, p90 {np.percentile(err, 90):.2f} px, "
          f"{len(rep.downweighted_relative)}/{scene.num_relative_priors} priors down-weighted, "
          f"consecutive poses moved {rot:.2f} deg, {1000 * pos:.1f} mm (medians)")

    assert all(s.termination != jba.Termination.FAILURE for s in r.summaries)
    assert rep.in_front.all()
    assert np.median(err) < 1.5
    assert refined.mean() >= 0.75
    assert P.num_points >= 2000 * r.num_groups
    assert len(rep.downweighted_relative) <= 0.15 * scene.num_relative_priors
    # The tracker is right to within a degree and a few centimetres between neighbours.
    assert rot < 1.0 and pos < 0.03
