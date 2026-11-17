#!/usr/bin/env python3
"""Write a COLMAP database and text model from a capture file with a reference, for tests.

    colmap_fixture.py capture.jbc out_dir [--camera-per-image]

out_dir/database.db gets COLMAP's tables (cameras, images, keypoints, matches,
two_view_geometries) and out_dir/sparse/ its text model (cameras.txt, images.txt,
points3D.txt), from the file's capture and reference. Ids are not the file's indices:
images are numbered in reverse with gaps, cameras and points from odd starts, keypoints
carry six columns as SIFT's affine shapes do, so colmap_capture.py has to map them back.
The same matches go into both match tables. It follows COLMAP's schema as documented;
it is not COLMAP's own output.
"""

import argparse
import array
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jbc  # noqa: E402

MODEL_NAMES = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL", 4: "OPENCV", 5: "OPENCV_FISHEYE"}
PAIR_ID_BASE = 2147483647

SCHEMA = """
CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, model INTEGER NOT NULL,
  width INTEGER NOT NULL, height INTEGER NOT NULL, params BLOB, prior_focal_length INTEGER NOT NULL);
CREATE TABLE images (image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, name TEXT NOT NULL UNIQUE,
  camera_id INTEGER NOT NULL);
CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL, cols INTEGER NOT NULL,
  data BLOB);
CREATE TABLE matches (pair_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL, cols INTEGER NOT NULL,
  data BLOB);
CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL,
  cols INTEGER NOT NULL, data BLOB, config INTEGER NOT NULL, F BLOB, E BLOB, H BLOB, qvec BLOB, tvec BLOB);
"""


def le(a):
    if sys.byteorder != "little":
        a = array.array(a.typecode, a)
        a.byteswap()
    return a.tobytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("capture")
    ap.add_argument("out_dir")
    ap.add_argument("--camera-per-image", action="store_true", help="one camera row per image, same values")
    args = ap.parse_args()

    s = jbc.read(args.capture)
    if "reference.registered" not in s:
        sys.exit("colmap_fixture: the capture file has no reference")
    model = s["camera.model"][2][0]
    np_ = s["intrinsics"][1][1]
    intr = s["intrinsics"][2]
    size = s["camera.size"][2] if "camera.size" in s else None
    kp_begin = s["keypoint.begin"][2]
    kp_xy = s["keypoint.xy"][2]
    n_img = len(kp_begin) - 1
    block = s["image.intrinsics"][2] if "image.intrinsics" in s else [0] * n_img
    names = bytes(s["image.names"][2]).decode().split("\n")
    pair_images, pair_begin, match = s["pair.images"][2], s["pair.begin"][2], s["pair.match"][2]
    ref_intr = s["reference.intrinsics"][2] if "reference.intrinsics" in s else intr
    registered, pose = s["reference.registered"][2], s["reference.pose"][2]
    xyz = s["reference.point.xyz"][2]
    obs_i, obs_p, obs_k = (s[f"reference.obs.{n}"][2] for n in ("image", "point", "keypoint"))

    image_id = [3 * (n_img - i) + 1 for i in range(n_img)]  # reversed, with gaps
    if args.camera_per_image:
        camera_id = [5 + 2 * i for i in range(n_img)]
        camera_block = {camera_id[i]: block[i] for i in range(n_img)}
    else:
        blocks = len(intr) // np_
        camera_id = [7 + block[i] for i in range(n_img)]
        camera_block = {7 + b: b for b in range(blocks)}

    os.makedirs(os.path.join(args.out_dir, "sparse"), exist_ok=True)
    db_path = os.path.join(args.out_dir, "database.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    db = sqlite3.connect(db_path)
    db.executescript(SCHEMA)
    for cid, b in sorted(camera_block.items()):
        w, h = (size[2 * b], size[2 * b + 1]) if size else (0, 0)
        db.execute("INSERT INTO cameras VALUES (?, ?, ?, ?, ?, 0)",
                   (cid, model, w, h, le(array.array("d", intr[np_ * b:np_ * b + np_]))))
    for i in range(n_img):
        db.execute("INSERT INTO images VALUES (?, ?, ?)", (image_id[i], names[i], camera_id[i]))
        rows = kp_begin[i + 1] - kp_begin[i]
        data = array.array("f")
        for k in range(kp_begin[i], kp_begin[i + 1]):
            data.extend([kp_xy[2 * k], kp_xy[2 * k + 1], 1.0, 0.0, 0.0, 1.0])
        db.execute("INSERT INTO keypoints VALUES (?, ?, 6, ?)", (image_id[i], rows, le(data)))
    for r in range(len(pair_images) // 2):
        a, b = pair_images[2 * r], pair_images[2 * r + 1]
        m = array.array("I", match[2 * pair_begin[r]:2 * pair_begin[r + 1]])
        id1, id2 = image_id[a], image_id[b]
        if id1 > id2:  # COLMAP keeps the smaller id first and swaps the columns with it
            id1, id2 = id2, id1
            m = array.array("I", [m[k ^ 1] for k in range(len(m))])
        pair_id = id1 * PAIR_ID_BASE + id2
        rows = len(m) // 2
        db.execute("INSERT INTO matches VALUES (?, ?, 2, ?)", (pair_id, rows, le(m)))
        db.execute("INSERT INTO two_view_geometries VALUES (?, ?, 2, ?, 2, NULL, NULL, NULL, NULL, NULL)",
                   (pair_id, rows, le(m)))
    db.commit()
    db.close()

    sparse = os.path.join(args.out_dir, "sparse")
    with open(os.path.join(sparse, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        for cid, b in sorted(camera_block.items()):
            w, h = (size[2 * b], size[2 * b + 1]) if size else (0, 0)
            params = " ".join(repr(v) for v in ref_intr[np_ * b:np_ * b + np_])
            f.write(f"{cid} {MODEL_NAMES[model]} {w} {h} {params}\n")
    point_id = [101 + 2 * j for j in range(len(xyz) // 3)]
    by_image = {}
    for k in range(len(obs_i)):
        by_image.setdefault(obs_i[k], {})[obs_k[k]] = point_id[obs_p[k]]
    with open(os.path.join(sparse, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        for i in range(n_img):
            if not registered[i]:
                continue
            p = " ".join(repr(v) for v in pose[7 * i:7 * i + 7])
            f.write(f"{image_id[i]} {p} {camera_id[i]} {names[i]}\n")
            seen = by_image.get(i, {})
            pts = [f"{kp_xy[2 * k]!r} {kp_xy[2 * k + 1]!r} {seen.get(k - kp_begin[i], -1)}"
                   for k in range(kp_begin[i], kp_begin[i + 1])]
            f.write(" ".join(pts) + "\n")
    tracks = {}
    for k in range(len(obs_i)):
        tracks.setdefault(obs_p[k], []).append(f"{image_id[obs_i[k]]} {obs_k[k]}")
    with open(os.path.join(sparse, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        for j in range(len(xyz) // 3):
            X = " ".join(repr(v) for v in xyz[3 * j:3 * j + 3])
            f.write(f"{point_id[j]} {X} 128 128 128 0.5 {' '.join(tracks.get(j, []))}\n")
    print(f"{args.out_dir}: database.db and sparse/ for {n_img} images")


if __name__ == "__main__":
    main()
