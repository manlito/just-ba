#!/usr/bin/env python3
"""Write a capture file (.jbc) for jba_demo from a COLMAP database, with COLMAP's model.

    colmap_capture.py database.db -o capture.jbc [--model sparse/0] [--raw] [--min-matches 15]

From the database: the cameras, the images, their keypoints and the matches of every
pair. By default the verified matches (two_view_geometries), so both reconstructions
start from the same matches (document 05, level 3); --raw takes the matcher's output
(matches) instead, for the demo's own verification. With --model, COLMAP's
reconstruction in its text format (cameras.txt, images.txt, points3D.txt) becomes the
file's reference: its refined intrinsics, poses, points and observations. A binary
model is converted first:

    colmap model_converter --input_path sparse/0 --output_path sparse/0-txt --output_type TXT

Standard library only. The cameras of the images must share one model of the six the
library has.
"""

import argparse
import array
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jbc  # noqa: E402

MODEL_IDS = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1, "SIMPLE_RADIAL": 2, "RADIAL": 3, "OPENCV": 4, "OPENCV_FISHEYE": 5}
NUM_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8}
PAIR_ID_BASE = 2147483647  # COLMAP: pair_id = id1 * 2147483647 + id2 with id1 < id2


def blob(data, typecode):
    a = array.array(typecode)
    a.frombytes(data or b"")
    if sys.byteorder != "little":
        a.byteswap()
    return a


def read_database(path, raw, min_matches):
    if not os.path.isfile(path):
        sys.exit(f"colmap_capture: no database at {path}")
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    cameras = {}
    for camera_id, model, width, height, params in db.execute(
            "SELECT camera_id, model, width, height, params FROM cameras ORDER BY camera_id"):
        cameras[camera_id] = (model, width, height, list(blob(params, "d")))
    images = list(db.execute("SELECT image_id, name, camera_id FROM images ORDER BY image_id"))
    if not images:
        sys.exit("colmap_capture: the database has no images")
    index = {image_id: i for i, (image_id, _, _) in enumerate(images)}

    keypoints = [array.array("d") for _ in images]
    for image_id, rows, cols, data in db.execute("SELECT image_id, rows, cols, data FROM keypoints"):
        if image_id not in index or rows == 0:
            continue
        a = blob(data, "f")
        if len(a) != rows * cols or cols < 2:
            sys.exit(f"colmap_capture: keypoints of image {image_id} have a bad shape")
        xy = keypoints[index[image_id]]
        for r in range(rows):
            xy.append(a[r * cols])
            xy.append(a[r * cols + 1])

    table = "matches" if raw else "two_view_geometries"
    pairs = []
    for pair_id, rows, cols, data in db.execute(f"SELECT pair_id, rows, cols, data FROM {table} WHERE rows > 0"):
        id2 = pair_id % PAIR_ID_BASE
        id1 = (pair_id - id2) // PAIR_ID_BASE
        if id1 not in index or id2 not in index or rows < min_matches:
            continue
        m = blob(data, "I")
        if cols != 2 or len(m) != 2 * rows:
            sys.exit(f"colmap_capture: matches of pair {pair_id} have a bad shape")
        a, b = index[id1], index[id2]
        if a > b:  # images are ordered by id, so this does not happen; kept for safety
            a, b = b, a
            m = array.array("I", [m[k ^ 1] for k in range(len(m))])
        pairs.append((a, b, m))
    pairs.sort(key=lambda p: (p[0], p[1]))
    db.close()
    return cameras, images, keypoints, pairs


def read_model(directory):
    names = {f: os.path.join(directory, f) for f in ("cameras.txt", "images.txt", "points3D.txt")}
    if not all(os.path.isfile(p) for p in names.values()):
        hint = (" (a binary model: convert it with colmap model_converter --output_type TXT)"
                if os.path.isfile(os.path.join(directory, "images.bin")) else "")
        sys.exit(f"colmap_capture: {directory} lacks cameras.txt, images.txt or points3D.txt{hint}")

    def lines(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    yield line

    cameras = {}
    for line in lines(names["cameras.txt"]):
        v = line.split()
        if v[1] not in MODEL_IDS:
            sys.exit(f"colmap_capture: camera model {v[1]} is not one of {', '.join(MODEL_IDS)}")
        cameras[int(v[0])] = (MODEL_IDS[v[1]], [float(x) for x in v[4:]])
    images = {}
    with open(names["images.txt"]) as f:
        rows = [line.rstrip("\n") for line in f if not line.startswith("#")]
    for k in range(0, len(rows), 2):  # a pose line, then its POINTS2D line (possibly empty)
        v = rows[k].split()
        if not v:
            continue
        images[int(v[0])] = ([float(x) for x in v[1:8]], int(v[8]), " ".join(v[9:]))
    points = []
    for line in lines(names["points3D.txt"]):
        v = line.split()
        track = [int(x) for x in v[8:]]
        points.append(([float(x) for x in v[1:4]], list(zip(track[0::2], track[1::2]))))
    return cameras, images, points


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("database")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--model", help="COLMAP's reconstruction, text format, as the reference")
    ap.add_argument("--raw", action="store_true", help="the matcher's matches, not the verified ones")
    ap.add_argument("--min-matches", type=int, default=15, help="pairs with fewer are left out (default 15)")
    args = ap.parse_args()

    cameras, images, keypoints, pairs = read_database(args.database, args.raw, args.min_matches)
    used = sorted({camera_id for _, _, camera_id in images})
    missing = [c for c in used if c not in cameras]
    if missing:
        sys.exit(f"colmap_capture: images name cameras {missing} the database does not have")
    models = {cameras[c][0] for c in used}
    if len(models) != 1 or next(iter(models)) not in NUM_PARAMS:
        sys.exit("colmap_capture: the images' cameras must share one model of: " + ", ".join(MODEL_IDS))
    model = models.pop()
    np_ = NUM_PARAMS[model]
    block = {c: b for b, c in enumerate(used)}
    intrinsics = []
    for c in used:
        if len(cameras[c][3]) != np_:
            sys.exit(f"colmap_capture: camera {c} has {len(cameras[c][3])} parameters, its model {np_}")
        intrinsics += cameras[c][3]

    n_img = len(images)
    kp_begin = [0]
    kp_xy = array.array("d")
    for xy in keypoints:
        kp_xy.extend(xy)
        kp_begin.append(len(kp_xy) // 2)
    pair_images, pair_begin, match = [], [0], array.array("i")
    for a, b, m in pairs:
        pair_images += [a, b]
        match.frombytes(m.tobytes())  # u32 to i32: keypoint indices stay below 2^31
        pair_begin.append(len(match) // 2)
    names = "\n".join(name for _, name, _ in images).encode()
    sections = [
        ("camera.model", jbc.I32, (1,), [model]),
        ("intrinsics", jbc.F64, (len(used), np_), intrinsics),
        ("camera.size", jbc.I32, (len(used), 2), [v for c in used for v in cameras[c][1:3]]),
        ("image.intrinsics", jbc.I32, (n_img,), [block[c] for _, _, c in images]),
        ("image.names", jbc.U8, (len(names),), names),
        ("keypoint.begin", jbc.I64, (n_img + 1,), kp_begin),
        ("keypoint.xy", jbc.F64, (len(kp_xy) // 2, 2), kp_xy),
        ("pair.images", jbc.I32, (len(pairs), 2), pair_images),
        ("pair.begin", jbc.I64, (len(pairs) + 1,), pair_begin),
        ("pair.match", jbc.I32, (len(match) // 2, 2), match),
    ]

    summary = f"{n_img} images, {len(used)} cameras ({model}), {len(kp_xy) // 2} keypoints, " \
              f"{len(pairs)} pairs, {len(match) // 2} {'raw' if args.raw else 'verified'} matches"
    if args.model:
        m_cameras, m_images, m_points = read_model(args.model)
        index = {image_id: i for i, (image_id, _, _) in enumerate(images)}
        ref_intrinsics = []
        for c in used:
            params = m_cameras.get(c, (model, cameras[c][3]))[1]
            if m_cameras.get(c, (model,))[0] != model or len(params) != np_:
                sys.exit(f"colmap_capture: the model's camera {c} differs in model from the database's")
            ref_intrinsics += params
        registered = [0] * n_img
        poses = [1.0, 0, 0, 0, 0, 0, 0] * n_img
        for image_id, (pose, camera_id, name) in m_images.items():
            i = index.get(image_id)
            if i is None or images[i][1] != name or images[i][2] != camera_id:
                sys.exit(f"colmap_capture: the model's image {image_id} ({name}) is not the database's")
            registered[i] = 1
            poses[7 * i:7 * i + 7] = pose
        xyz, obs_image, obs_point, obs_keypoint = [], [], [], []
        for X, track in m_points:
            j = len(xyz) // 3
            xyz += X
            for image_id, k in track:
                i = index[image_id]
                if not registered[i] or k >= len(keypoints[i]) // 2:
                    sys.exit(f"colmap_capture: a point's track names keypoint {k} of image {image_id}, "
                             "which the database or the model does not have")
                obs_image.append(i)
                obs_point.append(j)
                obs_keypoint.append(k)
        n_obs = len(obs_image)
        sections += [
            ("reference.intrinsics", jbc.F64, (len(used), np_), ref_intrinsics),
            ("reference.registered", jbc.U8, (n_img,), registered),
            ("reference.pose", jbc.F64, (n_img, 7), poses),
            ("reference.point.xyz", jbc.F64, (len(xyz) // 3, 3), xyz),
            ("reference.obs.image", jbc.I32, (n_obs,), obs_image),
            ("reference.obs.point", jbc.I32, (n_obs,), obs_point),
            ("reference.obs.keypoint", jbc.I32, (n_obs,), obs_keypoint),
        ]
        summary += f"; reference: {sum(registered)} images, {len(xyz) // 3} points, {n_obs} observations"
    jbc.write(args.output, sections)
    print(f"{args.output}: {summary}")


if __name__ == "__main__":
    main()
