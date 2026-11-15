"""Cost 0.5 * sum r^2 of a COLMAP text model (SIMPLE_RADIAL, PINHOLE or RADIAL), in double.

  python3 -I bench/competitors/colmap_cost.py <model_txt_dir>

Prints the number of observations and the cost. Used for Caspar's results, whose own
score is computed in single precision (bench/competitors/README.md).
"""
import sys
from pathlib import Path


def quat_to_R(q):
    w, x, y, z = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def main(d):
    d = Path(d)
    cams = {}
    for line in (d / "cameras.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        cams[int(t[0])] = (t[1], [float(v) for v in t[4:]])
    pts = {}
    for line in (d / "points3D.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        pts[int(t[0])] = [float(v) for v in t[1:4]]
    lines = [l for l in (d / "images.txt").read_text().splitlines() if not l.startswith("#")]
    total, n = 0.0, 0
    for a, b in zip(lines[0::2], lines[1::2]):
        t = a.split()
        R = quat_to_R([float(v) for v in t[1:5]])
        tv = [float(v) for v in t[5:8]]
        model, prm = cams[int(t[8])]
        b = b.split()
        for k in range(0, len(b), 3):
            x, y, pid = float(b[k]), float(b[k + 1]), int(b[k + 2])
            if pid < 0 or pid not in pts:
                continue
            X = pts[pid]
            P = [R[r][0] * X[0] + R[r][1] * X[1] + R[r][2] * X[2] + tv[r] for r in range(3)]
            u, v = P[0] / P[2], P[1] / P[2]
            r2 = u * u + v * v
            if model == "SIMPLE_RADIAL":
                f, cx, cy, kk = prm
                ex, ey = f * (1 + kk * r2) * u + cx - x, f * (1 + kk * r2) * v + cy - y
            elif model == "RADIAL":
                f, cx, cy, k1, k2 = prm
                dd = 1 + k1 * r2 + k2 * r2 * r2
                ex, ey = f * dd * u + cx - x, f * dd * v + cy - y
            elif model == "PINHOLE":
                fx, fy, cx, cy = prm
                ex, ey = fx * u + cx - x, fy * v + cy - y
            else:
                raise SystemExit(f"unsupported model {model}")
            total += ex * ex + ey * ey
            n += 1
    print(f"{n} {0.5 * total!r}")


main(sys.argv[1])
