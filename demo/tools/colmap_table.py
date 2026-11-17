#!/usr/bin/env python3
"""The comparison with COLMAP over several scenes, as Markdown tables.

    colmap_table.py path/to/jba_demo capture.jbc... [-- jba_demo run options]

Runs `jba_demo run <capture> --json` on each capture file and prints two tables: ours
against COLMAP's reconstruction (document 05, level 3), and both against the ground
truth for the scenes that have one. Points, observations and track length are given as
ours, and our difference from COLMAP's in percent.
"""

import json
import os
import subprocess
import sys
import tempfile


def pct(a, b):
    return f"{100.0 * (a - b) / b:+.1f} %" if b else "-"


def focal(d):
    """Our refined focal length against COLMAP's, median over blocks, and where it started."""
    f0, f1, fr = d.get("focal_given", 0), d.get("focal_ours", 0), d.get("focal_reference", 0)
    if not fr:
        return "-"
    if abs(f1 - f0) < 1e-9 * max(f0, 1):  # kept fixed
        return f"{f1:,.0f} px, fixed ({pct(f1, fr)})"
    return f"{f1:,.0f} px ({pct(f1, fr)}), from {f0:,.0f}"


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    demo, rest = sys.argv[1], sys.argv[2:]
    extra = []
    if "--" in rest:
        k = rest.index("--")
        rest, extra = rest[:k], rest[k + 1:]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for capture in rest:
            out = os.path.join(tmp, "r.json")
            r = subprocess.run([demo, "run", capture, "--json", out] + extra, capture_output=True, text=True)
            if r.returncode not in (0, 1) or not os.path.exists(out):
                sys.exit(f"jba_demo failed on {capture}:\n{r.stdout}{r.stderr}")
            with open(out) as f:
                d = json.load(f)
            d["scene"] = os.path.splitext(os.path.basename(capture))[0]
            rows.append(d)
            os.remove(out)

    print("| Scene | Images | Registered, ours / COLMAP | Rotation median / max | Centre median / max | "
          "Points | Observations | Track length | Reprojection, ours / COLMAP | Focal, ours / COLMAP | "
          "COLMAP points split | Our run |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in rows:
        o, c, p = d["ours"], d["reference"], d["ours_vs_reference"]
        print(f"| {d['scene']} | {d['images']} | {o['registered']} / {c['registered']} | "
              f"{p['rot_median_deg']:.4f}° / {p['rot_max_deg']:.3f}° | "
              f"{p['centre_median_pct']:.4f} % / {p['centre_max_pct']:.3f} % | "
              f"{o['points']:,} ({pct(o['points'], c['points'])}) | "
              f"{o['observations']:,} ({pct(o['observations'], c['observations'])}) | "
              f"{o['track_length']:.2f} ({pct(o['track_length'], c['track_length'])}) | "
              f"{o['reprojection_px']:.3f} / {c['reprojection_px']:.3f} px | "
              f"{focal(d)} | "
              f"{d['reference_points_split']:,} ({100.0 * d['reference_points_split'] / max(c['points'], 1):.1f} %) | "
              f"{d['seconds']:.1f} s |")
    truth = [d for d in rows if "ours_vs_truth" in d]
    if truth:
        print()
        print("| Scene | Truth images | Ours: registered, rotation median / max, centre median | "
              "COLMAP: registered, rotation median / max, centre median |")
        # Centre errors in mm assume the truth is in metres, as ETH3D's is.
        print("|---|---|---|---|")
        for d in truth:
            cells = []
            for k in ("ours_vs_truth", "reference_vs_truth"):
                a = d[k]
                cells.append(f"{a['common']}, {a['rot_median_deg']:.3f}° / {a['rot_max_deg']:.3f}°, "
                             f"{1000 * a['centre_median']:.1f} mm ({a['centre_median_pct']:.3f} %)")
            print(f"| {d['scene']} | {d['truth_images']} | {cells[0]} | {cells[1]} |")


if __name__ == "__main__":
    main()
