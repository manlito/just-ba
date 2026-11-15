"""Writes the generated pages of the documentation site into docs-site/_generated/
(D119), from the sources they describe, so they cannot drift:

- residuals.md and cameras.md from residuals/symbolic (the definitions the kernels
  are generated from): each module's formulas, each residual's inputs and outputs
  with their shapes, and each camera model's projection as LaTeX (via SymPy);
- api.md, the Python API reference, from the just_ba module: every public name in
  one of the groups below (an unplaced name is an error, so new ones are not missed);
- benchmarks.md from bench/baseline/*.json.

Run by conf.py on every build; needs wrenfold and sympy (docs-site/requirements.txt)
and an importable just_ba.
"""

from __future__ import annotations

import inspect
import json
import re
import sys
import typing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "residuals" / "symbolic"))

API_GROUPS = [
    ("Problems and files", "A problem holds the camera, poses, points, observations and priors.",
     ["Problem", "make_problem", "CameraModel", "num_params", "FrameRelation", "DepthKind", "DepthForm",
      "DEPTH_SCALE", "DEPTH_OFFSET", "add_relative_priors_from_poses", "read_problem", "write_problem", "read_bal",
      "write_result", "export_colmap"]),
    ("Refining a capture", "Keypoints, matches and pose priors in, a refined reconstruction out.",
     ["refine", "RefineOptions", "RefineResult", "initialize_poses", "PoseInitialization", "build_tracks", "Tracks",
      "TrackOptions", "TrackStats", "ConflictPolicy", "TriangulationOptions"]),
    ("Solving", "Levenberg-Marquardt on a problem that already has its points and observations.",
     ["solve", "SolverOptions", "SolverSummary", "IterationSummary", "Termination", "Loss", "LossType",
      "PriorDefaults", "GaugeMode", "Gauge", "choose_gauge", "store_losses", "with_problem_losses"]),
    ("Backends", "Where a solve runs.",
     ["Backend", "LinearSolver", "backend_available", "compiled_backends", "resolve_linear_solver"]),
    ("Diagnostics and uncertainty", "", ["evaluate", "ResidualReport", "covariance", "Covariance"]),
    ("Manhattan prior", "Line segments assigned to the axes of a Manhattan frame.",
     ["assign_lines", "apply_assignment", "ManhattanOptions", "ManhattanAssignment"]),
    ("Poses and conventions", "Converting tracker poses to the library's.",
     ["poses_from_camera_to_world", "camera_to_world_from_poses", "quat_from_matrix", "matrix_from_quat",
      "camera_centres"]),
    ("Package", "Which package and build this is.", ["__version__", "VARIANT", "CUDA_VERSION"]),
]
NOT_API = {"np", "annotations"}

MACHINES = {"apple-m5pro": "Apple M5 Pro", "rtx2070maxq": "Intel i7-9750H with an RTX 2070 Max-Q (laptop, Linux)"}
BAL_NAMES = {"problem-49-7776-pre": "ladybug-49 (BAL)", "problem-21-11315-pre": "trafalgar-21 (BAL)",
             "problem-16-22106-pre": "dubrovnik-16 (BAL)", "problem-52-64053-pre": "venice-52 (BAL)",
             "problem-93-61203-pre": "final-93 (BAL)"}
LIBRARIES = {"accelerate": "Apple Accelerate", "suite_sparse": "SuiteSparse", "eigen": "Eigen", "cuda": "cuDSS"}
SOLVER = {"iterative_schur": "iterative Schur (conjugate gradients)", "sparse_schur": "sparse Schur (direct)"}


def shape_of(annotation) -> tuple[int, int]:
    for meta in typing.get_args(annotation)[1:]:
        if hasattr(meta, "rows"):
            return meta.rows, meta.cols
    return 1, 1


def shape_text(r: int, c: int) -> str:
    if (r, c) == (1, 1):
        return "scalar"
    return f"{r}-vector" if c == 1 else f"{r} x {c}"


def call_symbolic(fn):
    from wrenfold import sym
    args = []
    for name, p in inspect.signature(fn).parameters.items():
        r, c = shape_of(p.annotation)
        if (r, c) == (1, 1):
            args.append(sym.symbols(name))
        else:
            args.append(sym.matrix([[sym.symbols(f"{name}_{i}_{j}") for j in range(c)] for i in range(r)]))
    return fn(*args)


def residual_block(fn) -> list[str]:
    lines = [f"### `{fn.__name__}`", "", "| Input | Shape |", "|---|---|"]
    for name, p in inspect.signature(fn).parameters.items():
        lines.append(f"| `{name}` | {shape_text(*shape_of(p.annotation))} |")
    lines += ["", "| Output | Shape | |", "|---|---|---|"]
    for out in call_symbolic(fn):
        shape = getattr(out.expression, "shape", (1, 1))
        lines.append(f"| `{out.name}` | {shape_text(*shape)} | {'optional' if out.is_optional else 'always'} |")
    return lines + [""]


def residuals_page() -> str:
    import depth
    import directions
    import priors
    import reprojection
    out = ["# Residual reference", "",
           "Generated from `residuals/symbolic`, the definitions the CPU, Metal and CUDA kernels are generated "
           "from. Jacobians are row-major, with respect to the tangent of each block: 6 values per pose, rotation "
           "first (a left perturbation, radians), then translation.", ""]
    modules = [("Reprojection", reprojection, [reprojection.make_reprojection(m) for m in reprojection.CAMERA_MODELS[:1]]),
               ("Pose priors", priors, priors.FUNCTIONS), ("Gravity and Manhattan lines", directions, directions.FUNCTIONS),
               ("Depth at keypoints", depth, depth.FUNCTIONS)]
    for title, module, functions in modules:
        out += [f"## {title}", "", "```text", inspect.cleandoc(module.__doc__ or ""), "```", ""]
        if module is reprojection:
            out += ["One function per camera model (see the camera models page); their inputs and outputs differ "
                    "only in the length of `intrinsics`. For SIMPLE_PINHOLE:", ""]
        for fn in functions:
            out += residual_block(fn)
    return "\n".join(out) + "\n"


def cameras_page() -> str:
    import sympy
    from wrenfold import sym
    from wrenfold.sympy_conversion import to_sympy
    import reprojection
    out = ["# Camera models", "",
           "Generated from `residuals/symbolic/reprojection.py`. A point $X$ in world coordinates is moved to the "
           "camera, $P = R X + t$, and divided by depth, $u = P_x / P_z$, $v = P_y / P_z$; the model maps $(u, v)$ "
           "to pixels. Names, ids and parameter order follow COLMAP.", "",
           "| Model | COLMAP id | Parameters |", "|---|---|---|"]
    for m in reprojection.CAMERA_MODELS:
        out.append(f"| `{m.name.upper()}` | {m.colmap_id} | {', '.join(f'`{p}`' for p in m.params)} |")
    out.append("")
    r_text = r"\sqrt{u^{2} + v^{2}}"
    for m in reprojection.CAMERA_MODELS:
        u, v = sym.symbols("u, v")
        k = [sym.symbols(p) for p in m.params]
        x, y = m.project(u, v, k)
        lx, ly = (sympy.latex(to_sympy(e)) for e in (x, y))
        note = ""
        if "atan" in lx:  # the fisheye model: name the angle so the formula reads
            theta = r"\operatorname{atan}{\left(" + r_text + r" \right)}"
            lx, ly = (s.replace(theta, r"\theta").replace(r_text, "r") for s in (lx, ly))
            note = r"with $r = \sqrt{u^2 + v^2}$ and $\theta = \operatorname{atan} r$."
        out += [f"## {m.name.upper()}", "", "$$", f"x = {lx}", "$$", "", "$$", f"y = {ly}", "$$", ""]
        if note:
            out += [note, ""]
    return "\n".join(out) + "\n"


def api_page() -> str:
    import just_ba
    public = {n for n in getattr(just_ba, "__all__", dir(just_ba)) if not n.startswith("_")} - NOT_API
    public |= {"__version__"}
    placed = [n for _, _, names in API_GROUPS for n in names]
    missing = sorted(public - set(placed))
    if missing:
        sys.exit(f"docs-site/gen/generate.py: public names not in any API group: {missing}")
    out = ["# Python API", "", "Everything is in the `just_ba` module (`import just_ba as jba`).", ""]
    for title, text, names in API_GROUPS:
        out += [f"## {title}", ""] + ([text, ""] if text else [])
        for n in names:
            obj = getattr(just_ba, n, None)
            if obj is None:
                sys.exit(f"docs-site/gen/generate.py: just_ba has no {n}")
            if inspect.isclass(obj):
                out += ["```{eval-rst}", f".. autoclass:: just_ba.{n}", "   :members:", "   :undoc-members:", "```", ""]
            elif callable(obj):
                out += ["```{eval-rst}", f".. autofunction:: just_ba.{n}", "```", ""]
            else:
                out += [f"### `{n}`", "", f"`{obj!r}` in this build.", ""]
    return "\n".join(out) + "\n"


def benchmarks_page() -> str:
    out = ["# Benchmarks", "",
           "Generated from the stored baselines (`bench/baseline`). Times are of the whole solver call, set-up "
           "included: warm is the best of three calls in a process that has already solved once, cold one call "
           "in a fresh process. Where shown, Ceres Solver solves the same problems with SPARSE_SCHUR and the same gauge; "
           "*equal cost* compares the time each needs to reach the worse of the two final costs.", ""]
    for path in sorted((ROOT / "bench" / "baseline").glob("*.json")):
        d = json.loads(path.read_text())
        stem = path.stem
        machine = next((v for k, v in MACHINES.items() if stem.startswith(k)), d.get("machine", {}).get("cpu", stem))
        solver = SOLVER.get(d.get("linear_solver"), d.get("linear_solver"))
        out += [f"## {d['backend'].upper()}, {solver}: {machine}", "",
                f"Commit `{d.get('commit', '?')}`, {d.get('threads') or 'all'} threads."]
        has_ceres = any("ceres" in p for p in d["problems"].values())
        if has_ceres:
            c = next(p["ceres"] for p in d["problems"].values() if "ceres" in p)
            out.append(f"Ceres {c['version']} with {LIBRARIES.get(c['sparse_library'], c['sparse_library'])}.")
        out.append("")
        head = "| Problem | Images | Observations | Iterations | Final cost | Warm (s) | Cold (s) |"
        rule = "|---|---:|---:|---:|---:|---:|---:|"
        if has_ceres:
            head += " Ceres (s) | Speed-up | At equal cost |"
            rule += "---:|---:|---:|"
        out += [head, rule]
        for name, p in d["problems"].items():
            row = (f"| {BAL_NAMES.get(name, name)} | {p['images']:,} | {p['observations']:,} | {p['iterations']} | {p['final_cost']:.6g} "
                   f"| {p['time_seconds']:.3f} | {p['cold_time_seconds']:.3f} |")
            if has_ceres:
                c = p.get("ceres")
                row += (f" {c['time_seconds']:.3f} | {c['speedup']:.1f}x | "
                        + (f"{c['equal_cost_speedup']:.1f}x |" if "equal_cost_speedup" in c else "- |")) if c else " - | - | - |"
            out.append(row)
        out.append("")
    return "\n".join(out) + "\n"


def main(dest: str) -> None:
    d = Path(dest)
    d.mkdir(parents=True, exist_ok=True)
    pages = {"residuals.md": residuals_page, "cameras.md": cameras_page, "api.md": api_page,
             "benchmarks.md": benchmarks_page}
    for name, make in pages.items():
        text = make()
        path = d / name
        if not path.exists() or path.read_text() != text:  # unchanged files keep Sphinx's cache
            path.write_text(text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "_generated"))
