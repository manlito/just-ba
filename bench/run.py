#!/usr/bin/env python3
"""Speed and iteration-count benchmark of the solver (document 05, regression gate;
document 12, item 2), on the CPU or the Metal backend (--backend, --linear-solver).

Solves a fixed set of problems with the CLI and writes one JSON file: per problem the
iteration count, accepted steps, linear (conjugate gradient) iterations, termination,
initial and final cost, and the best wall time of several runs. With --compare it checks
a run against a stored baseline of the same machine class:

  - counts and termination must be identical, and final costs agree to 1e-9 relative:
    these are deterministic for a given build and platform (D59), so any change means the
    arithmetic changed;
  - time per iteration may not grow by more than --max-slowdown (default 10%).

Counts are only comparable within a machine class (fused multiply-add on Apple and ARM
changes rounding; document 12). Times are taken by hand on an otherwise idle machine,
one problem at a time; hosted CI runners are too noisy for them.

  python3 bench/run.py --build build/dev --out bench/results/run.json
  python3 bench/run.py --build build/dev --compare bench/baseline/apple-m5pro.json
  python3 bench/run.py --build build/dev --backend metal --linear-solver sparse_schur \
      --compare bench/baseline/apple-m5pro-metal-sparse.json

A baseline records its backend and linear solver; --compare refuses a baseline of
another configuration.

--ceres <jba_ceres_solve> also solves every problem with Ceres (SPARSE_SCHUR, the same
gauge, all threads; --ceres-sparse-library picks the sparse Cholesky: Eigen's,
Accelerate's, SuiteSparse's or cuDSS, as that Ceres was built) and records its iterations, final cost and best time next to ours. Speed
reference of milestone 4: a Ceres built with third_party/ceres/build.cmake
-DACCELERATE=ON, and --ceres-sparse-library accelerate.

Problems: the five BAL problems (python3 tests/data/fetch_bal.py) and synthetic ones
written by jba_make_problem, which is built with the tests. Standard library only.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAL = [
    "problem-49-7776-pre",   # ladybug-49
    "problem-21-11315-pre",  # trafalgar-21
    "problem-16-22106-pre",  # dubrovnik-16
    "problem-52-64053-pre",  # venice-52
    "problem-93-61203-pre",  # final-93
]
# Synthetic: name, jba_make_problem options. Seeds are fixed, so the files are the same
# on every run of a given build. Every point is seen by every image in these, so they
# stay small (90,000 to 200,000 observations).
SYNTHETIC = [
    ("synthetic-pinhole-50", "--model PINHOLE --images 50 --points 4000 --seed 11"),
    ("synthetic-radial-free-30", "--model RADIAL --images 30 --points 3000 --free-intrinsics 7 --seed 12"),
    ("synthetic-relative-priors-50", "--model PINHOLE --images 50 --points 4000 --relative-priors 3 "
                                     "--prior-outliers 0.05 --seed 13"),
    ("synthetic-global-similarity-50", "--model PINHOLE --images 50 --points 4000 --relative-priors 2 "
                                       "--global-priors --global-relation similarity --seed 14"),
    ("synthetic-opencv-30", "--model OPENCV --images 30 --points 3000 --seed 15"),
]
# The main size range (D5: 50 to 2,000 images), with --large: a walked capture, each
# point seen by 8 neighbouring images, relative priors to the 2 next images. Not in
# the stored baselines; the CPU and Ceres take minutes on the largest.
LARGE = [
    ("synthetic-walk-500", "--model PINHOLE --images 500 --points 60000 --track-length 8 --relative-priors 2 "
                           "--seed 21"),
    ("synthetic-walk-2000", "--model PINHOLE --images 2000 --points 240000 --track-length 8 --relative-priors 2 "
                            "--seed 22"),
]
COUNTS = ("iterations", "accepted_steps", "linear_iterations", "termination")


def exe(build, *parts):
    for cfg in ("", "Release", "RelWithDebInfo", "Debug"):
        p = os.path.join(build, *parts[:-1], cfg, parts[-1])
        for cand in (p, p + ".exe"):
            if os.path.isfile(cand):
                return cand
    sys.exit(f"{os.path.join(build, *parts)} not found; build the CLI and the tests first")


def git_commit():
    try:
        sha = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def machine():
    info = {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
            "cpus": os.cpu_count()}
    if platform.system() == "Darwin":
        try:
            info["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                         text=True).stdout.strip()
        except OSError:
            pass
    return info


def solve(cli, path, args):
    out = subprocess.run([cli, "solve", path, "--threads", str(args.threads), "--backend", args.backend,
                          "--linear-solver", args.linear_solver], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{path}: {out.stderr.strip() or out.stdout.strip()}")
    return json.loads(out.stdout)


def ceres_solve(tool, path, args):
    out = subprocess.run([tool, path, "--linear-solver", "sparse_schur", "--sparse-library",
                          args.ceres_sparse_library, "--gauge", "auto"], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"ceres {path}: {out.stderr.strip() or out.stdout.strip()}")
    return json.loads(out.stdout)


def run(args):
    cli = exe(args.build, "cli", "just-ba")
    make = exe(args.build, "tests", "jba_make_problem")
    problems = []
    for name in BAL:
        p = os.path.join(ROOT, "tests", "data", "bal", name + ".txt")
        if os.path.isfile(p):
            problems.append((name, p))
        else:
            print(f"skipping {name}: not downloaded (python3 tests/data/fetch_bal.py)", file=sys.stderr)
    tmp = tempfile.mkdtemp(prefix="jba_bench_")
    for name, opts in SYNTHETIC + (LARGE if args.large else []):
        p = os.path.join(tmp, name + ".jba")
        subprocess.run([make, p, *opts.split()], check=True, capture_output=True)
        problems.append((name, p))
    if args.only:
        problems = [(n, p) for n, p in problems if any(o in n for o in args.only.split(","))]

    result = {"commit": git_commit(), "machine": machine(), "backend": args.backend,
              "linear_solver": args.linear_solver, "threads": args.threads, "repeats": args.repeats,
              "problems": {}}
    for name, path in problems:
        runs = [solve(cli, path, args) for _ in range(args.repeats)]
        first = runs[0]
        # The solver auto stands for, as the library resolved it for this build.
        result["linear_solver"] = first["linear_solver"]
        for r in runs[1:]:
            if any(r[k] != first[k] for k in COUNTS) or r["final_cost"] != first["final_cost"]:
                raise RuntimeError(f"{name}: repeated runs differ; the solver should be deterministic (D59)")
        best = min(r["time_seconds"] for r in runs)
        entry = {k: first[k] for k in ("images", "points", "observations", "relative_priors", "global_priors",
                                      "initial_cost", "final_cost", *COUNTS)}
        entry["time_seconds"] = best
        entry["time_per_iteration"] = best / max(1, first["iterations"])
        line = (f"{name:34s} {first['iterations']:3d} it {first['linear_iterations']:5d} lin  "
                f"{best:8.3f} s  {1000 * entry['time_per_iteration']:8.2f} ms/it  {first['termination']}")
        if args.ceres:
            cruns = [ceres_solve(args.ceres, path, args) for _ in range(args.repeats)]
            entry["ceres"] = {"sparse_library": args.ceres_sparse_library, "iterations": cruns[0]["iterations"],
                              "final_cost": cruns[0]["final_cost"], "termination": cruns[0]["termination"],
                              "time_seconds": min(r["time_seconds"] for r in cruns)}
            c = entry["ceres"]
            line += (f"  | ceres {c['iterations']:3d} it {c['time_seconds']:8.3f} s  x{c['time_seconds'] / best:5.2f}"
                     f"  cost diff {abs(first['final_cost'] - c['final_cost']) / c['final_cost']:.1e}")
        result["problems"][name] = entry
        print(line, file=sys.stderr)
    return result


def compare(result, baseline, max_slowdown):
    problems = []
    for key, default in (("backend", "cpu"), ("linear_solver", "iterative_schur")):
        if baseline.get(key, default) != result[key]:
            return [f"baseline is for {key} {baseline.get(key, default)}, this run is {result[key]}"]
    for name, b in baseline["problems"].items():
        r = result["problems"].get(name)
        if r is None:
            problems.append(f"{name}: not run")
            continue
        for k in COUNTS:
            if r[k] != b[k]:
                problems.append(f"{name}: {k} {b[k]} -> {r[k]}")
        rel = abs(r["final_cost"] - b["final_cost"]) / max(abs(b["final_cost"]), 1e-300)
        if rel > 1e-9:
            problems.append(f"{name}: final cost {b['final_cost']:.17g} -> {r['final_cost']:.17g} ({rel:.1e} relative)")
        ratio = r["time_per_iteration"] / b["time_per_iteration"]
        mark = "  SLOWER" if ratio > 1 + max_slowdown else ""
        print(f"{name:34s} time per iteration {1000 * b['time_per_iteration']:8.2f} -> "
              f"{1000 * r['time_per_iteration']:8.2f} ms ({100 * (ratio - 1):+5.1f}%){mark}", file=sys.stderr)
        if mark:
            problems.append(f"{name}: time per iteration {100 * (ratio - 1):+.1f}% (limit {100 * max_slowdown:.0f}%)")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", default=os.path.join(ROOT, "build"), help="build directory (default build/)")
    ap.add_argument("--out", help="write the results JSON here")
    ap.add_argument("--compare", help="baseline JSON of the same machine class")
    ap.add_argument("--max-slowdown", type=float, default=0.10, help="allowed growth of time per iteration")
    ap.add_argument("--repeats", type=int, default=3, help="runs per problem; the best time is kept")
    ap.add_argument("--threads", type=int, default=0, help="solver threads (0: all)")
    ap.add_argument("--backend", default="cpu", choices=["cpu", "metal", "cuda"])
    ap.add_argument("--linear-solver", default="auto", choices=["auto", "iterative_schur", "sparse_schur"],
                    help="auto: the library's rule for the backend, as this build resolves it")
    ap.add_argument("--large", action="store_true", help="add the main-size-range problems (LARGE)")
    ap.add_argument("--ceres", help="jba_ceres_solve to run on every problem too (speed reference)")
    ap.add_argument("--ceres-sparse-library", default="accelerate",
                    choices=["eigen", "accelerate", "suite_sparse", "cuda"])
    ap.add_argument("--only", help="comma-separated substrings of problem names")
    args = ap.parse_args()
    result = run(args)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
            f.write("\n")
    if args.compare:
        with open(args.compare) as f:
            baseline = json.load(f)
        problems = compare(result, baseline, args.max_slowdown)
        for p in problems:
            print("REGRESSION " + p, file=sys.stderr)
        sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
