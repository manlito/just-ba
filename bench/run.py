#!/usr/bin/env python3
"""Speed and iteration-count benchmark of the solver (document 05, regression gate;
document 12, item 2), on the CPU, Metal or CUDA backend (--backend, --linear-solver).

Solves a fixed set of problems with the CLI and writes one JSON file: per problem the
iteration count, accepted steps, linear (conjugate gradient) iterations, termination,
initial and final cost, the linear solver used, and times of the whole solver call
(set-up included, D101): `time_seconds`, the best of several warm calls (the CLI's
--warm-up: the device already started, as in a process that solves more than once),
its `setup_seconds`, and `cold_time_seconds`, one call in a fresh process (device
start-up included, as a command-line run pays it). With --compare it checks a run
against a stored baseline of the same machine class:

  - counts and termination must be identical, and final costs agree to 1e-9 relative:
    these are deterministic for a given build and platform (D59), so any change means the
    arithmetic changed;
  - time per iteration (warm call time over iterations) may not grow by more than
    --max-slowdown (default 10%).

Counts are only comparable within a machine class (fused multiply-add on Apple and ARM
changes rounding; document 12). Times are taken by hand on an otherwise idle machine,
one problem at a time; hosted CI runners are too noisy for them.

  python3 bench/run.py --build build/dev --out bench/results/run.json
  python3 bench/run.py --build build/dev --compare bench/baseline/apple-m5pro.json
  python3 bench/run.py --build build/dev --backend metal --linear-solver sparse_schur \
      --compare bench/baseline/apple-m5pro-metal-sparse.json

A baseline records its backend and linear solver; --compare refuses a baseline of
another configuration, and with --only compares only the problems run. --skip leaves
problems out (CI leaves out venice-52, as its quick mode does).

Tolerances: the defaults above suit a baseline recorded on the same machine. A
baseline from another machine of the same class, such as a hosted CI runner whose CPU
varies between runs, takes looser ones: --cost-rtol (final cost, relative),
--count-slack (iteration, accepted-step and linear-iteration counts, relative) and
--max-slowdown. CI uses 1e-6 (document 05's Level 2 tolerance), 0.1 and 0.5, with
--time-same-cpu: times are compared only on a runner with the baseline's CPU model,
since hosted runners of one label differ in speed by up to 2x (D117).

Stored results (document 05: per commit, so trends can be plotted): --store <machine>
also writes the run to bench/results/<machine>/<date>-<commit>-<backend>-<solver>.json,
with the full commit, the date, and the build's compiler and build type. Commit them.

--ceres <jba_ceres_solve> also solves every problem with Ceres (SPARSE_SCHUR, the same
gauge, all threads; --ceres-sparse-library picks the sparse Cholesky: Eigen's,
Accelerate's, SuiteSparse's or cuDSS, as that Ceres was built), column "ceres";
--ceres-gpu <jba_ceres_solve> another Ceres with cuDSS, column "ceres_gpu". Each
records Ceres's iterations, final cost, wall time of the call (building its problem
included) and the speed-up at equal accuracy: the time each solver needs to reach the
worse of the two final costs (relative 1e-6), from the per-iteration histories (D99).
Speed reference of milestone 4: a Ceres built with third_party/ceres/build.cmake
-DACCELERATE=ON, and --ceres-sparse-library accelerate.

Problems: the five BAL problems (python3 tests/data/fetch_bal.py) and synthetic ones
written by jba_make_problem, which is built with the tests; --large and --huge add
walked captures of 500 to 10,000 images. Standard library only.
"""

import argparse
import datetime
import glob
import json
import os
import platform
import re
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
# With --huge: document 07's 10,000-image walk (9.6 million observations).
HUGE = [
    ("synthetic-walk-10000", "--model PINHOLE --images 10000 --points 1200000 --track-length 8 --relative-priors 2 "
                             "--seed 23"),
]
EQUAL_COST = 1e-6  # D99
COUNTS = ("iterations", "accepted_steps", "linear_iterations", "termination")


def exe(build, *parts):
    for cfg in ("", "Release", "RelWithDebInfo", "Debug"):
        p = os.path.join(build, *parts[:-1], cfg, parts[-1])
        for cand in (p, p + ".exe"):
            if os.path.isfile(cand):
                return cand
    sys.exit(f"{os.path.join(build, *parts)} not found; build the CLI and the tests first")


def git_commit(short=True):
    try:
        cmd = ["git", "-C", ROOT, "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
        sha = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_info(build):
    """Compiler and build type of the build directory, from CMake's own files."""
    info = {}
    try:
        with open(os.path.join(build, "CMakeCache.txt")) as f:
            for line in f:
                m = re.match(r"CMAKE_BUILD_TYPE:\w+=(.*)", line)
                if m:
                    info["build_type"] = m.group(1).strip()
    except OSError:
        pass
    for path in glob.glob(os.path.join(build, "CMakeFiles", "*", "CMakeCXXCompiler.cmake")):
        with open(path) as f:
            text = f.read()
        cid = re.search(r'set\(CMAKE_CXX_COMPILER_ID "([^"]*)"\)', text)
        ver = re.search(r'set\(CMAKE_CXX_COMPILER_VERSION "([^"]*)"\)', text)
        if cid and ver:
            info["compiler"] = f"{cid.group(1)} {ver.group(1)}"
    return info


def machine():
    info = {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
            "cpus": os.cpu_count()}
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                names = [l.split(":", 1)[1].strip() for l in f if l.startswith("model name")]
            if names:
                info["cpu"] = names[0]
        except OSError:
            pass
    elif platform.system() == "Windows":
        info["cpu"] = platform.processor()
    if platform.system() == "Darwin":
        try:
            info["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                         text=True).stdout.strip()
        except OSError:
            pass
    return info


def solve(cli, path, args, warm=True, history=None):
    cmd = [cli, "solve", path, "--threads", str(args.threads), "--backend", args.backend, "--linear-solver",
           args.linear_solver]
    if warm:
        cmd.append("--warm-up")
    if history:
        cmd += ["--history", history]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{path}: {out.stderr.strip() or out.stdout.strip()}")
    r = json.loads(out.stdout)
    if history:
        with open(history) as f:
            r["history"] = json.load(f)
    return r


def ceres_solve(tool, library, path):
    out = subprocess.run([tool, path, "--linear-solver", "sparse_schur", "--sparse-library", library, "--gauge", "auto"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"ceres {path}: {out.stderr.strip() or out.stdout.strip()}")
    return json.loads(out.stdout)


def time_to(costs, times, target):
    """First time at which the cost is at or below target (within EQUAL_COST)."""
    for c, t in zip(costs, times):
        if c <= target * (1 + EQUAL_COST):
            return t
    return None


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
    synthetic = SYNTHETIC + (LARGE if args.large or args.huge else []) + (HUGE if args.huge else [])
    if args.only:
        keep = lambda n: any(o in n for o in args.only.split(","))
        problems = [(n, p) for n, p in problems if keep(n)]
        synthetic = [(n, o) for n, o in synthetic if keep(n)]
    if args.skip:
        drop = lambda n: any(o in n for o in args.skip.split(","))
        problems = [(n, p) for n, p in problems if not drop(n)]
        synthetic = [(n, o) for n, o in synthetic if not drop(n)]
    with tempfile.TemporaryDirectory(prefix="jba_bench_") as tmp:
        for name, opts in synthetic:
            p = os.path.join(tmp, name + ".jba")
            subprocess.run([make, p, *opts.split()], check=True, capture_output=True)
            problems.append((name, p))
        return run_problems(args, cli, problems, tmp)


def run_problems(args, cli, problems, tmp):
    result = {"commit": git_commit(), "commit_full": git_commit(short=False),
              "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
              "build": build_info(args.build), "machine": machine(), "backend": args.backend,
              "linear_solver": args.linear_solver, "threads": args.threads, "repeats": args.repeats,
              "timing": "whole call, set-up included; warm (best of repeats) and cold (a fresh process)",
              "problems": {}}
    ceres_tools = []
    if args.ceres:
        ceres_tools.append(("ceres", args.ceres, args.ceres_sparse_library))
    if args.ceres_gpu:
        ceres_tools.append(("ceres_gpu", args.ceres_gpu, "cuda"))
    for name, path in problems:
        cold = solve(cli, path, args, warm=False)
        hist = os.path.join(tmp, name + ".history.json")
        runs = [solve(cli, path, args, history=hist + str(r)) for r in range(args.repeats)]
        first = runs[0]
        # The solver auto stood for, as the library resolved it for this problem.
        result["linear_solver"] = first["linear_solver"]
        for r in runs[1:] + [cold]:
            if any(r[k] != first[k] for k in COUNTS) or r["final_cost"] != first["final_cost"]:
                raise RuntimeError(f"{name}: repeated runs differ; the solver should be deterministic (D59)")
        best = min(runs, key=lambda r: r["time_seconds"])
        entry = {k: first[k] for k in ("images", "points", "observations", "relative_priors", "global_priors",
                                      "initial_cost", "final_cost", "linear_solver", *COUNTS)}
        entry["time_seconds"] = best["time_seconds"]
        entry["setup_seconds"] = best["setup_seconds"]
        entry["cold_time_seconds"] = cold["time_seconds"]
        entry["time_per_iteration"] = best["time_seconds"] / max(1, first["iterations"])
        line = (f"{name:30s} {first['iterations']:3d} it {first['linear_iterations']:5d} lin  "
                f"{best['time_seconds']:7.3f} s (set-up {best['setup_seconds']:6.3f}, cold {cold['time_seconds']:7.3f})"
                f"  {first['termination']}")
        ours_costs = [h["cost"] for h in best["history"]]
        ours_times = [h["time_seconds"] for h in best["history"]]
        for column, tool, library in ceres_tools:
            cruns = [ceres_solve(tool, library, path) for _ in range(args.repeats)]
            cb = min(cruns, key=lambda r: r["wall_seconds"])
            c = {"sparse_library": library, "version": cb["ceres_version"], "iterations": cb["iterations"],
                 "final_cost": cb["final_cost"], "termination": cb["termination"],
                 "time_seconds": cb["wall_seconds"], "solve_seconds": cb["time_seconds"]}
            c["speedup"] = c["time_seconds"] / best["time_seconds"]
            # At equal accuracy (D99): time to the worse of the two final costs.
            target = max(first["final_cost"], c["final_cost"])
            t_ours = time_to(ours_costs, ours_times, target)
            t_ceres = time_to(cb["history_cost"], cb["history_time"], target)
            if t_ours and t_ceres:
                c["equal_cost_target"] = target
                c["equal_cost_seconds"] = t_ceres
                c["equal_cost_seconds_ours"] = t_ours
                c["equal_cost_speedup"] = t_ceres / t_ours
            entry[column] = c
            line += (f"  | {column} {c['iterations']:3d} it {c['time_seconds']:7.3f} s  x{c['speedup']:5.2f}"
                     f" (equal cost x{c.get('equal_cost_speedup', float('nan')):5.2f})")
        result["problems"][name] = entry
        print(line, file=sys.stderr)
    return result


def compare(result, baseline, max_slowdown, only=None, cost_rtol=1e-9, count_slack=0.0, time_same_cpu=False):
    problems = []
    check_time = True
    if time_same_cpu:
        b_cpu, r_cpu = baseline.get("machine", {}).get("cpu"), result["machine"].get("cpu")
        check_time = b_cpu is not None and b_cpu == r_cpu
        if not check_time:
            print(f"times not compared: this CPU ({r_cpu}) is not the baseline's ({b_cpu})", file=sys.stderr)
    if "timing" not in baseline:
        return ["baseline was recorded before times included the set-up (D101); record it again"]
    for key, default in (("backend", "cpu"), ("linear_solver", "iterative_schur")):
        if baseline.get(key, default) != result[key]:
            return [f"baseline is for {key} {baseline.get(key, default)}, this run is {result[key]}"]
    for name, b in baseline["problems"].items():
        r = result["problems"].get(name)
        if r is None:
            if not only:
                problems.append(f"{name}: not run")
            continue
        for k in COUNTS:
            if isinstance(b[k], (int, float)) and count_slack > 0:
                if abs(r[k] - b[k]) > count_slack * max(abs(b[k]), 1):
                    problems.append(f"{name}: {k} {b[k]} -> {r[k]} (limit {100 * count_slack:.0f}%)")
            elif r[k] != b[k]:
                problems.append(f"{name}: {k} {b[k]} -> {r[k]}")
        rel = abs(r["final_cost"] - b["final_cost"]) / max(abs(b["final_cost"]), 1e-300)
        if rel > cost_rtol:
            problems.append(f"{name}: final cost {b['final_cost']:.17g} -> {r['final_cost']:.17g} ({rel:.1e} relative)")
        ratio = r["time_per_iteration"] / b["time_per_iteration"]
        mark = "  SLOWER" if check_time and ratio > 1 + max_slowdown else ""
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
    ap.add_argument("--huge", action="store_true", help="add LARGE and the 10,000-image walk (HUGE)")
    ap.add_argument("--ceres", help="jba_ceres_solve to run on every problem too (speed reference)")
    ap.add_argument("--ceres-sparse-library", default="accelerate",
                    choices=["eigen", "accelerate", "suite_sparse", "cuda"])
    ap.add_argument("--ceres-gpu", help="a second jba_ceres_solve, built with cuDSS (column ceres_gpu)")
    ap.add_argument("--only", help="comma-separated substrings of problem names")
    ap.add_argument("--skip", help="comma-separated substrings of problem names to leave out")
    ap.add_argument("--cost-rtol", type=float, default=1e-9, help="allowed relative change of the final cost")
    ap.add_argument("--count-slack", type=float, default=0.0,
                    help="allowed relative change of the counts (0: identical); termination must match")
    ap.add_argument("--store", metavar="MACHINE", help="also write the run under bench/results/MACHINE/")
    ap.add_argument("--time-same-cpu", action="store_true",
                    help="compare times only when the CPU model is the baseline's (hosted runners vary)")
    args = ap.parse_args()
    result = run(args)
    outs = [args.out] if args.out else []
    if args.store:
        stamp = result["date"][:10]
        outs.append(os.path.join(ROOT, "bench", "results", args.store,
                                 f"{stamp}-{result['commit']}-{args.backend}-{result['linear_solver']}.json"))
    for out in outs:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=1)
            f.write("\n")
        print(f"wrote {out}", file=sys.stderr)
    if args.compare:
        with open(args.compare) as f:
            baseline = json.load(f)
        problems = compare(result, baseline, args.max_slowdown, args.only or args.skip, args.cost_rtol,
                           args.count_slack, args.time_same_cpu)
        for p in problems:
            print("REGRESSION " + p, file=sys.stderr)
        sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
