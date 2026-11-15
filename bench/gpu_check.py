#!/usr/bin/env python3
"""The GPU check run by hand (documents 05 and 08, D40): one command that builds the
GPU backend of this machine, runs the GPU tests and the Python tests, runs the
benchmark for each linear solver against this machine's baselines, stores the
results, and prints one summary. Not a CI job: CI has no GPU (D39).

  python3 bench/gpu_check.py                                 # the Mac: Metal
  python3 bench/gpu_check.py --machine rtx2070maxq --cudss-dir <cuDSS>/lib/cmake/cudss
  python bench/gpu_check.py --machine rtx2070maxq-windows --cudss-dir ...   # Windows

Steps (each reported; the exit status is non-zero if any fails):
  1. configure and build build/gpu-<backend> (Release, the Python bindings when this
     interpreter has nanobind, CUDA for the GPU present unless --cuda-arch is given);
  2. ctest -L gpu, then the Python tests (label python, real captures excluded);
  3. bench/run.py for sparse_schur and iterative_schur (sparse_schur on CUDA only
     with cuDSS), each compared with bench/baseline/<machine>-<backend>-<sparse|
     iterative>.json when it exists, and stored under bench/results/<machine>/.
     A machine without a baseline gets its runs stored, and they can become it.
     The BAL problems are fetched first when missing.

--wheel <file> instead checks a built wheel (release checklist, document 08): a
fresh virtual environment, the wheel and pytest installed, the Python tests run
from outside the source tree, and the backend expected on this machine available.

Times are meaningful only on an otherwise idle machine. Standard library only.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLVERS = {"sparse": "sparse_schur", "iterative": "iterative_schur"}


def run(cmd, **kw):
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kw).returncode == 0


def default_backend():
    if platform.system() == "Darwin":
        return "metal"
    if shutil.which("nvcc") or os.environ.get("CUDA_PATH"):
        return "cuda"
    sys.exit("gpu_check.py: no GPU backend for this machine (macOS for Metal, nvcc on PATH for CUDA)")


def default_machine(backend):
    if platform.system() == "Darwin":
        brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                               text=True).stdout.strip()
        return "apple-" + brand.lower().replace("apple", "").replace(" ", "")  # "Apple M5 Pro" -> apple-m5pro
    sys.exit("gpu_check.py: name this machine with --machine (its baselines are "
             f"bench/baseline/<machine>-{backend}-*.json, for example rtx2070maxq)")


def has_nanobind(python):
    return subprocess.run([python, "-c", "import nanobind, numpy, pytest"], capture_output=True).returncode == 0


def build_and_test(args, steps):
    build = args.build_dir or os.path.join(ROOT, "build", f"gpu-{args.backend}")
    python = sys.executable
    cfg = ["cmake", "-S", ROOT, "-B", build, "-DCMAKE_BUILD_TYPE=Release", "-DJBA_BUILD_TESTS=ON",
           "-DJBA_BUILD_CLI=ON"]
    with_python = has_nanobind(python)
    if with_python:
        cfg += ["-DJBA_BUILD_PYTHON=ON", f"-DPython_EXECUTABLE={python}"]
    if args.backend == "metal":
        cfg.append("-DJBA_WITH_METAL=ON")
    else:
        cfg += ["-DJBA_WITH_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}"]
        if args.cudss_dir:
            cfg += ["-DJBA_WITH_CUDSS=ON", f"-Dcudss_DIR={args.cudss_dir}"]
    ok = run(cfg) and run(["cmake", "--build", build, "--config", "Release", "--parallel"])
    steps.append(("build " + os.path.relpath(build, ROOT), ok))
    if not ok:
        return None
    ctest = ["ctest", "--test-dir", build, "-C", "Release", "--output-on-failure"]
    steps.append((f"GPU tests (ctest -L gpu, {args.backend})", run(ctest + ["-L", "gpu"])))
    if with_python:
        steps.append(("Python tests", run(ctest + ["-L", "python", "-LE", "real"])))
    else:
        steps.append((f"Python tests (skipped: {python} lacks nanobind, numpy or pytest; "
                      "see bindings/python/requirements.txt)", True))
    return build


def bench(args, build, steps):
    bal = os.path.join(ROOT, "tests", "data", "bal")
    if not os.path.isdir(bal) or len(os.listdir(bal)) < 5:
        run([sys.executable, os.path.join(ROOT, "tests", "data", "fetch_bal.py")])
    solvers = ["sparse", "iterative"] if args.backend == "metal" or args.cudss_dir else ["iterative"]
    for short in solvers:
        cmd = [sys.executable, os.path.join(ROOT, "bench", "run.py"), "--build", build, "--backend", args.backend,
               "--linear-solver", SOLVERS[short], "--store", args.machine]
        baseline = os.path.join(ROOT, "bench", "baseline", f"{args.machine}-{args.backend}-{short}.json")
        label = f"benchmark {args.backend} {SOLVERS[short]}"
        if os.path.isfile(baseline):
            with open(baseline) as f:
                text = f.read()
            # Run what the baseline holds: the CUDA baselines include the walks.
            if "synthetic-walk-10000" in text:
                cmd.append("--huge")
            elif "synthetic-walk-" in text:
                cmd.append("--large")
            cmd += ["--compare", baseline]
            label += " against " + os.path.relpath(baseline, ROOT)
        else:
            label += " (no baseline for this machine: stored only)"
        steps.append((label, run(cmd)))


def check_wheel(args, steps):
    with tempfile.TemporaryDirectory(prefix="jba_wheel_") as tmp:
        venv = os.path.join(tmp, "venv")
        py = os.path.join(venv, "Scripts" if os.name == "nt" else "bin", "python")
        ok = run([sys.executable, "-m", "venv", venv]) and \
            run([py, "-m", "pip", "install", "-q", os.path.abspath(args.wheel), "pytest"])
        steps.append((f"install {os.path.basename(args.wheel)} in a fresh environment", ok))
        if not ok:
            return
        steps.append(("Python tests against the installed wheel",
                      run([py, "-m", "pytest", "-q", os.path.join(ROOT, "bindings", "python", "tests")], cwd=tmp)))
        probe = ("import just_ba as j; print(j.__version__, j.VARIANT, j.CUDA_VERSION or '-', "
                 "[b.name for b in j.compiled_backends()]); "
                 f"import sys; sys.exit(0 if j.backend_available(j.Backend.{args.backend.upper()}) else 1)")
        steps.append((f"{args.backend} available from the installed wheel", run([py, "-c", probe], cwd=tmp)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", choices=["metal", "cuda"], help="default: metal on macOS, cuda with nvcc")
    ap.add_argument("--machine", help="name of this machine's baselines and results (default on a Mac: "
                                      "apple-<chip>, e.g. apple-m5pro)")
    ap.add_argument("--build-dir", help="default build/gpu-<backend>")
    ap.add_argument("--cudss-dir", help="cuDSS's CMake directory (<cuDSS>/lib/cmake/cudss): CUDA's sparse_schur")
    ap.add_argument("--cuda-arch", default="native", help="CMAKE_CUDA_ARCHITECTURES (default native)")
    ap.add_argument("--no-bench", action="store_true", help="tests only")
    ap.add_argument("--wheel", help="check this built wheel instead of building from source")
    args = ap.parse_args()
    args.backend = args.backend or default_backend()

    steps = []
    if args.wheel:
        check_wheel(args, steps)
    else:
        args.machine = args.machine or default_machine(args.backend)
        build = build_and_test(args, steps)
        if build and not args.no_bench:
            bench(args, build, steps)

    print("\nGPU check summary")
    for label, ok in steps:
        print(f"  {'pass' if ok else 'FAIL'}  {label}")
    if not args.wheel and not args.no_bench:
        print(f"  results stored under bench/results/{args.machine}/ (commit them)")
    sys.exit(0 if all(ok for _, ok in steps) else 1)


if __name__ == "__main__":
    main()
