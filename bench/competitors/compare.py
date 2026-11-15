#!/usr/bin/env python3
"""Reads the raw outputs of the milestone 5 comparison with Caspar and bae
(bench/competitors/README.md) and writes what was measured as one JSON file, so the
figures in document 07 can be checked and redone (review of milestone 5, finding 6).

  python3 bench/competitors/compare.py <results dir> <out.json>

<results dir> holds caspar/ (caspar.sh's outputs, per problem) and bae/ (bae_bal.py's
output and just-ba's solves, per BAL problem). Per problem and competitor it records
the competitor's time and final cost, its cost history, just-ba's cold and best warm
call (time, set-up, final cost, cost history with times since the call), and:

  - end_to_end: the competitor's time over just-ba's, cold and warm;
  - equal_cost: the time each needs to reach the worse of the two final costs
    (relative 1e-6), from the histories (D99), and their ratio, cold and warm.

just-ba's times are of the whole call, set-up included (D101). Caspar's time runs from
"Creating Caspar bundle adjuster" to its termination line ("Caspar: MAX_ITERATIONS" or another status; glog timestamps), its
history from its per-iteration lines (dt_tot, from its own start, so the equal-cost
figure credits Caspar as if it had stopped when it got there); its final cost is
recomputed in double (colmap_cost.py). bae's time runs from its first step, which
includes that step's set-up; its history from its per-step lines. Standard library
only.
"""

import datetime
import glob
import json
import os
import re
import sys

EQUAL_COST = 1e-6


def time_to(history, target):
    for cost, t in history:
        if cost <= target * (1 + EQUAL_COST):
            return t
    return None


def jba_runs(d, name):
    """just-ba's cold call and best warm call on one problem, with histories."""
    def load(prefix):
        with open(os.path.join(d, prefix + ".json")) as f:
            r = json.load(f)
        with open(os.path.join(d, prefix + ".hist.json")) as f:
            h = json.load(f)
        return {"time_seconds": r["time_seconds"], "setup_seconds": r["setup_seconds"],
                "final_cost": r["final_cost"], "iterations": r["iterations"], "linear_solver": r["linear_solver"],
                "history": [[x["cost"], x["time_seconds"]] for x in h]}
    cold = load(f"{name}.jba-cold")
    warm = [load(f"{name}.jba{r}") for r in (1, 2, 3) if os.path.isfile(os.path.join(d, f"{name}.jba{r}.json"))]
    return cold, min(warm, key=lambda r: r["time_seconds"])


def ratios(comp, cold, warm):
    out = {"end_to_end_cold": comp["time_seconds"] / cold["time_seconds"],
           "end_to_end_warm": comp["time_seconds"] / warm["time_seconds"]}
    for label, ours in (("cold", cold), ("warm", warm)):
        target = max(comp["final_cost"], ours["final_cost"])
        tc, to = time_to(comp["history"], target), time_to(ours["history"], target)
        out[f"equal_cost_target_{label}"] = target
        out[f"equal_cost_seconds_{label}"] = [tc, to]
        out[f"equal_cost_{label}"] = tc / to if tc and to else None
    return out


def glog_time(line):
    m = re.match(r"[IWE](\d{8}) (\d\d):(\d\d):(\d\d)\.(\d+)", line)
    if not m:
        return None
    day = datetime.datetime.strptime(m.group(1), "%Y%m%d")
    return day.timestamp() + int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4)) + float("0." + m.group(5))


def caspar(d):
    out = {}
    for conv in sorted(glob.glob(os.path.join(d, "*.convert.json"))):
        name = os.path.basename(conv)[: -len(".convert.json")]
        runs = []
        for r in (1, 2, 3):
            log, it = os.path.join(d, f"{name}.caspar{r}.log"), os.path.join(d, f"{name}.caspar{r}.out")
            if not os.path.isfile(log):
                continue
            t0 = t1 = None
            for line in open(log, errors="replace"):
                if "Creating Caspar bundle adjuster" in line and t0 is None:
                    t0 = glog_time(line)
                if re.search(r"\] Caspar: [A-Z_]+", line):  # its termination line
                    t1 = glog_time(line)
            hist = []
            for line in open(it, errors="replace"):
                sb = re.search(r"score_best[=: ]+([0-9.eE+-]+)", line)
                dt = re.search(r"dt_tot[=: ]+([0-9.eE+-]+)", line)
                if sb and dt:
                    hist.append([float(sb.group(1)), float(dt.group(1))])
            runs.append({"time_seconds": t1 - t0 if t0 and t1 else None, "history": hist})
        cost_file = os.path.join(d, f"{name}.caspar.cost")
        with open(cost_file) as f:
            final = float(f.read().split()[-1])
        timed = [r for r in runs if r["time_seconds"]]
        best = min(timed, key=lambda r: r["time_seconds"])
        # Caspar's history is its own single-precision score; its last entry is
        # replaced by the cost recomputed in double, against which just-ba is compared.
        hist = best["history"]
        if hist:
            hist[-1][0] = final
        comp = {"time_seconds": best["time_seconds"], "times_all_runs": [r["time_seconds"] for r in runs],
                "final_cost": final, "history": hist}
        cold, warm = jba_runs(d, name)
        with open(conv) as f:
            shape = json.load(f)
        out[name] = {"problem": shape, "caspar": comp, "jba_cold": cold, "jba_warm": warm,
                     **ratios(comp, cold, warm)}
    return out


def bae(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.bae.out"))):
        name = os.path.basename(f)[: -len(".bae.out")]
        hist, total, final, init = [], None, None, None
        for line in open(f, errors="replace"):
            m = re.match(r"initial loss .* cost ([0-9.eE+-]+)", line)
            if m:
                init = float(m.group(1))
            m = re.match(r"BAE iter \d+ loss \S+ cost (\S+) time (\S+)", line)
            if m:
                hist.append([float(m.group(1)), float(m.group(2))])
            m = re.match(r"BAE final: .* cost (\S+) time_seconds (\S+)", line)
            if m:
                final, total = float(m.group(1)), float(m.group(2))
        comp = {"time_seconds": total, "final_cost": final, "history": [[init, 0.0]] + hist}
        cold, warm = jba_runs(d, name)
        out[name] = {"bae": comp, "jba_cold": cold, "jba_warm": warm, **ratios(comp, cold, warm)}
    return out


def main():
    root, dest = sys.argv[1], sys.argv[2]
    result = {"what": __doc__.split("\n\n")[0], "caspar": caspar(os.path.join(root, "caspar")),
              "bae": bae(os.path.join(root, "bae"))}
    with open(dest, "w") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    for kind in ("caspar", "bae"):
        for name, e in result[kind].items():
            print(f"{kind:6s} {name:14s} {kind} {e[kind]['time_seconds']:8.3f} s cost {e[kind]['final_cost']:.6g} | "
                  f"jba cold {e['jba_cold']['time_seconds']:7.3f} warm {e['jba_warm']['time_seconds']:7.3f} "
                  f"cost {e['jba_warm']['final_cost']:.6g} | end-to-end x{e['end_to_end_cold']:.2f} cold "
                  f"x{e['end_to_end_warm']:.2f} warm | equal cost x{e['equal_cost_cold'] or float('nan'):.2f} cold "
                  f"x{e['equal_cost_warm'] or float('nan'):.2f} warm")


if __name__ == "__main__":
    main()
