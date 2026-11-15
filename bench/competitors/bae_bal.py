"""bae's BAL example (ba_example.py, tag 0.2) on a local BAL file (milestone 5 speed
comparison; bench/competitors/README.md).

  BAE_SRC=<bae checkout> <venv>/bin/python bench/competitors/bae_bal.py <problem.txt> [--iters 20]

Same model and solver settings as ba_example.py: SE3 pose (quaternion) + f, k1, k2
per camera, LM with TrustRegion(up=2, down=0.5**4), PCG(tol=1e-4, maxiter=250),
reject=30, float64, 20 LM steps. Only the loader differs: read_bal_data on a local
file instead of get_problem, which downloads from the BAL website.
bae's loss is sum ||r||^2; Ceres's cost is half of it, printed as cost.
"""
import argparse
import os
import sys
from time import perf_counter

# Appended, so the installed bae (with compiled extensions) wins over the source tree.
sys.path.append(os.environ["BAE_SRC"])
import ba_example as ex  # noqa: E402  (project, Residual, least_square_error)
import torch  # noqa: E402
import pypose as pp  # noqa: E402
from datapipes.bal_loader import read_bal_data  # noqa: E402
from bae.optim import LM  # noqa: E402
from bae.utils.pysolvers import PCG  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("problem")
ap.add_argument("--iters", type=int, default=20)
a = ap.parse_args()

t_load = perf_counter()
d = read_bal_data(a.problem, use_quat=True)
d = {k: v.to("cuda") for k, v in d.items() if isinstance(v, torch.Tensor)}
inp = {"observes": d["points_2d"], "cidx": d["camera_index_of_observations"],
       "pidx": d["point_index_of_observations"]}
model = ex.Residual(d["camera_params"][:, :ex.NUM_CAMERA_PARAMS].clone(), d["points_3d"].clone()).to("cuda")
opt = LM(model, strategy=pp.optim.strategy.TrustRegion(up=2.0, down=0.5**4),
         solver=PCG(tol=1e-4, maxiter=250), reject=30)
torch.cuda.synchronize()
print(f"loaded {a.problem}: {d['camera_params'].shape[0]} cameras, {d['points_3d'].shape[0]} points, "
      f"{d['points_2d'].shape[0]} observations, {perf_counter() - t_load:.3f} s")
init = opt.model.loss(inp, None).item()
print(f"initial loss (sum r^2) {init!r}  cost {init / 2!r}")
start = perf_counter()
for i in range(a.iters):
    loss = opt.step(inp)
    torch.cuda.synchronize()
    print(f"BAE iter {i} loss {loss.item()!r} cost {loss.item() / 2!r} time {perf_counter() - start:.4f}")
torch.cuda.synchronize()
total = perf_counter() - start
final = opt.model.loss(inp, None).item()
print(f"BAE final: iterations {a.iters} loss {final!r} cost {final / 2!r} time_seconds {total:.4f}")
