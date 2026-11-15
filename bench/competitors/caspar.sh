#!/bin/bash
# One problem through COLMAP's Caspar backend and through just-ba (milestone 5 speed
# comparison; bench/competitors/README.md).
#
#   COLMAP=<colmap binary> BUILD=<just-ba build> bench/competitors/caspar.sh <name> <problem> <refine> [--simple-radial]
#
# <problem> is a BAL file or a problem file; <refine> 1 lets Caspar refine the focal
# length and distortion (BAL), 0 keeps the intrinsics fixed (the synthetic walks).
# Writes into the current directory: <name>.jba and <name>-model (the same problem,
# jba_to_colmap), three Caspar runs (<name>.caspar<r>.log with glog's timestamps,
# .out with its per-iteration lines), the cost of Caspar's result in double
# (<name>.caspar.cost: observations, cost), and three just-ba solves with histories.
set -e
name=$1; in=$2; ref=$3; shift 3
"$BUILD/tests/jba_to_colmap" "$in" "$name.jba" "$name-model" "$@" > "$name.convert.json"
for r in 1 2 3; do
  rm -rf "$name-out"; mkdir -p "$name-out"
  "$COLMAP" bundle_adjuster --log_target stderr --log_level 2 --input_path "$name-model" --output_path "$name-out" \
    --BundleAdjustment.backend CASPAR --BundleAdjustment.refine_focal_length "$ref" \
    --BundleAdjustment.refine_extra_params "$ref" --BundleAdjustment.refine_principal_point 0 \
    > "$name.caspar$r.out" 2> "$name.caspar$r.log"
done
rm -rf "$name-out-txt"; mkdir -p "$name-out-txt"
"$COLMAP" model_converter --input_path "$name-out" --output_path "$name-out-txt" --output_type TXT > /dev/null 2>&1
python3 -I "$(dirname "$0")/colmap_cost.py" "$name-out-txt" > "$name.caspar.cost"
for r in 1 2 3; do
  "$BUILD/cli/just-ba" solve "$name.jba" --backend cuda --history "$name.jba$r.hist.json" > "$name.jba$r.json"
done
