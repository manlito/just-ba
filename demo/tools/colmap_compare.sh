#!/usr/bin/env bash
# The comparison with COLMAP (document 05, level 3; D136) for one scene, run by hand on a
# machine that has COLMAP. COLMAP is a reference here, not a dependency: install it
# anywhere disposable (a temporary prefix, a conda environment) and remove it afterwards.
#
#   demo/tools/colmap_compare.sh <images_dir> <work_dir> [jba_demo options...]
#
# COLMAP extracts features (one SIMPLE_RADIAL camera for the scene), matches every pair
# and reconstructs; its model is converted to text; colmap_capture.py writes the capture
# file with COLMAP's verified matches and model; jba_demo runs on it and prints both
# sides. Uses only COLMAP options that have been stable for years. Environment:
#   COLMAP    the colmap executable (default: colmap on the path)
#   JBA_DEMO  the jba_demo executable (default: build/demo/demo/jba_demo)
#   MATCHER   exhaustive_matcher (default) or sequential_matcher, for long sequences
set -euo pipefail

if [ $# -lt 2 ]; then
  sed -n '2,15p' "$0"
  exit 2
fi
images=$1
work=$2
shift 2
colmap=${COLMAP:-colmap}
demo=${JBA_DEMO:-build/demo/demo/jba_demo}
matcher=${MATCHER:-exhaustive_matcher}
here=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$work/sparse"
db="$work/database.db"
log="$work/colmap.log"
if [ ! -f "$work/sparse/0/cameras.bin" ] && [ ! -f "$work/sparse/0/cameras.txt" ]; then
  rm -f "$db"
  echo "COLMAP $("$colmap" help 2>&1 | head -1)" | tee "$log"
  start=$(date +%s)
  "$colmap" feature_extractor --database_path "$db" --image_path "$images" \
    --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_RADIAL >>"$log" 2>&1
  "$colmap" "$matcher" --database_path "$db" >>"$log" 2>&1
  matched=$(date +%s)
  "$colmap" mapper --database_path "$db" --image_path "$images" --output_path "$work/sparse" >>"$log" 2>&1
  done_at=$(date +%s)
  echo "features and matching: $((matched - start)) s; mapper: $((done_at - matched)) s" | tee -a "$log"
fi
models=$(find "$work/sparse" -mindepth 1 -maxdepth 1 -type d ! -name '*-txt' | sort)
[ -n "$models" ] || { echo "COLMAP made no model; see $log"; exit 1; }
[ "$(echo "$models" | wc -l)" -eq 1 ] || echo "note: COLMAP made several models; comparing with sparse/0 only"
mkdir -p "$work/sparse/0-txt"
"$colmap" model_converter --input_path "$work/sparse/0" --output_path "$work/sparse/0-txt" --output_type TXT >>"$log" 2>&1

python3 "$here/colmap_capture.py" "$db" --model "$work/sparse/0-txt" -o "$work/scene.jbc"
"$demo" run "$work/scene.jbc" --export "$work/ours" "$@" | tee "$work/compare.txt"
