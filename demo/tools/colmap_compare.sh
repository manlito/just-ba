#!/usr/bin/env bash
# The comparison with COLMAP (document 05, level 3; D138) for one scene, run by hand on a
# machine that has COLMAP. COLMAP is a reference here, not a dependency: install it
# anywhere disposable (a temporary prefix, a conda environment) and remove it afterwards.
#
#   demo/tools/colmap_compare.sh <images_dir> <work_dir> [jba_demo options...]
#
# COLMAP extracts features (SIMPLE_RADIAL cameras, one for the scene by default), matches every pair
# and reconstructs; its model is converted to text; colmap_capture.py writes the capture
# file with COLMAP's verified matches and model; jba_demo runs on it and prints both
# sides. Option names are COLMAP 4.2.1's (FeatureExtraction, FeatureMatching). Environment:
#   COLMAP    the colmap executable (default: colmap on the path)
#   JBA_DEMO  the jba_demo executable (default: build/demo/demo/jba_demo)
#   MATCHER   exhaustive_matcher (default) or sequential_matcher, for long sequences
#   USE_GPU   1 for COLMAP's GPU SIFT and matching (a CUDA build, or a display for
#             OpenGL); default 0, which also works headless over SSH
#   THREADS   feature extraction threads (default 4: CPU SIFT takes about 2 GB per
#             thread on large images, and every thread ran a 15 GB laptop out of memory)
#   MAX_IMAGE_SIZE  longest side SIFT sees, in pixels (default 3200; keypoints are
#             scaled back to the full image)
#   SINGLE_CAMERA  1 (default): one camera for the scene; 0: one per image, for images
#             of different sizes or cameras
#   TRUTH     a ground-truth model in COLMAP's text format (ETH3D's
#             dslr_calibration_undistorted), stored in the capture file and compared with
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
use_gpu=${USE_GPU:-0}
threads=${THREADS:-4}
max_image_size=${MAX_IMAGE_SIZE:-3200}
single_camera=${SINGLE_CAMERA:-1}
here=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$work/sparse"
db="$work/database.db"
log="$work/colmap.log"
if [ ! -f "$work/sparse/0/cameras.bin" ] && [ ! -f "$work/sparse/0/cameras.txt" ]; then
  rm -f "$db"
  echo "COLMAP $("$colmap" help 2>&1 | head -1)" | tee "$log"
  start=$(date +%s)
  "$colmap" feature_extractor --database_path "$db" --image_path "$images" \
    --ImageReader.single_camera "$single_camera" --ImageReader.camera_model SIMPLE_RADIAL \
    --FeatureExtraction.use_gpu "$use_gpu" --FeatureExtraction.num_threads "$threads" \
    --FeatureExtraction.max_image_size "$max_image_size" >>"$log" 2>&1
  "$colmap" "$matcher" --database_path "$db" --FeatureMatching.use_gpu "$use_gpu" >>"$log" 2>&1
  matched=$(date +%s)
  "$colmap" mapper --database_path "$db" --image_path "$images" --output_path "$work/sparse" >>"$log" 2>&1
  done_at=$(date +%s)
  echo "features and matching: $((matched - start)) s; mapper: $((done_at - matched)) s" | tee -a "$log"
fi
# COLMAP may make several models; the reference is the one with the most images (not
# always sparse/0: on ETH3D's office, sparse/0 held 7 images and sparse/1 all 26).
best="" best_n=-1
for m in $(find "$work/sparse" -mindepth 1 -maxdepth 1 -type d ! -name '*-txt' | sort); do
  n=$("$colmap" model_analyzer --path "$m" 2>&1 | grep -i -E "registered (images|frames)" | head -1 | grep -o -E "[0-9]+$" || echo 0)
  echo "model $(basename "$m"): ${n:-0} images" | tee -a "$log"
  if [ "${n:-0}" -gt "$best_n" ]; then best=$m best_n=${n:-0}; fi
done
[ -n "$best" ] || { echo "COLMAP made no model; see $log"; exit 1; }
mkdir -p "$work/sparse/best-txt"
"$colmap" model_converter --input_path "$best" --output_path "$work/sparse/best-txt" --output_type TXT >>"$log" 2>&1

truth_args=()
[ -n "${TRUTH:-}" ] && truth_args=(--truth "$TRUTH")
python3 "$here/colmap_capture.py" "$db" --model "$work/sparse/best-txt" "${truth_args[@]}" -o "$work/scene.jbc"
"$demo" run "$work/scene.jbc" --export "$work/ours" "$@" | tee "$work/compare.txt"
