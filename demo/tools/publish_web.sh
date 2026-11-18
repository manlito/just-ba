#!/usr/bin/env bash
# Builds the live demo as WebAssembly and uploads it to the GitHub release that the
# docs workflow takes it from (D148). Run by hand on a machine with Emscripten and
# poselib-gpu's sources: a hosted job cannot check poselib-gpu out (D134).
#
#   demo/tools/publish_web.sh            # build, record the commits, upload
#   demo/tools/publish_web.sh --dry-run  # build and record; print the gh commands only
#
# Then `gh workflow run docs -f deploy=true` publishes the site with this build.
#
# The build is configured on first use (Release, WebGPU) and rebuilt every run, so
# what is uploaded matches the checkout. Next to the three files the site serves, the
# upload carries demo-web.json: the commits of just-ba and poselib-gpu it was built
# from. The docs workflow prints it and does not publish it. Environment:
#   BUILD_DIR             the WebAssembly build (default build/demo-wasm)
#   JBA_POSELIB_GPU_DIR   poselib-gpu's sources, for a new build (default: CMake's,
#                         ../poselib-gpu next to the repository)
#   JBA_EMDAWNWEBGPU_DIR  emdawnwebgpu_pkg, for a new build (default: poselib-gpu's
#                         third_party/webgpu/emdawnwebgpu/emdawnwebgpu_pkg)
#   RELEASE_TAG           the release (default demo-web)
#   ALLOW_DIRTY           1 to upload from checkouts with uncommitted changes
#   EMSDK_PYTHON          Python 3.10 or later for Emscripten (default: the first found)
set -euo pipefail

dry_run=0
case "${1:-}" in
  "") ;;
  --dry-run) dry_run=1 ;;
  *) echo "usage: $0 [--dry-run]" >&2; exit 2 ;;
esac

root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$root"
build=${BUILD_DIR:-build/demo-wasm}
tag=${RELEASE_TAG:-demo-web}

# Emscripten needs Python 3.10 or later; macOS's first python3 is often Xcode's 3.9.
if [[ -z ${EMSDK_PYTHON:-} ]]; then
  for py in python3 python3.15 python3.14 python3.13 python3.12 python3.11 python3.10 /opt/homebrew/bin/python3; do
    if command -v "$py" >/dev/null && "$py" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
      EMSDK_PYTHON=$(command -v "$py"); export EMSDK_PYTHON; break
    fi
  done
fi

if [[ ! -f "$build/CMakeCache.txt" ]]; then
  pose_dir=${JBA_POSELIB_GPU_DIR:-$root/../poselib-gpu}
  dawn_dir=${JBA_EMDAWNWEBGPU_DIR:-$pose_dir/third_party/webgpu/emdawnwebgpu/emdawnwebgpu_pkg}
  emcmake cmake -S . -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DJBA_BUILD_DEMO=ON \
    -DJBA_WITH_WEBGPU=ON -DJBA_POSELIB_GPU_DIR="$pose_dir" -DJBA_EMDAWNWEBGPU_DIR="$dawn_dir"
fi
cmake --build "$build" --target jba_demo_wasm

# The poselib-gpu the build actually used, from its cache.
pose_dir=$(sed -n 's/^JBA_POSELIB_GPU_DIR:PATH=//p' "$build/CMakeCache.txt")
web="$build/demo/web"
files=("$web/index.html" "$web/jba_demo.js" "$web/jba_demo.wasm")
for f in "${files[@]}"; do
  [[ -f "$f" ]] || { echo "missing $f after the build" >&2; exit 1; }
done

dirty() { [[ -n "$(git -C "$1" status --porcelain --untracked-files=no)" ]] && echo true || echo false; }
jba_commit=$(git rev-parse HEAD)
jba_dirty=$(dirty .)
pose_commit=$(git -C "$pose_dir" rev-parse HEAD)
pose_dirty=$(dirty "$pose_dir")
if [[ ($jba_dirty == true || $pose_dirty == true) && ${ALLOW_DIRTY:-0} != 1 ]]; then
  echo "uncommitted changes (just-ba: $jba_dirty, poselib-gpu: $pose_dirty);" \
       "commit them, or set ALLOW_DIRTY=1" >&2
  exit 1
fi
emscripten=$(emcc --version | head -n 1)
built=$(date -u +%Y-%m-%dT%H:%M:%SZ)
cat > "$web/demo-web.json" <<EOF
{
  "just_ba": "$jba_commit",
  "just_ba_dirty": $jba_dirty,
  "poselib_gpu": "$pose_commit",
  "poselib_gpu_dirty": $pose_dirty,
  "emscripten": "$emscripten",
  "built": "$built"
}
EOF
files+=("$web/demo-web.json")
cat "$web/demo-web.json"
ls -l "${files[@]}"

notes="The live demo's WebAssembly build, which the docs workflow puts on the site (D148).
Uploaded by demo/tools/publish_web.sh; the tag only holds the files.

Built $built from just-ba $jba_commit and poselib-gpu $pose_commit."

run() {
  if [[ $dry_run == 1 ]]; then printf '+'; printf ' %q' "$@"; printf '\n'; else "$@"; fi
}
if ! gh release view "$tag" >/dev/null 2>&1; then
  run gh release create "$tag" --prerelease --latest=false --title "Live demo (WebAssembly)" --notes "$notes"
fi
run gh release upload "$tag" "${files[@]}" --clobber
run gh release edit "$tag" --notes "$notes"
