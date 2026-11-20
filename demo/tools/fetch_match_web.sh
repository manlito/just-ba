#!/usr/bin/env bash
# Fetch the live demo's keypoints and matcher (D156): just-keypoints' browser module with
# ALIKED's and XFeat's weights, and just-matching's with LightGlue's (Apache-2.0, float16) and
# LighterGlue's (XFeat's LightGlue, Apache-2.0), from their
# documentation sites into third_party/match-web (not committed), checked against pinned
# hashes:
#
#   demo/tools/fetch_match_web.sh
#
# The files are what those repositories' main branches build with Emscripten. When one of
# them deploys a new module the check fails: run the script with MATCH_WEB_UPDATE=1 to
# print the new hashes, check the demo with them, then paste them below. The demo's
# CMakeLists.txt copies them next to the page (JBA_DEMO_MATCH_WEB_DIR).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$ROOT/third_party/match-web"
FILES=(
  "jkp https://just-keypoints.pages.dev/lib jkp.js d459bd7cbb03187b0662a289e8235ce4b70d83969b93b3a750f71e89972d0262"
  "jkp https://just-keypoints.pages.dev/lib jkp_wasm.js 54d4ddebaa5187427297c2c1ef401ca8ba4016367ad763f96a4bc58b7e36ae9a"
  "jkp https://just-keypoints.pages.dev/lib jkp_wasm.wasm 6f11dd0839b57488d3f63e48e07e8b1fa3cdeaca7381e9302800ee67389d5919"
  "jkp https://just-keypoints.pages.dev/lib aliked-n16.jkpw 8002a58d4f374335b58941dd3eac210daa2d19270105932c60fbf6ae2d33f3a8"
  "jmt https://just-matching.pages.dev/lib jmt.js 481d6a5c383bd22f02f5f4701f6a1899e3f974bcd5a4d5882399d65c382eed14"
  "jmt https://just-matching.pages.dev/lib jmt_wasm.js 18718e16d4af9b0966e9605404286bef30d9356fcce175a607bcfb9ac3ad2015"
  "jmt https://just-matching.pages.dev/lib jmt_wasm.wasm 69f1da090b7eab2b0fb435d9376649ea266ee577902bc3d1804caa1c2e7e6082"
  "jmt https://just-matching.pages.dev/lib lightglue-aliked-fp16.jmtw 3541ba0088b4578f7009b97c54468c0712d6b98dd39fa50c52d7f33bfc5b5b97"
  "jkp https://just-keypoints.pages.dev/lib xfeat.jkpw c67999d3f9cc647d9ad03258b5df758cb4269ece10270f14168249d06ced3434"
  "jmt https://just-matching.pages.dev/lib lightglue-xfeat.jmtw 61df510fb7b4e00878075f7e6e1486351112820b9002acec92754cc55c4e7a7c"
)

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

fail=0
for entry in "${FILES[@]}"; do
  read -r dir base name want <<<"$entry"
  mkdir -p "$DEST/$dir"
  out="$DEST/$dir/$name"
  if [ -f "$out" ] && [ "$(sha256 "$out")" = "$want" ]; then continue; fi
  curl -fsSL -o "$out.part" "$base/$name"
  got="$(sha256 "$out.part")"
  if [ "$got" != "$want" ]; then
    if [ "${MATCH_WEB_UPDATE:-0}" = 1 ]; then
      echo "  \"$dir $base $name $got\""
      mv "$out.part" "$out"
    else
      echo "$dir/$name: sha256 $got, expected $want (a new module deployed? see the top of $0)" >&2
      rm -f "$out.part"
      fail=1
    fi
    continue
  fi
  mv "$out.part" "$out"
done
[ "$fail" = 0 ] || exit 1
echo "keypoints and matcher modules: $DEST"
