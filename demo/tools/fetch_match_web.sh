#!/usr/bin/env bash
# Fetch the live demo's keypoints and matcher (D156): just-keypoints' browser module with
# ALIKED's weights, and just-matching's with LightGlue's (Apache-2.0, float16), from their
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
  "jkp https://just-keypoints.pages.dev/lib jkp_wasm.wasm 3fc11898ad410815e8a399c7e110bc9cb218362088c91bba6f84ec8a12711d20"
  "jkp https://just-keypoints.pages.dev/lib aliked-n16.jkpw 8002a58d4f374335b58941dd3eac210daa2d19270105932c60fbf6ae2d33f3a8"
  "jmt https://just-matching.pages.dev/lib jmt.js a49941abea10e1a4a396057d4791942f5eab1ebc5734dab545122c84cf7cb914"
  "jmt https://just-matching.pages.dev/lib jmt_wasm.js ef1c61658f8a2004fe5573cda1cc5b9534c986084ff4c4b899d895f5da5691a8"
  "jmt https://just-matching.pages.dev/lib jmt_wasm.wasm ec25dcc8bd3e8ee07944ee01613531f74c44574041d58df4e9e454de16311a78"
  "jmt https://just-matching.pages.dev/lib lightglue-aliked-fp16.jmtw 3541ba0088b4578f7009b97c54468c0712d6b98dd39fa50c52d7f33bfc5b5b97"
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
