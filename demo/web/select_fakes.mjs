// A generated scene for the selector's tests (select.test.mjs, video_steps.test.mjs): a
// long textured strip the camera slides along, with a keypoint detector, a matcher and a
// pose check that know it. Not served with the pages.

export const W = 320, H = 240;

// Deterministic draws.
function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
}

/** Points on a strip `length` pixels long and H high, `per` of them in each W of it. */
export function strip(length, per = 220, seed = 1) {
  const r = rng(seed), pts = [];
  for (let k = 0, n = Math.round(per * length / W); k < n; k++) pts.push([r() * length, 4 + r() * (H - 8)]);
  return pts;
}

const pixels = (() => {
  const r = rng(9), d = new Uint8ClampedArray(4 * W * H);
  for (let i = 0; i < d.length; i++) d[i] = i % 4 === 3 ? 255 : Math.floor(256 * r());
  return d;
})();

/** The frame of a camera at `x` along the strip: an ImageData-like with where it is. */
export function frameAt(x, { blur = false } = {}) {
  return { width: W, height: H, data: blur ? new Uint8ClampedArray(4 * W * H).fill(128) : pixels, x };
}

/** just-keypoints' extractor: the strip's points in view, each one's index as its descriptor. */
export function extractor(pts) {
  return {
    async extract(image) {
      const xy = [], ids = [];
      pts.forEach((p, k) => {
        if (p[0] >= image.x && p[0] < image.x + W) { xy.push(p[0] - image.x, p[1]); ids.push(k); }
      });
      return { n: ids.length, xy: Float32Array.from(xy), desc: Int32Array.from(ids), score: new Float32Array(ids.length).fill(1) };
    },
  };
}

/** just-matching's matcher (LightGlue's shape): keypoints of the same point match. Counts its features alive. */
export function matcher() {
  const m = {
    alive: 0, matched: 0,
    async describeKeypoints(image, xy, desc) {
      m.alive++;
      return { ids: desc, destroy() { m.alive--; this.ids = null; } };
    },
    async match(fa, xyA, fb) {
      m.matched++;
      const at = new Map();
      fa.ids.forEach((id, i) => at.set(id, i));
      const out = [];
      fb.ids.forEach((id, j) => { if (at.has(id)) out.push(at.get(id), j); });
      return { n: out.length / 2, matches: Int32Array.from(out), scores: new Float32Array(out.length / 2).fill(1) };
    },
  };
  return m;
}

/** demo_pair_inliers: every match agrees (the scene has no wrong ones). */
export function verify(refXy, curXy, matches) {
  const seen = [];
  if (matches.length / 2 < 15) return seen;
  for (let k = 0; k < matches.length; k += 2) seen.push([matches[k], matches[k + 1]]);
  return seen;
}
