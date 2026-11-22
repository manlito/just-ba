// The live capture's keep rules (D157), ported from Like Recon's capture loop (its
// src-tauri/src/capture.rs, M6 and M11) so the browser keeps the frames the desktop app
// would: which frame of the camera's stream becomes a keyframe, from how much of the
// last keyframe it still sees (the overlap), how many verified matches link the two and
// how much of both frames they cover (the support), its confident keypoints and its
// sharpness against the session's sharp level. Pure functions and small state objects,
// tested by gates.test.mjs (node --test), the Rust tests' cases.
//
// Left out of the port: the online map's tracking of the live frame (landmarks, and a
// keep as soon as the map took the last keyframe), which needs PnP against the map on
// every frame.

/** Below this overlap the frame no longer sees the last keyframe well enough to chain from it. */
export const LOST_OVERLAP = 0.15;
/** Frames are held as candidates from this far above the target overlap. */
export const BAND = 0.08;
/** When the gates refuse every frame of the band, the best is kept this far under the target. */
export const RELAX = 0.1;
/** Redundancy: kept when the overlap with the keyframe before last falls to this. */
export const REDUNDANT_OVERLAP = 0.38;
/** Keeps on support or redundancy wait for the overlap to fall to this. */
export const EARLY_MAX_OVERLAP = 0.8;
/** Support under this multiple of its floor: kept. */
export const NEAR_FLOOR = 1.25;
/** Under this multiple of the floor, frames are held as candidates. */
export const BAND_FLOOR = 1.6;
/** The share of the pipeline's verified matches the live layer-0 check finds. */
export const LIVE_SHARE = 0.6;
/** The sharp level halves in this many frames unless a sharper frame lifts it. */
export const BLUR_HALF_LIFE = 90;
/** Blurry: under this fraction of the sharp level. */
export const BLUR_FRACTION = 0.4;
export const SPEED_WEIGHT = 0.3;
/** Overlap falling faster than this a frame: "slow down". */
export const SLOW_DOWN_DROP = 0.05;
export const RECENT_KEYFRAMES = 4;
export const SPARSE_EVERY = 5;
export const SPARSE_KEYFRAMES = 6;
/** Held keyframes tried a frame while lost; a Shoot now tries all. */
export const RELOCALIZE_PER_FRAME = 3;
/** A keypoint at least this sure counts towards a frame's quality. */
export const CONFIDENT_SCORE = 0.2;
/** CONFIDENT_SCORE for XFeat, whose scores run lower: as many of a frame's keypoints reach it (Like Recon). */
export const CONFIDENT_SCORE_XFEAT = 0.08;
/** Matches that agree with a relative pose within this many pixels count as seen. */
export const INLIER_PX = 4;
/** Fewer agreeing matches than this and the frames are taken not to overlap. */
export const MIN_INLIERS = 15;
/** A keyframe with fewer keypoints than this can't anchor the chain. */
export const MIN_REFERENCE_KEYPOINTS = 4 * MIN_INLIERS;

/** The capture settings: Like Recon's defaults. */
export function defaultSettings() {
  return { auto: true, targetOverlap: 0.55, minKeypoints: 800, rejectBlur: true, redundant: true };
}

/** The floor of the live support, from the pipeline's pair rules (50 matches, 5 % coverage). */
export function floorOf(minMatches = 50, minCoverage = 0.05) {
  return { inliers: Math.max(Math.round(minMatches * LIVE_SHARE), MIN_INLIERS), coverage: minCoverage };
}

const under = (floor, s, k) => s.inliers < k * floor.inliers || s.coverage < k * floor.coverage;

/** The frame's own gates, whatever it overlaps: 'too_few_keypoints', 'blurry' or null. */
export function ownGate(s, f) {
  return f.keypoints < s.minKeypoints ? 'too_few_keypoints' : s.rejectBlur && f.blurry ? 'blurry' : null;
}

/**
 * The rules. `f`: { shoot, hasMatcher, hasReference, keypoints (confident), blurry,
 * overlap (null without a reference), support: { inliers, coverage }, overlap2 (null) }.
 * Returns { gate, urge, acceptable }: gate one of 'kept', 'waiting', 'too_few_keypoints',
 * 'blurry', 'lost', 'paused', 'no_matcher'; urge { kind: 'no' | 'band' | 'due' |
 * 'overdue', why } with why one of 'first', 'shoot', 'target', 'support', 'redundancy',
 * 'lost'.
 */
export function decide(s, floor, f) {
  const v = (gate, urge, acceptable) => ({ gate, urge, acceptable });
  const no = { kind: 'no' };
  if (f.shoot) return v('kept', { kind: 'due', why: 'shoot' }, true);
  if (!s.auto) return v('paused', no, false);
  if (!f.hasMatcher) return v('no_matcher', no, false);
  const own = ownGate(s, f);
  if (!f.hasReference) return own ? v(own, no, false) : v('kept', { kind: 'due', why: 'first' }, true);
  const o = f.overlap ?? 0;
  if (o < LOST_OVERLAP) return v('lost', { kind: 'overdue', why: 'lost' }, false);
  const early = o <= EARLY_MAX_OVERLAP;
  const o2 = s.redundant ? f.overlap2 : null;
  const support = f.support ?? { inliers: 0, coverage: 0 };
  const why = (slack, k) => {
    if (o <= s.targetOverlap - slack) return 'target';
    if (early && under(floor, support, k)) return 'support';
    if (early && o2 != null && o2 <= REDUNDANT_OVERLAP - slack) return 'redundancy';
    return null;
  };
  let urge;
  let w;
  if ((w = why(RELAX, 1))) urge = { kind: 'overdue', why: w };
  else if ((w = why(0, NEAR_FLOOR))) urge = { kind: 'due', why: w };
  else if (why(-BAND, BAND_FLOOR)) urge = { kind: 'band' };
  else urge = no;
  return v(own ?? 'waiting', urge, own == null && !under(floor, support, 1));
}

/** The best frames of the band: one the gates pass and one they refused, by quality. */
export class Picker {
  constructor() { this.clear(); }
  clear() { this.good = null; this.refused = null; }
  holds() { return this.good != null || this.refused != null; }
  /**
   * Offers a frame ({ quality, acceptable, frame }, or null when it can't be a keyframe:
   * a lost one); returns { frame, why, relaxed } when one is to be kept.
   */
  step(urge, offer) {
    if (urge.kind === 'no') {
      this.clear();
      return null;
    }
    if (offer) {
      const slot = offer.acceptable ? 'good' : 'refused';
      // On a tie the later frame, nearer the target.
      if (this[slot] == null || offer.quality >= this[slot].quality) this[slot] = { quality: offer.quality, frame: offer.frame };
    }
    if (urge.kind === 'band') return null;
    let pick = null;
    if (this.good) pick = { frame: this.good.frame, why: urge.why, relaxed: false };
    else if (urge.kind === 'overdue' && this.refused) pick = { frame: this.refused.frame, why: urge.why, relaxed: true };
    else return null;
    this.clear();
    return pick;
  }
}

/** How fast the overlap with the last keyframe falls, per frame. */
export class Speed {
  constructor() { this.last = null; this.drop = 0; }
  update(overlap) {
    if (this.last != null && overlap != null) this.drop += SPEED_WEIGHT * (this.last - overlap - this.drop);
    this.last = overlap;
    return this.drop;
  }
  rebase() { this.last = null; }
}

/**
 * The overlap the rules judge a frame by: what it will be at the next check, at the
 * current speed (the smoothed drop per check). A page that checks a frame every few
 * hundred milliseconds (a phone) otherwise keeps its keyframes past the target, or loses
 * the last one between two checks. Not anticipated below LOST_OVERLAP (lost is lost) nor
 * into it.
 */
export function anticipate(overlap, drop) {
  if (overlap == null || overlap < LOST_OVERLAP || !(drop > 0)) return overlap;
  return Math.max(LOST_OVERLAP, overlap - drop);
}

/** Whether to ask for slower motion. */
export function slowDown(drop, blurry) {
  return drop > SLOW_DOWN_DROP || (blurry && drop > SLOW_DOWN_DROP / 2);
}

/**
 * Blur against the session's sharp level, a slowly decaying maximum of the sharpness.
 * `halfLife`: frames in which the level halves (a video's frames come at their own rate).
 */
export class Blur {
  constructor(halfLife = BLUR_HALF_LIFE) { this.level = 0; this.halfLife = halfLife; }
  blurry(s) {
    const decay = Math.pow(0.5, 1 / this.halfLife);
    this.level = Math.max(this.level * decay, s);
    return s < BLUR_FRACTION * this.level;
  }
  ratio(s) { return this.level > 0 ? Math.min(1, Math.max(0, s / this.level)) : 1; }
}

/** Which held keyframes to keep holding (seqs ascending, newest last). */
export function hold(seqs) {
  const n = seqs.length;
  const keep = new Array(n).fill(false);
  let sparse = 0;
  for (let i = n - 1; i >= 0; --i) {
    if (n - i <= RECENT_KEYFRAMES) keep[i] = true;
    else if (seqs[i] % SPARSE_EVERY === 0 && sparse < SPARSE_KEYFRAMES) {
      keep[i] = true;
      ++sparse;
    }
  }
  return keep;
}

/** How many keypoint scores are at least CONFIDENT_SCORE (model 'xfeat': CONFIDENT_SCORE_XFEAT). */
export function confident(scores, model = 'aliked') {
  const at = model === 'xfeat' ? CONFIDENT_SCORE_XFEAT : CONFIDENT_SCORE;
  let n = 0;
  for (const s of scores) if (s >= at) ++n;
  return n;
}

// ---- geometry ----------------------------------------------------------------------------

const pointsOf = (xy) => {
  const out = [];
  for (let i = 0; i + 1 < xy.length; i += 2) out.push([xy[i], xy[i + 1]]);
  return out;
};

/** Convex hull, counter-clockwise (Andrew's monotone chain); [] for fewer than 3 points or a line. */
export function hull(pts) {
  if (pts.length < 3) return [];
  const p = pts.map((q) => [q[0], q[1]]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const u = p.filter((q, i) => i === 0 || q[0] !== p[i - 1][0] || q[1] !== p[i - 1][1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const h = [];
  for (let pass = 0; pass < 2; ++pass) {
    const start = h.length;
    const seq = pass === 0 ? u : [...u].reverse();
    for (const q of seq) {
      while (h.length >= start + 2 && cross(h[h.length - 2], h[h.length - 1], q) <= 0) h.pop();
      h.push(q);
    }
    h.pop();
  }
  return h.length < 3 ? [] : h;
}

export function signedArea(poly) {
  let a = 0;
  for (let k = 0; k < poly.length; ++k) {
    const u = poly[k], v = poly[(k + 1) % poly.length];
    a += u[0] * v[1] - v[0] * u[1];
  }
  return a / 2;
}
export const polygonArea = (poly) => Math.abs(signedArea(poly));
export const hullArea = (pts) => polygonArea(hull(pts));

/** `subject` clipped to the convex, counter-clockwise polygon `by` (Sutherland–Hodgman). */
export function clip(subject, by) {
  let out = subject.slice();
  for (let k = 0; k < by.length && out.length; ++k) {
    const a = by[k], b = by[(k + 1) % by.length];
    const side = (p) => (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    const input = out;
    out = [];
    for (let i = 0; i < input.length; ++i) {
      const p = input[i], q = input[(i + 1) % input.length];
      const sp = side(p), sq = side(q);
      if (sp >= 0) out.push(p);
      if ((sp >= 0) !== (sq >= 0)) {
        const t = sp / (sp - sq);
        out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
      }
    }
  }
  return out;
}

const apply = (m, p) => [m[0] * p[0] + m[1] * p[1] + m[2], m[3] * p[0] + m[4] * p[1] + m[5]];

/**
 * Least-squares affine map taking each pair's first point to its second ([[p, q], ...]),
 * fitted again without the pairs far off the first fit. [a, b, c, d, e, f] or null.
 */
export function fitAffine(pairs) {
  const solve = (use) => {
    let cx = 0, cy = 0, n = 0;
    pairs.forEach(([p], k) => { if (use(k)) { cx += p[0]; cy += p[1]; ++n; } });
    if (n < 3) return null;
    cx /= n;
    cy /= n;
    const ata = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    const atb = [[0, 0, 0], [0, 0, 0]];
    pairs.forEach(([p, q], k) => {
      if (!use(k)) return;
      const r = [p[0] - cx, p[1] - cy, 1];
      for (let i = 0; i < 3; ++i) {
        for (let j = 0; j < 3; ++j) ata[i][j] += r[i] * r[j];
        atb[0][i] += r[i] * q[0];
        atb[1][i] += r[i] * q[1];
      }
    });
    const det3 = (m) => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
      m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    const d = det3(ata);
    if (Math.abs(d) < 1e-9 * Math.max(ata[0][0] * ata[1][1], 1)) return null;  // matches on a line
    const sol = [[0, 0, 0], [0, 0, 0]];
    for (let o = 0; o < 2; ++o)
      for (let c = 0; c < 3; ++c) {
        const m = ata.map((row, r) => row.map((v, j) => (j === c ? atb[o][r] : v)));
        sol[o][c] = det3(m) / d;
      }
    const [a, b, c] = sol[0], [dd, e, f] = sol[1];
    return [a, b, c - a * cx - b * cy, dd, e, f - dd * cx - e * cy];
  };
  const first = solve(() => true);
  if (!first) return null;
  const residual = (k) => {
    const [p, q] = pairs[k];
    const m = apply(first, p);
    return Math.hypot(m[0] - q[0], m[1] - q[1]);
  };
  const res = pairs.map((_, k) => residual(k)).sort((x, y) => x - y);
  const cut = Math.max(3 * res[res.length >> 1], 2);
  return solve((k) => residual(k) <= cut) ?? first;
}

/**
 * How much of the keyframe the current frame still sees: the motion between them fitted
 * from the verified matches (`pairs`: [keyframe keypoint, frame keypoint] indices into
 * refXy and curXy) as an affine map, the current frame (w x h) carried into the keyframe
 * by it, and the share of the keyframe's keypoint hull it covers.
 */
export function overlap(refXy, curXy, pairs, w, h) {
  const region = hull(pointsOf(refXy));
  const total = polygonArea(region);
  if (total <= 0) return 0;
  const pts = [];
  for (const [i, j] of pairs) {
    if (2 * j + 1 >= curXy.length || 2 * i + 1 >= refXy.length) continue;
    pts.push([[curXy[2 * j], curXy[2 * j + 1]], [refXy[2 * i], refXy[2 * i + 1]]]);
  }
  const map = fitAffine(pts);
  if (!map) return 0;
  const view = [[0, 0], [w, 0], [w, h], [0, h]].map((p) => apply(map, p));
  if (signedArea(view) <= 0) return 0;  // mirrored or collapsed: not a camera motion
  return Math.min(1, Math.max(0, polygonArea(clip(region, view)) / total));
}

/** Share of a w x h frame the convex hull of xy covers. */
export function coverage(xy, w, h) {
  const area = w * h;
  return area > 0 ? Math.min(1, Math.max(0, hullArea(pointsOf(xy)) / area)) : 0;
}

/** The support of verified matches `seen` ([i, j] into refXy, curXy), both frames w x h. */
export function supportOf(refXy, curXy, seen, w, h) {
  const a = [], b = [];
  for (const [i, j] of seen) {
    a.push(refXy[2 * i], refXy[2 * i + 1]);
    b.push(curXy[2 * j], curXy[2 * j + 1]);
  }
  return { inliers: seen.length, coverage: Math.min(coverage(a, w, h), coverage(b, w, h)) };
}

/** Whether keypoints at xy can anchor the chain: enough of them, not all on a line. */
export function usableReference(xy) {
  return xy.length / 2 >= MIN_REFERENCE_KEYPOINTS && hullArea(pointsOf(xy)) > 0;
}

/**
 * Variance of the Laplacian of the frame's luma, sampled to at most ~640 px across (the
 * usual focus measure; only its ratio to the sharp level is used). `rgba`: Uint8ClampedArray
 * or Uint8Array of w x h pixels.
 */
export function sharpness(rgba, w, h) {
  const step = Math.max(1, Math.ceil(w / 640));
  const sw = Math.floor(w / step), sh = Math.floor(h / step);
  if (sw < 3 || sh < 3) return 0;
  const g = new Float32Array(sw * sh);
  for (let y = 0; y < sh; ++y)
    for (let x = 0; x < sw; ++x) {
      const k = 4 * (y * step * w + x * step);
      g[y * sw + x] = 0.299 * rgba[k] + 0.587 * rgba[k + 1] + 0.114 * rgba[k + 2];
    }
  let sum = 0, sq = 0, n = 0;
  for (let y = 1; y < sh - 1; ++y)
    for (let x = 1; x < sw - 1; ++x) {
      const i = y * sw + x;
      const l = g[i - 1] + g[i + 1] + g[i - sw] + g[i + sw] - 4 * g[i];
      sum += l;
      sq += l * l;
      ++n;
    }
  const mean = sum / n;
  return sq / n - mean * mean;
}

/** The strongest `cap` matches ([n][2] Int32Array with scores [n]) by score, in order. */
export function strongest(matches, scores, cap) {
  const n = scores.length;
  if (!cap || n <= cap) return matches.slice(0, 2 * n);
  const order = Array.from({ length: n }, (_, k) => k).sort((x, y) => scores[y] - scores[x] || x - y).slice(0, cap);
  order.sort((x, y) => x - y);
  const out = new Int32Array(2 * order.length);
  order.forEach((k, i) => { out[2 * i] = matches[2 * k]; out[2 * i + 1] = matches[2 * k + 1]; });
  return out;
}
