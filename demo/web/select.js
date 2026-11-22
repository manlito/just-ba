// Which frames of a stream become keyframes (D157, D167): the live capture's keep loop,
// apart from its page, so the live demo's videos are selected by the same rules. Per
// frame: keypoints, the matcher's features, the matches to the last keyframe at the live
// depth checked against a relative pose (demo_pair_inliers), the overlap and support
// those give, the blur, and gates.js's rules with the best frame of the band and
// re-localization against held keyframes. A kept frame becomes the reference for the
// next ones. What a page does with a keyframe (a map, a thumbnail) is the page's.
import * as G from './gates.js';

// LightGlue's live depth: on a Pixel 9a at 640 px, 2 layers take 160 ms (342 matches),
// 3 take 215 ms (481), 5 take 330 ms, adaptive 9 about 510 ms. LighterGlue (XFeat's) at
// the same depth: its layers are about a third of LightGlue's work. FlexGlue: its
// layer-0 head.
export const FIXED = { depthConfidence: -1, widthConfidence: -1 };
export const LIVE = { flexglue: { layers: 0 }, lightglue: { layers: 2, ...FIXED }, lighterglue: { layers: 2, ...FIXED } };

/** Like Recon's 800 confident keypoints are for a 1280 x 720 frame. */
export function minKeypointsFor(w, h) {
  return Math.max(150, Math.round(800 * Math.min(1, (w * h) / (1280 * 720))));
}

/**
 * A frame's matches checked against a relative pose on the CPU, by the demo's module `M`
 * (demo_pair_inliers): (refXy, curXy, matches, w, h) -> the [i, j] pairs that agree.
 */
export function verifier(M) {
  return (refXy, curXy, matches, w, h) => {
    const n = matches.length / 2;
    if (n < G.MIN_INLIERS) return [];
    const a = new Float32Array(2 * n), b = new Float32Array(2 * n);
    for (let k = 0; k < n; k++) {
      const i = matches[2 * k], j = matches[2 * k + 1];
      a[2 * k] = refXy[2 * i]; a[2 * k + 1] = refXy[2 * i + 1];
      b[2 * k] = curXy[2 * j]; b[2 * k + 1] = curXy[2 * j + 1];
    }
    const pa = M._malloc(8 * n), pb = M._malloc(8 * n), pm = M._malloc(n);
    try {
      M.HEAPF32.set(a, pa >> 2);
      M.HEAPF32.set(b, pb >> 2);
      const got = M._demo_pair_inliers(n, pa, pb, w, h, G.INLIER_PX, pm);
      if (got < 0) throw new Error(JSON.parse(M.UTF8ToString(M._demo_last_error())).error);
      const mask = M.HEAPU8.subarray(pm, pm + n);
      const seen = [];
      for (let k = 0; k < n; k++) if (mask[k]) seen.push([matches[2 * k], matches[2 * k + 1]]);
      return seen;
    } finally {
      M._free(pa); M._free(pb); M._free(pm);
    }
  };
}

const now = () => performance.now();

export class FrameSelector {
  /**
   * extractor, matcher: just-keypoints' and just-matching's; matcherKind 'flexglue',
   * 'lightglue' or 'lighterglue'; kpModel 'aliked' or 'xfeat'; module: the demo's, for
   * demo_pair_inliers (or `verify`, a function as verifier()'s); live: the matcher's
   * options for a live match; settings, floor: gates.js's; phone: the keyframe before
   * last measured every other frame; anticipate: the overlap judged at the next check
   * (a camera; not a video stepped through); relocalizePerFrame: held keyframes tried a
   * frame while lost; blurHalfLife: frames in which the sharp level halves.
   */
  constructor({ extractor, matcher, matcherKind, kpModel = 'aliked', module = null, verify = null, live = LIVE[matcherKind],
                settings = G.defaultSettings(), floor = G.floorOf(), phone = false, anticipate = true,
                relocalizePerFrame = G.RELOCALIZE_PER_FRAME, blurHalfLife = G.BLUR_HALF_LIFE }) {
    Object.assign(this, { extractor, matcher, matcherKind, kpModel, live, settings, floor, phone, anticipate, relocalizePerFrame });
    this.verify = verify ?? verifier(module);
    this.reference = null;   // the keyframe the frames are measured against
    this.before = null;      // the one before it (redundancy)
    this.twoBack = false;
    this.lastOverlap2 = null;
    this.held = [];          // keyframes held for re-localization
    this.cursor = 0;
    this.count = 0;          // keyframes kept
    this.frames = 0;         // frames stepped
    this.picker = new G.Picker();
    this.speed = new G.Speed();
    this.blur = new G.Blur(blurHalfLife);
  }

  describe(image, kp) {
    return this.matcherKind === 'flexglue' ? this.matcher.describe(image) : this.matcher.describeKeypoints(image, kp.xy, kp.desc);
  }

  // The frame (features f, keypoints xy, size w x h) against keyframe r.
  async measure(r, f, xy, w, h) {
    if (!r.feat || !xy.length || !r.xy.length) return { overlap: 0, support: { inliers: 0, coverage: 0 }, seen: [] };
    const m = await this.matcher.match(r.feat, r.xy, f, xy, this.live);
    const seen = this.verify(r.xy, xy, m.matches, w, h);
    return {
      overlap: seen.length >= G.MIN_INLIERS ? G.overlap(r.xy, xy, seen, w, h) : 0,
      support: G.supportOf(r.xy, xy, seen, w, h),
      seen,
    };
  }

  // Held keyframes other than the reference, a few a frame (all for a Shoot now): the one it overlaps most.
  async relocalize(f, xy, w, h, all) {
    const others = this.held.filter((k) => k !== this.reference && k.feat).reverse();
    if (!others.length) return null;
    const n = all ? others.length : Math.min(this.relocalizePerFrame, others.length);
    const start = all ? 0 : this.cursor % others.length;
    let best = null;
    for (let k = 0; k < n; k++) {
      const r = others[(start + k) % others.length];
      const found = await this.measure(r, f, xy, w, h);
      if (found.overlap >= G.LOST_OVERLAP && (!best || found.overlap > best.m.overlap)) best = { r, m: found };
    }
    this.cursor = start + n;
    return best;
  }

  // Candidates hold GPU features: those neither held by the picker nor kept are freed.
  freeCandidate(c) {
    if (c && !c.kept && c !== this.picker.good?.frame && c !== this.picker.refused?.frame) c.feat?.destroy();
  }

  clearPicker() {
    const held = [this.picker.good?.frame, this.picker.refused?.frame];
    this.picker.clear();
    for (const c of held) this.freeCandidate(c);
  }

  /**
   * One frame (ImageData). `shoot`: keep it now. `window` (a video sampled evenly, no
   * chain): { close }, the frames of one window ranked by their own gates and quality,
   * the best kept when it closes. Returns { gate, why, overlap, overlap2, slowDown,
   * relocalized, kept, keptImage, keypoints, confident, ms }: kept is the new keyframe
   * ({ index, seq, xy, desc, feat, w, h, shoot, why, relocalized, relocalizedTo, tag,
   * ms }, its features now the caller's to free) or null, keptImage its pixels (it may
   * be an earlier frame than this one: the best of the band); `tag` is passed through
   * to the keyframe made from this frame.
   */
  async step(image, { shoot = false, window = null, tag = null } = {}) {
    const w = image.width, h = image.height;
    const tKp = now();
    const kp = await this.extractor.extract(image);
    const tDescribe = now();
    const confidentNow = G.confident(kp.score, this.kpModel);
    const sharp = G.sharpness(image.data, w, h);
    const blurry = this.blur.blurry(sharp);
    const ratio = this.blur.ratio(sharp);
    const f = await this.describe(image, kp);
    const tMatch = now();
    let found = null, overlap2 = null, relocalized = false;
    if (this.reference && !window) {
      found = await this.measure(this.reference, f, kp.xy, w, h);
      if (found.overlap < G.LOST_OVERLAP && (shoot || !this.picker.holds())) {
        const other = await this.relocalize(f, kp.xy, w, h, shoot);
        if (other) {
          this.reference = other.r;
          this.before = null;
          this.lastOverlap2 = null;
          this.twoBack = false;
          this.clearPicker();
          this.speed.rebase();
          found = other.m;
          relocalized = true;
        }
      }
      // The keyframe before last (redundancy): every frame on a computer; every other frame
      // on a phone, where each live match is ~160 ms, keeping the last measure in between.
      // Without it a phone capture chained keyframes to their neighbour only (pairs two
      // apart had 2-21 matches) and broke into pieces.
      // Also above EARLY_MAX_OVERLAP, where it can't make a keep: it is what arms the rule
      // (twoBack), which a keyframe kept late would otherwise never do.
      if (this.settings.redundant && this.before?.feat) {
        if (!this.phone || this.frames % 2 === 0 || this.lastOverlap2 == null || relocalized) {
          const m2 = await this.measure(this.before, f, kp.xy, w, h);
          this.twoBack ||= m2.overlap > G.REDUNDANT_OVERLAP;
          this.lastOverlap2 = m2.overlap;
        }
        overlap2 = this.lastOverlap2;
      }
    }
    const tRules = now();
    const overlapNow = found ? found.overlap : null;
    const speed = this.speed.update(overlapNow);
    // Judged at the next check (G.anticipate): the speed is the drop per check.
    const judged = (o) => (this.anticipate ? G.anticipate(o, speed) : o);
    const own = G.ownGate(this.settings, { keypoints: confidentNow, blurry });
    const verdict = window
      ? { gate: own ?? 'waiting', urge: window.close ? { kind: 'overdue', why: 'window' } : { kind: 'band' }, acceptable: own == null }
      : G.decide(this.settings, this.floor, {
        shoot, hasMatcher: true, hasReference: !!this.reference, keypoints: confidentNow, blurry,
        overlap: judged(overlapNow), support: found?.support ?? { inliers: 0, coverage: 0 },
        overlap2: this.twoBack ? judged(overlap2) : null,
      });
    // The descriptors are kept whatever the matcher: a page may match again with LightGlue (ALIKED's) or
    // LighterGlue (XFeat's).
    const cand = { xy: kp.xy, desc: kp.desc, feat: f, w, h, image, overlap: overlapNow, relocalized, tag,
                   ms: { keypoints: tDescribe - tKp, describe: tMatch - tDescribe } };
    let pick;
    if (shoot) {
      this.clearPicker();
      pick = { frame: cand, why: 'shoot', relaxed: false };
    } else {
      const quality = confidentNow * (0.5 + 0.5 * ratio);
      const before = [this.picker.good?.frame, this.picker.refused?.frame];
      pick = this.picker.step(verdict.urge, verdict.gate !== 'lost' ? { quality, acceptable: verdict.acceptable, frame: cand } : null);
      if (pick) pick.frame.kept = true;
      for (const c of before) this.freeCandidate(c);
    }
    const kept = pick ? this.keep(pick.frame, pick.why, shoot) : null;
    this.freeCandidate(cand);
    this.frames++;
    return {
      gate: pick ? 'kept' : verdict.gate === 'kept' ? 'waiting' : verdict.gate, why: pick?.why ?? null,
      overlap: overlapNow, overlap2, slowDown: G.slowDown(speed, blurry && this.settings.rejectBlur), relocalized,
      kept, keptImage: pick ? pick.frame.image : null, keypoints: kp.n, confident: confidentNow,
      ms: { keypoints: tDescribe - tKp, describe: tMatch - tDescribe, match: tRules - tMatch },
    };
  }

  // A kept frame becomes a keyframe, and the reference for the next frames when it can
  // anchor the chain.
  keep(c, why, shoot) {
    c.kept = true;
    const kf = { index: this.count, seq: this.count + 1, xy: c.xy, desc: c.desc, feat: c.feat, w: c.w, h: c.h, shoot, why,
                 relocalized: c.relocalized, relocalizedTo: c.relocalized ? this.reference : null, tag: c.tag, ms: c.ms };
    this.count++;
    if (G.usableReference(c.xy)) {
      this.before = this.reference;
      this.reference = kf;
      this.twoBack = false;
      this.lastOverlap2 = null;
      this.speed.rebase();
      // G.hold over those held and the new one: a keyframe let go is never held again.
      const pool = [...this.held, kf];
      const keepIt = G.hold(pool.map((k) => k.seq));
      this.held = pool.filter((_, i) => keepIt[i]);
    }
    return kf;
  }

  /** The keyframes the next frames may be measured against: their features must stay. */
  measured() { return [this.reference, this.before, ...this.held]; }

  /**
   * The chain broken (a video that moved on to another place, D167): no reference, so
   * the next frame that passes its own gates is kept as a first one. The held keyframes
   * stay, for re-localization later.
   */
  breakChain() {
    this.reference = null;
    this.before = null;
    this.twoBack = false;
    this.lastOverlap2 = null;
    this.clearPicker();
    this.speed.rebase();
  }

  /**
   * The end of a stream: the frame held as the best since the last keyframe (one the
   * gates pass) is kept, why 'end'. Returns { gate: 'kept', why, kept, keptImage } or null.
   */
  flush() {
    const c = this.picker.good?.frame;
    if (!c) {
      this.clearPicker();
      return null;
    }
    this.picker.good = null;
    this.clearPicker();
    const kept = this.keep(c, 'end', false);
    return { gate: 'kept', why: 'end', kept, keptImage: c.image, overlap: c.overlap, relocalized: false, slowDown: false };
  }

  /** Frees the candidates held. The keyframes' features are the caller's. */
  dispose() {
    this.clearPicker();
    this.reference = this.before = null;
    this.held = [];
  }
}
