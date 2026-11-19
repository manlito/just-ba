// just-ba's live capture (D157): the camera's frames, keyframes kept as Like Recon keeps
// them (gates.js), a map grown from the keyframes while capturing (demo::Session in
// capture_worker.js), then Finish, the whole capture reconstructed from that map.
//
// Per camera frame, on this page's thread: ALIKED keypoints (just-keypoints), the
// matcher's features of the frame (just-matching: FlexGlue's dense features, or
// LightGlue's of the keypoints), its matches to the last keyframe at the live depth
// (FlexGlue's layer-0 head), checked against a relative pose on the CPU
// (demo_pair_inliers), the overlap and support those give, the blur, and the rules. A
// kept frame is matched to the last few keyframes at the online depth, one pair per
// camera frame so the camera never waits, and goes to the worker's session; one the
// map can't place is matched to frames chosen by appearance (FlexGlue's shortlist) and
// tried again. Finish matches more pairs (the shortlist over every keyframe) and runs
// the mapper from the live map.
//
// URL parameters: matcher=flexglue|lightglue, keypoints=1024|2048, side=<px> (the
// working image's longer side), debug=1 (timings in the console), video=<url> (a recorded
// video instead of the camera, for testing; rate=<playback rate>, 0.5 by default).
import * as G from './gates.js';
import { ModelView } from './view.js';

const params = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);
const DEBUG = params.get('debug') === '1';

// FlexGlue's released checkpoint (just-matching's tools/fetch_upstream.sh pins it).
const FLEXGLUE = {
  sha256: 'd7371f743b8cdc4adf723c2aa05b181587fe4bc177fced139697975dff790068',
  bytes: 567328059,
  stored: 'flexglue-fp16.jmtw',
  storedBytes: 283537600,
};
// Like Recon's numbers (src-tauri/src/capture.rs, online.rs).
// LightGlue's live and map depths: on a Pixel 9a at 640 px, 2 layers take 160 ms (342
// matches), 3 take 215 ms (481), 5 take 330 ms, adaptive 9 about 510 ms.
const LIVE = { flexglue: { layers: 0 }, lightglue: { layers: 2, depthConfidence: -1, widthConfidence: -1 } };
const ONLINE = { flexglue: { layers: 3 }, lightglue: { layers: 3, depthConfidence: -1, widthConfidence: -1 } };
// Finish's pairs: FlexGlue as the map's; LightGlue at full, adaptive depth (it has the time).
const FINISH = { flexglue: { layers: 3 }, lightglue: { layers: 9, depthConfidence: 0.95, widthConfidence: 0.99 } };
const SEQUENTIAL = 3;            // keyframes each new one is matched to
const MATCHES_PER_PAIR = 1024;   // the strongest, per online pair
const WINDOW = 8;                // keyframes adjusted together
const RESTART_AFTER = 10;        // keyframes in a row not placed before a new piece
const ONLINE_MIN_PAIR_INLIERS = 15;
const APPEARANCE_PARTNERS = 4;   // more keyframes for one the map can't place (FlexGlue)
const FINISH_SHORTLIST = 10;     // pairs per keyframe for the finish (FlexGlue's shortlist)
const FINISH_SEQUENTIAL = 5;     // LightGlue's finish: each keyframe with the five before it
const MIN_FINISH_MATCHES = 15;

// A phone: LightGlue by default (FlexGlue's backbone is ~1 GB of GPU memory and takes ~2 s a
// frame on a Pixel 9a), a 640 px working image (ALIKED 136 ms there) and 1,024 keypoints.
const phone = matchMedia('(pointer: coarse)').matches && Math.min(screen.width, screen.height) < 820;
const MAX_SIDE = Number(params.get('side')) || (phone ? 640 : 1280);
const FEATURE_CACHE = phone ? 24 : 48;
// Held keyframes tried a frame while lost: each is a live match (~160 ms on a Pixel 9a).
const RELOCALIZE_PER_FRAME = phone ? 1 : G.RELOCALIZE_PER_FRAME;

const S = {
  gpu: null, precision: 'fp32',
  live: null,          // the demo's module on this thread, for demo_pair_inliers
  worker: null, calls: new Map(), nextId: 1,
  extractor: null, extractorK: 0,
  matcher: null, matcherKind: null,
  weights: null,       // FlexGlue: { stored } or { bytes } (this visit only)
  stream: null, track: null, wakeLock: null,
  running: false, finishing: false, shoot: false, busy: false,
  session: null,       // { width, height }
  keyframes: [],       // { index, seq, xy, desc, feat, w, h, jpeg, placed, shoot, thumb }
  reference: null, before: null, twoBack: false, held: [], cursor: 0,
  picker: new G.Picker(), speed: new G.Speed(), blur: new G.Blur(),
  floor: G.floorOf(), settings: G.defaultSettings(),
  jobs: [],            // online work: one keyframe at a time
  links: new Set(),    // "a-b" pairs already matched for the map
  map: null, lastStep: null, result: null,
  stats: { frameMs: 0, mapMs: 0, frames: 0, kpMs: 0, describeMs: 0, matchMs: 0, loopMs: 0 },
};

// ---- small things

function setStatus(text, error = false) {
  $('status').textContent = text;
  $('status').classList.toggle('error', error);
}
function setProgress(done, total) {
  const p = $('progress');
  p.hidden = done == null;
  if (done != null) { p.max = total || 1; p.value = done; }
}
function notice(text) {
  $('notice').hidden = !text;
  $('notice').textContent = text ?? '';
}
const fmt = (v, d = 2) => (v == null || !Number.isFinite(v) ? '–' : Number(v).toFixed(d));
const int = (v) => Number(v).toLocaleString('en-US');
const matcherKind = () => document.querySelector('input[name=matcher]:checked')?.value ?? 'lightglue';
const topK = () => Number(document.querySelector('input[name=keypoints]:checked')?.value ?? (phone ? 1024 : 2048));
// The next camera frame (or a moment, when none comes: a video that ended).
const nextFrame = () => new Promise((r) => {
  const v = $('video');
  const t = setTimeout(r, 150);
  if (v.requestVideoFrameCallback && !document.hidden) v.requestVideoFrameCallback(() => { clearTimeout(t); r(); });
});

// ---- the worker's session

function call(op, args = {}) {
  const id = S.nextId++;
  return new Promise((resolve, reject) => {
    S.calls.set(id, { resolve, reject });
    S.worker.postMessage({ id, op, ...args });
  });
}
function startWorker() {
  S.worker = new Worker(new URL('capture_worker.js', import.meta.url), { type: 'module' });
  S.worker.onmessage = ({ data }) => {
    if (data.progress) return onFinishProgress(data.progress);
    const c = S.calls.get(data.id);
    if (!c) return;
    S.calls.delete(data.id);
    data.ok ? c.resolve(data.result) : c.reject(new Error(data.error));
  };
  S.worker.onerror = (e) => setStatus('The mapper stopped: ' + (e.message || 'worker error'), true);
}

// ---- a frame's matches checked against a relative pose (on the CPU, this thread)

function verified(refXy, curXy, matches, w, h) {
  const M = S.live;
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
}

// The frame (features f, keypoints xy, size w x h) against keyframe r.
async function measure(r, f, xy, w, h) {
  if (!r.feat || !xy.length || !r.xy.length) return { overlap: 0, support: { inliers: 0, coverage: 0 }, seen: [] };
  const m = await S.matcher.match(r.feat, r.xy, f, xy, LIVE[S.matcherKind]);
  const seen = verified(r.xy, xy, m.matches, w, h);
  return {
    overlap: seen.length >= G.MIN_INLIERS ? G.overlap(r.xy, xy, seen, w, h) : 0,
    support: G.supportOf(r.xy, xy, seen, w, h),
    seen,
  };
}

// Held keyframes other than the reference, a few a frame (all for a Shoot now): the one it overlaps most.
async function relocalize(f, xy, w, h, all) {
  const others = S.held.filter((k) => k !== S.reference && k.feat).reverse();
  if (!others.length) return null;
  const n = all ? others.length : Math.min(RELOCALIZE_PER_FRAME, others.length);
  const start = all ? 0 : S.cursor % others.length;
  let best = null;
  for (let k = 0; k < n; k++) {
    const r = others[(start + k) % others.length];
    const found = await measure(r, f, xy, w, h);
    if (found.overlap >= G.LOST_OVERLAP && (!best || found.overlap > best.m.overlap)) best = { r, m: found };
  }
  S.cursor = start + n;
  return best;
}

// ---- the camera's loop

const grabCanvas = document.createElement('canvas');
const grabCtx = grabCanvas.getContext('2d', { willReadFrequently: true });

function workingSize() {
  const v = $('video');
  const s = Math.min(1, MAX_SIDE / Math.max(v.videoWidth, v.videoHeight));
  return [Math.round(v.videoWidth * s), Math.round(v.videoHeight * s)];
}

function grab() {
  const [w, h] = workingSize();
  if (grabCanvas.width !== w || grabCanvas.height !== h) { grabCanvas.width = w; grabCanvas.height = h; }
  grabCtx.drawImage($('video'), 0, 0, w, h);
  return { w, h, image: grabCtx.getImageData(0, 0, w, h) };
}

async function describe(image, kp) {
  return S.matcherKind === 'flexglue' ? S.matcher.describe(image) : S.matcher.describeKeypoints(image, kp.xy, kp.desc);
}

// Candidates hold GPU features: those neither held by the picker nor kept are freed.
function freeCandidate(c) {
  if (c && !c.kept && c !== S.picker.good?.frame && c !== S.picker.refused?.frame) c.feat?.destroy();
}

async function liveStep() {
  const t0 = performance.now();
  const frame = grab();
  const { w, h } = frame;
  if (w !== S.session.width || h !== S.session.height) {
    showChip('Turn the camera back the way it was', 'warn');
    return;
  }
  const shoot = S.shoot;
  S.shoot = false;
  const tKp = performance.now();
  const kp = await S.extractor.extract(frame.image);
  const tDescribe = performance.now();
  const confidentNow = G.confident(kp.score);
  const sharp = G.sharpness(frame.image.data, w, h);
  const blurry = S.blur.blurry(sharp);
  const ratio = S.blur.ratio(sharp);
  const f = await describe(frame.image, kp);
  const tMatch = performance.now();
  let now = null, overlap2 = null, relocalized = false;
  if (S.reference) {
    now = await measure(S.reference, f, kp.xy, w, h);
    if (now.overlap < G.LOST_OVERLAP && (shoot || !S.picker.holds())) {
      const found = await relocalize(f, kp.xy, w, h, shoot);
      if (found) {
        S.reference = found.r;
        S.before = null;
        S.lastOverlap2 = null;
        S.twoBack = false;
        clearPicker();
        S.speed.rebase();
        now = found.m;
        relocalized = true;
      }
    }
    // The keyframe before last (redundancy): every frame on a computer; every other frame
    // on a phone, where each live match is ~160 ms, keeping the last measure in between.
    // Without it a phone capture chained keyframes to their neighbour only (pairs two
    // apart had 2-21 matches) and broke into pieces.
    if (S.settings.redundant && S.before?.feat) {
      if (!phone || S.stats.frames % 2 === 0 || S.lastOverlap2 == null || relocalized) {
        const m2 = await measure(S.before, f, kp.xy, w, h);
        S.twoBack ||= m2.overlap > G.REDUNDANT_OVERLAP;
        S.lastOverlap2 = m2.overlap;
      }
      overlap2 = S.lastOverlap2;
    }
  }
  const tRules = performance.now();
  const overlapNow = now ? now.overlap : null;
  const speed = S.speed.update(overlapNow);
  const verdict = G.decide(S.settings, S.floor, {
    shoot, hasMatcher: true, hasReference: !!S.reference, keypoints: confidentNow, blurry,
    overlap: overlapNow, support: now?.support ?? { inliers: 0, coverage: 0 }, overlap2: S.twoBack ? overlap2 : null,
  });
  const cand = { xy: kp.xy, desc: S.matcherKind === 'lightglue' ? kp.desc : null, feat: f, w, h, image: frame.image,
                 overlap: overlapNow, relocalized };
  let pick;
  if (shoot) {
    clearPicker();
    pick = { frame: cand, why: 'shoot', relaxed: false };
  } else {
    const quality = confidentNow * (0.5 + 0.5 * ratio);
    const before = [S.picker.good?.frame, S.picker.refused?.frame];
    pick = S.picker.step(verdict.urge, verdict.gate !== 'lost' ? { quality, acceptable: verdict.acceptable, frame: cand } : null);
    if (pick) pick.frame.kept = true;
    for (const c of before) freeCandidate(c);
  }
  if (pick) await keep(pick.frame, pick.why, shoot);
  freeCandidate(cand);

  drawOverlay(kp.xy, now?.seen, w, h);
  const gate = pick ? 'kept' : verdict.gate === 'kept' ? 'waiting' : verdict.gate;
  showGate(gate, pick?.why, overlapNow, G.slowDown(speed, blurry && S.settings.rejectBlur), relocalized);
  const ms = performance.now() - t0;
  const avg = (old, v) => (S.stats.frames ? 0.8 * old + 0.2 * v : v);
  S.stats.frameMs = avg(S.stats.frameMs, ms);
  S.stats.kpMs = avg(S.stats.kpMs, tDescribe - tKp);
  S.stats.describeMs = avg(S.stats.describeMs, tMatch - tDescribe);
  S.stats.matchMs = avg(S.stats.matchMs, tRules - tMatch);
  S.stats.frames++;
  if (DEBUG) console.log(`frame ${ms.toFixed(0)} ms, ${kp.n} kp (${confidentNow} sure), overlap ${fmt(overlapNow)}, ${gate}`);
}

function clearPicker() {
  const held = [S.picker.good?.frame, S.picker.refused?.frame];
  S.picker.clear();
  for (const c of held) freeCandidate(c);
}

// A kept frame becomes a keyframe: the reference for the next frames (when it can anchor
// the chain), and the map's next image.
async function keep(c, why, shoot) {
  c.kept = true;
  const kf = { index: S.keyframes.length, seq: S.keyframes.length + 1, xy: c.xy, desc: c.desc, feat: c.feat, w: c.w,
               h: c.h, shoot, why, placed: false, jpeg: null };
  S.keyframes.push(kf);
  // A small JPEG of it, for its thumbnail, the points' colours and the finish's features.
  const cv = new OffscreenCanvas(c.w, c.h);
  cv.getContext('2d').putImageData(c.image, 0, 0);
  kf.jpeg = await cv.convertToBlob({ type: 'image/jpeg', quality: 0.9 });
  addThumb(kf, cv);
  const partners = S.keyframes.slice(Math.max(0, kf.index - SEQUENTIAL), kf.index);
  if (c.relocalized && S.reference && !partners.includes(S.reference)) partners.push(S.reference);
  if (G.usableReference(c.xy)) {
    S.before = S.reference;
    S.reference = kf;
    S.twoBack = false;
    S.lastOverlap2 = null;
    S.speed.rebase();
    const anchors = S.keyframes.filter((k) => G.usableReference(k.xy));
    const keepIt = G.hold(anchors.map((k) => k.seq));
    S.held = anchors.filter((_, i) => keepIt[i]);
  }
  S.jobs.push({ kf, partners, pairs: [], stage: 'match', done: new Set(), tried: new Set(partners.map((p) => p.index)) });
  freeFeatures();
  setButtons();
}

// Features stay for the keyframes the live loop measures against, the latest ones (the next
// keyframes' partners) and, thinned, older ones for the appearance shortlist.
function freeFeatures() {
  const withFeat = S.keyframes.filter((k) => k.feat);
  const keepSet = new Set([S.reference, S.before, ...S.held]);
  for (const j of S.jobs) { keepSet.add(j.kf); for (const p of j.partners) keepSet.add(p); }
  // Past the cache, every other one of the older half goes (online.rs's thin).
  if (withFeat.length > FEATURE_CACHE) {
    const old = withFeat.length - FEATURE_CACHE / 2;
    withFeat.forEach((k, i) => {
      if (i < old && i % 2 === 1 && !keepSet.has(k)) { k.feat.destroy(); k.feat = null; }
    });
  }
}

// One piece of the map's work: one pair matched, or a keyframe handed to the worker (which
// runs on its own) or its answer taken in. While the worker places a keyframe, the next
// keyframes' pairs are matched; they go to the worker in order, each after the one before
// it is placed or tried again. Returns false when there is nothing to do now.
async function onlineStep() {
  const head = S.jobs[0];
  if (!head) return false;
  if (head.failed) {
    S.jobs.shift();
    notice('The map could not take a keyframe: ' + head.failed.message);
    return true;
  }
  if (head.answer) return takeAnswer(head);
  for (const job of S.jobs) {
    if (job.stage !== 'match' && job.stage !== 'appearance') continue;
    const todo = job.stage === 'match' ? job.partners : job.more;
    const p = todo.find((q) => !job.done.has(q.index));
    if (p) {
      job.done.add(p.index);
      if (p.feat && job.kf.feat) {
        const m = await S.matcher.match(p.feat, p.xy, job.kf.feat, job.kf.xy, ONLINE[S.matcherKind]);
        const matches = G.strongest(m.matches, m.scores, MATCHES_PER_PAIR);
        (job.stage === 'match' ? job.pairs : job.morePairs).push({ earlier: p.index, matches });
        S.links.add(`${p.index}-${job.kf.index}`);
      }
      return true;
    }
    if (job !== head) continue;  // matched, waiting for its turn
    job.done = new Set();
    const t0 = performance.now();
    const canRetry = S.matcherKind === 'flexglue';
    const wait = job.stage === 'match'
      ? call('add', { xy: job.kf.xy, pairs: job.pairs, deferSplit: canRetry })
      : call('retry', { pairs: job.morePairs, newPiece: job.kf.shoot });
    wait.then((r) => { job.answer = r; job.mapMs = performance.now() - t0; }, (e) => { job.failed = e; });
    job.stage = job.stage === 'match' ? 'added' : 'retried';
    return true;
  }
  return false;  // the worker is at it, and every pair is matched
}

async function takeAnswer(job) {
  const { step, map } = job.answer;
  job.answer = null;
  if (DEBUG) {
    const pairs = job.stage === 'added' ? job.pairs : job.morePairs;
    console.log(`keyframe ${job.kf.index} ${job.stage} in ${job.mapMs.toFixed(0)} ms: ${step.ok ? '' : step.error + ' '}` +
      `placed [${step.placed}], pairs ${pairs.map((p) => `${p.earlier}:${p.matches.length / 2}`).join(' ')}`);
  }
  takeMap(step, map, job.mapMs);
  const placed = step.placed?.includes(job.kf.index);
  if (job.stage === 'added' && !placed && S.matcherKind === 'flexglue' && map.started) {
    job.more = await appearancePartners(job);
    job.morePairs = [];
    job.stage = 'appearance';
    return true;
  }
  S.jobs.shift();
  freeFeatures();
  return true;
}

// Keyframes chosen by appearance over the session for one the map couldn't place: among
// those before it (the ones kept since are not in the session yet).
async function appearancePartners(job) {
  const cached = S.keyframes.filter((k) => k.feat && k.index < job.kf.index && !job.tried.has(k.index));
  if (!cached.length) return [];
  const all = [...cached, job.kf];
  const me = all.length - 1;
  const list = await S.matcher.shortlist(all.map((k) => k.feat), { k: 2 * APPEARANCE_PARTNERS });
  const chosen = [];
  for (let p = 0; p < list.n && chosen.length < APPEARANCE_PARTNERS; p++) {
    const a = list.pairs[2 * p], b = list.pairs[2 * p + 1];
    const other = a === me ? b : b === me ? a : -1;
    if (other >= 0 && !chosen.includes(all[other])) chosen.push(all[other]);
  }
  for (const k of chosen) job.tried.add(k.index);
  return chosen;
}

function takeMap(step, map, ms) {
  S.lastStep = step;
  S.map = map;
  if (ms != null) S.stats.mapMs = ms;
  if (step && !step.ok) notice('The map: ' + step.error);
  const placed = new Set(map.cameras.map((c) => c.image));
  for (const k of S.keyframes) {
    k.placed = placed.has(k.index);
    k.thumb?.classList.toggle('placed', k.placed);
    k.thumb?.classList.toggle('unplaced', !k.placed && k.index < map.images);
  }
  $('empty').hidden = map.cameras.length > 0;
  view.set({ cameras: map.cameras, points: map.points, colours: null, added: new Set(step?.placed ?? []) });
  renderNumbers();
}

// The map's share of each camera frame: more when it falls behind. The desktop app matches
// the map's pairs on a thread of its own; here they share the GPU module with the camera.
// On a phone a map pair alone takes ~200 ms: one a frame at most, while the camera moves.
const MAP_BUDGET_MS = phone ? 0 : 120, MAP_BEHIND_MS = phone ? 0 : 400;

async function loop() {
  const v = $('video');
  let lastTime = -1;
  while (S.running) {
    if (document.hidden) { await new Promise((r) => setTimeout(r, 200)); continue; }
    try {
      // A frame the camera hasn't moved on from (a paused or ended video) is measured once.
      const fresh = v.currentTime !== lastTime || S.shoot;
      lastTime = v.currentTime;
      const tLoop = performance.now();
      if (fresh) await liveStep();
      const t0 = performance.now();
      const budget = !fresh ? Math.max(MAP_BEHIND_MS, 400) : S.jobs.length > 2 ? MAP_BEHIND_MS : MAP_BUDGET_MS;
      // At least one piece of the map's work a frame, then more while the budget lasts.
      if (await onlineStep()) while (performance.now() - t0 < budget && (await onlineStep()));
      if (!fresh && !S.jobs.length) await new Promise((r) => setTimeout(r, 100));
      if (fresh) S.stats.loopMs = S.stats.frames > 1 ? 0.8 * S.stats.loopMs + 0.2 * (performance.now() - tLoop) : performance.now() - tLoop;
    } catch (e) {
      console.error(e);
      setStatus('Something went wrong: ' + e.message, true);
      S.running = false;
      break;
    }
    await nextFrame();
  }
}

// ---- what the camera view shows

const overlay = $('overlay');
function drawOverlay(xy, seen, w, h) {
  const v = $('video'), dpr = window.devicePixelRatio || 1;
  const cw = v.clientWidth, ch = v.clientHeight;
  if (overlay.width !== Math.round(cw * dpr) || overlay.height !== Math.round(ch * dpr)) {
    overlay.width = Math.round(cw * dpr);
    overlay.height = Math.round(ch * dpr);
  }
  const g = overlay.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, cw, ch);
  const sx = cw / w, sy = ch / h;
  const inl = new Set((seen ?? []).map((p) => p[1]));
  const colKp = getComputedStyle(document.documentElement).getPropertyValue('--kp').trim();
  const colIn = getComputedStyle(document.documentElement).getPropertyValue('--kp-in').trim();
  for (let i = 0; 2 * i < xy.length; i++) {
    g.fillStyle = inl.has(i) ? colIn : colKp;
    g.globalAlpha = inl.has(i) ? 0.95 : 0.55;
    g.fillRect(xy[2 * i] * sx - 1.5, xy[2 * i + 1] * sy - 1.5, 3, 3);
  }
  g.globalAlpha = 1;
}

const GATES = {
  kept: 'Keyframe', waiting: 'Keep moving', too_few_keypoints: 'Too little texture', blurry: 'Blurry: move slower',
  lost: 'Lost: point back at the last keyframe', paused: 'Paused',
};
const WHY = { first: 'first', shoot: 'shot', target: '', support: 'matches thinning', redundancy: 'two back leaving',
              lost: 'kept before losing it' };
function showGate(gate, why, overlap, slow, relocalized) {
  let text = GATES[gate] ?? gate;
  if (gate === 'kept' && WHY[why]) text += ` (${WHY[why]})`;
  if (gate !== 'kept' && slow) text = 'Slow down';
  if (relocalized) text = 'Found an earlier keyframe again';
  showChip(text, gate === 'kept' ? 'kept' : gate === 'lost' || slow ? 'warn' : '');
  const m = $('meter');
  m.hidden = overlap == null;
  if (overlap != null) {
    m.querySelector('i').style.width = `${Math.round(100 * overlap)}%`;
    m.querySelector('b').style.left = `${100 * S.settings.targetOverlap}%`;
  }
}
function showChip(text, kind) {
  const c = $('chip');
  c.hidden = !text;
  c.textContent = text;
  c.className = 'chip' + (kind ? ' ' + kind : '');
}

function addThumb(kf, cv) {
  const t = document.createElement('canvas');
  const th = 54 * (window.devicePixelRatio || 1);
  t.height = th;
  t.width = Math.round(th * kf.w / kf.h);
  t.getContext('2d').drawImage(cv, 0, 0, t.width, t.height);
  t.title = `Keyframe ${kf.index + 1}${kf.why ? ` (${kf.why})` : ''}`;
  $('strip').append(t);
  t.scrollIntoView({ block: 'nearest', inline: 'end' });
  kf.thumb = t;
}

function renderNumbers() {
  const m = S.map, r = S.result;
  const rows = [
    ['Keyframes', int(S.keyframes.length)],
    ['In the map', m ? `${int(m.cameras.length)}${m.pieces > 1 ? ` (${m.pieces} pieces)` : ''}` : '–'],
    ['Points', m ? int(m.points.length / 3) : '–'],
    ['Reprojection error', m ? `${fmt(m.mean_error_px)} px` : '–'],
    ['Focal length', m?.intrinsics?.length ? `${fmt(m.intrinsics[0], 0)} px` : '–'],
    ['Camera frame', `${fmt(S.stats.frameMs, 0)} ms`],
    ['Keypoints · describe · match', `${fmt(S.stats.kpMs, 0)} · ${fmt(S.stats.describeMs, 0)} · ${fmt(S.stats.matchMs, 0)} ms`],
    ['Loop, with the map', `${fmt(S.stats.loopMs, 0)} ms`],
    ['Map, last keyframe', `${fmt(S.stats.mapMs, 0)} ms`],
  ];
  if (r) {
    rows.push(['Finished: registered', `${int(r.registered)} of ${int(r.images)}`]);
    rows.push(['Finished: points', int(r.points.length / 3)]);
    rows.push(['Finished: error', `${fmt(r.mean_error_px)} px`]);
  }
  $('numbers').innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td class="n">${v}</td></tr>`).join('');
}

// ---- the finish

function onFinishProgress(p) {
  const what = { 'focal search': 'searching for the focal length', verify: 'verifying every pair', tracks: 'building tracks',
                 resume: 'starting from the live map', 'first pair': 'starting from the first pair', register: 'registering keyframes',
                 adjust: 'bundle adjustment', final: 'final adjustment' }[p.stage] ?? p.stage;
  setStatus(`Finishing: ${what}${p.images ? `, ${p.registered} of ${p.images} keyframes` : ''}…`);
  if (p.images) setProgress(p.registered, p.images);
}

async function imageOf(kf) {
  return createImageBitmap(kf.jpeg);
}

async function finish() {
  S.running = false;
  S.finishing = true;
  S.cancelFinish = false;
  setButtons();
  try {
    await S.loopDone;
    stopCamera();
    showChip('', '');
    $('meter').hidden = true;
    overlay.getContext('2d').clearRect(0, 0, overlay.width, overlay.height);
    // The map's queue first: every keyframe placed or tried.
    while (S.jobs.length) {
      setStatus(`Adding the last keyframes to the map (${S.jobs.length} left)…`);
      if (!(await onlineStep())) await new Promise((r) => setTimeout(r, 30));
    }
    const n = S.keyframes.length;
    if (n < 2) throw new Error('fewer than two keyframes');
    // More pairs: the shortlist over every keyframe (FlexGlue), or each with the five before it.
    const wanted = [];
    if (S.matcherKind === 'flexglue') {
      for (const [i, k] of S.keyframes.entries()) {
        if (S.cancelFinish) throw new Error('stopped');
        if (k.feat) continue;
        setStatus(`Describing keyframe ${i + 1} of ${n} again…`);
        setProgress(i, n);
        const bm = await imageOf(k);
        k.feat = await S.matcher.describe(bm);
        bm.close();
      }
      setStatus('Choosing the pairs to match…');
      const list = await S.matcher.shortlist(S.keyframes.map((k) => k.feat), { k: FINISH_SHORTLIST });
      for (let p = 0; p < list.n; p++) wanted.push([Math.min(list.pairs[2 * p], list.pairs[2 * p + 1]), Math.max(list.pairs[2 * p], list.pairs[2 * p + 1])]);
    } else {
      for (let j = 1; j < n; j++) for (let i = Math.max(0, j - FINISH_SEQUENTIAL); i < j; i++) wanted.push([i, j]);
    }
    const todo = wanted.filter(([a, b]) => !S.links.has(`${a}-${b}`));
    const pairs = [];
    for (const [p, [a, b]] of todo.entries()) {
      if (S.cancelFinish) throw new Error('stopped');
      if (p % 2 === 0) { setStatus(`Matching pair ${p + 1} of ${todo.length}…`); setProgress(p, todo.length); }
      const A = S.keyframes[a], B = S.keyframes[b];
      for (const K of [A, B])
        if (!K.feat) K.feat = await S.matcher.describeKeypoints({ width: K.w, height: K.h }, K.xy, K.desc);
      const m = await S.matcher.match(A.feat, A.xy, B.feat, B.xy, FINISH[S.matcherKind]);
      if (m.n >= MIN_FINISH_MATCHES) pairs.push({ a, b, matches: m.matches });
    }
    setProgress(null);
    setStatus('Finishing from the live map…');
    const r = await call('finish', { pairs });
    setProgress(null);
    if (!r.ok && !r.cancelled) throw new Error(r.error);
    if (r.cancelled) throw new Error('stopped');
    S.result = r;
    await showResult(r);
    setStatus(`Done: ${r.registered} of ${r.images} keyframes, ${int(r.points.length / 3)} points, ${fmt(r.mean_error_px)} px, in ${fmt(r.seconds, 1)} s.`);
  } catch (e) {
    setProgress(null);
    setStatus(e.message === 'stopped' ? 'Finish stopped. The live map is shown.' : 'Finish failed: ' + e.message, e.message !== 'stopped');
  } finally {
    S.finishing = false;
    S.done = true;
    for (const k of S.keyframes) { k.feat?.destroy(); k.feat = null; }
    setButtons();
    renderNumbers();
  }
}

// The finished model, its points coloured by a keyframe that sees them.
async function showResult(r) {
  const n = r.points.length / 3;
  const colours = new Uint8Array(3 * n).fill(128);
  const pixels = new Map();
  for (let p = 0; p < n; p++) {
    const i = r.colourObs[2 * p], k = r.colourObs[2 * p + 1];
    const kf = S.keyframes[i];
    if (!kf) continue;
    if (!pixels.has(i)) {
      const bm = await imageOf(kf);
      const cv = new OffscreenCanvas(kf.w, kf.h);
      const g = cv.getContext('2d', { willReadFrequently: true });
      g.drawImage(bm, 0, 0);
      bm.close();
      pixels.set(i, g.getImageData(0, 0, kf.w, kf.h).data);
    }
    const px = pixels.get(i);
    const x = Math.min(kf.w - 1, Math.max(0, Math.round(kf.xy[2 * k]))), y = Math.min(kf.h - 1, Math.max(0, Math.round(kf.xy[2 * k + 1])));
    const o = 4 * (y * kf.w + x);
    colours.set([px[o], px[o + 1], px[o + 2]], 3 * p);
  }
  r.colours = colours;
  const placed = new Set(r.cameras.map((c) => c.image));
  for (const k of S.keyframes) {
    k.thumb?.classList.toggle('placed', placed.has(k.index));
    k.thumb?.classList.toggle('unplaced', !placed.has(k.index));
  }
  $('empty').hidden = r.cameras.length > 0;
  view.set({ cameras: r.cameras, points: r.points, colours, added: new Set() }, { refit: true });
  selectTab('model');
}

function downloadPly() {
  const r = S.result;
  if (!r) return;
  const n = r.points.length / 3;
  const head = `ply\nformat binary_little_endian 1.0\ncomment just-ba live capture\nelement vertex ${n}\n` +
    'property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n';
  const body = new DataView(new ArrayBuffer(15 * n));
  for (let p = 0; p < n; p++) {
    for (let d = 0; d < 3; d++) body.setFloat32(15 * p + 4 * d, r.points[3 * p + d], true);
    for (let d = 0; d < 3; d++) body.setUint8(15 * p + 12 + d, r.colours[3 * p + d]);
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([head, body.buffer], { type: 'application/octet-stream' }));
  a.download = 'live-capture.ply';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
}

// ---- camera

async function openCamera(deviceId) {
  stopCamera();
  if (params.get('video')) {
    const v = $('video');
    v.src = new URL(params.get('video'), location.href).href;
    v.playbackRate = Number(params.get('rate')) || 0.5;
    await new Promise((r, e) => { v.onloadeddata = r; v.onerror = () => e(new Error('the video did not load')); });
    await v.play();
    S.stream = { getTracks: () => [] };
    v.hidden = false;
    overlay.hidden = false;
    $('placeholder').hidden = true;
    return;
  }
  const video = { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } };
  if (deviceId) video.deviceId = { exact: deviceId };
  else video.facingMode = { ideal: 'environment' };
  S.stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
  S.track = S.stream.getVideoTracks()[0];
  const v = $('video');
  v.srcObject = S.stream;
  await v.play();
  v.hidden = false;
  overlay.hidden = false;
  $('placeholder').hidden = true;
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
  const sel = $('camera');
  const current = S.track.getSettings().deviceId;
  sel.innerHTML = devices.map((d, i) => `<option value="${d.deviceId}">${d.label || `Camera ${i + 1}`}</option>`).join('');
  sel.value = current;
  sel.disabled = devices.length < 2;
}

function stopCamera() {
  S.stream?.getTracks().forEach((t) => t.stop());
  S.stream = null;
  S.wakeLock?.release().catch(() => {});
  S.wakeLock = null;
}

// ---- modules and weights

async function ensureModules() {
  const K = topK();
  if (!S.extractor || S.extractorK !== K) {
    setStatus('Loading the keypoint detector (ALIKED)…');
    const jkp = await import(new URL('jkp/jkp.js', import.meta.url).href);
    S.extractor?.destroy?.();
    S.extractor = await jkp.create({ weightsUrl: new URL('jkp/aliked-n16.jkpw', import.meta.url).href, topK: K });
    S.extractorK = K;
  }
  const kind = matcherKind();
  if (!S.matcher || S.matcherKind !== kind) {
    const jmt = await import(new URL('jmt/jmt.js', import.meta.url).href);
    S.matcher?.destroy?.();
    S.matcher = null;
    let weights;
    if (kind === 'lightglue') {
      weights = await jmt.fetchWeights(new URL('jmt/lightglue-aliked-fp16.jmtw', import.meta.url).href, {
        onProgress: (got, total) => { setStatus(`LightGlue's weights: ${(got / 1e6).toFixed(0)} MB…`); if (total) setProgress(got, total); },
      });
    } else {
      setStatus('Loading FlexGlue\'s weights from this browser…');
      weights = S.weights?.bytes ?? new Uint8Array(await (await storedWeights()).arrayBuffer());
    }
    setProgress(null);
    setStatus('Starting the matcher…');
    S.matcher = await jmt.create({ weights, precision: S.precision });
    S.matcherKind = kind;
  }
}

async function storedWeights() {
  try {
    const dir = await navigator.storage.getDirectory();
    const f = await (await dir.getFileHandle(FLEXGLUE.stored)).getFile();
    return f.size === FLEXGLUE.storedBytes ? f : null;
  } catch {
    return null;
  }
}

async function refreshWeights() {
  const kind = matcherKind();
  $('flexglue').hidden = kind !== 'flexglue';
  $('matcher-hint').textContent = kind === 'flexglue'
    ? 'FlexGlue: the best matches, from your own copy of its weights; heavy on a phone.'
    : 'LightGlue: lighter, served with the page (24 MB); sequential pairs only.';
  if (kind === 'flexglue') {
    const f = S.weights?.bytes ? null : await storedWeights();
    const ready = !!(f || S.weights?.bytes);
    $('weights-status').innerHTML = ready
      ? `<span class="ok">FlexGlue's weights are ready</span>${f ? ` (kept in this browser, ${(f.size / 1e6).toFixed(0)} MB)` : ' (for this visit)'}.`
      : 'FlexGlue needs its weights once.';
    $('weights-get').hidden = ready;
    $('remove').hidden = !f;
    S.weightsReady = ready;
  }
  setButtons();
}

async function useCheckpoint(file) {
  if (!file) return;
  if (!$('accept').checked) { notice('Tick the box to accept the licences first.'); return; }
  if (file.size !== FLEXGLUE.bytes) {
    notice(`${file.name} is ${int(file.size)} bytes; FlexGlue's released flexglue.pth is ${int(FLEXGLUE.bytes)}.`);
    return;
  }
  notice(null);
  S.busy = true;
  setButtons();
  try {
    await navigator.storage.persist?.();
    const w = new Worker(new URL('weights_worker.js', import.meta.url), { type: 'module' });
    const out = await new Promise((resolve, reject) => {
      w.onmessage = ({ data }) => {
        if (data.stage) {
          const label = { check: 'Checking flexglue.pth', read: 'Converting it to half precision', store: 'Keeping it in this browser' }[data.stage];
          setStatus(`${label}… ${data.total ? Math.round(100 * data.done / data.total) + ' %' : ''}`);
          setProgress(data.done, data.total);
          return;
        }
        data.ok ? resolve(data) : reject(new Error(data.error));
      };
      w.onerror = (e) => reject(new Error(e.message || 'the conversion failed'));
      w.postMessage({ file, name: FLEXGLUE.stored, sha256: FLEXGLUE.sha256, keep: true });
    });
    w.terminate();  // its ~1.4 GB go with it
    if (!out.stored) S.weights = { bytes: out.bytes };
    setStatus(out.stored ? 'FlexGlue\'s weights are kept in this browser.' : 'This browser cannot keep files: the weights are ready for this visit.');
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    setProgress(null);
    S.busy = false;
    await refreshWeights();
  }
}

// ---- controls

function setButtons() {
  const capturing = S.running, ready = S.supported && !S.busy && !S.finishing;
  const weightsOk = matcherKind() !== 'flexglue' || S.weightsReady;
  $('start').disabled = !ready || capturing || !weightsOk || S.done;
  $('start').textContent = capturing ? 'Capturing…' : S.stream ? 'Start capture' : 'Start';
  $('shoot').disabled = !capturing;
  $('finish').disabled = !(capturing || (S.finishing && !S.cancelFinish)) || (!S.finishing && S.keyframes.length < 2);
  $('finish').textContent = S.finishing ? 'Stop' : 'Finish';
  $('reset').hidden = !(S.done || (S.keyframes.length && !capturing && !S.finishing));
  $('download').hidden = !S.result;
  for (const i of document.querySelectorAll('input[name=matcher], input[name=keypoints]')) i.disabled = capturing || S.finishing || S.busy;
  $('pick').disabled = S.busy || !$('accept').checked;
}

async function start() {
  if (S.running) return;
  S.busy = true;
  setButtons();
  try {
    if (!S.stream) {
      setStatus('Opening the camera…');
      await openCamera();
    }
    await ensureModules();
    const [w, h] = workingSize();
    setStatus('Starting the map…');
    const backends = S.backends;
    await call('new', {
      width: w, height: h, focal: 1.2 * Math.max(w, h), focalUnknown: true, window: WINDOW, restartAfter: RESTART_AFTER,
      minPairInliers: ONLINE_MIN_PAIR_INLIERS, poses: backends.poses.includes('webgpu') ? 'webgpu' : 'cpu', ba: 'cpu',
    });
    S.session = { width: w, height: h };
    // Like Recon's 800 confident keypoints are for a 1280 x 720 frame.
    S.settings.minKeypoints = Math.max(150, Math.round(800 * Math.min(1, (w * h) / (1280 * 720))));
    S.wakeLock = await navigator.wakeLock?.request('screen').catch(() => null);
    S.running = true;
    S.busy = false;
    setStatus('Capturing: move slowly around the scene. Finish when you have gone round it.');
    setButtons();
    selectTab('camera');
    S.loopDone = loop();
    await S.loopDone;
  } catch (e) {
    S.running = false;
    setStatus((e.name === 'NotAllowedError' ? 'The camera was not allowed: ' : 'Could not start: ') + e.message, true);
  } finally {
    S.busy = false;
    setButtons();
  }
}

async function reset() {
  for (const k of S.keyframes) k.feat?.destroy();
  clearPicker();
  Object.assign(S, {
    keyframes: [], reference: null, before: null, twoBack: false, lastOverlap2: null, held: [], cursor: 0, jobs: [], links: new Set(),
    map: null, lastStep: null, result: null, done: false, session: null,
    picker: new G.Picker(), speed: new G.Speed(), blur: new G.Blur(), stats: { frameMs: 0, mapMs: 0, frames: 0, kpMs: 0, describeMs: 0, matchMs: 0, loopMs: 0 },
  });
  await call('free');
  $('strip').innerHTML = '';
  $('empty').hidden = false;
  view.set(null);
  showChip('', '');
  $('meter').hidden = true;
  notice(null);
  renderNumbers();
  setStatus('Press Start for a new capture.');
  setButtons();
}

function selectTab(t) {
  $('stage').dataset.tab = t;
  document.querySelector(`input[name=tab][value=${t}]`).checked = true;
  if (t === 'model') view.draw();
}

const view = new ModelView($('view'));

$('start').addEventListener('click', start);
$('shoot').addEventListener('click', () => { S.shoot = true; });
$('finish').addEventListener('click', () => {
  if (S.finishing) { S.cancelFinish = true; S.worker.postMessage({ op: 'cancel' }); setStatus('Stopping…'); setButtons(); return; }
  finish();
});
$('reset').addEventListener('click', reset);
$('download').addEventListener('click', downloadPly);
$('camera').addEventListener('change', (e) => { if (!S.running) openCamera(e.target.value).catch((err) => setStatus(err.message, true)); });
for (const i of document.querySelectorAll('input[name=tab]')) i.addEventListener('change', () => selectTab(i.value));
for (const i of document.querySelectorAll('input[name=matcher]')) i.addEventListener('change', refreshWeights);
$('accept').addEventListener('change', setButtons);
$('pick').addEventListener('click', () => $('file').click());
$('file').addEventListener('change', (e) => { useCheckpoint(e.target.files[0]); e.target.value = ''; });
const drop = $('drop');
for (const ev of ['dragenter', 'dragover']) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); });
for (const ev of ['dragleave', 'drop']) drop.addEventListener(ev, () => drop.classList.remove('over'));
drop.addEventListener('drop', (e) => { e.preventDefault(); useCheckpoint(e.dataTransfer.files[0]); });
$('remove').addEventListener('click', async () => {
  try { await (await navigator.storage.getDirectory()).removeEntry(FLEXGLUE.stored); } catch { /* gone already */ }
  S.weights = null;
  if (S.matcherKind === 'flexglue') { S.matcher?.destroy(); S.matcher = null; }
  setStatus('FlexGlue\'s weights were removed from this browser.');
  refreshWeights();
});
document.addEventListener('visibilitychange', async () => {
  if (!document.hidden && S.running && !S.wakeLock) S.wakeLock = await navigator.wakeLock?.request('screen').catch(() => null);
});

// ---- start-up: what this browser can do

async function boot() {
  const missing = [];
  if (!window.isSecureContext) missing.push('a secure (https) page');
  if (typeof WebAssembly.Suspending !== 'function') missing.push('WebAssembly JavaScript Promise Integration (JSPI)');
  let adapter = null;
  try { adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' }); } catch { /* none */ }
  if (!adapter) missing.push('WebGPU');
  if (!navigator.mediaDevices?.getUserMedia) missing.push('camera access');
  if (!('Worker' in window)) missing.push('workers');
  document.querySelector(`input[name=matcher][value=${params.get('matcher') ?? (phone ? 'lightglue' : 'flexglue')}]`).checked = true;
  document.querySelector(`input[name=keypoints][value="${params.get('keypoints') ?? (phone ? 1024 : 2048)}"]`).checked = true;
  if (missing.length) {
    const u = $('unsupported');
    u.hidden = false;
    u.innerHTML = `<strong>Live capture can't run in this browser.</strong> It needs ${missing.join(', ')}. Chrome and Edge ` +
      '137 or later have everything, on a computer and on Android. Safari (iPhone, iPad, Mac) has WebGPU but not JSPI yet.';
    setStatus('');
    S.supported = false;
    setButtons();
    for (const b of document.querySelectorAll('button, input, select')) b.disabled = true;
    return;
  }
  S.supported = true;
  S.precision = adapter.features.has('shader-f16') ? 'fp16' : 'fp32';
  const i = adapter.info ?? {};
  $('gpu-info').textContent = `GPU: ${[i.vendor, i.architecture, i.description].filter(Boolean).join(' / ') || 'WebGPU adapter'}; ` +
    `the matcher runs in ${S.precision === 'fp16' ? 'FP16' : 'FP32'}.`;
  setStatus('Loading the mapper…');
  const { default: createDemo } = await import('./jba_demo.js');
  S.live = await createDemo();
  startWorker();
  S.backends = await call('init');
  await refreshWeights();
  renderNumbers();
  setStatus(matcherKind() === 'flexglue' && !S.weightsReady ? 'FlexGlue needs its weights (below), or choose LightGlue.' : 'Press Start and allow the camera.');
}

if (DEBUG) window.capture = S;
boot().catch((e) => setStatus('Could not load the capture: ' + e.message, true));
