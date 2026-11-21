// just-ba's live capture (D157): the camera's frames, keyframes kept as Like Recon keeps
// them (gates.js), a map grown from the keyframes while capturing (demo::Session in
// capture_worker.js), then Finish, the whole capture reconstructed from that map.
//
// Per camera frame, on this page's thread (select.js's FrameSelector): ALIKED keypoints (just-keypoints; XFeat's with
// LighterGlue), the matcher's features of the frame (just-matching: FlexGlue's dense
// features, or LightGlue's or LighterGlue's of the keypoints), its matches to the last keyframe at the live depth
// (FlexGlue's layer-0 head), checked against a relative pose on the CPU
// (demo_pair_inliers), the overlap and support those give, the blur, and the rules. A
// kept frame is matched to the last few keyframes at the online depth and goes to the
// worker's session. With LightGlue or LighterGlue the worker matches those pairs itself,
// with a matcher of its own (D159), and the camera's loop never waits for the map. With
// FlexGlue they are matched here between camera frames, a share of each, and a keyframe
// the map can't place is matched to frames chosen by appearance (FlexGlue's shortlist) and
// tried again. Finish matches more pairs (the shortlist over every keyframe) and runs
// the mapper from the live map.
//
// URL parameters: matcher=flexglue|lightglue|lighterglue, keypoints=1024|2048, side=<px> (the
// working image's longer side), debug=1 (timings in the console), video=<url> (a recorded
// video instead of the camera, for testing; rate=<playback rate>, 0.5 by default). The
// Camera panel takes a video file the same way.
import * as G from './gates.js';
import { FIXED, FrameSelector, minKeypointsFor } from './select.js';
import { ModelView } from './view.js';
import * as FW from './flexglue_weights.js';

const params = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);
const DEBUG = params.get('debug') === '1';

// Like Recon's numbers (src-tauri/src/capture.rs, online.rs). The live depths are
// select.js's; the map's: on a Pixel 9a at 640 px, LightGlue's 2 layers take 160 ms (342
// matches), 3 take 215 ms (481), 5 take 330 ms, adaptive 9 about 510 ms.
const ONLINE = { flexglue: { layers: 3 }, lightglue: { layers: 3, ...FIXED }, lighterglue: { layers: 3, ...FIXED } };
// Finish's pairs: FlexGlue as the map's; LightGlue and LighterGlue at full, adaptive depth (they have the time; the
// matcher's own adaptivity: LightGlue 0.95 and 0.99, LighterGlue all six layers and width 0.95).
const FINISH = { flexglue: { layers: 3 }, lightglue: { layers: 9 }, lighterglue: { layers: 9 } };
// LightGlue and LighterGlue match keypoints with their descriptors: ALIKED's, XFeat's.
const sparse = (kind) => kind === 'lightglue' || kind === 'lighterglue';
const SPARSE_WEIGHTS = { lightglue: 'jmt/lightglue-aliked-fp16.jmtw', lighterglue: 'jmt/lightglue-xfeat.jmtw' };
const weightsUrl = (kind) => new URL(SPARSE_WEIGHTS[kind], import.meta.url).href;
const keypointModel = (kind) => (kind === 'lighterglue' ? 'xfeat' : 'aliked');
const MATCHER_NAME = { flexglue: 'FlexGlue', lightglue: 'LightGlue', lighterglue: 'LighterGlue' };
const SEQUENTIAL = 3;            // keyframes each new one is matched to
const MATCHES_PER_PAIR = 1024;   // the strongest, per online pair
const WINDOW = 8;                // keyframes adjusted together
const RESTART_AFTER = 10;        // keyframes in a row not placed before a new piece
const ONLINE_MIN_PAIR_INLIERS = 15;
const APPEARANCE_PARTNERS = 4;   // more keyframes for one the map can't place (FlexGlue)
const FINISH_SHORTLIST = 10;     // pairs per keyframe for the finish (FlexGlue's shortlist)
const FINISH_SEQUENTIAL = 5;     // LightGlue's and LighterGlue's finish: each keyframe with the five before it
const MIN_FINISH_MATCHES = 15;
// FlexGlue's finish from the live map placing fewer than this share of the keyframes is run again from scratch.
const FINISH_FROM_SCRATCH = 0.8;

// A phone: LighterGlue by default, with XFeat keypoints (FlexGlue's backbone is ~1 GB of GPU
// memory and takes ~2 s a frame on a Pixel 9a; LightGlue was the default before: ALIKED 140 ms
// there, a 2-layer match 158 ms; XFeat is about a third of ALIKED's time and LighterGlue about
// half of LightGlue's on an M5 Pro, not yet measured on a phone), a 640 px working image and
// 1,024 keypoints. 480 px and 768 keypoints save ~80 ms a check but leave
// the map too few points: on the test video both placed 6 of ~40, so the speed comes from
// the checks skipped and the overlap anticipated instead.
const phone = matchMedia('(pointer: coarse)').matches && Math.min(screen.width, screen.height) < 820;
const MAX_SIDE = Number(params.get('side')) || (phone ? 640 : 1280);
const FEATURE_CACHE = phone ? 24 : 48;
// Held keyframes tried a frame while lost: each is a live match (~160 ms on a Pixel 9a).
const RELOCALIZE_PER_FRAME = phone ? 1 : G.RELOCALIZE_PER_FRAME;

const S = {
  gpu: null, precision: 'fp32',
  live: null,          // the demo's module on this thread, for demo_pair_inliers
  worker: null, calls: new Map(), nextId: 1,
  extractor: null, extractorKey: '', kpModel: 'aliked',   // the detector, and its keypoints' model
  matcher: null, matcherKind: null,
  weights: null,       // FlexGlue: { stored } or { bytes } (this visit only)
  stream: null, track: null, wakeLock: null,
  running: false, finishing: false, shoot: false, busy: false,
  session: null,       // { width, height }
  keyframes: [],       // { index, seq, xy, desc, feat, w, h, jpeg, placed, shoot, thumb }
  keyframesModel: null, // the keypoints' model of this capture's keyframes ('aliked' or 'xfeat')
  selector: null,      // the keep loop (select.js): the reference, the held keyframes, the picker
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
  const r = await S.selector.step(frame.image, { shoot });
  if (r.kept) await keep(r.kept, r.keptImage);
  showGate(r.gate, r.why, r.overlap, r.slowDown, r.relocalized);
  const ms = performance.now() - t0;
  const avg = (old, v) => (S.stats.frames ? 0.8 * old + 0.2 * v : v);
  S.stats.frameMs = avg(S.stats.frameMs, ms);
  S.stats.kpMs = avg(S.stats.kpMs, r.ms.keypoints);
  S.stats.describeMs = avg(S.stats.describeMs, r.ms.describe);
  S.stats.matchMs = avg(S.stats.matchMs, r.ms.match);
  S.stats.frames++;
  if (DEBUG) console.log(`frame ${ms.toFixed(0)} ms, ${r.keypoints} kp (${r.confident} sure), overlap ${fmt(r.overlap)}, ${r.gate}`);
}

// A frame the selector kept (its keyframe, already the reference for the next frames when
// it can anchor the chain) becomes the map's next image.
async function keep(kf, image) {
  Object.assign(kf, { placed: false, jpeg: null, keptAt: performance.now() });
  S.keyframes.push(kf);
  S.keyframesModel = S.kpModel;
  // A small JPEG of it, for its thumbnail, the points' colours and the finish's features.
  const cv = new OffscreenCanvas(kf.w, kf.h);
  cv.getContext('2d').putImageData(image, 0, 0);
  kf.jpeg = await cv.convertToBlob({ type: 'image/jpeg', quality: 0.9 });
  addThumb(kf, cv);
  showLastKeyframe(cv);
  const partners = S.keyframes.slice(Math.max(0, kf.index - SEQUENTIAL), kf.index);
  if (kf.relocalizedTo && !partners.includes(kf.relocalizedTo)) partners.push(kf.relocalizedTo);
  if (sparse(S.matcherKind)) {
    // The worker matches the pairs and places the keyframe (D159); the answer is taken in
    // between camera frames (onlineStep), in order.
    const job = { kf, partners: [], pairs: [], stage: 'added' };
    const sent = (k) => ({ index: k.index, xy: k.xy, desc: k.desc, width: k.w, height: k.h });
    call('place', { kf: sent(kf), partners: partners.map(sent), options: ONLINE[S.matcherKind], maxMatches: MATCHES_PER_PAIR })
      .then((r) => { job.answer = r; job.mapMs = r.ms; job.matchMs = r.matchMs; }, (e) => { job.failed = e; });
    S.jobs.push(job);
  } else {
    S.jobs.push({ kf, partners, pairs: [], stage: 'match', done: new Set(), tried: new Set(partners.map((p) => p.index)) });
  }
  freeFeatures();
  setButtons();
}

// Features stay for the keyframes the live loop measures against, the latest ones (the next
// keyframes' partners) and, thinned, older ones for the appearance shortlist.
function freeFeatures() {
  const withFeat = S.keyframes.filter((k) => k.feat);
  const keepSet = new Set(S.selector?.measured() ?? []);
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
// it is placed or tried again. Returns 'gpu' after work on the GPU (a pair matched, the
// appearance shortlist), 'cpu' after the rest (a keyframe sent, an answer taken in), and
// false when there is nothing to do now, or when the next piece needs the GPU and `gpu`
// is false.
async function onlineStep(gpu = true) {
  const head = S.jobs[0];
  if (!head) return false;
  if (head.failed) {
    S.jobs.shift();
    notice('The map could not take a keyframe: ' + head.failed.message);
    return 'cpu';
  }
  if (head.answer) {
    if (!gpu && wantsAppearance(head)) return false;
    return takeAnswer(head);
  }
  for (const job of S.jobs) {
    if (job.stage !== 'match' && job.stage !== 'appearance') continue;
    const todo = job.stage === 'match' ? job.partners : job.more;
    const p = todo.find((q) => !job.done.has(q.index));
    if (p) {
      if (!(p.feat && job.kf.feat)) {
        job.done.add(p.index);
        return 'cpu';
      }
      if (!gpu) return false;
      job.done.add(p.index);
      const m = await S.matcher.match(p.feat, p.xy, job.kf.feat, job.kf.xy, ONLINE[S.matcherKind]);
      const matches = G.strongest(m.matches, m.scores, MATCHES_PER_PAIR);
      (job.stage === 'match' ? job.pairs : job.morePairs).push({ earlier: p.index, matches });
      S.links.add(`${p.index}-${job.kf.index}`);
      return 'gpu';
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
    return 'cpu';
  }
  return false;  // the worker is at it, and every pair is matched
}

// A keyframe the map didn't place, to be tried again with partners chosen by appearance
// (FlexGlue's shortlist, on the GPU).
function wantsAppearance(job) {
  const { step, map } = job.answer;
  return job.stage === 'added' && !step.placed?.includes(job.kf.index) && S.matcherKind === 'flexglue' && map.started;
}

async function takeAnswer(job) {
  const { step, map } = job.answer;
  const again = wantsAppearance(job);
  if (DEBUG) {
    // The worker's pairs come as [earlier, matches]; the page's as { earlier, matches }.
    const pairs = job.answer.pairs ?? (job.stage === 'added' ? job.pairs : job.morePairs).map((p) => [p.earlier, p.matches.length / 2]);
    const matched = job.matchMs != null ? `, its pairs ${job.matchMs.toFixed(0)} ms of it` : '';
    console.log(`keyframe ${job.kf.index} ${job.stage} in ${job.mapMs.toFixed(0)} ms${matched} ` +
      `(${(performance.now() - job.kf.keptAt).toFixed(0)} ms after it was kept, ${S.jobs.length - 1} more queued): ` +
      `${step.ok ? '' : step.error + ' '}placed [${step.placed}], pairs ${pairs.map(([a, n]) => `${a}:${n}`).join(' ')}`);
  }
  job.answer = null;
  takeMap(step, map, job.mapMs);
  if (again) {
    job.more = await appearancePartners(job);
    job.morePairs = [];
    job.stage = 'appearance';
    return 'gpu';
  }
  S.jobs.shift();
  freeFeatures();
  return 'cpu';
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
  // A new piece (or the current one started again) is a new frame: the view fits afresh.
  const fresh = step?.started || step?.rejoined || step?.piece !== S.viewPiece;
  S.viewPiece = step?.piece;
  view.set({ cameras: map.cameras, points: map.points, colours: null, added: new Set(step?.placed ?? []) }, { refit: fresh });
  renderNumbers();
}

// FlexGlue's map: its share of each camera frame, more when it falls behind. The desktop
// app matches the map's pairs on a thread of its own; FlexGlue's here share the GPU module
// with the camera (LightGlue's and LighterGlue's are matched in the worker, D159, and
// this loop only takes the worker's answers in). Only the GPU's work counts against it: sending a keyframe to the worker and taking its
// answer in go through every frame. On a phone a map pair alone takes ~200 ms: one a frame
// while the map keeps up, two once more than MAP_BEHIND_JOBS keyframes wait. A keyframe
// needs three pairs and the camera keeps one every 2-3 frames; at one pair a frame the
// queue grew all through a capture (11 keyframes, 15 s behind on the test video).
const MAP_BUDGET_MS = phone ? 0 : 120, MAP_BEHIND_MS = phone ? 0 : 400;
const MAP_PAIRS = 1, MAP_BEHIND_PAIRS = phone ? 2 : 1, MAP_BEHIND_JOBS = 2;

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
      const behind = S.jobs.length > MAP_BEHIND_JOBS;
      const budget = !fresh ? Math.max(MAP_BEHIND_MS, 400) : behind ? MAP_BEHIND_MS : MAP_BUDGET_MS;
      const least = behind ? MAP_BEHIND_PAIRS : MAP_PAIRS;
      // At least `least` pairs a frame, then more while the budget lasts; the CPU's pieces
      // whenever they come.
      for (let pairs = 0; ;) {
        const did = await onlineStep(pairs < least || performance.now() - t0 < budget);
        if (!did) break;
        if (did === 'gpu') pairs++;
      }
      if (!fresh && !S.jobs.length) await new Promise((r) => setTimeout(r, 100));
      if (fresh) S.stats.loopMs = S.stats.frames > 1 ? 0.8 * S.stats.loopMs + 0.2 * (performance.now() - tLoop) : performance.now() - tLoop;
    } catch (e) {
      console.error(e);
      setStatus('Something went wrong: ' + e.message + '. Finish what was kept, or start a new capture.', true);
      S.running = false;
      keepAwake(false);
      setButtons();
      break;
    }
    await nextFrame();
  }
}

// ---- what the view shows: a pill for the tracking, the last keyframe in the corner

const GATES = {
  kept: 'Keyframe', waiting: 'Tracking', too_few_keypoints: 'Too little texture here', blurry: 'Blurry: move slower',
  lost: 'Lost: point back at the last keyframe', paused: 'Paused',
};
function showGate(gate, why, overlap, slow, relocalized) {
  let text = GATES[gate] ?? gate, kind = gate === 'kept' || gate === 'waiting' ? 'ok' : 'warn';
  if (gate === 'kept') text = `Keyframe ${S.keyframes.length}`;
  else if (gate === 'waiting' && slow) { text = 'Slow down'; kind = 'warn'; }
  if (relocalized) { text = 'Found the map again'; kind = 'ok'; }
  showChip(text, kind);
}
function showChip(text, kind) {
  const c = $('chip');
  c.hidden = !text;
  c.textContent = text;
  c.className = 'chip' + (kind ? ' ' + kind : '');
}

// The corner: the last keyframe, flashed when one is kept; the live camera inset in it.
function showLastKeyframe(cv) {
  const k = $('lastkf');
  k.width = cv.width;
  k.height = cv.height;
  k.getContext('2d').drawImage(cv, 0, 0);
  k.hidden = false;
  const pip = $('pip');
  pip.classList.add('has-kf', 'flash');
  setTimeout(() => pip.classList.remove('flash'), 400);
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
  // On the map: keyframes kept, placed, points.
  const placed = r ? r.registered : m ? m.cameras.length : 0;
  const pts = r ? r.points.length / 3 : m ? m.points.length / 3 : 0;
  $('stats').textContent = S.keyframes.length ? `${S.keyframes.length} kept · ${placed} placed · ${int(pts)} points` : '';
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
  await keepAwake(true);
  try {
    await S.loopDone;
    stopCamera();
    showChip('', '');
    // The map's queue first: every keyframe placed or tried.
    while (S.jobs.length) {
      setStatus(`Adding the last keyframes to the map (${S.jobs.length} left)…`);
      if (!(await onlineStep())) await new Promise((r) => setTimeout(r, 30));
    }
    const n = S.keyframes.length;
    if (n < 2) throw new Error('fewer than two keyframes');
    // The live features go, the Picker's held frames too (before the matcher that made them):
    // the pairs are matched with features made afresh (matchKeyframes).
    S.selector?.clearPicker();
    for (const k of S.keyframes) { k.feat?.destroy(); k.feat = null; }
    let r;
    if (sparse(S.matcherKind)) {
      // LightGlue (and LighterGlue): every pair matched again at full depth and the mapper from scratch. The live
      // map's pairs (3 layers) hold the result back: on the test video the same 39 keyframes
      // gave 17 registered from 3-layer pairs and 36 from full-depth ones.
      const pairs = await matchKeyframes(S.matcherKind, new Set());
      releaseModules();
      setStatus('Reconstructing…');
      r = await call('reconstruct', reconstructArgs(pairs));
      setProgress(null);
      if (r.cancelled) throw new Error('stopped');
      if (!r.ok) throw new Error(r.error);
    } else {
      const pairs = await matchKeyframes(S.matcherKind, S.links);
      releaseModules();
      setStatus('Finishing from the live map…');
      r = await call('finish', { pairs });
      setProgress(null);
      if (!r.ok && !r.cancelled) throw new Error(r.error);
      if (r.cancelled) throw new Error('stopped');
      // A live map in pieces can hold the finish back: under FINISH_FROM_SCRATCH of the
      // keyframes placed, again from scratch with the same pairs, the better kept.
      if (r.registered < FINISH_FROM_SCRATCH * r.images) {
        setStatus(`The live map placed ${r.registered} of ${r.images}: reconstructing from scratch…`);
        const again = await call('finish', { pairs, fromScratch: true });
        setProgress(null);
        if (again.cancelled) throw new Error('stopped');
        if (again.ok && again.registered > r.registered) r = again;
      }
    }
    S.result = r;
    await showResult(r);
    setStatus(`Done: ${r.registered} of ${r.images} keyframes, ${int(r.points.length / 3)} points, ${fmt(r.mean_error_px)} px, in ${fmt(r.seconds, 1)} s.`);
  } catch (e) {
    setProgress(null);
    setStatus(e.message === 'stopped' ? 'Finish stopped. The live map is shown.' : 'Finish failed: ' + e.message, e.message !== 'stopped');
  } finally {
    S.finishing = false;
    S.done = true;
    keepAwake(false);
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
}

// The worker's reconstruct op for the keyframes and `pairs`: from scratch, the focal length searched for.
function reconstructArgs(pairs) {
  const { w, h } = S.keyframes[0];
  return { width: w, height: h, focal: 1.2 * Math.max(w, h), xy: S.keyframes.map((k) => k.xy), pairs,
           poses: S.backends.poses.includes('webgpu') ? 'webgpu' : 'cpu' };
}

// The keypoint detector and the matcher let go before the mapper's long adjustments: phones
// share their memory with the GPU, and a Pixel 9a under memory pressure had its low-memory
// killer take the page in the middle of a finish (the modules' own heaps were under 200 MB).
// ensureModules makes them again for Reconstruct again or a new capture.
function releaseModules() {
  S.extractor?.destroy?.();
  S.extractor = null;
  S.extractorK = 0;
  S.matcher?.destroy?.();
  S.matcher = null;
  S.matcherKind = null;
  // The worker's too (D159), before its adjustments: its calls run in order.
  call('dropMatcher').catch(() => {});
}

// The pairs of keyframes for the finish and Reconstruct again, matched with `kind`, leaving
// out those in `skip` ("a-b"). FlexGlue on a computer: every keyframe described, the
// shortlist over all of them and the three before each. Otherwise (LightGlue, LighterGlue, or a phone):
// each keyframe with the five before it, described as they come and only those six held:
// on a Pixel 9a, FlexGlue's features of every keyframe at once and the shortlist over them
// crashed the browser's GPU process.
async function matchKeyframes(kind, skip) {
  const kfs = S.keyframes, n = kfs.length;
  const describeOne = async (k) => {
    if (sparse(kind)) return S.matcher.describeKeypoints({ width: k.w, height: k.h }, k.xy, k.desc);
    const bm = await createImageBitmap(k.jpeg);
    try { return await S.matcher.describe(bm); } finally { bm.close(); }
  };
  const feats = new Map();
  const pairs = [];
  const match = async (a, b) => {
    if (skip.has(`${a}-${b}`)) return;
    const m = await S.matcher.match(feats.get(a), kfs[a].xy, feats.get(b), kfs[b].xy, FINISH[kind]);
    if (m.n >= MIN_FINISH_MATCHES) pairs.push({ a, b, matches: m.matches });
  };
  try {
    if (kind === 'flexglue' && !phone) {
      for (let i = 0; i < n; i++) {
        if (S.cancelFinish) throw new Error('stopped');
        setStatus(`Describing keyframe ${i + 1} of ${n}…`);
        setProgress(i, n);
        feats.set(i, await describeOne(kfs[i]));
      }
      const wanted = new Set();
      const add = (a, b) => wanted.add(`${Math.min(a, b)}-${Math.max(a, b)}`);
      for (let j = 1; j < n; j++) for (let i = Math.max(0, j - SEQUENTIAL); i < j; i++) add(i, j);
      setStatus('Choosing the pairs to match…');
      const list = await S.matcher.shortlist([...feats.values()], { k: FINISH_SHORTLIST });
      for (let p = 0; p < list.n; p++) add(list.pairs[2 * p], list.pairs[2 * p + 1]);
      const todo = [...wanted].map((x) => x.split('-').map(Number));
      for (const [p, [a, b]] of todo.entries()) {
        if (S.cancelFinish) throw new Error('stopped');
        if (p % 2 === 0) { setStatus(`Matching pair ${p + 1} of ${todo.length}…`); setProgress(p, todo.length); }
        await match(a, b);
      }
    } else {
      for (let j = 0; j < n; j++) {
        if (S.cancelFinish) throw new Error('stopped');
        setStatus(`Matching keyframe ${j + 1} of ${n} with the ones before it…`);
        setProgress(j, n);
        feats.set(j, await describeOne(kfs[j]));
        for (let i = Math.max(0, j - FINISH_SEQUENTIAL); i < j; i++) await match(i, j);
        const old = j - FINISH_SEQUENTIAL;
        if (feats.has(old)) { feats.get(old).destroy(); feats.delete(old); }
      }
    }
  } finally {
    for (const f of feats.values()) f.destroy();
    setProgress(null);
  }
  return pairs;
}

// Reconstruct again: the keyframes matched again by the matcher chosen now
// (matchKeyframes), then the mapper from scratch. Needs neither the camera nor the live map.
async function reconstructAgain() {
  S.finishing = true;
  S.cancelFinish = false;
  setButtons();
  await keepAwake(true);
  try {
    const kind = matcherKind();
    if (sparse(kind) && S.keyframesModel && keypointModel(kind) !== S.keyframesModel) {
      throw new Error(`${MATCHER_NAME[kind]} matches ${keypointModel(kind) === 'xfeat' ? 'XFeat' : 'ALIKED'} keypoints, ` +
                      `and this capture has ${S.keyframesModel === 'xfeat' ? 'XFeat' : 'ALIKED'}'s: choose ` +
                      `${S.keyframesModel === 'xfeat' ? 'LighterGlue' : 'LightGlue'} or FlexGlue`);
    }
    await ensureModules(S.keyframesModel ?? keypointModel(kind));
    const pairs = await matchKeyframes(kind, new Set());
    releaseModules();
    setProgress(null);
    setStatus('Reconstructing…');
    const r = await call('reconstruct', reconstructArgs(pairs));
    if (r.cancelled) throw new Error('stopped');
    if (!r.ok) throw new Error(r.error);
    S.result = r;
    await showResult(r);
    setStatus(`Done with ${MATCHER_NAME[kind]}: ${r.registered} of ${r.images} keyframes, ` +
              `${int(r.points.length / 3)} points, ${fmt(r.mean_error_px)} px.`);
  } catch (e) {
    setProgress(null);
    setStatus(e.message === 'stopped' ? 'Stopped. The last model is shown.' : 'Reconstructing failed: ' + e.message,
              e.message !== 'stopped');
  } finally {
    S.finishing = false;
    keepAwake(false);
    setButtons();
    renderNumbers();
  }
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
  // A recorded video instead: the visitor's file (the Camera panel), or video=<url>.
  const recorded = S.videoFile ?? (params.get('video') ? new URL(params.get('video'), location.href).href : null);
  if (recorded) {
    const v = $('video');
    v.srcObject = null;
    v.src = recorded;
    await new Promise((r, e) => { v.onloadeddata = r; v.onerror = () => e(new Error('the video did not load')); });
    // Its first frame, waiting: it plays once the capture is ready for frames (start), so none of it is missed.
    v.pause();
    S.recorded = true;
    S.stream = { getTracks: () => [] };
    v.hidden = false;
    $('placeholder').hidden = true;
    return;
  }
  S.recorded = false;
  const video = { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } };
  if (deviceId) video.deviceId = { exact: deviceId };
  else video.facingMode = { ideal: 'environment' };
  S.stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
  S.track = S.stream.getVideoTracks()[0];
  const v = $('video');
  v.srcObject = S.stream;
  await v.play();
  v.hidden = false;
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
}

// The screen stays on while the page works (capturing, the finish, Reconstruct again): a
// phone whose screen sleeps sends the browser to the background, where Android reclaimed
// its GPU process in the middle of a finish (a Pixel 9a).
async function keepAwake(on) {
  if (on && !S.wakeLock) S.wakeLock = await navigator.wakeLock?.request('screen').catch(() => null);
  if (!on) {
    S.wakeLock?.release().catch(() => {});
    S.wakeLock = null;
  }
}

// ---- modules and weights

// The detector follows the matcher (XFeat for LighterGlue, else ALIKED); `model` overrides it (Reconstruct again
// keeps the capture's keypoints).
async function ensureModules(model = keypointModel(matcherKind())) {
  const K = topK();
  if (!S.extractor || S.extractorKey !== `${model}/${K}`) {
    setStatus(`Loading the keypoint detector (${model === 'xfeat' ? 'XFeat' : 'ALIKED'})…`);
    const jkp = await import(new URL('jkp/jkp.js', import.meta.url).href);
    S.extractor?.destroy?.();
    S.extractor = null;
    const weightsUrl = new URL(model === 'xfeat' ? 'jkp/xfeat.jkpw' : 'jkp/aliked-n16.jkpw', import.meta.url).href;
    S.extractor = await jkp.create({ weightsUrl, topK: K });
    S.extractorKey = `${model}/${K}`;
    S.kpModel = model;
  }
  const kind = matcherKind();
  if (!S.matcher || S.matcherKind !== kind) {
    const jmt = await import(new URL('jmt/jmt.js', import.meta.url).href);
    S.matcher?.destroy?.();
    S.matcher = null;
    let weights;
    if (sparse(kind)) {
      weights = await jmt.fetchWeights(weightsUrl(kind), {
        onProgress: (got, total) => { setStatus(`${MATCHER_NAME[kind]}'s weights: ${(got / 1e6).toFixed(0)} MB…`); if (total) setProgress(got, total); },
      });
    } else {
      setStatus('Loading FlexGlue\'s weights from this browser…');
      weights = S.weights?.bytes ?? await FW.load();
      if (!weights) throw new Error('FlexGlue\'s weights are gone from this browser: give flexglue.pth again');
    }
    setProgress(null);
    setStatus('Starting the matcher…');
    S.matcher = await jmt.create({ weights, precision: S.precision });
    S.matcherKind = kind;
  }
}

async function refreshWeights() {
  const kind = matcherKind();
  $('flexglue').hidden = kind !== 'flexglue';
  $('matcher-hint').textContent = kind === 'flexglue'
    ? 'FlexGlue: the best matches, from your own copy of its weights; heavy on a phone.'
    : kind === 'lighterglue'
    ? 'LighterGlue: XFeat keypoints and the lightest matcher, served with the page (4.5 MB); sequential pairs only.'
    : 'LightGlue: lighter, served with the page (24 MB); sequential pairs only.';
  if (kind === 'flexglue') {
    const f = S.weights?.bytes ? null : await FW.stored();
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
  notice(null);
  S.busy = true;
  setButtons();
  try {
    const out = await FW.convert(file, (data) => {
      setStatus(`${FW.STAGES[data.stage]}… ${data.total ? Math.round(100 * data.done / data.total) + ' %' : ''}`);
      setProgress(data.done, data.total);
    });
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
  // A capture that stopped (finished, or the loop failed) is finished or put away with New capture first.
  $('start').disabled = !ready || capturing || !weightsOk || S.done || S.keyframes.length > 0;
  $('start').textContent = capturing ? 'Capturing…' : S.stream ? 'Start capture' : 'Start';
  $('shoot').disabled = !capturing;
  const stopped = !capturing && !S.finishing && !S.done && S.keyframes.length >= 2;  // the loop failed
  $('finish').disabled = !(capturing || stopped || (S.finishing && !S.cancelFinish)) || (!S.finishing && S.keyframes.length < 2);
  $('finish').textContent = S.finishing ? 'Stop' : 'Finish';
  $('reset').hidden = !(S.done || (S.keyframes.length && !capturing)) || S.finishing;
  // After a capture: match the keyframes again with the matcher chosen, and reconstruct.
  $('again').hidden = !(S.done && S.keyframes.length >= 2) || S.finishing;
  $('again').disabled = S.busy || (matcherKind() === 'flexglue' && !S.weightsReady);
  $('download').hidden = !S.result;
  for (const i of document.querySelectorAll('input[name=matcher], input[name=keypoints]')) i.disabled = capturing || S.finishing || S.busy;
  $('pick').disabled = S.busy || !$('accept').checked;
  $('video-pick').disabled = $('video-clear').disabled = capturing || S.finishing || S.busy || S.done || S.keyframes.length > 0;
  $('video-clear').hidden = !S.videoFile;
  // The weights can't go while the matcher made from them is at work.
  $('remove').disabled = capturing || S.finishing || S.busy;
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
    if (sparse(S.matcherKind)) {
      setStatus('Starting the map\'s matcher…');
      await call('matcher', { url: weightsUrl(S.matcherKind), precision: S.precision });
    }
    const [w, h] = workingSize();
    S.selector?.dispose();
    S.selector = new FrameSelector({
      extractor: S.extractor, matcher: S.matcher, matcherKind: S.matcherKind, kpModel: S.kpModel, module: S.live,
      settings: S.settings, floor: S.floor, phone, anticipate: true, relocalizePerFrame: RELOCALIZE_PER_FRAME,
    });
    setStatus('Starting the map…');
    const backends = S.backends;
    await call('new', {
      width: w, height: h, focal: 1.2 * Math.max(w, h), focalUnknown: true, window: WINDOW, restartAfter: RESTART_AFTER,
      minPairInliers: ONLINE_MIN_PAIR_INLIERS, poses: backends.poses.includes('webgpu') ? 'webgpu' : 'cpu', ba: 'cpu',
    });
    S.session = { width: w, height: h };
    S.settings.minKeypoints = minKeypointsFor(w, h);
    // A phone checks a few frames a second: at walking pace the overlap falls ~0.2 between
    // checks. Before the overlap was anticipated (G.anticipate), keyframes landed far past
    // the target and pairs two apart had 4-42 matches; a slightly higher target leaves
    // overlap two back.
    if (phone) S.settings.targetOverlap = 0.6;
    await keepAwake(true);
    S.running = true;
    S.busy = false;
    setStatus('Capturing. Move slowly; Finish when done.');
    setButtons();
    if (S.recorded) {
      $('video').playbackRate = Number(params.get('rate')) || 0.5;
      await $('video').play();
    }
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
  S.selector?.dispose();
  Object.assign(S, {
    keyframes: [], keyframesModel: null, selector: null, jobs: [], links: new Set(),
    map: null, lastStep: null, result: null, done: false, session: null,
    stats: { frameMs: 0, mapMs: 0, frames: 0, kpMs: 0, describeMs: 0, matchMs: 0, loopMs: 0 },
  });
  await call('free');
  $('strip').innerHTML = '';
  $('empty').hidden = false;
  view.set(null);
  showChip('', '');
  $('lastkf').hidden = true;
  $('pip').classList.remove('has-kf');
  notice(null);
  renderNumbers();
  setStatus('Press Start for a new capture.');
  setButtons();
}

const view = new ModelView($('view'));

$('start').addEventListener('click', start);
$('shoot').addEventListener('click', () => { S.shoot = true; });
$('finish').addEventListener('click', () => {
  if (S.finishing) { S.cancelFinish = true; S.worker.postMessage({ op: 'cancel' }); setStatus('Stopping…'); setButtons(); return; }
  finish();
});
$('reset').addEventListener('click', reset);
$('again').addEventListener('click', reconstructAgain);
$('download').addEventListener('click', downloadPly);
// From a picked video file back to the camera (opened at the next Start, or now when one is chosen).
function backToCamera() {
  if (!S.videoFile) return;
  URL.revokeObjectURL(S.videoFile);
  S.videoFile = null;
  stopCamera();
  $('video').removeAttribute('src');
  $('video').load();
  $('video-name').textContent = 'A recorded video instead of the camera, played at half speed.';
  setButtons();
}
$('camera').addEventListener('change', (e) => {
  if (S.running) return;
  backToCamera();
  openCamera(e.target.value).catch((err) => setStatus(err.message, true));
});
$('video-clear').addEventListener('click', () => { if (!S.running && !S.finishing) backToCamera(); });
// A video file as the camera: played at rate= (0.5 by default) from Start, as video=<url> is.
$('video-pick').addEventListener('click', () => $('video-file').click());
$('video-file').addEventListener('change', (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file || S.running || S.finishing) return;
  if (S.videoFile) URL.revokeObjectURL(S.videoFile);
  S.videoFile = URL.createObjectURL(file);
  stopCamera();
  $('video-name').textContent = `${file.name}: played as the camera from Start.`;
  setButtons();
});
for (const i of document.querySelectorAll('input[name=matcher]')) i.addEventListener('change', refreshWeights);
$('accept').addEventListener('change', setButtons);
$('pick').addEventListener('click', () => $('file').click());
$('file').addEventListener('change', (e) => { useCheckpoint(e.target.files[0]); e.target.value = ''; });
const drop = $('drop');
for (const ev of ['dragenter', 'dragover']) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); });
for (const ev of ['dragleave', 'drop']) drop.addEventListener(ev, () => drop.classList.remove('over'));
drop.addEventListener('drop', (e) => { e.preventDefault(); useCheckpoint(e.dataTransfer.files[0]); });
$('remove').addEventListener('click', async () => {
  await FW.remove();
  S.weights = null;
  if (S.matcherKind === 'flexglue') { S.matcher?.destroy(); S.matcher = null; }
  setStatus('FlexGlue\'s weights were removed from this browser.');
  refreshWeights();
});
// A wake lock goes when the page is hidden; it comes back with the page while work goes on.
document.addEventListener('visibilitychange', () => {
  if (!document.hidden && (S.running || S.finishing)) { S.wakeLock = null; keepAwake(true); }
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
  document.querySelector(`input[name=matcher][value=${params.get('matcher') ?? (phone ? 'lighterglue' : 'flexglue')}]`).checked = true;
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

if (DEBUG) {
  window.capture = S;
  // Memory of each WebAssembly module, MB: this page's mapper, the keypoints, the matcher, the worker's mapper.
  window.captureMemory = async () => {
    const mb = (b) => Math.round(b / 1e6);
    const jkp = await import(new URL('jkp/jkp.js', import.meta.url).href).catch(() => null);
    const jmt = await import(new URL('jmt/jmt.js', import.meta.url).href).catch(() => null);
    return {
      live: mb(S.live?.HEAPU8.length ?? 0),
      keypoints: mb((await jkp?.module?.())?.HEAPU8?.length ?? 0),
      matcher: mb((await jmt?.module?.())?.HEAPU8?.length ?? 0),
      worker: mb((await call('mem')).bytes),
      jsHeap: performance.memory ? mb(performance.memory.usedJSHeapSize) : null,
    };
  };
}
boot().catch((e) => setStatus('Could not load the capture: ' + e.message, true));
