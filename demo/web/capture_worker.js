// The live capture's map (D157): the demo's module in a worker, holding one
// demo::Session (D153), so that placing a keyframe, adjusting the window and the finish
// never hold up the camera's loop on the page. With LightGlue or LighterGlue the map's
// pairs are matched here too (D159), by a matcher of the worker's own (just-matching, its
// own GPU device): the page sends a keyframe's keypoints and descriptors and goes on with
// the camera. FlexGlue's are matched on the page (a second FlexGlue is ~1 GB of GPU memory).
// The page posts { id, op, ... } and gets { id, ok, result } back (or { id, ok: false, error });
// during a finish it also gets { progress } messages, and { op: 'cancel' } stops the finish
// at its next report.
//
// Ops:
//   init { }                               -> { poses: [...], ba: [...] } backends
//   new { width, height, focal, focalUnknown, window, restartAfter, minPairInliers, poses, ba }
//   add { xy: Float32Array, pairs: [{ earlier, matches: Int32Array }], deferSplit }  -> step + map
//   retry { pairs, newPiece }              -> step + map
//   matcher { url, precision }             -> the worker's matcher, from the weights at url
//   place { kf, partners, options, maxMatches }  -> step + map, pairs ([earlier, matches]), ms
//                                            kf and each partner { index, xy, desc, width, height }:
//                                            kf matched to each partner, the strongest maxMatches
//                                            kept, then added as `add` does (not deferred)
//   dropMatcher { }                        -> the worker's matcher and features let go
//   finish { pairs: [{ a, b, matches }], fromScratch }  -> result with points, tracks, colourObs
//   reconstruct { width, height, focal, xy: [Float32Array per keyframe], pairs, poses }
//                                          -> the same, from scratch (Reconstruct again)
//   free { }
// A map or a finish's result has its points as a Float32Array (3 per point), tracks as an
// Int32Array, the finish's colourObs too (an image and a keypoint per point); they are transferred.
import createDemo from './jba_demo.js';
import { strongest } from './gates.js';

let M = null;
let cancel = false;

// The map's matcher (D159): LightGlue's or LighterGlue's, and the features of the latest
// keyframes, by keyframe index (a Map in the order last used). A partner whose features
// went is described again from what the page sends.
const matching = { jmt: null, matcher: null, key: '', feats: new Map() };
const KEEP_FEATURES = 8;

async function featuresOf(k) {
  let f = matching.feats.get(k.index);
  if (f) matching.feats.delete(k.index);
  else f = await matching.matcher.describeKeypoints({ width: k.width, height: k.height }, k.xy, k.desc);
  matching.feats.set(k.index, f);
  return f;
}

function forgetFeatures(keep = KEEP_FEATURES) {
  for (const [index, f] of matching.feats) {
    if (matching.feats.size <= keep) break;
    f.destroy();
    matching.feats.delete(index);
  }
}

function dropMatcher() {
  forgetFeatures(0);
  matching.matcher?.destroy();
  matching.matcher = null;
  matching.key = '';
}

async function module() {
  if (!M) {
    M = await createDemo();
    // The finish's progress: posted to the page; false stops the run (Stop).
    M.onProgress = (json) => {
      postMessage({ progress: JSON.parse(json) });
      return !cancel;
    };
  }
  return M;
}

// Module memory for typed arrays, freed together.
function arena() {
  const ptrs = [];
  return {
    put(arr) {
      const p = M._malloc(Math.max(arr.byteLength, 4));
      if (!p) throw new Error('out of memory');
      new Uint8Array(M.HEAPU8.buffer, p, arr.byteLength).set(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength));
      ptrs.push(p);
      return p;
    },
    free() { for (const p of ptrs) M._free(p); },
  };
}

// Pairs as the module takes them: per pair one or two images, then offsets into the matches.
function packPairs(pairs, two) {
  const images = new Int32Array((two ? 2 : 1) * pairs.length);
  const begin = new Int32Array(pairs.length + 1);
  let total = 0;
  pairs.forEach((p, k) => {
    if (two) { images[2 * k] = p.a; images[2 * k + 1] = p.b; } else images[k] = p.earlier;
    begin[k] = total;
    total += p.matches.length / 2;
  });
  begin[pairs.length] = total;
  const matches = new Int32Array(2 * total);
  let at = 0;
  for (const p of pairs) { matches.set(p.matches, at); at += p.matches.length; }
  return { images, begin, matches };
}

function arrays(n, withColour) {
  const points = M.HEAPF32.slice(M._demo_points() >> 2, (M._demo_points() >> 2) + 3 * n);
  const tracks = M.HEAP32.slice(M._demo_tracks() >> 2, (M._demo_tracks() >> 2) + n);
  const out = { points, tracks };
  if (withColour) out.colourObs = M.HEAP32.slice(M._demo_colour_obs() >> 2, (M._demo_colour_obs() >> 2) + 2 * n);
  return out;
}

async function snapshot() {
  const map = JSON.parse(await M.ccall('demo_session_snapshot', 'string', [], []));
  if (!map.ok) throw new Error(map.error);
  return { ...map, ...arrays(map.points, false) };
}

const ops = {
  async init() {
    await module();
    return JSON.parse(await M.ccall('demo_backends', 'string', [], [], { async: true }));
  },
  async new(o) {
    await module();
    forgetFeatures(0);
    const r = JSON.parse(await M.ccall('demo_session_new', 'string',
      ['number', 'number', 'number', 'number', 'number', 'number', 'number', 'string', 'string'],
      [o.width, o.height, o.focal, o.focalUnknown ? 1 : 0, o.window, o.restartAfter, o.minPairInliers, o.poses, o.ba],
      { async: true }));
    if (!r.ok) throw new Error(r.error);
    return r;
  },
  async add({ xy, pairs, deferSplit }) {
    const a = arena();
    try {
      const pk = packPairs(pairs, false);
      const step = JSON.parse(await M.ccall('demo_session_add', 'string',
        ['number', 'number', 'number', 'number', 'number', 'number', 'number'],
        [xy.length / 2, a.put(xy), pairs.length, a.put(pk.images), a.put(pk.begin), a.put(pk.matches), deferSplit ? 1 : 0],
        { async: true }));
      return { step, map: await snapshot() };
    } finally {
      a.free();
    }
  },
  async retry({ pairs, newPiece }) {
    const a = arena();
    try {
      const pk = packPairs(pairs, false);
      const step = JSON.parse(await M.ccall('demo_session_retry', 'string',
        ['number', 'number', 'number', 'number', 'number'],
        [pairs.length, a.put(pk.images), a.put(pk.begin), a.put(pk.matches), newPiece ? 1 : 0], { async: true }));
      return { step, map: await snapshot() };
    } finally {
      a.free();
    }
  },
  async matcher({ url, precision }) {
    const key = `${url}/${precision}`;
    if (matching.matcher && matching.key === key) return {};
    dropMatcher();
    matching.jmt ??= await import(new URL('jmt/jmt.js', import.meta.url).href);
    const weights = await matching.jmt.fetchWeights(url);
    matching.matcher = await matching.jmt.create({ weights, precision });
    matching.key = key;
    return {};
  },
  async place({ kf, partners, options, maxMatches }) {
    if (!matching.matcher) throw new Error('no matcher in the worker');
    const t0 = performance.now();
    const fk = await featuresOf(kf);
    const pairs = [];
    for (const p of partners) {
      const fp = await featuresOf(p);
      const m = await matching.matcher.match(fp, p.xy, fk, kf.xy, options);
      pairs.push({ earlier: p.index, matches: strongest(m.matches, m.scores, maxMatches) });
    }
    // The newest keyframe stays: the next ones are matched to it.
    matching.feats.delete(kf.index);
    matching.feats.set(kf.index, fk);
    forgetFeatures();
    const matchMs = performance.now() - t0;
    const r = await ops.add({ xy: kf.xy, pairs, deferSplit: false });
    return { ...r, pairs: pairs.map((p) => [p.earlier, p.matches.length / 2]), matchMs, ms: performance.now() - t0 };
  },
  async dropMatcher() {
    dropMatcher();
    return {};
  },
  async finish({ pairs, fromScratch }) {
    const a = arena();
    try {
      const pk = packPairs(pairs, true);
      const r = JSON.parse(await M.ccall('demo_session_finish', 'string', ['number', 'number', 'number', 'number', 'number'],
        [pairs.length, a.put(pk.images), a.put(pk.begin), a.put(pk.matches), fromScratch ? 1 : 0], { async: true }));
      return r.ok || r.cancelled ? { ...r, ...arrays(r.points, true) } : r;
    } finally {
      a.free();
    }
  },
  async reconstruct({ width, height, focal, xy, pairs, poses }) {
    await module();
    const a = arena();
    try {
      const begin = new Int32Array(xy.length + 1);
      let total = 0;
      xy.forEach((k, i) => { begin[i] = total; total += k.length / 2; });
      begin[xy.length] = total;
      const all = new Float32Array(2 * total);
      xy.forEach((k, i) => all.set(k, 2 * begin[i]));
      const pk = packPairs(pairs, true);
      const r = JSON.parse(await M.ccall('demo_capture_reconstruct', 'string',
        ['number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'string'],
        [xy.length, width, height, focal, a.put(begin), a.put(all), pairs.length, a.put(pk.images), a.put(pk.begin),
         a.put(pk.matches), poses], { async: true }));
      return r.ok || r.cancelled ? { ...r, ...arrays(r.points, true) } : r;
    } finally {
      a.free();
    }
  },
  // The module's memory, bytes (a WebAssembly module's never shrinks).
  async mem() {
    return { bytes: M ? M.HEAPU8.length : 0 };
  },
  async free() {
    forgetFeatures(0);  // a new session numbers its keyframes from 0 again
    if (M) M._demo_session_free();
    return {};
  },
};

// One op at a time, in order; a cancel goes through at once.
let queue = Promise.resolve();
onmessage = ({ data }) => {
  if (data.op === 'cancel') {
    cancel = true;
    return;
  }
  queue = queue.then(async () => {
    try {
      // A Stop holds until the next finish or reconstruct, which it ends; any op clears it after.
      const result = await ops[data.op](data).finally(() => { if (data.op === 'finish' || data.op === 'reconstruct') cancel = false; });
      const transfer = [];
      for (const v of [result?.map?.points, result?.map?.tracks, result?.points, result?.tracks, result?.colourObs])
        if (v?.buffer) transfer.push(v.buffer);
      postMessage({ id: data.id, ok: true, result }, transfer);
    } catch (e) {
      postMessage({ id: data.id, ok: false, error: String(e?.message ?? e) });
    }
  });
};
