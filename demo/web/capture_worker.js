// The live capture's map (D157): the demo's module in a worker, holding one
// demo::Session (D153), so that placing a keyframe, adjusting the window and the finish
// never hold up the camera's loop on the page. The page posts { id, op, ... } and gets
// { id, ok, result } back (or { id, ok: false, error }); during a finish it also gets
// { progress } messages, and { op: 'cancel' } stops the finish at its next report.
//
// Ops:
//   init { }                               -> { poses: [...], ba: [...] } backends
//   new { width, height, focal, focalUnknown, window, restartAfter, minPairInliers, poses, ba }
//   add { xy: Float32Array, pairs: [{ earlier, matches: Int32Array }], deferSplit }  -> step + map
//   retry { pairs, newPiece }              -> step + map
//   finish { pairs: [{ a, b, matches }] }  -> result with points, tracks, colourObs
//   reconstruct { width, height, focal, xy: [Float32Array per keyframe], pairs, poses }
//                                          -> the same, from scratch (Reconstruct again)
//   free { }
// A map or a finish's result has its points as a Float32Array (3 per point), tracks as an
// Int32Array, the finish's colourObs too (an image and a keypoint per point); they are transferred.
import createDemo from './jba_demo.js';

let M = null;
let cancel = false;

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
  async finish({ pairs }) {
    cancel = false;
    const a = arena();
    try {
      const pk = packPairs(pairs, true);
      const r = JSON.parse(await M.ccall('demo_session_finish', 'string', ['number', 'number', 'number', 'number'],
        [pairs.length, a.put(pk.images), a.put(pk.begin), a.put(pk.matches)], { async: true }));
      return r.ok || r.cancelled ? { ...r, ...arrays(r.points, true) } : r;
    } finally {
      a.free();
    }
  },
  async reconstruct({ width, height, focal, xy, pairs, poses }) {
    cancel = false;
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
  async free() {
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
      const result = await ops[data.op](data);
      const transfer = [];
      for (const v of [result?.map?.points, result?.map?.tracks, result?.points, result?.tracks, result?.colourObs])
        if (v?.buffer) transfer.push(v.buffer);
      postMessage({ id: data.id, ok: true, result }, transfer);
    } catch (e) {
      postMessage({ id: data.id, ok: false, error: String(e?.message ?? e) });
    }
  });
};
