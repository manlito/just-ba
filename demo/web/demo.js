// just-ba's live demo (D152): structure from motion on real photos, in the browser.
// - Keypoints and descriptors: ALIKED, from just-keypoints' module (jkp/), on WebGPU.
// - Matches: LightGlue on every pair, from just-matching's module (jmt/), on WebGPU.
// - Reconstruction: the demo's mapper (jba_demo.js): poselib-gpu verifies the pairs and registers images, just-ba
//   builds tracks, triangulates and adjusts; on the CPU or WebGPU, once on each to compare.
// Keypoints and matches are kept until the images or their settings change, so reconstructing again on another
// backend costs only the reconstruction. URL parameters: run=1 runs once the sample set is loaded; n=k keeps the
// first k sample images; backends=cpu|webgpu|both; focal=guess drops the samples' focal lengths, as for photos of
// the visitor's own (1.2 times the larger side, then searched for).
const params = new URLSearchParams(location.search);
const $ = id => document.getElementById(id);
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const MAX_IMAGES = 60;
const MIN_MATCHES = 15;  // pairs with fewer are not handed to the mapper (as the COLMAP path's --min-matches)
const LIGHTGLUE = { layers: 9, depthConfidence: 0.95, widthConfidence: 0.99 };  // its adaptive depth and width

const state = {
  images: [],         // { name, file, bitmap, data, focal (pixels of the scaled image, NaN: unknown), kp, feat, tKp, tDesc, thumb }
  extractor: null, extractorK: 0,
  matcher: null, weights: null, precision: 'fp32',
  matches: null,      // { key, pairs: [{ i, j, m: Int32Array }], tMatch, nPairs }
  runs: {},           // by backend: the mapper's answer, with points, tracks and colours
  shown: null,
  busy: false, cancel: false,
  timing: {},
  gpu: false,
};

// ---- status

function setStatus(text, error = false) {
  $('status').textContent = text;
  $('status').classList.toggle('error', error);
}
function setProgress(done, total) {
  const p = $('progress');
  if (done == null) return void (p.hidden = true);
  p.hidden = false;
  p.max = total;
  p.value = done;
}
// One task, not throttled in a background tab as setTimeout is.
const tick = () => new Promise(r => {
  const ch = new MessageChannel();
  ch.port1.onmessage = () => r();
  ch.port2.postMessage(0);
});
const median = xs => {
  if (!xs.length) return NaN;
  const s = [...xs].sort((a, b) => a - b), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const fmt = (v, d = 2) => (v == null || !Number.isFinite(v) ? '–' : Number(v).toFixed(d));
const int = v => Number(v).toLocaleString('en-US');
const setting = name => document.querySelector(`input[name=${name}]:checked`).value;

// ---- images

function settingsKey() { return `${setting('keypoints')}/${setting('side')}`; }

async function bitmapOf(blob) {
  const side = Number(setting('side'));
  let bitmap = await createImageBitmap(blob, { imageOrientation: 'from-image' });
  const s = Math.min(1, side / Math.max(bitmap.width, bitmap.height));
  if (s < 1) {
    const scaled = await createImageBitmap(bitmap, { resizeWidth: Math.round(bitmap.width * s),
                                                     resizeHeight: Math.round(bitmap.height * s), resizeQuality: 'high' });
    bitmap.close();
    bitmap = scaled;
  }
  return { bitmap, scale: s };
}

function rgba(bitmap) {
  const c = new OffscreenCanvas(bitmap.width, bitmap.height);
  const g = c.getContext('2d', { willReadFrequently: true });
  g.drawImage(bitmap, 0, 0);
  return g.getImageData(0, 0, bitmap.width, bitmap.height);
}

// entries: [{ name, blob, focal (pixels of the original file, or NaN) }]
async function addImages(entries) {
  for (const { name, blob, focal } of entries) {
    if (state.images.length >= MAX_IMAGES) {
      setStatus(`At most ${MAX_IMAGES} images.`, true);
      break;
    }
    try {
      const { bitmap, scale } = await bitmapOf(blob);
      state.images.push({ name, blob, focalOriginal: focal, focal: focal * scale, bitmap, data: rgba(bitmap) });
    } catch (e) {
      setStatus(`Cannot read ${name}: ${e.message}`, true);
    }
  }
  invalidate();
  renderThumbs();
}

// Images, keypoints and matches follow the size setting: read the files again.
async function rescale() {
  const entries = state.images.map(im => ({ name: im.name, blob: im.blob, focal: im.focalOriginal }));
  for (const im of state.images) im.feat?.destroy(), im.bitmap.close();
  state.images = [];
  await addImages(entries);
}

function invalidate() {
  for (const im of state.images) {
    if (im.kpKey !== settingsKey()) {
      im.kp = null;
      im.feat?.destroy();
      im.feat = null;
    }
  }
  state.matches = null;
  state.runs = {};
  state.shown = null;
  renderAll();
}

function clearImages() {
  for (const im of state.images) im.feat?.destroy(), im.bitmap.close();
  state.images = [];
  invalidate();
  renderThumbs();
}

async function loadSamples() {
  clearImages();
  setStatus('Loading the sample images…');
  const list = await (await fetch('samples/samples.json')).json();
  const n = Math.min(list.images.length, Number(params.get('n')) || Infinity);
  let done = 0;
  const entries = await Promise.all(list.images.slice(0, n).map(async im => {
    const r = await fetch(`samples/${im.file}`);
    if (!r.ok) throw new Error(`samples/${im.file}: ${r.status}`);
    const blob = await r.blob();
    setProgress(++done, n);
    return { name: im.file, blob, focal: params.get('focal') === 'guess' ? NaN : im.focal ?? NaN };
  }));
  setProgress(null);
  await addImages(entries);
  setStatus(`${state.images.length} images. Press Run.`);
}

function renderThumbs() {
  const run = state.runs[state.shown];
  $('thumbs').replaceChildren(...state.images.map((im, i) => {
    const div = document.createElement('div');
    div.className = 'thumb';
    const reg = run?.registeredSet;
    if (reg && !reg.has(i)) div.classList.add('out');
    div.title = `${i + 1}: ${im.name} (${im.bitmap.width} x ${im.bitmap.height})` +
                (im.kp ? `, ${im.kp.xy.length / 2} keypoints` : '') +
                (run?.image_error_px?.[i] != null ? `, ${fmt(run.image_error_px[i], 2)} px` : '');
    const c = document.createElement('canvas');
    c.width = 120;
    c.height = 80;
    const g = c.getContext('2d');
    const s = Math.max(c.width / im.bitmap.width, c.height / im.bitmap.height);
    const w = im.bitmap.width * s, h = im.bitmap.height * s;
    g.drawImage(im.bitmap, (c.width - w) / 2, (c.height - h) / 2, w, h);
    const label = document.createElement('span');
    label.textContent = i + 1;
    div.append(c, label);
    return div;
  }));
}

// ---- keypoints and matches (WebGPU)

async function ensureModules() {
  const K = Number(setting('keypoints'));
  if (!state.extractor || state.extractorK !== K) {
    setStatus('Loading just-keypoints (ALIKED)…');
    const t0 = performance.now();
    const jkp = await import(new URL('jkp/jkp.js', location.href).href);
    state.extractor?.destroy?.();
    state.extractor = await jkp.create({ weightsUrl: 'jkp/aliked-n16.jkpw', topK: K });
    state.extractorK = K;
    state.timing.loadKeypoints = performance.now() - t0;
  }
  if (!state.matcher) {
    const jmt = await import(new URL('jmt/jmt.js', location.href).href);
    const t0 = performance.now();
    if (!state.weights) {
      state.weights = await jmt.fetchWeights('jmt/lightglue-aliked-fp16.jmtw', {
        onProgress(got, total) {
          setStatus(total ? `LightGlue's weights: ${(got / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB…` : `LightGlue's weights…`);
          if (total) setProgress(got, total);
        },
      });
    }
    setProgress(null);
    setStatus('Creating the matcher…');
    state.matcher = await jmt.create({ weights: state.weights, precision: state.precision });
    state.timing.loadMatcher = performance.now() - t0;
  }
}

async function findMatches() {
  const n = state.images.length, key = settingsKey();
  for (const [i, im] of state.images.entries()) {
    if (state.cancel) throw new Error('stopped');
    if (im.kp && im.feat && im.kpKey === key) continue;
    setStatus(`Image ${i + 1} of ${n}: keypoints…`);
    setProgress(i, n);
    let t0 = performance.now();
    const k = await state.extractor.extract(im.data);
    im.kp = { xy: k.xy, desc: k.desc };
    im.tKp = performance.now() - t0;
    t0 = performance.now();
    im.feat?.destroy();
    im.feat = await state.matcher.describeKeypoints(im.data, im.kp.xy, im.kp.desc);
    await state.matcher.finish();
    im.tDesc = performance.now() - t0;
    im.kpKey = key;
  }
  renderThumbs();
  const sig = key + '|' + state.images.map(im => im.name).join('|');
  if (state.matches?.key === sig) return false;
  const total = n * (n - 1) / 2, pairs = [];
  const t0 = performance.now();
  let p = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++, p++) {
      if (state.cancel) throw new Error('stopped');
      if (p % 4 === 0) {
        setStatus(`Matching pair ${p + 1} of ${total}…`);
        setProgress(p, total);
      }
      const a = state.images[i], b = state.images[j];
      const r = await state.matcher.match(a.feat, a.kp.xy, b.feat, b.kp.xy, LIGHTGLUE);
      if (r.matches.length / 2 >= MIN_MATCHES) pairs.push({ i, j, m: r.matches });
    }
  }
  state.matches = { key: sig, pairs, nPairs: total, tMatch: performance.now() - t0 };
  setProgress(null);
  return true;
}

// ---- the reconstruction (the demo's module)

let M = null;
async function loadMapper() {
  const { default: createDemo } = await import('./jba_demo.js');
  M = await createDemo();
  return JSON.parse(await M.ccall('demo_backends', 'string', [], [], { async: true }));
}

const STAGES = {
  'focal search': 'searching for the focal length',
  verify: 'verifying every pair',
  tracks: 'building tracks',
  'first pair': 'starting from the first pair',
  register: 'registering images',
  adjust: 'bundle adjustment',
  final: 'final adjustment',
};

async function reconstruct(backend) {
  const ims = state.images, n = ims.length, pairs = state.matches.pairs;
  const sizes = new Int32Array(2 * n), focal = new Float64Array(n), begin = new Int32Array(n + 1);
  let K = 0;
  for (const [i, im] of ims.entries()) {
    sizes[2 * i] = im.bitmap.width;
    sizes[2 * i + 1] = im.bitmap.height;
    focal[i] = Number.isFinite(im.focal) ? im.focal : NaN;
    begin[i] = K;
    K += im.kp.xy.length / 2;
  }
  begin[n] = K;
  // COLMAP's pixel convention: the centre of the first pixel at 0.5, 0.5.
  const xy = new Float32Array(2 * K);
  for (const [i, im] of ims.entries()) for (let k = 0; k < im.kp.xy.length; k++) xy[2 * begin[i] + k] = im.kp.xy[k] + 0.5;
  const R = pairs.length, pimg = new Int32Array(2 * R), pbeg = new Int32Array(R + 1);
  let Mt = 0;
  for (const [p, q] of pairs.entries()) {
    pimg[2 * p] = q.i;
    pimg[2 * p + 1] = q.j;
    pbeg[p] = Mt;
    Mt += q.m.length / 2;
  }
  pbeg[R] = Mt;
  const mm = new Int32Array(2 * Mt);
  for (const [p, q] of pairs.entries()) mm.set(q.m, 2 * pbeg[p]);

  const bufs = [];
  const put = (arr) => {
    const ptr = M._malloc(Math.max(4, arr.byteLength));
    bufs.push(ptr);
    M.HEAPU8.set(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength), ptr);
    return ptr;
  };
  const args = [n, put(sizes), put(focal), put(begin), put(xy), R, put(pimg), put(pbeg), put(mm), backend, backend];
  const name = backend === 'cpu' ? 'CPU' : 'WebGPU';
  M.onProgress = json => {
    const p = JSON.parse(json);
    const what = STAGES[p.stage] ?? p.stage;
    const model = p.model > 0 ? `, model ${p.model + 1}` : '';
    const reg = p.model >= 0 ? `: ${p.registered} of ${p.images} images${model}` : '';
    setStatus(`${name}: ${what}${reg} (${p.seconds.toFixed(1)} s)`);
    if (p.model >= 0) setProgress(p.registered, p.images);
    return !state.cancel;
  };
  const t0 = performance.now();
  let out;
  try {
    out = await M.ccall('demo_reconstruct', 'string',
      ['number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'string', 'string'],
      args, { async: true });
  } finally {
    for (const b of bufs) M._free(b);
    M.onProgress = null;
    setProgress(null);
  }
  const r = JSON.parse(out);
  r.wall = (performance.now() - t0) / 1000;
  r.backend = backend;
  if (r.steps?.length) {
    const last = r.steps[r.steps.length - 1];
    const total = last.first_point + last.num_points;
    r.pointsAll = M.HEAPF32.slice(M._demo_points() >> 2, (M._demo_points() >> 2) + 3 * total);
    r.tracksAll = M.HEAP32.slice(M._demo_tracks() >> 2, (M._demo_tracks() >> 2) + total);
    const obs = M.HEAP32.slice(M._demo_colour_obs() >> 2, (M._demo_colour_obs() >> 2) + 2 * last.num_points);
    // Each track's colour, from an image that sees its point in the final model.
    r.colour = new Map();
    for (let p = 0; p < last.num_points; p++) {
      const im = ims[obs[2 * p]], k = obs[2 * p + 1];
      if (!im) continue;
      const x = Math.min(im.data.width - 1, Math.max(0, Math.round(im.kp.xy[2 * k])));
      const y = Math.min(im.data.height - 1, Math.max(0, Math.round(im.kp.xy[2 * k + 1])));
      const o = 4 * (y * im.data.width + x), d = im.data.data;
      r.colour.set(r.tracksAll[last.first_point + p], `rgb(${d[o]},${d[o + 1]},${d[o + 2]})`);
    }
    r.registeredSet = new Set(last.cameras.map(c => c.image));
  }
  return r;
}

// ---- the run

async function run() {
  if (state.busy) return;
  if (state.images.length < 3) return setStatus('Add at least three images.', true);
  state.busy = true;
  state.cancel = false;
  $('run').disabled = true;
  $('stop').hidden = false;
  stopPlaying();
  const tRun = performance.now();
  try {
    await ensureModules();
    const fresh = await findMatches();
    renderMatchTimes();
    if (!state.matches.pairs.length) throw new Error('no pair has enough matches');
    const which = setting('backends');
    const list = which === 'both' ? ['cpu', 'webgpu'] : [which];
    if (fresh) state.runs = {};
    for (const b of list) {
      if (!state.gpu && b === 'webgpu') continue;
      state.runs[b] = await reconstruct(b);
      state.runs[b].warm = (state.warm ??= new Set()).has(b);
      state.warm.add(b);
      if (state.runs[b].cancelled) throw new Error('stopped');
      state.shown = b;
      renderAll();
      if (state.runs[b].steps?.length) setStep(state.runs[b].steps.length - 1);
      await tick();
    }
    const r = state.runs[state.shown];
    const total = (performance.now() - tRun) / 1000;
    if (!r?.ok) setStatus(`The reconstruction failed: ${r?.error || 'no answer'}.`, true);
    else setStatus(`Done in ${total.toFixed(1)} s: ${r.registered} of ${r.images} images registered, ` +
                   `${int(r.points)} points.` + (fresh ? '' : ' Keypoints and matches were reused.'));
    if (r?.steps?.length > 1) play();
  } catch (e) {
    setProgress(null);
    setStatus(e.message === 'stopped' ? 'Stopped.' : 'Error: ' + e.message, e.message !== 'stopped');
    if (e.message !== 'stopped') console.error(e);
  } finally {
    state.busy = false;
    $('run').disabled = false;
    $('stop').hidden = true;
  }
}

// ---- tables

function renderMatchTimes() {
  const ims = state.images.filter(im => im.kp), m = state.matches;
  if (!ims.length || !m) {
    $('match-times').innerHTML = '<tr><td class="muted">Not run yet.</td></tr>';
    return;
  }
  const nm = m.pairs.reduce((s, q) => s + q.m.length / 2, 0);
  const rows = [
    ['Images', state.images.length],
    ['Keypoints per image, median', int(median(ims.map(im => im.kp.xy.length / 2)))],
    ['ALIKED per image, median', `${fmt(median(ims.map(im => im.tKp)), 0)} ms`],
    ['LightGlue features per image, median', `${fmt(median(ims.map(im => im.tDesc)), 0)} ms`],
    ['Pairs matched', `${int(m.nPairs)} in ${fmt(m.tMatch / 1000, 1)} s (${fmt(m.tMatch / m.nPairs, 0)} ms each)`],
    [`Pairs with ${MIN_MATCHES} matches or more`, `${int(m.pairs.length)}, ${int(nm)} matches`],
  ];
  $('match-times').innerHTML = rows.map(([a, b]) => `<tr><td>${a}</td><td class="n">${b}</td></tr>`).join('');
}

function renderCompare() {
  const cpu = state.runs.cpu, gpu = state.runs.webgpu, runs = [cpu, gpu].filter(Boolean);
  if (!runs.length) {
    $('compare').innerHTML = '<tr><td class="muted">Run to compare.</td></tr>';
    $('bars').hidden = true;
    $('compare-note').textContent = '';
    return;
  }
  const head = `<tr><th></th>${cpu ? '<th class="n">CPU</th>' : ''}${gpu ? '<th class="n">WebGPU</th>' : ''}` +
               `${cpu && gpu ? '<th class="n">CPU / WebGPU</th>' : ''}</tr>`;
  const timeRow = (label, key, cls = '') => {
    const a = cpu?.time?.[key], b = gpu?.time?.[key];
    if (!(a > 0.0005) && !(b > 0.0005)) return '';
    const ratio = a > 0 && b > 0 ? a / b : NaN;
    return `<tr class="${cls}"><td>${label}</td>${cpu ? `<td class="n">${fmt(a, 2)} s</td>` : ''}` +
           `${gpu ? `<td class="n">${fmt(b, 2)} s</td>` : ''}` +
           `${cpu && gpu ? `<td class="n ${ratio > 1 ? 'speed' : ''}">${Number.isFinite(ratio) ? fmt(ratio, 1) + '×' : '–'}</td>` : ''}</tr>`;
  };
  const valRow = (label, f) => `<tr><td>${label}</td>${cpu ? `<td class="n">${f(cpu)}</td>` : ''}` +
                               `${gpu ? `<td class="n">${f(gpu)}</td>` : ''}${cpu && gpu ? '<td></td>' : ''}</tr>`;
  const adj = r => r.adjustments ?? 0;
  $('compare').innerHTML = head +
    timeRow('Focal length search', 'focal_search') +
    timeRow('Verify every pair (relative pose)', 'verify') +
    timeRow('Build tracks', 'tracks') +
    timeRow('Register images (PnP)', 'pnp') +
    timeRow('Triangulate', 'triangulate') +
    timeRow(`Bundle adjustment (${runs.map(adj).join(' / ')} runs)`, 'adjust') +
    timeRow('Reconstruction', 'total', 'total') +
    valRow('Images registered', r => (r.ok ? `${r.registered} / ${r.images}` : 'failed')) +
    valRow('Points', r => int(r.points ?? 0)) +
    valRow('Observations', r => int(r.observations ?? 0)) +
    valRow('Mean reprojection error', r => `${fmt(r.mean_error_px, 3)} px`) +
    valRow('Focal length found', r => (r.intrinsics?.length ? `${fmt(r.intrinsics[0], 0)} px` : '–')) +
    valRow('Pairs kept', r => `${r.pairs_kept ?? 0} / ${r.pairs ?? 0}`);
  // Bars: the reconstruction's time on each.
  if (cpu && gpu) {
    const max = Math.max(cpu.time.total, gpu.time.total);
    const bar = (label, v, cls) => `<span>${label}</span><div class="track"><div class="fill ${cls}" style="width:${(100 * v / max).toFixed(1)}%"></div></div><span class="v">${fmt(v, 2)} s</span>`;
    $('bars').innerHTML = bar('CPU', cpu.time.total, 'cpu') + bar('WebGPU', gpu.time.total, '') +
      bar('CPU, adjustment', cpu.time.adjust, 'cpu') + bar('WebGPU, adjustment', gpu.time.adjust, '');
    $('bars').hidden = false;
  } else $('bars').hidden = true;
  const cold = gpu && !gpu.warm;
  $('compare-note').textContent = 'Times measured inside the reconstruction, keypoints and matching not included. ' +
    (cold ? 'This was the first WebGPU run on this page: it includes starting the device and compiling its kernels. ' +
            'Run again (keypoints and matches are reused) for warm times.' : '');
}

function renderAll() {
  renderMatchTimes();
  renderCompare();
  renderThumbs();
  const have = Object.keys(state.runs);
  $('shown').hidden = have.length < 2;
  for (const inp of document.querySelectorAll('input[name=shown]')) {
    inp.disabled = !state.runs[inp.value];
    inp.checked = inp.value === state.shown;
  }
  const r = state.runs[state.shown];
  $('empty').hidden = !!r?.steps?.length;
  $('step').max = String(Math.max(0, (r?.steps?.length ?? 1) - 1));
  $('step').disabled = $('play').disabled = !(r?.steps?.length > 1);
  if (r?.steps?.length) fitView(r);
  else $('step-label').textContent = 'no run yet';
  draw();
}

// ---- steps

let step = 0, playing = null;
function setStep(k) {
  const r = state.runs[state.shown];
  if (!r?.steps?.length) return;
  step = Math.max(0, Math.min(k, r.steps.length - 1));
  $('step').value = String(step);
  const s = r.steps[step];
  $('step-label').textContent = `${s.stage}: ${s.cameras.length} images, ${int(s.num_points)} points`;
  draw();
}
function stopPlaying() {
  if (playing) clearInterval(playing);
  playing = null;
  $('play').textContent = 'Play';
}
function play() {
  const r = state.runs[state.shown];
  stopPlaying();
  if (step + 1 >= r.steps.length) setStep(0);
  $('play').textContent = 'Pause';
  const dt = Math.max(120, Math.min(600, 9000 / r.steps.length));
  playing = setInterval(() => {
    if (step + 1 >= r.steps.length) return stopPlaying();
    setStep(step + 1);
  }, dt);
}

// ---- the 3D view: points and camera frusta on a 2D canvas

const canvas = $('view');
const ctx = canvas.getContext('2d');
const view = { target: [0, 0, 0], yaw: 0, pitch: 0.3, dist: 10, size: 1, up: [0, -1, 0], R: null };
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = a => { const n = Math.hypot(...a) || 1; return [a[0] / n, a[1] / n, a[2] / n]; };
const quantile = (xs, q) => { const s = Float64Array.from(xs).sort(); return s[Math.min(s.length - 1, Math.floor(q * s.length))]; };

// The model's frame is its first camera's. Up is the cameras' mean up (their -y axis); the view is turned so that
// up is up, centred on the bulk of the points, and looks from behind the cameras.
function fitView(r) {
  const last = r.steps[r.steps.length - 1], cams = last.cameras;
  let up = [0, 0, 0], fwd = [0, 0, 0];
  for (const c of cams) {
    const p = c.pose;  // C (3), then R row-major: rows are the camera's x, y, z axes in the world
    up = sub(up, [p[6], p[7], p[8]]);
    fwd = [fwd[0] + p[9], fwd[1] + p[10], fwd[2] + p[11]];
  }
  up = norm(up);
  fwd = norm(sub(fwd, up.map(v => v * dot(fwd, up))));
  const right = cross(fwd, up);
  view.R = [right, up, [-fwd[0], -fwd[1], -fwd[2]]];  // world -> display (x right, y up, z towards the viewer)
  const P = r.pointsAll, o = 3 * last.first_point, np = last.num_points;
  const disp = q => [dot(view.R[0], q), dot(view.R[1], q), dot(view.R[2], q)];
  const xs = [[], [], []];
  for (let p = 0; p < np; p++) {
    const d = disp([P[o + 3 * p], P[o + 3 * p + 1], P[o + 3 * p + 2]]);
    for (let k = 0; k < 3; k++) xs[k].push(d[k]);
  }
  for (const c of cams) {
    const d = disp(c.pose);
    for (let k = 0; k < 3; k++) xs[k].push(d[k]);
  }
  const lo = xs.map(v => quantile(v, 0.05)), hi = xs.map(v => quantile(v, 0.95));
  view.target = lo.map((v, d) => 0.5 * (v + hi[d]));
  view.size = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1;
  view.dist = 0.8 * view.size;
  // Frusta sized by the cameras' spread, not the points': a walk around a building has its cameras close together.
  const cs = [0, 1, 2].map(k => cams.map(c => c.pose[k]));
  const camSize = Math.hypot(...cs.map(v => Math.max(...v) - Math.min(...v)));
  view.frustum = Math.max(0.006 * view.size, Math.min(0.03 * view.size, 0.06 * camSize));
  view.yaw = 0;
  view.pitch = 0.45;
}

function eyeOf() {
  const cp = Math.cos(view.pitch);
  const eye = [view.target[0] + view.dist * cp * Math.sin(view.yaw), view.target[1] + view.dist * Math.sin(view.pitch),
               view.target[2] + view.dist * cp * Math.cos(view.yaw)];
  const fw = norm(sub(view.target, eye));
  const r = norm(cross(fw, [0, 1, 0]));
  return { eye, fw, r, u: cross(r, fw) };
}

function draw() {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const r = state.runs[state.shown];
  $('overlay').textContent = r?.steps?.length ? `${r.backend === 'cpu' ? 'CPU' : 'WebGPU'}'s model` : '';
  if (!r?.steps?.length || !view.R) return;
  const s = r.steps[Math.min(step, r.steps.length - 1)];
  const cam = eyeOf();
  const F = 0.9 * Math.min(w, h);
  const disp = q => [dot(view.R[0], q), dot(view.R[1], q), dot(view.R[2], q)];
  const project = q => {
    const d = sub(disp(q), cam.eye);
    const z = dot(d, cam.fw);
    if (z < 1e-3 * view.size) return null;
    return [w / 2 + F * dot(d, cam.r) / z, h / 2 - F * dot(d, cam.u) / z, z];
  };
  const P = r.pointsAll, T = r.tracksAll, o = s.first_point;
  const grey = css('--point');
  const pt = Math.max(1.5, Math.min(3, 2 * dpr * 1200 / Math.sqrt(s.num_points + 1) / 25));
  for (let p = 0; p < s.num_points; p++) {
    const i = o + p;
    const q = project([P[3 * i], P[3 * i + 1], P[3 * i + 2]]);
    if (!q) continue;
    ctx.fillStyle = r.colour.get(T[i]) ?? grey;
    ctx.fillRect(q[0] - pt / 2, q[1] - pt / 2, pt, pt);
  }
  const frustum = (c, colour, width) => {
    const C = [c[0], c[1], c[2]];
    const x = [c[3], c[4], c[5]], y = [c[6], c[7], c[8]], z = [c[9], c[10], c[11]];
    const k = view.frustum;
    const corner = (sx, sy) => [0, 1, 2].map(d => C[d] + k * (z[d] + 0.5 * sx * x[d] + 0.34 * sy * y[d]));
    const q0 = project(C), qs = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)].map(project);
    if (!q0 || qs.some(q => !q)) return;
    ctx.strokeStyle = colour;
    ctx.lineWidth = width;
    ctx.beginPath();
    for (const q of qs) ctx.moveTo(q0[0], q0[1]), ctx.lineTo(q[0], q[1]);
    ctx.moveTo(qs[3][0], qs[3][1]);
    for (const q of qs) ctx.lineTo(q[0], q[1]);
    ctx.stroke();
    ctx.beginPath();  // the top edge, thicker, so up shows
    ctx.moveTo(qs[0][0], qs[0][1]);
    ctx.lineTo(qs[1][0], qs[1][1]);
    ctx.lineWidth = width + 1;
    ctx.stroke();
  };
  const added = new Set(s.added), colCam = css('--cam'), colNew = css('--cam-new');
  for (const c of s.cameras) frustum(c.pose, added.has(c.image) ? colNew : colCam, added.has(c.image) ? 2 : 1);
}

const pointers = new Map();
let pinch = 0;
canvas.addEventListener('pointerdown', e => { canvas.setPointerCapture(e.pointerId); pointers.set(e.pointerId, [e.clientX, e.clientY]); canvas.style.cursor = 'grabbing'; });
canvas.addEventListener('pointerup', e => { pointers.delete(e.pointerId); pinch = 0; canvas.style.cursor = 'grab'; });
canvas.addEventListener('pointercancel', e => { pointers.delete(e.pointerId); pinch = 0; });
canvas.addEventListener('pointermove', e => {
  if (!pointers.has(e.pointerId)) return;
  const [px, py] = pointers.get(e.pointerId);
  pointers.set(e.pointerId, [e.clientX, e.clientY]);
  if (pointers.size === 2) {
    const [a, b] = [...pointers.values()];
    const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
    if (pinch) view.dist *= pinch / d;
    pinch = d;
  } else {
    view.yaw -= (e.clientX - px) * 0.008;
    view.pitch = Math.max(-1.45, Math.min(1.45, view.pitch + (e.clientY - py) * 0.008));
  }
  draw();
});
canvas.addEventListener('wheel', e => { e.preventDefault(); view.dist *= Math.exp(e.deltaY * 0.001); draw(); }, { passive: false });
window.addEventListener('resize', draw);
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);

// ---- controls

$('run').addEventListener('click', run);
$('stop').addEventListener('click', () => { state.cancel = true; setStatus('Stopping…'); });
$('samples').addEventListener('click', () => !state.busy && loadSamples().catch(e => setStatus('Error: ' + e.message, true)));
$('add').addEventListener('click', () => $('files').click());
$('clear').addEventListener('click', () => {
  if (state.busy) return;
  clearImages();
  setStatus('Add at least three images.');
});
const fromFiles = files => [...files].filter(f => f.type.startsWith('image/')).map(f => ({ name: f.name, blob: f, focal: NaN }));
$('files').addEventListener('change', async e => {
  if (state.busy) return;
  await addImages(fromFiles(e.target.files));
  e.target.value = '';
  setStatus(`${state.images.length} images. Press Run.`);
});
const drop = $('drop');
for (const ev of ['dragenter', 'dragover']) drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); });
for (const ev of ['dragleave', 'drop']) drop.addEventListener(ev, () => drop.classList.remove('over'));
drop.addEventListener('drop', async e => {
  e.preventDefault();
  if (state.busy) return;
  await addImages(fromFiles(e.dataTransfer.files));
  setStatus(`${state.images.length} images. Press Run.`);
});
for (const inp of document.querySelectorAll('input[name=keypoints]')) inp.addEventListener('change', invalidate);
for (const inp of document.querySelectorAll('input[name=side]')) inp.addEventListener('change', () => rescale());
for (const inp of document.querySelectorAll('input[name=shown]')) {
  inp.addEventListener('change', () => {
    stopPlaying();
    state.shown = inp.value;
    renderAll();
    setStep(state.runs[state.shown].steps.length - 1);
  });
}
$('step').addEventListener('input', () => { stopPlaying(); setStep(Number($('step').value)); });
$('play').addEventListener('click', () => (playing ? stopPlaying() : play()));
if (params.get('backends')) {
  const b = document.querySelector(`input[name=backends][value="${params.get('backends')}"]`);
  if (b) b.checked = true;
}

// ---- start

async function start() {
  if (typeof WebAssembly.Suspending !== 'function') {
    $('notice').hidden = false;
    $('notice').textContent = 'This browser cannot run the demo: it needs WebAssembly JavaScript Promise Integration ' +
                              '(JSPI) and WebGPU, which current Chrome and Edge have.';
    setStatus('');
    return;
  }
  let info = 'no WebGPU adapter';
  try {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    if (adapter) {
      const i = adapter.info ?? {};
      info = [i.vendor, i.architecture, i.description].filter(Boolean).join(' / ') || 'WebGPU adapter';
      state.precision = adapter.features.has('shader-f16') ? 'fp16' : 'fp32';
      state.gpu = true;
    }
  } catch { /* no WebGPU */ }
  $('gpu-info').textContent = `GPU: ${info}. LightGlue runs in ${state.precision === 'fp16' ? 'FP16' : 'FP32'}.`;
  if (!state.gpu) {
    $('notice').hidden = false;
    $('notice').textContent = 'No WebGPU device here: keypoints and matching need one, so the demo cannot run.';
    setStatus('');
    return;
  }
  setStatus('Loading the reconstruction module…');
  const b = await loadMapper();
  if (!b.ba.includes('webgpu') || !b.poses.includes('webgpu')) {
    state.gpu = false;
    document.querySelector('input[name=backends][value=cpu]').checked = true;
    for (const v of ['both', 'webgpu']) document.querySelector(`input[name=backends][value=${v}]`).disabled = true;
  }
  await loadSamples();
  $('run').disabled = false;
  if (params.get('run') === '1') run();
}
start().catch(e => setStatus('Could not start the demo: ' + e.message, true));
