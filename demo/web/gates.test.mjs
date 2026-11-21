// The live capture's keep rules (gates.js) on the cases of Like Recon's own tests
// (src-tauri/src/capture.rs). Run: node --test demo/web/gates.test.mjs
import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  Blur, LOST_OVERLAP, MIN_REFERENCE_KEYPOINTS, anticipate, Picker, RECENT_KEYFRAMES, SPARSE_KEYFRAMES, Speed, confident, decide,
  defaultSettings, floorOf, hold, overlap, sharpness, slowDown, strongest, usableReference,
} from './gates.js';

const facts = (o) => ({
  hasMatcher: true, hasReference: true, keypoints: 2000, overlap: o, support: { inliers: 200, coverage: 0.5 },
  overlap2: null, blurry: false, shoot: false,
});
const close = (a, b, eps, msg) => assert.ok(Math.abs(a - b) < eps, `${msg ?? ''} ${a} vs ${b}`);

test('gates in order', () => {
  const s = defaultSettings();
  const fl = floorOf();
  const d = (f) => decide(s, fl, f);
  const first = { ...facts(0), hasReference: false, overlap: null };
  const shoot = { ...first, shoot: true, hasMatcher: false, keypoints: 0, blurry: true };
  assert.equal(d(shoot).gate, 'kept');
  assert.deepEqual(d(shoot).urge, { kind: 'due', why: 'shoot' });
  assert.equal(d({ ...facts(0.3), hasMatcher: false }).gate, 'no_matcher');
  assert.equal(d(first).gate, 'kept');
  assert.deepEqual(d(first).urge, { kind: 'due', why: 'first' });
  const few = { ...first, keypoints: 500 };
  assert.equal(d(few).gate, 'too_few_keypoints');
  assert.equal(d(few).urge.kind, 'no');
  const blurry = { ...first, blurry: true };
  assert.equal(d(blurry).gate, 'blurry');
  assert.equal(d(facts(0.8)).urge.kind, 'no');
  assert.equal(d(facts(0.8)).gate, 'waiting');
  assert.equal(d(facts(0.6)).urge.kind, 'band');
  assert.deepEqual(d(facts(0.55)).urge, { kind: 'due', why: 'target' });
  assert.deepEqual(d(facts(0.5)).urge, { kind: 'due', why: 'target' });
  assert.deepEqual(d(facts(0.4)).urge, { kind: 'overdue', why: 'target' });
  assert.equal(d(facts(0.1)).gate, 'lost');
  assert.deepEqual(d(facts(0.1)).urge, { kind: 'overdue', why: 'lost' });
  assert.ok(d(facts(0.5)).acceptable);
  const refused = { ...facts(0.5), keypoints: 500 };
  assert.equal(d(refused).gate, 'too_few_keypoints');
  assert.deepEqual(d(refused).urge, { kind: 'due', why: 'target' });
  assert.ok(!d(refused).acceptable);
  assert.equal(decide({ ...s, auto: false }, fl, facts(0.2)).gate, 'paused');
  assert.equal(decide({ ...s, rejectBlur: false }, fl, blurry).gate, 'kept');
});

test('the floor comes from the pipeline pair rules', () => {
  const fl = floorOf();
  assert.equal(fl.inliers, 30);
  close(fl.coverage, 0.05, 1e-6);
});

test('keeps early when the support nears its floor', () => {
  const s = defaultSettings();
  const fl = floorOf();
  const w = (o, inliers, coverage) => decide(s, fl, { ...facts(o), support: { inliers, coverage } });
  assert.equal(w(0.7, 45, 0.3).urge.kind, 'band');
  assert.deepEqual(w(0.7, 36, 0.3).urge, { kind: 'due', why: 'support' });
  assert.deepEqual(w(0.7, 100, 0.06).urge, { kind: 'due', why: 'support' });
  assert.ok(w(0.7, 36, 0.3).acceptable);
  assert.deepEqual(w(0.7, 25, 0.3).urge, { kind: 'overdue', why: 'support' });
  assert.ok(!w(0.7, 25, 0.3).acceptable);
  assert.equal(w(0.9, 25, 0.3).urge.kind, 'no');
  const f = { ...facts(0.7), support: { inliers: 70, coverage: 0.3 } };
  assert.equal(decide(s, fl, f).urge.kind, 'no');
  assert.deepEqual(decide(s, floorOf(100), f).urge, { kind: 'due', why: 'support' });
});

test('redundancy keeps when the keyframe before last leaves the view', () => {
  const s = defaultSettings();
  const fl = floorOf();
  const w = (o, o2) => decide(s, fl, { ...facts(o), overlap2: o2 }).urge;
  assert.equal(w(0.75, 0.5).kind, 'no');
  assert.equal(w(0.75, 0.44).kind, 'band');
  assert.deepEqual(w(0.72, 0.38), { kind: 'due', why: 'redundancy' });
  assert.deepEqual(w(0.7, 0.25), { kind: 'overdue', why: 'redundancy' });
  assert.equal(w(0.9, 0.25).kind, 'no');
  assert.equal(decide({ ...s, redundant: false }, fl, { ...facts(0.72), overlap2: 0.3 }).urge.kind, 'no');
});

function replay(frames) {
  const s = defaultSettings();
  const fl = floorOf();
  const p = new Picker();
  const out = [];
  frames.forEach(([o, kp, blurry], i) => {
    const v = decide(s, fl, { ...facts(o), keypoints: kp, blurry });
    const k = p.step(v.urge, v.gate !== 'lost' ? { quality: kp, acceptable: v.acceptable, frame: i } : null);
    if (k) out.push([k.frame, i, k.why, k.relaxed]);
  });
  return out;
}

test('the best frame of the band is kept at the target', () => {
  assert.deepEqual(replay([[0.8, 1500, false], [0.62, 1200, false], [0.6, 1900, false], [0.57, 1000, false],
    [0.54, 1100, false]]), [[2, 4, 'target', false]]);
  assert.deepEqual(replay([[0.6, 1900, false], [0.9, 1000, false], [0.54, 1100, false]]), [[2, 2, 'target', false]]);
});

test('the gates do not eat the overlap', () => {
  assert.deepEqual(replay([[0.62, 400, false], [0.58, 700, false], [0.52, 600, true], [0.47, 500, false],
    [0.44, 500, false]]), [[1, 4, 'target', true]]);
  assert.deepEqual(replay([[0.62, 1900, true], [0.6, 900, false], [0.55, 1900, true]]), [[1, 2, 'target', false]]);
  assert.deepEqual(replay([[0.6, 1900, true], [0.1, 2000, false]]), [[0, 1, 'lost', true]]);
  assert.deepEqual(replay([[0.9, 2000, false], [0.1, 2000, false]]), []);
});

test('keyframes held for relocalization', () => {
  const held = (n) => {
    const seqs = Array.from({ length: n }, (_, k) => k + 1);
    const keep = hold(seqs);
    return seqs.filter((_, k) => keep[k]);
  };
  assert.deepEqual(held(3), [1, 2, 3]);
  assert.deepEqual(held(12), [5, 9, 10, 11, 12]);
  const long = held(100);
  assert.equal(long.length, RECENT_KEYFRAMES + SPARSE_KEYFRAMES);
  assert.deepEqual(long.slice(-4), [97, 98, 99, 100]);
  assert.deepEqual(long.slice(0, 6), [70, 75, 80, 85, 90, 95]);
  assert.ok(hold(long).every((k) => k));
});

test('speed is how fast the overlap falls', () => {
  const sp = new Speed();
  assert.equal(sp.update(null), 0);
  assert.equal(sp.update(1), 0);
  let d = 0;
  for (let k = 1; k <= 10; ++k) d = sp.update(1 - 0.1 * k);
  assert.ok(d > 0.09, `${d}`);
  assert.ok(slowDown(d, false));
  sp.rebase();
  assert.equal(sp.update(1), d);
  const calm = new Speed();
  for (let k = 0; k < 20; ++k) d = calm.update(1 - 0.01 * k);
  assert.ok(!slowDown(d, false));
  assert.ok(!slowDown(d, true));
  assert.ok(slowDown(0.03, true));
});

test('a soft frame is blurry next to the sharp level', () => {
  let b = new Blur();
  for (let k = 0; k < 10; ++k) assert.ok(!b.blurry(100));
  assert.ok(b.blurry(30));
  assert.ok(!b.blurry(80));
  const soft = new Blur();
  for (let k = 0; k < 10; ++k) assert.ok(!soft.blurry(20));
  b = new Blur();
  for (let k = 0; k < 10; ++k) b.blurry(3762);
  for (let k = 0; k < 60; ++k) assert.ok(b.blurry(70));
  assert.ok(b.ratio(70) < 0.1);
  let n = 0;
  while (n < 2000 && b.blurry(70)) ++n;
  assert.ok(n >= 200 && n < 900, `${n}`);
});

test('only confident keypoints count', () => {
  assert.equal(confident([]), 0);
  assert.equal(confident([0.05, 0.19, 0.2, 0.9, 0.31]), 3);
  assert.equal(confident([0.05, 0.079, 0.08, 0.19, 0.31], 'xfeat'), 3);
  const scores = new Float32Array(2048).fill(0.05);
  scores.fill(0.5, 0, 300);
  const v = decide(defaultSettings(), floorOf(), { ...facts(0.5), keypoints: confident(scores) });
  assert.equal(v.gate, 'too_few_keypoints');
  assert.ok(!v.acceptable);
});

function gridPoints(w, h, step) {
  const xy = [];
  for (let y = step / 2; y < h; y += step) for (let x = step / 2; x < w; x += step) xy.push(x, y);
  return xy;
}
function moved(xy, f, keep) {
  const cur = [], pairs = [];
  for (let i = 0; 2 * i < xy.length; ++i) {
    const [x, y] = f(xy[2 * i], xy[2 * i + 1]);
    if (keep(i, x, y)) {
      pairs.push([i, cur.length / 2]);
      cur.push(x, y);
    }
  }
  return [cur, pairs];
}

test('a blank or flat frame cannot anchor the chain', () => {
  const grid = [];
  for (let i = 0; i < 100; ++i) grid.push((i % 10) * 10, Math.floor(i / 10) * 10);
  assert.ok(usableReference(grid));
  assert.ok(!usableReference(grid.slice(0, 2 * (MIN_REFERENCE_KEYPOINTS - 1))));
  assert.ok(!usableReference([]));
  const line = [];
  for (let i = 0; i < 100; ++i) line.push(i, 2 * i);
  assert.ok(!usableReference(line));
});

test('overlap is the share of the keypoint area still in view', () => {
  const w = 1280, h = 720;
  const inside = (x, y) => x >= 0 && x < w && y >= 0 && y < h;
  const xy = gridPoints(w, h, 16);
  const ov = (cur, pairs) => overlap(xy, cur, pairs, w, h);
  let [cur, pairs] = moved(xy, (x, y) => [x, y], () => true);
  close(ov(cur, pairs), 1, 1e-6);
  [cur, pairs] = moved(xy, (x, y) => [x, y], (i) => i % 3 === 0);
  close(ov(cur, pairs), 1, 1e-6);
  assert.equal(ov([], []), 0);
  [cur, pairs] = moved(xy, (x, y) => [x - 512, y], (i, x, y) => i % 3 === 0 && inside(x, y));
  const o = ov(cur, pairs);
  close(o, 0.6, 0.02, 'pan');
  const cur2 = cur.slice(), pairs2 = pairs.slice();
  [0, 1, 2, 3].forEach((i, k) => { pairs2.push([i, cur2.length / 2]); cur2.push(1200 - 40 * k, 700); });
  close(ov(cur2, pairs2), o, 0.02, 'wrong matches');
  [cur, pairs] = moved(xy, (x, y) => [2 * x - 640, 2 * y - 360], (_, x, y) => inside(x, y));
  close(ov(cur, pairs), 0.25, 0.02, 'zoom in');
  [cur, pairs] = moved(xy, (x, y) => [0.5 * x + 320, 0.5 * y + 180], () => true);
  close(ov(cur, pairs), 1, 1e-6, 'zoom out');
  [cur, pairs] = moved(xy, (x, y) => [1280 - x, y], () => true);
  assert.equal(ov(cur, pairs), 0);
  const half = [];
  for (let i = 0; i < xy.length; i += 2) if (xy[i] >= 640) half.push(xy[i], xy[i + 1]);
  [cur, pairs] = moved(half, (x, y) => [x, y], () => true);
  close(overlap(half, cur, pairs, w, h), 1, 1e-6, 'half');
});

test('sharpness drops with blur', () => {
  const w = 320, h = 240;
  const sharp = new Uint8ClampedArray(4 * w * h);
  for (let y = 0; y < h; ++y)
    for (let x = 0; x < w; ++x) {
      const v = (Math.floor(x / 4) + Math.floor(y / 4)) % 2 === 0 ? 230 : 20;
      sharp.set([v, v, v, 255], 4 * (y * w + x));
    }
  // A box blur, 7 px wide, twice.
  let soft = sharp;
  for (let pass = 0; pass < 2; ++pass) {
    const out = new Uint8ClampedArray(soft.length);
    for (let y = 0; y < h; ++y)
      for (let x = 0; x < w; ++x) {
        let s = 0, n = 0;
        for (let dy = -3; dy <= 3; ++dy)
          for (let dx = -3; dx <= 3; ++dx) {
            const xx = x + dx, yy = y + dy;
            if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
            s += soft[4 * (yy * w + xx)];
            ++n;
          }
        const v = s / n;
        out.set([v, v, v, 255], 4 * (y * w + x));
      }
    soft = out;
  }
  assert.ok(sharpness(soft, w, h) < 0.2 * sharpness(sharp, w, h));
});

test('the strongest matches, in order', () => {
  const m = new Int32Array([0, 0, 1, 1, 2, 2, 3, 3]);
  const s = new Float32Array([0.1, 0.9, 0.5, 0.9]);
  assert.deepEqual([...strongest(m, s, 2)], [1, 1, 3, 3]);
  assert.deepEqual([...strongest(m, s, 0)], [...m]);
});

test('the overlap is judged at the next check', () => {
  assert.equal(anticipate(0.7, 0), 0.7);
  close(anticipate(0.7, 0.2), 0.5, 1e-9);
  assert.equal(anticipate(0.2, 0.2), LOST_OVERLAP, 'not anticipated into lost');
  assert.equal(anticipate(0.1, 0.2), 0.1, 'lost stays lost');
  assert.equal(anticipate(null, 0.2), null);
  assert.equal(anticipate(0.7, -0.1), 0.7, 'coming back is not anticipated');
  // At a phone's pace (a drop of 0.2 a check), the frame at 0.7 is kept rather than the next at 0.5.
  assert.deepEqual(decide(defaultSettings(), floorOf(), { ...facts(anticipate(0.7, 0.2)) }).urge, { kind: 'due', why: 'target' });
});
