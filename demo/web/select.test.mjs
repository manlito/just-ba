// The keep loop apart from its page (select.js), on a generated scene: node --test demo/web/select.test.mjs
import assert from 'node:assert/strict';
import { test } from 'node:test';

import * as G from './gates.js';
import { FrameSelector, minKeypointsFor } from './select.js';
import { W, extractor, frameAt, matcher, strip, verify } from './select_fakes.mjs';

function selector(pts, more = {}) {
  const m = matcher();
  const s = new FrameSelector({
    extractor: extractor(pts), matcher: m, matcherKind: 'lightglue', verify, anticipate: false,
    settings: { ...G.defaultSettings(), minKeypoints: 100 }, ...more,
  });
  return { s, m };
}

test('keyframes are kept as the overlap reaches the target', async () => {
  const { s, m } = selector(strip(12 * W), { settings: { ...G.defaultSettings(), minKeypoints: 100, redundant: false } });
  const kept = [];
  for (let x = 0; x <= 6 * W; x += 0.02 * W) {
    const r = await s.step(frameAt(x), { tag: { x } });
    if (r.kept) kept.push(r);
    else assert.ok(['waiting'].includes(r.gate), r.gate);
  }
  assert.equal(kept[0].why, 'first');
  assert.equal(kept[0].kept.tag.x, 0);
  assert.ok(kept.length >= 12 && kept.length <= 15, `${kept.length} keyframes over six frame widths`);
  for (let k = 1; k < kept.length; k++) {
    assert.equal(kept[k].why, 'target');
    assert.equal(kept[k].kept.index, k);
    // Each one sees about the target of the one before: 1 - shift / W.
    const seen = 1 - (kept[k].kept.tag.x - kept[k - 1].kept.tag.x) / W;
    assert.ok(seen > 0.5 && seen < 0.64, `keyframe ${k} sees ${seen.toFixed(2)} of the one before`);
  }
  // Only the keyframes' features are alive: every other frame's were freed.
  assert.equal(m.alive, kept.length + (s.picker.holds() ? 1 : 0));
  assert.deepEqual(s.measured().slice(0, 2), [kept.at(-1).kept, kept.at(-2).kept]);
  s.dispose();
  assert.equal(m.alive, kept.length);
});

test('a keyframe is kept when the one before last leaves the view', async () => {
  const { s } = selector(strip(12 * W));
  const kept = [];
  for (let x = 0; x <= 3 * W; x += 0.02 * W) {
    const r = await s.step(frameAt(x), { tag: { x } });
    if (r.kept) kept.push(r);
  }
  assert.deepEqual(kept.slice(0, 3).map((r) => r.why), ['first', 'target', 'redundancy']);
  // At the redundancy keep the first keyframe is down to REDUNDANT_OVERLAP.
  assert.ok(Math.abs(1 - kept[2].kept.tag.x / W - G.REDUNDANT_OVERLAP) < 0.03);
  assert.ok(kept[2].overlap2 <= G.REDUNDANT_OVERLAP + 1e-6);
});

test('a broken chain starts again with a first keyframe', async () => {
  const { s } = selector(strip(40 * W));
  for (let x = 0; x <= 0.2 * W; x += 0.02 * W) await s.step(frameAt(x));
  // The video cuts to another place: lost, nothing kept, for as long as it stays there.
  for (let k = 0; k < 5; k++) {
    const r = await s.step(frameAt(20 * W + k));
    assert.equal(r.gate, 'lost');
    assert.equal(r.kept, null);
    assert.equal(r.relocalized, false);
  }
  s.breakChain();
  const r = await s.step(frameAt(20 * W + 5));
  assert.equal(r.gate, 'kept');
  assert.equal(r.why, 'first');
  assert.equal(r.overlap, null);
  // The chain goes on from there.
  const next = await s.step(frameAt(20 * W + 0.1 * W));
  assert.equal(next.gate, 'waiting');
  assert.ok(Math.abs(next.overlap - 0.9) < 0.05);
});

test('a frame back at an earlier keyframe finds it again', async () => {
  const { s } = selector(strip(12 * W));
  const kept = [];
  for (let x = 0; x <= 1.6 * W; x += 0.02 * W) {
    const r = await s.step(frameAt(x), { tag: { x } });
    if (r.kept) kept.push(r.kept);
  }
  assert.ok(kept.length >= 4);
  assert.ok(s.held.includes(kept.at(-4)) && s.reference === kept.at(-1));
  // Nothing held by the picker, then a jump back to an earlier keyframe's place.
  s.clearPicker();
  const there = kept.at(-4);
  const back = await s.step(frameAt(there.tag.x - 0.08 * W));
  assert.equal(back.relocalized, true);
  assert.equal(s.reference, there);
  assert.ok(back.overlap > 0.9);
  assert.equal(s.before, null);
  assert.equal(back.kept, null);
});

test('a keyframe kept from a re-localized frame names the keyframe it found', async () => {
  const { s } = selector(strip(12 * W));
  const kept = [];
  for (let x = 0; x <= 1.6 * W; x += 0.02 * W) {
    const r = await s.step(frameAt(x), { tag: { x } });
    if (r.kept) kept.push(r.kept);
  }
  s.clearPicker();
  // Back by an earlier keyframe, at an overlap already in the band: held, then kept.
  const there = kept.at(-4);
  let r = await s.step(frameAt(there.tag.x - 0.4 * W));
  assert.equal(r.relocalized, true);
  assert.equal(s.reference, there);
  r = await s.step(frameAt(there.tag.x - 0.47 * W));
  assert.equal(r.gate, 'kept');
  assert.equal(r.kept.relocalizedTo, there);
  assert.equal(s.reference, r.kept);
});

test('the keyframes held are the newest four and every fifth of the older, up to six', async () => {
  const { s } = selector(strip(60 * W), { settings: { ...G.defaultSettings(), minKeypoints: 100, redundant: false } });
  const all = [];
  for (let x = 0; all.length < 45; x += 0.05 * W) {
    const r = await s.step(frameAt(x));
    if (!r.kept) continue;
    all.push(r.kept);
    const keep = G.hold(all.map((k) => k.seq));
    assert.deepEqual(s.held, all.filter((_, i) => keep[i]));
  }
  assert.equal(s.held.length, G.RECENT_KEYFRAMES + G.SPARSE_KEYFRAMES);
});

test('too little texture and blur refuse a first keyframe', async () => {
  const { s } = selector(strip(12 * W, 40));
  assert.equal((await s.step(frameAt(0))).gate, 'too_few_keypoints');
  const { s: sharp } = selector(strip(12 * W));
  assert.equal((await sharp.step(frameAt(0))).gate, 'kept');
  const blurry = await sharp.step(frameAt(0.3 * W, { blur: true }));
  assert.equal(blurry.gate, 'blurry');
});

test('a window keeps its best frame when it closes, whatever it overlaps', async () => {
  const { s, m } = selector(strip(40 * W));
  const matchedBefore = m.matched;
  assert.equal((await s.step(frameAt(0), { window: { close: false }, tag: 'a' })).gate, 'waiting');
  assert.equal((await s.step(frameAt(5 * W, { blur: true }), { window: { close: false }, tag: 'b' })).gate, 'blurry');
  const r = await s.step(frameAt(9 * W, { blur: true }), { window: { close: true }, tag: 'c' });
  assert.equal(r.gate, 'kept');
  assert.equal(r.why, 'window');
  assert.equal(r.kept.tag, 'a');
  assert.equal(m.matched, matchedBefore, 'no frame is matched');
  // A window whose every frame is refused keeps the best of them.
  await s.step(frameAt(12 * W, { blur: true }), { window: { close: false }, tag: 'd' });
  const refused = await s.step(frameAt(13 * W, { blur: true }), { window: { close: true }, tag: 'e' });
  assert.equal(refused.gate, 'kept');
  assert.equal(refused.kept.tag, 'e');
});

test('Like Recon\'s keypoint floor follows the frame size', () => {
  assert.equal(minKeypointsFor(1280, 720), 800);
  assert.equal(minKeypointsFor(1920, 1080), 800);
  assert.equal(minKeypointsFor(640, 360), 200);
  assert.equal(minKeypointsFor(320, 180), 150);
});
