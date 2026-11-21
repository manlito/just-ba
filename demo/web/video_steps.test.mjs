// Stepping through a video (video_source.js): node --test demo/web/video_steps.test.mjs
import assert from 'node:assert/strict';
import { test } from 'node:test';

import * as G from './gates.js';
import { FrameSelector } from './select.js';
import { W, extractor, frameAt, matcher, strip, verify } from './select_fakes.mjs';
import { BREAK_AFTER, MAX_HALVINGS, Stepper, evenCount, focalFromMm35, parseMoov, selectFrames } from './video_source.js';

const calm = { gate: 'waiting', kept: null, relocalized: false, slowDown: false };
const lost = { gate: 'lost', kept: null, relocalized: false, slowDown: false };
const near = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;

// Runs the stepper with `say(t)` as the selector; the times it looked at.
function run(stepper, say, most = 10000) {
  const seen = [];
  for (let c; seen.length < most && (c = stepper.next());) {
    seen.push(c);
    stepper.report(say(c.t, c));
  }
  return seen;
}

test('a calm video is looked at on the grid', () => {
  const s = new Stepper({ duration: 2, fps: 30, rate: 6 });
  const seen = run(s, () => calm);
  assert.equal(seen.length, 12);
  seen.forEach((c, k) => assert.ok(near(c.t, (k + 0.5) / 6) && !c.restart && !c.close));
  assert.deepEqual(s.stats, { seen: 12, kept: 0, breaks: 0, rewinds: 0, limit: false });
});

test('the grid is never finer than the video\'s frames', () => {
  const s = new Stepper({ duration: 1, fps: 4, rate: 6 });
  assert.deepEqual(run(s, () => calm).map((c) => c.t), [0.125, 0.375, 0.625, 0.875]);
});

test('"slow down" halves the step going on, and calm frames double it back', () => {
  const s = new Stepper({ duration: 100, fps: 30, rate: 6 });
  const slow = { ...calm, slowDown: true };
  const seen = run(s, (t) => (t < 1 ? slow : calm), 40);
  const steps = seen.slice(1).map((c, k) => c.t - seen[k].t);
  // 1/6 once (the step that was pending), then 1/12, 1/24, and 1/30: a frame, the floor.
  assert.ok(near(steps[0], 1 / 12) && near(steps[1], 1 / 24) && near(steps[2], 1 / 30));
  assert.equal(s.halvings, 0);
  const after = seen.findIndex((c) => c.t >= 1);
  const back = seen.slice(after).map((c, k, a) => (k ? c.t - a[k - 1].t : null)).slice(1);
  // Four calm frames at each step on the way back: three more at a frame, then 4 at 1/24, 4 at 1/12, then 1/6.
  assert.ok(back.slice(0, 3).every((d) => near(d, 1 / 30)));
  assert.ok(back.slice(3, 7).every((d) => near(d, 1 / 24)));
  assert.ok(back.slice(7, 11).every((d) => near(d, 1 / 12)));
  assert.ok(back.slice(11, 15).every((d) => near(d, 1 / 6)));
});

test('at most three halvings', () => {
  const s = new Stepper({ duration: 100, fps: 240, rate: 6 });
  run(s, () => ({ ...calm, slowDown: true }), 20);
  assert.equal(s.halvings, MAX_HALVINGS);
  assert.ok(near(s.step, 1 / 48));
});

test('a lost frame is looked for again from the last one that was not, at half the step', () => {
  const s = new Stepper({ duration: 100, fps: 30, rate: 6 });
  // The view is lost when the stepper jumps more than 0.1 s at once past t = 1.
  let last = 0;
  const seen = run(s, (t) => {
    const r = t > 1 && t - last > 0.1 ? lost : calm;
    if (r === calm) last = t;
    return r;
  }, 10);
  const k = seen.findIndex((c) => c.t > 1);
  assert.equal(k, 6);
  assert.ok(near(seen[k].t, 6.5 / 6));               // lost: a 1/6 step past 1
  assert.ok(near(seen[k + 1].t, 5.5 / 6 + 1 / 12));  // back to 5.5/6, half the step: found
  assert.ok(near(seen[k + 2].t, 5.5 / 6 + 2 / 12));
  assert.equal(s.stats.rewinds, 1);
  assert.equal(s.stats.breaks, 0);
  // Never back before the last good time.
  seen.slice(k).forEach((c) => assert.ok(c.t > 5.5 / 6));
});

test('still lost a second on at the smallest step: the chain is broken', () => {
  const s = new Stepper({ duration: 100, fps: 30, rate: 6 });
  // A cut at t = 2: everything after is lost until the selector starts again.
  let broken = false;
  const seen = run(s, (t, c) => {
    if (c.restart) broken = true;
    if (t < 2) s.lastGoodSeen = Math.max(s.lastGoodSeen ?? 0, t);
    return t >= 2 && !broken ? lost : calm;
  }, 200);
  const k = seen.findIndex((c) => c.t >= 2);
  const good = seen[k - 1].t;
  // Three looks back towards the cut, at 1/12, 1/24, then a frame (1/48 is under it); the
  // last good time ends within a frame of it.
  assert.ok(near(seen[k].t - good, 1 / 6) && near(seen[k + 1].t, good + 1 / 12));
  assert.equal(s.stats.rewinds, 3);
  assert.ok(2 - s.lastGoodSeen < 1 / 30 + 1e-9);
  const r = seen.findIndex((c) => c.restart);
  assert.ok(r > 0);
  assert.equal(s.stats.breaks, 1);
  // From the first lost frame at the smallest step to the restart: a second of video, a frame at a time.
  const smallest = seen.slice(k, r).filter((c, i, a) => i > 0 && near(c.t - a[i - 1].t, 1 / 30));
  assert.ok(smallest.length >= 29 && smallest.length <= 32, String(smallest.length));
  assert.ok(seen[r].t - 2 >= BREAK_AFTER - 0.1 && seen[r].t - 2 < BREAK_AFTER + 0.2);
  // After the break the step is the grid's again.
  assert.ok(near(seen[r + 2].t - seen[r + 1].t, 1 / 6));
  assert.equal(seen.filter((c) => c.restart).length, 1);
});

test('a lost frame that kept one or found a keyframe again is not looked for again', () => {
  for (const r of [{ gate: 'kept', why: 'lost', kept: {}, keptT: null, relocalized: false, slowDown: false }, { ...lost, relocalized: true }]) {
    const s = new Stepper({ duration: 1, fps: 30, rate: 6 });
    assert.equal(run(s, () => r).length, 6);
    assert.equal(s.stats.rewinds, 0);
  }
});

test('after a keep made by a lost frame, the next lost one is looked for from the kept frame', () => {
  const s = new Stepper({ duration: 100, fps: 30, rate: 6 });
  // Calm to 1 s; the frame at 6.5/6 loses the keyframe and the selector keeps the one it held, from 0.75 s;
  // the frame after is lost too.
  const seen = run(s, (t) => (t < 1 ? calm : near(t, 6.5 / 6) ? { gate: 'kept', why: 'lost', kept: {}, keptT: 0.75, relocalized: false, slowDown: false } : lost), 9);
  assert.ok(near(seen[6].t, 6.5 / 6) && near(seen[7].t, 7.5 / 6));
  assert.ok(near(seen[8].t, 0.75 + 1 / 12), `back to the kept frame, half a step on: ${seen[8].t}`);
  // A keep for another reason leaves the good time at the frame looked at.
  const o = new Stepper({ duration: 100, fps: 30, rate: 6 });
  const other = run(o, (t) => (t < 1 ? { gate: 'kept', why: 'target', kept: {}, keptT: t - 0.1, relocalized: false, slowDown: false } : lost), 8);
  assert.ok(near(other[7].t, 5.5 / 6 + 1 / 12));
});

test('evenly: windows, their frames, and the close', () => {
  assert.equal(evenCount(5), 20);
  assert.equal(evenCount(30), 60);
  assert.equal(evenCount(600), 150);
  // 10 s in 5 windows of 2 s: five frames each, at sixths.
  let s = new Stepper({ duration: 10, fps: 30, mode: 'evenly', count: 5 });
  let seen = run(s, () => calm);
  assert.equal(seen.length, 25);
  assert.ok(seen.slice(0, 5).every((c, j) => near(c.t, 2 * (j + 1) / 6)));
  assert.deepEqual(seen.map((c) => c.close), Array.from({ length: 25 }, (_, k) => k % 5 === 4));
  assert.ok(seen.every((c) => !c.restart));
  // 6 s by default: 20 windows of 0.3 s, three frames each at 25, 50 and 75 %.
  s = new Stepper({ duration: 6, fps: 30, mode: 'evenly' });
  seen = run(s, () => calm);
  assert.equal(seen.length, 60);
  assert.ok(near(seen[0].t, 0.075) && near(seen[1].t, 0.15) && near(seen[2].t, 0.225) && seen[2].close && !seen[1].close);
  // One every 2 s; windows shorter than four frames have one frame, the middle.
  s = new Stepper({ duration: 9, fps: 30, mode: 'evenly', interval: 2 });
  assert.equal(run(s, () => calm).filter((c) => c.close).length, 4);
  s = new Stepper({ duration: 1, fps: 30, mode: 'evenly', count: 10 });
  seen = run(s, () => calm);
  assert.equal(seen.length, 10);
  assert.ok(seen.every((c) => c.close) && near(seen[0].t, 0.05));
});

test('it ends once enough frames are kept', () => {
  for (const mode of ['overlap', 'evenly']) {
    const s = new Stepper({ duration: 100, fps: 30, mode, maxFrames: 7 });
    const seen = run(s, (t, c) => ({ ...calm, kept: mode === 'overlap' || c.close ? {} : null }));
    assert.equal(s.stats.kept, 7);
    assert.equal(s.stats.limit, true);  // the video had more: a chain cut short, or 7 windows for the 150 asked
    assert.ok(seen.at(-1).t < 100);
  }
  // A video that ends first is not cut short.
  const s = new Stepper({ duration: 1, fps: 30, maxFrames: 7 });
  run(s, () => ({ ...calm, kept: {} }));
  assert.equal(s.stats.kept, 6);
  assert.equal(s.stats.limit, false);
});

// ---- the stepper and the selector together, on the generated strip

test('a video of the strip: fast stretches looked at again, a cut breaks the chain', async () => {
  // The camera's place at time t: slow, a fast stretch (2.5 frame widths a second: lost at
  // six looks a second), slow again, a cut to another place at t = 8, slow to the end.
  const x = (t) => (t < 3 ? 0.3 * t : t < 4 ? 0.9 + 2.5 * (t - 3) : t < 8 ? 3.4 + 0.3 * (t - 4) : 30 + 0.3 * (t - 8)) * W;
  const pts = strip(40 * W);
  const m = matcher();
  const selector = new FrameSelector({ extractor: extractor(pts), matcher: m, matcherKind: 'lightglue', verify, anticipate: false,
                                       settings: { ...G.defaultSettings(), minKeypoints: 100 } });
  const stepper = new Stepper({ duration: 12, fps: 30, rate: 6 });
  const kept = [], restarts = [];
  const stats = await selectFrames({
    stepper, selector, frameAt: async (t) => ({ image: frameAt(x(t)), t }),
    onStep: (c, r) => {
      if (c?.restart) restarts.push(c.t);
      if (r.kept) kept.push({ t: r.kept.tag.t, why: r.why });
    },
  });
  assert.equal(stats.breaks, 1);
  assert.equal(restarts.length, 1);
  assert.ok(restarts[0] > 8.9 && restarts[0] < 9.3, `the chain breaks about a second after the cut: ${restarts[0]}`);
  assert.equal(kept.filter((k) => k.why === 'first').length, 2);
  // Every keyframe but the two first ones sees its predecessor: no gap in either chain.
  for (let k = 1; k < kept.length; k++) {
    if (kept[k].why === 'first') continue;
    const shift = Math.abs(x(kept[k].t) - x(kept[k - 1].t)) / W;
    assert.ok(shift < 0.6, `keyframe at ${kept[k].t.toFixed(2)} s is ${shift.toFixed(2)} of a frame from the one before`);
  }
  // The fast stretch is covered: keyframes inside it, found by looking again at a finer step.
  assert.ok(kept.filter((k) => k.t > 3 && k.t < 4).length >= 4);
  assert.ok(stats.seen > 12 * 6);
  selector.dispose();
  assert.equal(m.alive, kept.length);
});

test('evenly, on the strip: one keyframe a window, no matching', async () => {
  const pts = strip(40 * W);
  const m = matcher();
  const selector = new FrameSelector({ extractor: extractor(pts), matcher: m, matcherKind: 'lightglue', verify,
                                       settings: { ...G.defaultSettings(), minKeypoints: 100 } });
  const stepper = new Stepper({ duration: 10, fps: 30, mode: 'evenly', count: 8 });
  let kept = 0;
  const stats = await selectFrames({ stepper, selector, frameAt: async (t) => ({ image: frameAt(30 * t), t }), onStep: (c, r) => { kept += r.kept ? 1 : 0; } });
  assert.equal(kept, 8);
  assert.equal(stats.kept, 8);
  assert.equal(m.matched, 0);
});

test('the best frame since the last keyframe is kept at the end, and on Stop', async () => {
  const make = () => new FrameSelector({ extractor: extractor(strip(8 * W)), matcher: matcher(), matcherKind: 'lightglue', verify, anticipate: false,
                                         settings: { ...G.defaultSettings(), minKeypoints: 100, redundant: false } });
  // 0.44 frame widths in 2 s: the first frame kept, then the overlap falls into the band (0.63 at 1.7 s) and never to the target.
  const world = async (t) => ({ image: frameAt(0.22 * t * W), t });
  for (const stopAt of [Infinity, 11]) {
    const kept = [];
    let n = 0;
    const stats = await selectFrames({ stepper: new Stepper({ duration: 2, fps: 30 }), selector: make(), frameAt: world, cancelled: () => n >= stopAt,
                                       onStep: (c, r) => { n++; if (r.kept) kept.push({ why: r.why, t: r.kept.tag.t, last: c == null }); } });
    assert.deepEqual(kept.map((k) => k.why), ['first', 'end']);
    assert.equal(kept[1].last, true);
    assert.ok(kept[1].t > 1.7 && kept[1].t < (stopAt === 11 ? 1.8 : 2), String(kept[1].t));
    assert.equal(stats.kept, 2);
  }
  // Not when enough were kept already.
  const kept = [];
  const stats = await selectFrames({ stepper: new Stepper({ duration: 2, fps: 30, maxFrames: 1 }), selector: make(), frameAt: world,
                                     onStep: (c, r) => { if (r.kept) kept.push(r.why); } });
  assert.deepEqual(kept, ['first']);
  assert.equal(stats.limit, true);
  // Nothing held, nothing kept.
  const s = make();
  await s.step(frameAt(0));
  assert.equal(s.flush(), null);
});

test('Stop ends the selection where it is', async () => {
  const selector = new FrameSelector({ extractor: extractor(strip(4 * W)), matcher: matcher(), matcherKind: 'lightglue', verify,
                                       settings: { ...G.defaultSettings(), minKeypoints: 100 } });
  let n = 0;
  const stats = await selectFrames({ stepper: new Stepper({ duration: 10, fps: 30 }), selector, frameAt: async (t) => ({ image: frameAt(0), t }),
                                     onStep: () => { n++; }, cancelled: () => n >= 3 });
  assert.equal(stats.seen, 3);
});

// ---- the file's metadata

const u32 = (v) => [v >>> 24, (v >>> 16) & 255, (v >>> 8) & 255, v & 255];
const str = (s) => [...new TextEncoder().encode(s)];
const box = (type, ...parts) => {
  const body = parts.flat();
  return [...u32(8 + body.length), ...str(type), ...body];
};
const full = (type, ...parts) => box(type, [0, 0, 0, 0], ...parts);
const f32 = (x) => [...new Uint8Array(new Float32Array([x]).buffer)].reverse();

// A moov's content: a 4 s movie, a video track of 120 frames (hvc1), QuickTime keys.
function moov({ keys = true, iso = false, wide = false, udta = false } = {}) {
  const mvhd = full('mvhd', u32(0), u32(0), u32(600), u32(2400), new Array(80).fill(0));
  const mdhd = full('mdhd', u32(0), u32(0), u32(30000), u32(120000), [0, 0, 0, 0]);
  const hdlr = (kind) => full('hdlr', u32(0), str(kind), new Array(12).fill(0), [0]);
  const stsd = full('stsd', u32(1), box('hvc1', new Array(78).fill(0)));
  const stts = full('stts', u32(2), u32(100), u32(1000), u32(20), u32(1000));
  const video = box('trak', full('tkhd', new Array(80).fill(0)), box('mdia', mdhd, hdlr('vide'), box('minf', box('stbl', stsd, stts))));
  const sound = box('trak', box('mdia', full('mdhd', u32(0), u32(0), u32(44100), u32(176400), [0, 0, 0, 0]), hdlr('soun'),
                                box('minf', box('stbl', full('stts', u32(1), u32(999), u32(1))))));
  const names = ['com.apple.quicktime.make', 'com.apple.quicktime.model', 'com.apple.quicktime.camera.focal_length.35mm_equivalent',
                 'com.apple.quicktime.camera.lens_model'];
  const key = (n) => [...u32(8 + n.length), ...str('mdta'), ...str(n)];
  const item = (index, kind, value) => [...u32(8 + 16 + value.length), ...u32(index), ...box('data', u32(kind), u32(0), value)];
  const inside = [hdlr('mdta'), full('keys', u32(names.length), names.map(key).flat()),
                  box('ilst', item(1, 1, str('Apple')), item(2, 1, str('iPhone 17 Pro')), item(3, wide ? 23 : 1, wide ? f32(13) : str('26')),
                      item(4, 1, str('iPhone 17 Pro back camera 6.765mm f/1.78')))];
  const meta = iso ? full('meta', ...inside) : box('meta', ...inside);
  return new Uint8Array([...mvhd, ...sound, ...video, ...(keys ? (udta ? box('udta', meta) : meta) : [])]);
}

test('a video\'s frame rate, codec and camera come from its boxes', () => {
  const m = parseMoov(moov());
  assert.equal(m.duration, 4);
  assert.equal(m.frames, 120);
  assert.equal(m.fps, 30);
  assert.equal(m.codec, 'hvc1');
  assert.equal(m.make, 'Apple');
  assert.equal(m.model, 'iPhone 17 Pro');
  assert.equal(m.mm35, 26);
  assert.ok(m.lens.startsWith('iPhone 17 Pro back camera'));
  // ISO's meta (a full box) and a float value read the same.
  assert.equal(parseMoov(moov({ iso: true })).mm35, 26);
  assert.equal(parseMoov(moov({ wide: true })).mm35, 13);
  // ffmpeg writes the keys into the movie's user data.
  assert.equal(parseMoov(moov({ iso: true, udta: true })).make, 'Apple');
  // No keys: no focal length, the rest stands.
  const bare = parseMoov(moov({ keys: false }));
  assert.equal(bare.mm35, null);
  assert.equal(bare.make, null);
  assert.equal(bare.fps, 30);
  // An ArrayBuffer, and bytes at an offset of a larger buffer.
  const bytes = moov();
  assert.equal(parseMoov(bytes.buffer.slice(0)).mm35, 26);
  const shifted = new Uint8Array(bytes.length + 5);
  shifted.set(bytes, 5);
  assert.equal(parseMoov(shifted.subarray(5)).model, 'iPhone 17 Pro');
});

test('a cut-off or damaged moov never throws', () => {
  const bytes = moov();
  for (let n = 0; n <= bytes.length; n += 3) parseMoov(bytes.slice(0, n));
  const noise = bytes.slice();
  for (let k = 0; k < noise.length; k += 7) noise[k] = 255 - noise[k];
  parseMoov(noise);
  assert.equal(parseMoov(new Uint8Array(0)).fps, null);
});

test('a 35 mm equivalent as pixels', () => {
  // 26 mm on a 3840-wide frame: about 2885 px (Like Recon's video::focal_prior).
  assert.ok(Math.abs(focalFromMm35(26, 3840) - 2885) < 1);
  assert.ok(Math.abs(focalFromMm35(13, 1280) / focalFromMm35(26, 1280) - 0.5) < 1e-12);
  assert.ok(Number.isNaN(focalFromMm35(0.5, 1280)));
  assert.ok(Number.isNaN(focalFromMm35(5000, 1280)));
});
