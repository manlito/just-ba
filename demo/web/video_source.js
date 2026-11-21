// Videos as a source of frames for the live demo (D167): a video file opened in a
// <video> element and stepped through by seeking (not played), the times to look at
// chosen by a Stepper from what select.js's FrameSelector says of each frame, and the
// file's own metadata (frame rate, the camera's 35 mm equivalent focal length) read from
// its boxes. The Stepper and the metadata reader are pure (video_steps.test.mjs); Like
// Recon's video import steps the same way (its src/lib/video/stepper.ts mirrors Stepper).

/** Frames looked at per second of video, before the stepper adapts. */
export const RATE = 6;
/** At most this many frames kept from one video. */
export const MAX_FRAMES = 300;
/** The step is halved at most this many times (and never below one frame). */
export const MAX_HALVINGS = 3;
/** After this many calm frames in a row (not lost, no "slow down") the step doubles back. */
export const CALM_FRAMES = 4;
/** Lost for this many seconds of video at the smallest step: the chain is broken. */
export const BREAK_AFTER = 1;

/** The frames to keep from `seconds` of video sampled evenly: two a second, 20 to 150. */
export function evenCount(seconds) {
  return Math.min(150, Math.max(20, Math.round(2 * seconds)));
}

/**
 * The times of a video to look at, one at a time: next() gives { t, restart, close } (or
 * null at the end), report() takes what the selector said of that frame ({ gate, why,
 * kept, keptT (the kept frame's own time), relocalized, slowDown }).
 *
 * mode 'overlap' (the keep rules decide): times (k + 0.5) / rate, then adapted.
 * - A frame that comes back lost, with nothing kept and no re-localization: back to the
 *   last time that was not lost, at half the step (a live capture says "slow down"; a
 *   video can be looked at again). At most MAX_HALVINGS halvings, never under 1 / fps.
 * - A lost frame that made the selector keep the frame it held (why 'lost'): the last
 *   good time is that kept frame's, so a next lost frame is looked for from there.
 * - "Slow down": the step is halved, going on.
 * - After CALM_FRAMES frames in a row neither lost nor "slow down", the step doubles
 *   back towards 1 / rate.
 * - Still lost BREAK_AFTER seconds of video on at the smallest step (a live capture says
 *   "move back"; a video can't): the next frame comes with restart, for the selector's
 *   breakChain(), and the step starts over.
 *
 * mode 'evenly' (no matcher needed): `count` windows (default evenCount(duration); or
 * one every `interval` seconds), in each 3 frames at 25, 50 and 75 % (5, at sixths, when
 * a window is a second or longer; 1, the middle, when it is shorter than 4 frames); the
 * last of a window comes with close, where the selector keeps the window's best.
 *
 * Either way at most maxFrames are kept: stats.limit says the video had more to give (a
 * chain stopped before its end, or fewer windows than asked for).
 */
export class Stepper {
  constructor({ duration, fps = 30, rate = RATE, mode = 'overlap', maxFrames = MAX_FRAMES, count = null, interval = null }) {
    this.duration = duration;
    this.mode = mode;
    this.maxFrames = maxFrames;
    this.frame = 1 / (fps > 0 ? fps : 30);
    this.stats = { seen: 0, kept: 0, breaks: 0, rewinds: 0, limit: false };
    this.current = null;
    if (mode === 'evenly') {
      const n = interval > 0 ? Math.floor(duration / interval) : count ?? evenCount(duration);
      this.windows = Math.max(1, Math.min(n, maxFrames));
      this.capped = n > maxFrames;
      this.width = duration / this.windows;
      this.perWindow = this.width < 4 * this.frame ? 1 : this.width >= 1 ? 5 : 3;
      this.window = 0;
      this.inWindow = 0;
    } else {
      this.base = Math.max(1 / rate, this.frame);
      this.halvings = 0;
      this.upcoming = 0.5 * this.base;
      this.lastGood = null;   // the last time that was not lost
      this.lostFrom = null;   // the first lost time at the smallest step
      this.calm = 0;
      this.restart = false;
    }
  }

  /** The step now, seconds (overlap). */
  get step() { return Math.max(this.base / 2 ** this.halvings, this.frame); }
  canHalve() { return this.halvings < MAX_HALVINGS && this.step > this.frame * (1 + 1e-9); }

  next() {
    this.current = this.mode === 'evenly' ? this.nextWindowed() : this.nextChained();
    return this.current;
  }

  report(r) {
    this.stats.seen++;
    if (r.kept) this.stats.kept++;
    if (this.mode === 'evenly') this.reportWindowed();
    else this.reportChained(r, this.current.t);
  }

  nextWindowed() {
    if (this.window >= this.windows) {
      this.stats.limit = this.capped;
      return null;
    }
    const t = (this.window + (this.inWindow + 1) / (this.perWindow + 1)) * this.width;
    return { t, restart: false, close: this.inWindow === this.perWindow - 1 };
  }

  reportWindowed() {
    if (++this.inWindow >= this.perWindow) {
      this.inWindow = 0;
      this.window++;
    }
  }

  nextChained() {
    if (!(this.upcoming < this.duration)) return null;
    if (this.stats.kept >= this.maxFrames) {
      this.stats.limit = true;
      return null;
    }
    const c = { t: this.upcoming, restart: this.restart, close: false };
    this.restart = false;
    return c;
  }

  reportChained(r, t) {
    if (r.gate === 'lost' && !r.kept && !r.relocalized) {
      this.calm = 0;
      if (this.lastGood != null && this.canHalve()) {
        this.halvings++;
        this.stats.rewinds++;
        this.upcoming = this.lastGood + this.step;
        return;
      }
      this.lostFrom ??= t;
      if (t - this.lostFrom >= BREAK_AFTER - 1e-9) {
        this.restart = true;
        this.stats.breaks++;
        this.halvings = 0;
        this.lostFrom = null;
        this.lastGood = null;
      }
      this.upcoming = t + this.step;
      return;
    }
    // A frame that lost the keyframe and made the selector keep the one it held is not
    // itself known to see that one: the good time is the kept frame's.
    this.lastGood = r.kept && r.why === 'lost' && r.keptT != null ? Math.min(r.keptT, t) : t;
    this.lostFrom = null;
    if (r.slowDown) {
      this.calm = 0;
      if (this.canHalve()) this.halvings++;
    } else if (++this.calm >= CALM_FRAMES && this.halvings > 0) {
      this.halvings--;
      this.calm = 0;
    }
    this.upcoming = t + this.step;
  }
}

/**
 * A video through the stepper and the selector. frameAt(t) -> { image (ImageData), t (the
 * frame's own time) }; onStep(candidate, result, time) after each frame, where a kept
 * keyframe is taken (result.kept, its tag { t }); cancelled() true stops. At the end of
 * the video, and on a stop, the frame the selector still held as the best since the last
 * keyframe is kept too (onStep(null, result, its time)); not when maxFrames ended it.
 * Returns the stepper's stats.
 */
export async function selectFrames({ stepper, selector, frameAt, onStep = null, cancelled = () => false }) {
  for (let c; !cancelled() && (c = stepper.next());) {
    const { image, t } = await frameAt(c.t);
    if (c.restart) selector.breakChain();
    const r = await selector.step(image, { window: stepper.mode === 'evenly' ? { close: c.close } : null, tag: { t } });
    r.keptT = r.kept?.tag?.t ?? null;
    stepper.report(r);
    if (onStep) await onStep(c, r, t);
  }
  if (!stepper.stats.limit) {
    const r = selector.flush();
    if (r) {
      stepper.stats.kept++;
      if (onStep) await onStep(null, r, r.kept.tag.t);
    }
  }
  return stepper.stats;
}

// ---- the <video> element

// Waits for one of `events` on `target`, up to ms, checking cancelled() as it waits.
function waitFor(target, events, ms, cancelled = () => false) {
  return new Promise((resolve, reject) => {
    const off = () => {
      clearTimeout(timer);
      clearInterval(poll);
      for (const e of events) target.removeEventListener(e, on);
      target.removeEventListener('error', bad);
    };
    const on = (e) => { off(); resolve(e.type); };
    const bad = () => { off(); reject(new Error('this browser cannot play the video' + (target.error?.message ? `: ${target.error.message}` : ''))); };
    const timer = setTimeout(() => { off(); reject(new Error('the video took too long to open')); }, ms);
    const poll = setInterval(() => { if (cancelled()) { off(); reject(new Error('stopped')); } }, 100);
    for (const e of events) target.addEventListener(e, on);
    target.addEventListener('error', bad);
  });
}

/**
 * A video (a File, a Blob or a URL) in a paused, muted <video>, its first frame ready.
 * Throws when it can't be played, has no picture, has no length the browser can tell (a
 * stream recorded without one), takes over timeoutMs, or cancelled() turns true
 * ('stopped'); the element is left empty then.
 */
export async function openVideo(source, video = document.createElement('video'), { timeoutMs = 30000, cancelled = () => false } = {}) {
  const url = typeof source === 'string' ? source : URL.createObjectURL(source);
  const close = () => {
    video.removeAttribute('src');
    video.load();
    if (typeof source !== 'string') URL.revokeObjectURL(url);
  };
  try {
    video.muted = true;
    video.playsInline = true;
    video.preload = 'auto';
    if (typeof source === 'string') video.crossOrigin = 'anonymous';
    const loaded = waitFor(video, ['loadeddata'], timeoutMs, cancelled);
    video.src = url;
    await loaded;
    video.pause();
    if (!video.videoWidth) throw new Error('the video has no picture this browser can decode');
    if (!Number.isFinite(video.duration)) {
      // A recording without a length in its header: the browser finds it by seeking to the end.
      const found = waitFor(video, ['durationchange', 'seeked'], Math.min(timeoutMs, 5000), cancelled).catch((e) => { if (e.message === 'stopped') throw e; });
      video.currentTime = 1e9;
      await found;
    }
    if (!Number.isFinite(video.duration) || !(video.duration > 0)) throw new Error('the video does not say how long it is');
    await seekTo(video, 0);
  } catch (e) {
    close();
    throw e;
  }
  return { video, duration: video.duration, width: video.videoWidth, height: video.videoHeight, close };
}

/** A frame is taken this long after 'seeked' when no frame callback says it is shown. */
const SEEKED_GRACE_MS = 200;

/**
 * Seeks a paused video to t; resolves with the time of the frame then shown. 'seeked'
 * comes before the frame is presented, so the frame callback is waited for where there
 * is one and the page is visible (its mediaTime is the frame's own timestamp); a short
 * while after 'seeked' otherwise, or when no callback comes (the same frame again).
 */
export function seekTo(video, t, timeoutMs = 5000) {
  return new Promise((resolve, reject) => {
    let done = false, grace = null;
    const finish = (time) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      clearTimeout(grace);
      video.removeEventListener('seeked', onSeeked);
      resolve(time);
    };
    const frames = !!video.requestVideoFrameCallback && !document.hidden;
    const onSeeked = () => { grace = setTimeout(() => finish(video.currentTime), frames ? SEEKED_GRACE_MS : 30); };
    // A frame shown while still seeking is the one before: wait for the next.
    const onFrame = (_, meta) => {
      if (done) return;
      if (video.seeking) video.requestVideoFrameCallback(onFrame);
      else finish(meta.mediaTime);
    };
    const timer = setTimeout(() => {
      if (video.seeking) { done = true; video.removeEventListener('seeked', onSeeked); reject(new Error('the video did not seek')); }
      else finish(video.currentTime);
    }, timeoutMs);
    if (Math.abs(video.currentTime - t) < 1e-6 && !video.seeking && video.readyState >= 2) return finish(video.currentTime);
    video.addEventListener('seeked', onSeeked);
    video.currentTime = t;
    if (frames) video.requestVideoFrameCallback(onFrame);
  });
}

/** The video's frame now, scaled so its longer side is at most `side`, as ImageData. */
export function grab(video, canvas, side) {
  const s = Math.min(1, side / Math.max(video.videoWidth, video.videoHeight));
  const w = Math.round(video.videoWidth * s), h = Math.round(video.videoHeight * s);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const g = canvas.getContext('2d', { willReadFrequently: true });
  g.drawImage(video, 0, 0, w, h);
  return g.getImageData(0, 0, w, h);
}

// ---- the file's metadata (ISO base media: MP4, MOV)

const fourcc = (v, o) => String.fromCharCode(v.getUint8(o), v.getUint8(o + 1), v.getUint8(o + 2), v.getUint8(o + 3));

// The boxes in [from, to) of `v`: [{ type, start (of its content), end }].
function boxes(v, from, to) {
  const out = [];
  for (let o = from; o + 8 <= to;) {
    let size = v.getUint32(o), head = 8;
    if (size === 1 && o + 16 <= to) { size = Number(v.getBigUint64(o + 8)); head = 16; }
    else if (size === 0) size = to - o;
    if (size < head || o + size > to) break;
    out.push({ type: fourcc(v, o + 4), start: o + head, end: o + size });
    o += size;
  }
  return out;
}

// QuickTime's keys and their values ('meta' with 'keys' and 'ilst'), into `out`.
function readKeys(v, meta, out) {
  // ISO's meta is a full box (version and flags first); QuickTime's is not.
  const start = meta.start + 8 <= meta.end && fourcc(v, meta.start + 4) === 'hdlr' ? meta.start : meta.start + 4;
  const inside = boxes(v, start, meta.end);
  const keys = inside.find((b) => b.type === 'keys'), ilst = inside.find((b) => b.type === 'ilst');
  if (!keys || !ilst || keys.start + 8 > keys.end) return;
  const names = [];
  for (let o = keys.start + 8, k = v.getUint32(keys.start + 4); k > 0 && o + 8 <= keys.end; k--) {
    const size = v.getUint32(o);
    if (size < 8 || o + size > keys.end) break;
    names.push(new TextDecoder().decode(new Uint8Array(v.buffer, v.byteOffset + o + 8, size - 8)));
    o += size;
  }
  for (let o = ilst.start; o + 8 <= ilst.end;) {
    const size = v.getUint32(o), index = v.getUint32(o + 4);
    if (size < 8 || o + size > ilst.end) break;
    const data = boxes(v, o + 8, o + size).find((b) => b.type === 'data');
    const name = names[index - 1];
    if (data && name && data.start + 8 <= data.end) {
      const kind = v.getUint32(data.start) & 0xffffff, at = data.start + 8, n = data.end - at;
      if (kind === 1) out[name] = new TextDecoder().decode(new Uint8Array(v.buffer, v.byteOffset + at, n));
      else if (kind === 23 && n >= 4) out[name] = v.getFloat32(at);
      else if (kind === 24 && n >= 8) out[name] = v.getFloat64(at);
      else if ((kind === 21 || kind === 22) && n === 4) out[name] = kind === 21 ? v.getInt32(at) : v.getUint32(at);
      else if ((kind === 21 || kind === 22) && n === 2) out[name] = kind === 21 ? v.getInt16(at) : v.getUint16(at);
      else if ((kind === 21 || kind === 22) && n === 1) out[name] = kind === 21 ? v.getInt8(at) : v.getUint8(at);
    }
    o += size;
  }
}

/**
 * What a 'moov' box (its content, an ArrayBuffer or a typed array) says of the video:
 * { duration (s), fps, frames, codec, make, model, lens, mm35 (the 35 mm equivalent focal
 * length the camera wrote: Apple's com.apple.quicktime.camera.focal_length.35mm_equivalent) },
 * each null when absent. Never throws on a damaged box: what was read so far is returned.
 */
export function parseMoov(bytes) {
  const u = bytes instanceof ArrayBuffer ? new Uint8Array(bytes) : bytes;
  const v = new DataView(u.buffer, u.byteOffset, u.byteLength);
  const out = { duration: null, fps: null, frames: null, codec: null, make: null, model: null, lens: null, mm35: null };
  const tags = {};
  try {
    const top = boxes(v, 0, v.byteLength);
    const mvhd = top.find((b) => b.type === 'mvhd');
    if (mvhd) {
      const v1 = v.getUint8(mvhd.start) === 1;
      const scale = v.getUint32(mvhd.start + (v1 ? 20 : 12));
      const d = v1 ? Number(v.getBigUint64(mvhd.start + 24)) : v.getUint32(mvhd.start + 16);
      if (scale > 0) out.duration = d / scale;
    }
    for (const trak of top.filter((b) => b.type === 'trak')) {
      const inTrak = boxes(v, trak.start, trak.end);
      const mdia = inTrak.find((b) => b.type === 'mdia');
      if (!mdia) continue;
      const inMdia = boxes(v, mdia.start, mdia.end);
      const hdlr = inMdia.find((b) => b.type === 'hdlr'), mdhd = inMdia.find((b) => b.type === 'mdhd');
      if (!hdlr || fourcc(v, hdlr.start + 8) !== 'vide' || out.frames != null) continue;
      const minf = inMdia.find((b) => b.type === 'minf');
      const stbl = minf && boxes(v, minf.start, minf.end).find((b) => b.type === 'stbl');
      const inStbl = stbl ? boxes(v, stbl.start, stbl.end) : [];
      const stts = inStbl.find((b) => b.type === 'stts'), stsd = inStbl.find((b) => b.type === 'stsd');
      if (stsd && stsd.start + 16 <= stsd.end) out.codec = fourcc(v, stsd.start + 12);
      if (stts && mdhd) {
        let frames = 0;
        for (let k = 0, n = v.getUint32(stts.start + 4); k < n && stts.start + 16 + 8 * k <= stts.end; k++) frames += v.getUint32(stts.start + 8 + 8 * k);
        const v1 = v.getUint8(mdhd.start) === 1;
        const scale = v.getUint32(mdhd.start + (v1 ? 20 : 12));
        const d = v1 ? Number(v.getBigUint64(mdhd.start + 24)) : v.getUint32(mdhd.start + 16);
        out.frames = frames;
        if (scale > 0 && d > 0 && frames > 0) out.fps = frames / (d / scale);
      }
      const meta = inTrak.find((b) => b.type === 'meta');
      if (meta) readKeys(v, meta, tags);
    }
    // The file's own keys win over a track's: in the movie's user data (ffmpeg's place for them), then the movie's (Apple's).
    const udta = top.find((b) => b.type === 'udta');
    const inUdta = udta && boxes(v, udta.start, udta.end).find((b) => b.type === 'meta');
    if (inUdta) readKeys(v, inUdta, tags);
    const meta = top.find((b) => b.type === 'meta');
    if (meta) readKeys(v, meta, tags);
  } catch { /* a damaged box: what was read stands */ }
  const text = (k) => (typeof tags[k] === 'string' && tags[k].trim() ? tags[k].trim() : null);
  out.make = text('com.apple.quicktime.make');
  out.model = text('com.apple.quicktime.model');
  out.lens = text('com.apple.quicktime.camera.lens_model');
  const mm = Number(tags['com.apple.quicktime.camera.focal_length.35mm_equivalent']);
  out.mm35 = Number.isFinite(mm) && mm > 0 ? mm : null;
  return out;
}

/** parseMoov of a video file (a Blob); null when it has no 'moov' or it is over 64 MB. */
export async function readVideoMeta(file) {
  for (let o = 0; o + 8 <= file.size;) {
    const v = new DataView(await file.slice(o, o + 16).arrayBuffer());
    let size = v.getUint32(0), head = 8;
    if (size === 1 && v.byteLength >= 16) { size = Number(v.getBigUint64(8)); head = 16; }
    else if (size === 0) size = file.size - o;
    if (size < head) return null;
    if (fourcc(v, 4) === 'moov') {
      if (size > 64 * 1024 * 1024) return null;
      return parseMoov(await file.slice(o + head, o + size).arrayBuffer());
    }
    o += size;
  }
  return null;
}

/**
 * A focal length in pixels from a 35 mm equivalent, for a frame whose longer side is
 * `side` pixels: the equivalent is on the diagonal of the camera's 4:3 photo (43.27 mm
 * for 35 mm film), whose full width a video uses, 0.8 of that diagonal. NaN when it is
 * not a plausible focal length (outside 0.1 to 20 times the longer side).
 */
export function focalFromMm35(mm35, side) {
  const f = (mm35 * side * 1.25) / 43.2666;
  return f >= 0.1 * side && f <= 20 * side ? f : NaN;
}
