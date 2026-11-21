// FlexGlue's weights for the live capture (D157), from the visitor's own copy of
// flexglue.pth: checked against the released file's SHA-256, converted to a float16
// .jmtw file with just-matching's module (jmt.convert) and written to this origin's
// private file system, so later visits load 284 MB from the browser and convert
// nothing. In a worker because converting takes the module's memory to about 1.4 GB,
// which a WebAssembly module never gives back: the page ends the worker when it is done.
//
// The page posts { file: File, name: 'flexglue-fp16.jmtw', sha256, keep }; the worker
// posts { stage, done, total } as it goes, then { ok: true, stored, size } (stored: in
// the private file system; else bytes: the file, for this visit only) or { ok: false, error }.
import * as jmt from './jmt/jmt.js';

// SHA-256, fed in chunks (crypto.subtle.digest takes only a whole buffer).
class Sha256 {
  constructor() {
    this.h = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
    this.buf = new Uint8Array(64);
    this.n = 0;      // bytes in buf
    this.len = 0;    // bytes in all
    this.w = new Uint32Array(64);
  }
  static K = new Uint32Array([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
    0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
    0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]);
  block(b, o) {
    const w = this.w, K = Sha256.K, h = this.h;
    for (let i = 0; i < 16; i++) w[i] = (b[o + 4 * i] << 24) | (b[o + 4 * i + 1] << 16) | (b[o + 4 * i + 2] << 8) | b[o + 4 * i + 3];
    for (let i = 16; i < 64; i++) {
      const x = w[i - 15], y = w[i - 2];
      const s0 = ((x >>> 7) | (x << 25)) ^ ((x >>> 18) | (x << 14)) ^ (x >>> 3);
      const s1 = ((y >>> 17) | (y << 15)) ^ ((y >>> 19) | (y << 13)) ^ (y >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) | 0;
    }
    let [a, bb, c, d, e, f, g, hh] = h;
    for (let i = 0; i < 64; i++) {
      const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
      const t1 = (hh + S1 + ((e & f) ^ (~e & g)) + K[i] + w[i]) | 0;
      const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
      const t2 = (S0 + ((a & bb) ^ (a & c) ^ (bb & c))) | 0;
      hh = g; g = f; f = e; e = (d + t1) | 0; d = c; c = bb; bb = a; a = (t1 + t2) | 0;
    }
    h[0] += a; h[1] += bb; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }
  update(data) {
    let i = 0;
    this.len += data.length;
    if (this.n) {
      while (this.n < 64 && i < data.length) this.buf[this.n++] = data[i++];
      if (this.n < 64) return;
      this.block(this.buf, 0);
      this.n = 0;
    }
    for (; i + 64 <= data.length; i += 64) this.block(data, i);
    while (i < data.length) this.buf[this.n++] = data[i++];
  }
  hex() {
    const bits = this.len * 8;
    const pad = new Uint8Array(((this.n < 56 ? 56 : 120) - this.n) + 8);
    pad[0] = 0x80;
    const hi = Math.floor(bits / 2 ** 32), lo = bits >>> 0;
    const at = pad.length - 8;
    for (let k = 0; k < 4; k++) { pad[at + k] = (hi >>> (24 - 8 * k)) & 255; pad[at + 4 + k] = (lo >>> (24 - 8 * k)) & 255; }
    this.update(pad);
    return [...this.h].map((v) => (v >>> 0).toString(16).padStart(8, '0')).join('');
  }
}

async function sha256Of(file) {
  const s = new Sha256();
  let done = 0;
  for (const reader = file.stream().getReader(); ;) {
    const { done: end, value } = await reader.read();
    if (end) break;
    s.update(value);
    done += value.length;
    postMessage({ stage: 'check', done, total: file.size });
  }
  return s.hex();
}

async function store(name, bytes) {
  const dir = await navigator.storage.getDirectory();
  const tmp = await dir.getFileHandle(name + '.part', { create: true });
  const h = await tmp.createSyncAccessHandle();
  try {
    h.truncate(0);
    for (let at = 0; at < bytes.length; at += 1 << 24) h.write(bytes.subarray(at, at + (1 << 24)), { at });
    h.flush();
  } finally {
    h.close();
  }
  // A file takes its name only once it is whole.
  await tmp.move(name);
}

onmessage = async ({ data }) => {
  try {
    const got = await sha256Of(data.file);
    if (got !== data.sha256) {
      throw new Error(`${data.file.name} is not FlexGlue's released flexglue.pth (SHA-256 ${got.slice(0, 12)}…, ` +
                      `expected ${data.sha256.slice(0, 12)}…).`);
    }
    if (typeof jmt.convert !== 'function') throw new Error('this build of just-matching\'s module cannot convert checkpoints');
    const bytes = await jmt.convert(data.file, {
      fp16: true,
      onProgress: (done, total) => postMessage({ stage: 'read', done, total }),
    });
    postMessage({ stage: 'store', done: 0, total: bytes.length });
    let stored = false;
    if (data.keep) {
      try {
        await store(data.name, bytes);
        stored = true;
      } catch { /* no private file system here (a private window): keep it for this visit */ }
    }
    if (stored) postMessage({ ok: true, stored, size: bytes.length });
    else postMessage({ ok: true, stored, size: bytes.length, bytes }, [bytes.buffer]);
  } catch (e) {
    postMessage({ ok: false, error: String(e?.message ?? e) });
  }
};
