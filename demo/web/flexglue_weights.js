// FlexGlue's weights from the visitor's own flexglue.pth (D157), for the live capture and the live demo (D160).
// Both pages are on one origin, so the converted file kept in its private file system serves both.
//
//   stored()            the kept file (a File), or null
//   load()              its bytes, or null
//   convert(file, cb)   checks and converts flexglue.pth in weights_worker.js; cb({ stage, done, total });
//                       resolves to { stored: true } or { stored: false, bytes } (this visit only)
//   remove()            forgets the kept file

// FlexGlue's released checkpoint (just-matching's tools/fetch_upstream.sh pins it).
export const FLEXGLUE = {
  sha256: 'd7371f743b8cdc4adf723c2aa05b181587fe4bc177fced139697975dff790068',
  bytes: 567328059,
  stored: 'flexglue-fp16.jmtw',
  storedBytes: 283537600,
};

export const STAGES = { check: 'Checking flexglue.pth', read: 'Converting it to half precision', store: 'Keeping it in this browser' };

export async function stored() {
  try {
    const dir = await navigator.storage.getDirectory();
    const f = await (await dir.getFileHandle(FLEXGLUE.stored)).getFile();
    return f.size === FLEXGLUE.storedBytes ? f : null;
  } catch {
    return null;
  }
}

export async function load() {
  const f = await stored();
  return f ? new Uint8Array(await f.arrayBuffer()) : null;
}

// The size is checked here; the SHA-256 in the worker. Throws with a message for the visitor.
export async function convert(file, onStage) {
  if (file.size !== FLEXGLUE.bytes) {
    const int = (v) => Number(v).toLocaleString('en-US');
    throw new Error(`${file.name} is ${int(file.size)} bytes; FlexGlue's released flexglue.pth is ${int(FLEXGLUE.bytes)}.`);
  }
  await navigator.storage.persist?.();
  const w = new Worker(new URL('weights_worker.js', import.meta.url), { type: 'module' });
  try {
    const out = await new Promise((resolve, reject) => {
      w.onmessage = ({ data }) => {
        if (data.stage) return void onStage?.(data);
        data.ok ? resolve(data) : reject(new Error(data.error));
      };
      w.onerror = (e) => reject(new Error(e.message || 'the conversion failed'));
      w.postMessage({ file, name: FLEXGLUE.stored, sha256: FLEXGLUE.sha256, keep: true });
    });
    return out.stored ? { stored: true } : { stored: false, bytes: out.bytes };
  } finally {
    w.terminate();  // its ~1.4 GB go with it
  }
}

export async function remove() {
  try { await (await navigator.storage.getDirectory()).removeEntry(FLEXGLUE.stored); } catch { /* gone already */ }
}
