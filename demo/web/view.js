// A model's points and cameras on a 2D canvas, for the live capture (D157): drag to turn,
// scroll or pinch to zoom. Cameras are [C (3), R row-major (9)] as the module writes them
// (rows: the camera's x, y, z axes in the world). The view takes the cameras' mean up
// (their -y axis) as up and looks from behind them; it fits itself to the model until the
// visitor turns or zooms it.
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const n = Math.hypot(...a) || 1; return [a[0] / n, a[1] / n, a[2] / n]; };
const quantile = (xs, q) => { const s = Float64Array.from(xs).sort(); return s[Math.min(s.length - 1, Math.floor(q * s.length))]; };
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export class ModelView {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.v = { target: [0, 0, 0], yaw: 0, pitch: 0.45, dist: 10, size: 1, R: null, frustum: 0.05 };
    this.model = null;
    this.touched = false;  // the visitor moved the view: stop fitting it
    const pointers = new Map();
    let pinch = 0;
    canvas.addEventListener('pointerdown', (e) => { canvas.setPointerCapture(e.pointerId); pointers.set(e.pointerId, [e.clientX, e.clientY]); });
    const up = (e) => { pointers.delete(e.pointerId); pinch = 0; };
    canvas.addEventListener('pointerup', up);
    canvas.addEventListener('pointercancel', up);
    canvas.addEventListener('pointermove', (e) => {
      if (!pointers.has(e.pointerId)) return;
      const [px, py] = pointers.get(e.pointerId);
      pointers.set(e.pointerId, [e.clientX, e.clientY]);
      this.touched = true;
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
        if (pinch) this.v.dist *= pinch / d;
        pinch = d;
      } else {
        this.v.yaw -= (e.clientX - px) * 0.008;
        this.v.pitch = Math.max(-1.45, Math.min(1.45, this.v.pitch + (e.clientY - py) * 0.008));
      }
      this.draw();
    });
    canvas.addEventListener('wheel', (e) => { e.preventDefault(); this.touched = true; this.v.dist *= Math.exp(e.deltaY * 0.001); this.draw(); }, { passive: false });
    new ResizeObserver(() => this.draw()).observe(canvas);
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => this.draw());
  }

  /** model: { cameras: [{ image, pose }], points: Float32Array, colours: Uint8Array (3 per point) or null, added: Set }. */
  set(model, { refit = false } = {}) {
    this.model = model;
    if (refit) this.touched = false;
    if (!this.touched && model?.cameras?.length) this.fit();
    this.draw();
  }

  fit() {
    const { cameras: cams, points: P } = this.model;
    let up = [0, 0, 0], fwd = [0, 0, 0];
    for (const c of cams) {
      const p = c.pose;
      up = sub(up, [p[6], p[7], p[8]]);
      fwd = [fwd[0] + p[9], fwd[1] + p[10], fwd[2] + p[11]];
    }
    up = norm(up);
    fwd = norm(sub(fwd, up.map((x) => x * dot(fwd, up))));
    const v = this.v;
    v.R = [cross(fwd, up), up, [-fwd[0], -fwd[1], -fwd[2]]];
    const disp = (q) => [dot(v.R[0], q), dot(v.R[1], q), dot(v.R[2], q)];
    const xs = [[], [], []];
    const n = P ? P.length / 3 : 0, every = Math.max(1, Math.floor(n / 4000));
    for (let p = 0; p < n; p += every) {
      const d = disp([P[3 * p], P[3 * p + 1], P[3 * p + 2]]);
      for (let k = 0; k < 3; k++) xs[k].push(d[k]);
    }
    for (const c of cams) {
      const d = disp(c.pose);
      for (let k = 0; k < 3; k++) xs[k].push(d[k]);
    }
    const lo = xs.map((a) => quantile(a, 0.05)), hi = xs.map((a) => quantile(a, 0.95));
    v.target = lo.map((x, d) => 0.5 * (x + hi[d]));
    v.size = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1;
    v.dist = 1.4 * v.size;  // far enough back that the cameras, at the edge of the points, stay in view
    const cs = [0, 1, 2].map((k) => cams.map((c) => c.pose[k]));
    const camSize = Math.hypot(...cs.map((a) => Math.max(...a) - Math.min(...a)));
    v.frustum = Math.max(0.01 * v.size, Math.min(0.04 * v.size, 0.08 * camSize || 0.03 * v.size));
    v.yaw = 0;
    v.pitch = 0.45;
  }

  draw() {
    const { canvas, ctx, v } = this;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const m = this.model;
    if (!m?.cameras?.length || !v.R) return;
    const cp = Math.cos(v.pitch);
    const eye = [v.target[0] + v.dist * cp * Math.sin(v.yaw), v.target[1] + v.dist * Math.sin(v.pitch), v.target[2] + v.dist * cp * Math.cos(v.yaw)];
    const fw = norm(sub(v.target, eye)), r = norm(cross(fw, [0, 1, 0])), u = cross(r, fw);
    const F = 0.9 * Math.min(w, h);
    const disp = (q) => [dot(v.R[0], q), dot(v.R[1], q), dot(v.R[2], q)];
    const project = (q) => {
      const d = sub(disp(q), eye);
      const z = dot(d, fw);
      if (z < 1e-3 * v.size) return null;
      return [w / 2 + F * dot(d, r) / z, h / 2 - F * dot(d, u) / z];
    };
    const P = m.points, C = m.colours, n = P ? P.length / 3 : 0;
    const grey = css('--point') || '#888';
    const pt = Math.max(1.5, Math.min(3, 2400 / Math.sqrt(n + 1) / 25));
    ctx.fillStyle = grey;
    for (let p = 0; p < n; p++) {
      const q = project([P[3 * p], P[3 * p + 1], P[3 * p + 2]]);
      if (!q) continue;
      if (C) ctx.fillStyle = `rgb(${C[3 * p]},${C[3 * p + 1]},${C[3 * p + 2]})`;
      ctx.fillRect(q[0] - pt / 2, q[1] - pt / 2, pt, pt);
    }
    const colCam = css('--cam') || '#222', colNew = css('--cam-new') || '#d9480f';
    for (const c of m.cameras) {
      const isNew = m.added?.has(c.image);
      const p = c.pose, k = v.frustum;
      const Cc = [p[0], p[1], p[2]], x = [p[3], p[4], p[5]], y = [p[6], p[7], p[8]], z = [p[9], p[10], p[11]];
      const corner = (sx, sy) => [0, 1, 2].map((d) => Cc[d] + k * (z[d] + 0.5 * sx * x[d] + 0.34 * sy * y[d]));
      const q0 = project(Cc), qs = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)].map(project);
      if (!q0 || qs.some((q) => !q)) continue;
      ctx.strokeStyle = isNew ? colNew : colCam;
      ctx.lineWidth = isNew ? 2 : 1;
      ctx.beginPath();
      for (const q of qs) { ctx.moveTo(q0[0], q0[1]); ctx.lineTo(q[0], q[1]); }
      ctx.moveTo(qs[3][0], qs[3][1]);
      for (const q of qs) ctx.lineTo(q[0], q[1]);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(qs[0][0], qs[0][1]);
      ctx.lineTo(qs[1][0], qs[1][1]);
      ctx.lineWidth += 1;
      ctx.stroke();
    }
  }
}
