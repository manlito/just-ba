#include "jba/manhattan.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "jba/camera_model.h"
#include "jba/geometry.h"

namespace jba {
namespace {

constexpr double kPi = 3.141592653589793;

double dot3(const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
void cross3(const double* a, const double* b, double* c) {
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}
double norm3(const double* a) { return std::sqrt(dot3(a, a)); }

// Eigenvector of the smallest eigenvalue of a symmetric 3 x 3 matrix, by cyclic
// Jacobi rotations.
void smallest_eigenvector3(const double* A_in, double* v_out) {
  double A[9], V[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::copy_n(A_in, 9, A);
  for (int sweep = 0; sweep < 50; ++sweep) {
    const double off = A[1] * A[1] + A[2] * A[2] + A[5] * A[5];
    if (off < 1e-30) break;
    for (int p = 0; p < 2; ++p)
      for (int q = p + 1; q < 3; ++q) {
        const double apq = A[3 * p + q];
        if (std::abs(apq) < 1e-300) continue;
        const double theta = (A[3 * q + q] - A[3 * p + p]) / (2 * apq);
        const double t = (theta >= 0 ? 1 : -1) / (std::abs(theta) + std::sqrt(theta * theta + 1));
        const double c = 1 / std::sqrt(t * t + 1), s = t * c;
        for (int k = 0; k < 3; ++k) {
          const double akp = A[3 * k + p], akq = A[3 * k + q];
          A[3 * k + p] = c * akp - s * akq;
          A[3 * k + q] = s * akp + c * akq;
        }
        for (int k = 0; k < 3; ++k) {
          const double apk = A[3 * p + k], aqk = A[3 * q + k];
          A[3 * p + k] = c * apk - s * aqk;
          A[3 * q + k] = s * apk + c * aqk;
        }
        for (int k = 0; k < 3; ++k) {
          const double vkp = V[3 * k + p], vkq = V[3 * k + q];
          V[3 * k + p] = c * vkp - s * vkq;
          V[3 * k + q] = s * vkp + c * vkq;
        }
      }
  }
  int best = 0;
  for (int k = 1; k < 3; ++k)
    if (A[4 * k] < A[4 * best]) best = k;
  for (int k = 0; k < 3; ++k) v_out[k] = V[3 * k + best];
}

// Unit x, y completing v (unit) to a right-handed basis (x, y, v), chosen from v alone.
void complete_basis(const double* v, double* x, double* y) {
  int k = 0;
  for (int c = 1; c < 3; ++c)
    if (std::abs(v[c]) < std::abs(v[k])) k = c;
  double e[3] = {0, 0, 0};
  e[k] = 1;
  cross3(e, v, x);
  const double n = norm3(x);
  for (int a = 0; a < 3; ++a) x[a] /= n;
  cross3(v, x, y);
}

// Angle folded into [-pi/4, pi/4).
double fold90(double a) {
  const double q = kPi / 2;
  a = std::fmod(a + kPi / 4, q);
  if (a < 0) a += q;
  return a - kPi / 4;
}

struct Segment {
  double n[3];    // plane normal, working frame
  double length;  // pixels
  bool ok;
};

}  // namespace

bool line_normal(const Problem& p, std::size_t k, double* n) {
  const auto i = static_cast<std::size_t>(p.line_image[k]);
  const double* K = &p.intrinsics[static_cast<std::size_t>(num_params(p.model)) *
                                  static_cast<std::size_t>(p.intrinsics_of_image(i))];
  const double* e = &p.line_xy[4 * k];
  double b0[3], b1[3];
  if (!unproject(p.model, K, e[0], e[1], b0) || !unproject(p.model, K, e[2], e[3], b1)) return false;
  cross3(b0, b1, n);
  const double len = norm3(n);
  if (!(len > 1e-12)) return false;
  for (int a = 0; a < 3; ++a) n[a] /= len;
  return true;
}

double line_residual_sigma(const Problem& p, std::size_t k, const PriorDefaults& d) {
  const double* e = &p.line_xy[4 * k];
  const double length = std::hypot(e[2] - e[0], e[3] - e[1]);
  const double s = p.line_sigma.empty() ? d.line_endpoint_sigma : p.line_sigma[k];
  return std::max(d.line_min_sigma, std::sqrt(2.0) * s / length);
}

ManhattanAssignment assign_lines(const Problem& p, const ManhattanOptions& o, const double* frame_in) {
  if (auto err = validate(p); !err.empty()) throw std::invalid_argument("assign_lines: invalid problem: " + err);
  if (o.dof != 1 && o.dof != 3) throw std::invalid_argument("assign_lines: dof must be 1 or 3");
  ManhattanAssignment out;
  const std::size_t L = p.num_lines();
  out.axis.assign(L, -1);
  const double thr = std::sin(o.max_angle_deg * kPi / 180);
  const double flat = std::sin(o.horizontal_plane_deg * kPi / 180);

  // Plane normals in the working frame: n_w = R^T n_c.
  std::vector<Segment> seg(L);
  for (std::size_t k = 0; k < L; ++k) {
    Segment& s = seg[k];
    double nc[3];
    s.ok = line_normal(p, k, nc);
    const double* e = &p.line_xy[4 * k];
    s.length = std::hypot(e[2] - e[0], e[3] - e[1]);
    if (!s.ok) {
      ++out.unusable;
      continue;
    }
    const double* q = &p.poses[7 * static_cast<std::size_t>(p.line_image[k])];
    const double qc[4] = {q[0], -q[1], -q[2], -q[3]};
    quat_rotate(qc, nc, s.n);
  }

  double g[3];
  gravity_direction(p, g);
  const double up[3] = {-g[0], -g[1], -g[2]};
  double R[9];  // columns: the building's axes in the working frame
  if (frame_in) {
    out.frame.assign(frame_in, frame_in + 4);
    for (int c = 0; c < 3; ++c) {
      double e[3] = {0, 0, 0}, d[3];
      e[c] = 1;
      quat_rotate(frame_in, e, d);
      for (int r = 0; r < 3; ++r) R[3 * r + c] = d[r];
    }
  } else {
    // The vertical: up, or with three freedoms the direction lying in most planes.
    double v[3] = {up[0], up[1], up[2]};
    if (o.dof == 3) {
      std::vector<std::size_t> order;
      for (std::size_t k = 0; k < L; ++k)
        if (seg[k].ok) order.push_back(k);
      std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return seg[a].length > seg[b].length; });
      if (order.size() > 150) order.resize(150);
      double best = -1;
      for (std::size_t a = 0; a < order.size(); ++a)
        for (std::size_t b = a + 1; b < order.size(); ++b) {
          double c[3];
          cross3(seg[order[a]].n, seg[order[b]].n, c);
          const double cn = norm3(c);
          if (cn < 0.2) continue;
          for (double& x : c) x /= cn;
          double score = 0;
          for (std::size_t k = 0; k < L; ++k)
            if (seg[k].ok && std::abs(dot3(seg[k].n, c)) < thr) score += seg[k].length;
          if (score > best) {
            best = score;
            std::copy_n(c, 3, v);
          }
        }
      // Refine: the direction most nearly in every inlier plane.
      double M[9] = {};
      for (std::size_t k = 0; k < L; ++k) {
        if (!seg[k].ok || std::abs(dot3(seg[k].n, v)) >= thr) continue;
        for (int r = 0; r < 3; ++r)
          for (int c = 0; c < 3; ++c) M[3 * r + c] += seg[k].length * seg[k].n[r] * seg[k].n[c];
      }
      if (best > 0) smallest_eigenvector3(M, v);
      if (dot3(v, up) < 0)
        for (double& x : v) x = -x;
    }
    // Heading of the horizontal axes: a vote modulo 90 degrees.
    double xh[3], yh[3];
    complete_basis(v, xh, yh);
    std::vector<double> psi;
    std::vector<double> weight;
    for (std::size_t k = 0; k < L; ++k) {
      const Segment& s = seg[k];
      if (!s.ok || std::abs(dot3(s.n, v)) < thr) continue;  // along the vertical
      double h[3];
      cross3(v, s.n, h);
      const double hn = norm3(h);
      if (hn < flat) continue;  // plane nearly horizontal
      psi.push_back(fold90(std::atan2(dot3(h, yh), dot3(h, xh))));
      weight.push_back(s.length);
    }
    const double width = o.max_angle_deg * kPi / 180;
    double best_score = -1, heading = 0;
    for (int bin = 0; bin < 360; ++bin) {
      const double c = -kPi / 4 + (bin + 0.5) * (kPi / 2) / 360;
      double score = 0;
      for (std::size_t a = 0; a < psi.size(); ++a) {
        const double d = std::abs(fold90(psi[a] - c));
        if (d < width) score += weight[a] * (1 - d / width);
      }
      if (score > best_score) {
        best_score = score;
        heading = c;
      }
    }
    // Refine: weighted mean of the votes near the peak.
    double sw = 0, sd = 0;
    for (std::size_t a = 0; a < psi.size(); ++a) {
      const double d = fold90(psi[a] - heading);
      if (std::abs(d) < width) {
        sw += weight[a];
        sd += weight[a] * d;
      }
    }
    if (sw > 0) heading += sd / sw;
    const double ch = std::cos(heading), sh = std::sin(heading);
    double e0[3], e1[3];
    for (int a = 0; a < 3; ++a) e0[a] = ch * xh[a] + sh * yh[a];
    cross3(v, e0, e1);
    for (int r = 0; r < 3; ++r) {
      R[3 * r] = e0[r];
      R[3 * r + 1] = e1[r];
      R[3 * r + 2] = v[r];
    }
    out.frame.resize(4);
    quat_from_matrix(R, out.frame.data());
  }

  // Assignment, and the support test.
  double informative = 0, explained = 0;
  for (std::size_t k = 0; k < L; ++k) {
    const Segment& s = seg[k];
    if (!s.ok) continue;
    double miss[3];
    int fits = 0, axis = -1;
    for (int c = 0; c < 3; ++c) {
      const double d[3] = {R[c], R[3 + c], R[6 + c]};
      miss[c] = std::abs(dot3(s.n, d));
      if (miss[c] < thr) {
        ++fits;
        axis = c;
      }
    }
    if (fits == 1) {
      out.axis[k] = axis;
      ++out.assigned[axis];
    }
    const double v[3] = {R[2], R[5], R[8]};
    double h[3];
    cross3(v, s.n, h);
    if (miss[2] >= thr && norm3(h) >= flat) {
      informative += s.length;
      if (fits == 1) explained += s.length;
    }
  }
  out.support = informative > 0 ? explained / informative : 0;
  const int total = out.assigned[0] + out.assigned[1] + out.assigned[2];
  char buf[200];
  if (informative <= 0) {
    out.message = "no segment carries heading information";
  } else if (total < o.min_segments) {
    std::snprintf(buf, sizeof(buf), "only %d segments fit an axis (at least %d needed)", total, o.min_segments);
    out.message = buf;
  } else if (out.support < o.min_support) {
    std::snprintf(buf, sizeof(buf), "the scene does not look Manhattan: the best frame explains %.0f%% of the "
                  "horizontal segment length (at least %.0f%% needed)", 100 * out.support, 100 * o.min_support);
    out.message = buf;
  } else {
    out.accepted = true;
  }
  return out;
}

void apply_assignment(Problem& p, const ManhattanAssignment& a, int dof) {
  if (!a.accepted) {
    p.line_axis.clear();
    p.manhattan_frame.clear();
    return;
  }
  p.line_axis = a.axis;
  p.manhattan_frame = a.frame;
  p.manhattan_dof = dof;
}

}  // namespace jba
