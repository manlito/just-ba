#include "jba/initialize.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "jba/geometry.h"

namespace jba {
namespace {

// Largest eigenvector of a symmetric 4 x 4 matrix, by cyclic Jacobi rotations.
void largest_eigenvector4(const double* A_in, double* v_out) {
  double A[16], V[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  std::copy_n(A_in, 16, A);
  for (int sweep = 0; sweep < 50; ++sweep) {
    double off = 0;
    for (int p = 0; p < 4; ++p)
      for (int q = p + 1; q < 4; ++q) off += A[4 * p + q] * A[4 * p + q];
    if (off < 1e-30) break;
    for (int p = 0; p < 4; ++p) {
      for (int q = p + 1; q < 4; ++q) {
        const double apq = A[4 * p + q];
        if (std::abs(apq) < 1e-300) continue;
        const double theta = (A[4 * q + q] - A[4 * p + p]) / (2 * apq);
        const double t = (theta >= 0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1));
        const double c = 1 / std::sqrt(t * t + 1), s = t * c;
        for (int k = 0; k < 4; ++k) {
          const double akp = A[4 * k + p], akq = A[4 * k + q];
          A[4 * k + p] = c * akp - s * akq;
          A[4 * k + q] = s * akp + c * akq;
        }
        for (int k = 0; k < 4; ++k) {
          const double apk = A[4 * p + k], aqk = A[4 * q + k];
          A[4 * p + k] = c * apk - s * aqk;
          A[4 * q + k] = s * apk + c * aqk;
        }
        for (int k = 0; k < 4; ++k) {
          const double vkp = V[4 * k + p], vkq = V[4 * k + q];
          V[4 * k + p] = c * vkp - s * vkq;
          V[4 * k + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  int best = 0;
  for (int k = 1; k < 4; ++k)
    if (A[5 * k] > A[5 * best]) best = k;
  for (int k = 0; k < 4; ++k) v_out[k] = V[4 * k + best];
  // Sign convention: w >= 0.
  if (v_out[0] < 0)
    for (int k = 0; k < 4; ++k) v_out[k] = -v_out[k];
}

void conj(const double* q, double* c) {
  c[0] = q[0];
  c[1] = -q[1];
  c[2] = -q[2];
  c[3] = -q[3];
}

void centre(const double* pose, double* C) {
  double c[4];
  conj(pose, c);
  quat_rotate(c, pose + 4, C);
  for (int a = 0; a < 3; ++a) C[a] = -C[a];
}

// Pose from rotation q and centre C: t = -R C.
void make_pose(const double* q, const double* C, double* pose) {
  std::copy_n(q, 4, pose);
  quat_normalize(pose);
  double RC[3];
  quat_rotate(pose, C, RC);
  for (int a = 0; a < 3; ++a) pose[4 + a] = -RC[a];
}

// Applies X_new = s R X + t to a world-to-camera pose: R_new = R_cam R^T.
void transform_pose(const double* G, const double* pose, double* out) {
  double C[3], RC[3], gc[4], q[4];
  centre(pose, C);
  quat_rotate(G, C, RC);
  for (int a = 0; a < 3; ++a) RC[a] = G[7] * RC[a] + G[4 + a];
  conj(G, gc);
  quat_multiply(pose, gc, q);
  make_pose(q, RC, out);
}

struct UnionFind {
  std::vector<std::size_t> parent;
  explicit UnionFind(std::size_t n) : parent(n) { std::iota(parent.begin(), parent.end(), std::size_t(0)); }
  std::size_t find(std::size_t a) {
    while (parent[a] != a) a = parent[a] = parent[parent[a]];
    return a;
  }
  bool join(std::size_t a, std::size_t b) {
    a = find(a);
    b = find(b);
    if (a == b) return false;
    if (a < b) parent[b] = a;
    else parent[a] = b;
    return true;
  }
};


// Motion T_b T_a^-1 from relative prior k, in either direction.
void prior_motion(const Problem& p, std::size_t k, std::size_t a, double* T) {
  const double* P = &p.rel_pose[7 * k];
  if (static_cast<std::size_t>(p.rel_images[2 * k]) == a) {
    std::copy_n(P, 7, T);
    return;
  }
  // Inverse: (R^T, -R^T t).
  double c[4];
  conj(P, c);
  std::copy_n(c, 4, T);
  double Rt[3];
  quat_rotate(c, P + 4, Rt);
  for (int i = 0; i < 3; ++i) T[4 + i] = -Rt[i];
}

// Rotation and translation standard deviations a whitening matrix implies (from
// its diagonal); +inf when that part is left out.
void sigmas_of(const double* W, double& sr, double& st) {
  const double wr = (W[0] + W[7] + W[14]) / 3, wt = (W[21] + W[28] + W[35]) / 3;
  sr = wr > 0 ? 1 / wr : std::numeric_limits<double>::infinity();
  st = wt > 0 ? 1 / wt : std::numeric_limits<double>::infinity();
}

// Median disagreement, in standard deviations, of relative prior k with the
// triangles it closes: (i, j) against (i, m) then (m, j). Zero when it closes none.
double triangle_inconsistency(const Problem& p, const std::vector<double>& W, std::size_t k) {
  const auto i = static_cast<std::size_t>(p.rel_images[2 * k]), j = static_cast<std::size_t>(p.rel_images[2 * k + 1]);
  double sr0, st0;
  sigmas_of(&W[36 * k], sr0, st0);
  // Priors touching i and j, by their other image.
  std::vector<std::pair<std::size_t, std::size_t>> at_i, at_j;
  for (std::size_t q = 0; q < p.num_relative_priors(); ++q) {
    if (q == k) continue;
    const auto a = static_cast<std::size_t>(p.rel_images[2 * q]), b = static_cast<std::size_t>(p.rel_images[2 * q + 1]);
    if (a == i && b != j) at_i.push_back({b, q});
    else if (b == i && a != j) at_i.push_back({a, q});
    if (a == j && b != i) at_j.push_back({b, q});
    else if (b == j && a != i) at_j.push_back({a, q});
  }
  std::vector<double> errors;
  for (const auto& [m, qi] : at_i) {
    for (const auto& [m2, qj] : at_j) {
      if (m2 != m) continue;
      double T_mi[7], T_jm[7], T_ji[7];
      prior_motion(p, qi, i, T_mi);  // i -> m
      prior_motion(p, qj, m, T_jm);  // m -> j
      // Composed i -> j: (R_jm R_mi, R_jm t_mi + t_jm).
      quat_multiply(T_jm, T_mi, T_ji);
      double Rt[3];
      quat_rotate(T_jm, T_mi + 4, Rt);
      for (int c = 0; c < 3; ++c) T_ji[4 + c] = Rt[c] + T_jm[4 + c];
      const double* P = &p.rel_pose[7 * k];
      double Pc[4], d[4];
      conj(P, Pc);
      quat_multiply(Pc, T_ji, d);
      const double v = std::sqrt(d[1] * d[1] + d[2] * d[2] + d[3] * d[3]);
      const double angle = 2 * std::atan2(v, std::abs(d[0]));
      double dist = 0;
      for (int c = 0; c < 3; ++c) dist += (T_ji[4 + c] - P[4 + c]) * (T_ji[4 + c] - P[4 + c]);
      dist = std::sqrt(dist);
      double sr1, st1, sr2, st2;
      sigmas_of(&W[36 * qi], sr1, st1);
      sigmas_of(&W[36 * qj], sr2, st2);
      const double sr = std::sqrt(sr0 * sr0 + sr1 * sr1 + sr2 * sr2), st = std::sqrt(st0 * st0 + st1 * st1 + st2 * st2);
      const double er = std::isfinite(sr) ? angle / sr : 0, et = std::isfinite(st) ? dist / st : 0;
      errors.push_back(std::sqrt(er * er + et * et));
    }
  }
  if (errors.empty()) return 0;
  // Lower median: with two triangles, one of them through a different bad prior
  // must not make this one look bad.
  const std::size_t mid = (errors.size() - 1) / 2;
  std::nth_element(errors.begin(), errors.begin() + static_cast<std::ptrdiff_t>(mid), errors.end());
  return errors[mid];
}

// Rigid or similarity fit of `from` onto `to` with per-pair rotations, robust to a
// few bad pairs: refit without pairs whose centre or rotation residual exceeds
// three times the median (and a floor), twice.
void robust_fit(const std::vector<double>& from, const std::vector<double>& to, const std::vector<double>& rots,
                bool with_scale, double* G) {
  std::vector<char> use(from.size() / 3, 1);
  for (int round = 0; round < 3; ++round) {
    std::vector<double> f, t, r;
    for (std::size_t q = 0; q < use.size(); ++q) {
      if (!use[q]) continue;
      f.insert(f.end(), &from[3 * q], &from[3 * q] + 3);
      t.insert(t.end(), &to[3 * q], &to[3 * q] + 3);
      r.insert(r.end(), &rots[4 * q], &rots[4 * q] + 4);
    }
    fit_similarity(f, t, &r, with_scale && f.size() >= 6, G);
    if (round == 2 || use.size() < 3) return;
    std::vector<double> ec(use.size()), er(use.size());
    for (std::size_t q = 0; q < use.size(); ++q) {
      double RC[3];
      quat_rotate(G, &from[3 * q], RC);
      double d = 0;
      for (int c = 0; c < 3; ++c) {
        const double v = G[7] * RC[c] + G[4 + c] - to[3 * q + static_cast<std::size_t>(c)];
        d += v * v;
      }
      ec[q] = std::sqrt(d);
      const double* rq = &rots[4 * q];
      er[q] = 2 * std::acos(std::min(1.0, std::abs(rq[0] * G[0] + rq[1] * G[1] + rq[2] * G[2] + rq[3] * G[3])));
    }
    auto median = [](std::vector<double> v) {
      std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2), v.end());
      return v[v.size() / 2];
    };
    const double mc = median(ec), mr = median(er);
    for (std::size_t q = 0; q < use.size(); ++q) use[q] = ec[q] <= std::max(3 * mc, 1e-9) && er[q] <= std::max(3 * mr, 1e-9);
  }
}

}  // namespace

void average_quaternions(const std::vector<double>& quats, double* out) {
  double M[16] = {};
  for (std::size_t k = 0; k + 3 < quats.size(); k += 4) {
    const double* q = &quats[k];
    for (int a = 0; a < 4; ++a)
      for (int b = 0; b < 4; ++b) M[4 * a + b] += q[a] * q[b];
  }
  largest_eigenvector4(M, out);
  quat_normalize(out);
}

void fit_similarity(const std::vector<double>& from, const std::vector<double>& to,
                    const std::vector<double>* rotations, bool with_scale, double* out) {
  const std::size_t n = from.size() / 3;
  if (n == 0 || to.size() != from.size()) throw std::invalid_argument("fit_similarity: need matching point lists");
  double mf[3] = {0, 0, 0}, mt[3] = {0, 0, 0};
  for (std::size_t i = 0; i < n; ++i)
    for (int a = 0; a < 3; ++a) {
      mf[a] += from[3 * i + static_cast<std::size_t>(a)] / static_cast<double>(n);
      mt[a] += to[3 * i + static_cast<std::size_t>(a)] / static_cast<double>(n);
    }
  double q[4];
  if (rotations != nullptr && !rotations->empty()) {
    average_quaternions(*rotations, q);
  } else {
    // Horn: the rotation is the largest eigenvector of N built from the
    // cross-covariance S = sum (from - mf)(to - mt)^T.
    double S[9] = {};
    for (std::size_t i = 0; i < n; ++i)
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
          S[3 * a + b] += (from[3 * i + static_cast<std::size_t>(a)] - mf[a]) * (to[3 * i + static_cast<std::size_t>(b)] - mt[b]);
    const double Sxx = S[0], Sxy = S[1], Sxz = S[2], Syx = S[3], Syy = S[4], Syz = S[5], Szx = S[6], Szy = S[7], Szz = S[8];
    const double N[16] = {Sxx + Syy + Szz, Syz - Szy,       Szx - Sxz,        Sxy - Syx,
                          Syz - Szy,       Sxx - Syy - Szz, Sxy + Syx,        Szx + Sxz,
                          Szx - Sxz,       Sxy + Syx,       -Sxx + Syy - Szz, Syz + Szy,
                          Sxy - Syx,       Szx + Sxz,       Syz + Szy,        -Sxx - Syy + Szz};
    largest_eigenvector4(N, q);
    quat_normalize(q);
  }
  double s = 1;
  if (with_scale) {
    double num = 0, den = 0;
    for (std::size_t i = 0; i < n; ++i) {
      double d[3], Rd[3];
      for (int a = 0; a < 3; ++a) d[a] = from[3 * i + static_cast<std::size_t>(a)] - mf[a];
      quat_rotate(q, d, Rd);
      for (int a = 0; a < 3; ++a) {
        num += Rd[a] * (to[3 * i + static_cast<std::size_t>(a)] - mt[a]);
        den += d[a] * d[a];
      }
    }
    if (den > 1e-24 && num > 0) s = num / den;
  }
  double Rm[3];
  quat_rotate(q, mf, Rm);
  std::copy_n(q, 4, out);
  for (int a = 0; a < 3; ++a) out[4 + a] = mt[a] - s * Rm[a];
  out[7] = s;
}

PoseInitialization initialize_poses(const Problem& p, const PriorDefaults& defaults) {
  const std::size_t n_img = p.num_images(), n_rel = p.num_relative_priors(), n_glob = p.num_global_priors();
  const std::size_t n_sets = p.num_prior_sets();
  PoseInitialization out;
  out.poses = p.poses;
  out.set_transform.resize(8 * n_sets);
  for (std::size_t s = 0; s < n_sets; ++s) {
    static constexpr double kId[8] = {1, 0, 0, 0, 0, 0, 0, 1};
    if (p.set_transform.empty()) std::copy_n(kId, 8, &out.set_transform[8 * s]);
    else std::copy_n(&p.set_transform[8 * s], 8, &out.set_transform[8 * s]);
  }
  out.group.assign(n_img, -1);

  std::vector<double> W_rel, W_glob;
  prior_whitening(p, defaults, W_rel, W_glob);

  // Components of the relative prior graph, and a maximum-trust spanning tree.
  // Trust: the prior's weight, lowered when it disagrees with the loops it closes.
  std::vector<double> trust(n_rel);
  for (std::size_t k = 0; k < n_rel; ++k) {
    double t = 0;
    for (int a = 0; a < 36; ++a) t += W_rel[36 * k + static_cast<std::size_t>(a)] * W_rel[36 * k + static_cast<std::size_t>(a)];
    const double inc = triangle_inconsistency(p, W_rel, k);
    trust[k] = t / (1 + (inc / 3) * (inc / 3));
  }
  std::vector<std::size_t> order(n_rel);
  std::iota(order.begin(), order.end(), std::size_t(0));
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return trust[a] > trust[b]; });
  UnionFind uf(n_img);
  std::vector<std::vector<std::size_t>> tree(n_img);  // tree edges (prior indices) per image
  std::vector<int> degree(n_img, 0);
  std::vector<char> has_prior(n_img, 0);
  for (std::size_t k = 0; k < n_rel; ++k) {
    const auto i = static_cast<std::size_t>(p.rel_images[2 * k]), j = static_cast<std::size_t>(p.rel_images[2 * k + 1]);
    ++degree[i];
    ++degree[j];
    has_prior[i] = has_prior[j] = 1;
  }
  for (std::size_t k : order) {
    const auto i = static_cast<std::size_t>(p.rel_images[2 * k]), j = static_cast<std::size_t>(p.rel_images[2 * k + 1]);
    if (uf.join(i, j)) {
      tree[i].push_back(k);
      tree[j].push_back(k);
    }
  }
  for (auto& edges : tree) std::sort(edges.begin(), edges.end());
  for (std::size_t k = 0; k < n_glob; ++k) has_prior[static_cast<std::size_t>(p.glob_image[k])] = 1;

  // Components: images with any prior. comp_of[i], members, whether it has
  // relative priors.
  std::vector<int> comp_of(n_img, -1);
  std::vector<std::vector<std::size_t>> members;
  for (std::size_t i = 0; i < n_img; ++i) {
    if (!has_prior[i]) continue;
    const std::size_t root = uf.find(i);
    if (comp_of[root] < 0) {
      comp_of[root] = static_cast<int>(members.size());
      members.emplace_back();
    }
    comp_of[i] = comp_of[root];
    members[static_cast<std::size_t>(comp_of[i])].push_back(i);
  }
  const std::size_t n_comp = members.size();

  // Chain each component from its root along the tree.
  std::vector<double> chain = p.poses;
  std::vector<char> comp_relative(n_comp, 0);
  for (std::size_t c = 0; c < n_comp; ++c) {
    std::size_t root = members[c][0];
    for (std::size_t i : members[c])
      if (degree[i] > degree[root]) root = i;
    comp_relative[c] = degree[root] > 0;
    std::vector<char> seen(n_img, 0);
    std::vector<std::size_t> stack = {root};
    seen[root] = 1;
    while (!stack.empty()) {
      const std::size_t a = stack.back();
      stack.pop_back();
      for (auto it = tree[a].rbegin(); it != tree[a].rend(); ++it) {
        const std::size_t k = *it;
        const auto i = static_cast<std::size_t>(p.rel_images[2 * k]), j = static_cast<std::size_t>(p.rel_images[2 * k + 1]);
        const std::size_t b = a == i ? j : i;
        if (seen[b]) continue;
        seen[b] = 1;
        const double* T = &p.rel_pose[7 * k];
        const double* Pa = &chain[7 * a];
        double* Pb = &chain[7 * b];
        if (a == i) {
          // T_j = T_p T_i
          double Rt[3];
          quat_multiply(T, Pa, Pb);
          quat_normalize(Pb);
          quat_rotate(T, Pa + 4, Rt);
          for (int c3 = 0; c3 < 3; ++c3) Pb[4 + c3] = Rt[c3] + T[4 + c3];
        } else {
          // T_i = T_p^-1 T_j
          double Tc[4], d[3], Rd[3];
          conj(T, Tc);
          quat_multiply(Tc, Pa, Pb);
          quat_normalize(Pb);
          for (int c3 = 0; c3 < 3; ++c3) d[c3] = Pa[4 + c3] - T[4 + c3];
          quat_rotate(Tc, d, Rd);
          for (int c3 = 0; c3 < 3; ++c3) Pb[4 + c3] = Rd[c3];
        }
        stack.push_back(b);
      }
    }
  }

  // Global priors by component, as targets in the working frame once their set's
  // transform is known.
  std::vector<std::vector<std::size_t>> comp_globals(n_comp);
  for (std::size_t k = 0; k < n_glob; ++k) comp_globals[static_cast<std::size_t>(comp_of[static_cast<std::size_t>(p.glob_image[k])])].push_back(k);
  std::vector<char> set_known(n_sets, 0);
  std::vector<int> set_frame(n_sets, -1);
  for (std::size_t s = 0; s < n_sets; ++s) {
    if (p.relation_of_set(s) == FrameRelation::Identity) {
      set_known[s] = 1;
      set_frame[s] = 0;
    }
  }
  std::vector<int> comp_frame(n_comp, -1);  // 0: working frame; others: independent frames
  int next_frame = 1;

  // Working-frame target of global prior k (its set's transform must be known).
  auto target = [&](std::size_t k, double* q, double* C) {
    const auto s = static_cast<std::size_t>(p.set_of_global_prior(k));
    const double* S = &out.set_transform[8 * s];
    const double* g = &p.glob_pose[7 * k];
    double Cs[3], RC[3], Sc[4];
    centre(g, Cs);
    quat_rotate(S, Cs, RC);
    for (int a = 0; a < 3; ++a) C[a] = S[7] * RC[a] + S[4 + a];
    conj(S, Sc);
    quat_multiply(g, Sc, q);
    quat_normalize(q);
  };

  // Places a component from the targets of its priors in known sets.
  auto place = [&](std::size_t c, int frame) {
    std::vector<double> from, to, rots;
    std::vector<std::size_t> direct;
    for (std::size_t k : comp_globals[c]) {
      const auto s = static_cast<std::size_t>(p.set_of_global_prior(k));
      if (!set_known[s]) continue;
      const auto i = static_cast<std::size_t>(p.glob_image[k]);
      double q[4], C[3], Cc[3], qc[4], r[4];
      target(k, q, C);
      centre(&chain[7 * i], Cc);
      from.insert(from.end(), Cc, Cc + 3);
      to.insert(to.end(), C, C + 3);
      conj(q, qc);
      quat_multiply(qc, &chain[7 * i], r);
      rots.insert(rots.end(), r, r + 4);
      direct.push_back(k);
    }
    if (direct.empty()) return false;
    if (comp_relative[c]) {
      // The chained poses, moved as a whole onto the priors; a few bad global
      // priors do not move their images away from their neighbours.
      double G[8];
      robust_fit(from, to, rots, false, G);
      for (std::size_t i : members[c]) transform_pose(G, &chain[7 * i], &out.poses[7 * i]);
    } else {
      // A single image: at its prior.
      double q[4], C[3];
      target(direct[0], q, C);
      make_pose(q, C, &out.poses[7 * static_cast<std::size_t>(p.glob_image[direct[0]])]);
    }
    comp_frame[c] = frame;
    return true;
  };

  for (;;) {
    bool progress = false;
    // Components with priors in known sets.
    for (std::size_t c = 0; c < n_comp; ++c) {
      if (comp_frame[c] >= 0) continue;
      int frame = -1;
      for (std::size_t k : comp_globals[c]) {
        const auto s = static_cast<std::size_t>(p.set_of_global_prior(k));
        if (set_known[s] && (frame < 0 || set_frame[s] < frame)) frame = set_frame[s];
      }
      if (frame >= 0 && place(c, frame)) progress = true;
    }
    // Sets with placed images: fit their transforms.
    for (std::size_t s = 0; s < n_sets && !progress; ++s) {
      if (set_known[s]) continue;
      std::vector<double> from, to, rots;
      int frame = -1;
      for (std::size_t k = 0; k < n_glob; ++k) {
        if (static_cast<std::size_t>(p.set_of_global_prior(k)) != s) continue;
        const auto i = static_cast<std::size_t>(p.glob_image[k]);
        const int f = comp_frame[static_cast<std::size_t>(comp_of[i])];
        if (f < 0) continue;
        if (frame < 0 || f < frame) frame = f;
      }
      if (frame < 0) continue;
      for (std::size_t k = 0; k < n_glob; ++k) {
        if (static_cast<std::size_t>(p.set_of_global_prior(k)) != s) continue;
        const auto i = static_cast<std::size_t>(p.glob_image[k]);
        if (comp_frame[static_cast<std::size_t>(comp_of[i])] != frame) continue;
        const double* g = &p.glob_pose[7 * k];
        const double* w = &out.poses[7 * i];
        double Cs[3], Cw[3], wc[4], r[4];
        centre(g, Cs);
        centre(w, Cw);
        from.insert(from.end(), Cs, Cs + 3);
        to.insert(to.end(), Cw, Cw + 3);
        // R_w = R_g R_S^T, so R_S = R_w^T R_g.
        conj(w, wc);
        quat_multiply(wc, g, r);
        rots.insert(rots.end(), r, r + 4);
      }
      robust_fit(from, to, rots, p.relation_of_set(s) == FrameRelation::Similarity, &out.set_transform[8 * s]);
      if (p.relation_of_set(s) != FrameRelation::Similarity) out.set_transform[8 * s + 7] = 1;
      set_known[s] = 1;
      set_frame[s] = frame;
      progress = true;
    }
    if (progress) continue;
    // Nothing placed through anything known: take the lowest set used by an
    // unplaced component as it is, in a new frame.
    for (std::size_t c = 0; c < n_comp && !progress; ++c) {
      if (comp_frame[c] >= 0) continue;
      for (std::size_t k : comp_globals[c]) {
        const auto s = static_cast<std::size_t>(p.set_of_global_prior(k));
        if (set_known[s]) continue;
        set_known[s] = 1;
        set_frame[s] = next_frame++;
        progress = true;
        break;
      }
    }
    if (!progress) break;
  }
  // Components with relative priors only: their own frames.
  for (std::size_t c = 0; c < n_comp; ++c) {
    if (comp_frame[c] >= 0) continue;
    for (std::size_t i : members[c]) std::copy_n(&chain[7 * i], 7, &out.poses[7 * i]);
    comp_frame[c] = next_frame++;
  }

  // Groups: frames, numbered by decreasing size; the working frame (0) first when
  // a set that is the working frame placed anything.
  std::vector<std::size_t> frame_size(static_cast<std::size_t>(next_frame), 0);
  std::vector<std::size_t> frame_first(static_cast<std::size_t>(next_frame), n_img);
  for (std::size_t c = 0; c < n_comp; ++c) {
    const auto f = static_cast<std::size_t>(comp_frame[c]);
    frame_size[f] += members[c].size();
    frame_first[f] = std::min(frame_first[f], members[c][0]);
  }
  std::vector<std::size_t> frames;
  for (std::size_t f = 0; f < frame_size.size(); ++f)
    if (frame_size[f] > 0) frames.push_back(f);
  std::stable_sort(frames.begin(), frames.end(), [&](std::size_t a, std::size_t b) {
    if ((a == 0) != (b == 0)) return a == 0;
    if (frame_size[a] != frame_size[b]) return frame_size[a] > frame_size[b];
    return frame_first[a] < frame_first[b];
  });
  std::vector<int> group_of_frame(frame_size.size(), -1);
  for (std::size_t g = 0; g < frames.size(); ++g) group_of_frame[frames[g]] = static_cast<int>(g);
  for (std::size_t c = 0; c < n_comp; ++c)
    for (std::size_t i : members[c]) out.group[i] = group_of_frame[static_cast<std::size_t>(comp_frame[c])];
  out.num_groups = static_cast<int>(frames.size());
  return out;
}

}  // namespace jba
