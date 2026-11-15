#include <algorithm>
#include <cmath>
#include <vector>

#include "jba/geometry.h"
#include "jba/loss.h"
#include "jba/solver.h"

namespace jba {

std::string_view loss_name(LossType type) {
  switch (type) {
    case LossType::None: return "none";
    case LossType::Huber: return "huber";
    case LossType::Cauchy: return "cauchy";
    case LossType::GemanMcClure: return "geman_mcclure";
  }
  return "none";
}

std::optional<LossType> loss_from_name(std::string_view name) {
  for (LossType t : {LossType::None, LossType::Huber, LossType::Cauchy, LossType::GemanMcClure}) {
    if (loss_name(t) == name) return t;
  }
  return std::nullopt;
}

std::string_view termination_name(Termination t) {
  switch (t) {
    case Termination::Convergence: return "convergence";
    case Termination::NoConvergence: return "no_convergence";
    case Termination::Failure: return "failure";
    case Termination::Cancelled: return "cancelled";
  }
  return "failure";
}

namespace {

struct UnionFind {
  std::vector<std::size_t> parent;
  explicit UnionFind(std::size_t n) : parent(n) {
    for (std::size_t i = 0; i < n; ++i) parent[i] = i;
  }
  std::size_t find(std::size_t a) {
    while (parent[a] != a) a = parent[a] = parent[parent[a]];
    return a;
  }
  // The smaller index becomes the root, so roots do not depend on the join order.
  void join(std::size_t a, std::size_t b) {
    a = find(a);
    b = find(b);
    if (a == b) return;
    if (a < b) parent[b] = a;
    else parent[a] = b;
  }
};

// Translation coordinate that sets the scale with `ref`'s pose held (D57): the
// remaining freedom is a scaling about C1, which moves t2 along R2 (C2 - C1)
// = -t2 - R2 C1. Hold its largest coordinate.
int scale_coordinate(const Problem& p, std::size_t ref, std::size_t second) {
  const double* p1 = &p.poses[7 * ref];
  const double* p2 = &p.poses[7 * second];
  const double conj1[4] = {p1[0], -p1[1], -p1[2], -p1[3]};
  double C1[3];
  quat_rotate(conj1, p1 + 4, C1);
  for (double& v : C1) v = -v;
  double R2C1[3];
  quat_rotate(p2, C1, R2C1);
  int coord = 0;
  double best = -1;
  for (int c = 0; c < 3; ++c) {
    const double v = std::abs(-p2[4 + c] - R2C1[c]);
    if (v > best) {
      best = v;
      coord = c;
    }
  }
  return coord;
}

}  // namespace

Gauge choose_gauge(const Problem& p) {
  Gauge g;
  const bool any_image_fixed = std::any_of(p.image_fixed.begin(), p.image_fixed.end(), [](auto v) { return v != 0; });
  const bool any_point_fixed = std::any_of(p.point_fixed.begin(), p.point_fixed.end(), [](auto v) { return v != 0; });
  if (any_image_fixed || any_point_fixed) return g;

  const std::size_t n_img = p.num_images(), n_sets = p.num_prior_sets();
  std::vector<std::size_t> n_obs(n_img, 0), n_priors(n_img, 0);
  for (auto i : p.obs_image) ++n_obs[static_cast<std::size_t>(i)];

  // Connectivity: images and one node per prior set.
  UnionFind uf(n_img + n_sets);
  {
    std::vector<std::int64_t> first_image(p.num_points(), -1);
    for (std::size_t k = 0; k < p.num_observations(); ++k) {
      const auto j = static_cast<std::size_t>(p.obs_point[k]);
      const auto i = static_cast<std::size_t>(p.obs_image[k]);
      if (first_image[j] < 0) first_image[j] = static_cast<std::int64_t>(i);
      else uf.join(static_cast<std::size_t>(first_image[j]), i);
    }
  }
  for (std::size_t k = 0; k < p.num_relative_priors(); ++k) {
    const auto i = static_cast<std::size_t>(p.rel_images[2 * k]), j = static_cast<std::size_t>(p.rel_images[2 * k + 1]);
    uf.join(i, j);
    ++n_priors[i];
    ++n_priors[j];
  }
  for (std::size_t k = 0; k < p.num_global_priors(); ++k) {
    const auto i = static_cast<std::size_t>(p.glob_image[k]);
    const auto s = static_cast<std::size_t>(p.set_of_global_prior(k));
    ++n_priors[i];
    if (p.relation_of_set(s) != FrameRelation::Identity) uf.join(i, n_img + s);
  }
  for (auto i : p.grav_image) ++n_priors[static_cast<std::size_t>(i)];
  const bool manhattan = p.has_manhattan();
  if (manhattan)
    for (std::size_t k = 0; k < p.num_lines(); ++k)
      if (p.line_axis[k] >= 0) ++n_priors[static_cast<std::size_t>(p.line_image[k])];

  // What each group has.
  struct Group {
    bool frame_defined = false, relative = false, rigid = false, vertical = false, gravity = false;
    int line_axes = 0;  // bit per building axis with an assigned segment
    int similarity_set = -1;
    int ref = -1;
  };
  std::vector<Group> groups(n_img + n_sets);
  for (std::size_t k = 0; k < p.num_relative_priors(); ++k) groups[uf.find(static_cast<std::size_t>(p.rel_images[2 * k]))].relative = true;
  for (std::size_t k = 0; k < p.num_global_priors(); ++k) {
    const auto s = static_cast<std::size_t>(p.set_of_global_prior(k));
    Group& gr = groups[uf.find(static_cast<std::size_t>(p.glob_image[k]))];
    switch (p.relation_of_set(s)) {
      case FrameRelation::Identity: gr.frame_defined = true; break;
      case FrameRelation::Rigid: gr.rigid = true; break;
      case FrameRelation::Similarity:
        if (gr.similarity_set < 0 || static_cast<int>(s) < gr.similarity_set) gr.similarity_set = static_cast<int>(s);
        break;
    }
  }
  for (auto i : p.grav_image) {
    Group& gr = groups[uf.find(static_cast<std::size_t>(i))];
    gr.vertical = gr.gravity = true;
  }
  if (manhattan) {
    for (std::size_t k = 0; k < p.num_lines(); ++k) {
      if (p.line_axis[k] < 0) continue;
      Group& gr = groups[uf.find(static_cast<std::size_t>(p.line_image[k]))];
      gr.line_axes |= 1 << p.line_axis[k];
      if (p.manhattan_dof == 1) gr.vertical = true;
    }
  }
  // Whether the Manhattan frame sets a group's rotation: with one degree of freedom a
  // horizontal axis sets the heading (the vertical does the tilt); with three, two
  // axes set the whole rotation.
  auto rotation_from_frame = [&](const Group& gr) {
    if (!manhattan) return false;
    if (p.manhattan_dof == 1) return (gr.line_axes & 3) != 0;
    return gr.line_axes == 3 || gr.line_axes == 5 || gr.line_axes == 6 || gr.line_axes == 7;
  };
  for (std::size_t i = 0; i < n_img; ++i) {
    if (n_obs[i] == 0 && n_priors[i] == 0) continue;  // not a variable
    Group& gr = groups[uf.find(i)];
    if (gr.ref < 0) {
      gr.ref = static_cast<int>(i);
      continue;
    }
    const auto r = static_cast<std::size_t>(gr.ref);
    if (n_obs[i] > n_obs[r] || (n_obs[i] == n_obs[r] && n_priors[i] > n_priors[r])) gr.ref = static_cast<int>(i);
  }

  // The Manhattan frame is shared, so it ties the rotations of every group whose
  // rotation it sets (D86). It is held, and those groups hold their reference's
  // translation alone, unless some such group's frame is defined by global priors
  // (then the frame is free), or the frame has three freedoms and gravity priors
  // also set a vertical (holding it would fight them). Then the first such group
  // (by reference: most observations, most priors, lowest index) holds a rotation
  // as usual, unless one has its frame defined, and the others their translation.
  int rotation_group = -1;
  if (manhattan) {
    bool defined = false, gravity = false;
    for (std::size_t root = 0; root < groups.size(); ++root) {
      const Group& gr = groups[root];
      if (gr.ref < 0 || !rotation_from_frame(gr)) continue;
      defined = defined || gr.frame_defined;
      gravity = gravity || gr.gravity;
      if (gr.frame_defined) continue;
      if (rotation_group < 0) {
        rotation_group = static_cast<int>(root);
        continue;
      }
      const auto a = static_cast<std::size_t>(gr.ref), b = static_cast<std::size_t>(groups[static_cast<std::size_t>(rotation_group)].ref);
      if (n_obs[a] > n_obs[b] || (n_obs[a] == n_obs[b] && (n_priors[a] > n_priors[b] || (n_priors[a] == n_priors[b] && a < b))))
        rotation_group = static_cast<int>(root);
    }
    g.manhattan_frame_held = !defined && !(p.manhattan_dof == 3 && gravity);
    if (defined || g.manhattan_frame_held) rotation_group = -1;
  }
  for (std::size_t root = 0; root < groups.size(); ++root) {
    const Group& gr = groups[root];
    if (gr.ref < 0 || gr.frame_defined) continue;
    // Gravity sets the vertical: hold the reference's translation and heading only.
    // A group whose rotation the Manhattan frame sets holds its translation only.
    if (rotation_from_frame(gr) && static_cast<int>(root) != rotation_group) g.translation_images.push_back(gr.ref);
    else if (gr.vertical) g.heading_images.push_back(gr.ref);
    else g.fixed_images.push_back(gr.ref);
    if (gr.relative || gr.rigid) continue;
    if (gr.similarity_set >= 0) {
      g.fixed_set_scales.push_back(gr.similarity_set);
      continue;
    }
    const auto ref = static_cast<std::size_t>(gr.ref);
    if (n_obs[ref] == 0) continue;
    std::vector<char> seen(p.num_points(), 0);
    for (std::size_t k = 0; k < p.num_observations(); ++k) {
      if (static_cast<std::size_t>(p.obs_image[k]) == ref) seen[static_cast<std::size_t>(p.obs_point[k])] = 1;
    }
    std::vector<std::size_t> shared(n_img, 0);
    for (std::size_t k = 0; k < p.num_observations(); ++k) {
      const auto i = static_cast<std::size_t>(p.obs_image[k]);
      if (i != ref && seen[static_cast<std::size_t>(p.obs_point[k])]) ++shared[i];
    }
    std::size_t second = ref;
    for (std::size_t i = 0; i < n_img; ++i) {
      if (i == ref || uf.find(i) != root) continue;
      if (second == ref || shared[i] > shared[second]) second = i;
    }
    if (second == ref || shared[second] == 0) continue;
    g.fixed_translations.push_back({static_cast<int>(second), scale_coordinate(p, ref, second)});
  }
  std::sort(g.fixed_images.begin(), g.fixed_images.end());
  std::sort(g.heading_images.begin(), g.heading_images.end());
  std::sort(g.translation_images.begin(), g.translation_images.end());
  std::sort(g.fixed_translations.begin(), g.fixed_translations.end());
  std::sort(g.fixed_set_scales.begin(), g.fixed_set_scales.end());
  return g;
}

Loss* loss_of(SolverOptions& o, std::size_t t) {
  switch (t) {
    case kLossReprojection: return &o.loss;
    case kLossRelativePrior: return &o.relative_prior_loss;
    case kLossGlobalPrior: return &o.global_prior_loss;
    case kLossGravity: return &o.gravity_loss;
    case kLossLine: return &o.line_loss;
    default: return nullptr;
  }
}

const Loss* loss_of(const SolverOptions& o, std::size_t t) { return loss_of(const_cast<SolverOptions&>(o), t); }

std::string_view residual_type_name(std::size_t t) {
  switch (t) {
    case kLossReprojection: return "reprojection";
    case kLossRelativePrior: return "relative_prior";
    case kLossGlobalPrior: return "global_prior";
    case kLossGravity: return "gravity";
    case kLossLine: return "line";
    default: return "";
  }
}

SolverOptions with_problem_losses(SolverOptions options, const Problem& problem) {
  const std::size_t n = std::min(problem.loss_type.size(), problem.loss_scale.size());
  for (std::size_t t = 0; t < n; ++t) {
    if (Loss* l = loss_of(options, t)) *l = {static_cast<LossType>(problem.loss_type[t]), problem.loss_scale[t]};
  }
  return options;
}

void store_losses(Problem& problem, const SolverOptions& options) {
  problem.loss_type.assign(kNumLossTypes, 0);
  problem.loss_scale.assign(kNumLossTypes, 1.0);
  for (std::size_t t = 0; t < kNumLossTypes; ++t) {
    const Loss* l = loss_of(options, t);
    problem.loss_type[t] = static_cast<std::int32_t>(l->type);
    problem.loss_scale[t] = l->scale;
  }
}

}  // namespace jba
