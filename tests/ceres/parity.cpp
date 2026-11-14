// Level 2 parity with Ceres (document 05): same problem, same start, same loss,
// same gauge, same solver settings, into both solvers.
//
//   jba_parity <problem.jba | bal.txt> [options]
//
// Ceres runs on one thread unless --ceres-threads says otherwise (D63): with several
// its end point varies from run to run, which just-ba's does not, and the comparison
// would carry that noise. --threads sets just-ba's threads only.
//
// Checks: final cost (relative), iterations (ours <= ratio * Ceres's), and with
// --check-parameters the median camera centre difference (relative to scene scale)
// and median rotation difference. Prints JSON; exits non-zero on any failed check.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "ceres_reference.h"
#include "jba/geometry.h"
#include "jba/problem_io.h"
#include "jba/solver.h"

namespace {

bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void centre(const double* pose, double* C) {
  const double conj[4] = {pose[0], -pose[1], -pose[2], -pose[3]};
  jba::quat_rotate(conj, pose + 4, C);
  for (int i = 0; i < 3; ++i) C[i] = -C[i];
}

double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2), v.end());
  return v[v.size() / 2];
}

// RMS distance of camera centres from their mean.
double scene_scale(const jba::Problem& p) {
  const std::size_t n = p.num_images();
  double mean[3] = {0, 0, 0};
  std::vector<double> C(3 * n);
  for (std::size_t i = 0; i < n; ++i) {
    centre(&p.poses[7 * i], &C[3 * i]);
    for (int a = 0; a < 3; ++a) mean[a] += C[3 * i + a] / static_cast<double>(n);
  }
  double s = 0;
  for (std::size_t i = 0; i < n; ++i)
    for (int a = 0; a < 3; ++a) s += (C[3 * i + a] - mean[a]) * (C[3 * i + a] - mean[a]);
  return std::sqrt(s / static_cast<double>(n));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::fprintf(stderr, "usage: %s <problem> [--loss name:scale] [--gauge auto|none] [--max-iterations n] "
                           "[--function-tolerance x] [--linear-solver iterative_schur|sparse_schur|dense_schur] "
                           "[--check-parameters] [--max-cost-diff x] [--max-iteration-ratio x] "
                           "[--max-centre-diff x] [--max-rotation-diff x] [--threads n] [--ceres-threads n]\n", argv[0]);
      return 2;
    }
    const std::string input = argv[1];
    jba::SolverOptions o;
    auto linear = jba_ceres::LinearSolver::IterativeSchur;
    std::string linear_name = "iterative_schur";
    bool check_parameters = false, verbose = false;
    int ceres_threads = 1;
    double max_cost_diff = 1e-6, max_iteration_ratio = 1.5, max_centre_diff = 1e-6, max_rotation_diff = 1e-6;
    for (int a = 2; a < argc; ++a) {
      const std::string key = argv[a];
      if (key == "--check-parameters") {
        check_parameters = true;
        continue;
      }
      if (key == "--verbose") {
        verbose = true;
        continue;
      }
      if (a + 1 >= argc) throw std::invalid_argument("missing value for " + key);
      const std::string val = argv[++a];
      if (key == "--loss") {
        const auto colon = val.find(':');
        const auto type = jba::loss_from_name(val.substr(0, colon));
        if (!type) throw std::invalid_argument("unknown loss " + val);
        o.loss.type = *type;
        if (colon != std::string::npos) o.loss.scale = std::stod(val.substr(colon + 1));
      } else if (key == "--gauge") {
        o.gauge = val == "none" ? jba::GaugeMode::None : jba::GaugeMode::Auto;
      } else if (key == "--max-iterations") {
        o.max_iterations = std::stoi(val);
      } else if (key == "--function-tolerance") {
        o.function_tolerance = std::stod(val);
      } else if (key == "--parameter-tolerance") {
        o.parameter_tolerance = std::stod(val);
      } else if (key == "--linear-solver") {
        linear_name = val;
        if (val == "iterative_schur") linear = jba_ceres::LinearSolver::IterativeSchur;
        else if (val == "sparse_schur") linear = jba_ceres::LinearSolver::SparseSchur;
        else if (val == "dense_schur") linear = jba_ceres::LinearSolver::DenseSchur;
        else throw std::invalid_argument("unknown linear solver " + val);
      } else if (key == "--max-cost-diff") {
        max_cost_diff = std::stod(val);
      } else if (key == "--max-iteration-ratio") {
        max_iteration_ratio = std::stod(val);
      } else if (key == "--max-centre-diff") {
        max_centre_diff = std::stod(val);
      } else if (key == "--max-rotation-diff") {
        max_rotation_diff = std::stod(val);
      } else if (key == "--threads") {
        o.num_threads = std::stoi(val);
      } else if (key == "--ceres-threads") {
        ceres_threads = std::stoi(val);
        if (ceres_threads < 1) throw std::invalid_argument("--ceres-threads must be at least 1");
      } else {
        throw std::invalid_argument("unknown option " + key);
      }
    }

    const jba::Problem initial = ends_with(input, ".txt") ? jba::read_bal(input) : jba::read_problem(input);
    jba::Problem ours = initial, ref = initial;
    auto copts = jba_ceres::options_from(o, initial, linear);
    copts.num_threads = ceres_threads;
    const jba::SolverSummary s = jba::solve(ours, o);
    const jba_ceres::Result c = jba_ceres::solve(ref, copts);

    const double cost_diff = std::abs(s.final_cost - c.final_cost) / std::max(1e-300, std::abs(c.final_cost));
    const double initial_diff = std::abs(s.initial_cost - c.initial_cost) / std::max(1e-300, std::abs(c.initial_cost));
    const double ratio = c.iterations > 0 ? static_cast<double>(s.iterations) / c.iterations : 0.0;

    double centre_diff = 0, rotation_diff = 0;
    if (check_parameters) {
      const double scale = scene_scale(initial);
      std::vector<double> dc, dr;
      for (std::size_t i = 0; i < initial.num_images(); ++i) {
        double C1[3], C2[3];
        centre(&ours.poses[7 * i], C1);
        centre(&ref.poses[7 * i], C2);
        dc.push_back(std::sqrt((C1[0] - C2[0]) * (C1[0] - C2[0]) + (C1[1] - C2[1]) * (C1[1] - C2[1]) +
                               (C1[2] - C2[2]) * (C1[2] - C2[2])) / scale);
        const double* q1 = &ours.poses[7 * i];
        const double* q2 = &ref.poses[7 * i];
        const double d = std::min(1.0, std::abs(q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]));
        dr.push_back(2 * std::acos(d));
      }
      centre_diff = median(dc);
      rotation_diff = median(dr);
    }

    // Where the two cost histories first differ by more than 1e-6 relative.
    int first_divergence = -1;
    const std::size_t common = std::min(s.history.size(), c.history_cost.size());
    for (std::size_t i = 0; i < common; ++i) {
      const double a = s.history[i].cost, b = c.history_cost[i];
      if (std::abs(a - b) > 1e-6 * std::abs(b)) {
        first_divergence = static_cast<int>(i);
        break;
      }
    }

    std::vector<std::string> failures;
    if (!(initial_diff < 1e-12)) failures.push_back("initial cost differs");
    if (!(cost_diff <= max_cost_diff)) failures.push_back("final cost differs");
    if (c.iterations > 0 && !(ratio <= max_iteration_ratio)) failures.push_back("too many iterations");
    if (check_parameters && !(centre_diff <= max_centre_diff)) failures.push_back("camera centres differ");
    if (check_parameters && !(rotation_diff <= max_rotation_diff)) failures.push_back("rotations differ");
    if (s.termination == jba::Termination::Failure) failures.push_back("just-ba failed: " + s.message);
    if (!c.usable) failures.push_back("Ceres failed");

    std::printf("{\n  \"input\": %s, \"loss\": \"%s\", \"loss_scale\": %g, \"gauge\": \"%s\", "
                "\"linear_solver\": \"%s\", \"ceres_threads\": %d,\n",
                jba_ceres::json_string(input).c_str(), std::string(jba::loss_name(o.loss.type)).c_str(), o.loss.scale,
                o.gauge == jba::GaugeMode::Auto ? "auto" : "none", linear_name.c_str(), ceres_threads);
    std::printf("  \"images\": %zu, \"points\": %zu, \"observations\": %zu,\n", initial.num_images(),
                initial.num_points(), initial.num_observations());
    std::printf("  \"jba\":   {\"initial_cost\": %.17g, \"final_cost\": %.17g, \"iterations\": %d, "
                "\"linear_iterations\": %d, \"termination\": \"%s\", \"time_seconds\": %.4f},\n",
                s.initial_cost, s.final_cost, s.iterations, s.linear_iterations,
                std::string(jba::termination_name(s.termination)).c_str(), s.time_seconds);
    std::printf("  \"ceres\": {\"initial_cost\": %.17g, \"final_cost\": %.17g, \"iterations\": %d, "
                "\"linear_iterations\": %d, \"termination\": \"%s\", \"time_seconds\": %.4f},\n",
                c.initial_cost, c.final_cost, c.iterations, c.linear_iterations, c.termination.c_str(),
                c.time_seconds);
    std::printf("  \"final_cost_rel_diff\": %.3g, \"iteration_ratio\": %.3g, \"histories_agree_until\": %d",
                cost_diff, ratio, first_divergence < 0 ? static_cast<int>(common) - 1 : first_divergence - 1);
    if (check_parameters)
      std::printf(", \"median_centre_diff\": %.3g, \"median_rotation_diff_rad\": %.3g", centre_diff, rotation_diff);
    std::printf(",\n  \"pass\": %s\n}\n", failures.empty() ? "true" : "false");
    if (verbose) {
      std::fprintf(stderr, "iter | jba: cost, linear its, radius, rel. decrease, accepted | ceres: same\n");
      for (std::size_t i = 0; i < common; ++i) {
        const auto& h = s.history[i];
        std::fprintf(stderr, "%4zu | %.12g %4d %10.4g %10.6g %d | %.12g %4d %10.4g %10.6g %d\n", i, h.cost,
                     h.linear_iterations, h.trust_region_radius, h.relative_decrease, int(h.step_accepted),
                     c.history_cost[i], c.history_linear_iterations[i], c.history_radius[i],
                     c.history_relative_decrease[i], int(c.history_accepted[i]));
      }
    }
    for (const auto& f : failures) std::fprintf(stderr, "parity check failed: %s\n", f.c_str());
    return failures.empty() ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "jba_parity: %s\n", e.what());
    return 1;
  }
}
