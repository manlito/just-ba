// Reference solve with Ceres 2.2.0 alone (test-only; document 05).
//
//   jba_ceres_solve <problem.jba | bal.txt> [--linear-solver dense_schur|sparse_schur|iterative_schur]
//                   [--max-iterations N] [--output solved.jba] [--threads N]
//
// Prints a JSON summary. Exits non-zero if Ceres's initial cost disagrees with
// just-ba's CPU evaluation, or the solve fails. jba_parity compares the two solvers.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>

#include "ceres_reference.h"
#include "jba/cpu/cost.h"
#include "jba/problem_io.h"

namespace {

bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::fprintf(stderr, "usage: %s <problem.jba | bal.txt> [--linear-solver S] [--max-iterations N] "
                           "[--output out.jba] [--threads N]\n", argv[0]);
      return 2;
    }
    const std::string input = argv[1];
    jba_ceres::Options o;
    o.linear_solver = jba_ceres::LinearSolver::SparseSchur;
    o.num_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    std::string linear = "sparse_schur", output;
    for (int a = 2; a + 1 < argc; a += 2) {
      const std::string key = argv[a], val = argv[a + 1];
      if (key == "--max-iterations") o.max_iterations = std::stoi(val);
      else if (key == "--function-tolerance") o.function_tolerance = std::stod(val);
      else if (key == "--output") output = val;
      else if (key == "--threads") o.num_threads = std::stoi(val);
      else if (key == "--linear-solver") {
        linear = val;
        if (val == "dense_schur") o.linear_solver = jba_ceres::LinearSolver::DenseSchur;
        else if (val == "sparse_schur") o.linear_solver = jba_ceres::LinearSolver::SparseSchur;
        else if (val == "iterative_schur") o.linear_solver = jba_ceres::LinearSolver::IterativeSchur;
        else throw std::invalid_argument("unknown linear solver " + val);
      } else {
        throw std::invalid_argument("unknown option " + key);
      }
    }

    jba::Problem p = ends_with(input, ".txt") ? jba::read_bal(input) : jba::read_problem(input);
    const double jba_initial = jba::cpu::reprojection_cost(p);
    const jba_ceres::Result r = jba_ceres::solve(p, o);
    if (!output.empty()) jba::write_problem(output, p);

    const double rel = std::abs(r.initial_cost - jba_initial) / std::max(1.0, std::abs(r.initial_cost));
    std::printf("{\n  \"input\": %s,\n  \"ceres_version\": \"%s\",\n  \"model\": \"%s\",\n", jba_ceres::json_string(input).c_str(),
                CERES_VERSION_STRING, std::string(jba::model_name(p.model)).c_str());
    std::printf("  \"images\": %zu, \"points\": %zu, \"observations\": %zu,\n", p.num_images(), p.num_points(),
                p.num_observations());
    std::printf("  \"linear_solver\": \"%s\", \"threads\": %d,\n", linear.c_str(), o.num_threads);
    std::printf("  \"initial_cost_ceres\": %.17g,\n  \"initial_cost_jba\": %.17g,\n  \"initial_cost_rel_diff\": %.3g,\n",
                r.initial_cost, jba_initial, rel);
    std::printf("  \"final_cost\": %.17g,\n  \"iterations\": %d,\n  \"termination\": \"%s\",\n  \"time_seconds\": %.4f,\n",
                r.final_cost, r.iterations, r.termination.c_str(), r.time_seconds);
    std::printf("  \"history_cost\": [");
    for (std::size_t i = 0; i < r.history_cost.size(); ++i) std::printf("%s%.17g", i ? ", " : "", r.history_cost[i]);
    std::printf("]\n}\n");

    if (rel > 1e-9) {
      std::fprintf(stderr, "initial cost mismatch between Ceres and just-ba: %.3g\n", rel);
      return 1;
    }
    if (!r.usable || !(r.final_cost <= r.initial_cost)) {
      std::fprintf(stderr, "Ceres solve failed\n");
      return 1;
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "jba_ceres_solve: %s\n", e.what());
    return 1;
  }
}
