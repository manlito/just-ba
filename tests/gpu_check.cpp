// A GPU backend against the CPU reference on one problem file (milestones 4 and 5).
//
//   jba_gpu_check <problem> [--backend metal|cuda] [--gauge auto|none]
//                 [--loss NAME[:SCALE]] [--<type>-loss NAME[:SCALE]]
//                 [--max-iterations N] [--function-tolerance X] [--parameter-tolerance X]
//                 [--max-cost-diff X] [--max-centre-diff X] [--max-rotation-diff X] [--no-repeat]
//                 [--linear-solver iterative_schur|sparse_schur] [--expect-fallback]
//
// Solves with the CPU and the GPU backend (default metal) from the same start and
// options (the CPU always with iterative_schur, the GPU with --linear-solver), then
// checks:
// - final costs: relative difference at most --max-cost-diff (default 1e-6);
// - median camera centre difference relative to the scene scale, and median rotation
//   difference in radians, at most --max-centre-diff and --max-rotation-diff
//   (defaults 1e-4); 0 skips a check;
// - a second GPU solve gives bit-identical parameters (unless --no-repeat);
// - with --expect-fallback, at least one sparse_schur step fell back to conjugate
//   gradients (so that path is tested; the option fails with iterative_schur).
// Document 05 asks for 1e-3 in cost in single precision; the defaults are what the
// backend actually reaches on the synthetic cases. Prints JSON; skips (exit 77)
// without a device for the backend.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "jba/geometry.h"
#include "jba/problem_io.h"
#include "jba/solver.h"

namespace {

bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Camera centre C = -R^T t.
void centre(const double* pose, double* C) {
  const double conj[4] = {pose[0], -pose[1], -pose[2], -pose[3]};
  jba::quat_rotate(conj, pose + 4, C);
  for (int c = 0; c < 3; ++c) C[c] = -C[c];
}

double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::fprintf(stderr, "usage: %s <problem> [options]\n", argv[0]);
      return 2;
    }
    const std::string path = argv[1];
    jba::BackendKind backend = jba::BackendKind::Metal;
    for (int a = 2; a + 1 < argc; ++a)
      if (std::string(argv[a]) == "--backend") {
        const auto b = jba::backend_from_name(argv[a + 1]);
        if (!b || *b == jba::BackendKind::Cpu) throw std::invalid_argument(std::string("not a GPU backend: ") + argv[a + 1]);
        backend = *b;
      }
    if (!jba::backend_available(backend)) {
      std::printf("no %s device; skipped\n", std::string(jba::backend_name(backend)).c_str());
      return 77;
    }
    jba::Problem start = ends_with(path, ".txt") ? jba::read_bal(path) : jba::read_problem(path);
    jba::SolverOptions o = jba::with_problem_losses({}, start);
    o.max_iterations = 200;
    o.function_tolerance = 1e-12;
    o.parameter_tolerance = 1e-12;
    double max_cost = 1e-6, max_centre = 1e-4, max_rotation = 1e-4;
    bool repeat = true, expect_fallback = false;
    jba::LinearSolverType gpu_solver = jba::LinearSolverType::IterativeSchur;  // the flag's default
    for (int a = 2; a < argc; ++a) {
      const std::string key = argv[a];
      if (key == "--no-repeat") {
        repeat = false;
        continue;
      }
      if (key == "--expect-fallback") {
        expect_fallback = true;
        continue;
      }
      if (a + 1 >= argc) throw std::invalid_argument("missing value for " + key);
      const std::string val = argv[++a];
      auto loss = [&] {
        const auto colon = val.find(':');
        const auto type = jba::loss_from_name(val.substr(0, colon));
        if (!type) throw std::invalid_argument("unknown loss " + val);
        return jba::Loss{*type, colon != std::string::npos ? std::stod(val.substr(colon + 1)) : 1.0};
      };
      if (key == "--backend") continue;  // read above
      if (key == "--gauge") o.gauge = val == "none" ? jba::GaugeMode::None : jba::GaugeMode::Auto;
      else if (key == "--max-iterations") o.max_iterations = std::stoi(val);
      else if (key == "--function-tolerance") o.function_tolerance = std::stod(val);
      else if (key == "--parameter-tolerance") o.parameter_tolerance = std::stod(val);
      else if (key == "--max-cost-diff") max_cost = std::stod(val);
      else if (key == "--max-centre-diff") max_centre = std::stod(val);
      else if (key == "--max-rotation-diff") max_rotation = std::stod(val);
      else if (key == "--loss") o.loss = loss();
      else if (key == "--linear-solver") {
        const auto l = jba::linear_solver_from_name(val);
        if (!l) throw std::invalid_argument("unknown linear solver " + val);
        gpu_solver = *l;
      }
      else {
        bool found = false;
        for (std::size_t t = 1; t < jba::kNumLossTypes; ++t) {
          std::string flag = "--" + std::string(jba::residual_type_name(t)) + "-loss";
          for (char& c : flag)
            if (c == '_') c = '-';
          if (key == flag) {
            *jba::loss_of(o, t) = loss();
            found = true;
          }
        }
        if (!found) throw std::invalid_argument("unknown option " + key);
      }
    }

    auto run = [&](jba::BackendKind kind, jba::Problem& p, double& seconds) {
      jba::SolverOptions opts = o;
      opts.backend = kind;
      opts.linear_solver = kind == jba::BackendKind::Cpu ? jba::LinearSolverType::IterativeSchur : gpu_solver;
      const auto t0 = std::chrono::steady_clock::now();
      const jba::SolverSummary s = jba::solve(p, opts);
      seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      return s;
    };
    jba::Problem cpu = start, gpu = start;
    double t_cpu = 0, t_gpu = 0;
    const jba::SolverSummary sc = run(jba::BackendKind::Cpu, cpu, t_cpu);
    const jba::SolverSummary sg = run(backend, gpu, t_gpu);

    const double cost_diff = std::abs(sg.final_cost - sc.final_cost) / std::max(sc.final_cost, 1e-300);
    // Scene scale: RMS distance of the camera centres from their mean.
    const std::size_t n = start.num_images();
    std::vector<double> C(3 * n), G(3 * n);
    double mean[3] = {0, 0, 0};
    for (std::size_t i = 0; i < n; ++i) {
      centre(&cpu.poses[7 * i], &C[3 * i]);
      centre(&gpu.poses[7 * i], &G[3 * i]);
      for (int c = 0; c < 3; ++c) mean[c] += C[3 * i + c] / static_cast<double>(n);
    }
    double scale = 0;
    for (std::size_t i = 0; i < n; ++i)
      for (int c = 0; c < 3; ++c) scale += (C[3 * i + c] - mean[c]) * (C[3 * i + c] - mean[c]);
    scale = std::sqrt(scale / static_cast<double>(std::max<std::size_t>(n, 1)));
    std::vector<double> dc(n), dr(n);
    for (std::size_t i = 0; i < n; ++i) {
      double s = 0;
      for (int c = 0; c < 3; ++c) s += (C[3 * i + c] - G[3 * i + c]) * (C[3 * i + c] - G[3 * i + c]);
      dc[i] = std::sqrt(s) / std::max(scale, 1e-300);
      double dot = 0;
      for (int c = 0; c < 4; ++c) dot += cpu.poses[7 * i + c] * gpu.poses[7 * i + c];
      dr[i] = 2 * std::acos(std::min(1.0, std::abs(dot)));
    }
    const double centre_diff = median(dc), rotation_diff = median(dr);

    bool identical = true;
    if (repeat) {
      jba::Problem again = start;
      double t = 0;
      run(backend, again, t);
      identical = again.poses == gpu.poses && again.points == gpu.points && again.intrinsics == gpu.intrinsics;
    }

    // sparse_schur takes one linear iteration per step when it factorizes, more when
    // the step fell back to conjugate gradients.
    int fallback_steps = 0;
    if (gpu_solver == jba::LinearSolverType::SparseSchur)
      for (const jba::IterationSummary& it : sg.history) fallback_steps += it.linear_iterations > 1;

    const bool ok = (!expect_fallback || fallback_steps > 0) && cost_diff <= max_cost &&
                    (max_centre == 0 || centre_diff <= max_centre) &&
                    (max_rotation == 0 || rotation_diff <= max_rotation) && identical &&
                    sg.termination != jba::Termination::Failure;
    std::printf(
        "{\"problem\": \"%s\", \"backend\": \"%s\", \"cpu_cost\": %.12g, \"gpu_cost\": %.12g, \"cost_diff\": %.3g, "
        "\"centre_diff\": %.3g, \"rotation_diff\": %.3g, \"cpu_iterations\": %d, \"gpu_iterations\": %d, "
        "\"gpu_termination\": \"%s\", \"gpu_message\": \"%s\", \"repeatable\": %s, \"cpu_seconds\": %.3f, "
        "\"gpu_seconds\": %.3f, \"fallback_steps\": %d, \"ok\": %s}\n",
        path.c_str(), std::string(jba::backend_name(backend)).c_str(), sc.final_cost, sg.final_cost, cost_diff, centre_diff, rotation_diff, sc.iterations,
        sg.iterations, std::string(jba::termination_name(sg.termination)).c_str(), sg.message.c_str(),
        repeat ? (identical ? "true" : "false") : "null", t_cpu, t_gpu, fallback_steps, ok ? "true" : "false");
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
