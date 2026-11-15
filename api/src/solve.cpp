// Public entry point: picks the backend and runs the trust-region loop.

#include <stdexcept>
#include <string>

#include "jba/cpu/backend.h"
#include "jba/solver.h"
#include "jba/trust_region.h"
#ifdef JBA_HAVE_METAL
#include "jba/metal/backend.h"
#include "jba/metal/reprojection.h"
#endif

namespace jba {

std::string_view backend_name(BackendKind kind) {
  switch (kind) {
    case BackendKind::Cpu: return "cpu";
    case BackendKind::Metal: return "metal";
  }
  return "cpu";
}

std::optional<BackendKind> backend_from_name(std::string_view name) {
  for (BackendKind k : {BackendKind::Cpu, BackendKind::Metal})
    if (backend_name(k) == name) return k;
  return std::nullopt;
}

bool backend_available(BackendKind kind) {
  switch (kind) {
    case BackendKind::Cpu: return true;
    case BackendKind::Metal:
#ifdef JBA_HAVE_METAL
      return metal::available();
#else
      return false;
#endif
  }
  return false;
}

std::string_view linear_solver_name(LinearSolverType type) {
  switch (type) {
    case LinearSolverType::IterativeSchur: return "iterative_schur";
    case LinearSolverType::SparseSchur: return "sparse_schur";
  }
  return "iterative_schur";
}

std::optional<LinearSolverType> linear_solver_from_name(std::string_view name) {
  for (LinearSolverType t : {LinearSolverType::IterativeSchur, LinearSolverType::SparseSchur})
    if (linear_solver_name(t) == name) return t;
  return std::nullopt;
}

namespace {

template <typename B>
SolverSummary run(B& backend, Problem& problem, const SolverOptions& options, const IterationCallback& callback,
                  const Gauge& gauge) {
  SolverSummary summary;
  summary.gauge = gauge;
  backend.cost_by_type(summary.initial_cost_by_type.data());
  run_trust_region(backend, options, callback, summary);
  backend.cost_by_type(summary.final_cost_by_type.data());
  backend.write_back(problem);
  return summary;
}

}  // namespace

SolverSummary solve(Problem& problem, const SolverOptions& options, const IterationCallback& callback) {
  if (auto err = validate(problem); !err.empty()) throw std::invalid_argument("invalid problem: " + err);
  if (!has_reprojection(problem.model))
    throw std::invalid_argument("camera model " + std::string(model_name(problem.model)) + " is not implemented yet");
  if (!backend_available(options.backend))
    throw std::invalid_argument("backend " + std::string(backend_name(options.backend)) +
                                " is not available in this build or on this machine");

  if (options.linear_solver == LinearSolverType::SparseSchur && options.backend != BackendKind::Metal)
    throw std::invalid_argument("linear solver sparse_schur needs the metal backend");

  const Gauge gauge = options.gauge == GaugeMode::Auto ? choose_gauge(problem) : Gauge{};
#ifdef JBA_HAVE_METAL
  if (options.backend == BackendKind::Metal) {
    metal::MetalBackend backend(problem, options, gauge);
    return run(backend, problem, options, callback, gauge);
  }
#endif
  cpu::CpuBackend backend(problem, options, gauge);
  return run(backend, problem, options, callback, gauge);
}

}  // namespace jba
