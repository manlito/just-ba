// Public entry point: picks the backend and runs the trust-region loop. Only the CPU
// backend exists so far; the Metal and CUDA backends will be selected here.

#include <stdexcept>

#include "jba/cpu/backend.h"
#include "jba/solver.h"
#include "jba/trust_region.h"

namespace jba {

SolverSummary solve(Problem& problem, const SolverOptions& options, const IterationCallback& callback) {
  if (auto err = validate(problem); !err.empty()) throw std::invalid_argument("invalid problem: " + err);
  if (!has_reprojection(problem.model))
    throw std::invalid_argument("camera model " + std::string(model_name(problem.model)) + " is not implemented yet");

  const Gauge gauge = options.gauge == GaugeMode::Auto ? choose_gauge(problem) : Gauge{};
  cpu::CpuBackend backend(problem, options, gauge);
  SolverSummary summary;
  summary.gauge = gauge;
  backend.cost_by_type(summary.initial_cost_by_type.data());
  run_trust_region(backend, options, callback, summary);
  backend.cost_by_type(summary.final_cost_by_type.data());
  backend.write_back(problem);
  return summary;
}

}  // namespace jba
