#pragma once

#include <functional>
#include <memory>
#include <string>

#include "jba/backend.h"
#include "jba/gpu/device.h"
#include "jba/problem.h"
#include "jba/solver.h"

namespace jba::gpu {

// The GPU backends' host side (document 03, D96), above a Device: the per-observation
// work in single precision on the device (linearization of the reprojection residual,
// column norms and gradient, point and camera blocks, the implicit Schur product and
// the conjugate gradients loop with Ceres's stopping test, or the reduced camera
// system of strategy B), every decision on the host in double (the cost of the
// current and the candidate parameters, accepting a step, the retraction).
// Camera-side factors and depth rows, few by comparison, are linearized on the host
// in double and uploaded.
//
// Two linear solvers (D92): sparse_schur forms the reduced camera system on the device
// and factorizes it in double (Device::direct_solver), falling back to conjugate
// gradients for a step whose single-precision system is not positive definite;
// iterative_schur runs the CPU backend's implicit Schur complement with preconditioned
// conjugate gradients entirely on the device. Results are the same from run to run.
//
// options.linear_solver may be Auto: the backend then picks for this problem and
// device (D102): sparse_schur where the device has a direct solver, unless the
// reduced system would not fit the device's memory or its 32-bit indices; and if the
// factorization runs out of memory part way, the solve continues with
// iterative_schur. Asked for explicitly, sparse_schur throws DoesNotFit instead.
//
// Throws std::runtime_error when a kernel fails or sparse_schur is asked of a device
// without a direct solver.
class GpuBackend : public Backend {
 public:
  // make_device is called once, or twice when Auto's first choice does not fit.
  GpuBackend(const Problem& problem, const SolverOptions& options, const Gauge& gauge,
             const std::function<std::unique_ptr<Device>()>& make_device);
  ~GpuBackend() override;

  double cost() override;
  void cost_by_type(double* out) override;
  void linearize(bool compute_scaling) override;
  double gradient_max_norm() const override;
  double parameter_norm() const override;
  StepResult compute_step(double radius, double eta, int min_linear_iterations, int max_linear_iterations) override;
  bool candidate_cost(double* cost) override;
  void accept_candidate() override;
  // 1e6: in single precision, damping below about 1e-6 of a diagonal entry is lost
  // in the sums, so a larger radius would not change the step.
  double max_trust_region_radius() const override { return 1e6; }

  // Copies the current parameters into `problem`.
  void write_back(Problem& problem) const;

  // The linear solver the steps use (after a fallback, IterativeSchur), and what the
  // solve should report about a change of solver; empty when there was none.
  LinearSolverType linear_solver() const;
  std::string note() const;
  // sparse_schur steps that fell back to conjugate gradients (D98).
  int fallback_steps() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Test hooks (tests/gpu_check.cpp): make the next `count` factorizations of strategy B
// report `status`, or the next set-up of strategy B throw DoesNotFit, so the fallbacks
// are tested without a broken matrix or a full device. Process-wide; not thread-safe.
namespace testing {
void fail_factorizations(int count, FactorStatus status);
void fail_direct_setup(bool fail);
}  // namespace testing

}  // namespace jba::gpu
