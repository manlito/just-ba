#pragma once

// Finite-difference checks of residual Jacobians on the manifold (document 05,
// Level 1): central differences with one Richardson step, any residual size.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

#include "check.h"

namespace jba_test {

using Residual = std::function<std::vector<double>(double)>;

// d/dh of f(h) at 0.
inline std::vector<double> fd_derivative(const Residual& f, double h = 1e-4) {
  auto central = [&](double step) {
    const auto a = f(step), b = f(-step);
    std::vector<double> d(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) d[i] = (a[i] - b[i]) / (2 * step);
    return d;
  };
  const auto d1 = central(h), d2 = central(h / 2);
  std::vector<double> out(d1.size());
  for (std::size_t i = 0; i < d1.size(); ++i) out[i] = (4 * d2[i] - d1[i]) / 3;
  return out;
}

struct FdStats {
  double worst = 0;  // largest relative error of a Jacobian column
};

// Compares column `col` of J (rows x stride, row-major) with a finite difference.
inline void check_fd_column(FdStats& st, const char* what, int col, const std::vector<double>& fd, const double* J,
                            int stride, double tol = 1e-6) {
  const int rows = static_cast<int>(fd.size());
  double scale = 1;
  for (int row = 0; row < rows; ++row) scale = std::max(scale, std::abs(J[row * stride + col]));
  for (int row = 0; row < rows; ++row) {
    const double err = std::abs(J[row * stride + col] - fd[static_cast<std::size_t>(row)]) / scale;
    st.worst = std::max(st.worst, err);
    if (!(err < tol)) {
      std::fprintf(stderr, "%s[%d][%d]: analytic %.12g, finite difference %.12g\n", what, row, col,
                   J[row * stride + col], fd[static_cast<std::size_t>(row)]);
      ++failures;
    }
  }
}

}  // namespace jba_test
