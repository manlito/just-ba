#pragma once

#include <cmath>
#include <optional>
#include <string_view>

namespace jba {

// Robust losses (document 02). Each maps the squared, whitened residual norm s to
// rho(s), with rho(s) ~ s near zero, and has the same definition and scale
// parameter `a` as the Ceres loss of the same name, so costs compare directly.
// The objective is 0.5 * sum rho(s).
enum class LossType {
  None,          // rho = s
  Huber,         // rho = s for s <= a^2, else 2 a sqrt(s) - a^2
  Cauchy,        // rho = a^2 log(1 + s / a^2)
  GemanMcClure,  // rho = a^2 s / (a^2 + s)       (no Ceres counterpart; tested on its own)
};

struct Loss {
  LossType type = LossType::None;
  double scale = 1.0;  // a, in whitened residual units (pixels / sigma)
};

std::string_view loss_name(LossType type);
std::optional<LossType> loss_from_name(std::string_view name);

// rho(s), rho'(s), rho''(s).
inline void evaluate_loss(const Loss& loss, double s, double rho[3]) {
  const double a2 = loss.scale * loss.scale;
  rho[0] = s;
  rho[1] = 1;
  rho[2] = 0;
  switch (loss.type) {
    case LossType::None:
      rho[0] = s;
      rho[1] = 1;
      rho[2] = 0;
      return;
    case LossType::Huber:
      if (s > a2) {
        const double r = std::sqrt(s);
        rho[0] = 2 * loss.scale * r - a2;
        rho[1] = loss.scale / r;
        rho[2] = -rho[1] / (2 * s);
      } else {
        rho[0] = s;
        rho[1] = 1;
        rho[2] = 0;
      }
      return;
    case LossType::Cauchy: {
      const double inv = 1 / (1 + s / a2);
      rho[0] = a2 * std::log1p(s / a2);
      rho[1] = inv;
      rho[2] = -inv * inv / a2;
      return;
    }
    case LossType::GemanMcClure: {
      const double d = a2 + s;
      rho[0] = a2 * s / d;
      rho[1] = a2 * a2 / (d * d);
      rho[2] = -2 * rho[1] / d;
      return;
    }
  }
}

}  // namespace jba
