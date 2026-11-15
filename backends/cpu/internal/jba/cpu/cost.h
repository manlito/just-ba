#pragma once

#include "jba/problem.h"

namespace jba::cpu {

// Total cost 0.5 * sum ||r / sigma||^2 over all reprojection residuals, with no
// robust loss. Same convention as Ceres, so the numbers compare directly.
double reprojection_cost(const Problem& problem);

}  // namespace jba::cpu
