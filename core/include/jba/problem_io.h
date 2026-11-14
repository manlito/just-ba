#pragma once

#include <string>

#include "jba/problem.h"

namespace jba {

// Problem file format, version 1. Specification: docs/11-problem-file-format.md.
// Both functions throw std::runtime_error on failure.
Problem read_problem(const std::string& path);
void write_problem(const std::string& path, const Problem& problem);

// Bundle Adjustment in the Large text format, converted into the RADIAL model
// with one intrinsics block per image (document 05). Throws std::runtime_error.
Problem read_bal(const std::string& path);

}  // namespace jba
