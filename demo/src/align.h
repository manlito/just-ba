// Aligning one reconstruction onto another: the similarity of the camera centres, the
// common rotation of the camera rotations, and small statistics.
#pragma once

#include <vector>

namespace demo {

// The similarity to ~ s R from + t best in least squares (3 values per point).
void fit_similarity(const std::vector<double>& from, const std::vector<double>& to, double R[9], double t[3], double* s);

// The rotation Q that best maps estimated rotations onto true ones (Re Q' ~ Rg), from
// their quaternions (4 per image, qw first).
void fit_rotation(const std::vector<double>& est_q, const std::vector<double>& true_q, double Q[9]);

double median(std::vector<double> v);
// Diagonal of the box around points (3 values each).
double bbox_diagonal(const std::vector<double>& pts);

}  // namespace demo
