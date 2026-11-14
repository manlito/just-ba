// Writes a small synthetic problem with perturbed points and poses, for the
// always-on Ceres test. Two cameras are held fixed to remove the gauge freedom.

#include <cstdio>
#include <random>

#include "jba/geometry.h"
#include "jba/problem_io.h"
#include "synthetic.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <out.jba>\n", argv[0]);
    return 2;
  }
  auto p = jba_test::random_problem(jba::CameraModel::SimpleRadial, 12, 300, 5);
  std::mt19937_64 rng(6);
  std::normal_distribution<double> n(0, 1);
  for (double& v : p.points) v += 0.02 * n(rng);
  for (std::size_t i = 2; i < p.num_images(); ++i) {
    double* pose = &p.poses[7 * i];
    const double step[6] = {0.01 * n(rng), 0.01 * n(rng), 0.01 * n(rng), 0.02 * n(rng), 0.02 * n(rng), 0.02 * n(rng)};
    double out[7];
    jba::pose_retract(pose, step, out);
    for (int a = 0; a < 7; ++a) pose[a] = out[a];
  }
  p.image_fixed.assign(p.num_images(), 0);
  p.image_fixed[0] = p.image_fixed[1] = 1;
  jba::write_problem(argv[1], p);
  return 0;
}
