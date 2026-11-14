// BAL conversion: the cost of the converted problem equals the cost computed with
// BAL's own camera model, implemented here independently (Rodrigues rotation,
// camera looking down -z, image y up).

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>

#include "check.h"
#include "jba/cpu/cost.h"
#include "jba/problem_io.h"
#include "synthetic.h"

namespace {

void rodrigues(const double* aa, const double* X, double* out) {
  const double theta = std::sqrt(aa[0] * aa[0] + aa[1] * aa[1] + aa[2] * aa[2]);
  const double k[3] = {aa[0] / theta, aa[1] / theta, aa[2] / theta};
  const double c = std::cos(theta), s = std::sin(theta);
  const double kx[3] = {k[1] * X[2] - k[2] * X[1], k[2] * X[0] - k[0] * X[2], k[0] * X[1] - k[1] * X[0]};
  const double kd = k[0] * X[0] + k[1] * X[1] + k[2] * X[2];
  for (int i = 0; i < 3; ++i) out[i] = X[i] * c + kx[i] * s + k[i] * kd * (1 - c);
}

}  // namespace

int main() {
  std::mt19937_64 rng(3);
  jba_test::Uniform u(-1, 1);
  const int n_cam = 3, n_pts = 6;
  double cams[n_cam][9], pts[n_pts][3];
  for (auto& c : cams) {
    c[0] = 0.3 * u(rng); c[1] = 0.3 * u(rng); c[2] = 0.3 * u(rng);
    c[3] = 0.2 * u(rng); c[4] = 0.2 * u(rng); c[5] = -5 + 0.2 * u(rng);  // points in front: P.z < 0
    c[6] = 500 + 50 * u(rng); c[7] = -0.1 + 0.01 * u(rng); c[8] = 0.01 * u(rng);
  }
  for (auto& p : pts) for (double& v : p) v = u(rng);

  const auto path = (std::filesystem::temp_directory_path() / "jba_test_tiny_bal.txt").string();
  double expected = 0;
  {
    std::ofstream out(path);
    out.precision(17);
    out << n_cam << " " << n_pts << " " << n_cam * n_pts << "\n";
    std::string body;
    for (int i = 0; i < n_cam; ++i) {
      for (int j = 0; j < n_pts; ++j) {
        double P[3];
        rodrigues(cams[i], pts[j], P);
        for (int a = 0; a < 3; ++a) P[a] += cams[i][3 + a];
        const double px = -P[0] / P[2], py = -P[1] / P[2];
        const double r2 = px * px + py * py;
        const double rad = 1 + cams[i][7] * r2 + cams[i][8] * r2 * r2;
        const double x = cams[i][6] * rad * px, y = cams[i][6] * rad * py;
        const double ox = x + 2 * u(rng), oy = y + 2 * u(rng);
        expected += 0.5 * ((x - ox) * (x - ox) + (y - oy) * (y - oy));
        out << i << " " << j << " " << ox << " " << oy << "\n";
      }
    }
    for (auto& c : cams) for (double v : c) out << v << "\n";
    for (auto& p : pts) for (double v : p) out << v << "\n";
  }

  const auto p = jba::read_bal(path);
  EXPECT(p.model == jba::CameraModel::Radial);
  EXPECT(p.num_images() == n_cam && p.num_points() == n_pts && p.num_observations() == n_cam * n_pts);
  EXPECT(p.num_intrinsics_blocks() == n_cam);
  const double cost = jba::cpu::reprojection_cost(p);
  std::printf("BAL cost %.12g, converted cost %.12g\n", expected, cost);
  EXPECT_NEAR(cost, expected, 1e-10 * expected);
  std::filesystem::remove(path);
  return TEST_MAIN_RESULT();
}
