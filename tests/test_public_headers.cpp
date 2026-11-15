// Every public header, compiled against the public include directories only (D79):
// this target does not link jba_internal, so a public header that reaches an
// internal one fails to build.

#include "jba/camera_model.h"
#include "jba/initialize.h"
#include "jba/loss.h"
#include "jba/manhattan.h"
#include "jba/priors.h"
#include "jba/problem.h"
#include "jba/problem_io.h"
#include "jba/refine.h"
#include "jba/report.h"
#include "jba/solver.h"
#include "jba/tracks.h"
#include "jba/triangulation.h"

int main() {
  jba::Problem p;
  return p.num_images() == 0 ? 0 : 1;
}
