// just-ba sample command-line binary. Milestone 0: problem file conversion and
// inspection. Solving arrives in milestone 1.

#include <cstdio>
#include <exception>
#include <string>
#include <string_view>

#include "jba/cpu/cost.h"
#include "jba/problem_io.h"

namespace {

constexpr const char* kUsage =
    "usage:\n"
    "  just-ba convert-bal <bal.txt> <out.jba>   convert a BAL problem to the just-ba format\n"
    "  just-ba info <problem.jba>                print sizes and the initial cost as JSON\n";

bool ends_with(std::string_view s, std::string_view suffix) {
  return s.size() >= suffix.size() && s.substr(s.size() - suffix.size()) == suffix;
}

jba::Problem load(const std::string& path) {
  return ends_with(path, ".txt") ? jba::read_bal(path) : jba::read_problem(path);
}

int info(const std::string& path) {
  const jba::Problem p = load(path);
  std::printf(
      "{\"model\": \"%s\", \"images\": %zu, \"points\": %zu, \"observations\": %zu, "
      "\"intrinsics_blocks\": %zu, \"cost\": %.17g}\n",
      std::string(jba::model_name(p.model)).c_str(), p.num_images(), p.num_points(),
      p.num_observations(), p.num_intrinsics_blocks(), jba::cpu::reprojection_cost(p));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string_view cmd = argc > 1 ? argv[1] : "";
    if (cmd == "convert-bal" && argc == 4) {
      jba::write_problem(argv[3], jba::read_bal(argv[2]));
      return 0;
    }
    if (cmd == "info" && argc == 3) return info(argv[2]);
    std::fputs(kUsage, stderr);
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "just-ba: %s\n", e.what());
    return 1;
  }
}
