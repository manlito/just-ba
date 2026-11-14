// just-ba sample command-line binary, also the benchmark driver (document 05).

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "jba/cpu/cost.h"
#include "jba/problem_io.h"
#include "jba/solver.h"

namespace {

constexpr const char* kUsage =
    "usage:\n"
    "  just-ba convert-bal <bal.txt> <out.jba>   convert a BAL problem to the just-ba format\n"
    "  just-ba info <problem>                    print sizes and the initial cost as JSON\n"
    "  just-ba solve <problem> [options]         solve on the CPU and print a JSON summary\n"
    "\n"
    "<problem> is a .jba file, or a BAL .txt file.\n"
    "\n"
    "solve options:\n"
    "  --output <file.jba>          write the solved problem\n"
    "  --history <file.json>        write the per-iteration history\n"
    "  --max-iterations <n>         default 50\n"
    "  --function-tolerance <x>     default 1e-6\n"
    "  --gradient-tolerance <x>     default 1e-10\n"
    "  --parameter-tolerance <x>    default 1e-8\n"
    "  --loss <name>[:<scale>]      none, huber, cauchy, geman_mcclure; default none, scale 1\n"
    "  --gauge <auto|none>          default auto\n"
    "  --threads <n>                default 0 (all hardware threads)\n";

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

void write_history(const std::string& path, const jba::SolverSummary& s) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error(path + ": cannot open for writing");
  char buf[512];
  out << "[";
  for (std::size_t i = 0; i < s.history.size(); ++i) {
    const auto& h = s.history[i];
    std::snprintf(buf, sizeof(buf),
                  "%s\n  {\"iteration\": %d, \"cost\": %.17g, \"cost_change\": %.17g, \"gradient_max_norm\": %.6g, "
                  "\"step_norm\": %.6g, \"relative_decrease\": %.6g, \"trust_region_radius\": %.6g, "
                  "\"linear_iterations\": %d, \"step_valid\": %s, \"step_accepted\": %s, \"time_seconds\": %.6f}",
                  i ? "," : "", h.iteration, h.cost, h.cost_change, h.gradient_max_norm, h.step_norm,
                  h.relative_decrease, h.trust_region_radius, h.linear_iterations, h.step_valid ? "true" : "false",
                  h.step_accepted ? "true" : "false", h.time_seconds);
    out << buf;
  }
  out << "\n]\n";
}

int solve(int argc, char** argv) {
  const std::string input = argv[2];
  std::string output, history;
  jba::SolverOptions o;
  for (int a = 3; a < argc; a += 2) {
    const std::string key = argv[a];
    if (a + 1 >= argc) throw std::invalid_argument("missing value for " + key);
    const std::string val = argv[a + 1];
    if (key == "--output") output = val;
    else if (key == "--history") history = val;
    else if (key == "--max-iterations") o.max_iterations = std::stoi(val);
    else if (key == "--function-tolerance") o.function_tolerance = std::stod(val);
    else if (key == "--gradient-tolerance") o.gradient_tolerance = std::stod(val);
    else if (key == "--parameter-tolerance") o.parameter_tolerance = std::stod(val);
    else if (key == "--threads") o.num_threads = std::stoi(val);
    else if (key == "--gauge") {
      if (val == "auto") o.gauge = jba::GaugeMode::Auto;
      else if (val == "none") o.gauge = jba::GaugeMode::None;
      else throw std::invalid_argument("unknown gauge " + val);
    } else if (key == "--loss") {
      const auto colon = val.find(':');
      const auto type = jba::loss_from_name(val.substr(0, colon));
      if (!type) throw std::invalid_argument("unknown loss " + val);
      o.loss.type = *type;
      if (colon != std::string::npos) o.loss.scale = std::stod(val.substr(colon + 1));
    } else {
      throw std::invalid_argument("unknown option " + key);
    }
  }

  jba::Problem p = load(input);
  const jba::SolverSummary s = jba::solve(p, o);
  if (!output.empty()) jba::write_problem(output, p);
  if (!history.empty()) write_history(history, s);
  std::printf(
      "{\"input\": \"%s\", \"model\": \"%s\", \"images\": %zu, \"points\": %zu, \"observations\": %zu, "
      "\"loss\": \"%s\", \"loss_scale\": %g, \"gauge_fixed_images\": %zu, "
      "\"initial_cost\": %.17g, \"final_cost\": %.17g, \"iterations\": %d, \"accepted_steps\": %d, "
      "\"linear_iterations\": %d, \"termination\": \"%s\", \"message\": \"%s\", \"time_seconds\": %.4f}\n",
      input.c_str(), std::string(jba::model_name(p.model)).c_str(), p.num_images(), p.num_points(),
      p.num_observations(), std::string(jba::loss_name(o.loss.type)).c_str(), o.loss.scale, s.gauge.fixed_images.size(),
      s.initial_cost, s.final_cost, s.iterations, s.accepted_steps, s.linear_iterations,
      std::string(jba::termination_name(s.termination)).c_str(), s.message.c_str(), s.time_seconds);
  return s.termination == jba::Termination::Failure ? 1 : 0;
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
    if (cmd == "solve" && argc >= 3) return solve(argc, argv);
    std::fputs(kUsage, stderr);
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "just-ba: %s\n", e.what());
    return 1;
  }
}
