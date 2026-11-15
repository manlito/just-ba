// just-ba sample command-line binary, also the benchmark driver (document 05).

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <optional>
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
    "  --loss <name>[:<scale>]      none, huber, cauchy, geman_mcclure, for reprojection; default\n"
    "                               the file's loss.* sections, else none, scale 1\n"
    "  --<type>-loss <name>[:<scale>]  the same for relative-prior, global-prior, gravity; default\n"
    "                               the file's loss.* sections, else the library's defaults\n"
    "  --gauge <auto|none>          default auto\n"
    "  --threads <n>                default 0 (all hardware threads)\n";

// A string as a JSON string literal, quotes included.
std::string json_string(std::string_view s) {
  std::string out = "\"";
  for (const char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(c)));
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out + "\"";
}

// The gauge held fixed (D66), as a JSON object.
std::string gauge_json(const jba::Gauge& g) {
  std::string out = "{\"fixed_images\": [";
  for (std::size_t a = 0; a < g.fixed_images.size(); ++a) out += (a ? ", " : "") + std::to_string(g.fixed_images[a]);
  out += "], \"heading_images\": [";
  for (std::size_t a = 0; a < g.heading_images.size(); ++a)
    out += (a ? ", " : "") + std::to_string(g.heading_images[a]);
  out += "], \"translation_images\": [";
  for (std::size_t a = 0; a < g.translation_images.size(); ++a)
    out += (a ? ", " : "") + std::to_string(g.translation_images[a]);
  out += "], \"manhattan_frame_held\": ";
  out += g.manhattan_frame_held ? "true" : "false";
  out += ", \"fixed_translations\": [";
  for (std::size_t a = 0; a < g.fixed_translations.size(); ++a)
    out += (a ? ", [" : "[") + std::to_string(g.fixed_translations[a][0]) + ", " +
           std::to_string(g.fixed_translations[a][1]) + "]";
  out += "], \"fixed_set_scales\": [";
  for (std::size_t a = 0; a < g.fixed_set_scales.size(); ++a)
    out += (a ? ", " : "") + std::to_string(g.fixed_set_scales[a]);
  return out + "]}";
}

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
  std::optional<jba::Loss> loss_flags[jba::kNumLossTypes];  // override the file's loss.* sections
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
    } else {
      std::size_t loss_type = jba::kNumLossTypes;
      if (key == "--loss") loss_type = jba::kLossReprojection;
      for (std::size_t t = 1; t < jba::kNumLossTypes; ++t) {
        std::string flag = "--" + std::string(jba::residual_type_name(t)) + "-loss";
        for (char& c : flag)
          if (c == '_') c = '-';
        if (key == flag) loss_type = t;
      }
      if (loss_type == jba::kNumLossTypes) throw std::invalid_argument("unknown option " + key);
      const auto colon = val.find(':');
      const auto type = jba::loss_from_name(val.substr(0, colon));
      if (!type) throw std::invalid_argument("unknown loss " + val);
      loss_flags[loss_type] = jba::Loss{*type, colon != std::string::npos ? std::stod(val.substr(colon + 1)) : 1.0};
    }
  }

  jba::Problem p = load(input);
  o = jba::with_problem_losses(o, p);  // loss.* sections, if any (D80)
  for (std::size_t t = 0; t < jba::kNumLossTypes; ++t)
    if (loss_flags[t]) *jba::loss_of(o, t) = *loss_flags[t];
  const jba::SolverSummary s = jba::solve(p, o);
  jba::store_losses(p, o);  // the output records the losses it was solved with
  if (!output.empty()) jba::write_problem(output, p);
  if (!history.empty()) write_history(history, s);
  std::string by_type = "{";
  for (std::size_t t = 0; t < jba::kNumLossTypes; ++t) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "%s\"%s\": [%.17g, %.17g]", t ? ", " : "",
                  std::string(jba::residual_type_name(t)).c_str(), s.initial_cost_by_type[t], s.final_cost_by_type[t]);
    by_type += buf;
  }
  by_type += "}";
  std::printf(
      "{\"input\": %s, \"model\": \"%s\", \"images\": %zu, \"points\": %zu, \"observations\": %zu, "
      "\"relative_priors\": %zu, \"global_priors\": %zu, \"gravity_priors\": %zu, \"loss\": \"%s\", "
      "\"loss_scale\": %g, \"gauge\": %s, "
      "\"initial_cost\": %.17g, \"final_cost\": %.17g, \"iterations\": %d, \"accepted_steps\": %d, "
      "\"linear_iterations\": %d, \"termination\": \"%s\", \"message\": %s, \"time_seconds\": %.4f, "
      "\"cost_by_type\": %s}\n",
      json_string(input).c_str(), std::string(jba::model_name(p.model)).c_str(), p.num_images(), p.num_points(),
      p.num_observations(), p.num_relative_priors(), p.num_global_priors(), p.num_gravity_priors(),
      std::string(jba::loss_name(o.loss.type)).c_str(), o.loss.scale, gauge_json(s.gauge).c_str(), s.initial_cost,
      s.final_cost, s.iterations, s.accepted_steps, s.linear_iterations,
      std::string(jba::termination_name(s.termination)).c_str(), json_string(s.message).c_str(), s.time_seconds,
      by_type.c_str());
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
